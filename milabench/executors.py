import asyncio
from hashlib import md5
import json
from typing import Dict, Generator, List, Tuple

from milabench.fs import XPath
from milabench.multi import clone_with
from milabench.pack import BasePackage
from voir.instruments.gpu import get_gpu_info


class Executor():
    def __init__(
            self,
            exec:"Executor"=None,
            pack:BasePackage=None,
            **kwargs
    ) -> None:
        if exec and pack:
            raise ValueError("`exec` and `pack` can not both be set")
        if exec is None and pack is None:
            raise ValueError("Missing `exec` or `pack`")
        self.exec = exec
        self._pack = pack
        self.kwargs = kwargs

    @property
    def pack(self) -> BasePackage:
        if self._pack:
            return self._pack
        return self.exec.pack

    def argv(self, *argv, **kwargs) -> List:
        if self.exec:
            return self.exec.argv(*argv, **kwargs)
        return [*argv]

    def commands(self) -> Generator[Tuple[BasePackage, List, Dict], None, None]:
        yield self.pack, self.argv(), self.kwargs

    async def execute(self):
        coro = []

        for pack, argv, kwargs in self.commands():
            coro.append(pack.execute(*argv, **kwargs))

        return await asyncio.gather(coro)


class PackExecutor(Executor):
    def __init__(
            self,
            *script_argv,
            exec:Executor=None,
            pack:BasePackage=None,
            **kwargs
    ) -> None:
        if exec:
            raise ValueError(f"{self.__class__} does not accept nested `exec`")
        super().__init__(exec=exec, pack=pack, **kwargs)
        self.script_argv = script_argv

    def argv(self, *argv, **kwargs) -> List:
        script = self.pack.main_script
        if not XPath(script).is_absolute():
            script = str(self.pack.dirs.code / script)  # Could this lead any
                                                        # unexpected path during
                                                        # exec?
        return super().argv(*argv, script, *self.script_argv, **kwargs)


class DockerRunExecutor(Executor):
    def __init__(
            self,
            image:str,
            *docker_argv,
            exec:Executor=None,
            pack:BasePackage=None,
            **kwargs
    ) -> None:
        super().__init__(exec=exec, pack=pack, **kwargs)
        self.image = image
        self.docker_argv = docker_argv

    def argv(self, *argv, **kwargs) -> List:
        argv = [
            *argv,
            "docker",
            "run", "-i", "--rm", 
            "--network", "host",
            "--privileged",
            "--gpus", "all",
            *self.docker_argv
        ]
        env = self.pack.make_env()
        for var in ('MILABENCH_CONFIG', 'XDG_CACHE_HOME', 'OMP_NUM_THREADS'):
            argv.append("--env")
            argv.append(f"{var}='{env[var]}'")
        argv.append(self.image)
        argv.append(f"{self.pack.dirs.code / 'activator'}") # What does this do?
                                                            # Should it be part
                                                            # of DockerExec or
                                                            # sub exec?
        argv.append(f"{self.pack.dirs.venv}")   # What does this do? Should it
                                                # be part of DockerExec or sub
                                                # exec?
        return super().argv(*argv, **kwargs)


class SSHExecutor(Executor):
    def __init__(
            self,
            host:str,
            *ssh_argv,
            user:str=None,
            key:str=None,
            exec:Executor=None,
            pack:BasePackage=None,
            **kwargs
    ) -> None:
        super().__init__(exec=exec, pack=pack, **kwargs)
        self.host = host
        self.ssh_argv = list(*ssh_argv)
        if user:
            self.ssh_argv.append(f"-l{user}")
        if key:
            self.ssh_argv.append(f"-i{key}")

    def argv(self, *argv, **kwargs) -> List:
        return super().argv(
            *argv,
            "ssh",
            "-oCheckHostIP=no",
            "-oStrictHostKeyChecking=no",
            *self.ssh_argv,
            self.host,
            **kwargs
        )


class VoirExecutor(Executor):
    def __init__(
            self,
            *voir_argv,
            exec:Executor=None,
            pack:BasePackage=None,
            **kwargs
    ) -> None:
        super().__init__(
            exec=exec,
            pack=pack,
            **{"setsid":True, **kwargs}
        )
        self.voir_argv = voir_argv

    def argv(self, *argv, **kwargs) -> List:
        if voirconf := self.pack.config.get("voir", None):
            hsh = md5(str(voirconf).encode("utf8"))
            voirconf_file = (
                self.dirs.extra / f"voirconf-{self.tag}-{hsh.hexdigest()}.json"
            )
            with open(voirconf_file, "w") as f:
                json.dump(fp=f, obj=voirconf, indent=4)
            voir_argv = ("--config", voirconf_file)
        else:
            voir_argv = ()

        return super().argv(
            *argv,
            "voir",
            *voir_argv,
            *self.voir_argv,    # Is this going to cause errors? Should we only
                                # *us voir_argv and remove self.voir_argv from
                                # class?
            **kwargs
        )


class PerGPU(Executor):
    def commands(self) -> Generator[Tuple[BasePackage, List, Dict], None, None]:
        gpus = get_gpu_info()["gpus"].values()
        ngpus = len(gpus)
        devices = gpus or [{
            "device": 0,
            "selection_variable": "CPU_VISIBLE_DEVICE"
        }]

        for gpu in devices:
            gid = gpu["device"]
            gcfg = {
                "tag": [*self.exec.pack.config["tag"], f"D{gid}"],
                "device": gid,
                "devices": [gid] if ngpus else [],
                "env": {gpu["selection_variable"]: str(gid)},
            }
            run = clone_with(self.exec.pack.config, gcfg)
            run_pack = self.exec.pack.copy(run)
            yield run_pack, self.argv(), self.kwargs


# What is NJobs supposed to do?
class NJobs(Executor):
    def __init__(
            self,
            n:int,
            exec:Executor=None,
            pack:BasePackage=None,
            **kwargs
    ) -> None:
        super().__init__(exec=exec, pack=pack, **kwargs)
        self.n = n

    def commands(self) -> Generator[Tuple[BasePackage, List, Dict], None, None]:
        for _ in range(self.n):
            yield from super().commands()


class AccelerateLaunchExecutor(Executor):
    def __init__(
            self,
            *accelerate_argv,
            exec:Executor=None,
            pack:BasePackage=None,
            **kwargs
    ) -> None:
        super().__init__(exec=exec, pack=pack, **kwargs)
        self.accelerate_argv = accelerate_argv

    def argv(self, *argv, rank, **kwargs) -> List:
        nproc = len(self.pack.config.get("devices", [])) * self.pack.config['num_machines']
        deepspeed_argv = [
            "--use_deepspeed",
            "--deepspeed_multinode_launcher=standard",
            "--zero_stage=2",
        ] if self.pack.config["use_deepspeed"] else ["--multi_gpu"]
        return super().argv(
            *argv,
            "accelerate",
            "launch",
            "--mixed_precision=fp16",
            "--dynamo_backend=no",
            f"--machine_rank={rank}",
            f"--num_machines={self.pack.config['num_machines']}",
            *deepspeed_argv,
            f"--gradient_accumulation_steps={self.pack.config['gradient_accumulation_steps']}",
            f"--num_cpu_threads_per_process={self.pack.config['cpus_per_gpu']}",
            f"--main_process_ip={self.pack.config['manager_addr']}",
            f"--main_process_port={self.pack.config['manager_port']}",
            f"--num_processes={nproc}",
            *self.accelerate_argv,
            str(self.pack.dirs.code / "main.py"),
            rank=rank,
            **kwargs
        )


class AccelerateLoopExecutor(Executor):
    class _Placeholder(Executor):
        def __init__(self) -> None:
            pass

    PLACEHOLDER=_Placeholder()

    def __init__(
            self,
            accelerate_exec:AccelerateLaunchExecutor,
            exec:Executor=None,
            pack:BasePackage=None,
            **kwargs
    ) -> None:
        if not isinstance(exec, SSHExecutor):
            raise ValueError(f"{self.__class__} only accepts"
                             f" {SSHExecutor.__class__} in `exec`")
        super().__init__(exec=exec, pack=pack, **kwargs)
        self.accelerate_exec = accelerate_exec
        _exec = self
        while _exec:
            if _exec.exec is AccelerateLoopExecutor.PLACEHOLDER:
                _exec.exec = self.accelerate_exec
            _exec = _exec.exec

    def commands(self) -> Generator[Tuple[BasePackage, List, Dict], None, None]:
        yield (
            self.pack,
            self.accelerate_exec.argv(rank=0),
            {"setsid":True, "use_stdout":True, **self.kwargs}
        )
        for i, worker in enumerate(self.pack.config.get('worker_addrs', [])):
            self.exec.host = worker
            run_pack = self.pack.copy({
                "tag": [*self.pack.config['tag'], worker]
            })
            yield run_pack, self.argv(rank=i + 1), self.kwargs
