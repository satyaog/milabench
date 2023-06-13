from __future__ import annotations

import os
import asyncio
from hashlib import md5
import json
from typing import Dict, Generator, List, Tuple
from copy import deepcopy
import socket

from .metadata import machine_metadata
from .fs import XPath
from .alt_async import destroy
from .merge import merge
from .pack import BasePackage


def clone_with(cfg, new_cfg):
    return merge(deepcopy(cfg), new_cfg)


async def force_terminate(pack, delay):
    await asyncio.sleep(delay)
    for proc in pack.processes:
        ret = proc.poll()
        if ret is None:
            await pack.message(
                f"Terminating process because it ran for longer than {delay} seconds."
            )
            destroy(proc)


class Executor:
    def __init__(self, pack_or_exec: Executor | BasePackage, **kwargs) -> None:
        if isinstance(pack_or_exec, Executor):
            self.exec = pack_or_exec
            self._pack = None
        elif isinstance(pack_or_exec, BasePackage):
            self.exec = None
            self._pack = pack_or_exec
        else:
            raise RuntimeError(f"Need to be pack or executor {pack_or_exec}")
    
        self._kwargs = kwargs

    @property
    def pack(self) -> BasePackage:
        if self._pack:
            return self._pack
        return self.exec.pack

    def argv(self, **kwargs) -> List:
        if self.exec:
            return self._argv(**kwargs) + self.exec.argv(**kwargs)
        return self._argv(**kwargs)

    def kwargs(self) -> Dict:
        kwargs = self._kwargs
        if self.exec:
            kwargs = {**self.exec.kwargs(), **kwargs}
        return kwargs

    def commands(self) -> Generator[Tuple[BasePackage, List, Dict], None, None]:
        yield self.pack, self.argv(), self.kwargs()

    async def execute(self, timeout=False, timeout_delay=600, **kwargs):
        coro = []
        for pack, argv, _kwargs in self.commands():
            await pack.send(event="config", data=pack.config)
            await pack.send(event="meta", data=machine_metadata())
            
            pack.phase = "run"
            coro.append(pack.execute(*argv, **{**_kwargs, **kwargs}))
            
            if timeout:
                delay = pack.config.get("max_duration", timeout_delay)
                asyncio.create_task(force_terminate(pack, delay))

        return await asyncio.gather(*coro)

    def _argv(self, **kwargs) -> List:
        del kwargs
        return []


# Leafs
class CmdExecutor(Executor):
    def __init__(
            self,
            pack:BasePackage,
            *cmd_argv,
            **kwargs
    ) -> None:
        if isinstance(pack, Executor):
            raise ValueError(f"{self.__class__.__name__} does not accept nested"
                             f" {Executor.__class__.__name__}")
        super().__init__(
            pack,
            **kwargs
        )
        self.cmd_argv = cmd_argv

    def _argv(self, **kwargs) -> List:
        del kwargs
        return [*self.cmd_argv]


class PackExecutor(CmdExecutor):
    def __init__(
            self,
            pack:BasePackage,
            *script_argv,
            **kwargs
    ) -> None:
        script = script_argv[:1]
        if script and XPath(script[0]).exists():
            script = script[0]
            script_argv = script_argv[1:]
        else:
            script = None

        super().__init__(pack, *script_argv, **kwargs)

        self.script = script

    def _argv(self, **kwargs) -> List:
        main = self.script or self.pack.main_script
        if not XPath(main).is_absolute():
            main = self.pack.dirs.code / main  # Could this lead any unexpected
            # path during exec?

        if not main.exists():
            raise FileNotFoundError(
                f"Cannot run main script because it does not exist: {main}"
            )

        if main.is_dir():
            main = ["-m", str(self.pack.main_script)]
        else:
            main = [str(main)]
        return main + super()._argv(**kwargs)


class VoidExecutor(Executor):
    class _FakePack:
        config = dict()

        async def execute(self, *args, **kwargs):
            pass
        
        async def send(self, *args, **kwargs):
            pass

    def __init__(self, *args, **kwargs) -> None:
        self._pack = VoidExecutor._FakePack()
        self.exec = None
        self._kwargs = dict()


# Branches or Roots
class DockerRunExecutor(Executor):
    def __init__(self, executor: Executor, image: str, *docker_argv, **kwargs) -> None:
        super().__init__(executor, **kwargs)
        self.image = image
        self.docker_argv = docker_argv

    def _argv(self, **kwargs) -> List:
        del kwargs
        if self.image is None or os.environ.get("MILABENCH_DOCKER", None):
            # No-op when there's no docker image to run or inside a docker
            # container
            return []
        argv = [
            "docker",
            "run",
            "-i",
            "--rm",
            "--network",
            "host",
            "--privileged",
            "--gpus",
            "all",
            *self.docker_argv,
        ]
        env = self.pack.make_env()
        for var in ("MILABENCH_CONFIG", "XDG_CACHE_HOME", "OMP_NUM_THREADS"):
            argv.append("--env")
            argv.append(f"{var}='{env[var]}'")
        argv.append(self.image)
        argv.append(f"{self.pack.dirs.code / 'activator'}")  # What does this do?
        # Should it be part
        # of DockerExec or
        # sub exec?
        argv.append(f"{self.pack.dirs.venv}")  # What does this do? Should it
        # be part of DockerExec or sub
        # exec?
        return argv


class SSHExecutor(Executor):
    def __init__(
        self,
        executor: Executor,
        host: str,
        *ssh_argv,
        user: str = None,
        key: str = None,
        **kwargs,
    ) -> None:
        super().__init__(executor, **kwargs)
        self.host = host
        self.ssh_argv = list(*ssh_argv)
        if user:
            self.ssh_argv.append(f"-l{user}")
        if key:
            self.ssh_argv.append(f"-i{key}")

    def _argv(self, **kwargs) -> List:
        del kwargs
        if socket.gethostname() == self.host:
            # No-op when executing on the main node
            return []
        return [
            "ssh",
            "-oCheckHostIP=no",
            "-oStrictHostKeyChecking=no",
            *self.ssh_argv,
            self.host,
        ]


class VoirExecutor(Executor):
    def __init__(self, executor: Executor, *voir_argv, **kwargs) -> None:
        super().__init__(executor, **{"setsid": True, **kwargs})
        self.voir_argv = voir_argv

    def _argv(self, **kwargs) -> List:
        del kwargs
        if voirconf := self.pack.config.get("voir", None):
            hsh = md5(str(voirconf).encode("utf8"))
            voirconf_file = (
                self.pack.dirs.extra
                / f"voirconf-{self.pack.tag}-{hsh.hexdigest()}.json"
            )
            with open(voirconf_file, "w") as f:
                json.dump(fp=f, obj=voirconf, indent=4)
            voir_argv = ("--config", voirconf_file)
        else:
            voir_argv = ()

        return [
            "voir",
            *voir_argv,
            *self.voir_argv,  # Is this going to cause errors? Should we only
            # *us voir_argv and remove self.voir_argv from
            # class?
        ]


class WrapperExecutor(Executor):
    def __init__(self, executor: Executor, *wrapper_argv, **kwargs) -> None:
        super().__init__(executor, **kwargs)
        self.wrapper_argv = wrapper_argv

    def _argv(self, **kwargs) -> List:
        del kwargs
        return [*self.wrapper_argv]


# Roots
class ListExecutor(Executor):
    """Runs n instance of the same job in parallel"""

    def __init__(self, *executors: Tuple[Executor], **kwargs) -> None:
        super().__init__(executors[0], **kwargs)
        self.executors = executors

    def commands(self) -> Generator[Tuple[BasePackage, List, Dict], None, None]:
        for executor in self.executors:
            yield from executor.commands()


class NJobs(ListExecutor):
    """Runs n instance of the same job in parallel"""

    def __init__(self, executor: Executor, n: int, **kwargs) -> None:
        super().__init__(*([executor] * n), **kwargs)


class PerGPU(Executor):
    def __init__(self, executor: Executor, gpus=None, **kwargs) -> None:
        super().__init__(executor, **kwargs)
        if gpus is None:
            gpus = [{"device": 0, "selection_variable": "CPU_VISIBLE_DEVICE"}]
        self.devices = gpus

    def commands(self) -> Generator[Tuple[BasePackage, List, Dict], None, None]:
        ngpus = len(self.devices)
        
        for gpu in self.devices:
            gid = gpu["device"]
            gcfg = {
                "tag": [*self.exec.pack.config["tag"], f"D{gid}"],
                "device": gid,
                "devices": [gid] if ngpus else [],
                "env": {gpu["selection_variable"]: str(gid)},
            }
            run = clone_with(self.exec.pack.config, gcfg)
            run_pack = self.exec.pack.copy(run)
            yield run_pack, self.argv(), self.kwargs()


# Accelerate
class AccelerateLaunchExecutor(Executor):
    def __init__(
            self,
            pack:BasePackage,
            *accelerate_argv,
            **kwargs
    ) -> None:
        super().__init__(
            pack,
            **kwargs
        )
        self.accelerate_argv = accelerate_argv

    def _argv(self, rank, **kwargs) -> List:
        del kwargs
        nproc = (
            len(self.pack.config.get("devices", [])) * self.pack.config["num_machines"]
        )
        deepspeed_argv = (
            [
                "--use_deepspeed",
                "--deepspeed_multinode_launcher=standard",
                "--zero_stage=2",
            ]
            if self.pack.config["use_deepspeed"]
            else ["--multi_gpu"]
        )
        return [
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
        ]


class AccelerateLoopExecutor(Executor):
    class _Placeholder(Executor):
        def __init__(self) -> None:
            pass

    PLACEHOLDER = _Placeholder()

    def __init__(
        self,
        accelerate_exec: AccelerateLaunchExecutor,
        executor: SSHExecutor = None,
        **kwargs,
    ) -> None:
        if not isinstance(executor, SSHExecutor):
            raise ValueError(f"{self.__class__.__name__} only accepts"
                             f" {SSHExecutor.__class__.__name__} as nested"
                             f" {Executor.__class__.__name__}")
        super().__init__(
            executor,
            **kwargs
        )
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
            {"setsid": True, "use_stdout": True, **self.kwargs()},
        )
        for i, worker in enumerate(self.pack.config.get("worker_addrs", [])):
            self.exec.host = worker
            run_pack = self.pack.copy({"tag": [*self.pack.config["tag"], worker]})
            yield run_pack, self.argv(rank=i + 1), self.kwargs()
