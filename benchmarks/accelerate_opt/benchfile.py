from milabench.executors import AccelerateLaunchExecutor, AccelerateLoopExecutor, DockerRunExecutor, SSHExecutor
from milabench.pack import Package


class AccelerateBenchmark(Package):
    base_requirements = "requirements.in"

    def make_env(self):
        env = super().make_env()
        env["OMP_NUM_THREADS"] = str(self.config["cpus_per_gpu"])
        return env

    def build_execution_plan(self):
        return AccelerateLoopExecutor(
            AccelerateLaunchExecutor(
                pack=self
            ),
            exec=SSHExecutor(
                "__host__",
                user=self.config["worker_user"],
                key="/milabench/id_milabench",
                exec=DockerRunExecutor(
                    self.config["docker_image"],
                    exec=AccelerateLoopExecutor.PLACEHOLDER
                )
            )
        )

    async def prepare(self):
        self.phase = "prepare"
        await self.execute(
            "accelerate",
            "launch",
            "--mixed_precision=fp16",
            "--num_machines=1",
            "--dynamo_backend=no",
            "--num_processes=1",
            "--num_cpu_threads_per_process=8",
            str(self.dirs.code / "main.py"),
            env={"MILABENCH_PREPARE_ONLY": "1"},
        )

    async def run(self):
        self.phase = "run"
        # XXX: this doesn't participate in the process timeout
        plan = self.build_execution_plan()
        await plan.execute()

__pack__ = AccelerateBenchmark
