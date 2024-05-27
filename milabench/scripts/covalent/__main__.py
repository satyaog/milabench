import argparse
import os
import pathlib
import subprocess
import sys
import tempfile


def serve(*argv):
    return subprocess.run([
        "covalent",
        *argv
    ]).returncode


def _get_executor_kwargs(args):
    return {
        **{k:v for k,v in vars(args).items() if k not in ("setup", "teardown")},
    }


def executor(executor_cls, args, *argv):
    import covalent as ct

    def _popen(cmd, *args, _env=None, **kwargs):
        _env = _env if _env is not None else {}

        for envvar in _env.keys():
            envvar_val = _env[envvar]

            if not envvar_val:
                continue

            envvar_val = pathlib.Path(envvar_val).expanduser()
            if str(envvar_val) != _env[envvar]:
                _env[envvar] = str(envvar_val)

        if "MILABENCH_CONFIG_CONTENT" in _env:
            _config_dir = pathlib.Path(_env["MILABENCH_CONFIG"]).parent
            with tempfile.NamedTemporaryFile("wt", dir=str(_config_dir), suffix=".yaml", delete=False) as _f:
                _f.write(_env["MILABENCH_CONFIG_CONTENT"])
                _env["MILABENCH_CONFIG"] = _f.name

        try:
            cmd = (str(pathlib.Path(cmd[0]).expanduser()), *cmd[1:])
        except IndexError:
            pass

        cwd = kwargs.pop("cwd", None)
        if cwd is not None:
            cwd = str(pathlib.Path(cwd).expanduser())
            kwargs["cwd"] = cwd

        _env = {**os.environ.copy(), **kwargs.pop("env", {}), **_env}

        kwargs = {
            **kwargs,
            "env": _env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        p = subprocess.Popen(cmd, *args, **kwargs)

        stdout_chunks = []
        while True:
            line = p.stdout.readline()
            if not line:
                break
            line_str = line.decode("utf-8").strip()
            stdout_chunks.append(line_str)
            print(line_str)

        _, stderr = p.communicate()
        stderr = stderr.decode("utf-8").strip()
        stdout = os.linesep.join(stdout_chunks)

        if p.returncode != 0:
            raise subprocess.CalledProcessError(
                p.returncode,
                (cmd, args, kwargs),
                stdout,
                stderr
            )
        return p.returncode, stdout, stderr

    executor:ct.executor.BaseExecutor = executor_cls(
        **_get_executor_kwargs(args),
    )
    return_code = 0
    try:
        if args.setup:
            dispatch_id = ct.dispatch(
                ct.lattice(executor.get_connection_attributes), disable_run=False
            )()

            result = ct.get_result(dispatch_id=dispatch_id, wait=True).result

            assert result and result[0]

            all_connection_attributes, _ = result
            for hostname, connection_attributes in all_connection_attributes.items():
                print(f"hostname::>{hostname}")
                for attribute,value in connection_attributes.items():
                    if attribute == "hostname":
                        continue
                    print(f"{attribute}::>{value}")

        if argv:
            dispatch_id = ct.dispatch(
                ct.lattice(
                    lambda:ct.electron(_popen, executor=executor)(argv)
                ),
                disable_run=False
            )()

            result = ct.get_result(dispatch_id=dispatch_id, wait=True).result

            return_code, _, _ = result if result is not None else (1, "", "")
    finally:
        if args.teardown:
            result = executor.stop_cloud_instance().result
            assert result is not None

    return return_code


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    try:
        import covalent as ct
    except (KeyError, ImportError):
        from ..utils import run_in_module_venv
        check_if_module = "import covalent"
        return run_in_module_venv(__file__, check_if_module, argv)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparser = subparsers.add_parser("serve")
    subparser.add_argument(f"argv", nargs=argparse.REMAINDER)
    for p in ("azure","ec2"):
        try:
            config = ct.get_config(f"executors.{p}")
        except KeyError:
            continue
        subparser = subparsers.add_parser(p)
        subparser.add_argument(f"--setup", action="store_true")
        subparser.add_argument(f"--teardown", action="store_true")
        for param, default in config.items():
            if param.startswith("_"):
                continue
            add_argument_kwargs = {}
            if isinstance(default, bool):
                add_argument_kwargs["action"] = "store_false" if default else "store_true"
            else:
                add_argument_kwargs["default"] = default
            subparser.add_argument(f"--{param.replace('_', '-')}", **add_argument_kwargs)

    try:
        cv_argv, argv = argv[:argv.index("--")], argv[argv.index("--")+1:]
    except ValueError:
        cv_argv, argv = argv, []

    args = parser.parse_args(cv_argv)

    if cv_argv[0] == "serve":
        assert not argv
        return serve(*args.argv)
    elif cv_argv[0] == "azure":
        executor_cls = ct.executor.AzureExecutor
    elif cv_argv[0] == "ec2":
        executor_cls = ct.executor.EC2Executor
    else:
        raise

    return executor(executor_cls, args, *argv)


if __name__ == "__main__":
    sys.exit(main())
