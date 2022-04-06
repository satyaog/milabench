import os
import runpy
import sys
from functools import partial

from coleo import Option, config as configuration, default, run_cli, tooled

from .fs import XPath
from .log import simple_dash, simple_report
from .merge import self_merge
from .multi import MultiPackage


def main():
    sys.path.insert(0, os.path.abspath(os.curdir))
    run_cli(Main)


def get_pack(defn):
    pack = XPath(defn["definition"]).expanduser()
    if not pack.is_absolute():
        pack = XPath(defn["config_base"]) / pack
        defn["definition"] = str(pack)
    pack_glb = runpy.run_path(str(pack / "benchfile.py"))
    pack_cls = pack_glb["__pack__"]
    pack_obj = pack_cls(defn)
    return pack_obj


@tooled
def _get_multipack(dev=False):
    # Configuration file
    # [positional: ?]
    config: Option & str = None

    if config is None:
        config = os.environ.get("MILABENCH_CONFIG", None)

    if config is None:
        sys.exit("Error: CONFIG argument not provided and no $MILABENCH_CONFIG")

    # Base path for code, venvs, data and runs
    base: Option & str = None

    # Whether to use the current environment
    use_current_env: Option & bool = False

    if dev:
        use_current_env = True

    # Packs to select
    select: Option & str = default("")

    # Packs to exclude
    exclude: Option & str = default("")

    if select:
        select = select.split(",")

    if exclude:
        exclude = exclude.split(",")

    config_base = str(XPath(config).parent.absolute())
    config = configuration(config)
    config["defaults"]["config_base"] = config_base
    if base is not None:
        config["defaults"]["dirs"]["base"] = base
    elif os.environ.get("MILABENCH_BASE", None):
        config["defaults"]["dirs"]["base"] = os.environ["MILABENCH_BASE"]
    elif not config["defaults"]["dirs"].get("base", None):
        sys.exit("Error: Neither --base nor $MILABENCH_BASE are set.")
    config = self_merge(config)

    objects = {}

    for name, defn in config["benchmarks"].items():
        if select and name not in select:
            continue
        if exclude and name in exclude:
            continue

        defn.setdefault("name", name)
        defn["tag"] = [defn["name"]]

        if use_current_env or defn["dirs"].get("venv", None) is None:
            venv = os.environ.get("CONDA_PREFIX", None)
            if venv is None:
                venv = os.environ.get("VIRTUAL_ENV", None)
            if venv is None:
                print("Could not find virtual environment", file=sys.stderr)
                sys.exit(1)
            defn["dirs"]["venv"] = venv

        def _format_path(pth):
            formatted = pth.format(**defn)
            xpth = XPath(formatted).expanduser()
            if formatted.startswith("."):
                xpth = xpth.absolute()
            return xpth

        dirs = {k: _format_path(v) for k, v in defn["dirs"].items()}
        dirs = {
            k: str(v if v.is_absolute() else dirs["base"] / v) for k, v in dirs.items()
        }
        defn["dirs"] = dirs
        objects[name] = get_pack(defn)

    return MultiPackage(objects)


class Main:
    def run():
        # Name of the run
        run: Option = None

        # Dev mode (adds --sync, current venv, only one run, no logging)
        dev: Option & bool = False

        # Sync changes to the benchmark directory
        sync: Option & bool = False

        mp = _get_multipack(dev=dev)

        if dev or sync:
            mp.do_install(dash=simple_dash, sync=True)

        if dev:
            mp.do_dev(dash=simple_dash)
        else:
            mp.do_run(dash=simple_dash, report=partial(simple_report, runname=run))

    def prepare():
        # Dev mode (does install --sync, uses current venv)
        dev: Option & bool = False

        # Sync changes to the benchmark directory
        sync: Option & bool = False

        mp = _get_multipack(dev=dev)

        if dev or sync:
            mp.do_install(dash=simple_dash, sync=True)

        mp.do_prepare(dash=simple_dash)

    def install():
        # Force install
        force: Option & bool = False

        # Sync changes to the benchmark directory
        sync: Option & bool = False

        # Dev mode (adds --sync, use current venv)
        dev: Option & bool = False

        mp = _get_multipack(dev=dev)
        mp.do_install(dash=simple_dash, force=force, sync=sync)