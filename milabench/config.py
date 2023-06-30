import socket

import yaml
from omegaconf import OmegaConf

from .fs import XPath
from .merge import merge


def relative_to(pth, cwd):
    pth = XPath(pth).expanduser()
    if not pth.is_absolute():
        pth = (XPath(cwd) / pth).resolve()
    return pth


def _config_layers(config_files):
    for config_file in config_files:
        if isinstance(config_file, dict):
            yield config_file
        else:
            config_file = XPath(config_file).absolute()
            config_base = config_file.parent
            with open(config_file) as cf:
                config = yaml.safe_load(cf)
                includes = config.pop("include", [])
                if isinstance(includes, str):
                    includes = [includes]
                yield from _config_layers(
                    relative_to(incl, config_base) for incl in includes
                )
                for v in config.values():
                    assert isinstance(v, dict)
                    v.setdefault("config_base", str(config_base))
                    v.setdefault("config_file", str(config_file))
                    v.setdefault("dirs", {})
                yield config


def resolve_inheritance(bench_config, all_configs):
    while inherit := bench_config.pop("inherits", None):
        parent = all_configs[inherit]
        tags = {*parent.get("tags", []), *bench_config.get("tags", [])}
        bench_config = merge(parent, bench_config)
        bench_config["tags"] = sorted(tags)

    if "*" in all_configs:
        bench_config = merge(bench_config, all_configs["*"])

    return bench_config


def finalize_config(name, bench_config):
    bench_config["name"] = name
    if "definition" in bench_config:
        pack = XPath(bench_config["definition"]).expanduser()
        if not pack.is_absolute():
            pack = (XPath(bench_config["config_base"]) / pack).resolve()
            bench_config["definition"] = str(pack)

    bench_config["tag"] = [bench_config["name"]]

    bench_config = OmegaConf.to_object(OmegaConf.create(bench_config))
    return bench_config


def build_config(*config_files):
    all_configs = {}
    for layer in _config_layers(config_files):
        all_configs = merge(all_configs, layer)
    for name, bench_config in all_configs.items():
        all_configs[name] = resolve_inheritance(bench_config, all_configs)
    for name, bench_config in all_configs.items():
        all_configs[name] = finalize_config(name, bench_config)
    return all_configs


def resolve_addresses(nodes):
    # Note: it is possible for self to be none
    # if we are running milabench on a node that is not part of the system
    # in that case it should still work; the local is then going to
    # ssh into the main node which will dispatch the work to the other nodes
    self = None

    for node in nodes:
        # Resolve the IP
        hostname, aliaslist, ipaddrlist = socket.gethostbyaddr(node["ip"])

        node["hostname"] = hostname
        node["aliaslist"] = aliaslist
        node["ipaddrlist"] = ipaddrlist

        is_local = hostname == socket.gethostname()
        node["local"] = is_local

        if is_local:
            self = node

    return self


def build_system_config(config_file, defaults=None):
    """Load the system configuration, verify its validity and resolve ip addresses

    Notes
    -----
    * node['local'] true when the code is executing on the machine directly
    * node["main"] true when the machine is in charge of distributing the workload
    """

    if config_file is None:
        config = {}
    else:
        config_file = XPath(config_file).absolute()
        with open(config_file) as cf:
            config = yaml.safe_load(cf)

    if defaults:
        config = merge(defaults, config)

    sys_cfg = config["system"]

    if sys_cfg["sshkey"] is not None:
        sys_cfg["sshkey"] = str(XPath(sys_cfg["sshkey"]).resolve())

    main_node = None
    aliases = {}
    for i, node in enumerate(sys_cfg["nodes"]):
        for field in ("name", "ip", "user"):
            _name = node.get("name", None)
            assert node[field], f"The `{field}` of the node `{_name}` is missing"

        assert node["name"] not in aliases, (
            f"Usage of name {node['name']} for multiple nodes"
        )

        aliases[node["name"]] = node

        if node.get("main", False) and not main_node:
            main_node = node
            sys_cfg["nodes"][i] = None

    # Helpers
    sys_cfg["nodes"] = [*([main_node] if main_node is not None else []),
                        *[n for n in sys_cfg["nodes"] if n is not None]]
    sys_cfg["self"] = resolve_addresses(sys_cfg["nodes"])
    sys_cfg["aliases"] = aliases

    return config
