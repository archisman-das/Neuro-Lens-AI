"""Central loader for config.yaml.

Previously nothing in the repo actually read config.yaml — every script used
argparse defaults instead, so the YAML file was decorative. This module loads
config.yaml once and exposes:

    load_config(path=None) -> dict
    section(name, default=None) -> dict | None
    merge_args(args, section_name, mapping=None) -> Namespace

`merge_args` lets training scripts use config values as defaults and override
them from the CLI. Example:

    from src.config_loader import merge_args
    args = parser.parse_args()
    args = merge_args(args, 'training')

Only the sections whose consumers exist in the repo are considered "live":
data, model, training, kfold, ablation, evaluation, experiment, paths,
hardware, random, robustness, uncertainty, multiclass.

The 3D-MRI / federated / SSL sections still exist in config.yaml as reference
parameter sets used by the corresponding classes in src/advanced_models.py.
They are no longer silently ignored — when the relevant module is invoked, it
can fetch its section from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config.yaml'
_CACHE: dict[str, Any] = {}


def load_config(path: Optional[str | Path] = None) -> dict:
    """Load and cache config.yaml. Pass a path to override the default location."""
    path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    cache_key = str(path.resolve())
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    with path.open('r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh) or {}
    _CACHE[cache_key] = cfg
    return cfg


def section(name: str, default: Optional[Mapping[str, Any]] = None, *, path: Optional[str | Path] = None) -> dict:
    """Return a top-level section of the config, or the default if missing."""
    cfg = load_config(path)
    value = cfg.get(name)
    if value is None:
        return dict(default) if default is not None else {}
    if not isinstance(value, Mapping):
        raise TypeError(f'Config section {name!r} must be a mapping, got {type(value).__name__}.')
    return dict(value)


def merge_args(args, section_name: str, mapping: Optional[Mapping[str, str]] = None, *, path: Optional[str | Path] = None):
    """Fill in argparse defaults from the named config section.

    `mapping` maps `argparse_attr_name -> config_key_name`. If None, attributes
    are matched against YAML keys of the same name.

    A CLI value (anything that differs from the argparse default) wins over the
    YAML value. The argparse default itself is treated as "unset" so YAML
    overrides it. To detect "default vs explicit", we compare against the
    parser's recorded defaults; that requires the caller to pass the parser via
    args._parser, OR to call this BEFORE parse_args by supplying the parser
    object directly (see set_yaml_defaults).
    """
    cfg = section(section_name, path=path)
    if not cfg:
        return args
    keys = mapping or {k: k for k in vars(args).keys() if k in cfg}
    for attr, yaml_key in keys.items():
        if yaml_key not in cfg:
            continue
        # Only override if the user did not pass the flag (i.e. attr equals the
        # parser's stored default). When we can't tell, prefer the CLI value.
        current = getattr(args, attr, None)
        parser_defaults = getattr(args, '_parser_defaults', {})
        default = parser_defaults.get(attr)
        if current == default:
            setattr(args, attr, cfg[yaml_key])
    return args


def set_yaml_defaults(parser, section_name: str, mapping: Optional[Mapping[str, str]] = None, *, path: Optional[str | Path] = None):
    """Apply YAML values as argparse defaults BEFORE parse_args is called.

    Preferred over merge_args when you control parser construction. Example:

        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=10)
        set_yaml_defaults(parser, 'training')   # may set parser default to 100
        args = parser.parse_args()              # CLI still overrides

    Returns the parser for chaining.
    """
    cfg = section(section_name, path=path)
    if not cfg:
        return parser
    if mapping is None:
        # Match by action.dest -> yaml key of the same name.
        for action in parser._actions:  # noqa: SLF001
            if action.dest in cfg:
                action.default = cfg[action.dest]
        return parser
    for dest, yaml_key in mapping.items():
        if yaml_key in cfg:
            for action in parser._actions:  # noqa: SLF001
                if action.dest == dest:
                    action.default = cfg[yaml_key]
                    break
    return parser
