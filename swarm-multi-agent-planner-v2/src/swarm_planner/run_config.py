"""`--config file.yaml` support for the run scripts.

A config file is a flat YAML mapping whose keys are the long option names of the script it is given
to (without the leading dashes), for example:

    aco: true
    agents_per_task: 8
    max_cost_usd: 12

Precedence, from strongest to weakest: command-line flags > config file > the script's built-in
defaults. So `--seed 1` always wins over a `seed:` entry, and a config only has to name what differs
from the defaults. Unknown keys and values outside an option's allowed choices are rejected, so a
typo in a config cannot silently run the wrong experiment.

Relative paths in a config (`data_path`, `run_dir`, `cache_path`) are resolved against the project
root (the folder that contains `configs/`), not against the current directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PATH_KEYS = ("data_path", "run_dir", "cache_path")
NO_VALUE_STRINGS = {"none"}  # `cache_path: none` disables the cache, exactly like `--cache_path none`


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        metavar="FILE.yaml",
        help="YAML file with default values for the options of this script (see configs/). "
        "Command-line flags override it.",
    )


def load_config_file(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping of option names to values, got {type(data).__name__}")
    return data


def _actions_by_dest(parser: argparse.ArgumentParser) -> Dict[str, argparse.Action]:
    return {a.dest: a for a in parser._actions if a.dest not in ("help", "config")}


def _coerce(parser: argparse.ArgumentParser, key: str, value: Any, action: argparse.Action, source: str) -> Any:
    if value is not None and not isinstance(value, bool) and callable(action.type):
        try:
            value = action.type(value)
        except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
            parser.error(f"{source}: bad value for '{key}': {value!r} ({exc})")
    if action.choices is not None and value not in action.choices:
        parser.error(f"{source}: '{key}' must be one of {list(action.choices)}, got {value!r}")
    return value


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Optional[List[str]] = None,
    project_root: str | Path | None = None,
) -> argparse.Namespace:
    """Parse `argv`, using the file named by `--config` (if any) for defaults.

    `parser` must already have been given `add_config_argument`.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)

    if known.config:
        source = str(known.config)
        values = load_config_file(known.config)
        actions = _actions_by_dest(parser)
        unknown = sorted(set(values) - set(actions))
        if unknown:
            parser.error(f"{source}: unknown option(s) {unknown}; valid keys: {sorted(actions)}")
        resolved: Dict[str, Any] = {}
        for key, value in values.items():
            value = _coerce(parser, key, value, actions[key], source)
            if key in PATH_KEYS and isinstance(value, str) and value.strip().lower() not in NO_VALUE_STRINGS:
                path = Path(value)
                if not path.is_absolute() and project_root is not None:
                    value = str(Path(project_root) / path)
            resolved[key] = value
        parser.set_defaults(**resolved)

    return parser.parse_args(argv)
