"""--config support of the run scripts, and the shipped configs/ files."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIGS = ROOT / "configs"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_script(name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path: Path, text: str) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


# --- mechanics -----------------------------------------------------------------------------------


def test_without_config_the_defaults_are_unchanged():
    run_experiment = _load_script("run_experiment")
    args = run_experiment.parse_args([])
    assert args.aco is True and args.agents_per_task == 8 and args.iterations_per_task == 6
    assert args.selection_policy == "verified_last3" and args.max_cost_usd is None


def test_config_overrides_defaults_and_flags_override_config(tmp_path):
    run_experiment = _load_script("run_experiment")
    cfg = _write(tmp_path, "aco: false\nagents_per_task: 3\nmax_cost_usd: 12\nseed: 2\n")
    args = run_experiment.parse_args(["--config", cfg])
    assert args.aco is False and args.agents_per_task == 3 and args.seed == 2
    assert args.max_cost_usd == 12.0 and isinstance(args.max_cost_usd, float)
    args = run_experiment.parse_args(["--config", cfg, "--seed", "1", "--aco", "true"])
    assert args.seed == 1 and args.aco is True and args.agents_per_task == 3  # flags win


def test_relative_paths_resolve_against_the_project_root(tmp_path):
    run_experiment = _load_script("run_experiment")
    cfg = _write(tmp_path, "data_path: data/dev_100.json\ncache_path: none\n")
    args = run_experiment.parse_args(["--config", cfg])
    assert Path(args.data_path) == ROOT / "data" / "dev_100.json"
    assert args.cache_path == "none"  # the 'no cache' keyword is kept as it is


def test_unknown_key_is_rejected(tmp_path, capsys):
    run_experiment = _load_script("run_experiment")
    with pytest.raises(SystemExit):
        run_experiment.parse_args(["--config", _write(tmp_path, "agents_per_taks: 8\n")])
    assert "agents_per_taks" in capsys.readouterr().err


def test_value_outside_choices_or_wrong_type_is_rejected(tmp_path, capsys):
    run_experiment = _load_script("run_experiment")
    with pytest.raises(SystemExit):
        run_experiment.parse_args(["--config", _write(tmp_path, "selection_policy: not_a_rule\n")])
    assert "selection_policy" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run_experiment.parse_args(["--config", _write(tmp_path, "rho: lots\n")])
    assert "rho" in capsys.readouterr().err


def test_missing_and_malformed_config_files_fail_clearly(tmp_path):
    run_experiment = _load_script("run_experiment")
    with pytest.raises(FileNotFoundError):
        run_experiment.parse_args(["--config", str(tmp_path / "nope.yaml")])
    with pytest.raises(ValueError, match="mapping"):
        run_experiment.parse_args(["--config", _write(tmp_path, "- a\n- b\n")])


# --- the shipped configs -------------------------------------------------------------------------

SHIPPED = {
    "baseline": "run_baseline",
    "tot_only": "run_experiment",
    "tot_aco": "run_experiment",
    "sampling_control": "run_sampling_control",
    "verifier_solver": "run_verifier_solver",
}


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_every_shipped_config_is_valid_for_its_script(name):
    script = _load_script(SHIPPED[name])
    args = script.parse_args(["--config", str(CONFIGS / f"{name}.yaml")])
    assert Path(args.data_path) == ROOT / "data" / "curated_400.json"
    keys = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
    assert "seed" not in keys and "run_dir" not in keys  # seeds and output folders come from the command line


# resolved-config key in a finished run  ->  option name in the shipped config
RESOLVED_KEYS = {
    "model": "model",
    "agents_per_task": "agents_per_task",
    "iterations_per_task": "iterations_per_task",
    "alpha": "alpha",
    "epsilon": "epsilon",
    "rho": "rho",
    "aco_enabled": "aco",
    "selection_policy": "selection_policy",
    "samples_per_task": "samples",
    "task_concurrency": "task_concurrency",
    "max_concurrent_requests": "max_concurrent_requests",
    "max_cost_usd": "max_cost_usd",
}


@pytest.mark.parametrize("name", ["baseline", "tot_only", "tot_aco", "sampling_control"])
def test_shipped_config_matches_what_the_reported_runs_used(name):
    """configs/<name>.yaml must say what results/runs/<name>/seed_0 actually ran with."""
    resolved_path = ROOT / "results" / "runs" / name / "seed_0" / "config.resolved.json"
    if not resolved_path.exists():
        pytest.skip("reported runs are not present")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    script = _load_script(SHIPPED[name])
    args = vars(script.parse_args(["--config", str(CONFIGS / f"{name}.yaml")]))
    checked = 0
    for resolved_key, option in RESOLVED_KEYS.items():
        if resolved_key in resolved and option in args:
            if resolved_key == "selection_policy" and name in ("baseline", "sampling_control"):
                continue  # fixed by the runner, not an option
            assert args[option] == resolved[resolved_key], f"{name}: {option}"
            checked += 1
    if name == "sampling_control":
        assert args["temperature"] == resolved["llm_params"]["temperature"]
        checked += 1
    assert checked >= 4
