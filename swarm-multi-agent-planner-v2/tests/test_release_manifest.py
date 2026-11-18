"""scripts/make_release_manifest.py: hashing, exclusions, verification."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("make_release_manifest", ROOT / "scripts" / "make_release_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(tmp_path: Path) -> Path:
    (tmp_path / "results" / "runs" / "a" / "__pycache__").mkdir(parents=True)
    (tmp_path / "results" / "runs" / "a" / "predictions.json").write_text("{}", encoding="utf-8")
    (tmp_path / "results" / "runs" / "a" / "__pycache__" / "x.pyc").write_bytes(b"x")
    (tmp_path / "results" / "cache").mkdir()
    (tmp_path / "results" / "cache" / "llm_cache.sqlite").write_bytes(b"cache")
    (tmp_path / "results" / "cache" / "llm_cache.sqlite-wal").write_bytes(b"wal")
    (tmp_path / "results" / "cache" / "llm_cache.backup.sqlite").write_bytes(b"old")
    return tmp_path


def test_manifest_lists_included_files_with_correct_hashes(tmp_path):
    m = _load()
    root = _tree(tmp_path)
    manifest = root / "results" / "MANIFEST.sha256"
    entries = m.build_manifest(root, ["results/runs", "results/cache/llm_cache.sqlite", "missing_dir"], manifest)
    assert [rel for _, rel in entries] == ["results/cache/llm_cache.sqlite", "results/runs/a/predictions.json"]
    cache_hash = hashlib.sha256(b"cache").hexdigest()
    assert manifest.read_text(encoding="utf-8").splitlines()[0] == f"{cache_hash}  results/cache/llm_cache.sqlite"


def test_manifest_excludes_itself_sidecars_backups_and_pycache(tmp_path):
    m = _load()
    root = _tree(tmp_path)
    manifest = root / "results" / "MANIFEST.sha256"
    m.build_manifest(root, ["results"], manifest)
    m.build_manifest(root, ["results"], manifest)  # a second build must not list the first manifest
    text = manifest.read_text(encoding="utf-8")
    for unwanted in ("-wal", "backup", "__pycache__", "MANIFEST"):
        assert unwanted not in text


def test_verify_detects_changed_and_missing_files(tmp_path):
    m = _load()
    root = _tree(tmp_path)
    manifest = root / "results" / "MANIFEST.sha256"
    m.build_manifest(root, ["results/runs", "results/cache/llm_cache.sqlite"], manifest)
    assert m.verify_manifest(root, manifest) == (2, [], [])
    (root / "results" / "runs" / "a" / "predictions.json").write_text('{"x": 1}', encoding="utf-8")
    (root / "results" / "cache" / "llm_cache.sqlite").unlink()
    checked, missing, changed = m.verify_manifest(root, manifest)
    assert checked == 2
    assert missing == ["results/cache/llm_cache.sqlite"]
    assert changed == ["results/runs/a/predictions.json"]
