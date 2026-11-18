#!/usr/bin/env python3
"""SHA-256 manifest of everything that goes into the release archive (logs, results, cache, data, configs).

    python scripts/make_release_manifest.py                # write results/MANIFEST.sha256
    python scripts/make_release_manifest.py --verify       # check the files against that manifest

The manifest uses the `sha256sum` format (`<hash>  <relative path>`, forward slashes), so it can also be
checked with `sha256sum -c`. Included by default: configs/, data/, results/runs/, results/aggregate/,
results/path_table/, results/verifier_audit/ and the LLM cache results/cache/llm_cache.sqlite.
Left out: sqlite -wal/-shm sidecar files, cache backups, __pycache__ and the manifest itself.
Build the manifest only when no run is writing to the cache (the cache is hashed as one file).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = (
    "configs",
    "data",
    "results/runs",
    "results/aggregate",
    "results/path_table",
    "results/verifier_audit",
    "results/cache/llm_cache.sqlite",
)
DEFAULT_MANIFEST = "results/MANIFEST.sha256"
SKIP_SUFFIXES = (".sqlite-wal", ".sqlite-shm", ".pyc")
SKIP_NAMES = ("llm_cache.backup.sqlite",)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_files(root: Path, paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for entry in paths:
        target = root / entry
        if target.is_file():
            candidates = [target]
        elif target.is_dir():
            candidates = [p for p in target.rglob("*") if p.is_file()]
        else:
            print(f"warning: {entry} does not exist, skipped", file=sys.stderr)
            continue
        for path in candidates:
            if "__pycache__" in path.parts or path.name in SKIP_NAMES or path.name.endswith(SKIP_SUFFIXES):
                continue
            files.append(path)
    return sorted(set(files), key=lambda p: p.relative_to(root).as_posix())


def build_manifest(root: Path, paths: Iterable[str], manifest: Path) -> List[Tuple[str, str]]:
    entries = []
    for path in collect_files(root, paths):
        if path.resolve() == manifest.resolve():
            continue
        entries.append((sha256_of(path), path.relative_to(root).as_posix()))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(f"{h}  {rel}\n" for h, rel in entries), encoding="utf-8", newline="\n")
    return entries


def verify_manifest(root: Path, manifest: Path) -> Tuple[int, List[str], List[str]]:
    """Return (files checked, missing files, files whose hash differs)."""
    missing: List[str] = []
    changed: List[str] = []
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        path = root / rel
        if not path.is_file():
            missing.append(rel)
        elif sha256_of(path) != expected:
            changed.append(rel)
        checked += 1
    return checked, missing, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="Project root (paths are relative to it).")
    parser.add_argument("--paths", nargs="+", default=list(DEFAULT_PATHS), help="Files or folders to include.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Manifest file, relative to --root.")
    parser.add_argument("--verify", action="store_true", help="Check the files against the manifest instead of writing it.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest = Path(args.manifest) if Path(args.manifest).is_absolute() else root / args.manifest
    if args.verify:
        checked, missing, changed = verify_manifest(root, manifest)
        for rel in missing:
            print(f"MISSING  {rel}")
        for rel in changed:
            print(f"CHANGED  {rel}")
        print(f"{checked} files checked, {len(missing)} missing, {len(changed)} changed")
        return 1 if (missing or changed) else 0
    entries = build_manifest(root, args.paths, manifest)
    print(f"wrote {manifest} ({len(entries)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
