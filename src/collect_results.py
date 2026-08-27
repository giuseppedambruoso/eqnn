"""Collects the essential artifacts (final_model.pt + summary.json) from
every completed training run under outputs/ and multirun/ into a small,
git-trackable results_def/ directory — a curated subset, not a full mirror:
excludes loss_history.csv/.jpg, confusion_matrix.png, circuit.txt/.png,
which stay local-only (outputs/ and multirun/ are gitignored), since
wandb already has the complete artifact for every run.

Also builds results_def/summary.csv, one row per run, pulling the full input
config (architecture, N, seed, readout, augment_train, class1, class2,
num_qubits, reps, device, img_size, dataset) alongside the accuracy
metrics out of each summary.json — so every accuracy number is traceable
to the exact config that produced it without opening 200+ files. The
complete summary.json (including per-parameter final_params) is also
kept alongside each model's final_model.pt for the full detail.

Runs that share the exact same identity (architecture, N, seed, dataset,
readout, augment_train, class1, class2, device, num_qubits, reps — i.e.
an accidental duplicate local run, not two different seeds) are deduped
before copying: only the one with the most recently modified summary.json
is kept, so a duplicate never gets counted or copied twice.

Doesn't touch wandb or the training pipeline — a standalone filesystem
utility, no network access needed. Safe to re-run any time (idempotent):
each run overwrites its own results_def/ subdirectory and summary.csv is
rebuilt from scratch every time, so it naturally picks up new training
runs on the next call.

Usage:
    python3 src/collect_results.py
    python3 src/collect_results.py --outputs-dir outputs --multirun-dir multirun --results-dir results_def
"""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

SUMMARY_CSV_COLUMNS = [
    "run_dir",
    "run_name",
    "architecture",
    "N",
    "seed",
    "dataset",
    "readout",
    "augment_train",
    "class1",
    "class2",
    "device",
    "num_qubits",
    "reps",
    "img_size",
    "epochs_configured",
    "epochs_completed",
    "train_accuracy",
    "val_accuracy",
    "val_aug_accuracy",
    "p4m_is_invariant",
]

# A duplicate is a run sharing every one of these — NOT epochs/learning_rate/
# patience/min_delta (irrelevant to what model got trained) and NOT
# accuracy/timing fields (those are outcomes, not identity).
IDENTITY_FIELDS = [
    "architecture",
    "N",
    "seed",
    "dataset",
    "readout",
    "augment_train",
    "class1",
    "class2",
    "device",
    "num_qubits",
    "reps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--multirun-dir", default="multirun")
    parser.add_argument("--results-dir", default="results_def")
    return parser.parse_args()


def _find_run_dirs(*roots: str) -> list[Path]:
    """Every directory containing a final_model.pt, across all given roots."""
    run_dirs: list[Path] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        run_dirs.extend(p.parent for p in root_path.rglob("final_model.pt"))
    return sorted(run_dirs)


def _relative_mirror_path(path: Path) -> Path:
    """Strips any absolute anchor so a run_dir mirrors safely under
    results_root even if --outputs-dir/--multirun-dir were passed as
    absolute paths — joining an absolute Path onto another discards the
    left side entirely, which would make the "mirror" collide with the
    original (source == dest, breaking the copy)."""
    return Path(*path.parts[1:]) if path.is_absolute() else path


def _copy_essentials(
    run_dir: Path, results_root: Path, mirror_path: Path | None = None
) -> Path:
    """Mirrors run_dir's path under results_root and copies only
    final_model.pt + summary.json into it (skipping the other, bulkier
    per-run files). run_dir is always used to actually locate the source
    files; mirror_path controls where they land under results_root and
    defaults to run_dir itself (anchor-stripped) — pass it explicitly
    when run_dir is an absolute path that must be read regardless of the
    current working directory, but the mirrored structure should be
    anchored to some other root instead (e.g. train.py calling this after
    Hydra has already chdir'd into the job's own output directory)."""
    if mirror_path is None:
        mirror_path = _relative_mirror_path(run_dir)
    dest_dir = results_root / mirror_path
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "final_model.pt", dest_dir / "final_model.pt")
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        shutil.copy2(summary_path, dest_dir / "summary.json")
    return dest_dir


def _summary_row(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    config = summary.get("config", {})
    p4m = summary.get("p4m_equivariance", {})
    return {
        "run_dir": str(run_dir),
        "run_name": summary.get("run_name"),
        "architecture": config.get("architecture"),
        "N": summary.get("N"),
        "seed": summary.get("seed"),
        "dataset": summary.get("dataset"),
        "readout": config.get("readout"),
        "augment_train": config.get("augment_train"),
        "class1": config.get("class1", 3),
        "class2": config.get("class2", 4),
        "device": config.get("device"),
        "num_qubits": config.get("num_qubits"),
        "reps": config.get("reps"),
        "img_size": config.get("img_size"),
        "epochs_configured": summary.get("epochs_configured"),
        "epochs_completed": summary.get("epochs_completed"),
        "train_accuracy": summary.get("train_accuracy"),
        "val_accuracy": summary.get("val_accuracy"),
        "val_aug_accuracy": summary.get("val_aug_accuracy"),
        "p4m_is_invariant": p4m.get("is_invariant"),
    }


def _identity_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in IDENTITY_FIELDS)


def _dedupe_run_dirs(
    candidates: list[tuple[Path, dict[str, Any]]],
) -> tuple[list[tuple[Path, dict[str, Any]]], int]:
    """Groups candidates by identity and keeps only the one with the most
    recently modified summary.json per group — every candidate here
    already has a usable summary.json (that's how row was built), so
    there's no "finished vs crashed" distinction to make, just recency.
    Returns (kept, number of duplicates dropped)."""
    groups: dict[tuple[Any, ...], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for run_dir, row in candidates:
        groups[_identity_key(row)].append((run_dir, row))

    kept: list[tuple[Path, dict[str, Any]]] = []
    duplicates_dropped = 0
    for group in groups.values():
        if len(group) == 1:
            kept.append(group[0])
            continue
        group_sorted = sorted(
            group,
            key=lambda item: (item[0] / "summary.json").stat().st_mtime,
            reverse=True,
        )
        kept.append(group_sorted[0])
        duplicates_dropped += len(group_sorted) - 1
    return kept, duplicates_dropped


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_dir)

    run_dirs = _find_run_dirs(args.outputs_dir, args.multirun_dir)
    if not run_dirs:
        print(
            f"No final_model.pt found under {args.outputs_dir!r}/{args.multirun_dir!r}."
        )
        return

    candidates: list[tuple[Path, dict[str, Any]]] = []
    skipped_no_summary = 0
    for run_dir in run_dirs:
        row = _summary_row(run_dir)
        if row is None:
            skipped_no_summary += 1
            continue
        candidates.append((run_dir, row))

    kept, duplicates_dropped = _dedupe_run_dirs(candidates)

    rows = []
    for run_dir, row in kept:
        dest_dir = _copy_essentials(run_dir, results_root)
        row["run_dir"] = str(dest_dir)
        rows.append(row)

    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Found {len(run_dirs)} run(s) under "
        f"{args.outputs_dir!r}/{args.multirun_dir!r}: "
        f"{skipped_no_summary} skipped (no summary.json), "
        f"{duplicates_dropped} skipped (duplicate config+seed), "
        f"{len(rows)} collected into {results_root}/."
    )
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
