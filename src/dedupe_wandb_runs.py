"""Finds wandb runs that are exact duplicates of the same input
configuration — every training-config field identical, INCLUDING seed (two
different seeds of an otherwise-identical config are independent samples,
not duplicates — only an accidental re-run of the exact same job counts).

Defaults to a DRY RUN: prints, for every duplicate group, which run would
be kept (the one with a usable val/accuracy summary, breaking ties by the
most recently created) and which would be deleted, without deleting
anything. Pass --delete to actually delete the wandb runs marked "would
delete" — this is irreversible, review the dry-run output first.

Usage (reuses the eqnn image, which already has wandb + the .env API key
wired via docker-compose.yml's env_file):

    docker compose run --rm --entrypoint python3 eqnn src/dedupe_wandb_runs.py
    docker compose run --rm --entrypoint python3 eqnn src/dedupe_wandb_runs.py --architecture config6 config7
    docker compose run --rm --entrypoint python3 eqnn src/dedupe_wandb_runs.py --architecture config6 config7 --delete
"""

import argparse
import os
from collections import defaultdict
from typing import Any, Protocol

import wandb

IDENTITY_FIELDS = [
    "architecture",
    "N",
    "dataset",
    "img_size",
    "num_qubits",
    "reps",
    "device",
    "readout",
    "epochs",
    "learning_rate",
    "patience",
    "min_delta",
    "augment_train",
    "seed",
]


class RunLike(Protocol):
    """The subset of wandb.apis.public.Run this module relies on — narrowed
    to a Protocol so the grouping/selection logic can be unit-tested with
    lightweight fakes instead of live wandb runs."""

    id: str
    name: str
    created_at: str
    config: dict[str, Any]
    summary: dict[str, Any]

    def delete(self) -> None: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find (and optionally delete) wandb runs that duplicate the "
            "exact same input configuration, including seed."
        )
    )
    parser.add_argument(
        "--architecture",
        nargs="+",
        default=None,
        help="Restrict to these architectures (default: every run in the project).",
    )
    parser.add_argument(
        "--project", default=None, help="Defaults to $WANDB_PROJECT or 'eqnn'."
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="Defaults to your wandb account's default entity.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete the duplicate runs. Without this flag, only "
        "prints what would be deleted (dry run) — irreversible, review the "
        "dry-run output first.",
    )
    return parser.parse_args()


def _identity_key(cfg: dict[str, Any]) -> tuple[Any, ...] | None:
    """None if any identity field is missing — such a run can't be safely
    compared, so it's excluded from dedup entirely rather than risking a
    false-positive match on missing (None, None, ...) fields."""
    values = tuple(cfg.get(field) for field in IDENTITY_FIELDS)
    return None if None in values else values


def _pick_keep_and_drop(runs: list[RunLike]) -> tuple[RunLike, list[RunLike]]:
    """Within a duplicate group, keep the run with a usable val/accuracy
    summary (a crashed/incomplete run is a worse candidate to keep than a
    finished one, regardless of recency), breaking ties by the most
    recently created run — everything else in the group is to be dropped."""

    def sort_key(run: RunLike) -> tuple[bool, str]:
        return ("val/accuracy" in run.summary, run.created_at)

    runs_sorted = sorted(runs, key=sort_key, reverse=True)
    return runs_sorted[0], runs_sorted[1:]


def main() -> None:
    args = parse_args()
    project = args.project or os.environ.get("WANDB_PROJECT", "eqnn")

    api = wandb.Api()
    resolved_entity = args.entity or api.default_entity
    path = f"{resolved_entity}/{project}" if resolved_entity else project

    filters = (
        {"config.architecture": {"$in": args.architecture}} if args.architecture else {}
    )
    runs = api.runs(path, filters=filters)

    groups: dict[tuple[Any, ...], list[RunLike]] = defaultdict(list)
    skipped = 0
    for run in runs:
        key = _identity_key(run.config)
        if key is None:
            skipped += 1
            continue
        groups[key].append(run)

    if skipped:
        print(f"Skipped {skipped} run(s) missing one or more identity fields.")

    duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}
    if not duplicate_groups:
        print(f"No duplicate configurations found in {path!r}.")
        return

    total_to_drop = 0
    for key, group_runs in duplicate_groups.items():
        keep, drop = _pick_keep_and_drop(group_runs)
        total_to_drop += len(drop)
        config_desc = dict(zip(IDENTITY_FIELDS, key, strict=True))
        print(f"\n{config_desc}: {len(group_runs)} runs")
        print(
            f"  KEEP    {keep.id} ({keep.name}) created_at={keep.created_at} "
            f"val/accuracy={keep.summary.get('val/accuracy')}"
        )
        for run in drop:
            action = "DELETE" if args.delete else "would delete"
            print(
                f"  {action:<12} {run.id} ({run.name}) created_at={run.created_at} "
                f"val/accuracy={run.summary.get('val/accuracy')}"
            )
            if args.delete:
                run.delete()

    if args.delete:
        print(f"\nDeleted {total_to_drop} duplicate run(s).")
    else:
        print(
            f"\nDry run only — {total_to_drop} run(s) would be deleted, "
            "nothing was actually deleted. Re-run with --delete to apply."
        )


if __name__ == "__main__":
    main()
