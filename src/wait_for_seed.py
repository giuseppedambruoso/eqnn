"""Polls wandb until every requested (architecture, N, augment_train)
combination has a finished run (usable val/accuracy) for one specific
seed, then exits 0 — meant to gate a `&&` in a shell one-liner so a
running sweep can be stopped automatically as soon as a target seed
finishes, without waiting for the rest of the sweep to complete:

    docker compose run --rm --entrypoint python3 eqnn src/wait_for_seed.py \\
        --seed 5 --architecture config6 config7 --readout x0_xhalf \\
        --n-values 40 80 160 320 640 --augment-train none online once \\
    && kill -INT <sweep_pid>

Doesn't touch the training pipeline — a standalone polling utility over
already-logged wandb runs, alongside plot_wandb_results.py and
dedupe_wandb_runs.py (and it reuses the same legacy-bool augment_train
normalization those two need).
"""

import argparse
import os
import sys
import time
from itertools import product
from typing import Any

import wandb


def _normalize_augment_train(value: Any) -> Any:
    """See plot_wandb_results.py / dedupe_wandb_runs.py: runs logged
    before the "once" mode existed used a bool (True/False) instead of
    today's "online"/"none"."""
    if isinstance(value, bool):
        return "online" if value else "none"
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return "online" if value.lower() == "true" else "none"
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait until a sweep has finished every (architecture, N, "
            "augment_train) combination for one seed."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--architecture", nargs="+", required=True)
    parser.add_argument("--n-values", type=int, nargs="+", required=True)
    parser.add_argument(
        "--augment-train", nargs="+", default=["none", "online", "once"]
    )
    parser.add_argument(
        "--readout",
        default=None,
        help="Only count runs with this readout. Default: any readout.",
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
        "--poll-interval",
        type=float,
        default=60.0,
        help="Seconds between wandb checks (default: 60).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Give up and exit 1 after this many seconds (default: wait forever).",
    )
    return parser.parse_args()


def _expected_combos(
    architectures: list[str], n_values: list[int], augment_train_modes: list[str]
) -> set[tuple[str, int, str]]:
    return set(product(architectures, n_values, augment_train_modes))


def _finished_combos(
    architectures: list[str],
    seed: int,
    readout: str | None,
    project: str,
    entity: str | None,
) -> set[tuple[str, int, str]]:
    api = wandb.Api()
    resolved_entity = entity or api.default_entity
    path = f"{resolved_entity}/{project}" if resolved_entity else project

    finished: set[tuple[str, int, str]] = set()
    for architecture in architectures:
        runs = api.runs(
            path,
            filters={"config.architecture": architecture, "config.seed": seed},
        )
        for run in runs:
            cfg = run.config
            if run.summary.get("val/accuracy") is None:
                continue
            if readout is not None and cfg.get("readout") != readout:
                continue
            N = cfg.get("N")
            augment_train = _normalize_augment_train(cfg.get("augment_train"))
            if N is None or augment_train is None:
                continue
            finished.add((architecture, N, augment_train))
    return finished


def main() -> None:
    args = parse_args()
    project = args.project or os.environ.get("WANDB_PROJECT", "eqnn")

    expected = _expected_combos(args.architecture, args.n_values, args.augment_train)
    start = time.monotonic()

    while True:
        finished = _finished_combos(
            args.architecture, args.seed, args.readout, project, args.entity
        )
        done = expected & finished
        print(
            f"seed={args.seed}: {len(done)}/{len(expected)} combinations finished",
            flush=True,
        )
        if expected.issubset(finished):
            print("All target combinations finished.")
            return
        if args.timeout is not None and time.monotonic() - start > args.timeout:
            print("Timed out waiting.", file=sys.stderr)
            sys.exit(1)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
