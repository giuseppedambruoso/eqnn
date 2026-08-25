"""Plots val/accuracy and val_aug/accuracy vs N from wandb runs, averaged
over seeds, one line per augment_train value and one row of plots per
readout value found. Doesn't touch the training pipeline — a standalone
analysis utility over already-logged wandb runs.

Usage (reuses the eqnn image, which already has wandb + the .env API key
wired via docker-compose.yml's env_file):

    docker compose run --rm --entrypoint python3 eqnn src/plot_wandb_results.py --architecture config6

Output: a PNG (default val_accuracy_vs_N.png) saved in the working
directory (mount ./outputs or similar if you want it to land on the host).
"""

import argparse
import os
from collections import defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot val/accuracy and val_aug/accuracy vs N from wandb, "
            "averaged over seeds, split by augment_train and readout."
        )
    )
    parser.add_argument("--architecture", default="config6")
    parser.add_argument(
        "--project", default=None, help="Defaults to $WANDB_PROJECT or 'eqnn'."
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="Defaults to your wandb account's default entity.",
    )
    parser.add_argument("--output", default="val_accuracy_vs_N.png")
    return parser.parse_args()


def _fetch_grouped_results(
    architecture: str, project: str, entity: str | None
) -> dict[tuple[Any, Any], dict[int, dict[str, list[float]]]]:
    """Returns {(augment_train, readout): {N: {"val": [...], "val_aug": [...]}}},
    one accuracy value per finished run with a usable summary — runs that
    crashed or never reached the final validation step (no val/accuracy in
    their summary) are silently skipped.
    """
    api = wandb.Api()
    resolved_entity = entity or api.default_entity
    path = f"{resolved_entity}/{project}" if resolved_entity else project

    runs = api.runs(path, filters={"config.architecture": architecture})

    grouped: dict[tuple[Any, Any], dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"val": [], "val_aug": []})
    )
    for run in runs:
        cfg = run.config
        summary = run.summary
        N = cfg.get("N")
        augment_train = cfg.get("augment_train")
        readout = cfg.get("readout")
        val_acc = summary.get("val/accuracy")
        val_aug_acc = summary.get("val_aug/accuracy")
        if None in (N, augment_train, val_acc, val_aug_acc):
            continue
        grouped[(augment_train, readout)][N]["val"].append(val_acc)
        grouped[(augment_train, readout)][N]["val_aug"].append(val_aug_acc)

    return grouped


def main() -> None:
    args = parse_args()
    project = args.project or os.environ.get("WANDB_PROJECT", "eqnn")

    grouped = _fetch_grouped_results(args.architecture, project, args.entity)
    if not grouped:
        raise SystemExit(
            f"No finished runs with a logged val/accuracy found for "
            f"architecture={args.architecture!r} in project {project!r}."
        )

    readouts = sorted({key[1] for key in grouped}, key=str)
    fig, axes = plt.subplots(
        len(readouts), 2, figsize=(12, 5 * len(readouts)), squeeze=False
    )

    for row, readout in enumerate(readouts):
        ax_val, ax_val_aug = axes[row]
        for augment_train in (True, False):
            key = (augment_train, readout)
            if key not in grouped:
                continue
            points = sorted(grouped[key].items())
            Ns = [p[0] for p in points]
            val_means = [sum(p[1]["val"]) / len(p[1]["val"]) for p in points]
            val_aug_means = [
                sum(p[1]["val_aug"]) / len(p[1]["val_aug"]) for p in points
            ]
            n_seeds = [len(p[1]["val"]) for p in points]
            label = f"augment_train={augment_train} (n_seeds={n_seeds})"
            ax_val.plot(Ns, val_means, marker="o", label=label)
            ax_val_aug.plot(Ns, val_aug_means, marker="o", label=label)

        ax_val.set_title(f"val/accuracy vs N — readout={readout}")
        ax_val.set_xlabel("N")
        ax_val.set_ylabel("val/accuracy")
        ax_val.legend()
        ax_val.grid(True, alpha=0.3)

        ax_val_aug.set_title(f"val_aug/accuracy vs N — readout={readout}")
        ax_val_aug.set_xlabel("N")
        ax_val_aug.set_ylabel("val_aug/accuracy")
        ax_val_aug.legend()
        ax_val_aug.grid(True, alpha=0.3)

    fig.suptitle(f"{args.architecture}: accuracy vs N, averaged over seeds")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
