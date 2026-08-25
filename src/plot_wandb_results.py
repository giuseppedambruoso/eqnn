"""Plots val/accuracy and val_aug/accuracy vs N from wandb runs, averaged
over seeds, one line per augment_train value and one row of plots per
readout value found. Doesn't touch the training pipeline — a standalone
analysis utility over already-logged wandb runs.

Usage (reuses the eqnn image, which already has wandb + the .env API key
wired via docker-compose.yml's env_file):

    docker compose run --rm --entrypoint python3 eqnn src/plot_wandb_results.py --architecture config6
    docker compose run --rm --entrypoint python3 eqnn src/plot_wandb_results.py --architecture config6 --readout x0_xhalf

Always prints a diagnostic table first (N, augment_train, readout, the
exact seeds found) — check it before trusting the plot: a combination
with fewer seeds than expected, or a missing row entirely, means those
runs either weren't launched or aren't finished/logged in wandb yet.

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
        "--readout",
        default=None,
        help="Only plot this readout (e.g. x0_xhalf or avg_x). "
        "Default: plot every readout found, one row per value.",
    )
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


def _normalize_bool(value: Any) -> Any:
    """wandb's API sometimes hands back config booleans as the strings
    "true"/"false" instead of Python bool, depending on how they were
    logged — normalize so (True, False) grouping keys actually match."""
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return value


def _fetch_grouped_results(
    architecture: str, project: str, entity: str | None
) -> dict[tuple[Any, Any], dict[int, dict[str, list[float]]]]:
    """Returns {(augment_train, readout): {N: {"val": [...], "val_aug": [...],
    "seed": [...]}}}, one accuracy value per finished run with a usable
    summary — runs that crashed or never reached the final validation step
    (no val/accuracy in their summary) are silently skipped.
    """
    api = wandb.Api()
    resolved_entity = entity or api.default_entity
    path = f"{resolved_entity}/{project}" if resolved_entity else project

    runs = api.runs(path, filters={"config.architecture": architecture})

    grouped: dict[tuple[Any, Any], dict[int, dict[str, list[Any]]]] = defaultdict(
        lambda: defaultdict(lambda: {"val": [], "val_aug": [], "seed": []})
    )
    for run in runs:
        cfg = run.config
        summary = run.summary
        N = cfg.get("N")
        augment_train = _normalize_bool(cfg.get("augment_train"))
        readout = cfg.get("readout")
        seed = cfg.get("seed")
        val_acc = summary.get("val/accuracy")
        val_aug_acc = summary.get("val_aug/accuracy")
        if None in (N, augment_train, val_acc, val_aug_acc):
            continue
        bucket = grouped[(augment_train, readout)][N]
        bucket["val"].append(val_acc)
        bucket["val_aug"].append(val_aug_acc)
        bucket["seed"].append(seed)

    return grouped


def _print_diagnostics(
    grouped: dict[tuple[Any, Any], dict[int, dict[str, list[float]]]],
) -> None:
    header = (
        f"{'augment_train':<15}{'readout':<12}{'N':<8}{'n_seeds':<9}"
        f"{'mean_val':<10}{'mean_val_aug':<14}seeds"
    )
    print(header)
    print("-" * len(header))
    for (augment_train, readout), by_n in sorted(
        grouped.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))
    ):
        for N, bucket in sorted(by_n.items()):
            seeds = sorted(bucket["seed"])
            mean_val = sum(bucket["val"]) / len(bucket["val"])
            mean_val_aug = sum(bucket["val_aug"]) / len(bucket["val_aug"])
            print(
                f"{str(augment_train):<15}{str(readout):<12}{N:<8}"
                f"{len(seeds):<9}{mean_val:<10.4f}{mean_val_aug:<14.4f}{seeds}"
            )
    print("-" * len(header))


def main() -> None:
    args = parse_args()
    project = args.project or os.environ.get("WANDB_PROJECT", "eqnn")

    grouped = _fetch_grouped_results(args.architecture, project, args.entity)
    if not grouped:
        raise SystemExit(
            f"No finished runs with a logged val/accuracy found for "
            f"architecture={args.architecture!r} in project {project!r}."
        )

    _print_diagnostics(grouped)

    if args.readout is not None:
        grouped = {k: v for k, v in grouped.items() if k[1] == args.readout}
        if not grouped:
            raise SystemExit(f"No runs found with readout={args.readout!r}.")

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
