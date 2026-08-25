"""Plots val/accuracy and val_aug/accuracy vs N from wandb runs, averaged
over seeds. One row of plots per readout value found (or a single row if
--readout restricts to one); one line per (architecture, augment_train)
combination — color = architecture, solid/dashed = augment_train
True/False — so multiple architectures overlay directly comparable on the
same panel. Doesn't touch the training pipeline — a standalone analysis
utility over already-logged wandb runs.

Usage (reuses the eqnn image, which already has wandb + the .env API key
wired via docker-compose.yml's env_file):

    docker compose run --rm --entrypoint python3 eqnn src/plot_wandb_results.py --architecture config6
    docker compose run --rm --entrypoint python3 eqnn src/plot_wandb_results.py --architecture config6 config7 --readout x0_xhalf

Always prints a diagnostic table first (architecture, augment_train,
readout, N, seed count, mean val/val_aug accuracy, the exact seeds found)
— check it before trusting the plot: a combination with fewer seeds than
expected, or a missing row entirely, means those runs either weren't
launched or aren't finished/logged in wandb yet.

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

AUGMENT_TRAIN_LINESTYLES = {True: "-", False: "--"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot val/accuracy and val_aug/accuracy vs N from wandb, "
            "averaged over seeds, split by architecture, augment_train, "
            "and readout."
        )
    )
    parser.add_argument(
        "--architecture",
        nargs="+",
        default=["config6"],
        help="One or more architectures to overlay on the same plots "
        "(e.g. --architecture config6 config7).",
    )
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


Bucket = dict[str, list[Any]]
GroupKey = tuple[str, Any, Any]  # (architecture, augment_train, readout)
Grouped = dict[GroupKey, dict[int, Bucket]]


def _fetch_grouped_results(
    architectures: list[str], project: str, entity: str | None
) -> Grouped:
    """Returns {(architecture, augment_train, readout): {N: {"val": [...],
    "val_aug": [...], "seed": [...]}}}, one accuracy value per finished run
    with a usable summary — runs that crashed or never reached the final
    validation step (no val/accuracy in their summary) are silently
    skipped.
    """
    api = wandb.Api()
    resolved_entity = entity or api.default_entity
    path = f"{resolved_entity}/{project}" if resolved_entity else project

    grouped: Grouped = defaultdict(
        lambda: defaultdict(lambda: {"val": [], "val_aug": [], "seed": []})
    )
    for architecture in architectures:
        runs = api.runs(path, filters={"config.architecture": architecture})
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
            bucket = grouped[(architecture, augment_train, readout)][N]
            bucket["val"].append(val_acc)
            bucket["val_aug"].append(val_aug_acc)
            bucket["seed"].append(seed)

    return grouped


def _print_diagnostics(grouped: Grouped) -> None:
    header = (
        f"{'architecture':<14}{'augment_train':<15}{'readout':<12}{'N':<8}"
        f"{'n_seeds':<9}{'mean_val':<10}{'mean_val_aug':<14}seeds"
    )
    print(header)
    print("-" * len(header))
    for (architecture, augment_train, readout), by_n in sorted(
        grouped.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), str(kv[0][2]))
    ):
        for N, bucket in sorted(by_n.items()):
            seeds = sorted(bucket["seed"])
            mean_val = sum(bucket["val"]) / len(bucket["val"])
            mean_val_aug = sum(bucket["val_aug"]) / len(bucket["val_aug"])
            print(
                f"{architecture:<14}{str(augment_train):<15}{str(readout):<12}"
                f"{N:<8}{len(seeds):<9}{mean_val:<10.4f}{mean_val_aug:<14.4f}"
                f"{seeds}"
            )
    print("-" * len(header))


def main() -> None:
    args = parse_args()
    project = args.project or os.environ.get("WANDB_PROJECT", "eqnn")

    grouped = _fetch_grouped_results(args.architecture, project, args.entity)
    if not grouped:
        raise SystemExit(
            f"No finished runs with a logged val/accuracy found for "
            f"architecture(s)={args.architecture} in project {project!r}."
        )

    _print_diagnostics(grouped)

    if args.readout is not None:
        grouped = {k: v for k, v in grouped.items() if k[2] == args.readout}
        if not grouped:
            raise SystemExit(f"No runs found with readout={args.readout!r}.")

    architectures = sorted({key[0] for key in grouped})
    colors = {arch: f"C{i}" for i, arch in enumerate(architectures)}
    readouts = sorted({key[2] for key in grouped}, key=str)

    fig, axes = plt.subplots(
        len(readouts), 2, figsize=(12, 5 * len(readouts)), squeeze=False
    )

    for row, readout in enumerate(readouts):
        ax_val, ax_val_aug = axes[row]
        for architecture in architectures:
            for augment_train in (True, False):
                key = (architecture, augment_train, readout)
                if key not in grouped:
                    continue
                points = sorted(grouped[key].items())
                Ns = [p[0] for p in points]
                val_means = [sum(p[1]["val"]) / len(p[1]["val"]) for p in points]
                val_aug_means = [
                    sum(p[1]["val_aug"]) / len(p[1]["val_aug"]) for p in points
                ]
                n_seeds = [len(p[1]["val"]) for p in points]
                style = {
                    "color": colors[architecture],
                    "linestyle": AUGMENT_TRAIN_LINESTYLES[augment_train],
                    "marker": "o",
                }
                label = (
                    f"{architecture}, augment_train={augment_train} (n_seeds={n_seeds})"
                )
                ax_val.plot(Ns, val_means, label=label, **style)
                ax_val_aug.plot(Ns, val_aug_means, label=label, **style)

        ax_val.set_title(f"val/accuracy vs N — readout={readout}")
        ax_val.set_xlabel("N")
        ax_val.set_ylabel("val/accuracy")
        ax_val.legend(fontsize="small")
        ax_val.grid(True, alpha=0.3)

        ax_val_aug.set_title(f"val_aug/accuracy vs N — readout={readout}")
        ax_val_aug.set_xlabel("N")
        ax_val_aug.set_ylabel("val_aug/accuracy")
        ax_val_aug.legend(fontsize="small")
        ax_val_aug.grid(True, alpha=0.3)

    fig.suptitle(
        f"{', '.join(architectures)}: accuracy vs N, averaged over seeds "
        "(solid=augment_train, dashed=no augment_train)"
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
