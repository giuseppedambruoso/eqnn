"""Plots val/accuracy and val_aug/accuracy vs N from wandb runs, averaged
over seeds, as three side-by-side panels (one per augment_train mode:
none / online / once). Distinguishes: color = architecture, marker shape
= val vs val_aug. Requires a single readout (pass --readout, or it's
auto-picked when only one is present) — mixing readouts on one axis would
conflate two different measured quantities. Doesn't touch the training
pipeline — a standalone analysis utility over already-logged wandb runs.

Only picks up runs logged with the current string-valued augment_train
("none"/"online"/"once" — see src/data_loading.py); older runs logged
with a boolean augment_train won't match any panel and are silently
skipped (they still show up in the diagnostic table).

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

METRIC_LINESTYLES = {"val": "-", "val_aug": "--"}
METRIC_MARKERS = {"val": "o", "val_aug": "s"}
METRIC_LABELS = {"val": "val/accuracy", "val_aug": "val_aug/accuracy"}


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
            augment_train = cfg.get("augment_train")
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
    if len(readouts) > 1:
        raise SystemExit(
            f"Found multiple readouts {readouts} — a single combined plot "
            "would conflate two different measured quantities. Pass "
            "--readout to pick one."
        )
    readout = readouts[0]

    augment_train_modes = ["none", "online", "once"]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)
    panels = dict(zip(augment_train_modes, axes, strict=True))

    for augment_train, ax in panels.items():
        for architecture in architectures:
            key = (architecture, augment_train, readout)
            if key not in grouped:
                continue
            points = sorted(grouped[key].items())
            Ns = [p[0] for p in points]
            means = {
                metric: [sum(p[1][metric]) / len(p[1][metric]) for p in points]
                for metric in ("val", "val_aug")
            }
            n_seeds = [len(p[1]["val"]) for p in points]
            for metric in ("val", "val_aug"):
                label = f"{architecture}, {METRIC_LABELS[metric]} (n_seeds={n_seeds})"
                ax.plot(
                    Ns,
                    means[metric],
                    label=label,
                    color=colors[architecture],
                    linestyle=METRIC_LINESTYLES[metric],
                    marker=METRIC_MARKERS[metric],
                )
        ax.set_title(f"augment_train={augment_train}")
        ax.set_xlabel("N")
        ax.legend(fontsize="small", loc="best")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("accuracy")

    fig.suptitle(
        f"{', '.join(architectures)}: accuracy vs N, averaged over seeds "
        f"— readout={readout}\n"
        "(colore=architettura, continua ●=val, tratteggiata ■=val_aug)"
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
