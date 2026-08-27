"""Plots val/accuracy and val_aug/accuracy vs N from wandb runs, averaged
over seeds with ±SEM (standard error of the mean) error bars, as a grid
of panels: one column per augment_train mode (none / online / once), one
row per (class1, class2) digit pair found in the data (e.g. 3-vs-4 and
4-vs-5 side by side if both were run). Distinguishes: color =
architecture, marker shape = val vs val_aug. Requires a single readout
(pass --readout, or it's auto-picked when only one is present) — mixing
readouts on one axis would conflate two different measured quantities.
Doesn't touch the training pipeline — a standalone analysis utility over
already-logged wandb runs.

For each N, only the seeds present in EVERY (architecture, augment_train)
series with data at that N are averaged, computed separately within each
class pair (a classification task on 3-vs-4 has nothing to do with
seed availability on 4-vs-5) — so every line/panel in the final plot
reflects exactly the same set of trained models, whether comparing
across augment_train modes or across architectures. See
_restrict_to_common_seeds. The diagnostic table printed below still
shows the raw, unrestricted seed counts, so you can see what got dropped.

Older runs logged augment_train as a bool, before "once" existed —
True/"true" is normalized to "online" and False/"false" to "none" (they
were the same behavior, just re-randomized every epoch vs not), so those
runs still land in the right panel alongside newer string-valued ones.

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
import statistics
from collections import defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

METRIC_LINESTYLES = {"val": "-", "val_aug": "--"}
METRIC_MARKERS = {"val": "o", "val_aug": "s"}
METRIC_LABELS = {"val": "val/accuracy", "val_aug": "val_aug/accuracy"}


def _sem(values: list[float]) -> float:
    """Standard error of the mean — 0 for a single seed (no spread to
    estimate), not NaN (which would break the error bars)."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / len(values) ** 0.5


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


def _normalize_augment_train(value: Any) -> Any:
    """Older runs logged augment_train as a bool (True/False) or, via
    wandb's API, sometimes as the strings "true"/"false" — from before the
    "once" mode existed, when it was just "re-randomize every epoch or
    not". Map those onto today's "online"/"none" so old and new runs group
    together; anything else (already "none"/"online"/"once") passes
    through unchanged."""
    if isinstance(value, bool):
        return "online" if value else "none"
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return "online" if value.lower() == "true" else "none"
    return value


Bucket = dict[str, list[Any]]
GroupKey = tuple[
    str, Any, Any, int, int
]  # (architecture, augment_train, readout, class1, class2)
Grouped = dict[GroupKey, dict[int, Bucket]]


def _fetch_grouped_results(
    architectures: list[str], project: str, entity: str | None
) -> Grouped:
    """Returns {(architecture, augment_train, readout, class1, class2):
    {N: {"val": [...], "val_aug": [...], "seed": [...]}}}, one accuracy
    value per finished run with a usable summary — runs that crashed or
    never reached the final validation step (no val/accuracy in their
    summary) are silently skipped. Runs logged before DATA.class1/class2
    existed default to (3, 4), the only pair ever used back then.
    """
    api = wandb.Api()
    resolved_entity = entity or api.default_entity
    path = f"{resolved_entity}/{project}" if resolved_entity else project

    grouped: Grouped = defaultdict(
        lambda: defaultdict(lambda: {"val": [], "val_aug": [], "seed": []})
    )
    for architecture in architectures:
        # per_page defaults to 50 — with hundreds of accumulated runs that
        # means many sequential HTTP round trips; a large single page
        # cuts this down to (usually) one request per architecture.
        runs = api.runs(
            path, filters={"config.architecture": architecture}, per_page=1000
        )
        for run in runs:
            cfg = run.config
            summary = run.summary
            N = cfg.get("N")
            augment_train = _normalize_augment_train(cfg.get("augment_train"))
            readout = cfg.get("readout")
            seed = cfg.get("seed")
            class1 = cfg.get("class1", 3)
            class2 = cfg.get("class2", 4)
            val_acc = summary.get("val/accuracy")
            val_aug_acc = summary.get("val_aug/accuracy")
            if None in (N, augment_train, val_acc, val_aug_acc):
                continue
            bucket = grouped[(architecture, augment_train, readout, class1, class2)][N]
            bucket["val"].append(val_acc)
            bucket["val_aug"].append(val_aug_acc)
            bucket["seed"].append(seed)

    return grouped


def _print_diagnostics(grouped: Grouped) -> None:
    header = (
        f"{'architecture':<14}{'augment_train':<15}{'readout':<12}{'classes':<10}"
        f"{'N':<8}{'n_seeds':<9}{'mean_val':<10}{'mean_val_aug':<14}seeds"
    )
    print(header)
    print("-" * len(header))
    for (architecture, augment_train, readout, class1, class2), by_n in sorted(
        grouped.items(),
        key=lambda kv: (kv[0][0], str(kv[0][1]), str(kv[0][2]), kv[0][3], kv[0][4]),
    ):
        classes_str = f"{class1}v{class2}"
        for N, bucket in sorted(by_n.items()):
            seeds = sorted(bucket["seed"])
            mean_val = sum(bucket["val"]) / len(bucket["val"])
            mean_val_aug = sum(bucket["val_aug"]) / len(bucket["val_aug"])
            print(
                f"{architecture:<14}{str(augment_train):<15}{str(readout):<12}"
                f"{classes_str:<10}{N:<8}{len(seeds):<9}{mean_val:<10.4f}"
                f"{mean_val_aug:<14.4f}{seeds}"
            )
    print("-" * len(header))


def _restrict_to_common_seeds(grouped: Grouped) -> Grouped:
    """Restricts every N to only the seeds present in EVERY
    (architecture, augment_train) series that has data at that N — every
    line/panel visually compared in the final plot then reflects exactly
    the same set of trained models, whether the comparison is across
    augment_train modes (same architecture, same seed = same initial
    parameters) or across architectures (same N, same effort per seed).
    A series with extra seeds no other series reaches (or missing ones
    the rest have) no longer skews its own mean relative to the others.
    Computed separately per (class1, class2) — seed availability on one
    classification task says nothing about another. Assumes readout is
    already fixed to a single value (see main: this runs after the
    --readout filter)."""
    restricted: Grouped = defaultdict(
        lambda: defaultdict(lambda: {"val": [], "val_aug": [], "seed": []})
    )
    class_pairs = {(key[3], key[4]) for key in grouped}
    for class_pair in class_pairs:
        keys_for_pair = [key for key in grouped if (key[3], key[4]) == class_pair]
        n_values = {N for key in keys_for_pair for N in grouped[key]}
        for N in n_values:
            buckets = [grouped[key][N] for key in keys_for_pair if N in grouped[key]]
            common_seeds = set.intersection(*(set(b["seed"]) for b in buckets))
            for key in keys_for_pair:
                by_n = grouped[key]
                if N not in by_n:
                    continue
                bucket = by_n[N]
                keep = [i for i, s in enumerate(bucket["seed"]) if s in common_seeds]
                if not keep:
                    # No seed survives the intersection for this point —
                    # drop it entirely rather than leaving an empty bucket
                    # (which would divide by zero when averaged later).
                    continue
                restricted[key][N] = {
                    "val": [bucket["val"][i] for i in keep],
                    "val_aug": [bucket["val_aug"][i] for i in keep],
                    "seed": [bucket["seed"][i] for i in keep],
                }
    return restricted


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

    grouped = _restrict_to_common_seeds(grouped)

    class_pairs = sorted({(key[3], key[4]) for key in grouped})
    augment_train_modes = ["none", "online", "once"]
    fig, axes = plt.subplots(
        len(class_pairs),
        3,
        figsize=(21, 7 * len(class_pairs)),
        sharey=True,
        squeeze=False,
    )

    for row, (class1, class2) in enumerate(class_pairs):
        panels = dict(zip(augment_train_modes, axes[row], strict=True))
        for augment_train, ax in panels.items():
            for architecture in architectures:
                key = (architecture, augment_train, readout, class1, class2)
                if key not in grouped:
                    continue
                points = sorted(grouped[key].items())
                Ns = [p[0] for p in points]
                means = {
                    metric: [sum(p[1][metric]) / len(p[1][metric]) for p in points]
                    for metric in ("val", "val_aug")
                }
                sems = {
                    metric: [_sem(p[1][metric]) for p in points]
                    for metric in ("val", "val_aug")
                }
                n_seeds = [len(p[1]["val"]) for p in points]
                for metric in ("val", "val_aug"):
                    label = (
                        f"{architecture}, {METRIC_LABELS[metric]} (n_seeds={n_seeds})"
                    )
                    ax.errorbar(
                        Ns,
                        means[metric],
                        yerr=sems[metric],
                        label=label,
                        color=colors[architecture],
                        linestyle=METRIC_LINESTYLES[metric],
                        marker=METRIC_MARKERS[metric],
                        capsize=3,
                        elinewidth=1,
                    )
            ax.set_title(f"class {class1} vs {class2}, augment_train={augment_train}")
            ax.set_xlabel("N")
            ax.legend(fontsize="small", loc="best")
            ax.grid(True, alpha=0.3)
        axes[row][0].set_ylabel("accuracy")

    fig.suptitle(
        f"{', '.join(architectures)}: accuracy vs N, averaged over seeds "
        f"— readout={readout}\n"
        "(colore=architettura, continua ●=val, tratteggiata ■=val_aug, "
        "barre=±SEM sui seed comuni a tutte le serie della stessa coppia "
        "di classi; righe=coppia di classi, colonne=augment_train)"
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
