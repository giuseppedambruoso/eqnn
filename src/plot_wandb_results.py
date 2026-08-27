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

Only runs with seed in [--min-seed, --max-seed] (default 1-10) and N in
--n-values (default 40/80/160/320/640/1280/2560/5120) are ever
considered — this excludes older, unrelated experiment-phase runs (e.g.
seed=1234, N=30) that would otherwise silently pollute the mean. This
filter is applied at fetch time, before the diagnostic table, on both
the wandb and --local paths.

For each N, only the seeds present in EVERY (architecture, augment_train)
series with data at that N are averaged, computed separately within each
class pair (a classification task on 3-vs-4 has nothing to do with
seed availability on 4-vs-5) — so every line/panel in the final plot
reflects exactly the same set of trained models, whether comparing
across augment_train modes or across architectures. See
_restrict_to_common_seeds. The diagnostic table printed below still
shows the raw, unrestricted seed counts, so you can see what got dropped.

An exact duplicate run (same seed re-launched by mistake) is
automatically deduped — only the first occurrence of each seed counts
towards the mean — see _dedupe_by_seed. The diagnostic table still shows
the raw, undeduped seed list so duplicates remain visible even though
they no longer skew the plot.

Older runs logged augment_train as a bool, before "once" existed —
True/"true" is normalized to "online" and False/"false" to "none" (they
were the same behavior, just re-randomized every epoch vs not), so those
runs still land in the right panel alongside newer string-valued ones.

Usage (reuses the eqnn image, which already has wandb + the .env API key
wired via docker-compose.yml's env_file):

    docker compose run --rm --entrypoint python3 eqnn src/plot_wandb_results.py --architecture config6
    docker compose run --rm --entrypoint python3 eqnn src/plot_wandb_results.py --architecture config6 config7 --readout x0_xhalf

Pass --local to read summary.json files directly from outputs/ and
multirun/ instead of querying wandb — no network calls, so it's much
faster once there are hundreds of accumulated runs, and it also picks up
runs from a sweep that's still in progress and hasn't necessarily
finished uploading everything to wandb yet:

    docker compose run --rm --entrypoint python3 eqnn src/plot_wandb_results.py --architecture config6 config7 --readout x0_xhalf --local

Pass --save-local (without --local — it fetches from wandb as usual)
to also download every fetched run's model artifact into
<outputs-dir>/wandb_backfill/<run_id>/, so runs that only ever existed
on wandb (e.g. from an earlier machine/session, before local
persistence to results_def/ existed) become visible to a future --local
plot and to collect_results.py too:

    docker compose run --rm --entrypoint python3 eqnn src/plot_wandb_results.py --architecture config6 config7 --readout x0_xhalf --save-local

With --local, every run comes from re-scanning outputs/ and multirun/
from scratch each time, and re-running the same sweep after fixing a bug
leaves the old, stale directory sitting right there forever. Before
building the diagnostic table, --local PERMANENTLY DELETES (via
shutil.rmtree) every local run directory that duplicates a more recently
modified one's identity (architecture, N, seed, augment_train, readout,
class1, class2), printing exactly what got removed — see
_prune_stale_local_runs. A run still in progress is never touched (it
has no summary.json yet, so it's never a candidate).

Always prints a diagnostic table first (architecture, augment_train,
readout, N, seed count, mean val/val_aug accuracy, the exact seeds found)
— check it before trusting the plot: a combination with fewer seeds than
expected, or a missing row entirely, means those runs either weren't
launched or aren't finished/logged yet (in wandb, or on disk if --local).

Output: a PNG (default val_accuracy_vs_N.png) saved in the working
directory (mount ./outputs or similar if you want it to land on the host).
"""

import argparse
import json
import os
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
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


DEFAULT_N_VALUES = [40, 80, 160, 320, 640, 1280, 2560, 5120]


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
        "--min-seed",
        type=int,
        default=1,
        help="Only include runs with seed >= this (default: 1) — excludes "
        "old/unrelated runs (e.g. seed=1234 from an earlier experiment "
        "phase) that would otherwise pollute the mean.",
    )
    parser.add_argument(
        "--max-seed",
        type=int,
        default=10,
        help="Only include runs with seed <= this (default: 10).",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=DEFAULT_N_VALUES,
        help=f"Only include runs whose N is one of these (default: "
        f"{DEFAULT_N_VALUES}).",
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
    parser.add_argument(
        "--local",
        action="store_true",
        help="Read summary.json files from outputs/ and multirun/ instead "
        "of querying wandb — no network calls, much faster with many "
        "accumulated runs, and picks up an in-progress sweep too.",
    )
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--multirun-dir", default="multirun")
    parser.add_argument(
        "--save-local",
        action="store_true",
        help="Only meaningful without --local (i.e. fetching from wandb): "
        "downloads each fetched run's model artifact (final_model.pt, "
        "summary.json, ...) into <outputs-dir>/wandb_backfill/<run_id>/, "
        "so it becomes visible to a future --local plot and to "
        "collect_results.py. Skips runs already downloaded there.",
    )
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


def _seed_and_n_allowed(
    seed: Any, N: Any, min_seed: int, max_seed: int, n_values: list[int]
) -> bool:
    """Filters out runs outside the requested seed range / N whitelist —
    e.g. seed=1234 or N=30 from an earlier experiment phase, which would
    otherwise silently pollute the mean alongside the current sweep's
    seeds 1-10 and N in {40, 80, ..., 5120}."""
    if not isinstance(seed, int) or not (min_seed <= seed <= max_seed):
        return False
    return N in n_values


def _download_run_locally(run: Any, dest_root: Path) -> None:
    """Downloads this wandb run's logged model Artifact (final_model.pt,
    summary.json, and whatever else train.py attached — see train.py's
    wandb.Artifact block) into dest_root/<run.id>/, so a run that only
    exists on wandb also becomes visible to a future --local plot (see
    _fetch_grouped_results_local) and to collect_results.py. Skips runs
    already downloaded there (checked via summary.json's presence) —
    idempotent, safe to call on every fetched run every time. Best-effort:
    a download failure is reported but never aborts the whole fetch."""
    dest_dir = dest_root / run.id
    if (dest_dir / "summary.json").exists():
        return
    try:
        for artifact in run.logged_artifacts():
            if artifact.type == "model":
                artifact.download(root=str(dest_dir))
                return
    except Exception as exc:
        print(f"Warning: failed to download artifact for run {run.id}: {exc}")


def _fetch_grouped_results(
    architectures: list[str],
    project: str,
    entity: str | None,
    min_seed: int,
    max_seed: int,
    n_values: list[int],
    save_local_dir: Path | None = None,
    known_local_identities: set[tuple[Any, ...]] | None = None,
) -> Grouped:
    """Returns {(architecture, augment_train, readout, class1, class2):
    {N: {"val": [...], "val_aug": [...], "seed": [...]}}}, one accuracy
    value per finished run with a usable summary — runs that crashed or
    never reached the final validation step (no val/accuracy in their
    summary) are silently skipped. Runs logged before DATA.class1/class2
    existed default to (3, 4), the only pair ever used back then.

    known_local_identities (identity = (architecture, N, seed,
    augment_train, readout, class1, class2), matching _find_local_candidates)
    skips downloading (with save_local_dir) any run whose identity already
    exists locally — e.g. from actual local training, not just a previous
    backfill. Without this, --save-local would re-download a run into
    wandb_backfill/<run_id>/ even though the identical config+seed already
    exists locally under a different path, creating a duplicate that a
    later --local prune would delete — and the next --save-local would
    then re-download it again, forever."""
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
            if not _seed_and_n_allowed(seed, N, min_seed, max_seed, n_values):
                continue
            if save_local_dir is not None:
                identity = (
                    architecture,
                    N,
                    seed,
                    augment_train,
                    readout,
                    class1,
                    class2,
                )
                already_local = (
                    known_local_identities is not None
                    and identity in known_local_identities
                )
                if not already_local:
                    _download_run_locally(run, save_local_dir)
            bucket = grouped[(architecture, augment_train, readout, class1, class2)][N]
            bucket["val"].append(val_acc)
            bucket["val_aug"].append(val_aug_acc)
            bucket["seed"].append(seed)

    return grouped


LocalCandidate = tuple[tuple[Any, ...], Path, float, float, float]


def _find_local_candidates(
    architectures: list[str],
    outputs_dir: str,
    multirun_dir: str,
    min_seed: int,
    max_seed: int,
    n_values: list[int],
) -> list[LocalCandidate]:
    """Every local run with a usable summary.json, as (identity, run_dir,
    mtime, val_acc, val_aug_acc). identity = (architecture, N, seed,
    augment_train, readout, class1, class2) — a run missing from wandb or
    still uploading there is still included, since this only touches the
    local filesystem. Runs outside [min_seed, max_seed] or with an N not
    in n_values are excluded entirely (not pruned as duplicates — they're
    just out of scope, e.g. an old experiment phase's seed=1234)."""
    candidates: list[LocalCandidate] = []
    for root in (outputs_dir, multirun_dir):
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for summary_path in root_path.rglob("summary.json"):
            with open(summary_path) as f:
                summary = json.load(f)
            config = summary.get("config", {})
            architecture = config.get("architecture")
            if architecture not in architectures:
                continue
            N = summary.get("N")
            augment_train = _normalize_augment_train(
                config.get("augment_train", "none")
            )
            readout = config.get("readout")
            seed = summary.get("seed")
            class1 = config.get("class1", 3)
            class2 = config.get("class2", 4)
            val_acc = summary.get("val_accuracy")
            val_aug_acc = summary.get("val_aug_accuracy")
            if None in (N, augment_train, val_acc, val_aug_acc, seed):
                continue
            if not _seed_and_n_allowed(seed, N, min_seed, max_seed, n_values):
                continue
            identity = (architecture, N, seed, augment_train, readout, class1, class2)
            candidates.append(
                (
                    identity,
                    summary_path.parent,
                    summary_path.stat().st_mtime,
                    val_acc,
                    val_aug_acc,
                )
            )
    return candidates


def _prune_stale_local_runs(candidates: list[LocalCandidate]) -> list[LocalCandidate]:
    """outputs/ and multirun/ accumulate one directory per sweep launch
    forever — re-running the same sweep twice (e.g. after fixing a bug)
    leaves the old, stale local run sitting right next to the new one.
    Deletes every run_dir that isn't the most recently modified one for
    its identity (architecture, N, seed, augment_train, readout, class1,
    class2), and returns only the survivors. Never touches a run still in
    progress — those have no summary.json yet, so they're never in
    `candidates` to begin with (see _find_local_candidates)."""
    by_identity: dict[tuple[Any, ...], list[LocalCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_identity[candidate[0]].append(candidate)

    survivors: list[LocalCandidate] = []
    removed: list[Path] = []
    for group in by_identity.values():
        if len(group) == 1:
            survivors.append(group[0])
            continue
        group_sorted = sorted(group, key=lambda c: c[2], reverse=True)
        survivors.append(group_sorted[0])
        for stale in group_sorted[1:]:
            shutil.rmtree(stale[1], ignore_errors=True)
            removed.append(stale[1])

    if removed:
        print(f"Pruned {len(removed)} stale duplicate local run(s):")
        for path in removed:
            print(f"  removed {path}")

    return survivors


def _fetch_grouped_results_local(
    architectures: list[str],
    outputs_dir: str,
    multirun_dir: str,
    min_seed: int,
    max_seed: int,
    n_values: list[int],
) -> Grouped:
    """Same shape as _fetch_grouped_results, but scans local summary.json
    files under outputs_dir/multirun_dir instead of querying wandb — no
    network calls, and picks up runs from a sweep still in progress.
    train.py writes summary.json with underscore keys (val_accuracy,
    val_aug_accuracy), unlike wandb's slash-separated summary keys.
    Prunes stale duplicate local runs first — see _prune_stale_local_runs."""
    candidates = _find_local_candidates(
        architectures, outputs_dir, multirun_dir, min_seed, max_seed, n_values
    )
    survivors = _prune_stale_local_runs(candidates)

    grouped: Grouped = defaultdict(
        lambda: defaultdict(lambda: {"val": [], "val_aug": [], "seed": []})
    )
    for identity, _run_dir, _mtime, val_acc, val_aug_acc in survivors:
        architecture, N, seed, augment_train, readout, class1, class2 = identity
        bucket = grouped[(architecture, augment_train, readout, class1, class2)][N]
        bucket["val"].append(val_acc)
        bucket["val_aug"].append(val_aug_acc)
        bucket["seed"].append(seed)

    return grouped


def _dedupe_by_seed(grouped: Grouped) -> Grouped:
    """Keeps only the first occurrence of each seed within every (key, N)
    bucket — an exact duplicate run (e.g. an accidental sweep re-launch)
    would otherwise silently double-count that seed in the mean instead
    of contributing once, like every other seed. Applied unconditionally
    before plotting, on both the wandb and --local data paths, so the
    plot is correct even if the underlying duplicate run still exists."""
    deduped: Grouped = {}
    for key, by_n in grouped.items():
        deduped[key] = {}
        for N, bucket in by_n.items():
            seen: set[Any] = set()
            val: list[Any] = []
            val_aug: list[Any] = []
            seed: list[Any] = []
            for v, va, s in zip(
                bucket["val"], bucket["val_aug"], bucket["seed"], strict=True
            ):
                if s in seen:
                    continue
                seen.add(s)
                val.append(v)
                val_aug.append(va)
                seed.append(s)
            deduped[key][N] = {"val": val, "val_aug": val_aug, "seed": seed}
    return deduped


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

    if args.local:
        grouped = _fetch_grouped_results_local(
            args.architecture,
            args.outputs_dir,
            args.multirun_dir,
            args.min_seed,
            args.max_seed,
            args.n_values,
        )
        source_desc = f"{args.outputs_dir!r}/{args.multirun_dir!r} (local)"
    else:
        project = args.project or os.environ.get("WANDB_PROJECT", "eqnn")
        save_local_dir = None
        known_local_identities = None
        if args.save_local:
            save_local_dir = Path(args.outputs_dir) / "wandb_backfill"
            # Skip downloading a run whose identity already exists locally
            # (from actual training, or a previous backfill) — otherwise a
            # later --local prune deletes the redundant copy and the next
            # --save-local just re-downloads it, forever.
            existing_candidates = _find_local_candidates(
                args.architecture,
                args.outputs_dir,
                args.multirun_dir,
                args.min_seed,
                args.max_seed,
                args.n_values,
            )
            known_local_identities = {c[0] for c in existing_candidates}
        grouped = _fetch_grouped_results(
            args.architecture,
            project,
            args.entity,
            args.min_seed,
            args.max_seed,
            args.n_values,
            save_local_dir,
            known_local_identities,
        )
        source_desc = f"project {project!r}"
    if not grouped:
        raise SystemExit(
            f"No finished runs with a logged val/accuracy found for "
            f"architecture(s)={args.architecture} in {source_desc}."
        )

    _print_diagnostics(grouped)

    grouped = _dedupe_by_seed(grouped)

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
