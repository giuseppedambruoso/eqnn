import json
import os
import time
from pathlib import Path
from typing import cast

import src.plot_wandb_results as pwr
from src.plot_wandb_results import (
    _dedupe_by_seed,
    _download_run_locally,
    _fetch_grouped_results,
    _fetch_grouped_results_local,
    _find_local_candidates,
    _prune_stale_local_runs,
    _restrict_to_common_seeds,
    _seed_and_n_allowed,
)


def _bucket(seeds: list[int]) -> dict[str, list[object]]:
    return {
        "val": [0.1 * s for s in seeds],
        "val_aug": [0.2 * s for s in seeds],
        "seed": cast(list[object], list(seeds)),
        "dataset": cast(list[object], ["mnist"] * len(seeds)),
    }


def test_restricts_to_seeds_common_across_augment_train_modes():
    grouped = {
        ("config6", "none", "x0_xhalf", 3, 4): {40: _bucket([1, 2, 3])},
        ("config6", "online", "x0_xhalf", 3, 4): {40: _bucket([1, 2])},
        ("config6", "once", "x0_xhalf", 3, 4): {40: _bucket([1, 2, 3, 4])},
    }

    restricted = _restrict_to_common_seeds(grouped)

    for key in grouped:
        assert restricted[key][40]["seed"] == [1, 2]


def test_restricts_across_architectures_too():
    """A series with extra seeds no other architecture reaches yet must
    not keep them — every line in the final plot, including different
    colors (architectures), must reflect the same set of trained models."""
    grouped = {
        ("config6", "none", "x0_xhalf", 3, 4): {40: _bucket([1, 2, 3, 4, 5, 6, 7])},
        ("config6", "online", "x0_xhalf", 3, 4): {40: _bucket([1, 2, 3, 4, 5, 6, 7])},
        ("config6", "once", "x0_xhalf", 3, 4): {40: _bucket([1, 2, 3, 4, 5, 6, 7])},
        ("config7", "none", "x0_xhalf", 3, 4): {40: _bucket([1, 2, 3, 4, 5, 6])},
        ("config7", "online", "x0_xhalf", 3, 4): {40: _bucket([1, 2, 3, 4, 5, 6])},
        ("config7", "once", "x0_xhalf", 3, 4): {40: _bucket([1, 2, 3, 4, 5, 6])},
    }

    restricted = _restrict_to_common_seeds(grouped)

    for key in grouped:
        assert restricted[key][40]["seed"] == [1, 2, 3, 4, 5, 6]


def test_does_not_restrict_across_different_class_pairs():
    """Seed availability on one classification task (3-vs-4) must not
    affect another (4-vs-5) — they're unrelated training runs, computed
    as separate rows in the final plot."""
    grouped = {
        ("config6", "none", "x0_xhalf", 3, 4): {40: _bucket([1, 2, 3])},
        ("config6", "none", "x0_xhalf", 4, 5): {40: _bucket([1, 2])},
    }

    restricted = _restrict_to_common_seeds(grouped)

    assert restricted[("config6", "none", "x0_xhalf", 3, 4)][40]["seed"] == [1, 2, 3]
    assert restricted[("config6", "none", "x0_xhalf", 4, 5)][40]["seed"] == [1, 2]


def test_keeps_full_seed_set_when_only_one_mode_present():
    grouped = {("config6", "none", "x0_xhalf", 3, 4): {40: _bucket([1, 2, 3])}}

    restricted = _restrict_to_common_seeds(grouped)

    assert restricted[("config6", "none", "x0_xhalf", 3, 4)][40]["seed"] == [1, 2, 3]


def test_drops_point_entirely_when_no_seed_is_common():
    grouped = {
        ("config6", "none", "x0_xhalf", 3, 4): {40: _bucket([1, 2])},
        ("config6", "online", "x0_xhalf", 3, 4): {40: _bucket([3, 4])},
    }

    restricted = _restrict_to_common_seeds(grouped)

    assert 40 not in restricted[("config6", "none", "x0_xhalf", 3, 4)]
    assert 40 not in restricted[("config6", "online", "x0_xhalf", 3, 4)]


def test_restriction_is_per_n_not_global():
    grouped = {
        ("config6", "none", "x0_xhalf", 3, 4): {
            40: _bucket([1, 2]),
            80: _bucket([1, 2, 3]),
        },
        ("config6", "online", "x0_xhalf", 3, 4): {
            40: _bucket([1, 2, 3]),
            80: _bucket([1, 2]),
        },
    }

    restricted = _restrict_to_common_seeds(grouped)

    assert restricted[("config6", "none", "x0_xhalf", 3, 4)][40]["seed"] == [1, 2]
    assert restricted[("config6", "none", "x0_xhalf", 3, 4)][80]["seed"] == [1, 2]


def test_values_and_seeds_stay_aligned_after_filtering():
    grouped = {
        ("config6", "none", "x0_xhalf", 3, 4): {40: _bucket([1, 2, 3])},
        ("config6", "online", "x0_xhalf", 3, 4): {40: _bucket([2, 3])},
    }

    restricted = _restrict_to_common_seeds(grouped)

    bucket = restricted[("config6", "none", "x0_xhalf", 3, 4)][40]
    for seed, val in zip(bucket["seed"], bucket["val"], strict=True):
        assert val == 0.1 * seed


def test_dedupe_by_seed_drops_repeated_seed():
    grouped = {("config6", "none", "x0_xhalf", 3, 4): {40: _bucket([1, 1, 2])}}

    deduped = _dedupe_by_seed(grouped)

    bucket = deduped[("config6", "none", "x0_xhalf", 3, 4)][40]
    assert bucket["seed"] == [1, 2]
    assert bucket["val"] == [0.1 * 1, 0.1 * 2]


def test_dedupe_by_seed_keeps_unique_seeds_unchanged():
    grouped = {("config6", "none", "x0_xhalf", 3, 4): {40: _bucket([1, 2, 3])}}

    deduped = _dedupe_by_seed(grouped)

    assert deduped[("config6", "none", "x0_xhalf", 3, 4)][40]["seed"] == [1, 2, 3]


def _write_summary(run_dir: Path, **overrides: object) -> None:
    run_dir.mkdir(parents=True)
    summary = {
        "seed": 1,
        "N": 40,
        "val_accuracy": 0.9,
        "val_aug_accuracy": 0.85,
        "config": {
            "architecture": "config6",
            "readout": "x0_xhalf",
            "augment_train": "none",
            "class1": 3,
            "class2": 4,
        },
        **overrides,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))


def test_fetch_grouped_results_local_reads_underscore_keys(tmp_path: Path):
    """train.py writes summary.json with val_accuracy/val_aug_accuracy
    (underscores), unlike wandb's val/accuracy (slash) — the local fetch
    must read the local key names, not the wandb ones."""
    _write_summary(tmp_path / "multirun" / "job0")

    grouped = _fetch_grouped_results_local(
        ["config6"], str(tmp_path / "outputs"), str(tmp_path / "multirun"), 1, 10, [40]
    )

    key = ("config6", "none", "x0_xhalf", 3, 4)
    assert grouped[key][40]["val"] == [0.9]
    assert grouped[key][40]["val_aug"] == [0.85]
    assert grouped[key][40]["seed"] == [1]


def test_fetch_grouped_results_local_filters_by_architecture(tmp_path: Path):
    _write_summary(tmp_path / "multirun" / "job0", config={"architecture": "config7"})

    grouped = _fetch_grouped_results_local(
        ["config6"], str(tmp_path / "outputs"), str(tmp_path / "multirun"), 1, 10, [40]
    )

    assert grouped == {}


def test_fetch_grouped_results_local_skips_missing_dirs(tmp_path: Path):
    grouped = _fetch_grouped_results_local(
        ["config6"],
        str(tmp_path / "nonexistent_outputs"),
        str(tmp_path / "nonexistent_multirun"),
        1,
        10,
        [40],
    )
    assert grouped == {}


def test_prune_stale_local_runs_deletes_older_duplicate_directory(tmp_path: Path):
    """Two local runs sharing the same identity (architecture, N, seed,
    augment_train, readout, class1, class2) must be pruned to the most
    recently modified one — the older duplicate's ENTIRE directory is
    deleted from disk, not just skipped in memory."""
    old_dir = tmp_path / "multirun" / "old_run"
    new_dir = tmp_path / "multirun" / "new_run"
    _write_summary(old_dir, val_accuracy=0.5, val_aug_accuracy=0.5)
    _write_summary(new_dir, val_accuracy=0.9, val_aug_accuracy=0.85)
    old_time = time.time() - 1000
    new_time = time.time()
    os.utime(old_dir / "summary.json", (old_time, old_time))
    os.utime(new_dir / "summary.json", (new_time, new_time))

    candidates = _find_local_candidates(
        ["config6"], str(tmp_path / "outputs"), str(tmp_path / "multirun"), 1, 10, [40]
    )
    survivors = _prune_stale_local_runs(candidates)

    assert not old_dir.exists()
    assert new_dir.exists()
    assert len(survivors) == 1
    assert survivors[0][3] == 0.9


def test_prune_stale_local_runs_keeps_distinct_identities(tmp_path: Path):
    """Two runs with different seeds are NOT duplicates of each other —
    both must survive untouched."""
    run1 = tmp_path / "multirun" / "run1"
    run2 = tmp_path / "multirun" / "run2"
    _write_summary(run1, seed=1)
    _write_summary(run2, seed=2)

    candidates = _find_local_candidates(
        ["config6"], str(tmp_path / "outputs"), str(tmp_path / "multirun"), 1, 10, [40]
    )
    survivors = _prune_stale_local_runs(candidates)

    assert run1.exists()
    assert run2.exists()
    assert len(survivors) == 2


def test_fetch_grouped_results_local_prunes_before_aggregating(tmp_path: Path):
    old_dir = tmp_path / "multirun" / "old_run"
    new_dir = tmp_path / "multirun" / "new_run"
    _write_summary(old_dir, val_accuracy=0.5, val_aug_accuracy=0.5)
    _write_summary(new_dir, val_accuracy=0.9, val_aug_accuracy=0.85)
    old_time = time.time() - 1000
    new_time = time.time()
    os.utime(old_dir / "summary.json", (old_time, old_time))
    os.utime(new_dir / "summary.json", (new_time, new_time))

    grouped = _fetch_grouped_results_local(
        ["config6"], str(tmp_path / "outputs"), str(tmp_path / "multirun"), 1, 10, [40]
    )

    assert not old_dir.exists()
    key = ("config6", "none", "x0_xhalf", 3, 4)
    assert grouped[key][40]["val"] == [0.9]


def test_seed_and_n_allowed_filters_out_of_range_seed():
    assert _seed_and_n_allowed(5, 40, min_seed=1, max_seed=10, n_values=[40]) is True
    assert (
        _seed_and_n_allowed(1234, 40, min_seed=1, max_seed=10, n_values=[40]) is False
    )
    assert _seed_and_n_allowed(0, 40, min_seed=1, max_seed=10, n_values=[40]) is False


def test_seed_and_n_allowed_filters_unlisted_n():
    assert (
        _seed_and_n_allowed(1, 40, min_seed=1, max_seed=10, n_values=[40, 80]) is True
    )
    assert (
        _seed_and_n_allowed(1, 30, min_seed=1, max_seed=10, n_values=[40, 80]) is False
    )


def test_find_local_candidates_excludes_out_of_range_seed(tmp_path: Path):
    """An older experiment phase's seed=1234 run must not silently
    pollute the mean of the current sweep's seeds 1-10."""
    _write_summary(tmp_path / "multirun" / "old_phase", seed=1234)
    _write_summary(tmp_path / "multirun" / "current", seed=3)

    candidates = _find_local_candidates(
        ["config6"], str(tmp_path / "outputs"), str(tmp_path / "multirun"), 1, 10, [40]
    )

    assert len(candidates) == 1
    assert candidates[0][0][2] == 3  # identity = (architecture, N, seed, ...)


def test_find_local_candidates_excludes_unlisted_n(tmp_path: Path):
    _write_summary(tmp_path / "multirun" / "old_n", N=30)
    _write_summary(tmp_path / "multirun" / "current", N=40)

    candidates = _find_local_candidates(
        ["config6"], str(tmp_path / "outputs"), str(tmp_path / "multirun"), 1, 10, [40]
    )

    assert len(candidates) == 1
    assert candidates[0][0][1] == 40  # identity = (architecture, N, ...)


class _FakeArtifact:
    def __init__(self, artifact_type: str):
        self.type = artifact_type
        self.downloaded_to: str | None = None

    def download(self, root: str) -> None:
        self.downloaded_to = root
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / "summary.json").write_text("{}")


class _FakeRun:
    def __init__(self, run_id: str, artifacts: list[_FakeArtifact]):
        self.id = run_id
        self._artifacts = artifacts
        self.config: dict[str, object] = {}

    def logged_artifacts(self) -> list[_FakeArtifact]:
        return self._artifacts


def test_download_run_locally_downloads_model_artifact(tmp_path: Path):
    model_artifact = _FakeArtifact("model")
    run = _FakeRun("run123", [_FakeArtifact("code"), model_artifact])

    _download_run_locally(run, tmp_path)

    assert model_artifact.downloaded_to == str(tmp_path / "run123")
    assert (tmp_path / "run123" / "summary.json").exists()


def test_download_run_locally_skips_if_already_downloaded(tmp_path: Path):
    dest = tmp_path / "run123"
    dest.mkdir(parents=True)
    (dest / "summary.json").write_text("{}")
    model_artifact = _FakeArtifact("model")
    run = _FakeRun("run123", [model_artifact])

    _download_run_locally(run, tmp_path)

    assert model_artifact.downloaded_to is None


def test_download_run_locally_does_not_raise_on_failure(tmp_path: Path):
    class _BrokenRun:
        id = "broken"

        def logged_artifacts(self) -> list[_FakeArtifact]:
            raise RuntimeError("network error")

    _download_run_locally(_BrokenRun(), tmp_path)  # must not raise


class _FakeWandbApi:
    def __init__(self, runs: list[_FakeRun]):
        self.default_entity = "test-entity"
        self._runs = runs

    def runs(
        self, path: str, filters: object = None, per_page: object = None
    ) -> list[_FakeRun]:
        return self._runs


class _FakeWandbModule:
    def __init__(self, runs: list[_FakeRun]):
        self._runs = runs

    def Api(self) -> _FakeWandbApi:
        return _FakeWandbApi(self._runs)


def _fake_run(config: dict[str, object], summary: dict[str, object]) -> _FakeRun:
    run = _FakeRun("run1", [_FakeArtifact("model")])
    run.config = config  # type: ignore[attr-defined]
    run.summary = summary  # type: ignore[attr-defined]
    return run


def test_fetch_grouped_results_skips_download_for_known_local_identity(
    monkeypatch, tmp_path: Path
):
    """A run whose identity already exists locally (from actual training,
    or a previous backfill) must not be re-downloaded — otherwise a later
    --local prune deletes the redundant copy and the next --save-local
    just re-downloads it, forever."""
    run = _fake_run(
        config={
            "architecture": "config6",
            "N": 40,
            "seed": 1,
            "augment_train": "none",
            "readout": "x0_xhalf",
            "class1": 3,
            "class2": 4,
        },
        summary={"val/accuracy": 0.9, "val_aug/accuracy": 0.85},
    )
    monkeypatch.setattr(pwr, "wandb", _FakeWandbModule([run]))
    known_local_identities = {("config6", 40, 1, "none", "x0_xhalf", 3, 4, "mnist")}

    grouped = _fetch_grouped_results(
        ["config6"],
        "proj",
        None,
        1,
        10,
        [40],
        save_local_dir=tmp_path,
        known_local_identities=known_local_identities,
    )

    artifact = run.logged_artifacts()[0]
    assert artifact.downloaded_to is None
    assert grouped[("config6", "none", "x0_xhalf", 3, 4)][40]["val"] == [0.9]


def test_fetch_grouped_results_downloads_when_not_known_locally(
    monkeypatch, tmp_path: Path
):
    run = _fake_run(
        config={
            "architecture": "config6",
            "N": 40,
            "seed": 1,
            "augment_train": "none",
            "readout": "x0_xhalf",
            "class1": 3,
            "class2": 4,
        },
        summary={"val/accuracy": 0.9, "val_aug/accuracy": 0.85},
    )
    monkeypatch.setattr(pwr, "wandb", _FakeWandbModule([run]))

    _fetch_grouped_results(
        ["config6"],
        "proj",
        None,
        1,
        10,
        [40],
        save_local_dir=tmp_path,
        known_local_identities=set(),
    )

    artifact = run.logged_artifacts()[0]
    assert artifact.downloaded_to == str(tmp_path / "run1")


def test_mirror_into_results_def_copies_when_under_project_root(
    monkeypatch, tmp_path: Path
):
    fake_root = tmp_path / "fake_project"
    fake_root.mkdir()
    monkeypatch.setattr(pwr, "_PROJECT_ROOT", fake_root)
    dest_dir = fake_root / "outputs" / "wandb_backfill" / "run1"
    dest_dir.mkdir(parents=True)
    (dest_dir / "final_model.pt").write_bytes(b"model-bytes")
    (dest_dir / "summary.json").write_text("{}")

    pwr._mirror_into_results_def(dest_dir)

    mirrored = fake_root / "results_def" / "outputs" / "wandb_backfill" / "run1"
    assert (mirrored / "final_model.pt").read_bytes() == b"model-bytes"


def test_mirror_into_results_def_skips_when_outside_project_root(tmp_path: Path):
    """dest_dir under a pytest tmp_path (as in every other test in this
    file that calls _download_run_locally) must never resolve to
    somewhere under the REAL project root — that would silently write
    into the real, git-tracked results_def/ during a test run."""
    outside_dir = tmp_path / "elsewhere" / "run1"
    outside_dir.mkdir(parents=True)
    (outside_dir / "final_model.pt").write_bytes(b"model-bytes")
    (outside_dir / "summary.json").write_text("{}")

    pwr._mirror_into_results_def(outside_dir)  # must not raise

    assert not (pwr._PROJECT_ROOT / "results_def" / "elsewhere").exists()


def test_patch_downloaded_summary_config_syncs_stale_fields(tmp_path: Path):
    """train.py's checkpoint_config and wandb_extra_config are separate
    dicts — some historical runs have augment_train missing/null in the
    archived summary.json's config even though run.config (authoritative)
    has it correctly. The local file must get synced to match."""
    dest_dir = tmp_path / "run1"
    dest_dir.mkdir()
    (dest_dir / "summary.json").write_text(
        json.dumps(
            {"config": {"architecture": "config6", "augment_train": None, "class1": 3}}
        )
    )
    run = _FakeRun("run1", [])
    run.config = {
        "architecture": "config6",
        "augment_train": "once",
        "readout": "x0_xhalf",
        "class1": 3,
        "class2": 4,
    }

    pwr._patch_downloaded_summary_config(dest_dir, run)

    patched = json.loads((dest_dir / "summary.json").read_text())
    assert patched["config"]["augment_train"] == "once"
    assert patched["config"]["readout"] == "x0_xhalf"
    assert patched["config"]["class2"] == 4


def test_patch_downloaded_summary_config_leaves_matching_file_untouched(
    tmp_path: Path,
):
    dest_dir = tmp_path / "run1"
    dest_dir.mkdir()
    original = json.dumps({"config": {"augment_train": "once"}})
    (dest_dir / "summary.json").write_text(original)
    run = _FakeRun("run1", [])
    run.config = {"augment_train": "once"}

    pwr._patch_downloaded_summary_config(dest_dir, run)

    assert (dest_dir / "summary.json").read_text() == original


def test_download_run_locally_patches_and_remirrors_already_downloaded_file(
    monkeypatch, tmp_path: Path
):
    """A file downloaded before this patching existed must get fixed
    retroactively on the next call, not just on fresh downloads."""
    fake_root = tmp_path / "fake_project"
    fake_root.mkdir()
    monkeypatch.setattr(pwr, "_PROJECT_ROOT", fake_root)
    dest_root = fake_root / "outputs" / "wandb_backfill"
    dest_dir = dest_root / "run1"
    dest_dir.mkdir(parents=True)
    (dest_dir / "final_model.pt").write_bytes(b"model-bytes")
    (dest_dir / "summary.json").write_text(
        json.dumps({"config": {"augment_train": None}})
    )
    run = _FakeRun("run1", [])
    run.config = {"augment_train": "online"}

    pwr._download_run_locally(run, dest_root)

    patched = json.loads((dest_dir / "summary.json").read_text())
    assert patched["config"]["augment_train"] == "online"
    mirrored = fake_root / "results_def" / "outputs" / "wandb_backfill" / "run1"
    mirrored_summary = json.loads((mirrored / "summary.json").read_text())
    assert mirrored_summary["config"]["augment_train"] == "online"
