import csv
import json
from pathlib import Path

from src.collect_results import (
    _copy_essentials,
    _find_run_dirs,
    _relative_mirror_path,
    _summary_row,
    main,
)


def _make_run(run_dir: Path, **summary_overrides: object) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "final_model.pt").write_bytes(b"fake-checkpoint-bytes")
    (run_dir / "loss_history.jpg").write_bytes(b"not-essential")
    summary = {
        "run_name": "config6_N=40_seed=1",
        "seed": 1,
        "N": 40,
        "epochs_completed": 40,
        "train_accuracy": 0.9,
        "val_accuracy": 0.88,
        "val_aug_accuracy": 0.87,
        "p4m_equivariance": {"is_invariant": True},
        "config": {
            "architecture": "config6",
            "readout": "x0_xhalf",
            "augment_train": "none",
        },
        **summary_overrides,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))


def test_find_run_dirs_across_multiple_roots(tmp_path: Path):
    outputs = tmp_path / "outputs" / "2026-01-01" / "10-00-00"
    multirun = tmp_path / "multirun" / "2026-01-01" / "11-00-00" / "job0"
    _make_run(outputs)
    _make_run(multirun)

    found = _find_run_dirs(str(tmp_path / "outputs"), str(tmp_path / "multirun"))

    assert set(found) == {outputs, multirun}


def test_find_run_dirs_ignores_missing_roots(tmp_path: Path):
    assert _find_run_dirs(str(tmp_path / "nonexistent")) == []


def test_copy_essentials_excludes_bulkier_files(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "run1"
    _make_run(run_dir)
    results_root = tmp_path / "results"

    dest_dir = _copy_essentials(run_dir, results_root)

    assert (dest_dir / "final_model.pt").read_bytes() == b"fake-checkpoint-bytes"
    assert (dest_dir / "summary.json").exists()
    assert not (dest_dir / "loss_history.jpg").exists()


def test_copy_essentials_never_writes_onto_the_source(tmp_path: Path):
    """An absolute run_dir must not collide with its own mirror under
    results_root (joining an absolute Path onto another discards the left
    side entirely — dest_dir must never end up equal to run_dir)."""
    run_dir = tmp_path / "outputs" / "run1"
    _make_run(run_dir)
    assert run_dir.is_absolute()
    results_root = tmp_path / "results"

    dest_dir = _copy_essentials(run_dir, results_root)

    assert dest_dir != run_dir
    assert str(dest_dir).startswith(str(results_root))


def test_copy_essentials_reads_from_run_dir_but_mirrors_at_explicit_path(
    tmp_path: Path,
):
    """mirror_path lets the caller read from an absolute run_dir (e.g.
    Hydra having chdir'd into it) while anchoring the mirrored structure
    somewhere else entirely — train.py's project-root anchoring relies on
    this to avoid nesting results_def/ under each job's own directory."""
    run_dir = tmp_path / "some" / "absolute" / "job_dir"
    _make_run(run_dir)
    results_root = tmp_path / "results_def"
    mirror_path = Path("multirun") / "2026-01-01" / "job0"

    dest_dir = _copy_essentials(run_dir, results_root, mirror_path)

    assert dest_dir == results_root / mirror_path
    assert (dest_dir / "final_model.pt").read_bytes() == b"fake-checkpoint-bytes"


def test_relative_mirror_path_strips_absolute_anchor():
    assert not _relative_mirror_path(Path("/a/b/c")).is_absolute()
    assert _relative_mirror_path(Path("a/b/c")) == Path("a/b/c")


def test_summary_row_pulls_expected_fields(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "run1"
    _make_run(run_dir)

    row = _summary_row(run_dir)

    assert row is not None
    assert row["architecture"] == "config6"
    assert row["readout"] == "x0_xhalf"
    assert row["augment_train"] == "none"
    assert row["val_accuracy"] == 0.88
    assert row["p4m_is_invariant"] is True


def test_summary_row_missing_summary_json_returns_none(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "final_model.pt").write_bytes(b"x")

    assert _summary_row(run_dir) is None


def test_main_end_to_end(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_run(Path("outputs") / "2026-01-01" / "10-00-00")
    _make_run(Path("multirun") / "2026-01-01" / "11-00-00" / "job0", seed=2)

    import sys

    monkeypatch.setattr(sys, "argv", ["collect_results.py"])
    main()

    results_root = Path("results_def")
    csv_path = results_root / "summary.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {row["seed"] for row in rows} == {"1", "2"}
    for row in rows:
        run_dir = Path(row["run_dir"])
        assert (run_dir / "final_model.pt").exists()
        assert not (run_dir / "loss_history.jpg").exists()
