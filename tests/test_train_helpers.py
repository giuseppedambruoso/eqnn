from pathlib import Path

import torch

from src.train import _plot_confusion_matrix, _results_def_mirror_path


def test_results_def_mirror_path_relative_to_root(tmp_path):
    project_root = tmp_path / "project"
    job_dir = project_root / "multirun" / "2026-01-01" / "job0"
    job_dir.mkdir(parents=True)

    mirror = _results_def_mirror_path(str(job_dir), project_root)

    assert mirror == Path("multirun") / "2026-01-01" / "job0"


def test_results_def_mirror_path_none_outside_root(tmp_path):
    """pytest's tmp_path (as used by test_custom_training.py, where
    job_dir = os.getcwd() falls outside the repo) must not resolve to a
    mirror path — that would pollute the real, git-tracked results_def/
    with throwaway test artifacts, which is exactly the bug this guards
    against."""
    project_root = tmp_path / "project"
    outside_dir = tmp_path / "elsewhere" / "job0"
    outside_dir.mkdir(parents=True)

    assert _results_def_mirror_path(str(outside_dir), project_root) is None


def test_plot_confusion_matrix_labels_use_given_classes(tmp_path):
    """Axis labels must reflect the actual class1/class2 used, not a
    hardcoded 3/4 — a 4-vs-5 run must not display "3"/"4" tick labels."""
    predictions = torch.tensor([0.1, 0.9, 0.2, 0.8])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    destination = str(tmp_path / "confusion_matrix.png")

    _plot_confusion_matrix(predictions, labels, destination, class1=4, class2=5)

    assert Path(destination).exists()
