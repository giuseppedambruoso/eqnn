from dataclasses import dataclass, field
from typing import Any

from src.dedupe_wandb_runs import IDENTITY_FIELDS, _identity_key, _pick_keep_and_drop


@dataclass
class FakeRun:
    id: str
    created_at: str
    summary: dict[str, Any] = field(default_factory=dict)
    name: str = "fake-run"

    def delete(self) -> None:
        raise AssertionError("delete() should never be called by the pure logic")


def _full_config(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {field_name: field_name for field_name in IDENTITY_FIELDS}
    base["N"] = 40
    base["seed"] = 1
    base.update(overrides)
    return base


def test_identity_key_missing_field_is_none():
    cfg = _full_config()
    del cfg["readout"]
    assert _identity_key(cfg) is None


def test_identity_key_matches_only_when_every_field_matches():
    a = _identity_key(_full_config())
    b = _identity_key(_full_config())
    c = _identity_key(_full_config(seed=2))
    assert a == b
    assert a != c


def test_pick_keep_and_drop_prefers_finished_run():
    finished = FakeRun(
        id="finished", created_at="2024-01-01", summary={"val/accuracy": 0.9}
    )
    crashed = FakeRun(id="crashed", created_at="2024-06-01", summary={})
    keep, drop = _pick_keep_and_drop([crashed, finished])
    assert keep.id == "finished"
    assert [run.id for run in drop] == ["crashed"]


def test_pick_keep_and_drop_breaks_ties_by_recency():
    older = FakeRun(id="older", created_at="2024-01-01", summary={"val/accuracy": 0.8})
    newer = FakeRun(id="newer", created_at="2024-06-01", summary={"val/accuracy": 0.85})
    keep, drop = _pick_keep_and_drop([older, newer])
    assert keep.id == "newer"
    assert [run.id for run in drop] == ["older"]
