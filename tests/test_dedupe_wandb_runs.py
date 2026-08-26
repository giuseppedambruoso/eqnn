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


def test_identity_key_does_not_require_img_size():
    """img_size is never logged into run.config by train.py's wandb.init
    (see main.py) — it must NOT be an identity field, or every historical
    run (missing it) would be silently excluded from dedup, as happened
    before this field list was pruned to only fields train.py actually
    logs."""
    assert "img_size" not in IDENTITY_FIELDS


def test_identity_key_matches_only_when_every_field_matches():
    a = _identity_key(_full_config())
    b = _identity_key(_full_config())
    c = _identity_key(_full_config(seed=2))
    assert a == b
    assert a != c


def test_identity_key_normalizes_legacy_bool_augment_train():
    """Runs logged before "once" existed used a bool for augment_train —
    False must match "none" and True must match "online" so an old and a
    new run of the otherwise-same config are recognized as duplicates
    instead of silently splitting into two separate identity groups."""
    legacy_false = _identity_key(_full_config(augment_train=False))
    modern_none = _identity_key(_full_config(augment_train="none"))
    legacy_true = _identity_key(_full_config(augment_train=True))
    modern_online = _identity_key(_full_config(augment_train="online"))
    assert legacy_false == modern_none
    assert legacy_true == modern_online
    assert legacy_false != legacy_true


def test_identity_key_defaults_missing_class1_class2_to_3_and_4():
    """Runs logged before DATA.class1/class2 existed never had these
    fields at all — they must default to (3, 4), the only pair ever used
    back then, matching a run that explicitly logged class1=3/class2=4,
    instead of being excluded from dedup entirely."""
    cfg = _full_config()
    del cfg["class1"]
    del cfg["class2"]
    legacy = _identity_key(cfg)
    modern = _identity_key(_full_config(class1=3, class2=4))
    assert legacy == modern


def test_identity_key_distinguishes_different_class_pairs():
    """A 3-vs-4 run and an otherwise-identical 4-vs-5 run must NOT be
    treated as duplicates of each other — they're different
    classification tasks, not a re-run of the same job."""
    pair_3v4 = _identity_key(_full_config(class1=3, class2=4))
    pair_4v5 = _identity_key(_full_config(class1=4, class2=5))
    assert pair_3v4 != pair_4v5


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
