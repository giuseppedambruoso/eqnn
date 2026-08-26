from typing import cast

from src.plot_wandb_results import _restrict_to_common_seeds


def _bucket(seeds: list[int]) -> dict[str, list[object]]:
    return {
        "val": [0.1 * s for s in seeds],
        "val_aug": [0.2 * s for s in seeds],
        "seed": cast(list[object], list(seeds)),
    }


def test_restricts_to_seeds_common_across_augment_train_modes():
    grouped = {
        ("config6", "none", "x0_xhalf"): {40: _bucket([1, 2, 3])},
        ("config6", "online", "x0_xhalf"): {40: _bucket([1, 2])},
        ("config6", "once", "x0_xhalf"): {40: _bucket([1, 2, 3, 4])},
    }

    restricted = _restrict_to_common_seeds(grouped)

    for key in grouped:
        assert restricted[key][40]["seed"] == [1, 2]


def test_restricts_across_architectures_too():
    """A series with extra seeds no other architecture reaches yet must
    not keep them — every line in the final plot, including different
    colors (architectures), must reflect the same set of trained models."""
    grouped = {
        ("config6", "none", "x0_xhalf"): {40: _bucket([1, 2, 3, 4, 5, 6, 7])},
        ("config6", "online", "x0_xhalf"): {40: _bucket([1, 2, 3, 4, 5, 6, 7])},
        ("config6", "once", "x0_xhalf"): {40: _bucket([1, 2, 3, 4, 5, 6, 7])},
        ("config7", "none", "x0_xhalf"): {40: _bucket([1, 2, 3, 4, 5, 6])},
        ("config7", "online", "x0_xhalf"): {40: _bucket([1, 2, 3, 4, 5, 6])},
        ("config7", "once", "x0_xhalf"): {40: _bucket([1, 2, 3, 4, 5, 6])},
    }

    restricted = _restrict_to_common_seeds(grouped)

    for key in grouped:
        assert restricted[key][40]["seed"] == [1, 2, 3, 4, 5, 6]


def test_keeps_full_seed_set_when_only_one_mode_present():
    grouped = {("config6", "none", "x0_xhalf"): {40: _bucket([1, 2, 3])}}

    restricted = _restrict_to_common_seeds(grouped)

    assert restricted[("config6", "none", "x0_xhalf")][40]["seed"] == [1, 2, 3]


def test_drops_point_entirely_when_no_seed_is_common():
    grouped = {
        ("config6", "none", "x0_xhalf"): {40: _bucket([1, 2])},
        ("config6", "online", "x0_xhalf"): {40: _bucket([3, 4])},
    }

    restricted = _restrict_to_common_seeds(grouped)

    assert 40 not in restricted[("config6", "none", "x0_xhalf")]
    assert 40 not in restricted[("config6", "online", "x0_xhalf")]


def test_restriction_is_per_n_not_global():
    grouped = {
        ("config6", "none", "x0_xhalf"): {40: _bucket([1, 2]), 80: _bucket([1, 2, 3])},
        ("config6", "online", "x0_xhalf"): {
            40: _bucket([1, 2, 3]),
            80: _bucket([1, 2]),
        },
    }

    restricted = _restrict_to_common_seeds(grouped)

    assert restricted[("config6", "none", "x0_xhalf")][40]["seed"] == [1, 2]
    assert restricted[("config6", "none", "x0_xhalf")][80]["seed"] == [1, 2]


def test_values_and_seeds_stay_aligned_after_filtering():
    grouped = {
        ("config6", "none", "x0_xhalf"): {40: _bucket([1, 2, 3])},
        ("config6", "online", "x0_xhalf"): {40: _bucket([2, 3])},
    }

    restricted = _restrict_to_common_seeds(grouped)

    bucket = restricted[("config6", "none", "x0_xhalf")][40]
    for seed, val in zip(bucket["seed"], bucket["val"], strict=True):
        assert val == 0.1 * seed
