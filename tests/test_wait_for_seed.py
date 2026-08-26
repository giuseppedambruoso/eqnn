from src.wait_for_seed import _expected_combos, _normalize_augment_train


def test_expected_combos_is_cartesian_product():
    combos = _expected_combos(["config6", "config7"], [40, 80], ["none", "online"])
    assert combos == {
        ("config6", 40, "none"),
        ("config6", 40, "online"),
        ("config6", 80, "none"),
        ("config6", 80, "online"),
        ("config7", 40, "none"),
        ("config7", 40, "online"),
        ("config7", 80, "none"),
        ("config7", 80, "online"),
    }


def test_normalize_augment_train_maps_legacy_bool():
    assert _normalize_augment_train(True) == "online"
    assert _normalize_augment_train(False) == "none"
    assert _normalize_augment_train("true") == "online"
    assert _normalize_augment_train("false") == "none"
    assert _normalize_augment_train("once") == "once"
