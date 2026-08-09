from scripts.audit_mmp_transform_effect_train_only import canonical_key_and_sign


def test_canonical_key_preserves_direction_sign() -> None:
    assert canonical_key_and_sign("[*:1]C>>[*:1]N") == ("[*:1]C>>[*:1]N", 1)
    assert canonical_key_and_sign("[*:1]N>>[*:1]C") == ("[*:1]C>>[*:1]N", -1)
