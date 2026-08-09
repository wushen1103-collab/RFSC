from scripts.train_gine import singleton_safe_batch_size


def test_singleton_safe_batch_size_changes_only_singleton_remainder():
    assert singleton_safe_batch_size(4097, 64) == 63
    assert singleton_safe_batch_size(4098, 64) == 64
    assert singleton_safe_batch_size(32, 64) == 32
