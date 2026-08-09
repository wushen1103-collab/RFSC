import torch

from scripts.train_lac_mlp_scg import curriculum_keep_mask, lac_edge_losses


def test_lac_edge_loss_rewards_correct_delta_direction():
    truth = torch.tensor([3.0, 1.0])
    left = torch.tensor([0])
    right = torch.tensor([1])
    aligned = lac_edge_losses(truth, torch.tensor([2.0, 1.0]), left, right)
    reversed_delta = lac_edge_losses(truth, torch.tensor([1.0, 2.0]), left, right)
    assert aligned.item() < 0
    assert reversed_delta.item() > 0


def test_lac_curriculum_discards_bottom_fraction():
    loss = torch.arange(10, dtype=torch.float32)
    keep = curriculum_keep_mask(loss, 0.2)
    assert keep.sum().item() == 8
    assert not keep[0]
    assert not keep[1]
    assert keep[-1]
