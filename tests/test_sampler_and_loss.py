import torch

from dr_detection.loss import FocalLoss, OrdinalDistanceLoss
from dr_detection.sampler import BalancedBatchSampler, ClassAwareRandomSampler


def test_class_aware_random_sampler():
    # 80 zeros, 20 ones
    labels = [0] * 80 + [1] * 20
    sampler = ClassAwareRandomSampler(labels, num_samples=100)
    sampled = list(sampler)
    assert len(sampled) == 100
    sampled_labels = [labels[i] for i in sampled]
    # In expectation, class 0 and 1 should each be ~50%
    class_1_ratio = sum(sampled_labels) / len(sampled_labels)
    assert 0.35 <= class_1_ratio <= 0.65


def test_balanced_batch_sampler():
    # 5 classes with uneven distribution
    labels = [0] * 100 + [1] * 30 + [2] * 50 + [3] * 15 + [4] * 20
    samples_per_class = 4
    sampler = BalancedBatchSampler(labels, samples_per_class=samples_per_class)
    batches = list(sampler)
    assert len(batches) > 0
    for batch in batches:
        assert len(batch) == 5 * samples_per_class
        batch_labels = [labels[idx] for idx in batch]
        # Each class must be present exactly 4 times in every batch
        for cls in range(5):
            assert batch_labels.count(cls) == samples_per_class


def test_focal_loss():
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None


def test_ordinal_distance_loss():
    loss_fn = OrdinalDistanceLoss(num_classes=5, distance_weight=0.5)
    logits = torch.randn(8, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None
