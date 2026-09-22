from __future__ import annotations

import random
from collections.abc import Iterator
from typing import List

import numpy as np
import torch
from torch.utils.data import Sampler


class ClassAwareRandomSampler(Sampler[int]):
    """Samples elements such that each class has equal overall probability of being selected.
    Calculates weights inversely proportional to class frequencies.
    """
    def __init__(self, labels: list[int] | np.ndarray, num_samples: int | None = None, replacement: bool = True):
        self.labels = np.asarray(labels, dtype=int)
        classes, counts = np.unique(self.labels, return_counts=True)
        class_to_count = dict(zip(classes, counts, strict=True))
        
        # Calculate sample weights: 1.0 / count(class)
        self.weights = torch.tensor([1.0 / class_to_count[lbl] for lbl in self.labels], dtype=torch.double)
        self.num_samples = len(self.labels) if num_samples is None else num_samples
        self.replacement = replacement

    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(self.weights, self.num_samples, replacement=self.replacement)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class BalancedBatchSampler(Sampler[List[int]]):
    """Batch sampler that guarantees each mini-batch contains an exact equal quota of
    samples per class. Essential for preventing majority class dominance during
    stochastic gradient descent on heavily imbalanced medical datasets.
    
    Example:
        For 5 DR classes and samples_per_class=6, each batch has size 30 (6 per grade).
    """
    def __init__(self, labels: list[int] | np.ndarray, samples_per_class: int = 6):
        self.labels = np.asarray(labels, dtype=int)
        self.samples_per_class = samples_per_class
        
        self.classes = np.unique(self.labels).tolist()
        self.num_classes = len(self.classes)
        self.batch_size = self.num_classes * self.samples_per_class
        
        # Group indices by class
        self.class_indices: dict[int, list[int]] = {
            cls: np.where(self.labels == cls)[0].tolist() for cls in self.classes
        }
        
        # Determine number of batches per epoch based on total samples
        self.num_batches = len(self.labels) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices for each class independently
        shuffled_indices = {
            cls: random.sample(idxs, len(idxs)) for cls, idxs in self.class_indices.items()
        }
        pointers = {cls: 0 for cls in self.classes}
        
        for _ in range(self.num_batches):
            batch = []
            for cls in self.classes:
                ptr = pointers[cls]
                idxs = shuffled_indices[cls]
                
                # If exhausted, reshuffle and reset pointer
                if ptr + self.samples_per_class > len(idxs):
                    random.shuffle(idxs)
                    shuffled_indices[cls] = idxs
                    ptr = 0
                
                selected = idxs[ptr : ptr + self.samples_per_class]
                pointers[cls] = ptr + self.samples_per_class
                batch.extend(selected)
                
            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches
