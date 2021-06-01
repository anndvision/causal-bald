import abc

import torch
import numpy as np

from torch.utils import data


class IActiveLearningDataset(abc.ABC):
    @abc.abstractmethod
    def acquire(self, pool_indices):
        pass

    @abc.abstractmethod
    def is_empty(self):
        pass

    def get_random_pool_indices(self, size):
        assert 0 <= size <= len(self.pool_dataset)
        pool_indices = torch.randperm(len(self.pool_dataset))[:size]
        return pool_indices


class ActiveLearningDataset(IActiveLearningDataset):
    # Obtained from: https://github.com/BlackHC/batchbald_redux/blob/master/batchbald_redux/active_learning.py

    """Splits `dataset` into an active dataset and an available dataset."""

    """
    This class can be used as follows:
    active_learning_data = ActiveLearningData(train_dataset)
    validation_data = active_learning_data.extract_dataset_from_pool(1000)
    initial_samples = active_learning_data.get_rand_pool_indices(20)
    active_learning_data.acquire(initial_samples)
    """

    def __init__(self, dataset, start_indices=None):
        super().__init__()
        self.dataset = dataset

        self.training_mask = np.full((len(dataset),), False)
        self.pool_mask = np.full((len(dataset),), True)

        if start_indices is not None:
            self.training_mask[start_indices] = True
            self.pool_mask[start_indices] = False

        self.training_dataset = data.Subset(self.dataset, None)
        self.pool_dataset = data.Subset(self.dataset, None)

        self._update_indices()

    def _update_indices(self):
        self.training_dataset.indices = np.nonzero(self.training_mask)[0]
        self.pool_dataset.indices = np.nonzero(self.pool_mask)[0]

    @property
    def acquired_indices(self):
        return self.training_dataset.indices

    def is_empty(self):
        return len(self.pool_dataset) == 0

    def get_random_pool_indices(self, size):
        assert 0 <= size <= len(self.pool_dataset)
        pool_indices = torch.randperm(len(self.pool_dataset))[:size]
        return pool_indices

    def get_dataset_indices(self, pool_indices):
        """Transform indices (in `pool_dataset`) to indices in the original `dataset`."""
        indices = self.pool_dataset.indices[pool_indices]
        return indices

    def acquire(self, pool_indices):
        """Acquire elements from the pool dataset into the training dataset.
        Add them to training dataset & remove them from the pool dataset."""
        indices = self.get_dataset_indices(pool_indices)

        self.training_mask[indices] = True
        self.pool_mask[indices] = False
        self._update_indices()


class RandomFixedLengthSampler(data.Sampler):
    """
    Sometimes, you really want to do more with little data without increasing the number of epochs.
    This sampler takes a `dataset` and draws `target_length` samples from it (with repetition).
    """

    def __init__(self, dataset, target_length):
        super().__init__(dataset)
        self.dataset = dataset
        self.target_length = target_length

    def __iter__(self):
        # Ensure that we don't lose data by accident.
        if self.target_length < len(self.dataset):
            return iter(torch.randperm(len(self.dataset)).tolist())

        # Sample slightly more indices to avoid biasing towards start of dataset
        indices = torch.randperm(
            self.target_length + (-self.target_length % len(self.dataset))
        )

        return iter((indices[: self.target_length] % len(self.dataset)).tolist())

    def __len__(self):
        return self.target_length
