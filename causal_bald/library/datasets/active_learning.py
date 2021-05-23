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

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.training_mask = np.full((len(dataset),), False)
        self.pool_mask = np.full((len(dataset),), True)

        self.training_dataset = data.Subset(self.dataset, None)
        self.pool_dataset = data.Subset(self.dataset, None)

        self._update_indices()

    def _update_indices(self):
        self.training_dataset.indices = np.nonzero(self.training_mask)[0]
        self.pool_dataset.indices = np.nonzero(self.pool_mask)[0]

    def _remove_from_pool(self, pool_indices):
        indices = self._get_dataset_indices(pool_indices)

        self.pool_mask[indices] = False
        self._update_indices()

    def _get_dataset_indices(self, pool_indices):
        """Transform indices (in `pool_dataset`) to indices in the original `dataset`."""
        indices = self.pool_dataset.indices[pool_indices]
        return indices

    def acquire(self, pool_indices):
        """Acquire elements from the pool dataset into the training dataset.
        Add them to training dataset & remove them from the pool dataset."""

        self.training_mask[pool_indices] = True
        self._remove_from_pool(pool_indices)

    def is_empty(self):
        return len(self.pool_dataset) == 0

    def get_pool_dataset(self):
        return self.pool_dataset

    def get_training_dataset(self):
        return self.training_dataset
