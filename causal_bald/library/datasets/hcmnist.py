import typing

import numpy as np
import torch
from sklearn import model_selection
from torchvision import datasets

from causal_bald.library.datasets import utils


class HCMNIST(datasets.MNIST):
    def __init__(
        self,
        root: str,
        split: str = "train",
        mode: str = "mu",
        beta: float = 2.0,
        sigma_y: float = 0.01,
        domain: float = 3.0,
        subsample: dict = None,
        seed: int = 1331,
        transform: typing.Optional[typing.Callable] = None,
        target_transform: typing.Optional[typing.Callable] = None,
        download: bool = True,
    ) -> None:
        train = split == "train" or split == "valid"
        super(HCMNIST, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.data = self.data.view(len(self.targets), -1).numpy()
        self.targets = self.targets.numpy()

        rng = np.random.RandomState(seed=seed)

        if train:
            (
                data_train,
                data_valid,
                targets_train,
                targets_valid,
            ) = model_selection.train_test_split(
                self.data, self.targets, test_size=0.3, random_state=seed
            )
            self.data = data_train if split == "train" else data_valid
            self.targets = targets_train if split == "train" else targets_valid
            if subsample is not None:
                for digit, frequency in subsample.items():
                    idx = np.where(self.targets == digit)[0]
                    idx_delete = rng.choice(
                        idx, size=int(len(idx) * (1 - frequency)), replace=False
                    )
                    self.data = np.delete(self.data, idx_delete, axis=0)
                    self.targets = np.delete(self.targets, idx_delete, axis=0)

        self.x = ((self.data.astype("float32") / 255.0) - 0.1307) / 0.3081

        self.mode = mode
        self.dim_input = [1, 28, 28]
        self.dim_treatment = 1
        self.dim_output = 1

        self.phi_model = fit_phi_model(
            root=root, edges=torch.arange(-domain, domain + 0.1, (2 * domain) / 10),
        )

        phi = self.phi
        self.pi = (
            utils.complete_propensity(x=phi, u=0.0, lambda_=1.0, beta=beta)
            .astype("float32")
            .ravel()
        )
        self.t = rng.binomial(1, self.pi).astype("float32")
        eps = (sigma_y * rng.normal(size=self.t.shape)).astype("float32")
        self.mu0 = utils.f_mu(x=phi, t=0.0, u=0.0, gamma=0.0).astype("float32").ravel()
        self.mu1 = utils.f_mu(x=phi, t=1.0, u=0.0, gamma=0.0).astype("float32").ravel()
        self.y0 = self.mu0 + eps
        self.y1 = self.mu1 + eps
        self.y = self.t * self.y1 + (1 - self.t) * self.y0
        self.tau = self.mu1 - self.mu0
        self.y_mean = np.array([0.0], dtype="float32")
        self.y_std = np.array([1.0], dtype="float32")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        t = self.t[index : index + 1]
        if self.mode == "pi":
            return torch.from_numpy(x), torch.from_numpy(t)
        elif self.mode == "mu":
            y = self.y[index : index + 1]
            return torch.from_numpy(np.hstack([x, t])), torch.from_numpy(y)
        else:
            raise NotImplementedError(
                f"{self.mode} not supported. Choose from 'pi'  for propensity models or 'mu' for expected outcome models"
            )

    @property
    def phi(self):
        x = self.x
        z = np.zeros_like(self.targets.astype("float32"))
        for k, v in self.phi_model.items():
            ind = self.targets == k
            x_ind = x[ind].reshape(ind.sum(), -1)
            means = x_ind.mean(axis=-1)
            z[ind] = utils.linear_normalization(
                np.clip((means - v["mu"]) / v["sigma"], -1.4, 1.4), v["lo"], v["hi"]
            )
        return np.expand_dims(z, -1)


def fit_phi_model(root, edges):
    ds = datasets.MNIST(root=root, download=True)
    data = (ds.data.float().div(255) - 0.1307).div(0.3081).view(len(ds), -1)
    model = {}
    digits = torch.unique(ds.targets)
    for i, digit in enumerate(digits):
        lo, hi = edges[i : i + 2]
        ind = ds.targets == digit
        data_ind = data[ind].view(ind.sum(), -1)
        means = data_ind.mean(dim=-1)
        mu = means.mean()
        sigma = means.std()
        model.update(
            {
                digit.item(): {
                    "mu": mu.item(),
                    "sigma": sigma.item(),
                    "lo": lo.item(),
                    "hi": hi.item(),
                }
            }
        )
    return model
