import math
import torch
import gpytorch

from sklearn import cluster

from gpytorch import models
from gpytorch import priors
from gpytorch import kernels
from gpytorch import variational


class VariationalGP(models.ApproximateGP):
    def __init__(
        self,
        num_outputs,
        initial_lengthscale,
        initial_inducing_points,
        separate_inducing_points=False,
        kernel="RBF",
        ard=None,
        lengthscale_prior=False,
    ):
        n_inducing_points = initial_inducing_points.shape[0]
        if separate_inducing_points:
            # Use independent inducing points per output GP
            initial_inducing_points = initial_inducing_points.repeat(num_outputs, 1, 1)

        if num_outputs > 1:
            batch_shape = torch.Size([num_outputs])
        else:
            batch_shape = torch.Size([])

        variational_distribution = variational.CholeskyVariationalDistribution(
            n_inducing_points, batch_shape=batch_shape
        )

        variational_strategy = variational.VariationalStrategy(
            self,
            initial_inducing_points,
            variational_distribution,
        )

        if num_outputs > 1:
            variational_strategy = variational.IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=num_outputs
            )

        super(VariationalGP, self).__init__(variational_strategy)

        if lengthscale_prior:
            lengthscale_prior = priors.SmoothedBoxPrior(
                math.exp(-1), math.exp(1), sigma=0.1
            )
        else:
            lengthscale_prior = None

        kwargs = {
            "ard_num_dims": ard,
            "batch_shape": batch_shape,
            "lengthscale_prior": lengthscale_prior,
        }

        if kernel == "RBF":
            kernel = kernels.RBFKernel(**kwargs)
        elif kernel == "Matern12":
            kernel = kernels.MaternKernel(nu=1 / 2, **kwargs)
        elif kernel == "Matern32":
            kernel = kernels.MaternKernel(nu=3 / 2, **kwargs)
        elif kernel == "Matern52":
            kernel = kernels.MaternKernel(nu=5 / 2, **kwargs)
        elif kernel == "RQ":
            kernel = kernels.RQKernel(**kwargs)
        else:
            raise ValueError("Specified kernel not known.")

        kernel.lengthscale = initial_lengthscale * torch.ones_like(kernel.lengthscale)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel, batch_shape=batch_shape
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DeepKernelGP(gpytorch.Module):
    def __init__(self, encoder, gp):
        super(DeepKernelGP, self).__init__()

        self.encoder = encoder
        self.gp = gp

    def forward(self, inputs):
        phi = self.encoder(inputs[:, :-1])
        t = inputs[:, -1:]
        return self.gp(torch.cat([phi, t], dim=-1))


def initial_values_for_GP(train_dataset, feature_extractor, n_inducing_points, device):
    steps = 10

    idx = torch.randperm(len(train_dataset))[:1000].chunk(steps)
    f_X_samples = []

    with torch.no_grad():
        for i in range(steps):
            X_sample = torch.stack([train_dataset[j][0][:-1] for j in idx[i]])
            if torch.cuda.is_available():
                X_sample = X_sample.to(device)
                feature_extractor = feature_extractor.to(device)
            phi = feature_extractor(X_sample)
            phi = torch.cat([phi, torch.rand_like(phi[:, :1])], dim=-1)
            f_X_samples.append(phi.cpu())

    f_X_samples = torch.cat(f_X_samples)
    initial_inducing_points = _get_initial_inducing_points(
        f_X_samples.numpy(), min(f_X_samples.shape[0], n_inducing_points)
    )
    if initial_inducing_points.shape[0] < n_inducing_points:
        # Did not create enough inducing points
        initial_inducing_points = initial_inducing_points.repeat(
            n_inducing_points // initial_inducing_points.shape[0] + 1, 1
        )
        initial_inducing_points = initial_inducing_points[:n_inducing_points]
        # break ties between inducing points for numerical stability
        initial_inducing_points = torch.normal(initial_inducing_points, 0.01)

    initial_lengthscale = _get_initial_lengthscale(f_X_samples.to(device))

    return initial_inducing_points.to("cpu"), initial_lengthscale.to("cpu")


def _get_initial_inducing_points(f_X_sample, n_inducing_points):
    kmeans = cluster.MiniBatchKMeans(n_clusters=n_inducing_points)
    kmeans.fit(f_X_sample)
    initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)

    return initial_inducing_points


def _get_initial_lengthscale(f_X_samples):

    initial_lengthscale = torch.pdist(f_X_samples).mean()

    return initial_lengthscale
