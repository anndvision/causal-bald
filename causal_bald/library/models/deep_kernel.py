import torch

from torch import nn
from torch import optim
from torch.utils import data

from gpytorch import mlls
from gpytorch import likelihoods

from ignite import metrics

from causal_bald.library.models import core
from causal_bald.library.modules import dense
from causal_bald.library.modules import convolution
from causal_bald.library.modules import gaussian_process


class DeepKernelGP(core.PyTorchModel):
    def __init__(
        self,
        job_dir,
        kernel,
        num_inducing_points,
        inducing_point_dataset,
        architecture,
        dim_input,
        dim_hidden,
        dim_output,
        depth,
        negative_slope,
        batch_norm,
        spectral_norm,
        dropout_rate,
        weight_decay,
        learning_rate,
        batch_size,
        epochs,
        patience,
        num_workers,
        seed,
    ):
        super(DeepKernelGP, self).__init__(
            job_dir=job_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            num_workers=num_workers,
        )
        if isinstance(dim_input, list):
            self.encoder = convolution.ResNet(
                dim_input=dim_input,
                layers=[2] * depth,
                base_width=dim_hidden // 8,
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
                stem_kernel_size=5,
                stem_kernel_stride=1,
                stem_kernel_padding=2,
                stem_pool=False,
                activate_output=True,
            )
        else:
            self.encoder = nn.Sequential(
                dense.NeuralNetwork(
                    architecture=architecture,
                    dim_input=dim_input,
                    dim_hidden=dim_hidden,
                    depth=depth,
                    negative_slope=negative_slope,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate,
                    spectral_norm=spectral_norm,
                    activate_output=False,
                ),
                dense.Activation(
                    dim_input=None,
                    negative_slope=negative_slope,
                    dropout_rate=0.0,
                    batch_norm=batch_norm,
                ),
            )

        self.encoder.to(self.device)

        self.batch_size = batch_size
        self.best_loss = 1e7
        self.patience = patience
        (
            initial_inducing_points,
            initial_lengthscale,
        ) = gaussian_process.initial_values_for_GP(
            train_dataset=inducing_point_dataset,
            feature_extractor=self.encoder,
            n_inducing_points=num_inducing_points,
            device=self.device,
        )
        self.gp = gaussian_process.VariationalGP(
            num_outputs=dim_output,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            separate_inducing_points=False,
            kernel=kernel,
            ard=None,
            lengthscale_prior=False,
        ).to(self.device)
        self.network = gaussian_process.DeepKernelGP(
            encoder=self.encoder,
            gp=self.gp,
        )
        self.likelihood = likelihoods.GaussianLikelihood()
        self.optimizer = optim.Adam(
            params=[
                {"params": self.encoder.parameters(), "lr": self.learning_rate},
                {"params": self.gp.parameters(), "lr": 2 * self.learning_rate},
                {"params": self.likelihood.parameters(), "lr": 2 * self.learning_rate},
            ],
            weight_decay=weight_decay,
        )
        self.loss = mlls.VariationalELBO(
            likelihood=self.likelihood,
            model=self.network.gp,
            num_data=len(inducing_point_dataset),
        )
        self.metrics = {
            "loss": metrics.Average(
                output_transform=lambda x: -self.likelihood.expected_log_prob(
                    x["targets"].squeeze(), x["outputs"]
                ).mean(),
                device=self.device,
            )
        }
        self.network.to(self.device)
        self.likelihood.to(self.device)

    def train_step(self, engine, batch):
        self.network.train()
        self.likelihood.train()
        inputs, targets = self.preprocess(batch)
        self.optimizer.zero_grad()
        outputs = self.network(inputs)
        loss = -self.loss(outputs, targets.squeeze()).mean()
        loss.backward()
        self.optimizer.step()
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def tune_step(self, engine, batch):
        self.network.eval()
        self.likelihood.eval()
        inputs, targets = self.preprocess(batch)
        with torch.no_grad():
            outputs = self.network(inputs)
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def predict_mus(self, ds, batch_size=None):
        dl = data.DataLoader(
            ds,
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        mu_0 = []
        mu_1 = []
        self.network.eval()
        with torch.no_grad():
            for batch in dl:
                batch = self.preprocess(batch)
                covariates = torch.cat([batch[0][:, :-1], batch[0][:, :-1]], 0)
                treatments = torch.cat(
                    [
                        torch.zeros_like(batch[0][:, -1:]),
                        torch.ones_like(batch[0][:, -1:]),
                    ],
                    0,
                )
                inputs = torch.cat([covariates, treatments], -1)
                posterior_predictive = self.network(inputs)
                samples = posterior_predictive.sample(torch.Size([1000]))
                mus = samples.chunk(2, dim=1)
                mu_0.append(mus[0])
                mu_1.append(mus[1])
        return (
            torch.cat(mu_0, 1).to("cpu").numpy(),
            torch.cat(mu_1, 1).to("cpu").numpy(),
        )
