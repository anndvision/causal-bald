import json
import numpy as np

from pathlib import Path

from causal_bald.library import models
from causal_bald.library import datasets
from causal_bald.library import acquisitions


def active_deep_kernel_gp(config, experiment_dir, trial):
    # Set dataset seeds
    dataset_name = config.get("dataset_name")
    config["ds_train"]["seed"] = trial
    config["ds_valid"]["seed"] = trial + 1 if dataset_name == "synthetic" else trial
    config["ds_test"]["seed"] = trial + 2 if dataset_name == "synthetic" else trial
    # Get datasets
    ds_active = datasets.ActiveLearningDataset(
        datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
    )
    ds_valid = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))
    # Get model parameters from config
    kernel = config.get("kernel")
    num_inducing_points = config.get("num_inducing_points")
    dim_hidden = config.get("dim_hidden")
    dim_output = config.get("dim_output")
    depth = config.get("depth")
    negative_slope = config.get("negative_slope")
    dropout_rate = config.get("dropout_rate")
    spectral_norm = config.get("spectral_norm")
    learning_rate = config.get("learning_rate")
    batch_size = config.get("batch_size")
    epochs = config.get("epochs")
    # Set the trial dir
    trial_dir = (
        Path(experiment_dir)
        / "deep_kernel_gp"
        / f"kernel-{kernel}_ip-{num_inducing_points}-dh-{dim_hidden}_do-{dim_output}_dp-{depth}_ns-{negative_slope}_dr-{dropout_rate}_sn-{spectral_norm}_lr-{learning_rate}_bs-{batch_size}_ep-{epochs}"
        / f"trial-{trial:03d}"
    )
    trial_dir.mkdir(parents=True, exist_ok=True)
    # Write config for downstream use
    config_path = trial_dir / "config.json"
    with config_path.open(mode="w") as cp:
        json.dump(config, cp)
    # Get the acquisition function
    acquisition_function = acquisitions.FUNCTIONS.get(
        config.get("acquisition_function")
    )
    # Do active learning loop
    step_size = config.get("step_size")
    warm_start_size = config.get("warm_start_size")
    max_acquisitions = config.get("max_acquisitions")
    for i in range(max_acquisitions):
        model = models.DeepKernelGP(
            job_dir=trial_dir / f"acquisition-{i-1:03d}",
            kernel=kernel,
            num_inducing_points=num_inducing_points,
            inducing_point_dataset=ds_active.dataset,
            architecture="resnet",
            dim_input=ds_active.dataset.dim_input,
            dim_hidden=dim_hidden,
            dim_output=dim_output,
            depth=depth,
            negative_slope=negative_slope,
            batch_norm=False,
            spectral_norm=spectral_norm,
            dropout_rate=dropout_rate,
            weight_decay=(0.5 * (1 - config.get("dropout_rate")))
            / len(ds_active.dataset),
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=5,
            num_workers=0,
            seed=config.get("seed"),
        )
        model.load(load_best=True)

        mu_0, mu_1 = model.predict_mus(
            ds_active.dataset, batch_size=len(ds_active.dataset)
        )

        bald = (
            acquisition_function(mu_0=mu_0, mu_1=mu_1, t=ds_active.dataset.t, pt=None)
            ** 1.5
        )[ds_active.pool_dataset.indices]

        p = bald / bald.sum()
        idx = np.random.choice(
            range(len(p)),
            replace=False,
            p=p,
            size=warm_start_size if i == 0 else step_size,
        )
        ds_active.acquire(idx)

        acquisition_dir = trial_dir / f"acquisition-{i:03d}"
        model = models.DeepKernelGP(
            job_dir=acquisition_dir,
            kernel=kernel,
            num_inducing_points=num_inducing_points,
            inducing_point_dataset=ds_active.get_training_dataset(),
            architecture="resnet",
            dim_input=ds_active.dataset.dim_input,
            dim_hidden=dim_hidden,
            dim_output=dim_output,
            depth=depth,
            negative_slope=negative_slope,
            batch_norm=False,
            spectral_norm=spectral_norm,
            dropout_rate=dropout_rate,
            weight_decay=(0.5 * (1 - config.get("dropout_rate")))
            / len(ds_active.dataset),
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=5,
            num_workers=0,
            seed=config.get("seed"),
        )
        _ = model.fit(ds_active.get_training_dataset(), ds_valid)
        acquired_path = acquisition_dir / "aquired.json"
        with acquired_path.open(mode="w") as ap:
            json.dump({"aquired_indices": [int(a) for a in idx]}, ap)
