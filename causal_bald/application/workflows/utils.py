import numpy as np

from pathlib import Path

from causal_bald.library import models


def directory_ensemble(base_dir, config):
    # Get model parameters from config
    dim_hidden = config.get("dim_hidden")
    dim_output = config.get("dim_output")
    depth = config.get("depth")
    negative_slope = config.get("negative_slope")
    dropout_rate = config.get("dropout_rate")
    spectral_norm = config.get("spectral_norm")
    learning_rate = config.get("learning_rate")
    batch_size = config.get("batch_size")
    epochs = config.get("epochs")
    return (
        Path(base_dir)
        / "ensemble"
        / f"dh-{dim_hidden}_do-{dim_output}_dp-{depth}_ns-{negative_slope}_dr-{dropout_rate}_sn-{spectral_norm}_lr-{learning_rate}_bs-{batch_size}_ep-{epochs}"
    )


def directory_deep_kernel_gp(base_dir, config):
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
    return (
        Path(base_dir)
        / "deep_kernel_gp"
        / f"kernel-{kernel}_ip-{num_inducing_points}-dh-{dim_hidden}_do-{dim_output}_dp-{depth}_ns-{negative_slope}_dr-{dropout_rate}_sn-{spectral_norm}_lr-{learning_rate}_bs-{batch_size}_ep-{epochs}"
    )


DIRECTORIES = {
    "deep_kernel_gp": directory_deep_kernel_gp,
    "ensemble": directory_ensemble,
}


def train_ensemble(ds_train, ds_valid, job_dir, config, dim_input):
    ensemble_size = config.get("ensemble_size")
    dim_hidden = config.get("dim_hidden")
    dim_output = config.get("dim_output")
    depth = config.get("depth")
    negative_slope = config.get("negative_slope")
    dropout_rate = config.get("dropout_rate")
    spectral_norm = config.get("spectral_norm")
    learning_rate = config.get("learning_rate")
    batch_size = config.get("batch_size")
    epochs = config.get("epochs")
    for ensemble_id in range(ensemble_size):
        if job_dir is None:
            out_dir = job_dir
        else:
            out_dir = job_dir / f"model-{ensemble_id}"
            if (out_dir / "best_checkpoint.pt").exists():
                continue
        model = models.TARNet(
            job_dir=out_dir,
            architecture="resnet",
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            dim_output=dim_output,
            depth=depth,
            negative_slope=negative_slope,
            batch_norm=False,
            spectral_norm=spectral_norm,
            dropout_rate=dropout_rate,
            weight_decay=(0.5 * (1 - dropout_rate)) / len(ds_train),
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=20,
            num_workers=0,
            seed=None,
        )
        _ = model.fit(ds_train, ds_valid)


def train_deep_kernel_gp(ds_train, ds_valid, job_dir, config, dim_input):
    if not (job_dir / "best_checkpoint.pt").exists():
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
        model = models.DeepKernelGP(
            job_dir=job_dir,
            kernel=kernel,
            num_inducing_points=num_inducing_points,
            inducing_point_dataset=ds_train,
            architecture="resnet",
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            dim_output=dim_output,
            depth=depth,
            negative_slope=negative_slope,
            batch_norm=False,
            spectral_norm=spectral_norm,
            dropout_rate=dropout_rate,
            weight_decay=(0.5 * (1 - config.get("dropout_rate"))) / len(ds_train),
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=5,
            num_workers=0,
            seed=config.get("seed"),
        )
        _ = model.fit(ds_train, ds_valid)


TRAIN_FUNCTIONS = {
    "deep_kernel_gp": train_deep_kernel_gp,
    "ensemble": train_ensemble,
}


def predict_ensemble(dataset, job_dir, config):
    ensemble_size = config.get("ensemble_size")
    dim_hidden = config.get("dim_hidden")
    dim_output = config.get("dim_output")
    depth = config.get("depth")
    negative_slope = config.get("negative_slope")
    dropout_rate = config.get("dropout_rate")
    spectral_norm = config.get("spectral_norm")
    learning_rate = config.get("learning_rate")
    batch_size = config.get("batch_size")
    epochs = config.get("epochs")
    mu_0 = []
    mu_1 = []
    for ensemble_id in range(ensemble_size):
        out_dir = job_dir / f"model-{ensemble_id}"
        model = models.TARNet(
            job_dir=out_dir,
            architecture="resnet",
            dim_input=dataset.dim_input,
            dim_hidden=dim_hidden,
            dim_output=dim_output,
            depth=depth,
            negative_slope=negative_slope,
            batch_norm=False,
            spectral_norm=spectral_norm,
            dropout_rate=dropout_rate,
            weight_decay=(0.5 * (1 - dropout_rate)) / len(dataset),
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=10,
            num_workers=0,
            seed=None,
        )
        model.load()
        mus = model.predict_mus(dataset)
        mu_0.append(mus[0])
        mu_1.append(mus[1])
    return np.asarray(mu_0), np.asarray(mu_1)


def predict_deep_kernel_gp(dataset, job_dir, config):
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
    model = models.DeepKernelGP(
        job_dir=job_dir,
        kernel=kernel,
        num_inducing_points=num_inducing_points,
        inducing_point_dataset=dataset,
        architecture="resnet",
        dim_input=dataset.dim_input,
        dim_hidden=dim_hidden,
        dim_output=dim_output,
        depth=depth,
        negative_slope=negative_slope,
        batch_norm=False,
        spectral_norm=spectral_norm,
        dropout_rate=dropout_rate,
        weight_decay=(0.5 * (1 - config.get("dropout_rate"))) / len(dataset),
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=5,
        num_workers=0,
        seed=config.get("seed"),
    )
    model.load()
    return model.predict_mus(dataset)


PREDICT_FUNCTIONS = {
    "deep_kernel_gp": predict_deep_kernel_gp,
    "ensemble": predict_ensemble,
}
