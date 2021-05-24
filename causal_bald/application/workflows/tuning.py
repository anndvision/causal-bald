from ray import tune
from ray.tune import schedulers
from ray.tune.suggest import hyperopt

from causal_bald.library import models
from causal_bald.library import datasets


def tune_deep_kernel_gp(config):
    space = {
        "kernel": tune.choice(["RBF", "Matern12", "Matern32", "Matern52"]),
        "num_inducing_points": tune.choice([20, 50, 100, 200]),
        "dim_hidden": tune.choice([100, 200]),
        "depth": tune.choice([3, 4]),
        "negative_slope": tune.choice([-1.0, 0.0, 0.1, 0.2]),
        "dropout_rate": tune.choice([0.05, 0.1, 0.2, 0.3]),
        "spectral_norm": tune.choice([0.95, 1.5, 3.0, 6.0, 12.0, 24.0]),
        "learning_rate": tune.choice([1e-3]),
        "batch_size": tune.choice([32, 64, 100, 200]),
    }

    def func(config):
        dataset_name = config.get("dataset_name")
        ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
        ds_valid = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))

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
        outcome_model = models.DeepKernelGP(
            job_dir=None,
            kernel=kernel,
            num_inducing_points=num_inducing_points,
            inducing_point_dataset=ds_train,
            architecture="resnet",
            dim_input=ds_train.dim_input,
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
            patience=4,
            num_workers=0,
            seed=config.get("seed"),
        )
        _ = outcome_model.fit(ds_train, ds_valid)

    algorithm = hyperopt.HyperOptSearch(
        space,
        metric="mean_loss",
        mode="min",
        n_initial_points=100,
    )
    scheduler = schedulers.AsyncHyperBandScheduler(
        grace_period=100, max_t=config.get("epochs")
    )
    analysis = tune.run(
        run_or_experiment=func,
        metric="mean_loss",
        mode="min",
        name="hyperopt_deep_kernel_gp",
        resources_per_trial={
            "cpu": config.get("cpu_per_trial"),
            "gpu": config.get("gpu_per_trial"),
        },
        num_samples=config.get("max_samples"),
        search_alg=algorithm,
        scheduler=scheduler,
        local_dir=config.get("experiment_dir"),
        config=config,
    )
    print("Best hyperparameters found were: ", analysis.best_config)


def tune_tarnet(config):
    space = {
        "dim_hidden": tune.choice([200]),
        "depth": tune.choice([3]),
        "negative_slope": tune.choice([-1.0]),
        "dropout_rate": tune.choice([0.05, 0.1, 0.2, 0.3]),
        "spectral_norm": tune.choice([0.95, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0]),
        "learning_rate": tune.choice([1e-3]),
        "batch_size": tune.choice([32, 64, 100, 200]),
    }

    def func(config):
        dataset_name = config.get("dataset_name")
        ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
        ds_valid = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))

        dim_hidden = config.get("dim_hidden")
        dim_output = config.get("dim_output")
        depth = config.get("depth")
        negative_slope = config.get("negative_slope")
        dropout_rate = config.get("dropout_rate")
        spectral_norm = config.get("spectral_norm")
        learning_rate = config.get("learning_rate")
        batch_size = config.get("batch_size")
        epochs = config.get("epochs")

        outcome_model = models.TARNet(
            job_dir=None,
            architecture="resnet",
            dim_input=ds_train.dim_input,
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
            seed=config.get("seed"),
        )
        _ = outcome_model.fit(ds_train, ds_valid)

    algorithm = hyperopt.HyperOptSearch(
        space,
        metric="mean_loss",
        mode="min",
        n_initial_points=20,
    )
    scheduler = schedulers.AsyncHyperBandScheduler(
        grace_period=20, max_t=config.get("epochs")
    )
    analysis = tune.run(
        run_or_experiment=func,
        metric="mean_loss",
        mode="min",
        name="hyperopt_tarnet",
        resources_per_trial={
            "cpu": config.get("cpu_per_trial"),
            "gpu": config.get("gpu_per_trial"),
        },
        num_samples=config.get("max_samples"),
        search_alg=algorithm,
        scheduler=scheduler,
        local_dir=config.get("experiment_dir"),
        config=config,
    )
    print("Best hyperparameters found were: ", analysis.best_config)
