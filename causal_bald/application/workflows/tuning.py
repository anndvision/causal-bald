from ray import tune
from ray.tune import schedulers
from ray.tune.suggest import hyperopt

from causal_bald.library import datasets

from causal_bald.application.workflows import utils


def deep_kernel_gp_tuner(config):
    space = {
        "kernel": tune.choice(["RBF", "Matern12", "Matern32", "Matern52"]),
        "num_inducing_points": tune.choice([20, 50, 100, 200]),
        "dim_hidden": tune.choice([100, 200, 400]),
        "depth": tune.choice([2, 3, 4]),
        "negative_slope": tune.choice([-1.0, 0.0, 0.1, 0.2]),
        "dropout_rate": tune.choice([0.05, 0.1, 0.2, 0.5]),
        "spectral_norm": tune.choice([0.0, 0.95, 1.5, 3.0]),
        "learning_rate": tune.choice([2e-4, 5e-4, 1e-3]),
        "batch_size": tune.choice([32, 64, 100, 200]),
    }

    def func(config):
        dataset_name = config.get("dataset_name")
        ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
        ds_valid = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))

        utils.TRAIN_FUNCTIONS["deep_kernel_gp"](
            ds_train=ds_train,
            ds_valid=ds_valid,
            job_dir=None,
            config=config,
            dim_input=ds_train.dim_input,
        )

    algorithm = hyperopt.HyperOptSearch(
        space, metric="mean_loss", mode="min", n_initial_points=100,
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


def tarnet_tuner(config):
    space = {
        "dim_hidden": tune.choice([100, 200, 400]),
        "depth": tune.choice([2, 3, 4]),
        "negative_slope": tune.choice([-1.0, 0.0, 0.1, 0.2]),
        "dropout_rate": tune.choice([0.05, 0.1, 0.2, 0.5]),
        "spectral_norm": tune.choice([0.0, 0.95, 1.5, 3.0]),
        "learning_rate": tune.choice([2e-4, 5e-4, 1e-3]),
        "batch_size": tune.choice([32, 64, 100, 200]),
    }

    def func(config):
        dataset_name = config.get("dataset_name")
        ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
        ds_valid = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))

        utils.TRAIN_FUNCTIONS["ensemble"](
            ds_train=ds_train,
            ds_valid=ds_valid,
            job_dir=None,
            config=config,
            dim_input=ds_train.dim_input,
        )

    algorithm = hyperopt.HyperOptSearch(
        space, metric="mean_loss", mode="min", n_initial_points=100,
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
