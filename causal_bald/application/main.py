import ray
import click

from torch import cuda

from pathlib import Path

from causal_bald.application import workflows


@click.group(chain=True)
@click.pass_context
def cli(context):
    context.obj = {"n_gpu": cuda.device_count()}


@cli.command("tune")
@click.option(
    "--job-dir",
    type=str,
    required=True,
    help="location for writing checkpoints and results",
)
@click.option(
    "--max-samples",
    default=100,
    type=int,
    help="maximum number of search space samples, default=100",
)
@click.option(
    "--gpu-per-trial",
    default=0.0,
    type=float,
    help="number of gpus for each trial, default=0",
)
@click.option(
    "--cpu-per-trial",
    default=1.0,
    type=float,
    help="number of cpus for each trial, default=1",
)
@click.option(
    "--seed",
    default=1331,
    type=int,
    help="random number generator seed, default=1331",
)
@click.pass_context
def tune(
    context,
    job_dir,
    max_samples,
    gpu_per_trial,
    cpu_per_trial,
    seed,
):
    ray.init(
        num_gpus=context.obj["n_gpu"],
        dashboard_host="127.0.0.1",
        ignore_reinit_error=True,
    )
    gpu_per_trial = 0 if context.obj["n_gpu"] == 0 else gpu_per_trial
    job_dir = Path(job_dir) / "tuning"
    context.obj.update(
        {
            "job_dir": str(job_dir),
            "max_samples": max_samples,
            "gpu_per_trial": gpu_per_trial,
            "cpu_per_trial": cpu_per_trial,
            "seed": seed,
            "mode": "tune",
        }
    )


@cli.command("train")
@click.option(
    "--job-dir",
    type=str,
    required=True,
    help="location for writing checkpoints and results",
)
@click.option("--num-trials", default=1, type=int, help="number of trials, default=1")
@click.option(
    "--gpu-per-trial",
    default=0.0,
    type=float,
    help="number of gpus for each trial, default=0",
)
@click.option(
    "--cpu-per-trial",
    default=1.0,
    type=float,
    help="number of cpus for each trial, default=1",
)
@click.option("--verbose", default=False, type=bool, help="verbosity default=False")
@click.option(
    "--seed",
    default=1331,
    type=int,
    help="random number generator seed, default=1331",
)
@click.pass_context
def train(
    context,
    job_dir,
    num_trials,
    gpu_per_trial,
    cpu_per_trial,
    verbose,
    seed,
):
    ray.init(
        num_gpus=context.obj["n_gpu"],
        dashboard_host="127.0.0.1",
        ignore_reinit_error=True,
    )
    gpu_per_trial = 0 if context.obj["n_gpu"] == 0 else gpu_per_trial
    job_dir = Path(job_dir) / "training"
    context.obj.update(
        {
            "job_dir": str(job_dir),
            "num_trials": num_trials,
            "gpu_per_trial": gpu_per_trial,
            "cpu_per_trial": cpu_per_trial,
            "verbose": verbose,
            "seed": seed,
            "mode": "train",
        }
    )


@cli.command("active-learning")
@click.option(
    "--job-dir",
    type=str,
    required=True,
    help="location for writing checkpoints and results",
)
@click.option("--num-trials", default=1, type=int, help="number of trials, default=1")
@click.option(
    "--step-size",
    default=10,
    type=int,
    help="number of data points to acquire at each step, default=10",
)
@click.option(
    "--warm-start-size",
    default=50,
    type=int,
    help="number of data points to acquire at start, default=50",
)
@click.option(
    "--max-acquisitions",
    default=100,
    type=int,
    help="number of acquisition steps, default=100",
)
@click.option(
    "--acquisition-function",
    default="mu-rho",
    type=str,
    help="acquistion function, default=mu-rho",
)
@click.option(
    "--gpu-per-trial",
    default=0.0,
    type=float,
    help="number of gpus for each trial, default=0",
)
@click.option(
    "--cpu-per-trial",
    default=1.0,
    type=float,
    help="number of cpus for each trial, default=1",
)
@click.option("--verbose", default=False, type=bool, help="verbosity default=False")
@click.option(
    "--seed",
    default=1331,
    type=int,
    help="random number generator seed, default=1331",
)
@click.pass_context
def active_learning(
    context,
    job_dir,
    num_trials,
    step_size,
    warm_start_size,
    max_acquisitions,
    acquisition_function,
    gpu_per_trial,
    cpu_per_trial,
    verbose,
    seed,
):
    ray.init(
        num_gpus=context.obj["n_gpu"],
        dashboard_host="127.0.0.1",
        ignore_reinit_error=True,
    )
    gpu_per_trial = 0 if context.obj["n_gpu"] == 0 else gpu_per_trial
    job_dir = (
        Path(job_dir)
        / "active_learning"
        / f"ss-{step_size}_ws-{warm_start_size}_ma-{max_acquisitions}_af-{acquisition_function}"
    )
    context.obj.update(
        {
            "job_dir": str(job_dir),
            "num_trials": num_trials,
            "step_size": step_size,
            "warm_start_size": warm_start_size,
            "max_acquisitions": max_acquisitions,
            "acquisition_function": acquisition_function,
            "gpu_per_trial": gpu_per_trial,
            "cpu_per_trial": cpu_per_trial,
            "verbose": verbose,
            "seed": seed,
            "mode": "active",
        }
    )


@cli.command("evaluate")
@click.option(
    "--experiment-dir",
    type=str,
    required=True,
    help="location for reading checkpoints",
)
@click.option(
    "--output-dir",
    type=str,
    required=False,
    default=None,
    help="location for writing results",
)
@click.pass_context
def evaluate(
    context,
    experiment_dir,
    output_dir,
):
    output_dir = experiment_dir if output_dir is None else output_dir
    context.obj.update(
        {
            "experiment_dir": experiment_dir,
            "output_dir": output_dir,
        }
    )
    workflows.evaluation.evaluate(
        experiment_dir=Path(experiment_dir),
        output_dir=Path(output_dir),
    )


@cli.command("ihdp")
@click.pass_context
@click.option(
    "--root",
    type=str,
    required=True,
    help="location of dataset",
)
def ihdp(
    context,
    root,
):
    job_dir = Path(context.obj.get("job_dir"))
    dataset_name = "ihdp"
    experiment_dir = job_dir / dataset_name
    context.obj.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "ds_train": {
                "root": root,
                "split": "train",
                "mode": "mu",
                "seed": context.obj.get("seed"),
            },
            "ds_valid": {
                "root": root,
                "split": "valid",
                "mode": "mu",
                "seed": context.obj.get("seed"),
            },
            "ds_test": {
                "root": root,
                "split": "test",
                "mode": "mu",
                "seed": context.obj.get("seed"),
            },
        }
    )


@cli.command("ihdp-cov")
@click.pass_context
@click.option(
    "--root",
    type=str,
    required=True,
    help="location of dataset",
)
def ihdp_cov(
    context,
    root,
):
    job_dir = Path(context.obj.get("job_dir"))
    dataset_name = "ihdp-cov"
    experiment_dir = job_dir / dataset_name
    context.obj.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "ds_train": {
                "root": root,
                "split": "train",
                "mode": "mu",
                "seed": context.obj.get("seed"),
            },
            "ds_valid": {
                "root": root,
                "split": "valid",
                "mode": "mu",
                "seed": context.obj.get("seed"),
            },
            "ds_test": {
                "root": root,
                "split": "test",
                "mode": "mu",
                "seed": context.obj.get("seed"),
            },
        }
    )


@cli.command("synthetic")
@click.pass_context
@click.option(
    "--num-examples",
    default=10000,
    type=int,
    help="number of training examples, defaul=1000",
)
@click.option(
    "--beta",
    default=2.0,
    type=float,
    help="Coefficient for x effect on t, default=2.0",
)
@click.option(
    "--bimodal",
    default=False,
    type=bool,
    help="x sampled from bimodal distribution, default=False",
)
@click.option(
    "--sigma",
    default=1.0,
    type=float,
    help="standard deviation of random noise in y, default=1.0",
)
@click.option(
    "--domain-limit",
    default=2.5,
    type=float,
    help="Domain of x is [-domain_limit, domain_limit], default=2.5",
)
def synthetic(
    context,
    num_examples,
    beta,
    bimodal,
    sigma,
    domain_limit,
):
    job_dir = Path(context.obj.get("job_dir"))
    dataset_name = "synthetic"
    experiment_dir = (
        job_dir
        / dataset_name
        / f"ne-{num_examples}_be-{beta:.02f}_bi{bimodal}_si-{sigma:.02f}_dl-{domain_limit:.02f}"
    )
    context.obj.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "ds_train": {
                "num_examples": num_examples,
                "mode": "mu",
                "beta": beta,
                "bimodal": bimodal,
                "sigma_y": sigma,
                "seed": context.obj.get("seed"),
            },
            "ds_valid": {
                "num_examples": num_examples // 10,
                "mode": "mu",
                "beta": beta,
                "bimodal": bimodal,
                "sigma_y": sigma,
                "seed": context.obj.get("seed") + 1,
            },
            "ds_test": {
                "num_examples": min(num_examples, 2000),
                "mode": "mu",
                "beta": beta,
                "bimodal": bimodal,
                "sigma_y": sigma,
                "seed": context.obj.get("seed") + 2,
            },
        }
    )


@cli.command("ensemble")
@click.pass_context
@click.option("--dim-hidden", default=200, type=int, help="num neurons")
@click.option("--dim-output", default=2, type=int, help="output dimensionality")
@click.option("--depth", default=3, type=int, help="depth of feature extractor")
@click.option(
    "--negative-slope",
    default=-1,
    type=float,
    help="negative slope of leaky relu, default=-1 use elu",
)
@click.option(
    "--dropout-rate", default=0.2, type=float, help="dropout rate, default=0.1"
)
@click.option(
    "--spectral-norm",
    default=0.95,
    type=float,
    help="Spectral normalization coefficient. If 0.0 do not use spectral norm, default=0.0",
)
@click.option(
    "--learning-rate",
    default=1e-3,
    type=float,
    help="learning rate for gradient descent, default=1e-3",
)
@click.option(
    "--batch-size",
    default=100,
    type=int,
    help="number of examples to read during each training step, default=100",
)
@click.option(
    "--epochs", type=int, default=500, help="number of training epochs, default=50"
)
@click.option(
    "--ensemble-size",
    type=int,
    default=5,
    help="number of models in ensemble, default=1",
)
def ensemble(
    context,
    dim_hidden,
    dim_output,
    depth,
    negative_slope,
    dropout_rate,
    spectral_norm,
    learning_rate,
    batch_size,
    epochs,
    ensemble_size,
):
    if context.obj["mode"] == "tune":
        context.obj.update(
            {
                "dim_output": dim_output,
                "epochs": epochs,
                "ensemble_size": ensemble_size,
            }
        )
        workflows.tuning.tune_tarnet(config=context.obj)
    elif context.obj["mode"] == "train":
        context.obj.update(
            {
                "dim_hidden": dim_hidden,
                "depth": depth,
                "dim_output": dim_output,
                "negative_slope": negative_slope,
                "dropout_rate": dropout_rate,
                "spectral_norm": spectral_norm,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "ensemble_size": ensemble_size,
            }
        )

        @ray.remote(
            num_gpus=context.obj.get("gpu_per_trial"),
            num_cpus=context.obj.get("cpu_per_trial"),
        )
        def trainer(**kwargs):
            func = workflows.training.tarnet_trainer(**kwargs)
            return func

        results = []
        for trial in range(context.obj.get("num_trials")):
            for ensemble_id in range(ensemble_size):
                results.append(
                    trainer.remote(
                        config=context.obj,
                        experiment_dir=context.obj.get("experiment_dir"),
                        trial=trial,
                        ensemble_id=ensemble_id,
                    )
                )
        ray.get(results)


@cli.command("deep-kernel-gp")
@click.pass_context
@click.option("--kernel", default="Matern32", type=str, help="GP kernel")
@click.option(
    "--num-inducing-points",
    default=100,
    type=int,
    help="Number of Deep GP Inducing Points",
)
@click.option("--dim-hidden", default=200, type=int, help="num neurons")
@click.option("--dim-output", default=1, type=int, help="output dimensionality")
@click.option("--depth", default=3, type=int, help="depth of feature extractor")
@click.option(
    "--negative-slope",
    default=-1,
    type=float,
    help="negative slope of leaky relu, default=-1 use elu",
)
@click.option(
    "--dropout-rate", default=0.1, type=float, help="dropout rate, default=0.2"
)
@click.option(
    "--spectral-norm",
    default=0.95,
    type=float,
    help="Spectral normalization coefficient. If 0.0 do not use spectral norm, default=0.0",
)
@click.option(
    "--learning-rate",
    default=1e-3,
    type=float,
    help="learning rate for gradient descent, default=1e-3",
)
@click.option(
    "--batch-size",
    default=100,
    type=int,
    help="number of examples to read during each training step, default=100",
)
@click.option(
    "--epochs", type=int, default=1000, help="number of training epochs, default=500"
)
def deep_kernel_gp(
    context,
    kernel,
    num_inducing_points,
    dim_hidden,
    dim_output,
    depth,
    negative_slope,
    dropout_rate,
    spectral_norm,
    learning_rate,
    batch_size,
    epochs,
):
    if context.obj["mode"] == "tune":
        context.obj.update(
            {
                "epochs": epochs,
                "dim_output": dim_output,
            }
        )
        workflows.tuning.tune_deep_kernel_gp(config=context.obj)
    elif context.obj["mode"] == "train":
        context.obj.update(
            {
                "kernel": kernel,
                "num_inducing_points": num_inducing_points,
                "dim_hidden": dim_hidden,
                "depth": depth,
                "dim_output": dim_output,
                "negative_slope": negative_slope,
                "dropout_rate": dropout_rate,
                "spectral_norm": spectral_norm,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
            }
        )

        @ray.remote(
            num_gpus=context.obj.get("gpu_per_trial"),
            num_cpus=context.obj.get("cpu_per_trial"),
        )
        def trainer(**kwargs):
            func = workflows.training.train_deep_kernel_gp(**kwargs)
            return func

        results = []
        for trial in range(context.obj.get("num_trials")):
            results.append(
                trainer.remote(
                    config=context.obj,
                    experiment_dir=context.obj.get("experiment_dir"),
                    trial=trial,
                )
            )
        ray.get(results)
    elif context.obj["mode"] == "active":
        context.obj.update(
            {
                "kernel": kernel,
                "num_inducing_points": num_inducing_points,
                "dim_hidden": dim_hidden,
                "depth": depth,
                "dim_output": dim_output,
                "negative_slope": negative_slope,
                "dropout_rate": dropout_rate,
                "spectral_norm": spectral_norm,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
            }
        )

        @ray.remote(
            num_gpus=context.obj.get("gpu_per_trial"),
            num_cpus=context.obj.get("cpu_per_trial"),
        )
        def active_learner(**kwargs):
            func = workflows.active_learning.active_deep_kernel_gp(**kwargs)
            return func

        results = []
        for trial in range(context.obj.get("num_trials")):
            results.append(
                active_learner.remote(
                    config=context.obj,
                    experiment_dir=context.obj.get("experiment_dir"),
                    trial=trial,
                )
            )
        ray.get(results)


if __name__ == "__main__":
    cli()
