import json
import torch
import numpy as np

from scipy import stats

from torch.utils import data

from causal_bald.library import datasets
from causal_bald.library import plotting
from causal_bald.library import acquisitions

from causal_bald.application.workflows import utils

import seaborn as sns
import matplotlib.pyplot as plt


sns.set(style="whitegrid", palette="colorblind")
params = {
    "figure.constrained_layout.use": True,
    "axes.labelsize": 14,
    # "xtick.labelsize": 18,
    # "ytick.labelsize": 18,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
    "lines.markersize": 8,
    # "font.size": 24,
}
plt.rcParams.update(params)

styles = {
    "random": ("C5", "--", "X"),
    "mu": ("C9", "-", "x"),
    "tau": ("C3", "-", "+"),
    "rho": ("C2", "-", "|"),
    "mu-pi": ("C4", "-", "^"),
    "mu-rho": ("C0", "-", "o"),
    "pi": ("C8", "-", "*"),
    "sundin": ("C1", "-", "s"),
}


def plot_errorbars(experiment_dir):
    for trial_dir in sorted(experiment_dir.iterdir()):
        if "trial-" not in str(trial_dir):
            continue
        config_path = trial_dir / "config.json"
        with config_path.open(mode="r") as cp:
            config = json.load(cp)

        dataset_name = config.get("dataset_name")

        ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))
        model_dir = trial_dir / "checkpoints"
        mu_0, mu_1 = utils.PREDICT_FUNCTIONS[config.get("model_name")](
            dataset=ds_test, job_dir=model_dir, config=config
        )
        tau_pred = (mu_1 - mu_0) * ds_test.y_std[0]
        tau_true = ds_test.mu1 - ds_test.mu0
        plot_path = trial_dir / "scatter.png"
        plotting.errorbar(
            x=tau_true,
            y=tau_pred.mean(0),
            y_err=2 * tau_pred.std(0),
            x_label="True CATE",
            y_label="Predicted CATE",
            file_path=plot_path,
        )


def plot_convergence(experiment_dir, methods):
    _ = plt.figure(figsize=(682 / 72, 512 / 72), dpi=300)
    for acquisition_function in methods:
        pehe_path = experiment_dir / f"{acquisition_function}_pehe.json"
        with pehe_path.open(mode="r") as pp:
            pehe = json.load(pp)
            del pehe["acquisition_function"]
        pehes = []
        num_acquired = []
        for trial, results in pehe.items():
            pehes.append(results["value"])
            num_acquired.append(results["num_acquired"])
        pehes = np.asarray(pehes)
        mean_pehe = pehes.mean(0)
        sem_pehe = stats.sem(pehes, axis=0)
        x = np.asarray(num_acquired[0])
        _ = plt.plot(
            x,
            mean_pehe,
            color=styles[acquisition_function][0],
            marker=styles[acquisition_function][2],
            label=acquisition_function,
        )
        _ = plt.fill_between(
            x=x,
            y1=mean_pehe - sem_pehe,
            y2=mean_pehe + sem_pehe,
            color=styles[acquisition_function][0],
            alpha=0.3,
        )
        _ = plt.legend(loc=None, title=None)
    _ = plt.savefig(experiment_dir / "convergence.png", dpi=150)


def plot_evolution(experiment_dir, trial, num_steps):
    plot_args = {
        "x_pool": None,
        "t_pool": None,
        "x_acquired": None,
        "t_acquired": None,
        "tau_true": None,
        "tau_pred": None,
    }
    trial_dir = experiment_dir / f"trial-{trial:03d}"
    config_path = trial_dir / "config.json"
    with config_path.open(mode="r") as cp:
        config = json.load(cp)

    dataset_name = config.get("dataset_name")

    ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
    plot_args["x_pool"] = ds_train.x.ravel()
    plot_args["t_pool"] = ds_train.t
    plot_args["tau_true"] = ds_train.mu1 - ds_train.mu0
    out_dir = trial_dir / "evolution"
    out_dir.mkdir(parents=True, exist_ok=True)
    for acquisition_step in range(num_steps):
        # Load acquired indices
        acquired_path = (
            trial_dir / f"acquisition-{acquisition_step:03d}" / "aquired.json"
        )
        if not acquired_path.exists():
            break
        with acquired_path.open(mode="r") as ap:
            aquired_dict = json.load(ap)
        acquired_indices = aquired_dict["aquired_indices"]
        plot_args["x_acquired"] = ds_train.x[acquired_indices].ravel()
        plot_args["t_acquired"] = ds_train.t[acquired_indices]
        acquisition_dir = trial_dir / f"acquisition-{acquisition_step:03d}"
        domain = torch.arange(-3.5, 3.5, 0.01, dtype=torch.float32).unsqueeze(-1)
        ds = data.TensorDataset(torch.cat([domain, domain], -1), domain)
        ds.dim_input = 1
        mu_0, mu_1 = utils.PREDICT_FUNCTIONS[
            config.get("model_name", "deep_kernel_gp")
        ](dataset=ds, job_dir=acquisition_dir, config=config)
        plot_args["tau_pred"] = (mu_1 - mu_0) * ds_train.y_std[0]
        num_acquired = len(plot_args["t_acquired"])
        plot_args["domain"] = np.arange(-3.5, 3.5, 0.01)
        plot_args["legend_title"] = f"Acquired: {num_acquired:03d}"
        plot_args["file_path"] = out_dir / f"distribution_{acquisition_step:02d}.png"
        plotting.acquisition_hist(**plot_args)


def plot_distribution(experiment_dir, acquisition_step):
    plot_args = {
        "x_pool": [],
        "t_pool": [],
        "x_acquired": [],
        "t_acquired": [],
        "tau_true": [],
        "tau_pred": [],
    }
    trial = 0
    for trial_dir in sorted(experiment_dir.iterdir()):
        if "trial-" not in str(trial_dir):
            continue
        config_path = trial_dir / "config.json"
        with config_path.open(mode="r") as cp:
            config = json.load(cp)

        dataset_name = config.get("dataset_name")

        ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
        plot_args["x_pool"].append(ds_train.x.ravel())
        plot_args["t_pool"].append(ds_train.t)
        plot_args["tau_true"].append(ds_train.mu1 - ds_train.mu0)

        # Load acquired indices
        acquired_path = (
            trial_dir / f"acquisition-{acquisition_step:03d}" / "aquired.json"
        )
        if not acquired_path.exists():
            break
        with acquired_path.open(mode="r") as ap:
            aquired_dict = json.load(ap)
        acquired_indices = aquired_dict["aquired_indices"]
        plot_args["x_acquired"].append(ds_train.x[acquired_indices].ravel())
        plot_args["t_acquired"].append(ds_train.t[acquired_indices])
        acquisition_dir = trial_dir / f"acquisition-{acquisition_step:03d}"
        domain = torch.arange(-3.5, 3.5, 0.01, dtype=torch.float32).unsqueeze(-1)
        ds = data.TensorDataset(torch.cat([domain, domain], -1), domain)
        ds.dim_input = 1
        mu_0, mu_1 = utils.PREDICT_FUNCTIONS[
            config.get("model_name", "deep_kernel_gp")
        ](dataset=ds, job_dir=acquisition_dir, config=config)
        plot_args["tau_pred"].append((mu_1 - mu_0) * ds_train.y_std[0])
        trial += 1
    for k, v in plot_args.items():
        if k == "tau_pred":
            plot_args[k] = np.vstack(v)
        else:
            plot_args[k] = np.hstack(v)
    num_acquired = len(plot_args["t_acquired"]) // (trial)
    plot_args["domain"] = np.arange(-3.5, 3.5, 0.01)
    plot_args["legend_title"] = f"Acquired: {num_acquired}"
    plot_args["file_path"] = experiment_dir / f"distribution_{acquisition_step:02d}.png"
    plotting.acquisition_clean(**plot_args)


def plot_dataset(config, output_dir):
    dataset_name = config.get("dataset_name")
    ds = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
    if dataset_name == "cmnist":
        plotting.mnist(
            ds=ds, file_path=output_dir / "cmnist_dataset.png",
        )
    elif dataset_name == "synthetic":
        plotting.dataset(
            ds=ds, file_path=output_dir / "synthetic_dataset.png",
        )
    else:
        raise NotImplementedError(f"{dataset_name} dataset not supported")


def pehe(experiment_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    pehe = {}
    trial = 0
    for trial_dir in sorted(experiment_dir.iterdir()):
        if "trial-" not in str(trial_dir):
            continue
        trial_key = f"trial-{trial:03d}"
        config_path = trial_dir / "config.json"
        with config_path.open(mode="r") as cp:
            config = json.load(cp)

        dataset_name = config.get("dataset_name")

        ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))

        acquisition_function = acquisitions.FUNCTIONS.get(
            config.get("acquisition_function")
        )

        trial_pehe_path = trial_dir / "pehe.json"
        if not trial_pehe_path.exists():
            trial_pehe = {"value": [], "num_acquired": []}
            max_acquisitions = config.get("max_acquisitions")
            for i in range(max_acquisitions):
                # Load acquired indices
                acquired_path = trial_dir / f"acquisition-{i:03d}" / "aquired.json"
                if not acquired_path.exists():
                    break
                with acquired_path.open(mode="r") as ap:
                    aquired_dict = json.load(ap)
                acquired_indices = aquired_dict["aquired_indices"]
                num_acquired = len(acquired_indices)
                acquisition_dir = trial_dir / f"acquisition-{i:03d}"
                mu_0, mu_1 = utils.PREDICT_FUNCTIONS[config.get("model_name")](
                    dataset=ds_test, job_dir=acquisition_dir, config=config
                )
                tau_pred = (mu_1 - mu_0) * ds_test.y_std[0]
                tau_true = ds_test.mu1 - ds_test.mu0
                trial_pehe["value"].append(float(rmse_fn(tau_pred.mean(0), tau_true)))
                trial_pehe["num_acquired"].append(num_acquired)
            trial_pehe_path.write_text(json.dumps(trial_pehe, indent=4, sort_keys=True))
        else:
            trial_pehe = json.loads(trial_pehe_path.read_text())
        pehe[trial_key] = trial_pehe
        trial += 1
    pehes = []
    num_acquired = []
    for trial, results in pehe.items():
        pehes.append(results["value"])
        num_acquired.append(results["num_acquired"])
    pehes = np.asarray(pehes)
    print(num_acquired[0])
    print(list(pehes.mean(0)))
    print(list(stats.sem(pehes, axis=0)))
    acquisition_function = config.get("acquisition_function")
    pehe["acquisition_function"] = acquisition_function
    pehe_path = output_dir / f"{acquisition_function}_pehe.json"
    pehe_path.write_text(json.dumps(pehe, indent=4, sort_keys=True))


def rmse_fn(y_pred, y):
    return np.sqrt(np.mean(np.square(y_pred - y)))
