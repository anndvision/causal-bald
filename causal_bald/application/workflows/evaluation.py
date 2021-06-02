import torch
import json
import numpy as np

from scipy import stats

from torch.utils import data

from causal_bald.library import models
from causal_bald.library import datasets
from causal_bald.library import plotting
from causal_bald.library import acquisitions

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


def plot_errorbars(experiment_dir, output_dir):
    for trial_dir in sorted(experiment_dir.iterdir()):
        if "trial-" not in str(trial_dir):
            continue
        config_path = trial_dir / "config.json"
        with config_path.open(mode="r") as cp:
            config = json.load(cp)

        dataset_name = config.get("dataset_name")

        ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))
        model_dir = trial_dir / "checkpoints"
        mu_0, mu_1 = predict_due(ds=ds_test, job_dir=model_dir, config=config)
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


def plot_convergence(experiment_dir, output_dir, methods):
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
    _ = plt.savefig("convergence.png")


def plot_distribution(experiment_dir, output_dir, acquisition_step):
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
        mu_0, mu_1 = predict_due(ds=ds, job_dir=acquisition_dir, config=config)
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
    plot_args["file_path"] = experiment_dir / "distribution.png"
    plotting.acquisition_clean(**plot_args)


def pehe(experiment_dir, output_dir):
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
                mu_0, mu_1 = predict_due(
                    ds=ds_test, job_dir=acquisition_dir, config=config
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


def predict_due(ds, job_dir, config):
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
        inducing_point_dataset=ds,
        architecture="resnet",
        dim_input=ds.dim_input,
        dim_hidden=dim_hidden,
        dim_output=dim_output,
        depth=depth,
        negative_slope=negative_slope,
        batch_norm=False,
        spectral_norm=spectral_norm,
        dropout_rate=dropout_rate,
        weight_decay=(0.5 * (1 - config.get("dropout_rate"))) / len(ds),
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=20,
        num_workers=0,
        seed=config.get("seed"),
    )
    model.load()
    # Evaluate PEHE on test set
    return model.predict_mus(ds, batch_size=len(ds))


# def evaluate(experiment_dir, output_dir):
#     results = {
#         "rmse": [],
#         "rmse_random": [],
#         "rmse_uncertainty": [],
#     }
#     for i, trial_dir in enumerate(sorted(experiment_dir.iterdir())):
#         result_path = config_path = trial_dir / "results.json"
#         if not (result_path).exists():
#             config_path = trial_dir / "config.json"
#             with config_path.open(mode="r") as cp:
#                 config = json.load(cp)

#             dataset_name = config.get("dataset_name")
#             if dataset_name == "ihdp-cov":
#                 deferral_factor = 2
#             else:
#                 deferral_factor = 10
#             ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))

#             if "kernel" in config.keys():
#                 tau_pred = predict_due(config=config, trial_dir=trial_dir, ds=ds_test)
#             else:
#                 tau_pred = predict_ensemble(
#                     config=config, trial_dir=trial_dir, ds=ds_test
#                 )

#             tau_true = ds_test.mu1 - ds_test.mu0

#             n_examples = len(ds_test)
#             n_retained = n_examples - (n_examples // deferral_factor)
#             idx_random = np.random.choice(np.arange(n_examples), n_retained)
#             idx_sorted = np.argsort(tau_pred.var(0))[:n_retained]

#             rmse = rmse_fn(tau_pred.mean(0), tau_true)
#             rmse_rand = rmse_fn(tau_pred.mean(0)[idx_random], tau_true[idx_random])
#             rmse_unct = rmse_fn(tau_pred.mean(0)[idx_sorted], tau_true[idx_sorted])

#             result = {
#                 "rmse": float(rmse),
#                 "rmse_random": float(rmse_rand),
#                 "rmse_uncertainty": float(rmse_unct),
#             }
#             with result_path.open(mode="w") as rp:
#                 json.dump(result, rp)

#             figure_path = output_dir / f"trial-{i:03d}" / "errorbars.png"
#             plotting.errorbar(
#                 x=tau_true,
#                 y=tau_pred.mean(0),
#                 y_err=2 * tau_pred.std(0),
#                 x_label=r"$\tau(\mathbf{x})$",
#                 y_label=r"$\widehat{\tau}(\mathbf{x})$",
#                 x_pad=-20,
#                 y_pad=-45,
#                 file_path=figure_path,
#             )
#             idx_x = np.argsort(ds_test.x.ravel())
#             _ = plotting.functions(
#                 x=ds_test.x.ravel(),
#                 t=ds_test.t,
#                 domain=ds_test.x[idx_x],
#                 tau_true=ds_test.tau[idx_x],
#                 tau_mean=tau_pred[:10, idx_x],
#                 legend_title=None,
#                 legend_loc=(0.07, 0.62),
#                 file_path=output_dir / f"trial-{i:03d}" / "functions.png",
#             )
#         else:
#             with result_path.open(mode="r") as rp:
#                 result = json.load(rp)
#         for k, v in result.items():
#             results[k].append(v)
#     for k, v in results.items():
#         v = np.asarray(v)
#         print(f"{k}: mean-{v.mean():.03f} sem-{stats.sem(v):.03f}")
# if False:
#     # Predict train set
#     mu_0, mu_1 = model.predict_mus(ds_train, batch_size=len(ds_train))
#     tau_pred = (mu_1 - mu_0) * ds_train.y_std[0]
#     results[trial_key]["tau_pred"].append(tau_pred)
#     results[trial_key]["acquired_indices"].append(acquired_indices)

#     scores = acquisition_function(
#         mu_0=mu_0, mu_1=mu_1, t=ds_train.t, pt=None
#     ) ** (1 / temperature)

#     scores[acquired_indices] = ds_train.t[acquired_indices] - 2.0

#     plotting.acquisition_clean(
#         ds_pool=ds_train,
#         acquired_indices=acquired_indices,
#         tau_mean=tau_pred,
#         legend_title=f"Acquired: {num_acquired}",
#         file_path=trial_dir / f"distribution_{i:03d}.png",
#     )
#     _ = plotting.acquisition(
#         x=ds_train.x.ravel(),
#         t=ds_train.t,
#         tau_true=ds_train.tau,
#         bald=scores,
#         legend_loc=(0.1, 0.6),
#         file_path=trial_dir / f"prop_{i:03d}.png",
#     )
#     idx_x = np.argsort(ds_train.x.ravel())
#     _ = plotting.functions(
#         x=ds_train.x.ravel(),
#         t=ds_train.t,
#         domain=ds_train.x[idx_x],
#         tau_true=ds_train.tau[idx_x],
#         tau_mean=tau_pred[:20, idx_x],
#         legend_title=f"Acquired: {num_acquired}",
#         legend_loc=(0.07, 0.62),
#         file_path=trial_dir / f"ensemble_{i:03d}.png",
#     )


def rmse_fn(y_pred, y):
    return np.sqrt(np.mean(np.square(y_pred - y)))
