import json
import numpy as np

from scipy import stats

from pathlib import Path

from causal_bald.library import models
from causal_bald.library import datasets
from causal_bald.library import plotting
from causal_bald.library import acquisitions


def evaluate(experiment_dir, output_dir):
    for trial, trial_dir in enumerate(sorted(experiment_dir.iterdir())):
        config_path = trial_dir / "config.json"
        with config_path.open(mode="r") as cp:
            config = json.load(cp)

        dataset_name = config.get("dataset_name")

        config["ds_train"]["seed"] = trial
        config["ds_valid"]["seed"] = trial + 1 if dataset_name == "synthetic" else trial
        config["ds_test"]["seed"] = trial + 2 if dataset_name == "synthetic" else trial

        ds = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))

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

        acquisition_function = acquisitions.FUNCTIONS.get(
            config.get("acquisition_function")
        )

        acquired_indices = []
        step_size = config.get("step_size")
        warm_start_size = config.get("warm_start_size")
        max_acquisitions = config.get("max_acquisitions")
        for i in range(max_acquisitions):
            acquisition_dir = trial_dir / f"acquisition-{i:03d}"
            model = models.DeepKernelGP(
                job_dir=acquisition_dir,
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
            model.load(load_best=True)

            mu_0, mu_1 = model.predict_mus(ds, batch_size=len(ds))
            tau_pred = mu_1 - mu_0

            acquired_path = acquisition_dir / "aquired.json"
            with acquired_path.open(mode="r") as ap:
                aquired_dict = json.load(ap)
            acquired_indices.extend(aquired_dict["aquired_indices"])

            bald = acquisition_function(mu_0=mu_0, mu_1=mu_1, t=ds.t, pt=None)

            bald[acquired_indices] = -1.0 * (ds.t[acquired_indices] + 1)

            plotting.acquisition_clean(
                ds_pool=ds,
                acquired_indices=acquired_indices,
                tau_mean=tau_pred,
                legend_title=f"Acquired: {warm_start_size + i * step_size}",
                file_path=trial_dir / f"distribution_{i:03d}.png",
            )
            _ = plotting.acquisition(
                x=ds.x.ravel(),
                t=ds.t,
                tau_true=ds.tau,
                bald=bald,
                legend_loc=(0.1, 0.6),
                file_path=trial_dir / f"prop_{i:03d}.png",
            )
            idx_x = np.argsort(ds.x.ravel())
            _ = plotting.functions(
                x=ds.x.ravel(),
                t=ds.t,
                domain=ds.x[idx_x],
                tau_true=ds.tau[idx_x],
                tau_mean=tau_pred[:20, idx_x],
                legend_title=f"Acquired: {warm_start_size + i * step_size}",
                legend_loc=(0.07, 0.62),
                file_path=trial_dir / f"ensemble_{i:03d}.png",
            )


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


def predict_due(config, trial_dir, ds):
    model_dir = trial_dir / "checkpoints"
    outcome_model = models.DeepKernelGP(
        job_dir=model_dir,
        kernel=config.get("kernel"),
        num_inducing_points=config.get("num_inducing_points"),
        inducing_point_dataset=ds,
        architecture="resnet",
        dim_input=ds.dim_input,
        dim_hidden=config.get("dim_hidden"),
        dim_output=config.get("dim_output"),
        depth=config.get("depth"),
        negative_slope=config.get("negative_slope"),
        batch_norm=False,
        spectral_norm=config.get("spectral_norm"),
        dropout_rate=config.get("dropout_rate"),
        weight_decay=(0.5 * (1 - config.get("dropout_rate"))) / len(ds),
        learning_rate=config.get("learning_rate"),
        batch_size=config.get("batch_size"),
        epochs=config.get("epochs"),
        patience=20,
        num_workers=0,
        seed=config.get("seed"),
    )
    outcome_model.load(load_best=True)
    mu_0, mu_1 = outcome_model.predict_mus(ds, batch_size=len(ds))
    tau_pred = mu_1 - mu_0
    return tau_pred * ds.y_std[0]


def predict_ensemble(config, trial_dir, ds):
    ensemble = build_ensemble(config=config, experiment_dir=trial_dir, ds=ds)
    mus = []
    for model in ensemble:
        mus.append(model.predict_mus(ds))
    mu_0 = []
    mu_1 = []
    for mu in mus:
        mu_0.append(mu[0])
        mu_1.append(mu[1])

    mu_0 = np.asarray(mu_0)
    mu_1 = np.asarray(mu_1)
    tau_pred = mu_1 - mu_0
    return tau_pred * ds.y_std[0]


def build_ensemble(config, experiment_dir, ds):
    ensemble = []
    for i in range(config.get("ensemble_size")):
        model_dir = experiment_dir / "checkpoints" / f"model-{i}" / "mu"
        outcome_model = models.TARNet(
            job_dir=model_dir,
            architecture="resnet",
            dim_input=ds.dim_input,
            dim_hidden=config.get("dim_hidden"),
            dim_output=config.get("dim_output"),
            depth=config.get("depth"),
            negative_slope=config.get("negative_slope"),
            batch_norm=False,
            spectral_norm=config.get("spectral_norm"),
            dropout_rate=config.get("dropout_rate"),
            weight_decay=(0.5 * (1 - config.get("dropout_rate"))) / len(ds),
            learning_rate=config.get("learning_rate"),
            batch_size=config.get("batch_size"),
            epochs=config.get("epochs"),
            patience=20,
            num_workers=0,
            seed=config.get("seed"),
        )
        outcome_model.load(load_best=True)
        ensemble.append(outcome_model)
    return ensemble


def rmse_fn(y_pred, y):
    return np.sqrt(np.mean(np.square(y_pred - y)))
