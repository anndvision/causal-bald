import json
import numpy as np
import scipy.stats

from copy import deepcopy

from causal_bald.library import models
from causal_bald.library import datasets
from causal_bald.library import acquisitions

from causal_bald.application.workflows import utils


def active_learner(model_name, config, experiment_dir, trial):
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
    # Set the trial dir
    experiment_dir = utils.DIRECTORIES[model_name](
        base_dir=experiment_dir, config=config
    )
    trial_dir = experiment_dir / f"trial-{trial:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    # Write config for downstream use
    config_path = trial_dir / "config.json"
    with config_path.open(mode="w") as cp:
        json.dump(config, cp)
    # Get the acquisition function
    acquisition_function = acquisitions.FUNCTIONS.get(
        config.get("acquisition_function")
    )
    # Train pi model if needed by acquisition
    pt = get_propensities(trial_dir=trial_dir, config=config)
    # Do active learning loop
    step_size = config.get("step_size")
    warm_start_size = config.get("warm_start_size")
    max_acquisitions = config.get("max_acquisitions")
    temperature = config.get("temperature")
    use_gumbel = config.get("use_gumbel")
    for i in range(max_acquisitions):
        batch_size = warm_start_size if i == 0 else step_size
        acquisition_dir = trial_dir / f"acquisition-{i:03d}"
        acquired_path = acquisition_dir / "aquired.json"
        if not acquired_path.exists():
            if i == 0:
                scores = acquisitions.random(
                    mu_0=None, mu_1=None, t=ds_active.dataset.t, pt=pt, temperature=None
                )[ds_active.pool_dataset.indices]
            else:
                # Predict pool set
                mu_0, mu_1 = utils.PREDICT_FUNCTIONS[model_name](
                    dataset=ds_active.dataset,
                    job_dir=trial_dir / f"acquisition-{i-1:03d}",
                    config=config,
                )
                # Get acquisition scores
                scores = (
                    acquisition_function(
                        mu_0=mu_0,
                        mu_1=mu_1,
                        t=ds_active.dataset.t,
                        pt=pt,
                        temperature=temperature if temperature > 0.0 else 1.0,
                    )
                )[ds_active.pool_dataset.indices]
            if temperature > 0.0:
                if use_gumbel:
                    p = scores + scipy.stats.gumbel_r.rvs(
                        loc=0, scale=1, size=len(scores), random_state=None,
                    )
                    idx = np.argpartition(p, -batch_size)[-batch_size:]
                else:
                    scores = np.exp(scores)
                    p = scores / scores.sum()
                    idx = np.random.choice(
                        range(len(p)), replace=False, p=p, size=batch_size,
                    )
            else:
                idx = np.argsort(scores)[-batch_size:]
            ds_active.acquire(idx)
            # Train model
            utils.TRAIN_FUNCTIONS[model_name](
                ds_train=ds_active.training_dataset,
                ds_valid=ds_valid,
                job_dir=acquisition_dir,
                config=config,
                dim_input=ds_active.dataset.dim_input,
            )
            # Save acuired points
            with acquired_path.open(mode="w") as ap:
                json.dump(
                    {"aquired_indices": [int(a) for a in ds_active.acquired_indices]},
                    ap,
                )


def get_propensities(trial_dir, config):
    dataset_name = config.get("dataset_name")
    dim_hidden = config.get("dim_hidden")
    depth = config.get("depth")
    negative_slope = config.get("negative_slope")
    dropout_rate = config.get("dropout_rate")
    spectral_norm = config.get("spectral_norm")
    learning_rate = config.get("learning_rate")
    batch_size = config.get("batch_size")
    epochs = config.get("epochs")
    if config.get("acquisition_function") in ["pi", "mu-pi"]:
        config_pi_train = deepcopy(config.get("ds_train"))
        config_pi_train["mode"] = "pi"
        ds_pi_train = datasets.DATASETS.get(dataset_name)(**config_pi_train)
        pi_dir = trial_dir / "pi"
        pi_model = models.NeuralNetwork(
            job_dir=pi_dir,
            architecture="resnet",
            dim_input=ds_pi_train.dim_input,
            dim_hidden=dim_hidden,
            dim_output=1,
            depth=depth,
            negative_slope=negative_slope,
            batch_norm=False,
            spectral_norm=spectral_norm,
            dropout_rate=dropout_rate,
            weight_decay=(0.5 * (1 - config.get("dropout_rate"))) / len(ds_pi_train),
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=10,
            num_workers=0,
            seed=config.get("seed"),
        )
        if not (pi_dir / "best_checkpoint.pt").exists():
            config_pi_valid = deepcopy(config.get("ds_valid"))
            config_pi_valid["mode"] = "pi"
            ds_pi_valid = datasets.DATASETS.get(dataset_name)(**config_pi_valid)
            pi_model.fit(ds_pi_train, ds_pi_valid)
        pi_model.load()
        return pi_model.predict_mean(ds_pi_train).ravel()
    else:
        return None
