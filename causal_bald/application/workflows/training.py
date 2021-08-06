import json

from causal_bald.library import datasets

from causal_bald.application.workflows import utils


def trainer(
    config, experiment_dir, trial, model_name,
):
    dataset_name = config.get("dataset_name")

    config["ds_train"]["seed"] = trial
    config["ds_valid"]["seed"] = trial + 1 if dataset_name == "synthetic" else trial
    config["ds_test"]["seed"] = trial + 2 if dataset_name == "synthetic" else trial

    ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
    ds_valid = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))

    experiment_dir = utils.DIRECTORIES[model_name]
    experiment_dir = experiment_dir / f"trial-{trial:03d}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    config_path = experiment_dir / "config.json"
    with config_path.open(mode="w") as cp:
        json.dump(config, cp)

    out_dir = experiment_dir / "checkpoints"

    utils.TRAIN_FUNCTIONS[model_name](
        ds_train=ds_train,
        ds_valid=ds_valid,
        job_dir=out_dir,
        config=config,
        dim_input=ds_train.dim_input,
    )
    return -1
