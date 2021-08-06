import torch
import requests
import numpy as np
import pandas as pd

from pathlib import Path

from torch.utils import data

from sklearn import model_selection

_ORDINAL_COVARIATES = [
    "dlivord_min",
    "dtotord_min",
]

_BINARY_COVARIATES = [
    "alcohol",
    "anemia",
    "bord_0",
    "bord_1",
    "cardiac",
    "chyper",
    "csex",
    "diabetes",
    "dmar",
    "eclamp",
    "hemo",
    "herpes",
    "hydra",
    "incervix",
    "lung",
    "othermr",
    "phyper",
    "pre4000",
    "preterm",
    "renal",
    "rh",
    "tobacco",
    "uterine",
]

_CATEGORICAL_COVARIATES = [
    "adequacy",
    "birattnd",
    "brstate",
    "brstate_reg",
    "cigar6",
    "crace",
    "data_year",
    "dfageq",
    "drink5",
    "feduc6",
    "frace",
    "gestat10",
    "mager8",
    "meduc6",
    "mplbir",
    "mplbir_reg",
    "mpre5",
    "mrace",
    "nprevistq",
    "orfath",
    "ormoth",
    "pldel",
    "stoccfipb",
    "stoccfipb_reg",
]

_CYCLICAL_COVARIATES = [
    "birmon",
]


class Twins(data.Dataset):
    def __init__(self, root, split, mode, seed):
        super(Twins, self).__init__()
        rng = np.random.default_rng(seed)
        root = Path(root)
        x_path = root / "twin_pairs_X_3years_samesex.csv"
        t_path = root / "twin_pairs_T_3years_samesex.csv"
        y_path = root / "twin_pairs_Y_3years_samesex.csv"
        # Download data if necessary
        if not x_path.exists():
            root.mkdir(parents=True, exist_ok=True)
            r = requests.get(
                "https://github.com/AMLab-Amsterdam/CEVAE/raw/master/datasets/TWINS/twin_pairs_X_3years_samesex.csv"
            )
            with open(x_path, "wb") as f:
                f.write(r.content)
        if not t_path.exists():
            root.mkdir(parents=True, exist_ok=True)
            r = requests.get(
                "https://github.com/AMLab-Amsterdam/CEVAE/raw/master/datasets/TWINS/twin_pairs_T_3years_samesex.csv"
            )
            with open(t_path, "wb") as f:
                f.write(r.content)
        if not y_path.exists():
            root.mkdir(parents=True, exist_ok=True)
            r = requests.get(
                "https://github.com/AMLab-Amsterdam/CEVAE/raw/master/datasets/TWINS/twin_pairs_Y_3years_samesex.csv"
            )
            with open(y_path, "wb") as f:
                f.write(r.content)
        df = pd.concat(
            [
                pd.read_csv(x_path, sep=",", index_col=0),
                pd.read_csv(t_path, sep=",", index_col=0),
                pd.read_csv(y_path, sep=",", index_col=0),
            ],
            axis=1,
        )
        # Filter birth weight >= 2kg
        condition = (
            (df.dbirwt_0 < 2000) & (df.dbirwt_1 < 2000) & (df.dbirwt_0 != df.dbirwt_1)
        )
        df = df[condition]
        # Convert covariates to one hot encoding
        df = pd.get_dummies(
            df,
            columns=_CATEGORICAL_COVARIATES
            + _BINARY_COVARIATES
            + _CYCLICAL_COVARIATES
            + _ORDINAL_COVARIATES,
            prefix=_CATEGORICAL_COVARIATES
            + _BINARY_COVARIATES
            + _CYCLICAL_COVARIATES
            + _ORDINAL_COVARIATES,
            drop_first=True,
            dummy_na=True,
        )
        # Remove constant rows
        df = df.loc[:, (df != df.iloc[0]).any()]
        df["t"] = rng.integers(0, 2, len(df))
        df["y"] = df.t * df.mort_1 + (1 - df.t) * df.mort_0
        # Train test split
        df_train, df_test = model_selection.train_test_split(
            df, test_size=0.1, random_state=seed
        )
        self.mode = mode
        self.split = split
        # Set x, y, and t values
        self.y_mean = np.asarray([0.0], dtype="float32")
        self.y_std = np.asarray([1.0], dtype="float32")
        covars = (
            _CATEGORICAL_COVARIATES
            + _BINARY_COVARIATES
            + _CYCLICAL_COVARIATES
            + _ORDINAL_COVARIATES
        )
        covars = [col for col in df.columns for var in covars if var in col]
        self.dim_input = len(covars)
        self.dim_treatment = 1
        self.dim_output = 1
        if self.split == "test":
            self.x = df_test[covars].to_numpy(dtype="float32")
            self.t = df_test["t"].to_numpy(dtype="float32")
            self.mu0 = df_test["mort_0"].to_numpy(dtype="float32")
            self.mu1 = df_test["mort_1"].to_numpy(dtype="float32")
            self.y0 = df_test["mort_0"].to_numpy(dtype="float32")
            self.y1 = df_test["mort_1"].to_numpy(dtype="float32")
            if mode == "mu":
                self.y = self.mu1 - self.mu0
            elif mode == "pi":
                self.y = self.t
            else:
                raise NotImplementedError("Not a valid mode")
        else:
            df_train, df_valid = model_selection.train_test_split(
                df_train, test_size=0.3, random_state=seed
            )
            if split == "train":
                df = df_train
            elif split == "valid":
                df = df_valid
            else:
                raise NotImplementedError("Not a valid dataset split")
            self.x = df[covars].to_numpy(dtype="float32")
            self.t = df["t"].to_numpy(dtype="float32")
            self.mu0 = df["mort_0"].to_numpy(dtype="float32")
            self.mu1 = df["mort_1"].to_numpy(dtype="float32")
            self.y0 = df["mort_0"].to_numpy(dtype="float32")
            self.y1 = df["mort_1"].to_numpy(dtype="float32")
            if mode == "mu":
                self.y = df["y"].to_numpy(dtype="float32")
            elif mode == "pi":
                self.y = self.t
            else:
                raise NotImplementedError("Not a valid mode")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        inputs = (
            torch.from_numpy(self.x[idx]).float()
            if self.mode == "pi"
            else torch.from_numpy(np.hstack([self.x[idx], self.t[idx]])).float()
        )
        targets = torch.from_numpy((self.y[idx] - self.y_mean) / self.y_std).float()
        return inputs, targets
