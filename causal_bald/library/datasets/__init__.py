from causal_bald.library.datasets.ihdp import IHDP
from causal_bald.library.datasets.ihdp_cov import IHDPCov
from causal_bald.library.datasets.synthetic import Synthetic
from causal_bald.library.datasets.hcmnist import HCMNIST

from causal_bald.library.datasets.active_learning import ActiveLearningDataset
from causal_bald.library.datasets.active_learning import RandomFixedLengthSampler

DATASETS = {
    "ihdp": IHDP,
    "ihdp-cov": IHDPCov,
    "synthetic": Synthetic,
    "cmnist": HCMNIST,
}
