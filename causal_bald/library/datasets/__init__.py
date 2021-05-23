from causal_bald.library.datasets.ihdp import IHDP

from causal_bald.library.datasets.ihdp_cov import IHDPCov

from causal_bald.library.datasets.synthetic import Synthetic

from causal_bald.library.datasets.active_learning import ActiveLearningDataset

DATASETS = {
    "ihdp": IHDP,
    "ihdp-cov": IHDPCov,
    "synthetic": Synthetic,
}
