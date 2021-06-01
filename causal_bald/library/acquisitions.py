import numpy as np

from scipy import stats

_eps = 1e-7


def random(mu_0, mu_1, t, pt, temperature):
    return np.ones_like(mu_0.mean(0))


def tau(mu_0, mu_1, t, pt, temperature):
    return (mu_1 - mu_0).var(0) ** (1 / temperature)


def mu(mu_0, mu_1, t, pt, temperature):
    return (t * mu_1.var(0) + (1 - t) * mu_0.var(0)) ** (1 / temperature)


def rho(mu_0, mu_1, t, pt, temperature):
    return tau(mu_0, mu_1, t, pt, temperature) / (
        mu(mu_0, mu_1, 1 - t, pt, temperature) + _eps
    )


def mu_rho(mu_0, mu_1, t, pt, temperature):
    return mu(mu_0, mu_1, t, pt, temperature) * rho(mu_0, mu_1, t, pt, temperature)


def pi(mu_0, mu_1, t, pt, temperature):
    return t * (1 - pt) + (1 - t) * pt


def mu_pi(mu_0, mu_1, t, pt, temperature):
    return mu(mu_0, mu_1, t, pt, temperature) * pi(mu_0, mu_1, t, pt, temperature)


def sundin(mu_0, mu_1, t, pt, temperature):
    tau = mu_1 - mu_0
    gammas = np.clip(stats.norm().cdf(-np.abs(tau) / np.sqrt(2)), _eps, 1 - _eps)
    gamma = gammas.mean(0)
    predictive_entropy = stats.bernoulli(gamma).entropy()
    conditional_entropy = stats.bernoulli(gammas).entropy().mean(0)
    # it can get negative very small number because of numerical instabilities
    mi = predictive_entropy - conditional_entropy
    return mi


FUNCTIONS = {
    "random": random,
    "tau": tau,
    "mu": mu,
    "rho": rho,
    "mu-rho": mu_rho,
    "pi": pi,
    "mu-pi": mu_pi,
    "sundin": sundin,
}
