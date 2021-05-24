import numpy as np

_eps = 1e-7


def random(mu_0, mu_1, t, pt):
    return np.ones_like(mu_0.mean(0))


def tau(mu_0, mu_1, t, pt):
    return (mu_1 - mu_0).var(0)


def mu(mu_0, mu_1, t, pt):
    return t * mu_1.var(0) + (1 - t) * mu_0.var(0)


def rho(mu_0, mu_1, t, pt):
    return tau(mu_0, mu_1, t, pt) / (mu(mu_0, mu_1, 1 - t, pt) + _eps)


def mu_rho(mu_0, mu_1, t, pt):
    return mu(mu_0, mu_1, t, pt) * rho(mu_0, mu_1, t, pt)


FUNCTIONS = {
    "random": random,
    "tau": tau,
    "mu": mu,
    "rho": rho,
    "mu-rho": mu_rho,
}
