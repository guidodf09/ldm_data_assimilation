"""Inverse problem shared by every method: observations, forward model, likelihood.

Data vector ordering is QoI-major everywhere: d[q * n_times + t]
(same as rates[:, :n_times].transpose(2, 1, 0) and as C_surr).
"""
import numpy as np
from scipy import linalg

from .utils import model2tricat


def rates_to_d(rates, n_times):
    """rates (n, T_all, n_qoi) -> d (n_times * n_qoi, n), QoI-major."""
    r = rates[:, :n_times]
    return r.transpose(2, 1, 0).reshape(-1, r.shape[0])


class Observations:
    def __init__(self, cfg):
        o = cfg["observations"]
        self.n_times = o["n_hm_times"]
        self.true_rates = np.load(cfg["case"]["true_data"])[0]            # (T_all, n_qoi)
        self.d_true = rates_to_d(self.true_rates[None], self.n_times)     # (nd, 1)
        self.nd = self.d_true.shape[0]

        self.cd_diag = (o["error_frac"] * self.d_true[:, 0]) ** 2
        self.d_obs = self.d_true.copy()
        if o["add_noise"]:
            # Same call as the MCMC notebook, so the noise realization is identical
            rs = np.random.RandomState(o["noise_seed"])
            self.d_obs += rs.multivariate_normal(np.zeros(self.nd), np.diag(self.cd_diag))[:, None]

        self.c_surr = np.zeros((self.nd, self.nd))
        if o["use_surrogate_error"]:
            self.c_surr = np.load(cfg["paths"]["surrogate_error"])
            assert self.c_surr.shape == (self.nd, self.nd), \
                f"C_surr is {self.c_surr.shape}, expected ({self.nd}, {self.nd}): recompute for n_hm_times"

    @property
    def cd(self):
        return np.diag(self.cd_diag)

    @property
    def c_total(self):
        """Likelihood covariance Cd + C_surr."""
        return self.cd + self.c_surr


class GaussianLikelihood:
    """log p(d_obs | d) = -0.5 (d - d_obs)^T C^-1 (d - d_obs), vectorized over columns."""

    def __init__(self, d_obs, cov):
        self.d_obs = d_obs
        self.chol = linalg.cho_factor(cov, lower=True)

    def __call__(self, d):
        r = d - self.d_obs
        return -0.5 * np.sum(r * linalg.cho_solve(self.chol, r), axis=0)


class ForwardModel:
    """csi (n_latent, n) -> geomodels (n, nx, ny, nz), rates (n, T_all, n_qoi), d (nd, n)."""

    def __init__(self, ldm, surrogate, n_times):
        self.ldm, self.surrogate, self.n_times = ldm, surrogate, n_times
        self.n_latent = ldm.n_latent

    def __call__(self, csi):
        m = self.ldm.generate(csi)
        rates = self.surrogate.predict(m)
        return m, rates, rates_to_d(rates, self.n_times)


def load_true_model(cfg):
    """True geomodel: index id_true of the dataset shuffled with seed 0."""
    import h5py

    with h5py.File(cfg["paths"]["geomodels_h5"], "r") as f:
        dset = f["data"]
        idx = np.random.RandomState(0).permutation(dset.shape[0])[cfg["case"]["id_true"]]
        m = dset[idx, 0] / 255.0
    return model2tricat(m[None], *cfg["ldm"]["thresholds"])
