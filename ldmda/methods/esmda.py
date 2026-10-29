"""ES-MDA in the LDM latent space with (non-distance-based) localization.

Emerick & Reynolds (2013), Computers & Geosciences 55, 3-15.
Localization of the Kalman gain as in Lacerda et al. (2019), J. Petrol. Sci. Eng. 172, 690-706, Eq. (13):

    x_new = x + (R o K)(d_obs + sqrt(alpha) C^1/2 z - d),   K = C_xd (C_dd + alpha C)^-1,   C = Cd + C_surr

Localization options:
  none      R = 1
  pol       pseudo-optimal (Furrer & Bengtsson 2007; Lacerda Eqs. 18-19):
            R = c^2 / (c^2 + (c^2 + c_ii c_jj) / Ne), zeroed where |c| < eps sqrt(c_ii c_jj)
  cbl       correlation-based (Luo et al. 2018; Lacerda Eqs. 33-34): R = 1 if |rho| > theta else 0
  cbl-soft  correlation-based, R = |rho|^gamma, zeroed where |rho| <= theta
"""
import os
import time

import numpy as np

from ..utils import facies_to_codes


def localization_matrix(method, C_xd, x, d, theta=0.05, gamma=2.0, eps=1e-3):
    """R (n_x, nd) for the Kalman gain, or None for no localization. C_xd = cov(x, d)."""
    method = method.lower()
    if method == "none":
        return None
    if method == "pol":
        ne = x.shape[1]
        vv = np.outer(x.var(axis=1, ddof=1), d.var(axis=1, ddof=1))   # c_ii c_jj
        c2 = C_xd ** 2
        R = c2 / (c2 + (c2 + vv) / ne + 1e-300)
        R[np.abs(C_xd) < eps * np.sqrt(vv)] = 0.0
        return R

    s = np.outer(x.std(axis=1, ddof=1), d.std(axis=1, ddof=1))
    rho = np.divide(C_xd, s, out=np.zeros_like(C_xd), where=s > 0)
    if method == "cbl":
        return (np.abs(rho) > theta).astype(float)
    if method == "cbl-soft":
        R = np.abs(rho) ** gamma
        R[np.abs(rho) <= theta] = 0.0
        return R
    raise ValueError(f"unknown localization '{method}' (none | pol | cbl | cbl-soft)")


def run_esmda(fwd, obs, cfg, out_dir, rng):
    c = cfg["esmda"]
    nr, na = c["nr"], c["na"]
    alpha = np.asarray(c["alpha"] if c["alpha"] is not None else [na] * na, dtype=float)
    assert len(alpha) == na, "len(alpha) must equal na"
    if not np.isclose(np.sum(1.0 / alpha), 1.0):
        print(f"WARNING: sum(1/alpha) = {np.sum(1.0 / alpha):.4f} != 1")
    loc = c["localization"]
    n_dummy = c["dummy_params"]
    n_lat = fwd.n_latent

    # Prior: latents + optional dummy parameters (stacked, updated together)
    x = rng.standard_normal((n_lat + n_dummy, nr))
    np.save(os.path.join(out_dir, "prior_latents.npy"), x[:n_lat].T.astype(np.float32))
    if n_dummy:
        np.save(os.path.join(out_dir, "dummy_prior.npy"), x[n_lat:])

    c_err = obs.c_total                               # Cd + C_surr: used in the gain and the perturbations
    L_err = np.linalg.cholesky(c_err)
    rmse = []
    t0 = time.time()

    for it in range(na + 1):
        m, rates, d = fwd(x[:n_lat])
        np.save(os.path.join(out_dir, f"rates_iter{it}.npy"), rates.astype(np.float32))
        rmse.append(float(np.sqrt(np.mean((d - obs.d_obs) ** 2))))
        print(f"[ESMDA] iter {it}/{na}  data RMSE {rmse[-1]:.3f}  ({time.time() - t0:.0f} s)")
        if it == na:
            break

        # Perturbed observations, d_obs + N(0, alpha C)
        d_uc = obs.d_obs + np.sqrt(alpha[it]) * (L_err @ rng.standard_normal((obs.nd, nr)))

        # Ensemble covariances
        dx = x - x.mean(axis=1, keepdims=True)
        dd = d - d.mean(axis=1, keepdims=True)
        C_xd = dx @ dd.T / (nr - 1)
        C_dd = dd @ dd.T / (nr - 1)

        # Localized Kalman gain, R o K
        K = np.linalg.solve(C_dd + alpha[it] * c_err, C_xd.T).T     # system matrix is symmetric
        R = localization_matrix(loc["method"], C_xd, x, d,
                                theta=loc["theta"], gamma=loc["gamma"], eps=loc["eps"])
        if R is not None:
            K *= R
        x = x + K @ (d_uc - d)

    np.save(os.path.join(out_dir, "posterior_latents.npy"), x[:n_lat].T.astype(np.float32))
    np.save(os.path.join(out_dir, "posterior_rates.npy"), rates.astype(np.float32))
    np.save(os.path.join(out_dir, "posterior_models.npy"), facies_to_codes(m))
    np.save(os.path.join(out_dir, "rmse.npy"), np.array(rmse))
    if n_dummy:
        np.save(os.path.join(out_dir, "dummy_posterior.npy"), x[n_lat:])
    print(f"[ESMDA] done in {time.time() - t0:.0f} s")
