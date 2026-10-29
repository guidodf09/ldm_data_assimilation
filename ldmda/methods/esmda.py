"""ES-MDA in the LDM latent space with (non-distance-based) localization.

Localization options (Lacerda et al. 2019):
  none      R = 1
  pol       pseudo-optimal localization, R = c^2 / (c^2 + (c^2 + var_m var_d) / Ne)
  cbl       correlation-based, hard threshold |rho| > theta
  cbl-soft  correlation-based, R = |rho|^gamma, zeroed below theta
"""
import os
import time

import numpy as np

from ..utils import facies_to_codes


def localization_matrix(method, C_md, m, d, theta=0.05, gamma=2.0):
    method = method.lower()
    if method == "none":
        return np.ones_like(C_md)
    if method == "pol":
        ne = m.shape[1]
        c_ii = np.var(m, axis=1, ddof=1)[:, None]
        c_jj = np.var(d, axis=1, ddof=1)[None, :]
        c2 = C_md ** 2
        return c2 / (c2 + (c2 + c_ii * c_jj) / ne + 1e-12)

    Mm = m - m.mean(axis=1, keepdims=True)
    Dd = d - d.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(Mm, axis=1, keepdims=True) @ np.linalg.norm(Dd, axis=1, keepdims=True).T
    rho = np.divide(Mm @ Dd.T, denom, out=np.zeros_like(C_md), where=denom > 0)
    if method == "cbl":
        return (np.abs(rho) > theta).astype(float)
    if method in ("cbl-soft", "soft-cbl"):
        R = np.abs(rho) ** gamma
        R[R < theta] = 0.0
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

    c_err = obs.cd + obs.c_surr
    sqrt_cd = np.sqrt(obs.cd_diag)[:, None]
    rmse = []
    t0 = time.time()

    for it in range(na + 1):
        m, rates, d = fwd(x[:n_lat])
        np.save(os.path.join(out_dir, f"rates_iter{it}.npy"), rates.astype(np.float32))
        rmse.append(float(np.sqrt(np.mean((d - obs.d_obs) ** 2))))
        print(f"[ESMDA] iter {it}/{na}  data RMSE {rmse[-1]:.3f}  ({time.time() - t0:.0f} s)")
        if it == na:
            break

        # Perturbed observations
        d_uc = obs.d_obs + np.sqrt(alpha[it]) * sqrt_cd * rng.standard_normal((obs.nd, nr))

        # Ensemble covariances
        dx = x - x.mean(axis=1, keepdims=True)
        dd = d - d.mean(axis=1, keepdims=True)
        C_md = dx @ dd.T / (nr - 1)
        C_dd = dd @ dd.T / (nr - 1)

        R = localization_matrix(loc["method"], C_md, x, d, theta=loc["theta"], gamma=loc["gamma"])
        A = C_dd + alpha[it] * c_err                      # symmetric
        K = np.linalg.solve(A, (C_md * R).T).T            # (C_md o R) A^-1
        x = x + K @ (d_uc - d)

    np.save(os.path.join(out_dir, "posterior_latents.npy"), x[:n_lat].T.astype(np.float32))
    np.save(os.path.join(out_dir, "posterior_rates.npy"), rates.astype(np.float32))
    np.save(os.path.join(out_dir, "posterior_models.npy"), facies_to_codes(m))
    np.save(os.path.join(out_dir, "rmse.npy"), np.array(rmse))
    if n_dummy:
        np.save(os.path.join(out_dir, "dummy_posterior.npy"), x[n_lat:])
    print(f"[ESMDA] done in {time.time() - t0:.0f} s")
