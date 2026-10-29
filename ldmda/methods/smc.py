"""Adaptive tempering SMC in the LDM latent space.

Target at stage t: prior(x) * L(x)^epn_t, with 0 = epn_0 < ... < epn_T = 1.
Each new exponent is chosen so that ESS = ess_min * N (bisection), then particles are
resampled and moved with (len_chain - 1) pCN steps targeting the current tempered posterior.
Resampling and ESS come from the `particles` library (Chopin & Papaspiliopoulos 2020).
"""
import os
import time

import numpy as np
from particles import resampling as rs
from scipy import optimize

from ..utils import facies_to_codes


def next_exponent(epn, ess_min, ll):
    """Largest step keeping ESS(delta * ll) >= ess_min * N; capped at 1."""
    N = ll.shape[0]

    def f(delta):
        return (rs.essl(delta * ll) if delta > 0 else N) - ess_min * N

    if f(1.0 - epn) < 0.0:
        return epn + optimize.brentq(f, 0.0, 1.0 - epn)
    return 1.0


def pcn_move(x, ll, epn, loglik_fn, beta, n_steps, rng):
    """n_steps pCN steps on all particles, target prior * L^epn. x is (N, n_lat)."""
    x, ll = x.copy(), ll.copy()
    for k in range(n_steps):
        x_prop = np.sqrt(1 - beta ** 2) * x + beta * rng.standard_normal(x.shape)
        ll_prop = loglik_fn(x_prop)
        p_acc = np.exp(np.clip(epn * (ll_prop - ll), None, 0.0))
        acc = rng.random(x.shape[0]) < p_acc
        x[acc], ll[acc] = x_prop[acc], ll_prop[acc]
        print(f"    step {k + 1}/{n_steps}  acc {acc.mean():.3f}  mean log-lik {ll.mean():.1f}", flush=True)
    return x, ll, float(acc.mean())


def run_smc(fwd, obs, loglik, cfg, out_dir, rng):
    c = cfg["smc"]
    N = c["n_particles"]

    def loglik_fn(x):                 # x (N, n_lat)
        _, _, d = fwd(x.T)
        return loglik(d)

    # particles' resampling uses the legacy global numpy RNG
    np.random.seed(int(rng.integers(2 ** 31)))

    t0 = time.time()
    x = rng.standard_normal((N, fwd.n_latent))
    ll = loglik_fn(x)
    exponents, ess_hist, acc_hist = [0.0], [], []
    lw = np.zeros(N)

    for t in range(c["max_stages"]):
        if t > 0:
            W = rs.exp_and_normalise(lw)
            A = rs.resampling(c["resampling"], W, M=N)
            x, ll = x[A], ll[A]
            x, ll, acc = pcn_move(x, ll, exponents[-1], loglik_fn, c["beta"], c["len_chain"] - 1, rng)
            acc_hist.append(acc)
            lw = np.zeros(N)

        new = next_exponent(exponents[-1], c["ess_min"], ll)
        lw = lw + (new - exponents[-1]) * ll
        exponents.append(new)
        ess_hist.append(rs.essl(lw))
        print(f"[SMC] stage {t}  exponent {new:.4g}  ESS {ess_hist[-1]:.1f}  ({time.time() - t0:.0f} s)")
        if new >= 1.0:
            break

    W = rs.exp_and_normalise(lw)
    np.save(os.path.join(out_dir, "posterior_latents.npy"), x.astype(np.float32))
    np.save(os.path.join(out_dir, "posterior_weights.npy"), W)
    np.save(os.path.join(out_dir, "exponents.npy"), np.array(exponents))
    np.save(os.path.join(out_dir, "ess.npy"), np.array(ess_hist))
    np.save(os.path.join(out_dir, "acceptance_rate.npy"), np.array(acc_hist))

    m, rates, _ = fwd(x.T)
    np.save(os.path.join(out_dir, "posterior_rates.npy"), rates.astype(np.float32))
    np.save(os.path.join(out_dir, "posterior_models.npy"), facies_to_codes(m))
    print(f"[SMC] done: {len(exponents) - 1} stages in {time.time() - t0:.0f} s")
