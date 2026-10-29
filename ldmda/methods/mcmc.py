"""Multi-chain preconditioned Crank-Nicolson (pCN) MCMC in the LDM latent space.

Each chain starts from an independent prior draw csi ~ N(0, I). All chains are advanced
together (one batched forward call per iteration). The pCN proposal preserves the prior, so
acceptance uses the likelihood ratio only. Stops when the Stan diagnostics pass
`n_consecutive_pass` checks in a row, or at max_iter.

Memory: every iteration stores log-likelihood, acceptance and the monitored latent coordinates;
full latents and rates are stored every `thin` iterations only. Posterior geomodels are
regenerated from the saved latents at the end (DDIM is deterministic).
"""
import os
import time

import numpy as np

from .. import diagnostics as dg
from ..utils import facies_to_codes


def run_mcmc(fwd, obs, loglik, cfg, out_dir, rng):
    c = cfg["mcmc"]
    cc = c["convergence"]
    nc, beta, T = c["num_chains"], c["beta"], c["temperature"]
    max_iter, burn_in, thin = c["max_iter"], c["burn_in"], c["thin"]
    n_lat = fwd.n_latent
    ckpt = os.path.join(out_dir, "checkpoint.npz")

    n_mon = cc["n_monitor"] or n_lat
    mon_idx = np.sort(np.random.default_rng(0).choice(n_lat, n_mon, replace=False))

    # Per-iteration traces (preallocated, small) and thinned full states
    ll_chain = np.zeros((max_iter, nc))
    acc_chain = np.zeros((max_iter, nc), dtype=bool)
    xi_chain = np.zeros((max_iter, nc, n_mon), dtype=np.float32)
    thin_iters, thin_csi, thin_rates, diag_history = [], [], [], []

    if c["resume"] and os.path.exists(ckpt):
        s = np.load(ckpt, allow_pickle=True)
        start = int(s["iter"]) + 1
        ll_chain[:start], acc_chain[:start], xi_chain[:start] = s["ll_chain"], s["acc_chain"], s["xi_chain"]
        thin_iters, thin_csi, thin_rates = list(s["thin_iters"]), list(s["thin_csi"]), list(s["thin_rates"])
        diag_history = list(map(list, s["diag_history"]))
        rng.bit_generator.state = s["rng_state"].item()
        csi_cur = s["csi_cur"]
        _, rates_cur, d_cur = fwd(csi_cur)
        ll_cur = loglik(d_cur)
        print(f"[MCMC] resumed at iter {start} | log-lik {np.round(ll_cur, 2)}")
    else:
        start = 1
        csi_cur = rng.standard_normal((n_lat, nc))           # independent random starts
        _, rates_cur, d_cur = fwd(csi_cur)
        ll_cur = loglik(d_cur)
        print(f"[MCMC] initial log-lik per chain: {np.round(ll_cur, 2)}")
        ll_chain[0], xi_chain[0] = ll_cur, csi_cur[mon_idx].T
        thin_iters, thin_csi, thin_rates = [0], [csi_cur.T.astype(np.float32)], [rates_cur.astype(np.float32)]

    def save_checkpoint(i):
        tmp = ckpt + ".tmp.npz"
        np.savez(tmp, iter=i, csi_cur=csi_cur, ll_chain=ll_chain[: i + 1], acc_chain=acc_chain[: i + 1],
                 xi_chain=xi_chain[: i + 1], mon_idx=mon_idx,
                 thin_iters=np.array(thin_iters), thin_csi=np.array(thin_csi), thin_rates=np.array(thin_rates),
                 diag_history=np.array(diag_history, dtype=float).reshape(-1, 7),
                 rng_state=np.array(rng.bit_generator.state, dtype=object))
        os.replace(tmp, ckpt)

    t0, streak, converged, i = time.time(), 0, False, start - 1
    for i in range(start, max_iter):
        # 1) pCN proposal + forward model
        csi_k = np.sqrt(1 - beta ** 2) * csi_cur + beta * rng.standard_normal(csi_cur.shape)
        _, rates_k, d_k = fwd(csi_k)
        ll_k = loglik(d_k)

        # 2) Accept / reject per chain
        acc = np.log(rng.random(nc)) < (ll_k - ll_cur) / T
        csi_cur[:, acc], ll_cur[acc], rates_cur[acc] = csi_k[:, acc], ll_k[acc], rates_k[acc]

        # 3) Record state (a rejection repeats the previous state)
        ll_chain[i], acc_chain[i], xi_chain[i] = ll_cur, acc, csi_cur[mon_idx].T
        if i % thin == 0:
            thin_iters.append(i)
            thin_csi.append(csi_cur.T.astype(np.float32))
            thin_rates.append(rates_cur.astype(np.float32))

        # 4) Diagnostics
        if i % c["print_every"] == 0:
            el = time.time() - t0
            print(f"\n[MCMC] iter {i} ({el / 60:.1f} min) | acc rate {np.round(acc_chain[1:i + 1].mean(0), 3)} "
                  f"| log-lik {np.round(ll_cur, 2)}")
            if i > burn_in:
                rep = dg.convergence_report(xi_chain[burn_in:i + 1], ll_chain[burn_in:i + 1], cc)
                print(dg.format_report(rep, cc))
                diag_history.append([i, el, rep["max_rhat"], rep["min_bulk_ess"], rep["min_tail_ess"],
                                     rep["min_mean_ess"], rep["converged"]])
                streak = streak + 1 if rep["converged"] else 0
                converged = streak >= cc["n_consecutive_pass"]
                print(f"  rule passed {streak}/{cc['n_consecutive_pass']} in a row"
                      + (" -> CONVERGED" if converged else ""))

        if i % c["save_every"] == 0 or converged or i == max_iter - 1:
            save_checkpoint(i)
        if converged:
            break

    n_iter = i + 1
    print(f"[MCMC] {n_iter} iterations in {time.time() - t0:.0f} s, converged={converged}")
    np.save(os.path.join(out_dir, "acceptance_rate.npy"), acc_chain[1:n_iter].mean(axis=0))

    # Posterior pool: thinned states after burn-in, all chains pooled
    keep = np.where(np.array(thin_iters) > burn_in)[0]
    if len(keep) == 0:
        raise RuntimeError("no thinned states after burn-in: run longer or lower burn_in / thin")
    csi_pool = np.concatenate([thin_csi[k] for k in keep])        # (n_keep * nc, n_lat)
    rates_pool = np.concatenate([thin_rates[k] for k in keep])
    sel = rng.choice(len(csi_pool), size=min(c["n_posterior"], len(csi_pool)), replace=False)
    print(f"[MCMC] posterior pool {len(csi_pool)} states, saving {len(sel)}")

    csi_post = csi_pool[sel]
    m_post, _, _ = fwd(csi_post.T.astype(np.float64))
    np.save(os.path.join(out_dir, "posterior_latents.npy"), csi_post)
    np.save(os.path.join(out_dir, "posterior_rates.npy"), rates_pool[sel])
    np.save(os.path.join(out_dir, "posterior_models.npy"), facies_to_codes(m_post))
