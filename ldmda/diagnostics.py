"""Stan-style convergence diagnostics (Vehtari, Gelman, Simpson, Carpenter, Buerkner 2021).

Input shape for all public functions: (num_chains, num_samples) or (num_chains, num_samples, num_params).
"""
import numpy as np
from scipy.fft import next_fast_len
from scipy.stats import norm, rankdata

_MIN_SAMPLES = 4


def _autocov_fft(x):
    n = len(x)
    n_pad = next_fast_len(2 * n)
    f = np.fft.rfft(x, n=n_pad)
    return np.fft.irfft(f * np.conj(f), n=n_pad)[:n] / n


def _split_chains(chains):
    chains = np.asarray(chains, dtype=np.float64)
    n_chains, n_samples = chains.shape
    n_half = n_samples // 2
    if n_half == 0:
        return np.empty((2 * n_chains, 0), dtype=np.float64)
    return chains[:, : 2 * n_half].reshape(n_chains * 2, n_half)


def _z_scale(ary):
    ary = np.asarray(ary, dtype=np.float64)
    ranks = rankdata(ary.ravel(), method="average")
    return norm.ppf((ranks - 3.0 / 8.0) / (ary.size + 1.0 / 4.0)).reshape(ary.shape)


def _over_params(fn, chains):
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim == 2:
        return fn(chains)
    return np.array([fn(chains[..., p]) for p in range(chains.shape[2])])


def _rhat_kernel(chains):
    M, N = chains.shape
    if M < 2 or N < 2:
        return float("nan")
    W = chains.var(axis=1, ddof=1).mean()
    B = N * np.var(chains.mean(axis=1), ddof=1)
    if W == 0.0:
        return float("inf") if B > 0.0 else float("nan")
    return float(np.sqrt((((N - 1.0) / N) * W + B / N) / W))


def rhat(chains):
    """max(rank-normalized split R-hat, folded rank-normalized split R-hat)."""
    rank = np.asarray(_over_params(lambda a: _rhat_kernel(_z_scale(_split_chains(a))), chains))
    folded = np.asarray(_over_params(
        lambda a: _rhat_kernel(_z_scale(_split_chains(np.abs(a - np.median(a))))), chains))
    out = np.maximum(rank, folded)
    return float(out) if out.ndim == 0 else out


def _ess_kernel(chains):
    M, N = chains.shape
    if N < _MIN_SAMPLES or M < 2:
        return float("nan")
    chain_means = chains.mean(axis=1, keepdims=True)
    W = chains.var(axis=1, ddof=1).mean()
    if W == 0.0:
        return float("nan")
    B = N * np.var(chain_means.ravel(), ddof=1)
    V_plus = ((N - 1.0) / N) * W + B / N
    acov = np.array([_autocov_fft(chains[c] - chain_means[c, 0]) for c in range(M)])
    rho_hat = 1.0 - (W - acov.mean(axis=0)) / V_plus
    rho_hat[0] = 1.0
    n_pairs = N // 2
    pair_sums = rho_hat[0: 2 * n_pairs: 2] + rho_hat[1: 2 * n_pairs: 2]
    non_positive = np.where(pair_sums <= 0.0)[0]
    T_ = int(non_positive[0]) if len(non_positive) > 0 else n_pairs
    if T_ == 0:
        return float(M * N)
    monotone = np.minimum.accumulate(pair_sums[:T_])
    if 2 * T_ < N:
        boundary_even = max(rho_hat[2 * T_], 0.0)
        extra = 1.5 * monotone[-1] if boundary_even > monotone[-1] else boundary_even
    else:
        extra = 0.0
    tau_hat = max(-1.0 + 2.0 * monotone.sum() + extra, 1.0 / max(np.log10(M * N), 1.0))
    return float(M * N / tau_hat)


def bulk_ess(chains):
    return _over_params(lambda a: _ess_kernel(_z_scale(_split_chains(a))), chains)


def _quantile_ess(chains, prob):
    return _over_params(
        lambda a: _ess_kernel(_split_chains((a <= np.quantile(a, prob)).astype(np.float64))), chains)


def tail_ess(chains):
    """min(ESS of the 5% quantile, ESS of the 95% quantile)."""
    out = np.minimum(np.asarray(_quantile_ess(chains, 0.05)), np.asarray(_quantile_ess(chains, 0.95)))
    return float(out) if out.ndim == 0 else out


def ess_mean(chains):
    return _over_params(lambda a: _ess_kernel(_split_chains(a)), chains)


def convergence_report(xi, ll, cc):
    """
    xi : (n_iter, chains, n_monitor)  monitored latent coordinates (post burn-in)
    ll : (n_iter, chains)             log-likelihood
    cc : convergence config dict
    """
    xi_a, ll_a = xi.transpose(1, 0, 2), ll.T
    d = {
        "xi_rhat": np.asarray(rhat(xi_a), float), "xi_bulk": np.asarray(bulk_ess(xi_a), float),
        "xi_tail": np.asarray(tail_ess(xi_a), float), "xi_mean": np.asarray(ess_mean(xi_a), float),
        "ll_rhat": float(rhat(ll_a)), "ll_bulk": float(bulk_ess(ll_a)),
        "ll_tail": float(tail_ess(ll_a)), "ll_mean": float(ess_mean(ll_a)),
        "n_per_chain": ll_a.shape[1],
    }
    rh, bulk, tail, mean = (np.append(d[f"xi_{k}"], d[f"ll_{k}"]) for k in ("rhat", "bulk", "tail", "mean"))
    d["max_rhat"] = float(np.nanmax(rh))
    d["min_bulk_ess"] = float(np.nanmin(bulk))
    d["min_tail_ess"] = float(np.nanmin(tail))
    d["min_mean_ess"] = float(np.nanmin(mean))

    checks = []
    if cc["use_rhat"]:
        checks.append(bool(np.all(np.isfinite(rh)) and np.max(rh) < cc["rhat_threshold"]))
    if cc["use_ess"]:
        checks.append(bool(np.all(np.isfinite(bulk)) and np.all(np.isfinite(tail))
                           and np.min(bulk) >= cc["min_bulk_ess"] and np.min(tail) >= cc["min_tail_ess"]))
    d["converged"] = bool(checks) and all(checks)
    return d


def format_report(d, cc):
    return (f"  draws/chain {d['n_per_chain']} | max R-hat {d['max_rhat']:.4f} (<{cc['rhat_threshold']}) | "
            f"min bulk ESS {d['min_bulk_ess']:.0f} | min tail ESS {d['min_tail_ess']:.0f} "
            f"(>={cc['min_bulk_ess']}, {cc['min_tail_ess']})\n"
            f"  log-lik: R-hat {d['ll_rhat']:.4f}, bulk ESS {d['ll_bulk']:.0f}, tail ESS {d['ll_tail']:.0f}\n"
            f"  latents: R-hat median {np.nanmedian(d['xi_rhat']):.4f} max {np.nanmax(d['xi_rhat']):.4f}, "
            f"bulk ESS min {np.nanmin(d['xi_bulk']):.0f}")
