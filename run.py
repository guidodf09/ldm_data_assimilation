"""Run one history-matching method on the shared LDM + surrogate setup.

    python run.py configs/esmda.yaml
    python run.py configs/mcmc.yaml --set case.name=true2 case.true_model=data/true2_model.npy case.true_rates=data/true2_rates.npy
    python run.py configs/smc.yaml --name smc_case88_test
"""
import argparse
import os

import numpy as np
import torch

from ldmda.config import load_config, save_config
from ldmda.ldm import LDM
from ldmda.problem import ForwardModel, GaussianLikelihood, Observations
from ldmda.surrogate import Surrogate
from ldmda.utils import facies_to_codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="method config, e.g. configs/esmda.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    ap.add_argument("--name", default=None, help="run folder name (default: <case>_<method>)")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.set)
    method = cfg["method"]
    case = cfg["case"]["name"]
    run_dir = os.path.join(cfg["paths"]["results_dir"], args.name or f"{case}_{method}")
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    print(f"{method.upper()} | {case} | device {device} | -> {run_dir}")

    # ---- shared setup ----
    ldm = LDM(cfg, device)
    surrogate = Surrogate(cfg, device)
    obs = Observations(cfg)
    fwd = ForwardModel(ldm, surrogate, obs.n_times)
    loglik = GaussianLikelihood(obs.d_obs, obs.c_total)

    np.save(os.path.join(run_dir, "d_obs.npy"), obs.d_obs[:, 0])
    np.save(os.path.join(run_dir, "true_rates.npy"), obs.true_rates)
    np.save(os.path.join(run_dir, "true_model.npy"), facies_to_codes(np.squeeze(np.load(cfg["case"]["true_model"]))))

    if cfg["prior_samples"] > 0:   # own generator, so the method's random stream does not depend on this
        prior_rng = np.random.default_rng(cfg["seed"] + 1)
        m, rates, _ = fwd(prior_rng.standard_normal((ldm.n_latent, cfg["prior_samples"])))
        np.save(os.path.join(run_dir, "prior_rates.npy"), rates.astype(np.float32))
        np.save(os.path.join(run_dir, "prior_models.npy"), facies_to_codes(m))

    # ---- method ----
    if method == "esmda":
        from ldmda.methods import run_esmda
        run_esmda(fwd, obs, cfg, run_dir, rng)
    elif method == "mcmc":
        from ldmda.methods import run_mcmc
        run_mcmc(fwd, obs, loglik, cfg, run_dir, rng)
    elif method == "smc":
        from ldmda.methods import run_smc
        run_smc(fwd, obs, loglik, cfg, run_dir, rng)
    else:
        raise ValueError(f"unknown method '{method}'")


if __name__ == "__main__":
    main()
