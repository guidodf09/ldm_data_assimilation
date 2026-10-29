"""Surrogate error covariance C_surr on the surrogate's test set (run once per n_hm_times).

Reproduces the notebook split: dataset shuffled with seed 42, then 80/10/10 random_split
with torch.Generator().manual_seed(42). C_surr is order-invariant, so the test models are
read from the h5 file in sorted order.

    python scripts/compute_surrogate_error.py configs/base.yaml
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldmda.problem import rates_to_d  # noqa: E402
from ldmda.surrogate import Surrogate  # noqa: E402
from ldmda.utils import model2tricat  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config", nargs="?", default="configs/base.yaml")
args = ap.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f)
n_times = cfg["observations"]["n_hm_times"]
device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

# Same shuffle and split as in training
rd = cfg["paths"]["rates_dir"]
rates = np.concatenate([np.load(os.path.join(rd, f"all_data_all_times_part{k}.npy")) for k in (1, 2, 3)])
n = rates.shape[0]
order = np.random.RandomState(42).permutation(n)
n_train, n_val = int(0.8 * n), int(0.1 * n)
_, _, test = random_split(range(n), [n_train, n_val, n - n_train - n_val],
                          generator=torch.Generator().manual_seed(42))
test_orig = np.sort(order[np.array(test.indices)])          # indices in the h5 / rates files

with h5py.File(cfg["paths"]["geomodels_h5"], "r") as f:
    m_test = model2tricat(f["data"][test_orig, 0] / 255.0, *cfg["ldm"]["thresholds"])
true_rates = rates[test_orig]

pred_rates = Surrogate(cfg, device).predict(m_test)

eps = rates_to_d(true_rates, n_times) - rates_to_d(pred_rates, n_times)   # (nd, n_test), QoI-major
D = (eps - eps.mean(axis=1, keepdims=True)) / np.sqrt(eps.shape[1] - 1)
C_surr = D @ D.T

out = cfg["paths"]["surrogate_error"]
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
np.save(out, C_surr)
print(f"C_surr {C_surr.shape} from {eps.shape[1]} test models -> {out}")
print(f"mean |bias| / std: {np.mean(np.abs(eps.mean(1)) / np.sqrt(np.diag(C_surr))):.3f}")
