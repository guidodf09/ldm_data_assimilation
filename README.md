# Latent-space history matching with a 3D LDM

Companion code for the paper. Three history-matching methods share one setup:

- **Parameterization:** a pretrained 3D latent diffusion model (LDM) that maps a latent `csi ~ N(0, I)` (16×16×4) to a 128×128×32 facies model.
- **Forward model:** a pretrained 3D CNN-LSTM surrogate that maps the facies model to well rates (30 times × 15 QoIs).
- **Likelihood:** Gaussian, with covariance `Cd + C_surr` (measurement error plus surrogate error).

| Method | What it does |
|---|---|
| `esmda` | ES-MDA in latent space with non-distance localization (POL, CBL, soft CBL; Lacerda et al. 2019) |
| `mcmc`  | Multi-chain pCN MCMC; stops on Stan R-hat / bulk and tail ESS (Vehtari et al. 2021) |
| `smc`   | Adaptive-tempering SMC with pCN moves |

## Layout

```
configs/
  base.yaml        shared setup: paths, LDM, surrogate, observations
  esmda.yaml       method settings (merged on top of base.yaml)
  mcmc.yaml
  smc.yaml
ldmda/
  ldm.py           latent -> geomodel (VAE + UNet + DDIM)
  surrogate.py     geomodel -> rates (CNN-LSTM + scalers)
  problem.py       observations, forward model, likelihood
  diagnostics.py   R-hat / ESS
  methods/         esmda.py, mcmc.py, smc.py
scripts/
  compute_surrogate_error.py   builds C_surr from the surrogate test set (run once)
run.py             entry point
```

## Setup

```bash
pip install -r requirements.txt
```

The trained weights are not in this repo. Set the paths in `configs/base.yaml`:

- `ldm_dir` must contain `trained_vae.pt`, `trained_unet.pt`, `autoencoder_properties.pkl` and `unet_properties.pkl`.
- Also set `surrogate_weights`, the two scaler `.pkl` files, the geomodel `.h5` file and the true-data `.npy` file.

## Run

```bash
python scripts/compute_surrogate_error.py configs/base.yaml     # once -> data/C_surr_5times.npy

python run.py configs/esmda.yaml
python run.py configs/mcmc.yaml
python run.py configs/smc.yaml

# other true case / settings without editing files
python run.py configs/mcmc.yaml --set case.name=case81 case.id_true=81 case.true_data=/path/true_81.npy
python run.py configs/mcmc.yaml --set mcmc.resume=true          # continue an MCMC run
```

## Outputs

Each run writes to `results/<case>_<method>/`:

| File | Content |
|---|---|
| `config.yaml` | full resolved config of the run |
| `d_obs.npy`, `true_rates.npy`, `true_model.npy` | observations and reference |
| `posterior_latents.npy` | (n, 1024) |
| `posterior_rates.npy` | (n, 30, 15) |
| `posterior_models.npy` | (n, 128, 128, 32), uint8 facies codes 0/1/2; use `ldmda.utils.codes_to_facies` to get 0/0.5/1 |

Method-specific files:

- **ESMDA:** `rates_iter{k}.npy`, `rmse.npy`, `dummy_*.npy`
- **MCMC:** `checkpoint.npz` (traces, thinned states, diagnostics history), `acceptance_rate.npy`
- **SMC:** `posterior_weights.npy` (the final particles are weighted), `exponents.npy`, `ess.npy`

## Conventions

- Data vectors are QoI-major (`d[q * n_times + t]`) in every method and in `C_surr`.
- By default, all methods use the same noisy `d_obs` (seed 42).
