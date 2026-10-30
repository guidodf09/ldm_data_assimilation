# Latent-Space Data Assimilation with a 3D Latent Diffusion Model

Code for "Data assimilation for subsurface flow using latent diffusion model parameterization: performance of ensemble-Kalman and Monte Carlo techniques" (G. Di Federico, W. Teng, L. J. Durlofsky; under review).

## Summary
We compare data assimilation (DA) algorithms for large 3D channelized geomodels (128×128×32, channel-levee-mud facies) with uncertain geological scenario parameters (mud fraction, channel orientation and width). Geomodels are parameterized with a pretrained 3D latent diffusion model (3D-LDM, [ldm_3d_geomodel](https://github.com/guidodf09/ldm_3d_geomodel)), which maps a 1024-dimensional Gaussian latent variable to a full facies model. DA is performed on the latent variable, so posterior geomodels are always geologically realistic.

Rigorous Monte Carlo sampling needs ~10⁵–10⁶ forward runs, so the numerical simulator is replaced by a fast CNN-LSTM flow surrogate (geomodel → 30 report times × 15 well rates, ~1 ms per run on a GPU). Surrogate error is estimated on its test set and added to the likelihood as a full covariance, `C_tot = C_d + C_surr`.

Three algorithms run on this shared setup:
- **ESMDA** (ensemble smoother with multiple data assimilation) with pseudo-optimal localization of the Kalman gain
- **MCMC**: multi-chain preconditioned Crank-Nicolson (pCN), stopped on Stan R-hat / ESS diagnostics
- **SMC**: adaptive-tempering sequential Monte Carlo with pCN mutations

## Conclusions
- All three methods preserve geological realism, thanks to the 3D-LDM parameterization.
- MCMC and SMC agree closely with each other, for well rates, scenario parameters and spatial statistics.
- Latent-space ESMDA *overestimates* posterior uncertainty: its P10–P90 well-rate ranges are ~2.3–2.9× wider, with higher data mismatch. Large ensembles, localization, more iterations and larger latent spaces do not fix this. This is the opposite of the classical ensemble-collapse problem.
- The likely cause is Kalman-gain updates acting through a highly nonlinear parameterization.
- SMC combines ensemble-level parallelism with rigorous sampling, which makes it the recommended choice with deep-learning parameterizations, given a fast surrogate.

## Contents
- `configs/`
  - `base.yaml`: shared setup (true case, paths, LDM, surrogate, observations)
  - `esmda.yaml`, `mcmc.yaml`, `smc.yaml`: method settings, merged on top of `base.yaml`
  - `surrogate_train.yaml`: surrogate training settings
- `ldmda/`
  - `ldm.py`: latent → geomodel (VAE + U-net + DDIM)
  - `surrogate.py`: geomodel → well rates (CNN-LSTM + scalers)
  - `problem.py`: observations, forward model, Gaussian likelihood
  - `diagnostics.py`: Stan R-hat and bulk/tail ESS
  - `methods/`: `esmda.py`, `mcmc.py`, `smc.py`
- `scripts/`
  - `compute_surrogate_error.py`: builds `C_surr` from the surrogate test set
  - `train_surrogate.py`: (re)trains the flow surrogate
- `data/`: geomodel dataset, simulated rates, and true models/rates for Cases 1–3 (see `data/link_to_data.md`)
- `weights/`: trained VAE, U-net and surrogate (see `weights/link_to_models.md`)
- `template_folder/`: simulation deck used to generate the training and true rates (`test.DATA`, `KR.INC`; 128×128×32 grid, 3 injectors, 6 producers, BHP control, 30 × 50-day report steps). `PERMX.INC` and `PORO.INC` are written per geomodel
- `run.py`: entry point

## Quick Start
1. Install the requirements:
```bash
pip install -r requirements.txt
```
2. Download data and weights from [Google Drive](https://drive.google.com/drive/folders/1vhaRdzOv56gIDrDwPhm8xDW6S-g_nhXB?usp=sharing) and place them as:
```
data/
  geomodels_128_paper.h5
  true{1,2,3}_model.npy, true{1,2,3}_rates.npy
  rates/all_data_all_times_part{1,2,3}.npy
weights/
  trained_vae/        trained_vae.pt, autoencoder_properties.pkl
  trained_unet/       trained_unet.pt, unet_properties.pkl
  trained_surrogate/  model_trained.pth, scalers.joblib
```
3. Build the surrogate error covariance (once per surrogate):
```bash
python scripts/compute_surrogate_error.py configs/base.yaml
```
4. Run a method (from the repo root):
```bash
python run.py configs/esmda.yaml
python run.py configs/mcmc.yaml
python run.py configs/smc.yaml
```
Any config value can be overridden from the command line:
```bash
python run.py configs/smc.yaml --set case.name=true2 case.true_model=data/true2_model.npy case.true_rates=data/true2_rates.npy
python run.py configs/esmda.yaml --set esmda.localization.method=cbl
python run.py configs/mcmc.yaml --set mcmc.resume=true          # continue an MCMC run from its checkpoint
```
5. (Optional) Retrain the surrogate, then repeat step 3:
```bash
python scripts/train_surrogate.py configs/surrogate_train.yaml
```

## Outputs
Each run writes to `results/<case>_<method>/`: the resolved `config.yaml`, `d_obs.npy`, the true rates and model, and the posterior `posterior_latents.npy` (n, 1024), `posterior_rates.npy` (n, 30, 15) and `posterior_models.npy` (n, 128, 128, 32; uint8 codes 0/1/2, see `ldmda.utils.codes_to_facies`). Method-specific files:
- **ESMDA:** `rates_iter{k}.npy`, `rmse.npy`, `dummy_*.npy`
- **MCMC:** `checkpoint.npz` (traces, thinned states, diagnostics history), `acceptance_rate.npy`
- **SMC:** `posterior_weights.npy`, `exponents.npy`, `ess.npy`, `acceptance_rate.npy`

Data vectors are QoI-major (`d[q * n_times + t]`) everywhere, including `C_surr`. All methods use the same noisy observations (seed 42). The first 5 report times (250 days) are history-matched, and the remaining 1250 days are forecast.

## Software Requirements
`torch`, `monai`, `monai-generative`, `numpy`, `scipy`, `scikit-learn`, `joblib`, `h5py`, `pyyaml`, and `particles` (SMC resampling). A GPU is strongly recommended.

## Code implementations are based on the following repositories:
- [ldm_3d_geomodel](https://github.com/guidodf09/ldm_3d_geomodel) (3D-LDM training and generation)
- [monai](https://github.com/Project-MONAI/GenerativeModels) and [diffusers](https://github.com/huggingface/diffusers/)
- [particles](https://github.com/nchopin/particles) (Chopin & Papaspiliopoulos, 2020)

## Contact
Guido Di Federico, Wenchao Teng, Louis J. Durlofsky  
Department of Energy Science & Engineering, Stanford University  
Contact: gdifede@stanford.edu

## Examples
<figure style="text-align: center; margin-bottom: 100px;">
  <img src="./pics/true_88_no_mud.jpg?raw=true" width="250"/>
  <img src="./pics/true_81_no_mud.jpg?raw=true" width="250"/>
  <img src="./pics/true_28_no_mud.jpg?raw=true" width="250"/>
  <figcaption>Synthetic true models for Cases 1–3 (mud facies hidden), with injectors I1–I3 and producers P1–P6</figcaption>
</figure>

<figure style="text-align: center; margin-bottom: 100px;">
  <img src="./pics/flow_surrogate.jpg?raw=true" width="700"/>
  <figcaption>CNN-LSTM flow surrogate: 3D convolutional encoder, time embedding and LSTM decoder</figcaption>
</figure>

<figure style="text-align: center; margin-bottom: 100px;">
  <img src="./pics/post_sample_22_ESMDA.jpg?raw=true" width="250"/>
  <img src="./pics/post_sample_23_MCMC.jpg?raw=true" width="250"/>
  <img src="./pics/post_sample_42_SMC.jpg?raw=true" width="250"/>
  <figcaption>Random posterior realizations for Case 1: ESMDA, MCMC and SMC (left to right)</figcaption>
</figure>

<figure style="text-align: center; margin-bottom: 100px;">
  <img src="./pics/rates_P1_OIL_comparison_hm_case88.jpg?raw=true" width="350"/>
  <img src="./pics/rates_P1_WAT_comparison_hm_case88.jpg?raw=true" width="350"/>
  <img src="./pics/rates_I3_WAT_comparison_hm_case88.jpg?raw=true" width="350"/>
  <figcaption>Case 1 well rates: prior P10–P90 (gray), posterior P10–P90 for ESMDA (green), MCMC (blue) and SMC (purple), with observed (points) and true (red) data</figcaption>
</figure>

<figure style="text-align: center; margin-bottom: 100px;">
  <img src="./pics/variogram3d_45deg.jpg?raw=true" width="700"/>
  <img src="./pics/variogram3d_vertical.jpg?raw=true" width="700"/>
  <figcaption>Case 1 variograms along the diagonal (top) and vertical (bottom): true model (red), prior (gray) and posteriors</figcaption>
</figure>
