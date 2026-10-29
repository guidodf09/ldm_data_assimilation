"""3D latent diffusion model (LDM) parameterization: latent csi -> geomodel."""
import math
import os
import pickle

import numpy as np
import torch
from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler

from .utils import csi_to_z, model2tricat


def _load_state(path, device):
    state = torch.load(path, map_location=device)
    # VAE was saved from nn.DataParallel -> strip the "module." prefix
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}


class LDM:
    def __init__(self, cfg, device):
        c, g = cfg["ldm"], cfg["grid"]
        ldm_dir = cfg["paths"]["ldm_dir"]
        self.device = device
        self.latent_shape = tuple(c["latent_shape"])
        self.grid = (g["nx"], g["ny"], g["nz"])
        self.n_steps = c["n_steps"]
        self.batch_size = c["batch_size"]
        self.thresholds = c["thresholds"]

        with open(os.path.join(ldm_dir, "autoencoder_properties.pkl"), "rb") as f:
            p = pickle.load(f)
        self.vae = AutoencoderKL(
            spatial_dims=p["spatial_dims"], in_channels=p["in_channels"],
            out_channels=p["out_channels"], num_channels=p["num_channels"],
            latent_channels=p["latent_channels"], num_res_blocks=p["num_res_blocks"],
            norm_num_groups=p["norm_num_groups"], attention_levels=p["attention_levels"],
        ).to(device)
        self.vae.load_state_dict(_load_state(os.path.join(ldm_dir, "trained_vae.pt"), device))
        self.vae.eval()

        with open(os.path.join(ldm_dir, "unet_properties.pkl"), "rb") as f:
            p = pickle.load(f)
        self.unet = DiffusionModelUNet(
            spatial_dims=p["spatial_dims"], in_channels=p["in_channels"],
            out_channels=p["out_channels"], num_channels=p["num_channels"],
            num_res_blocks=c["unet_num_res_blocks"], attention_levels=p["attention_levels"],
            num_head_channels=p["num_head_channels"],
        ).to(device)
        self.unet.load_state_dict(_load_state(os.path.join(ldm_dir, "trained_unet.pt"), device))
        self.unet.eval()

        s = c["scheduler"]
        self.scheduler = DDIMScheduler(
            num_train_timesteps=s["num_train_timesteps"], schedule=s["schedule"],
            beta_start=s["beta_start"], beta_end=s["beta_end"], clip_sample=False,
        )
        self.inferer = LatentDiffusionInferer(self.scheduler, scale_factor=c["scale_factor"])

    @property
    def n_latent(self):
        return int(np.prod(self.latent_shape))

    @torch.no_grad()
    def generate(self, csi):
        """csi (n_latent, n) -> facies (n, nx, ny, nz) in {0, 0.5, 1}."""
        n = csi.shape[1]
        z = torch.tensor(csi_to_z(csi, self.latent_shape)[:, None], dtype=torch.float32)
        out = np.empty((n, *self.grid), dtype=np.float32)
        for b in range(math.ceil(n / self.batch_size)):
            sl = slice(b * self.batch_size, (b + 1) * self.batch_size)
            self.scheduler.set_timesteps(num_inference_steps=self.n_steps)
            img = self.inferer.sample(
                input_noise=z[sl].to(self.device), autoencoder_model=self.vae,
                diffusion_model=self.unet, scheduler=self.scheduler, verbose=False,
            )
            out[sl] = img[:, 0].float().cpu().numpy()
        return model2tricat(out, *self.thresholds)
