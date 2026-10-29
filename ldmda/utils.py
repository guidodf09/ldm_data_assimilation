"""Small array helpers shared by all methods.

Conventions
-----------
csi : (n_latent, n)  flattened latent vectors, one column per realization
z   : (n, *latent_shape)
m   : (n, nx, ny, nz) facies in {0, 0.5, 1}
"""
import numpy as np


def model2tricat(m, thresh1, thresh2):
    """Continuous LDM output -> facies {0, 0.5, 1}."""
    out = np.full_like(m, 0.5)
    out[m < thresh1] = 0.0
    out[m > thresh2] = 1.0
    return out


def csi_to_z(csi, latent_shape):
    return csi.T.reshape(csi.shape[1], *latent_shape)


def z_to_csi(z):
    return z.reshape(z.shape[0], -1).T


def facies_to_codes(m):
    """{0, 0.5, 1} float -> {0, 1, 2} uint8 (8x smaller on disk)."""
    return np.rint(m * 2).astype(np.uint8)


def codes_to_facies(c):
    return c.astype(np.float32) / 2.0

