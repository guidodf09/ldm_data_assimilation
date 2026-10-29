"""3D CNN-LSTM surrogate flow model: geomodel -> well rates (n, time_steps, n_qoi)."""
import math

import joblib
import numpy as np
import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class GeoToRatesModel(nn.Module):
    def __init__(self, input_shape=(1, 128, 128, 32), conv_channels=(32, 64, 128), latent_dim=256,
                 time_steps=30, lstm_hidden_size=128, lstm_layers=2, output_dim=15,
                 bidirectional=False, dropout_rate=0.3):
        super().__init__()
        self.time_steps = time_steps
        layers, in_ch = [], input_shape[0]
        for out_ch in conv_channels:
            layers.append(Conv3DBlock(in_ch, out_ch))
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)
        self.spatial_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        self.linear_latent = nn.Sequential(
            nn.Linear(conv_channels[-1], latent_dim), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )
        self.decoder = nn.LSTM(input_size=latent_dim, hidden_size=lstm_hidden_size,
                               num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.output_layer = nn.Linear(lstm_hidden_size * (2 if bidirectional else 1), output_dim)

    def forward(self, x):
        latent = self.linear_latent(self.flatten(self.spatial_pool(self.encoder(x))))
        rnn_out, _ = self.decoder(latent.unsqueeze(1).repeat(1, self.time_steps, 1))
        return self.output_layer(rnn_out)


class Surrogate:
    """Wraps the network with its input/output scalers."""

    def __init__(self, cfg, device):
        c, g = cfg["surrogate"], cfg["grid"]
        self.device = device
        self.batch_size = c["batch_size"]
        self.n_qoi = c["n_qoi"]
        self.net = GeoToRatesModel(
            input_shape=(1, g["nx"], g["ny"], g["nz"]), conv_channels=c["conv_channels"],
            latent_dim=c["latent_dim"], time_steps=c["time_steps"],
            lstm_hidden_size=c["lstm_hidden_size"], lstm_layers=c["lstm_layers"],
            bidirectional=c["bidirectional"], output_dim=c["n_qoi"], dropout_rate=c["dropout"],
        ).to(device)
        self.net.load_state_dict(torch.load(cfg["paths"]["surrogate_weights"], map_location=device))
        self.net.eval()
        self.model_scaler = joblib.load(cfg["paths"]["model_scaler"])
        self.rate_scaler = joblib.load(cfg["paths"]["rate_scaler"])

    @torch.no_grad()
    def predict(self, m):
        """m (n, nx, ny, nz) facies -> rates (n, time_steps, n_qoi) in physical units."""
        n, out = m.shape[0], []
        for b in range(math.ceil(n / self.batch_size)):
            mb = m[b * self.batch_size:(b + 1) * self.batch_size]
            x = self.model_scaler.transform(mb.reshape(mb.shape[0], -1)).reshape(mb.shape)
            y = self.net(torch.tensor(x[:, None], dtype=torch.float32, device=self.device))
            out.append(y.cpu().numpy())
        y = np.concatenate(out)
        return self.rate_scaler.inverse_transform(y.reshape(-1, self.n_qoi)).reshape(y.shape)
