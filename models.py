import torch
import torch.nn as nn
from torch.utils.data import Dataset


# -----------------------------
# 3D Conv Block
# -----------------------------
class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------
# Geo (3-channel) → Rates Model
#   channels: [poro, perm, ar]   (facies is NOT fed to the model)
# -----------------------------
class GeoToRatesModel(nn.Module):
    def __init__(
        self,
        input_shape=(3, 128, 128, 32),   # 3 property channels
        conv_channels=(32, 64, 128),
        latent_dim=256,
        time_steps=30,
        lstm_hidden_size=128,
        lstm_layers=2,
        output_dim=15,
        bidirectional=False,
        dropout_rate=0.3,
        pool_output=(4, 4, 2),
        use_coords=True,
    ):
        super().__init__()

        self.time_steps = time_steps
        self.bidirectional = bidirectional
        self.use_coords = use_coords

        # Normalized coordinate channels: wells sit at fixed grid positions,
        # so the encoder needs location info that conv+pool would otherwise discard.
        if use_coords:
            H, W, D = input_shape[1:]
            grids = torch.meshgrid(
                torch.linspace(-1, 1, H),
                torch.linspace(-1, 1, W),
                torch.linspace(-1, 1, D),
                indexing="ij",
            )
            self.register_buffer("coords", torch.stack(grids, dim=0).unsqueeze(0))  # (1, 3, H, W, D)

        # CNN Encoder
        encoder_layers = []
        in_ch = input_shape[0] + (3 if use_coords else 0)
        for out_ch in conv_channels:
            encoder_layers.append(Conv3DBlock(in_ch, out_ch, stride=1))
            encoder_layers.append(nn.MaxPool3d(kernel_size=2))
            in_ch = out_ch

        self.encoder = nn.Sequential(*encoder_layers)
        self.spatial_pool = nn.AdaptiveAvgPool3d(pool_output)
        self.flatten = nn.Flatten()

        pooled_cells = pool_output[0] * pool_output[1] * pool_output[2]
        self.linear_latent = nn.Sequential(
            nn.Linear(conv_channels[-1] * pooled_cells, latent_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        # Learned Time Embeddings
        self.time_embedding = nn.Embedding(time_steps, latent_dim)

        # LSTM Decoder (input = repeated latent + time embedding)
        self.decoder = nn.LSTM(
            input_size=latent_dim * 2,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if lstm_layers > 1 else 0.0
        )

        decoder_output_dim = lstm_hidden_size * (2 if bidirectional else 1)
        self.output_layer = nn.Linear(decoder_output_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x : (N, 3, H, W, D)  -- [poro, perm, ar] (facies already removed)
        returns: (N, T, output_dim)
        """
        N = x.size(0)

        if self.use_coords:
            x = torch.cat([x, self.coords.expand(N, -1, -1, -1, -1)], dim=1)

        x = self.encoder(x)
        x = self.spatial_pool(x)
        x = self.flatten(x)
        latent = self.linear_latent(x)                 # (N, latent_dim)

        t_idx = torch.arange(self.time_steps, device=latent.device)
        t_emb = self.time_embedding(t_idx).unsqueeze(0).repeat(N, 1, 1)  # (N, T, latent_dim)

        latent_seq = latent.unsqueeze(1).repeat(1, self.time_steps, 1)
        decoder_input = torch.cat([latent_seq, t_emb], dim=-1)

        rnn_out, _ = self.decoder(decoder_input)
        return self.output_layer(rnn_out)              # (N, T, output_dim)


# -----------------------------
# Dataset (no parameter vector)
# -----------------------------
class GeoRatesDataset(Dataset):
    def __init__(self, geo_data, rates_data):
        self.geo_data   = geo_data      # (N, 3, H, W, D)
        self.rates_data = rates_data    # (N, T, F)

    def __len__(self):
        return len(self.geo_data)

    def __getitem__(self, idx):
        return self.geo_data[idx], self.rates_data[idx]