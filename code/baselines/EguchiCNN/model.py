"""EguchiCNN: 2D CNN for semantic segmentation of protein structures.

Architecture from: Eguchi & Huang, Bioinformatics 2020.
"Multi-scale structural analysis of proteins by deep semantic segmentation"

The network takes a Ca distance matrix (L x L) as input and predicts
per-residue class labels via:
  1. Six convolutional encoding layers (down-sampling)
  2. PixelShuffle up-sampling back to L x L
  3. Rectangular convolution to collapse 2D map to 1D (1 x L x C)
  4. Final 1x1 convolution -> per-residue class logits
"""
import torch
import torch.nn as nn


class EguchiCNN(nn.Module):
    """Semantic segmentation CNN for protein contact maps.

    Args:
        max_len: padded input size (default 512)
        num_classes: number of output classes per residue
        dropout: dropout probability (default 0.1)
    """

    def __init__(self, max_len=512, num_classes=10, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.num_classes = num_classes

        # --- Encoding: 6 conv layers with down-sampling ---
        # Input: [B, 1, L, L]
        # Paper: six convolutional layers, each followed by BatchNorm + LeakyReLU
        # We use stride=2 for down-sampling (512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8)
        self.encoder = nn.Sequential(
            # Layer 1: 1 -> 32
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout),
            # Layer 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout),
            # Layer 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout),
            # Layer 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout),
            # Layer 5: 256 -> 256
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout),
            # Layer 6: 256 -> 256
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        # After encoder: [B, 256, 8, 8] (for max_len=512)

        # --- Decoding: PixelShuffle to up-sample back to L x L ---
        # PixelShuffle(r) reshapes [B, C*r*r, H, W] -> [B, C, H*r, W*r]
        # We need to go from 8x8 -> 512x512, factor = 64 = 2^6
        # Use a conv to expand channels, then PixelShuffle
        # r=64 means we need C_in * 64 * 64 channels before shuffle
        # That's too many. Instead, use multiple PixelShuffle stages.
        # 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512 (6 stages, r=2 each)
        shuffle_layers = []
        in_ch = 256
        for i in range(6):
            out_ch = 4  # after shuffle r=2: 4 channels -> 1 channel per stage
            # But we want to keep some channels for intermediate stages
            if i < 5:
                out_ch_before_shuffle = 64  # 64 / (2*2) = 16 channels after shuffle
                shuffle_layers.append(nn.Conv2d(in_ch, out_ch_before_shuffle, kernel_size=3, padding=1))
                shuffle_layers.append(nn.LeakyReLU(inplace=True))
                shuffle_layers.append(nn.PixelShuffle(2))  # channels / 4
                in_ch = out_ch_before_shuffle // 4  # = 16
            else:
                # Last stage: output 4 channels, shuffle to 1 channel at 512x512
                shuffle_layers.append(nn.Conv2d(in_ch, 4, kernel_size=3, padding=1))
                shuffle_layers.append(nn.LeakyReLU(inplace=True))
                shuffle_layers.append(nn.PixelShuffle(2))  # 4 -> 1 channel
                in_ch = 1

        self.decoder = nn.Sequential(*shuffle_layers)
        # After decoder: [B, 1, 512, 512]

        # --- Rectangular convolution: collapse rows to get 1D output ---
        # Paper: "Rectangular convolutions are used to reshape the map into
        # a 1x512@C tensor"
        # Use kernel (max_len, 1) to collapse the row dimension
        # But that's a huge kernel. Instead, use (4, 1) conv + adaptive pool.
        self.rect_conv = nn.Sequential(
            nn.Conv2d(1, num_classes, kernel_size=(4, 1), stride=(4, 1), padding=0),
            nn.LeakyReLU(inplace=True),
        )
        # After rect_conv: [B, num_classes, 128, 512]

        # Adaptive pool to collapse to [B, num_classes, 1, 512]
        self.pool = nn.AdaptiveAvgPool2d((1, max_len))

        # Final 1x1 conv (identity-like, refine logits)
        self.final_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1)

        # Xavier initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass.

        Args:
            x: [B, 1, L, L] distance matrix (scaled by /100)

        Returns:
            logits: [B, num_classes, L] per-residue class logits
        """
        B = x.shape[0]

        # Encode
        h = self.encoder(x)  # [B, 256, 8, 8]

        # Decode via PixelShuffle
        h = self.decoder(h)  # [B, 1, 512, 512]

        # Rectangular conv to collapse 2D -> quasi-1D
        h = self.rect_conv(h)  # [B, C, H', 512]

        # Pool to [B, C, 1, 512]
        h = self.pool(h)

        # Final conv
        h = self.final_conv(h)  # [B, C, 1, 512]

        # Squeeze spatial dim -> [B, C, L]
        logits = h.squeeze(2)  # [B, num_classes, max_len]

        return logits
