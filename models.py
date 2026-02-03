import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Cfg


def c2r(x: torch.Tensor) -> torch.Tensor:
    """Complex -> real concat [Re, Im]."""
    return torch.cat([x.real, x.imag], dim=-1)


class DetectorNetTransformer(nn.Module):
    """
    Transformer-based MU-MIMO detector (permutation-equivariant over users)

    Inputs:
      y: (B, Nr) complex
      H: (B, Nr, K) complex
    Output:
      logits: (B, K, M)  (e.g., M=4 for QPSK, M=16 for 16QAM)
    """
    def __init__(self, Nr: int, K: int, M: int = 4, d_model: int = 256, n_heads: int = 8, n_layers: int = 4, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.Nr = Nr
        self.K = K
        self.M = M

        token_dim = 2 * Nr + 2  # [Re/Im(h_k)] + [Re/Im(z_k)]
        self.in_proj = nn.Sequential(
            nn.Linear(token_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, M),
        )

    def forward(self, y: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        y: (B, Nr) complex
        H: (B, Nr, K) complex
        returns logits: (B, K, M)
        """
        assert torch.is_complex(y) and torch.is_complex(H), "y and H must be complex tensors"
        B, Nr = y.shape
        assert Nr == self.Nr, f"Expected Nr={{self.Nr}}, got {{Nr}}"
        assert H.shape == (B, self.Nr, self.K), f"Expected H shape {{(B,self.Nr,self.K)}}, got {{H.shape}}"

        # z = H^H y : (B, K)
        z = torch.matmul(H.conj().transpose(-2, -1), y.unsqueeze(-1))  # (B,K,1)

        # H_users: (B, K, Nr) where each token corresponds to a user channel vector h_k
        H_users = H.transpose(1, 2).contiguous()  # (B,K,Nr)

        # Build tokens: (B, K, 2Nr+2)
        tok = torch.cat([c2r(H_users), c2r(z)], dim=-1)

        x = self.in_proj(tok)      # (B,K,d_model)
        x = self.encoder(x)        # (B,K,d_model)
        logits = self.head(x)      # (B,K,M)
        return logits


class DetectorNet(nn.Module):
    """
    Input: y (B,Nr) complex, H (B,Nr,K) complex
    Build z = H^H y (B,K) and G = H^H H (B,K,K)
    Output: logits (B,K,4) for QPSK
    """
    def __init__(self, K: int, hidden: int = 256, depth: int = 4, M: int = 4):
        super().__init__()
        in_dim = 2*K + 2*K*K  # Re/Im of z and G flattened
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.LayerNorm(hidden)]
            d = hidden
        layers += [nn.Linear(hidden, K*M)]
        self.net = nn.Sequential(*layers)
        self.K = K
        self.M = M

    def forward(self, y: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # z = H^H y
        z = torch.matmul(H.conj().transpose(-2, -1), y.unsqueeze(-1)).squeeze(-1)  # (B,K)
        G = torch.matmul(H.conj().transpose(-2, -1), H)  # (B,K,K)

        feat = torch.cat([c2r(z), c2r(G).reshape(y.shape[0], -1)], dim=-1)  # (B, 2K+2K^2)
        out = self.net(feat).reshape(y.shape[0], self.K, self.M)  # (B,K,M)
        return out
