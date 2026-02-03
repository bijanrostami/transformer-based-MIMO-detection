import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Cfg
from data_loader import load_channels
from models import DetectorNetTransformer
from baseline_methods import ZFDetector, MMSEDetector, compute_ser


def sample_batch(H_low: np.ndarray, B: int, sigma2: float, cfg: Cfg):
    """
    Generate a batch of MIMO signals.
    
    Args:
        H_low: Channel matrix database of shape (N, subcarrier, K, Nr)
        B: Batch size
        sigma2: Noise variance
        cfg: Configuration object
        
    Returns:
        y: Received signal of shape (B, Nr)
        H: Channel matrix of shape (B, Nr, K)
        s: Transmitted symbols of shape (B, K)
        labels: QPSK symbol labels of shape (B, K)
    """
    n_ue = cfg.K
    Nr = cfg.Nr
    dtype = cfg.dtype

    sample_idx = np.random.choice(H_low.shape[0], B, replace=False)
    ue_select = np.random.choice(H_low.shape[1], n_ue, replace=False)
    H_np = H_low[sample_idx][:, ue_select, :]
    H_users = torch.from_numpy(H_np).to(cfg.device)  # (batch, n_ue, Nr)
    H = H_users.transpose(1, 2).contiguous()  # (batch, Nr, n_ue)

    # Random QPSK symbols
    bits = torch.randint(0, 2, (B, n_ue, 2), device=H.device)
    re = 2 * bits[..., 0] - 1
    im = 2 * bits[..., 1] - 1
    s = torch.complex(re.float(), im.float()) / math.sqrt(2.0)
    labels = 2 * bits[..., 0].long() + bits[..., 1].long()

    # AWGN
    re = torch.randn(B, Nr, device=H.device) * math.sqrt(sigma2 / 2)
    im = torch.randn(B, Nr, device=H.device) * math.sqrt(sigma2 / 2)
    complex_noise = torch.complex(re, im).to(dtype)

    # y = H s + n
    y = torch.matmul(H, s.unsqueeze(-1)).squeeze(-1) + complex_noise  # (batch, Nr)

    return y, H, s, labels


@torch.no_grad()
def ser_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Symbol Error Rate (SER) from logits.
    
    Args:
        logits: Output logits of shape (B, K, M)
        labels: True symbol labels of shape (B, K)
        
    Returns:
        Symbol Error Rate as a float
    """
    pred = logits.argmax(dim=-1)  # (B, K)
    return (pred != labels).float().mean().item()


if __name__ == "__main__":
    DATASET_LOW = (
        "./Dataset/Low_Mob_pre_process_full.hdf5"
    )
    DATASET_HIGH = (
        "./Dataset/High_Mob_pre_process_full.hdf5"
    )

    H_low, H_high = load_channels(DATASET_LOW, DATASET_HIGH)

    cfg = Cfg(Nr=64, K=32, snr_db=1)

    sigma2 = 10.0 ** (-cfg.snr_db / 10.0)  # unit symbol power assumption

    B_test = 10
    n_ue = 32
    Nr = cfg.Nr
    device = cfg.device
    dtype = cfg.dtype

    # Sampling MIMO channel and user selection
    y, H, s, labels = sample_batch(H_low, B_test, sigma2, cfg)

    # ========== ZF Detector ==========
    print("=" * 50)
    print("Zero Forcing (ZF) Detector")
    print("=" * 50)
    s_zf = ZFDetector.detect(y, H)
    s_hat_zf = ZFDetector.qpsk_decode(s_zf)
    ser_zf = compute_ser(s_hat_zf, s)
    print(f"ZF SER: {ser_zf:.6f}")

    # ========== MMSE Detector ==========
    print("\n" + "=" * 50)
    print("Minimum Mean Square Error (MMSE) Detector")
    print("=" * 50)
    s_mmse = MMSEDetector.detect(y, H, sigma2)
    s_hat_mmse = MMSEDetector.qpsk_decode(s_mmse)
    ser_mmse = compute_ser(s_hat_mmse, s)
    print(f"MMSE SER: {ser_mmse:.6f}")

    # ========== DNN Model Training ==========
    print("\n" + "=" * 50)
    print("Transformer-based Detector Training")
    print("=" * 50)
    steps = 500
    B = 256
    lr = 2e-4

    model = DetectorNetTransformer(Nr=cfg.Nr, K=n_ue, M=4).to(cfg.device)  # QPSK => M=4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for t in range(1, steps + 1):
        y_tr, H_tr, _, labels_tr = sample_batch(H_low, B, sigma2, cfg)
        logits = model(y_tr, H_tr)
        loss = F.cross_entropy(logits.reshape(-1, 4), labels_tr.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if t % 500 == 0:
            with torch.no_grad():
                ser = ser_from_logits(logits, labels_tr)
            print(f"step {t:4d} | loss {loss.item():.4f} | train SER {ser:.4f}")

    # ========== Model Evaluation ==========
    print("\n" + "=" * 50)
    print("Model Evaluation")
    print("=" * 50)
    logits_test = model(y, H)
    ser_model = ser_from_logits(logits_test, labels)
    print(f"Transformer Detector SER: {ser_model:.6f}")

    # ========== Performance Comparison ==========
    print("\n" + "=" * 50)
    print("Performance Comparison Summary")
    print("=" * 50)
    print(f"ZF Detector:        SER = {ser_zf:.6f}")
    print(f"MMSE Detector:      SER = {ser_mmse:.6f}")
    print(f"Transformer Model:  SER = {ser_model:.6f}")
    print("=" * 50)

