from dataclasses import dataclass
import torch


@dataclass
class Cfg:
    """Configuration class for MIMO detection experiments."""
    Nr: int = 16
    K: int = 8
    snr_db: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.complex64

