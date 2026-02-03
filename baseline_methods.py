"""
Baseline methods for MIMO detection: ZF (Zero Forcing) and MMSE (Minimum Mean Square Error)
"""

import math
import torch


class ZFDetector:
    """
    Zero Forcing (ZF) MIMO Detector
    
    The ZF detector computes the optimal linear detector that forces the channel
    interference to zero. This is also known as the pseudo-inverse approach.
    
    Weight computation: W = H(H^H H)^{-1}
    Detection: s_hat = W^H y
    """
    
    @staticmethod
    def compute_weights(H: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        Compute ZF weights.
        
        Args:
            H: Channel matrix of shape (B, Nr, K)
            eps: Regularization term for numerical stability
            
        Returns:
            ZF weights of shape (B, K, Nr)
        """
        B, Nr, K = H.shape
        device = H.device
        dtype = H.dtype
        
        # Compute Gram matrix: G = H^H H
        gram = H.conj().transpose(-2, -1) @ H  # (B, K, K)
        
        # Add regularization for numerical stability
        gram = gram + eps * torch.eye(K, device=device, dtype=dtype).unsqueeze(0)
        
        # Compute inverse: G^{-1}
        I = torch.eye(K, device=device, dtype=dtype).unsqueeze(0).expand(B, K, K)
        gram_inv = torch.linalg.solve(gram, I)
        
        # ZF weights: W = H G^{-1}
        zf_weights = H @ gram_inv  # (B, Nr, K)
        
        # Return transposed for detection: (B, K, Nr)
        return zf_weights.permute(0, 2, 1).contiguous()
    
    @staticmethod
    def detect(y: torch.Tensor, H: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        Perform ZF detection on received signal.
        
        Args:
            y: Received signal of shape (B, Nr)
            H: Channel matrix of shape (B, Nr, K)
            eps: Regularization term for numerical stability
            
        Returns:
            Detected symbols of shape (B, K)
        """
        zf_weights = ZFDetector.compute_weights(H, eps)
        # s_hat = W^H y
        s_detected = torch.matmul(torch.conj(zf_weights), y.unsqueeze(-1)).squeeze(-1)  # (B, K)
        return s_detected
    
    @staticmethod
    def qpsk_decode(s_detected: torch.Tensor) -> torch.Tensor:
        """
        Decode QPSK symbols by hard slicing.
        
        Args:
            s_detected: Detected symbols of shape (B, K)
            
        Returns:
            Decoded QPSK symbols of shape (B, K)
        """
        re = torch.where(s_detected.real >= 0, 1.0, -1.0)
        im = torch.where(s_detected.imag >= 0, 1.0, -1.0)
        return torch.complex(re, im) / math.sqrt(2.0)


class MMSEDetector:
    """
    Minimum Mean Square Error (MMSE) MIMO Detector
    
    The MMSE detector minimizes the mean square error between the true symbols
    and the estimated symbols, taking into account the noise power.
    
    Weight computation: W = (H H^H + sigma2 I)^{-1} H
    Detection: s_hat = W^H y
    """
    
    @staticmethod
    def compute_weights(H: torch.Tensor, sigma2: float, eps: float = 1e-10) -> torch.Tensor:
        """
        Compute MMSE weights.
        
        Args:
            H: Channel matrix of shape (B, Nr, K)
            sigma2: Noise power (variance)
            eps: Regularization term for numerical stability
            
        Returns:
            MMSE weights of shape (B, K, Nr)
        """
        B, Nr, K = H.shape
        device = H.device
        dtype = H.dtype
        
        # Compute H H^H + sigma2 I
        gram_mmse = H @ H.conj().transpose(-2, -1)  # (B, Nr, Nr)
        gram_mmse = gram_mmse + (sigma2 + eps) * torch.eye(Nr, device=device, dtype=dtype).unsqueeze(0)
        
        # Compute inverse
        I_mmse = torch.eye(Nr, device=device, dtype=dtype).unsqueeze(0).expand(B, Nr, Nr)
        gram_inv_mmse = torch.linalg.solve(gram_mmse, I_mmse)
        
        # MMSE weights: W = G^{-1} H
        mmse_weights = gram_inv_mmse @ H  # (B, Nr, K)
        
        # Return transposed for detection: (B, K, Nr)
        return mmse_weights.permute(0, 2, 1).contiguous()
    
    @staticmethod
    def detect(y: torch.Tensor, H: torch.Tensor, sigma2: float, eps: float = 1e-10) -> torch.Tensor:
        """
        Perform MMSE detection on received signal.
        
        Args:
            y: Received signal of shape (B, Nr)
            H: Channel matrix of shape (B, Nr, K)
            sigma2: Noise power (variance)
            eps: Regularization term for numerical stability
            
        Returns:
            Detected symbols of shape (B, K)
        """
        mmse_weights = MMSEDetector.compute_weights(H, sigma2, eps)
        # s_hat = W^H y
        s_detected = torch.matmul(torch.conj(mmse_weights), y.unsqueeze(-1)).squeeze(-1)  # (B, K)
        return s_detected
    
    @staticmethod
    def qpsk_decode(s_detected: torch.Tensor) -> torch.Tensor:
        """
        Decode QPSK symbols by hard slicing.
        
        Args:
            s_detected: Detected symbols of shape (B, K)
            
        Returns:
            Decoded QPSK symbols of shape (B, K)
        """
        re = torch.where(s_detected.real >= 0, 1.0, -1.0)
        im = torch.where(s_detected.imag >= 0, 1.0, -1.0)
        return torch.complex(re, im) / math.sqrt(2.0)


@torch.no_grad()
def compute_ser(s_hat: torch.Tensor, s_true: torch.Tensor) -> float:
    """
    Compute Symbol Error Rate (SER).
    
    Args:
        s_hat: Detected symbols of shape (B, K)
        s_true: True symbols of shape (B, K)
        
    Returns:
        Symbol Error Rate as a float
    """
    return (s_hat != s_true).float().mean().item()
