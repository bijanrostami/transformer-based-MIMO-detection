import h5py
import numpy as np


def load_channels(path_low: str, path_high: str):
    H_file_low = h5py.File(path_low, "r")
    H_file_high = h5py.File(path_high, "r")

    H_r_low = np.array(H_file_low.get("H_r"))
    H_i_low = np.array(H_file_low.get("H_i"))
    H_r_high = np.array(H_file_high.get("H_r"))
    H_i_high = np.array(H_file_high.get("H_i"))  # (TTI, Nr, K, subcarrier)

    H_low = np.array(H_r_low + 1j * H_i_low).transpose(0, 3, 2, 1).reshape(-1, 64, 64)
    H_high = np.array(H_r_high + 1j * H_i_high).transpose(0, 3, 2, 1).reshape(-1, 64, 64)

    return H_low, H_high
