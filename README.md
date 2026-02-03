# Transformer-Based MIMO Detection

[![Python Linting](https://github.com/YOUR-USERNAME/transformer-based-MIMO-detection/workflows/Python%20Linting/badge.svg)](https://github.com/YOUR-USERNAME/transformer-based-MIMO-detection/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a transformer-based architecture for MU-MIMO detection. MIMO detection is implemented using a transformer to include the user channel dependencies and related coupling effects. The detector models user-to-user channel dependencies by forming a token per user (built from each user's channel vector and the matched-filter statistic), then uses self-attention to capture inter-user coupling before predicting per-user symbol logits.

## Highlights
- Transformer detector that is permutation-equivariant over users
- Uses complex-valued channel observations with real/imaginary tokenization
- Includes ZF and MMSE baselines for comparison
- End-to-end training on synthetic QPSK symbols with AWGN
- Modular architecture with separated baseline methods and utilities

## Project Structure
- `MIMO_Detection.py`: main script with model definitions, data sampling, baselines, and training loop
- `config.py`: configuration class for experiment parameters
- `data_loader.py`: utilities for loading channel datasets
- `models.py`: neural network architectures (Transformer and MLP detectors)
- `baseline_methods.py`: Zero Forcing (ZF) and MMSE baseline detectors

## Datasets

This project uses two datasets representing different mobility scenarios:

**Dataset 1 - Low Mobility:** Derived from the RENEW project dataset repository, this dataset contains channel measurements from a 64-antenna massive MIMO base station at Rice University. Users are placed in multiple locations including 4 line-of-sight and non-line-of-sight clusters with 25 locations within each cluster, emulating a low-mobility network where users move within a cluster. The dataset includes channels from 64 users, with 52 frequency subcarriers per OFDM symbol and 500 frames.

**Dataset 2 - High Mobility:** Generated using the QuaDRiGa channel simulator with the 3GPP Urban Micro channel model, this dataset simulates a 64-antenna massive MIMO base station with 64 mobile users moving at 2.8 m/s. Like Dataset 1, it uses 52 subcarriers per user channel and includes multiple frames.

**Download Links:**
- [Google Drive Dataset Repository](https://drive.google.com/drive/folders/1zqbyl7yBQmVnAdiys_MMvnxXILg2TrWd)
- [RENEW Wireless Dataset](https://renew-wireless.org/dataset-iuc.html)

## Model Summary
- **Input**: received vector `y` and channel matrix `H`
- **Token design**: for each user `k`, token is `[Re/Im(h_k); Re/Im(z_k)]` where `z = H^H y`
- **Backbone**: Transformer encoder stack
- **Output**: per-user logits over modulation symbols (QPSK by default)

## Setup

### Requirements
- Python >= 3.7
- PyTorch >= 1.10.0
- NumPy >= 1.21.0
- h5py >= 3.0.0

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR-USERNAME/transformer-based-MIMO-detection.git
cd transformer-based-MIMO-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Update the dataset paths at the bottom of `MIMO_Detection.py`:

```python
DATASET_LOW = ".../Low_Mob_pre_process_full.hdf5"
DATASET_HIGH = ".../High_Mob_pre_process_full.hdf5"
```

2. Run the script:

```bash
python MIMO_Detection.py
```

The script will:
- Load low/high mobility datasets
- Sample a batch and compute ZF/MMSE baselines
- Train the transformer-based detector
- Report symbol error rates (SER)

## Performance Comparison

The script compares three detection methods:
- **Zero Forcing (ZF)**: Linear detector that forces channel interference to zero
- **MMSE**: Linear detector that minimizes mean square error with noise consideration
- **Transformer Detector**: Neural network-based detector that learns to exploit channel dependencies

## Notes
- The script expects QPSK symbols and uses a fixed `M=4` classification head
- `Cfg` controls the number of antennas/users, SNR, and device
- The transformer-based detector is designed to capture user channel dependencies through attention over per-user tokens

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{transformer_mimo_detection,
  title={Transformer-Based MIMO Detection},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/YOUR-USERNAME/transformer-based-MIMO-detection}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- RENEW Project for Dataset 1: [https://renew-wireless.org/](https://renew-wireless.org/)
- QuaDRiGa Channel Simulator for Dataset 2 generation
- Rice University for massive MIMO measurements

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
