# Counterfactual MRI: Explainability via Generative Models

A comprehensive framework for generating and analyzing counterfactual explanations for MRI classification models using diffusion-based generative models.

## Overview

This project implements a pipeline for:
- Training predictor models on MRI data
- Building diffusion models conditioned on predictor outputs
- Generating counterfactual MRI images that change predictions
- Computing explainability metrics (difference maps, SSIM, histogram analysis)
- Providing interactive visualizations and analysis tools

## Project Structure

```
counterfactual-mri/
├── data/                    # Data storage (raw, processed, metadata)
├── notebooks/               # Jupyter notebooks for exploration and analysis
├── src/
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model architectures
│   ├── training/           # Training scripts
│   ├── inference/          # Inference and generation
│   ├── explainability/     # Explainability metrics
│   └── utils/              # Utilities and configuration
├── app/                    # Next.js web interface
├── configs/                # Configuration files (YAML)
├── experiments/            # Experiment tracking and results
├── checkpoints/            # Model checkpoints
├── reports/                # Generated reports and figures
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.0+ (for GPU support, optional)
- Node.js 14+ (for web app)

### Setup

1. **Clone and navigate to repository**
```bash
cd counterfactual-mri
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Node.js dependencies** (for web app)
```bash
cd app
npm install
```

## Quick Start

### 1. Data Preparation
- Place raw DICOM files in `data/raw/`
- Run preprocessing notebooks to prepare data

### 2. Train Predictor Model
```bash
python src/training/train_predictor.py --config configs/predictor.yaml
```

### 3. Train Diffusion Model
```bash
python src/training/train_diffusion.py --config configs/diffusion.yaml
```

### 4. Generate Counterfactuals
```bash
python -c "
from src.inference.counterfactual_editor import CounterfactualEditor
# Generate counterfactuals for analysis
"
```

## Key Modules

### Data (`src/data/`)
- **dicom_parser.py**: Load and parse DICOM files
- **preprocess.py**: Data preprocessing and normalization
- **dataset.py**: PyTorch Dataset class for MRI data

### Models (`src/models/`)
- **predictor.py**: Classification/regression model
- **diffusion_unet.py**: U-Net architecture for diffusion
- **conditioning.py**: Conditioning mechanisms

### Training (`src/training/`)
- **train_predictor.py**: Predictor training loop
- **train_diffusion.py**: Diffusion model training

### Inference (`src/inference/`)
- **counterfactual_editor.py**: Counterfactual generation
- **sampler.py**: Sampling strategies (DDPM, DDIM)

### Explainability (`src/explainability/`)
- **difference_map.py**: Pixel-wise difference analysis
- **ssim_map.py**: Structural similarity metrics
- **histogram.py**: Intensity distribution analysis
- **validation.py**: Quality and validity metrics

## Configuration

Configuration files are YAML-based:
- `configs/data.yaml`: Data loading parameters
- `configs/predictor.yaml`: Predictor model hyperparameters
- `configs/diffusion.yaml`: Diffusion model hyperparameters

## Notebooks

- `notebooks/01_metadata_exploration.ipynb`: Dataset analysis
- `notebooks/02_preprocessing_checks.ipynb`: QA checks
- `notebooks/03_predictor_baseline.ipynb`: Baseline model training

## Web Interface

The Next.js application provides an interactive interface for:
- Visualizing counterfactual generations
- Exploring explainability metrics
- Comparing original and counterfactual images

```bash
cd app
npm run dev
# Visit http://localhost:3000
```

## Citation

If you use this project in your research, please cite:

```bibtex
@project{counterfactual_mri,
  title={Counterfactual MRI: Explainability via Generative Models},
  year={2024}
}
```

## License

[Add your license here]

## Contact

[Add contact information]
