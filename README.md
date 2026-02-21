# MambaSynth: Causal Mamba for Synthetic Time Series

Causal next-step prediction on synthetic time series using the [Mamba](https://arxiv.org/abs/2312.00752) selective state space model.

A Markov-switching process generates multi-dimensional time series from overlapping oscillatory circles. A Mamba model learns to predict the next time step autoregressively, building increasingly compressed representations of the underlying state space across its layers.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy matplotlib umap-learn scikit-learn
```

### Building the Mamba CUDA Kernel for RTX 5090 (Blackwell / sm_120)

The `mamba-ssm` package ships custom CUDA kernels (`selective_scan` and `causal_conv1d`) that must be compiled from source for Blackwell GPUs. Pre-built wheels do not exist for compute capability 12.0, and PyTorch's bundled CUDA runtime does not include `nvcc`, so a full CUDA toolkit is required.

**1. Install the CUDA 12.8 toolkit** (first version with Blackwell / sm_120 support):

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8
```

This installs `nvcc` to `/usr/local/cuda-12.8/bin/nvcc` (~2-3 GB download).

**2. Build `causal-conv1d`** (Mamba dependency):

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="12.0"
pip install causal-conv1d
```

**3. Clone and build `mamba-ssm` from source:**

```bash
git clone https://github.com/state-spaces/mamba.git /tmp/mamba
cd /tmp/mamba
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="12.0"
pip install -e .
```

The CUDA kernel compilation takes 5-10 minutes. The `setup.py` in `mamba-ssm >= 2.3.0` already handles sm_120 for CUDA 12.8+, so no source patching is needed.

**4. Verify:**

```bash
python -c "
from mamba_ssm import Mamba
import torch
m = Mamba(d_model=128, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 512, 128).cuda()
y = m(x)
print(f'OK: {tuple(x.shape)} -> {tuple(y.shape)}')
"
```

**Why not `pip install mamba-ssm` directly?** The pip package requires `nvcc` at build time. Without a system CUDA toolkit, the build fails with `bare_metal_version` undefined in `setup.py`. The `nvidia-cuda-nvcc-cu12` pip package is misleadingly named -- it contains `ptxas` but not the actual `nvcc` compiler. A full toolkit install via apt is the only reliable path.

**Why CUDA 12.8?** Earlier toolkit versions (e.g., 12.0 from `nvidia-cuda-toolkit` in Ubuntu's default repos) do not support Blackwell's sm_120 compute capability. CUDA 12.8 is the minimum version that can compile kernels for RTX 5090.

## Data Generation

Generate synthetic Markov-switching circle time series:

```bash
python markov_circles_timeseries.py
```

This creates `data/data.npz` and `data/config.json`.

## Training

Train the causal Mamba next-step prediction model:

```bash
python masked_model_gpu_mamba_next.py \
    --d-state 32 --seq-len 1024 --stride 256 \
    --no-compile --no-train-eval
```

### Gate Learning Rate Ablation

Scale or freeze Mamba's input-dependent gating parameters (x_proj, dt_proj) to study the contribution of selective gating:

```bash
# Freeze gates (approximates LTI / S4-like model)
python masked_model_gpu_mamba_next.py \
    --gate-lr-factor 0 --d-state 32 --seq-len 1024 --stride 256 \
    --no-compile --no-train-eval --checkpoint model_next_frozen.pt

# Gates learn 100x slower
python masked_model_gpu_mamba_next.py \
    --gate-lr-factor 0.01 --d-state 32 --seq-len 1024 --stride 256 \
    --no-compile --no-train-eval --checkpoint model_next_slow_gates.pt
```

### Multi-Horizon Loss

Train with multi-scale prediction horizons (1, 2, 4, 8, ... up to N) to encourage the model to capture slow dynamics:

```bash
python masked_model_gpu_mamba_next.py \
    --d-state 32 --seq-len 1024 --stride 256 \
    --no-compile --no-train-eval --max-horizon 64
```

## Evaluation

UMAP visualisation and Levina-Bickel intrinsic dimension estimation across all layers:

```bash
python evaluate_representations.py --checkpoint model_next.pt
python evaluate_representations.py --checkpoint model_next.pt --pca 20
```

## Generation

Autoregressive generation from a seed context:

```bash
python generate.py --checkpoint model_next.pt --seed-len 64 --gen-len 512
python generate.py --checkpoint model_next.pt --gen-len 2048 --temperature 0.5
```

Hidden-state perturbation bursts to encourage state transitions during generation:

```bash
python generate.py --checkpoint model_next.pt --gen-len 2048 \
    --perturb-hidden --perturb-interval 200 --perturb-scale 50 --perturb-duration 10
```

## Gate Visualisation

Visualise internal Mamba gating signals (delta, z, B, C) as heatmaps or UMAPs:

```bash
python visualize_gates.py --checkpoint model_next.pt --layer 7 --sample 0
python visualize_gates.py --checkpoint model_next.pt --umap
```

## Files

| File | Description |
|------|-------------|
| `masked_model_gpu_mamba_next.py` | Main training script (causal Mamba model) |
| `generate.py` | Autoregressive generation with perturbation support |
| `evaluate_representations.py` | UMAP + Levina-Bickel evaluation |
| `visualize_gates.py` | Mamba gate visualisation (heatmaps + UMAP) |
| `dataset.py` | Sliding-window dataset loader |
| `estimate_dimension.py` | Levina-Bickel intrinsic dimension estimator |
| `markov_circles_timeseries.py` | Synthetic data generator |

## Model Architecture

- **Input projection**: Linear(feature_dim → d_model)
- **N Mamba layers**: Pre-norm → Mamba block (selective SSM) → Dropout → Residual
- **Output head**: LayerNorm → Linear(d_model → d_ff) → GELU → Linear(d_ff → feature_dim)

Default configuration: d_model=128, 7 layers, d_state=32, ~982K parameters.
