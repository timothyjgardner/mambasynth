# MambaSynth: Causal Mamba for Synthetic Time Series

Causal next-step prediction on synthetic time series using the [Mamba](https://arxiv.org/abs/2312.00752) selective state space model.

A Markov-switching process generates multi-dimensional time series from overlapping oscillatory circles. A Mamba model learns to predict the next time step autoregressively, building increasingly compressed representations of the underlying state space across its layers.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy matplotlib mamba-ssm umap-learn scikit-learn
```

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
