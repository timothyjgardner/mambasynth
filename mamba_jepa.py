"""
Mamba-JEPA: Causal Mamba Joint Embedding Predictive Architecture — GPU-Optimised

Next-step prediction in *representation space* using causal Mamba encoders.
Since Mamba is inherently causal, each position only sees the past — no
masking is needed.  The information asymmetry comes from causality itself.

Architecture (standard)
-----------------------
    Context encoder : Input → Linear(D_in→D) → Mamba (N layers) → LayerNorm → D-dim reps
    Target encoder  : EMA copy (no gradients)
    Predictor       : context_rep[t] → MLP or small Mamba → D-dim
    Loss            : MSE(predictor(context_rep)[:-1], target_rep[1:])

Architecture (multi-scale, --strides)
-------------------------------------
    Branch(stride=S):
        Input (B, T, D_in) → CausalConv1d(kernel=2S, stride=S) → (B, T/S, D)
                            → Mamba (N layers) → LayerNorm → upsample → (B, T, D)
    MultiScaleEncoder:
        Parallel branches → concatenate → Linear(n_branches * D → D) → LayerNorm

Usage
-----
    python mamba_jepa.py                                   # train with defaults
    python mamba_jepa.py --strides 4                       # single conv branch
    python mamba_jepa.py --strides 1,4,16                  # multi-scale branches
    python mamba_jepa.py --per-layer-loss                  # deep supervision
    python mamba_jepa.py --eval mamba_jepa_model.pt        # evaluate
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from mamba_ssm import Mamba
from dataset import SyntheticSongDataset


# ---------------------------------------------------------------------------
# Causal Mamba Layer  (pre-norm residual, fused CUDA kernels)
# ---------------------------------------------------------------------------

class MambaLayer(nn.Module):
    """Single causal Mamba block with pre-norm and residual connection."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state,
                           d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


# ---------------------------------------------------------------------------
# Mamba Encoder (shared architecture for context & target)
# ---------------------------------------------------------------------------

class MambaEncoder(nn.Module):
    """Input projection + causal Mamba layers + final LayerNorm."""

    def __init__(self, feature_dim, d_model, n_layers, d_state, d_conv,
                 expand, dropout):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            MambaLayer(d_model=d_model, d_state=d_state, d_conv=d_conv,
                       expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        # Per-layer norms for stabilising per-layer loss (bounding magnitudes)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

    def forward(self, x, return_all_layers=False):
        """x: (batch, seq_len, feature_dim) → (batch, seq_len, d_model)

        If return_all_layers=True, returns list of normalised per-layer
        outputs (including final norm) instead of just the final output.
        """
        h = self.input_dropout(self.input_proj(x))
        if not return_all_layers:
            for layer in self.layers:
                h = layer(h)
            return self.final_norm(h)

        outputs = []
        for layer, norm in zip(self.layers, self.layer_norms):
            h = layer(h)
            outputs.append(norm(h))
        outputs.append(self.final_norm(h))
        return outputs

    @torch.no_grad()
    def forward_layers(self, x):
        """Return list of per-layer representations for analysis."""
        return self.forward(x, return_all_layers=True)


# ---------------------------------------------------------------------------
# Convolutional Branch — one temporal resolution
# ---------------------------------------------------------------------------

class ConvBranch(nn.Module):
    """Causal strided convolution → Mamba layers at a single timescale.

    Parameters
    ----------
    feature_dim : int
        Raw input dimension.
    d_model : int
        Internal model dimension.
    n_layers : int
        Number of Mamba layers in this branch.
    stride : int
        Temporal stride (1 = full resolution, 4 = 4× downsampled).
    kernel_factor : int
        Conv kernel = kernel_factor * stride (default 2, so kernel = 2*stride).
    """

    def __init__(self, feature_dim, d_model, n_layers, stride=1,
                 kernel_factor=2, d_state=16, d_conv=4, expand=2,
                 dropout=0.1):
        super().__init__()
        self.stride = stride
        self.d_model = d_model

        kernel_size = max(kernel_factor * stride, 1)
        padding = kernel_size - 1  # causal: left-pad so output depends only on past

        if stride == 1:
            self.conv = nn.Sequential(
                nn.Linear(feature_dim, d_model),
                nn.Dropout(dropout),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConstantPad1d((padding, 0), 0.0),
                nn.Conv1d(feature_dim, d_model, kernel_size=kernel_size,
                          stride=stride),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.layers = nn.ModuleList([
            MambaLayer(d_model=d_model, d_state=d_state, d_conv=d_conv,
                       expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

    def _upsample(self, h, T_orig):
        if self.stride <= 1:
            return h
        return h.repeat_interleave(self.stride, dim=1)[:, :T_orig, :]

    def forward(self, x, return_all_layers=False):
        """x: (B, T, D_in) → (B, T, d_model)

        If return_all_layers=True, returns list of normalised per-layer
        outputs (upsampled to original resolution) plus final norm.
        """
        T_orig = x.shape[1]
        if self.stride == 1:
            h = self.conv(x)
        else:
            h = x.transpose(1, 2)
            h = self.conv(h)
            h = h.transpose(1, 2)

        if not return_all_layers:
            for layer in self.layers:
                h = layer(h)
            return self._upsample(self.norm(h), T_orig)

        outputs = []
        for layer, ln in zip(self.layers, self.layer_norms):
            h = layer(h)
            outputs.append(self._upsample(ln(h), T_orig))
        outputs.append(self._upsample(self.norm(h), T_orig))
        return outputs

    @torch.no_grad()
    def forward_layers(self, x):
        return self.forward(x, return_all_layers=True)


# ---------------------------------------------------------------------------
# Multi-Scale Encoder
# ---------------------------------------------------------------------------

class MultiScaleEncoder(nn.Module):
    """Parallel convolutional branches at different strides, fused into a
    single representation stream.

    Parameters
    ----------
    strides : list of int
        Temporal strides for each branch (e.g. [1, 4, 16]).
    layers_per_branch : list of int or int
        Number of Mamba layers per branch.  If int, same for all branches.
    """

    def __init__(self, feature_dim, d_model, strides, layers_per_branch,
                 d_state=16, d_conv=4, expand=2, dropout=0.1,
                 kernel_factor=2):
        super().__init__()
        self.strides = strides
        self.d_model = d_model
        n_branches = len(strides)

        if isinstance(layers_per_branch, int):
            layers_per_branch = [layers_per_branch] * n_branches

        self.branches = nn.ModuleList([
            ConvBranch(
                feature_dim=feature_dim, d_model=d_model,
                n_layers=n_layers, stride=stride,
                kernel_factor=kernel_factor,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout,
            )
            for stride, n_layers in zip(strides, layers_per_branch)
        ])

        if n_branches > 1:
            self.fusion = nn.Sequential(
                nn.Linear(d_model * n_branches, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.fusion = nn.Identity()

    def forward(self, x, return_all_layers=False):
        """x: (B, T, D_in) → (B, T, d_model)

        If return_all_layers=True, returns list of per-layer fused outputs.
        """
        if not return_all_layers:
            branch_outs = [branch(x) for branch in self.branches]
            if len(branch_outs) == 1:
                return self.fusion(branch_outs[0])
            return self.fusion(torch.cat(branch_outs, dim=-1))

        if len(self.branches) == 1:
            return self.branches[0].forward(x, return_all_layers=True)

        all_branch_layers = [b.forward(x, return_all_layers=True)
                             for b in self.branches]
        max_depth = max(len(bl) for bl in all_branch_layers)
        outputs = []
        for d in range(max_depth):
            parts = []
            for bl in all_branch_layers:
                idx = min(d, len(bl) - 1)
                parts.append(bl[idx])
            fused = self.fusion(torch.cat(parts, dim=-1))
            outputs.append(fused)
        return outputs

    @torch.no_grad()
    def forward_layers(self, x):
        return self.forward(x, return_all_layers=True)


# ---------------------------------------------------------------------------
# Mamba Predictor (next-step in representation space)
# ---------------------------------------------------------------------------

class MambaPredictor(nn.Module):
    """Lightweight causal Mamba that maps context_rep[t] → target_rep[t+1].

    Runs a small causal Mamba scan over the full context representation
    sequence, then projects each position to predict the next target rep.
    """

    def __init__(self, d_model, n_layers, d_state, d_conv, expand, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaLayer(d_model=d_model, d_state=d_state, d_conv=d_conv,
                       expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, context_reps):
        h = context_reps
        for layer in self.layers:
            h = layer(h)
        return self.proj(self.norm(h))


class MLPPredictor(nn.Module):
    """Per-position MLP predictor — no temporal modelling.

    Forces the encoder to build representations that are individually
    predictive of the next target rep without any sequence-level context
    in the predictor.
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, context_reps):
        return self.net(context_reps)


# ---------------------------------------------------------------------------
# JEPA Model
# ---------------------------------------------------------------------------

class CausalMambaJEPA(nn.Module):
    """
    Causal Mamba JEPA: next-step prediction in representation space.

    The context encoder produces causal representations (position t sees
    only x_1..x_t).  The predictor maps context_rep[t] to a prediction
    of target_rep[t+1].  The target encoder (EMA, no grad) provides the
    learning signal.

    When ``strides`` is provided, the encoder switches from a plain
    MambaEncoder (linear input projection) to a MultiScaleEncoder with
    causal strided convolutions — one ConvBranch per stride value.

    Parameters
    ----------
    feature_dim : int
        Dimension of each time step.
    d_model : int
        Internal model dimension.
    n_layers : int
        Number of Mamba layers in each encoder (used when strides is None,
        and as default for layers_per_branch).
    d_state : int
        SSM state dimension N.
    d_conv : int
        Causal conv1d kernel size.
    expand : int
        Expansion factor (d_inner = expand × d_model).
    dropout : float
        Dropout rate.
    predictor_n_layers : int
        Number of Mamba layers in the predictor (typically 2).
    predictor_type : str
        'mamba' for a small causal Mamba predictor, 'mlp' for a
        per-position MLP (no temporal modelling — weaker predictor
        forces richer encoder representations).
    per_layer_loss : bool
        If True, each encoder layer (+ final norm) gets its own MLP
        predictor head.  The total loss is the mean of per-layer MSEs.
    strides : list of int or None
        When set, uses MultiScaleEncoder with one ConvBranch per stride
        (e.g. [4] for single slow branch, [1,4,16] for multi-scale).
        When None, uses standard MambaEncoder with linear projection.
    layers_per_branch : list of int, int, or None
        Mamba layers per branch.  Defaults to n_layers for all branches.
    kernel_factor : int
        Conv kernel = kernel_factor × stride.
    """

    def __init__(
        self,
        feature_dim=20,
        d_model=128,
        n_layers=7,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        predictor_n_layers=2,
        predictor_type='mamba',
        per_layer_loss=False,
        strides=None,
        layers_per_branch=None,
        kernel_factor=2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.per_layer_loss = per_layer_loss
        self.strides = strides

        if strides is not None:
            if layers_per_branch is None:
                layers_per_branch = n_layers
            enc_kwargs = dict(
                feature_dim=feature_dim, d_model=d_model,
                strides=strides, layers_per_branch=layers_per_branch,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, kernel_factor=kernel_factor,
            )
            self.context_encoder = MultiScaleEncoder(**enc_kwargs)
            self.target_encoder = MultiScaleEncoder(**enc_kwargs)

            if isinstance(layers_per_branch, int):
                lpb_list = [layers_per_branch] * len(strides)
            else:
                lpb_list = list(layers_per_branch)
            n_layer_outputs = (lpb_list[0] + 1 if len(strides) == 1
                               else max(lpb_list) + 1)
        else:
            enc_kwargs = dict(feature_dim=feature_dim, d_model=d_model,
                              n_layers=n_layers, d_state=d_state, d_conv=d_conv,
                              expand=expand, dropout=dropout)
            self.context_encoder = MambaEncoder(**enc_kwargs)
            self.target_encoder = MambaEncoder(**enc_kwargs)
            n_layer_outputs = n_layers + 1

        self._n_layer_outputs = n_layer_outputs

        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        if per_layer_loss:
            self.predictors = nn.ModuleList([
                MLPPredictor(d_model=d_model, dropout=dropout)
                for _ in range(n_layer_outputs)
            ])
            self.predictor = self.predictors[-1]
        elif predictor_type == 'mlp':
            self.predictor = MLPPredictor(d_model=d_model, dropout=dropout)
        else:
            self.predictor = MambaPredictor(
                d_model=d_model, n_layers=predictor_n_layers,
                d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (batch, seq_len, feature_dim) — raw observations

        Returns
        -------
        If per_layer_loss=False:
            pred_reps   : (batch, seq_len-1, d_model)
            target_reps : (batch, seq_len-1, d_model)

        If per_layer_loss=True:
            per_layer_preds   : list of (batch, seq_len-1, d_model)
            per_layer_targets : list of (batch, seq_len-1, d_model)
        """
        if not self.per_layer_loss:
            context_reps = self.context_encoder(x)
            with torch.no_grad():
                target_reps = self.target_encoder(x)
            pred_reps = self.predictor(context_reps)
            return pred_reps[:, :-1], target_reps[:, 1:]

        context_layers = self.context_encoder(x, return_all_layers=True)
        with torch.no_grad():
            target_layers = self.target_encoder(x, return_all_layers=True)

        per_layer_preds = []
        per_layer_targets = []
        for ctx, tgt, pred_head in zip(context_layers, target_layers,
                                       self.predictors):
            pred = pred_head(ctx)
            per_layer_preds.append(pred[:, :-1])
            per_layer_targets.append(tgt[:, 1:])

        return per_layer_preds, per_layer_targets

    @torch.no_grad()
    def update_target_encoder(self, momentum):
        """EMA update: θ_target = τ · θ_target + (1 − τ) · θ_context"""
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(),
                                self.target_encoder.parameters()):
            p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def encode(self, x):
        """Run input through the target encoder (no masking) and return
        per-layer representations.  Compatible with evaluate_representations.py.

        Returns
        -------
        layer_outputs : list of (batch, seq_len, d_model) tensors,
            one per encoder layer + final norm.
        """
        return self.target_encoder.forward_layers(x)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def jepa_loss(pred_reps, target_reps):
    """MSE between predicted and target representations (already shifted).

    Accepts either single tensors or lists (per-layer loss mode).
    Returns (total_loss, per_layer_losses) when given lists, else just total_loss.
    """
    if isinstance(pred_reps, list):
        layer_losses = [((p - t) ** 2).mean() for p, t in zip(pred_reps, target_reps)]
        total = sum(layer_losses) / len(layer_losses)
        return total, layer_losses
    return ((pred_reps - target_reps) ** 2).mean()


# ---------------------------------------------------------------------------
# EMA momentum schedule
# ---------------------------------------------------------------------------

def momentum_schedule(epoch, n_epochs, base_momentum=0.996):
    """Cosine schedule: base_momentum → 1.0 over training."""
    return 1.0 - (1.0 - base_momentum) * (
        1.0 + math.cos(math.pi * epoch / n_epochs)) / 2.0


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, amp_dtype,
                    momentum):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, _state, _mask in loader:
        x = x.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            pred_reps, target_reps = model(x)
            result = jepa_loss(pred_reps, target_reps)
            loss = result[0] if isinstance(result, tuple) else result

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(optimizer)
        scaler.update()

        raw = model._orig_mod if hasattr(model, '_orig_mod') else model
        raw.update_target_encoder(momentum)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    """Returns (loss, cosine_similarity, target_std[, per_layer_losses]).

    When model uses per_layer_loss, also returns a list of per-layer val losses.
    """
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    is_per_layer = getattr(raw_model, 'per_layer_loss', False)

    model.eval()
    total_loss = 0.0
    total_cos = 0.0
    total_tgt_std = 0.0
    n_batches = 0
    n_heads = None
    per_layer_totals = None

    for x, _state, _mask in loader:
        x = x.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            pred_reps, target_reps = model(x)
            result = jepa_loss(pred_reps, target_reps)

        if isinstance(result, tuple):
            loss, layer_losses = result
            if per_layer_totals is None:
                n_heads = len(layer_losses)
                per_layer_totals = [0.0] * n_heads
            for i, ll in enumerate(layer_losses):
                per_layer_totals[i] += ll.item()
        else:
            loss = result

        # Cosine sim / std on final layer only
        if isinstance(pred_reps, list):
            p, t = pred_reps[-1], target_reps[-1]
        else:
            p, t = pred_reps, target_reps
        pred_flat = p.reshape(-1, p.size(-1)).float()
        tgt_flat = t.reshape(-1, t.size(-1)).float()
        total_cos += nn.functional.cosine_similarity(
            pred_flat, tgt_flat, dim=-1).mean().item()
        total_tgt_std += t.float().std().item()

        total_loss += loss.item()
        n_batches += 1

    n = max(n_batches, 1)
    base = (total_loss / n, total_cos / n, total_tgt_std / n)
    if per_layer_totals is not None:
        return base + ([pl / n for pl in per_layer_totals],)
    return base


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_loss_curve(train_losses, val_losses, cos_sims, lrs=None,
                    save_path='training_loss.png'):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label='Train JEPA Loss',
             linewidth=2, alpha=0.5)
    ax1.plot(epochs, val_losses, label='Val JEPA Loss',
             linewidth=2, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('JEPA Loss (rep-space MSE)')
    ax1.set_title('Mamba-JEPA Training Progress')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, cos_sims, color='green', linewidth=1.5, alpha=0.7,
             label='Cosine Similarity')
    ax2.set_ylabel('Cosine Similarity', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def append_to_log(log_path, epoch, train_loss, val_loss, cos_sim,
                  target_std, lr, momentum, is_best):
    write_header = not Path(log_path).exists()
    with open(log_path, 'a') as f:
        if write_header:
            f.write('epoch,train_loss,val_loss,cos_sim,target_std,'
                    'lr,momentum,best\n')
        f.write(f'{epoch},{train_loss:.6f},{val_loss:.6f},'
                f'{cos_sim:.6f},{target_std:.6f},'
                f'{lr:.8f},{momentum:.6f},{"*" if is_best else ""}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train a Causal Mamba JEPA (next-step in rep space).')
    # Data
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--stride', type=int, default=32)
    # Model (encoder)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=7,
                        help='Mamba layers in each encoder')
    parser.add_argument('--d-state', type=int, default=16,
                        help='SSM state dimension N')
    parser.add_argument('--d-conv', type=int, default=4,
                        help='Causal conv1d kernel size')
    parser.add_argument('--expand', type=int, default=2,
                        help='Mamba expansion factor (d_inner = expand × d_model)')
    parser.add_argument('--dropout', type=float, default=0.1)
    # Model (predictor)
    parser.add_argument('--predictor-n-layers', type=int, default=2,
                        help='Mamba layers in the predictor '
                             '(keep shallow to avoid lazy encoder)')
    parser.add_argument('--predictor-mlp', action='store_true',
                        help='Use a per-position MLP predictor instead of '
                             'Mamba (weaker — forces richer encoder reps)')
    parser.add_argument('--per-layer-loss', action='store_true',
                        help='Each encoder layer gets its own MLP predictor '
                             'head with independent JEPA loss (deep supervision)')
    # Multi-scale (optional convolutional front-end)
    parser.add_argument('--strides', type=str, default=None,
                        help='Comma-separated conv strides per branch '
                             '(e.g. "4" for single slow branch, "1,4,16" '
                             'for multi-scale).  When not set, uses standard '
                             'linear input projection.')
    parser.add_argument('--layers-per-branch', type=str, default=None,
                        help='Comma-separated Mamba layers per branch, '
                             'or single int for all branches.  '
                             'Defaults to --n-layers for each branch.')
    parser.add_argument('--kernel-factor', type=int, default=2,
                        help='Conv kernel = kernel_factor × stride')
    # EMA
    parser.add_argument('--ema-base', type=float, default=0.996,
                        help='Base EMA momentum (annealed to 1.0 via cosine)')
    # Training
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup-epochs', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    # GPU options
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    # Checkpointing
    parser.add_argument('--checkpoint', type=str, default='mamba_jepa_model.pt')
    parser.add_argument('--eval', type=str, default=None, metavar='CKPT',
                        help='Evaluate from a checkpoint (skip training)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ---- Parse multi-scale options ----
    if args.strides is not None:
        branch_strides = [int(s) for s in args.strides.split(',')]
        for s in branch_strides:
            if args.seq_len % s != 0:
                parser.error(f"seq_len={args.seq_len} must be divisible by "
                             f"stride={s}")
        if args.layers_per_branch is not None:
            lpb_parts = args.layers_per_branch.split(',')
            if len(lpb_parts) == 1:
                layers_per_branch = int(lpb_parts[0])
            else:
                layers_per_branch = [int(x) for x in lpb_parts]
        else:
            layers_per_branch = args.n_layers
    else:
        branch_strides = None
        layers_per_branch = None

    # ---- Device ----
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        device = torch.device('cpu')
        args.no_amp = True
        args.no_compile = True
        args.num_workers = 0
    else:
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute_cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {gpu_name}  ({gpu_mem:.1f} GB, "
              f"compute {compute_cap[0]}.{compute_cap[1]})")

    if args.no_amp:
        amp_dtype = torch.float32
        print("Mixed precision: DISABLED (FP32)")
    elif device.type == 'cuda' and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        print("Mixed precision: BF16")
    else:
        amp_dtype = torch.float16
        print("Mixed precision: FP16")

    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Dataset (masking disabled — causality provides asymmetry) ----
    full_ds = SyntheticSongDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        mask_ratio=0.0,
        mask_patch_size=1,
        mask_seed=None,
    )

    n_val = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    use_persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=use_persistent,
    )

    print(f"Dataset: {len(full_ds)} windows  "
          f"(train={n_train}, val={n_val})")
    print(f"  seq_len={args.seq_len}, stride={args.stride}, "
          f"feature_dim={full_ds.feature_dim}")
    print(f"  batch_size={args.batch_size}")

    # ---- Model ----
    pred_type = 'mlp' if (args.predictor_mlp or args.per_layer_loss) else 'mamba'
    model = CausalMambaJEPA(
        feature_dim=full_ds.feature_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
        predictor_n_layers=args.predictor_n_layers,
        predictor_type=pred_type,
        per_layer_loss=args.per_layer_loss,
        strides=branch_strides,
        layers_per_branch=layers_per_branch,
        kernel_factor=args.kernel_factor,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    if args.per_layer_loss:
        pred_label = f"{model._n_layer_outputs} per-layer MLPs"
    else:
        pred_label = "MLP" if pred_type == 'mlp' else f"{args.predictor_n_layers}L Mamba"

    if branch_strides is not None:
        strides_str = ','.join(str(s) for s in branch_strides)
        lpb_str = (str(layers_per_branch) if isinstance(layers_per_branch, int)
                   else ','.join(str(x) for x in layers_per_branch))
        print(f"Model: Multi-Scale Mamba JEPA — {n_trainable:,} trainable, "
              f"{n_total:,} total params")
        print(f"  branches: strides=[{strides_str}], "
              f"layers_per_branch=[{lpb_str}]")
        print(f"  kernel_factor={args.kernel_factor}, predictor: {pred_label}")
    else:
        print(f"Model: Causal Mamba JEPA (next-step rep) — {n_trainable:,} trainable, "
              f"{n_total:,} total params")
        print(f"  encoder: {args.n_layers}L, predictor: {pred_label}")
    print(f"  d_model={args.d_model}, d_state={args.d_state}, "
          f"d_conv={args.d_conv}, expand={args.expand}")
    print(f"  EMA base momentum: {args.ema_base}")

    # ---- torch.compile ----
    if not args.no_compile and device.type == 'cuda':
        print("Compiling model with torch.compile (inductor)...")
        model = torch.compile(model)
        print("  Compilation deferred — first batch will be slower.")
    elif args.no_compile:
        print("torch.compile: DISABLED")

    # ---- GradScaler ----
    scaler = torch.amp.GradScaler(
        device='cuda',
        enabled=(not args.no_amp and device.type == 'cuda'),
    )

    # ---- Eval-only mode ----
    if args.eval is not None:
        ckpt = torch.load(args.eval, map_location=device, weights_only=True)
        state_dict = ckpt['model_state_dict']
        raw = model._orig_mod if hasattr(model, '_orig_mod') else model
        try:
            raw.load_state_dict(state_dict)
        except RuntimeError:
            cleaned = {k.replace('_orig_mod.', ''): v
                       for k, v in state_dict.items()}
            raw.load_state_dict(cleaned)

        val_loss, cos_sim, tgt_std = evaluate(
            model, val_loader, device, amp_dtype)
        print(f"\nVal JEPA Loss: {val_loss:.4f}")
        print(f"Cosine Similarity: {cos_sim:.4f}")
        print(f"Target Repr Std: {tgt_std:.4f}")
        return

    # ---- Optimiser (context encoder + predictor only) ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=0.01)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 0.01,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    # ---- Training loop ----
    best_val = float('inf')
    train_losses, val_losses, cos_sims, lrs = [], [], [], []
    log_path = 'mamba_jepa_training_log.csv'
    plot_every = 25

    if Path(log_path).exists():
        Path(log_path).unlink()

    header = (f"{'Epoch':<7} {'Train':<12} {'Val':<12} {'CosSim':<8} "
              f"{'TgtStd':<8} {'LR':<12} {'Mom':<8} {'Time':<8} {'Best'}")
    if args.per_layer_loss:
        header += "  Per-layer val losses"
    print(f"\n{header}")
    print('-' * (90 + (30 if args.per_layer_loss else 0)))

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        mom = momentum_schedule(epoch, args.epochs, args.ema_base)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, amp_dtype, mom)
        eval_result = evaluate(model, val_loader, device, amp_dtype)
        val_loss, cos_sim, tgt_std = eval_result[:3]
        per_layer_val = eval_result[3] if len(eval_result) > 3 else None

        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        cos_sims.append(cos_sim)
        lrs.append(lr)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            raw = model._orig_mod if hasattr(model, '_orig_mod') else model
            ckpt_args = {
                **vars(args),
                'model_type': 'mamba_jepa',
                'feature_dim': full_ds.feature_dim,
            }
            if branch_strides is not None:
                ckpt_args['branch_strides'] = branch_strides
                ckpt_args['layers_per_branch_parsed'] = (
                    layers_per_branch if isinstance(layers_per_branch, list)
                    else [layers_per_branch] * len(branch_strides)
                )
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'cos_sim': cos_sim,
                'args': ckpt_args,
            }, args.checkpoint)

        append_to_log(log_path, epoch, train_loss, val_loss, cos_sim,
                      tgt_std, lr, mom, is_best)

        marker = ' *' if is_best else ''
        if epoch <= 5 or epoch % 10 == 0 or is_best or epoch == args.epochs:
            line = (f"{epoch:<7} {train_loss:<12.4f} {val_loss:<12.4f} "
                    f"{cos_sim:<8.4f} {tgt_std:<8.4f} "
                    f"{lr:<12.6f} {mom:<8.4f} {elapsed:<8.2f}{marker}")
            if per_layer_val is not None:
                layer_str = '  [' + ', '.join(f'{v:.4f}' for v in per_layer_val) + ']'
                line += layer_str
            print(line)

        if epoch % plot_every == 0 or epoch == args.epochs:
            plot_loss_curve(train_losses, val_losses, cos_sims)

    print(f"\nBest val JEPA loss: {best_val:.4f}")
    print(f"Checkpoint saved to {args.checkpoint}")
    print(f"Training log saved to {log_path}")


if __name__ == '__main__':
    main()
