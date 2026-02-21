"""
Mamba-based Next-Step Time Series Prediction — GPU-Optimised (RTX 5090)

Causal (autoregressive) Mamba model: given x_1, ..., x_t, predict x_{t+1}.
This is the natural paradigm for Mamba — single-direction causal scan,
no mask tokens, no positional encoding, no bidirectional hacks.

Supports multi-horizon prediction loss (--max-horizon) to encourage the
model to capture slow dynamics alongside fast ones.

Supports --gate-lr-factor to scale the learning rate on Mamba's
input-dependent gating parameters (x_proj, dt_proj).  Set to 0 to
freeze gates (approximates an LTI / S4-like model).

Key components
--------------
  MambaLayer                : single causal Mamba block (pre-norm, residual)
  CausalTimeSeriesMamba     : full next-step prediction model

Usage
-----
    python masked_model_gpu_mamba_next.py                          # defaults
    python masked_model_gpu_mamba_next.py --gate-lr-factor 0       # freeze gates
    python masked_model_gpu_mamba_next.py --seq-len 4096 --stride 512
    python masked_model_gpu_mamba_next.py --max-horizon 64
    python masked_model_gpu_mamba_next.py --eval model_next.pt
"""

import argparse
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
    """Single causal Mamba block with pre-norm and residual connection.

    Architecture::

        residual ────────────────── (+) ── out
                  │                  ↑
                  └→ LayerNorm → Mamba → Dropout ──┘

    Parameters
    ----------
    d_model : int
        Model dimension.
    d_state : int
        SSM state dimension.
    d_conv : int
        Causal conv1d kernel size.
    expand : int
        Expansion factor for inner dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state,
                           d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x : (batch, seq_len, d_model) → (batch, seq_len, d_model)"""
        return x + self.dropout(self.mamba(self.norm(x)))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class CausalTimeSeriesMamba(nn.Module):
    """Next-step prediction model using causal Mamba.

    At each time step t, the output predicts x_{t+1} using only
    x_1, ..., x_t (causal — no future information leaks).

    Parameters
    ----------
    feature_dim : int
        Dimension of each time step.
    d_model : int
        Internal model dimension.
    n_layers : int
        Number of Mamba layers.
    d_state : int
        SSM state dimension N.
    d_conv : int
        Causal conv1d kernel size.
    expand : int
        Expansion factor (d_inner = expand × d_model).
    d_ff : int
        Output projection hidden dim.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        feature_dim=20,
        d_model=128,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        d_ff=512,
        dropout=0.1,
        max_horizon=1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state

        # Build log-spaced horizon list: 1, 2, 4, 8, ... up to max_horizon
        horizons = []
        k = 1
        while k <= max_horizon:
            horizons.append(k)
            k *= 2
        if horizons[-1] != max_horizon:
            horizons.append(max_horizon)
        self.horizons = horizons

        self.input_proj = nn.Linear(feature_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            MambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # One output head per horizon
        self.output_heads = nn.ModuleDict({
            str(k): nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, feature_dim),
            )
            for k in self.horizons
        })
    @staticmethod
    def migrate_state_dict(state_dict):
        """Remap old 'output_proj.*' keys to 'output_heads.1.*' for
        checkpoints saved before the multi-horizon refactor."""
        migrated = {}
        for k, v in state_dict.items():
            if k.startswith('output_proj.'):
                new_key = k.replace('output_proj.', 'output_heads.1.', 1)
                migrated[new_key] = v
            else:
                migrated[k] = v
        return migrated

    def forward(self, x):
        """
        Parameters
        ----------
        x : (batch, seq_len, feature_dim)

        Returns
        -------
        preds : dict {horizon_k: (batch, seq_len, feature_dim)}
            preds[k][:, t, :] is the model's prediction of x[:, t+k, :].
            When max_horizon=1, returns a single-entry dict.
        """
        h = self.input_dropout(self.input_proj(x))

        for layer in self.layers:
            h = layer(h)

        h = self.final_norm(h)
        return {k: head(h) for k, head in self.output_heads.items()}

    @torch.no_grad()
    def encode(self, x):
        """Run input through the model and return intermediate
        representations at every layer (for UMAP / intrinsic-dimension
        analysis).

        Returns
        -------
        layer_outputs : list of tensors
            [0..n_layers-1] : (batch, seq_len, d_model) after each layer
            [n_layers]      : (batch, seq_len, feature_dim) output projection
        """
        h = self.input_proj(x)

        layer_outputs = []
        for layer in self.layers:
            h = layer(h)
            layer_outputs.append(h)

        h = self.final_norm(h)
        layer_outputs.append(self.output_heads['1'](h))
        return layer_outputs


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def multi_horizon_mse_loss(preds, target):
    """Average MSE across all prediction horizons.

    For each horizon k, computes MSE between pred[:, :-k] and target[:, k:]
    (i.e. the model at time t predicts x_{t+k}).

    Longer horizons get equal weight so slow dynamics are not under-penalised.
    """
    total = 0.0
    for k_str, pred in preds.items():
        k = int(k_str)
        total = total + ((pred[:, :-k] - target[:, k:]) ** 2).mean()
    return total / len(preds)


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, amp_dtype):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, state, _mask in loader:
        x = x.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            preds = model(x)
            loss = multi_horizon_mse_loss(preds, x)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, state, _mask in loader:
        x = x.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            preds = model(x)
            loss = multi_horizon_mse_loss(preds, x)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_predictions(model, dataset, device, n_samples=3, save_dir='.',
                          amp_dtype=torch.bfloat16):
    """Show ground truth vs one-step-ahead prediction."""
    model.eval()
    save_dir = Path(save_dir)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)),
                         replace=False)

    for i, idx in enumerate(indices):
        x, state, _mask = dataset[idx]
        x_in = x.unsqueeze(0).to(device)

        with torch.no_grad(), torch.autocast(device_type='cuda',
                                             dtype=amp_dtype):
            preds = model(x_in)

        # Use the 1-step head for visualisation
        pred = preds['1']

        x_np = x.numpy()
        pred_np = pred[0].float().cpu().numpy()
        state_np = state.numpy()
        seq_len = x_np.shape[0]

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(
            4, 2, height_ratios=[1, 4, 4, 4],
            width_ratios=[1, 0.02], hspace=0.15, wspace=0.03,
        )

        # State bar
        ax_top = fig.add_subplot(gs[0, 0])
        for t in range(seq_len):
            ax_top.axvspan(t, t + 1,
                           color=plt.cm.tab10(state_np[t] / 10),
                           alpha=0.8, linewidth=0)
        ax_top.set_xlim(0, seq_len)
        ax_top.set_yticks([])
        ax_top.set_ylabel('State', fontsize=9)
        ax_top.set_title(f'Next-step prediction sample {i}', fontsize=12)
        plt.setp(ax_top.get_xticklabels(), visible=False)

        # Ground truth (shifted: x[1:])
        target_np = x_np[1:]
        pred_shifted = pred_np[:-1]
        vmax = max(abs(x_np.min()), abs(x_np.max()))

        ax_gt = fig.add_subplot(gs[1, 0], sharex=ax_top)
        im = ax_gt.imshow(
            target_np.T, aspect='auto', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            extent=[1, seq_len, x_np.shape[1] - 0.5, -0.5],
            interpolation='none',
        )
        ax_gt.set_ylabel('Dimension', fontsize=9)
        ax_gt.set_title('Ground truth  x[t+1]', fontsize=10)
        plt.setp(ax_gt.get_xticklabels(), visible=False)

        # Model prediction (pred[t] → estimate of x[t+1])
        ax_pred = fig.add_subplot(gs[2, 0], sharex=ax_top)
        ax_pred.imshow(
            pred_shifted.T, aspect='auto', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            extent=[1, seq_len, pred_shifted.shape[1] - 0.5, -0.5],
            interpolation='none',
        )
        ax_pred.set_ylabel('Dimension', fontsize=9)
        ax_pred.set_title('Prediction  pred[t] → x̂[t+1]', fontsize=10)
        plt.setp(ax_pred.get_xticklabels(), visible=False)

        # Prediction error
        error = pred_shifted - target_np
        ax_err = fig.add_subplot(gs[3, 0], sharex=ax_top)
        err_max = max(abs(error.min()), abs(error.max()), 1e-6)
        ax_err.imshow(
            error.T, aspect='auto', cmap='RdBu_r',
            vmin=-err_max, vmax=err_max,
            extent=[1, seq_len, error.shape[1] - 0.5, -0.5],
            interpolation='none',
        )
        ax_err.set_xlabel('Time step', fontsize=9)
        ax_err.set_ylabel('Dimension', fontsize=9)
        ax_err.set_title('Prediction error', fontsize=10)

        ax_cb = fig.add_subplot(gs[1, 1])
        plt.colorbar(im, cax=ax_cb)
        for row in [0, 2, 3]:
            fig.add_subplot(gs[row, 1]).axis('off')

        fname = save_dir / f'next_prediction_{i}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {fname}")


# ---------------------------------------------------------------------------
# Training loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(train_losses, val_losses, lrs=None,
                    train_eval_losses=None, save_path='training_loss.png'):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label='Train MSE (dropout ON)',
             linewidth=2, alpha=0.5)
    if train_eval_losses is not None:
        ax1.plot(epochs, train_eval_losses,
                 label='Train MSE (dropout OFF)', linewidth=2)
    ax1.plot(epochs, val_losses, label='Val MSE (dropout OFF)',
             linewidth=2, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Next-Step MSE Loss')
    ax1.set_title('Training Progress (Causal Mamba)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    if lrs is not None:
        ax2 = ax1.twinx()
        ax2.plot(epochs, lrs, color='grey', linewidth=1, alpha=0.5,
                 linestyle='--', label='LR')
        ax2.set_ylabel('Learning Rate', color='grey')
        ax2.tick_params(axis='y', labelcolor='grey')
        ax2.legend(loc='center right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def append_to_log(log_path, epoch, train_loss, train_eval_loss, val_loss,
                  lr, is_best):
    write_header = not Path(log_path).exists()
    with open(log_path, 'a') as f:
        if write_header:
            f.write('epoch,train_mse,train_eval_mse,val_mse,lr,best\n')
        f.write(f'{epoch},{train_loss:.6f},{train_eval_loss:.6f},'
                f'{val_loss:.6f},{lr:.8f},{"*" if is_best else ""}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train a causal Mamba next-step prediction model on '
                    'synthetic-song data.  (GPU-optimised for RTX 5090)')
    # Data
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--seq-len', type=int, default=512,
                        help='Sequence length per window')
    parser.add_argument('--stride', type=int, default=128)
    # Model
    parser.add_argument('--d-model', type=int, default=128,
                        help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=7,
                        help='Number of Mamba layers')
    parser.add_argument('--d-state', type=int, default=16,
                        help='SSM state dimension N')
    parser.add_argument('--d-conv', type=int, default=4,
                        help='Causal conv1d kernel size')
    parser.add_argument('--expand', type=int, default=2,
                        help='Mamba expansion factor (d_inner = expand × d_model)')
    parser.add_argument('--d-ff', type=int, default=512,
                        help='Output projection hidden dim')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-horizon', type=int, default=1,
                        help='Max prediction horizon for multi-scale loss '
                             '(e.g. 64).  Uses log-spaced horizons: '
                             '1, 2, 4, 8, ... up to this value.')
    parser.add_argument('--gate-lr-factor', type=float, default=1.0,
                        help='LR multiplier for Mamba gating params '
                             '(x_proj, dt_proj).  0 = freeze gates '
                             '(LTI-like), 1 = normal Mamba.')
    # Training
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup-epochs', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    # GPU options
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed-precision')
    parser.add_argument('--no-train-eval', action='store_true',
                        help='Skip eval-mode pass on training set')
    # Checkpointing
    parser.add_argument('--checkpoint', type=str, default='model_next.pt')
    parser.add_argument('--eval', type=str, default=None, metavar='CKPT',
                        help='Evaluate from checkpoint (skip training)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ---- Device ----
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU.")
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

    # ---- Dataset (masking disabled — we only use x and state) ----
    full_ds = SyntheticSongDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        mask_ratio=0.0,
        mask_patch_size=1,
        mask_seed=0,
    )

    n_val = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    use_persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=use_persistent,
    )

    print(f"Dataset: {len(full_ds)} windows  "
          f"(train={n_train}, val={n_val})")
    print(f"  seq_len={args.seq_len}, stride={args.stride}, "
          f"feature_dim={full_ds.feature_dim}")
    print(f"  batch_size={args.batch_size}, "
          f"num_workers={args.num_workers}")

    # ---- Model ----
    model = CausalTimeSeriesMamba(
        feature_dim=full_ds.feature_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_horizon=args.max_horizon,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Causal Mamba (fused CUDA) — {n_params:,} parameters")
    print(f"  d_model={args.d_model}, layers={args.n_layers}, "
          f"expand={args.expand}, d_state={args.d_state}, "
          f"d_conv={args.d_conv}")
    if args.max_horizon > 1:
        print(f"  multi-horizon loss: {model.horizons}")

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
        state_dict = CausalTimeSeriesMamba.migrate_state_dict(state_dict)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            cleaned = {k.replace('_orig_mod.', ''): v
                       for k, v in state_dict.items()}
            cleaned = CausalTimeSeriesMamba.migrate_state_dict(cleaned)
            model.load_state_dict(cleaned)

        val_loss = evaluate(model, val_loader, device, amp_dtype)
        print(f"\nValidation MSE: {val_loss:.4f}")

        visualize_predictions(model, full_ds, device, n_samples=3,
                              amp_dtype=amp_dtype)
        return

    # ---- Optimizer & scheduler ----
    gate_lr = args.lr * args.gate_lr_factor
    raw_m = model._orig_mod if hasattr(model, '_orig_mod') else model
    gate_params, other_params = [], []
    for name, param in raw_m.named_parameters():
        if '.mamba.x_proj.' in name or '.mamba.dt_proj.' in name:
            gate_params.append(param)
        else:
            other_params.append(param)
    if args.gate_lr_factor != 1.0:
        n_gate = sum(p.numel() for p in gate_params)
        n_other = sum(p.numel() for p in other_params)
        print(f"  gate-lr-factor={args.gate_lr_factor}  "
              f"(gate params: {n_gate:,}, other: {n_other:,})")
    param_groups = [
        {'params': other_params, 'lr': args.lr},
        {'params': gate_params, 'lr': gate_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr,
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
    train_losses, val_losses, train_eval_losses, lrs = [], [], [], []
    log_path = 'training_log_next.csv'
    plot_every = 25

    if Path(log_path).exists():
        Path(log_path).unlink()

    do_train_eval = not args.no_train_eval

    if do_train_eval:
        print(f"\n{'Epoch':<7} {'Train(drop)':<12} {'Train(eval)':<12} "
              f"{'Val MSE':<12} {'LR':<12} {'Time':<8} {'Best'}")
        print('-' * 80)
    else:
        print(f"\n{'Epoch':<7} {'Train MSE':<12} {'Val MSE':<12} "
              f"{'LR':<12} {'Time':<8} {'Best'}")
        print('-' * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     scaler, device, amp_dtype)
        train_eval_loss = (evaluate(model, train_loader, device, amp_dtype)
                           if do_train_eval else float('nan'))
        val_loss = evaluate(model, val_loader, device, amp_dtype)

        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        train_losses.append(train_loss)
        train_eval_losses.append(train_eval_loss)
        val_losses.append(val_loss)
        lrs.append(lr)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            raw_model = (model._orig_mod
                         if hasattr(model, '_orig_mod') else model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args),
            }, args.checkpoint)

        append_to_log(log_path, epoch, train_loss, train_eval_loss,
                      val_loss, lr, is_best)

        marker = ' *' if is_best else ''
        if epoch <= 5 or epoch % 10 == 0 or is_best or epoch == args.epochs:
            if do_train_eval:
                print(f"{epoch:<7} {train_loss:<12.4f} {train_eval_loss:<12.4f} "
                      f"{val_loss:<12.4f} {lr:<12.6f} {elapsed:<8.2f}{marker}")
            else:
                print(f"{epoch:<7} {train_loss:<12.4f} {val_loss:<12.4f} "
                      f"{lr:<12.6f} {elapsed:<8.2f}{marker}")

        if epoch % plot_every == 0 or epoch == args.epochs:
            plot_loss_curve(train_losses, val_losses, lrs,
                            train_eval_losses if do_train_eval else None)

    print(f"\nBest val MSE: {best_val:.4f}")
    print(f"Checkpoint saved to {args.checkpoint}")
    print(f"Training log saved to {log_path}")

    # ---- Visualize with best model ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    raw_model = (model._orig_mod
                 if hasattr(model, '_orig_mod') else model)
    raw_model.load_state_dict(ckpt['model_state_dict'])

    visualize_predictions(model, full_ds, device, n_samples=3,
                          amp_dtype=amp_dtype)


if __name__ == '__main__':
    main()
