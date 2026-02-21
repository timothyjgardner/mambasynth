"""
Generate synthetic time series using a trained Causal Mamba model.

Seeds the model with a real context window from the dataset, then
autoregressively generates new time steps by feeding each prediction
back as input.

Usage
-----
    python generate.py --checkpoint model_next.pt --seed-len 64 --gen-len 512
    python generate.py --checkpoint model_next.pt --gen-len 2048 --temperature 0.5
    python generate.py --checkpoint model_next.pt --n-samples 5
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from masked_model_gpu_mamba_next import CausalTimeSeriesMamba
from dataset import SyntheticSongDataset


def load_model(checkpoint_path, device):
    """Load a trained CausalTimeSeriesMamba from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = ckpt['args']

    model = CausalTimeSeriesMamba(
        feature_dim=args.get('feature_dim', 20),
        d_model=args['d_model'],
        n_layers=args['n_layers'],
        d_state=args['d_state'],
        d_conv=args.get('d_conv', 4),
        expand=args.get('expand', 2),
        d_ff=args.get('d_ff', 512),
        dropout=0.0,
        max_horizon=args.get('max_horizon', 1),
    ).to(device)

    state_dict = CausalTimeSeriesMamba.migrate_state_dict(ckpt['model_state_dict'])
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        cleaned = CausalTimeSeriesMamba.migrate_state_dict(cleaned)
        model.load_state_dict(cleaned)

    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val MSE {ckpt.get('val_loss', '?'):.4f})")
    return model, args


def _make_hidden_perturb_hook(scale, device):
    """Create a forward hook that adds noise to the last time step of a
    Mamba layer's output, perturbing the internal representation."""
    def hook(module, input, output):
        noise = torch.randn_like(output[:, -1:, :]) * scale
        output[:, -1:, :] = output[:, -1:, :] + noise
        return output
    return hook


@torch.no_grad()
def generate(model, seed, gen_len, temperature=0.0,
             perturb_interval=0, perturb_scale=0.0,
             perturb_hidden=False, perturb_layers=0, perturb_duration=1,
             device='cuda', amp_dtype=torch.bfloat16):
    """Autoregressively generate new time steps.

    Parameters
    ----------
    model : CausalTimeSeriesMamba
    seed : (1, seed_len, feature_dim) tensor
        Context window to condition on.
    gen_len : int
        Number of new time steps to generate.
    temperature : float
        Background Gaussian noise scale added to each generated step.
        0.0 = deterministic (greedy), >0 = stochastic.
    perturb_interval : int
        If >0, inject a perturbation burst every this many steps.
    perturb_scale : float
        Scale of perturbation pulses.
    perturb_hidden : bool
        If True, perturbation pulses are injected into the hidden
        representations between Mamba layers rather than the output space.
    perturb_layers : int
        Number of early layers to perturb (0 = all layers).  Setting
        this to e.g. 2 perturbs only layers 0-1 and lets deeper layers
        digest the noise so the output stays coherent.
    perturb_duration : int
        Number of consecutive steps to sustain each perturbation burst.
    device : str
    amp_dtype : torch.dtype

    Returns
    -------
    generated : (gen_len, feature_dim) numpy array
    """
    model.eval()
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    seq = seed.to(device)  # (1, seed_len, F)

    with torch.autocast(device_type='cuda', dtype=amp_dtype):
        preds = model(seq)
    next_step = preds['1'][:, -1:, :]  # (1, 1, F) — always use 1-step head

    generated = []
    perturb_remaining = 0

    for step in range(gen_len):
        if temperature > 0:
            noise = torch.randn_like(next_step) * temperature
            next_step = next_step + noise

        # Start a new perturbation burst at each interval
        if (perturb_interval > 0 and perturb_scale > 0
                and step > 0 and step % perturb_interval == 0):
            perturb_remaining = perturb_duration

        is_perturbing = perturb_remaining > 0

        if is_perturbing and not perturb_hidden:
            pulse = torch.randn_like(next_step) * perturb_scale
            next_step = next_step + pulse

        generated.append(next_step[0, 0].float().cpu().numpy())
        seq = torch.cat([seq, next_step], dim=1)

        # Register hidden-state hooks during perturbation burst
        hooks = []
        if is_perturbing and perturb_hidden:
            n = len(raw_model.layers) if perturb_layers <= 0 else perturb_layers
            for layer in raw_model.layers[:n]:
                h = layer.register_forward_hook(
                    _make_hidden_perturb_hook(perturb_scale, device))
                hooks.append(h)

        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            preds = model(seq)
        next_step = preds['1'][:, -1:, :]

        for h in hooks:
            h.remove()

        if perturb_remaining > 0:
            perturb_remaining -= 1

    return np.array(generated)  # (gen_len, feature_dim)


def visualize_generation(seed_np, gen_np, state_np, save_path,
                         feature_dim):
    """Plot seed context + generated continuation side by side."""
    seed_len = seed_np.shape[0]
    gen_len = gen_np.shape[0]
    total_len = seed_len + gen_len

    full = np.concatenate([seed_np, gen_np], axis=0)
    vmax = max(abs(full.min()), abs(full.max()))

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 5, 5],
                          width_ratios=[1, 0.02], hspace=0.12, wspace=0.03)

    # Top: state bar for seed region
    ax_top = fig.add_subplot(gs[0, 0])
    for t in range(seed_len):
        ax_top.axvspan(t, t + 1,
                       color=plt.cm.tab10(state_np[t] / 10),
                       alpha=0.8, linewidth=0)
    ax_top.axvspan(seed_len, total_len, color='#e0e0e0', alpha=0.6)
    ax_top.axvline(seed_len, color='black', linewidth=2, linestyle='--')
    ax_top.set_xlim(0, total_len)
    ax_top.set_yticks([])
    ax_top.set_ylabel('State', fontsize=9)
    ax_top.set_title(
        f'Generation: {seed_len} seed steps → {gen_len} generated steps',
        fontsize=12)
    ax_top.text(seed_len / 2, 0.5, 'seed (real)', ha='center', va='center',
                transform=ax_top.get_xaxis_transform(), fontsize=9)
    ax_top.text(seed_len + gen_len / 2, 0.5, 'generated', ha='center',
                va='center', transform=ax_top.get_xaxis_transform(),
                fontsize=9, color='#666666')
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # Middle: full heatmap
    ax_full = fig.add_subplot(gs[1, 0], sharex=ax_top)
    im = ax_full.imshow(
        full.T, aspect='auto', cmap='RdBu_r',
        vmin=-vmax, vmax=vmax,
        extent=[0, total_len, feature_dim - 0.5, -0.5],
        interpolation='none',
    )
    ax_full.axvline(seed_len, color='black', linewidth=2, linestyle='--')
    ax_full.set_ylabel('Dimension', fontsize=9)
    ax_full.set_title('Seed + Generated sequence', fontsize=10)
    plt.setp(ax_full.get_xticklabels(), visible=False)

    # Bottom: generated region only (zoomed)
    ax_gen = fig.add_subplot(gs[2, 0])
    ax_gen.imshow(
        gen_np.T, aspect='auto', cmap='RdBu_r',
        vmin=-vmax, vmax=vmax,
        extent=[0, gen_len, feature_dim - 0.5, -0.5],
        interpolation='none',
    )
    ax_gen.set_xlabel('Generated time step', fontsize=9)
    ax_gen.set_ylabel('Dimension', fontsize=9)
    ax_gen.set_title('Generated region (zoomed)', fontsize=10)

    ax_cb = fig.add_subplot(gs[1, 1])
    plt.colorbar(im, cax=ax_cb)
    fig.add_subplot(gs[0, 1]).axis('off')
    fig.add_subplot(gs[2, 1]).axis('off')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate time series from a trained Causal Mamba model.')
    parser.add_argument('--checkpoint', type=str, default='model_next.pt',
                        help='Path to trained checkpoint')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory (for seed context)')
    parser.add_argument('--seed-len', type=int, default=64,
                        help='Number of real time steps to seed the model')
    parser.add_argument('--gen-len', type=int, default=512,
                        help='Number of time steps to generate')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Background noise scale (0=deterministic)')
    parser.add_argument('--perturb-interval', type=int, default=0,
                        help='Inject a large noise pulse every N steps '
                             'to encourage state transitions (0=off)')
    parser.add_argument('--perturb-scale', type=float, default=2.0,
                        help='Scale of perturbation pulses')
    parser.add_argument('--perturb-hidden', action='store_true',
                        help='Inject perturbations into hidden Mamba layer '
                             'representations instead of the output space')
    parser.add_argument('--perturb-layers', type=int, default=0,
                        help='Number of early layers to perturb when using '
                             '--perturb-hidden (0 = all layers)')
    parser.add_argument('--perturb-duration', type=int, default=1,
                        help='Number of consecutive steps to sustain each '
                             'perturbation burst (default: 1)')
    parser.add_argument('--n-samples', type=int, default=3,
                        help='Number of samples to generate')
    parser.add_argument('--seed-offset', type=int, default=None,
                        help='Starting index into the dataset for the seed '
                             '(default: random)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-dir', type=str, default='.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = (torch.bfloat16
                 if device.type == 'cuda' and torch.cuda.is_bf16_supported()
                 else torch.float32)

    model, ckpt_args = load_model(args.checkpoint, device)

    # Load raw data for seeding
    ds = SyntheticSongDataset(
        data_dir=args.data_dir,
        seq_len=args.seed_len + args.gen_len,
        stride=args.seed_len,
        mask_ratio=0.0,
        mask_patch_size=1,
        mask_seed=0,
    )

    rng = np.random.default_rng(args.seed)
    if args.seed_offset is not None:
        offsets = [args.seed_offset]
    else:
        offsets = rng.choice(len(ds), size=min(args.n_samples, len(ds)),
                             replace=False)

    save_dir = Path(args.save_dir)
    for i, idx in enumerate(offsets):
        x_full, state_full, _ = ds[idx]

        seed = x_full[:args.seed_len]
        state_seed = state_full[:args.seed_len].numpy()
        ground_truth = x_full[args.seed_len:].numpy()

        perturb_info = ''
        if args.perturb_interval > 0:
            perturb_info = f', perturb every {args.perturb_interval} @ {args.perturb_scale}'
        print(f"\nSample {i}: seeding with {args.seed_len} steps, "
              f"generating {args.gen_len} steps "
              f"(temperature={args.temperature}{perturb_info})")

        seed_tensor = seed.unsqueeze(0).to(device)
        gen_np = generate(model, seed_tensor, args.gen_len,
                          temperature=args.temperature,
                          perturb_interval=args.perturb_interval,
                          perturb_scale=args.perturb_scale,
                          perturb_hidden=args.perturb_hidden,
                          perturb_layers=args.perturb_layers,
                          perturb_duration=args.perturb_duration,
                          device=device, amp_dtype=amp_dtype)

        # Visualize
        fname = save_dir / f'generation_{i}.png'
        visualize_generation(seed.numpy(), gen_np, state_seed, fname,
                             feature_dim=seed.shape[-1])

        # Also save a comparison with ground truth if available
        if ground_truth.shape[0] >= args.gen_len:
            gt = ground_truth[:args.gen_len]
            mse = ((gen_np - gt) ** 2).mean()
            print(f"  MSE vs ground truth: {mse:.4f}")

            fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
            vmax = max(abs(gt.min()), abs(gt.max()),
                       abs(gen_np.min()), abs(gen_np.max()))

            axes[0].imshow(gt.T, aspect='auto', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, interpolation='none')
            axes[0].set_ylabel('Dim')
            axes[0].set_title('Ground truth (continuation)')

            axes[1].imshow(gen_np.T, aspect='auto', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, interpolation='none')
            axes[1].set_ylabel('Dim')
            axes[1].set_title('Generated')

            err = gen_np - gt
            err_max = max(abs(err.min()), abs(err.max()), 1e-6)
            axes[2].imshow(err.T, aspect='auto', cmap='RdBu_r',
                           vmin=-err_max, vmax=err_max, interpolation='none')
            axes[2].set_ylabel('Dim')
            axes[2].set_xlabel('Time step')
            axes[2].set_title(f'Error  (MSE={mse:.4f})')

            plt.tight_layout()
            cmp_fname = save_dir / f'generation_vs_gt_{i}.png'
            plt.savefig(cmp_fname, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {cmp_fname}")


if __name__ == '__main__':
    main()
