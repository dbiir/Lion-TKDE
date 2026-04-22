"""
Train RNN and LSTM baselines on all four workload modes and print a
side-by-side comparison table.

Usage
-----
# Quick demo (first 5000 rows of each CSV, ~2 min total):
python train_baselines.py

# Full dataset (360k rows, several hours):
python train_baselines.py --n-rows 0

# Single mode:
python train_baselines.py --modes A

Options
-------
--modes       A B C D  (default: all four)
--n-rows N    rows from each CSV (0 = all, default 5000)
--seq-len     encoder window N  (default 96)
--pred-len    prediction horizon K  (default 48)
--epochs      max training epochs   (default 10)
--batch-size  (default 256)
--lr          Adam learning rate    (default 1e-3)
--patience    early stopping patience (default 3)
--seed        (default 42)
--out-dir     directory to save model checkpoints (default: checkpoints/)
--data-dir    directory containing workload CSVs  (default: data)
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from models import WorkloadRNN, WorkloadLSTM, WorkloadInformer
from workload_dataset import WorkloadDataset
from evaluate import slot_accuracy, set_accuracy, top3_accuracy, cross_entropy_loss

NUM_PARTITIONS = 64
SENTINEL       = 64
SLOTS          = 10

MODE_NAMES = {
    'A': 'harmonic_superposition',
    'B': 'phase_cascade',
    'C': 'frequency_modulated',
    'D': 'regime_with_precursors',
}


# ── Loss ──────────────────────────────────────────────────────────────────────

def masked_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy averaged over non-sentinel slots only.

    Parameters
    ----------
    logits : (B, K, 10, 64) float32
    target : (B, K, 10)     int64

    Returns
    -------
    scalar tensor (differentiable)
    """
    mask        = (target != SENTINEL)
    flat_logits = logits.reshape(-1, NUM_PARTITIONS)
    # Clamp sentinel indices so CE doesn't crash — masked rows are excluded from loss
    flat_target = target.reshape(-1).clamp(0, NUM_PARTITIONS - 1)
    flat_mask   = mask.reshape(-1)
    if flat_mask.sum() == 0:
        return logits.sum() * 0.0   # differentiable zero
    return F.cross_entropy(flat_logits[flat_mask], flat_target[flat_mask])


# ── Dataset helper ────────────────────────────────────────────────────────────

def load_loaders(csv_path: str, n_rows: int, seq_len: int,
                 label_len: int, pred_len: int, batch_size: int):
    """
    Build train/val/test DataLoaders, optionally using only the first n_rows.
    n_rows=0 means use all rows.
    """
    if n_rows > 0:
        df  = pd.read_csv(csv_path, nrows=n_rows)
        tmp = csv_path + f".tmp{n_rows}.csv"
        df.to_csv(tmp, index=False)
        src = tmp
    else:
        src = csv_path

    def _dl(flag, shuffle, sl, ll, pl):
        ds = WorkloadDataset(src, flag=flag, seq_len=sl,
                             label_len=ll, pred_len=pl)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, drop_last=False)

    # Auto-reduce window sizes if data is too small for the requested seq_len/pred_len.
    # The val split is 10% of n_rows (smallest split); we need seq_len + pred_len < that.
    eff_sl, eff_ll, eff_pl = seq_len, label_len, pred_len
    if n_rows > 0:
        val_rows = int(n_rows * 0.10)
        min_rows = seq_len + pred_len
        if val_rows < min_rows:
            # Scale down proportionally keeping seq_len : pred_len ratio
            ratio   = seq_len / (seq_len + pred_len)
            budget  = max(val_rows - 1, 2)
            eff_pl  = max(1, int(budget * (1 - ratio)))
            eff_sl  = max(1, budget - eff_pl)
            eff_ll  = eff_sl // 2
            print(f"  [INFO] n_rows={n_rows} is small; auto-reduced "
                  f"seq_len={seq_len}→{eff_sl}, pred_len={pred_len}→{eff_pl} "
                  f"to fit val split ({val_rows} rows).")

    train_dl = _dl('train', shuffle=True,  sl=eff_sl, ll=eff_ll, pl=eff_pl)
    val_dl   = _dl('val',   shuffle=False, sl=eff_sl, ll=eff_ll, pl=eff_pl)
    test_dl  = _dl('test',  shuffle=False, sl=eff_sl, ll=eff_ll, pl=eff_pl)

    if n_rows > 0 and os.path.exists(tmp):
        os.unlink(tmp)

    return train_dl, val_dl, test_dl, eff_sl, eff_pl


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        seq_x, seq_y, _, _ = batch
        seq_x  = seq_x.to(device)
        # seq_y is (B, label_len+K, 10); take only the K future targets
        label_len = seq_y.shape[1] - model.pred_len
        target = seq_y[:, label_len:, :].to(device)   # (B, K, 10)

        optimizer.zero_grad()
        logits = model(seq_x)                          # (B, K, 10, 64)
        loss   = masked_ce_loss(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate_loss(model, loader, device):
    """Return average masked CE loss with no gradient tracking."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            seq_x, seq_y, _, _ = batch
            seq_x     = seq_x.to(device)
            label_len = seq_y.shape[1] - model.pred_len
            target    = seq_y[:, label_len:, :].to(device)
            logits    = model(seq_x)
            total_loss += masked_ce_loss(logits, target).item()
    return total_loss / max(len(loader), 1)


def train_model(model, train_dl, val_dl, max_epochs, lr, patience, device):
    """Train with early stopping; return best val loss."""
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    best_val   = float('inf')
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        tr = train_one_epoch(model, train_dl, optimizer, device)
        vl = evaluate_loss(model, val_dl, device)
        print(f"    epoch {epoch:2d}  train={tr:.4f}  val={vl:.4f}")

        if vl < best_val - 1e-4:
            best_val   = vl
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    early stop at epoch {epoch}")
                break

    return best_val


# ── Evaluation ────────────────────────────────────────────────────────────────

def collect_predictions(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            seq_x, seq_y, _, _ = batch
            seq_x     = seq_x.to(device)
            label_len = seq_y.shape[1] - model.pred_len
            target    = seq_y[:, label_len:, :].numpy()  # (B, K, 10)
            logits    = model(seq_x).cpu().numpy()       # (B, K, 10, 64)
            all_logits.append(logits)
            all_targets.append(target)
    if not all_logits:
        # Empty test split — return zero-sample arrays with correct shape
        return (np.empty((0, model.pred_len, SLOTS, NUM_PARTITIONS), dtype=np.float32),
                np.empty((0, model.pred_len, SLOTS), dtype=np.int64))
    # NOTE: for --n-rows 0 (full 360k CSV), peak RAM is ~8.6 GB; reduce --n-rows if needed
    return np.concatenate(all_logits), np.concatenate(all_targets)


def compute_metrics(logits_arr, targets_arr):
    n = len(logits_arr)
    if n == 0:
        return {'slot_acc': float('nan'), 'set_acc': float('nan'),
                'top3_acc': float('nan'), 'ce_loss': float('nan')}
    pred_ids = np.argmax(logits_arr, axis=-1)   # (n, K, 10)

    slot_accs = [slot_accuracy(pred_ids[i], targets_arr[i]) for i in range(n)]
    set_accs  = [set_accuracy(pred_ids[i],  targets_arr[i]) for i in range(n)]
    top3_accs = [top3_accuracy(logits_arr[i], targets_arr[i]) for i in range(n)]
    ce        = cross_entropy_loss(
        logits_arr.reshape(-1, SLOTS, NUM_PARTITIONS),
        targets_arr.reshape(-1, SLOTS),
    )
    return {
        'slot_acc':  float(np.nanmean(slot_accs)),
        'set_acc':   float(np.nanmean(set_accs)),
        'top3_acc':  float(np.nanmean(top3_accs)),
        'ce_loss':   ce,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train RNN and LSTM baselines')
    parser.add_argument('--modes',      nargs='+', default=['A', 'B', 'C', 'D'])
    parser.add_argument('--n-rows',     type=int,   default=5000,
                        help='rows per CSV (0 = all 360k)')
    parser.add_argument('--seq-len',    type=int,   default=96)
    parser.add_argument('--pred-len',   type=int,   default=48)
    parser.add_argument('--epochs',     type=int,   default=10)
    parser.add_argument('--batch-size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--patience',   type=int,   default=3)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--out-dir',    default='checkpoints')
    parser.add_argument('--data-dir',   default='data')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_len = args.seq_len // 2

    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    eff_seq_len = args.seq_len
    eff_pred_len = args.pred_len

    # Each entry: (name, class, kwargs_fn(eff_seq_len, eff_pred_len) → dict)
    MODEL_CLASSES = [
        ('RNN',      WorkloadRNN,
         lambda sl, pl: dict(d_model=120, hidden_size=128, num_layers=2,
                             pred_len=pl, dropout=0.1)),
        ('LSTM',     WorkloadLSTM,
         lambda sl, pl: dict(d_model=120, hidden_size=128, num_layers=2,
                             pred_len=pl, dropout=0.1)),
        ('Informer', WorkloadInformer,
         lambda sl, pl: dict(d_model=160, n_heads=8, e_layers=2, d_layers=1,
                             d_ff=320, pred_len=pl, label_len=max(1, sl // 2),
                             dropout=0.1)),
    ]

    for mode in args.modes:
        csv_name = f"workload_{MODE_NAMES[mode]}.csv"
        csv_path = os.path.join(args.data_dir, csv_name)
        if not os.path.exists(csv_path):
            print(f"[WARN] {csv_path} not found — skipping mode {mode}")
            continue

        print(f"\n{'='*60}")
        print(f"Mode {mode} ({MODE_NAMES[mode]})")
        print(f"  CSV: {csv_path}  n_rows={args.n_rows or 'all'}")
        print(f"{'='*60}")

        train_dl, val_dl, test_dl, eff_seq_len, eff_pred_len = load_loaders(
            csv_path, args.n_rows, args.seq_len,
            label_len, args.pred_len, args.batch_size,
        )
        if len(train_dl.dataset) == 0 or len(val_dl.dataset) == 0 or len(test_dl.dataset) == 0:
            print(f"[WARN] Mode {mode}: n_rows={args.n_rows} is too small even after auto-scaling "
                  f"(train={len(train_dl.dataset)}, val={len(val_dl.dataset)}, "
                  f"test={len(test_dl.dataset)} samples). Skipping.")
            continue
        print(f"  train batches={len(train_dl)}  val={len(val_dl)}  test={len(test_dl)}")

        for model_name, ModelClass, kwargs_fn in MODEL_CLASSES:
            print(f"\n  --- {model_name} ---")
            t0    = time.time()
            model = ModelClass(
                **kwargs_fn(eff_seq_len, eff_pred_len)
            ).to(device)

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    parameters: {n_params:,}")

            train_model(model, train_dl, val_dl,
                        args.epochs, args.lr, args.patience, device)

            ckpt = os.path.join(args.out_dir, f"mode{mode}_{model_name.lower()}.pt")
            torch.save(model.state_dict(), ckpt)

            logits_arr, targets_arr = collect_predictions(model, test_dl, device)
            metrics = compute_metrics(logits_arr, targets_arr)
            results[(mode, model_name)] = metrics

            elapsed = time.time() - t0
            print(f"    slot_acc={metrics['slot_acc']:.4f}  "
                  f"set_acc={metrics['set_acc']:.4f}  "
                  f"top3_acc={metrics['top3_acc']:.4f}  "
                  f"ce_loss={metrics['ce_loss']:.4f}  "
                  f"({elapsed:.0f}s)")

    # ── Comparison table ──────────────────────────────────────────────────────
    model_names = ' vs '.join(name for name, *_ in MODEL_CLASSES)
    print(f"\n\n{'='*75}")
    print(f"RESULTS: {model_names} across workload modes")
    eff_note = ""
    if eff_seq_len != args.seq_len or eff_pred_len != args.pred_len:
        eff_note = f"  (requested: seq_len={args.seq_len} pred_len={args.pred_len})"
    print(f"  seq_len={eff_seq_len}  pred_len={eff_pred_len}  "
          f"n_rows={args.n_rows or 360000}{eff_note}")
    print(f"{'='*75}")
    print(f"{'Mode':<6} {'Model':<6} {'SlotAcc':>8} {'SetAcc':>8} "
          f"{'Top3Acc':>9} {'CELoss':>8}")
    print('-' * 75)

    for mode in args.modes:
        for model_name, _, _kw in MODEL_CLASSES:
            key = (mode, model_name)
            if key not in results:
                continue
            m = results[key]
            print(f"{mode:<6} {model_name:<6} "
                  f"{m['slot_acc']:>8.4f} {m['set_acc']:>8.4f} "
                  f"{m['top3_acc']:>9.4f} {m['ce_loss']:>8.4f}")
        print()

    print('=' * 75)


if __name__ == '__main__':
    main()
