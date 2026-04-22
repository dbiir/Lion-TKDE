"""
End-to-end example: workload data generation → GRU training → evaluation

Runs in under a minute on CPU with small parameters.
Demonstrates the full pipeline:
  1. Generate synthetic workload data (Mode A – Stable Hotspot)
  2. Build train/val/test DataLoaders via WorkloadDataModule
  3. Train a small GRU model (as a baseline)
  4. Save predictions as .npy files
  5. Run the WorkloadEvaluator to print all metrics

Usage:
  python example.py

No GPU required. Adjust N_INTERVALS / N_EPOCHS for speed vs quality.
"""

import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# ── Local modules ────────────────────────────────────────────────────────────
from workload_generator import generate_mode_A, NUM_PARTITIONS
from workload_dataset import WorkloadDataModule
from evaluate import WorkloadEvaluator, plot_loss_curves, plot_hotspot_heatmap

# ── Hyper-parameters (kept small for a quick demo) ───────────────────────────
N_INTERVALS  = 600      # number of scheduling intervals to generate
SEQ_LEN      = 30       # look-back window W
LABEL_LEN    = 15       # decoder prompt (Informer-style overlap)
PRED_LEN     = 1        # next-interval prediction
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
BATCH_SIZE   = 32
N_EPOCHS     = 20
LR           = 1e-3
PATIENCE     = 5        # early-stopping patience


# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate workload data
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Step 1 – Generate synthetic workload (Mode A: Stable Hotspot)")
print("=" * 60)

tmpdir = tempfile.mkdtemp()
csv_path = os.path.join(tmpdir, "workload_stable_hotspot.csv")
df = generate_mode_A(n_intervals=N_INTERVALS, seed=42)
df.to_csv(csv_path, index=False)
print(f"  Generated {len(df)} intervals × {len(df.columns)-1} features")
print(f"  Saved to {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 2 – Build DataLoaders (70/10/20 chronological split)")
print("=" * 60)

dm = WorkloadDataModule(
    csv_path,
    seq_len=SEQ_LEN, label_len=LABEL_LEN, pred_len=PRED_LEN,
    batch_size=BATCH_SIZE,
)
train_loader, val_loader, test_loader = dm.get_loaders()
n_feat = dm.n_features

print(f"  n_features   = {n_feat}")
print(f"  train batches= {len(train_loader)}")
print(f"  val   batches= {len(val_loader)}")
print(f"  test  batches= {len(test_loader)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Define a small GRU model
# ─────────────────────────────────────────────────────────────────────────────

class SmallGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=0.1 if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, input_size * pred_len)
        self.pred_len   = pred_len
        self.input_size = input_size

    def forward(self, x):
        # x: (B, seq_len, n_feat)
        out, _ = self.gru(x)
        last = out[:, -1, :]                      # (B, hidden_size)
        pred = self.head(last)                     # (B, n_feat * pred_len)
        return pred.view(-1, self.pred_len, self.input_size)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallGRU(n_feat, HIDDEN_SIZE, NUM_LAYERS, PRED_LEN).to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training loop with early stopping
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 3 – Train GRU model")
print("=" * 60)

train_losses, val_losses = [], []
best_val, best_state, patience_counter = float("inf"), None, 0

for epoch in range(1, N_EPOCHS + 1):
    # ── train ──
    model.train()
    epoch_loss = 0.0
    for seq_x, seq_y, _, _ in train_loader:
        # seq_x: (B, seq_len, n_feat)
        # seq_y: (B, label_len + pred_len, n_feat)  → target = last pred_len steps
        seq_x = seq_x.to(device)
        target = seq_y[:, -PRED_LEN:, :].to(device)   # (B, pred_len, n_feat)

        optimizer.zero_grad()
        pred = model(seq_x)                            # (B, pred_len, n_feat)
        loss = criterion(pred, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)

    # ── validate ──
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for seq_x, seq_y, _, _ in val_loader:
            seq_x  = seq_x.to(device)
            target = seq_y[:, -PRED_LEN:, :].to(device)
            pred   = model(seq_x)
            val_loss += criterion(pred, target).item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"  Epoch {epoch:3d}/{N_EPOCHS}  train={train_loss:.5f}  val={val_loss:.5f}")

    # ── early stopping ──
    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (best val={best_val:.5f})")
            break

model.load_state_dict(best_state)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Run inference on test set and collect predictions
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 4 – Collect test predictions")
print("=" * 60)

model.eval()
all_preds, all_trues = [], []

with torch.no_grad():
    for seq_x, seq_y, _, _ in test_loader:
        seq_x  = seq_x.to(device)
        target = seq_y[:, -PRED_LEN:, :]   # keep on CPU for numpy

        pred = model(seq_x).cpu()
        all_preds.append(pred.numpy())
        all_trues.append(target.numpy())

preds = np.concatenate(all_preds, axis=0)   # (n_test, pred_len, n_feat)
trues = np.concatenate(all_trues, axis=0)

# Inverse-transform to original scale for interpretable metrics
test_ds = dm.test_dataset
preds_orig = test_ds.inverse_transform(preds[:, 0, :])
trues_orig = test_ds.inverse_transform(trues[:, 0, :])

preds_path = os.path.join(tmpdir, "gru_stable_hotspot_preds.npy")
trues_path = os.path.join(tmpdir, "gru_stable_hotspot_trues.npy")
np.save(preds_path, preds_orig)
np.save(trues_path, trues_orig)
print(f"  preds shape: {preds_orig.shape}")
print(f"  Saved → {preds_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluate
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 5 – Evaluate predictions")
print("=" * 60)

ev = WorkloadEvaluator(
    preds_orig, trues_orig,
    train_losses=np.array(train_losses),
    val_losses=np.array(val_losses),
)
report = ev.full_report(k=5)

print("\n  ── Prediction Accuracy ──────────────────────")
print(f"  MAE  : {report['MAE']:.4f}")
print(f"  RMSE : {report['RMSE']:.4f}")
print(f"  MAPE : {report['MAPE']:.2f}%")

print("\n  ── Hotspot Forecasting (top-5 partitions) ───")
print(f"  Precision@5 : {report['Precision@5']:.3f}")
print(f"  Recall@5    : {report['Recall@5']:.3f}")

print("\n  ── Distribution Similarity ──────────────────")
print(f"  JS Divergence : {report['JS_Divergence']:.4f}")

if "best_val_epoch" in report:
    print("\n  ── Overfitting Analysis ─────────────────────")
    print(f"  Best val epoch     : {report['best_val_epoch']}")
    print(f"  Final train loss   : {report['final_train_loss']:.5f}")
    print(f"  Final val   loss   : {report['final_val_loss']:.5f}")
    print(f"  Train-val gap      : {report['train_val_loss_gap']:.5f}")
    print(f"  Overfit flag       : {report['overfit_flag']}")

# Optional: save plots
plots_dir = os.path.join(tmpdir, "plots")
os.makedirs(plots_dir, exist_ok=True)
plot_loss_curves(
    np.array(train_losses), np.array(val_losses),
    title="GRU – Stable Hotspot",
    out_path=os.path.join(plots_dir, "loss_curves.png"),
)
plot_hotspot_heatmap(
    preds_orig, trues_orig,
    out_path=os.path.join(plots_dir, "hotspot_heatmap.png"),
)
print(f"\n  Plots saved to {plots_dir}/")
print("\nDone.")
