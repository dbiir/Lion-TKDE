"""
Evaluation for transaction-level partition prediction.

All metrics ignore sentinel slots (SENTINEL = 64).

Functions
---------
slot_accuracy(pred, target)         → float
set_accuracy(pred, target)          → float
top3_accuracy(logits, target)       → float
cross_entropy_loss(logits, target)  → float
frequency_prior_baseline(targets)   → callable
evaluate_grid(model_fn, csv_path, n_values, k_values) → dict
"""

import numpy as np
import torch
import torch.nn.functional as F
from workload_dataset import WorkloadDataModule

SENTINEL     = 64
SLOTS        = 10
N_PARTITIONS = 64


def slot_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Fraction of non-sentinel slots where pred == target.

    Parameters
    ----------
    pred, target : (K, 10) int arrays
    """
    mask = target != SENTINEL
    if not mask.any():
        return float('nan')
    return float((pred[mask] == target[mask]).mean())


def set_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Fraction of K transactions where the predicted partition set exactly
    matches the ground-truth set (sentinel slots excluded).

    Parameters
    ----------
    pred, target : (K, 10) int arrays
    """
    K = target.shape[0]
    correct = 0
    counted = 0
    for k in range(K):
        t_set = set(target[k][target[k] != SENTINEL].tolist())
        if not t_set:   # all-sentinel row — skip (consistent with slot_accuracy nan-guard)
            continue
        counted += 1
        if set(pred[k][pred[k] != SENTINEL].tolist()) == t_set:
            correct += 1
    return float('nan') if counted == 0 else correct / counted


def top3_accuracy(logits: np.ndarray, target: np.ndarray) -> float:
    """
    Fraction of non-sentinel slots where the true partition ID appears
    in the top-3 predicted classes.

    Parameters
    ----------
    logits : (K, 10, 64) float array of raw scores
    target : (K, 10)     int array
    """
    top3 = np.argsort(logits, axis=-1)[..., -3:]   # (K, 10, 3)
    mask = target != SENTINEL
    if not mask.any():
        return float('nan')
    in_top3 = (top3 == target[..., np.newaxis]).any(axis=-1)   # (K, 10)
    return float(in_top3[mask].mean())


def cross_entropy_loss(logits: np.ndarray, target: np.ndarray) -> float:
    """
    Mean cross-entropy over non-sentinel slots.

    Parameters
    ----------
    logits : (K, 10, 64) float array
    target : (K, 10)     int array
    """
    mask        = (target != SENTINEL).flatten()
    flat_logits = torch.tensor(logits.reshape(-1, N_PARTITIONS)[mask], dtype=torch.float32)
    flat_target = torch.tensor(target.flatten()[mask], dtype=torch.long)
    return float(F.cross_entropy(flat_logits, flat_target).item())


def frequency_prior_baseline(train_targets: np.ndarray):
    """
    Build a predictor that always outputs the top-10 most frequent
    partition IDs from training data (sorted ascending, sentinel-padded).

    Parameters
    ----------
    train_targets : (M, 10) int array from training split

    Returns
    -------
    predict : callable  (target: np.ndarray (K, 10)) → pred: np.ndarray (K, 10)
    """
    flat  = train_targets.flatten()
    flat  = flat[flat != SENTINEL]
    ids, counts = np.unique(flat, return_counts=True)
    top10 = np.sort(ids[np.argsort(-counts)[:SLOTS]])
    row   = np.full(SLOTS, SENTINEL, dtype=np.int32)
    row[:len(top10)] = top10

    def predict(target: np.ndarray) -> np.ndarray:
        return np.broadcast_to(row, target.shape).copy()

    return predict


def evaluate_grid(
    model_fn,
    csv_path: str,
    n_values=(96, 192, 336, 720),
    k_values=(48, 96, 192, 336),
    label_len_fn=None,
    batch_size: int = 32,
) -> dict:
    """
    Evaluate a model across the 4×4 (N, K) benchmark grid.

    Parameters
    ----------
    model_fn    : callable (seq_x, seq_x_mark, seq_y_mark) → logits tensor
                  seq_x      : (B, N, 10)    int64
                  seq_x_mark : (B, N, 2)     float32
                  seq_y_mark : (B, label+K, 2) float32
                  returns    : (B, K, 10, 64) float32
    csv_path    : str
    n_values    : sequence of N (encoder window sizes)
    k_values    : sequence of K (prediction horizons)
    label_len_fn: callable N → label_len; defaults to N // 2
    batch_size  : int

    Returns
    -------
    dict keyed (N, K) → {'slot_acc', 'set_acc', 'top3_acc', 'ce_loss'}
    """
    if label_len_fn is None:
        label_len_fn = lambda n: n // 2

    results = {}
    for N in n_values:
        for K in k_values:
            L  = label_len_fn(N)
            dm = WorkloadDataModule(csv_path, seq_len=N, label_len=L,
                                    pred_len=K, batch_size=batch_size)
            all_logits  = []
            all_targets = []

            for seq_x, seq_y, seq_x_mark, seq_y_mark in dm.test_loader():
                with torch.no_grad():
                    logits = model_fn(seq_x, seq_x_mark, seq_y_mark)  # (B, K, 10, 64)
                target = seq_y[:, L:, :].numpy()                       # (B, K, 10)
                all_logits.append(logits.cpu().numpy())
                all_targets.append(target)

            logits_arr  = np.concatenate(all_logits,  axis=0)   # (N_test, K, 10, 64)
            targets_arr = np.concatenate(all_targets, axis=0)   # (N_test, K, 10)
            pred_ids    = np.argmax(logits_arr, axis=-1)         # (N_test, K, 10)
            n_samples   = len(pred_ids)

            results[(N, K)] = {
                'slot_acc': float(np.nanmean([
                    slot_accuracy(pred_ids[i], targets_arr[i])
                    for i in range(n_samples)])),
                'set_acc':  float(np.nanmean([
                    set_accuracy(pred_ids[i], targets_arr[i])
                    for i in range(n_samples)])),
                'top3_acc': float(np.nanmean([
                    top3_accuracy(logits_arr[i], targets_arr[i])
                    for i in range(n_samples)])),
                'ce_loss':  cross_entropy_loss(
                    logits_arr.reshape(-1, SLOTS, N_PARTITIONS),
                    targets_arr.reshape(-1, SLOTS)),
            }
    return results
