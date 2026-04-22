"""
Workload Dataset for Transaction-Level OLTP Prediction

CSV format (produced by workload_generator.py):
  date, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9

Returns integer tensors (no scaling — partition IDs are not numeric magnitudes).

Tensors:
  seq_x      : (N,           10) int64  — encoder input
  seq_y      : (label_len+K, 10) int64  — decoder prompt + K target transactions
  seq_x_mark : (N,           2)  float32 — [index_norm, second_of_day_norm]
  seq_y_mark : (label_len+K, 2)  float32

Split: train 70% / val 10% / test 20% (chronological).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

SLOTS           = 10
SENTINEL        = 64
SECONDS_PER_DAY = 86400.0


class WorkloadDataset(Dataset):
    """
    Sliding-window dataset over a transaction CSV.

    Parameters
    ----------
    csv_path  : str
    flag      : 'train', 'val', or 'test'
    seq_len   : int — encoder look-back N
    label_len : int — decoder prompt (typically seq_len // 2)
    pred_len  : int — prediction horizon K
    """

    def __init__(self, csv_path, flag='train', seq_len=96,
                 label_len=48, pred_len=48):
        assert flag in ('train', 'val', 'test')
        self.seq_len   = seq_len
        self.label_len = label_len
        self.pred_len  = pred_len
        self._load(csv_path, flag)

    def _load(self, csv_path, flag):
        df = pd.read_csv(csv_path, parse_dates=['date'])
        total = len(df)

        train_end = int(total * 0.70)
        val_end   = int(total * 0.80)
        borders   = {'train': (0, train_end),
                     'val':   (train_end, val_end),
                     'test':  (val_end, total)}
        lo, hi = borders[flag]

        # Include look-back context for val/test
        lo_ctx       = max(0, lo - self.seq_len)
        self._offset = lo - lo_ctx   # rows that are context-only

        part_cols    = [f'p{i}' for i in range(SLOTS)]
        self._data   = df[part_cols].values[lo_ctx:hi].astype(np.int64)
        self._n_samples = (hi - lo) - (self.seq_len + self.pred_len) + 1

        if self._n_samples < 0:
            raise ValueError(
                f"Split '{flag}' has only {hi - lo} rows, but seq_len + pred_len = "
                f"{self.seq_len + self.pred_len}. Reduce window sizes or use more data."
            )

        # Time marks: [index_norm, second_of_day_norm]
        dates      = pd.to_datetime(df['date'].values[lo_ctx:hi])
        idx_norm   = (np.arange(lo_ctx, hi) / max(total - 1, 1)).astype(np.float32)
        sec        = (dates.hour * 3600 + dates.minute * 60 + dates.second
                      + dates.microsecond / 1e6).astype(np.float32)
        day_norm   = sec / SECONDS_PER_DAY
        self._mark = np.stack([idx_norm, day_norm], axis=1)   # (window, 2)

    def __len__(self):
        return max(0, self._n_samples)

    def __getitem__(self, idx):
        i       = idx + self._offset
        s_begin = i
        s_end   = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len

        seq_x      = torch.from_numpy(self._data[s_begin:s_end])
        seq_y      = torch.from_numpy(self._data[r_begin:r_end])
        seq_x_mark = torch.from_numpy(self._mark[s_begin:s_end])
        seq_y_mark = torch.from_numpy(self._mark[r_begin:r_end])
        return seq_x, seq_y, seq_x_mark, seq_y_mark


class WorkloadDataModule:
    """Convenience wrapper that builds DataLoaders from a single CSV."""

    def __init__(self, csv_path, seq_len=96, label_len=48, pred_len=48,
                 batch_size=32, num_workers=0):
        self.csv_path    = csv_path
        self.seq_len     = seq_len
        self.label_len   = label_len
        self.pred_len    = pred_len
        self.batch_size  = batch_size
        self.num_workers = num_workers

    def _ds(self, flag):
        return WorkloadDataset(self.csv_path, flag=flag, seq_len=self.seq_len,
                               label_len=self.label_len, pred_len=self.pred_len)

    def train_loader(self):
        # shuffle=False: each sample is a self-contained window; temporal order within
        # the batch does not matter for seq2seq training, but we keep False to preserve
        # deterministic batching for reproducibility.
        return DataLoader(self._ds('train'), batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def val_loader(self):
        return DataLoader(self._ds('val'), batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_loader(self):
        return DataLoader(self._ds('test'), batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)


def build_workload_dataset(csv_path, flag, seq_len, label_len, pred_len):
    """Drop-in factory compatible with data_factory.py."""
    return WorkloadDataset(csv_path, flag=flag, seq_len=seq_len,
                           label_len=label_len, pred_len=pred_len)
