import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import PartitionEmbedding, WorkloadRNN, WorkloadLSTM, WorkloadInformer

BATCH  = 4
N      = 96
K      = 48
SLOTS  = 10
D      = 120    # d_model — must be divisible by SLOTS


def _make_input(batch=BATCH, seq=N):
    return torch.randint(0, 65, (batch, seq, SLOTS), dtype=torch.int64)


def test_partition_embedding_shape():
    embed = PartitionEmbedding(d_model=D)
    out = embed(_make_input())
    assert out.shape == (BATCH, N, D), f"Expected ({BATCH}, {N}, {D}), got {out.shape}"


def test_partition_embedding_dtype():
    embed = PartitionEmbedding(d_model=D)
    out = embed(_make_input())
    assert out.dtype == torch.float32


def test_partition_embedding_sentinel_ok():
    embed = PartitionEmbedding(d_model=D)
    x = torch.full((1, 1, SLOTS), 64, dtype=torch.int64)
    assert embed(x).shape == (1, 1, D)


def test_workload_rnn_output_shape():
    model = WorkloadRNN(d_model=D, hidden_size=64, num_layers=2, pred_len=K)
    assert model(_make_input()).shape == (BATCH, K, SLOTS, 64)


def test_workload_rnn_output_dtype():
    model = WorkloadRNN(d_model=D, hidden_size=64, num_layers=2, pred_len=K)
    assert model(_make_input()).dtype == torch.float32


def test_workload_rnn_no_nan():
    model = WorkloadRNN(d_model=D, hidden_size=64, num_layers=2, pred_len=K)
    assert not model(_make_input()).isnan().any()


def test_workload_rnn_backward():
    import torch.nn.functional as F
    model  = WorkloadRNN(d_model=D, hidden_size=64, num_layers=2, pred_len=K)
    logits = model(_make_input())
    target = torch.randint(0, 64, (BATCH, K, SLOTS))
    F.cross_entropy(logits.reshape(-1, 64), target.reshape(-1)).backward()


def test_workload_lstm_output_shape():
    model = WorkloadLSTM(d_model=D, hidden_size=64, num_layers=2, pred_len=K)
    assert model(_make_input()).shape == (BATCH, K, SLOTS, 64)


def test_workload_lstm_output_dtype():
    model = WorkloadLSTM(d_model=D, hidden_size=64, num_layers=2, pred_len=K)
    assert model(_make_input()).dtype == torch.float32


def test_workload_lstm_no_nan():
    model = WorkloadLSTM(d_model=D, hidden_size=64, num_layers=2, pred_len=K)
    assert not model(_make_input()).isnan().any()


def test_workload_lstm_backward():
    import torch.nn.functional as F
    model  = WorkloadLSTM(d_model=D, hidden_size=64, num_layers=2, pred_len=K)
    logits = model(_make_input())
    target = torch.randint(0, 64, (BATCH, K, SLOTS))
    F.cross_entropy(logits.reshape(-1, 64), target.reshape(-1)).backward()


def test_different_batch_sizes():
    rnn  = WorkloadRNN(d_model=D, hidden_size=64, num_layers=2, pred_len=K)
    lstm = WorkloadLSTM(d_model=D, hidden_size=64, num_layers=2, pred_len=K)
    for b in [1, 8, 32]:
        x = _make_input(batch=b)
        assert rnn(x).shape  == (b, K, SLOTS, 64)
        assert lstm(x).shape == (b, K, SLOTS, 64)


# ── WorkloadInformer ──────────────────────────────────────────────────────────

D_INF    = 160   # d_model for Informer (divisible by SLOTS=10 and n_heads=8)
LABEL    = K     # label_len == pred_len for tests


def _make_informer(pred_len=K, label_len=LABEL):
    return WorkloadInformer(
        d_model=D_INF, n_heads=8, e_layers=1, d_layers=1,
        d_ff=160, pred_len=pred_len, label_len=label_len, dropout=0.0,
    )


def test_workload_informer_output_shape():
    model = _make_informer()
    assert model(_make_input()).shape == (BATCH, K, SLOTS, 64)


def test_workload_informer_output_dtype():
    model = _make_informer()
    assert model(_make_input()).dtype == torch.float32


def test_workload_informer_no_nan():
    model = _make_informer()
    assert not model(_make_input()).isnan().any()


def test_workload_informer_backward():
    import torch.nn.functional as F
    model  = _make_informer()
    logits = model(_make_input())
    target = torch.randint(0, 64, (BATCH, K, SLOTS))
    F.cross_entropy(logits.reshape(-1, 64), target.reshape(-1)).backward()


def test_workload_informer_label_len_shorter_than_seq():
    """Informer should work when label_len < N (normal use-case)."""
    model = _make_informer(pred_len=K, label_len=K // 2)
    assert model(_make_input(seq=N)).shape == (BATCH, K, SLOTS, 64)


def test_workload_informer_different_batch_sizes():
    model = _make_informer()
    for b in [1, 8, 32]:
        assert model(_make_input(batch=b)).shape == (b, K, SLOTS, 64)
