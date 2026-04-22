"""
Seq2seq models for transaction-level partition ID prediction.

Input:  (B, N, 10) int64  — N past transactions, 10 partition slots each
Output: (B, K, 10, 64) float32 — K future transactions, 64-class logit per slot

Architecture:
  PartitionEmbedding : Embedding(65, slot_dim) per slot, concat → (B, N, d_model)
  WorkloadRNN        : 2-layer GRU  encoder → Linear head
  WorkloadLSTM       : 2-layer LSTM encoder → Linear head
"""

import math

import torch
import torch.nn as nn

NUM_PARTITIONS = 64
SENTINEL       = 64
SLOTS          = 10


class PartitionEmbedding(nn.Module):
    """
    Embed each of the 10 partition-ID slots independently, then concatenate.

    Parameters
    ----------
    d_model : int — total embedding dim; must be divisible by SLOTS (10)
    """

    def __init__(self, d_model: int = 120):
        super().__init__()
        assert d_model % SLOTS == 0, \
            f"d_model ({d_model}) must be divisible by SLOTS ({SLOTS})"
        self.slot_dim = d_model // SLOTS
        # 65 entries: IDs 0–63 plus sentinel 64
        self.embed = nn.Embedding(SENTINEL + 1, self.slot_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, SLOTS) int64
        → (B, N, d_model) float32
        """
        embedded = self.embed(x)           # (B, N, SLOTS, slot_dim)
        B, N, S, D = embedded.shape
        return embedded.reshape(B, N, S * D)


class WorkloadRNN(nn.Module):
    """
    2-layer GRU encoder → linear classification head.

    forward(seq_x) : (B, N, 10) int64 → (B, K, 10, 64) float32
    """

    def __init__(
        self,
        d_model: int     = 120,
        hidden_size: int = 128,
        num_layers: int  = 2,
        pred_len: int    = 48,
        dropout: float   = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.embed    = PartitionEmbedding(d_model)
        self.rnn      = nn.GRU(
            input_size  = d_model,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.head = nn.Linear(hidden_size, pred_len * SLOTS * NUM_PARTITIONS)

    def forward(self, seq_x: torch.Tensor) -> torch.Tensor:
        x      = self.embed(seq_x)              # (B, N, d_model)
        _, h_n = self.rnn(x)                    # h_n: (layers, B, hidden)
        h_last = h_n[-1]                        # (B, hidden)
        logits = self.head(h_last)              # (B, K*10*64)
        B      = seq_x.shape[0]
        return logits.view(B, self.pred_len, SLOTS, NUM_PARTITIONS)


class WorkloadLSTM(nn.Module):
    """
    2-layer LSTM encoder → linear classification head.

    forward(seq_x) : (B, N, 10) int64 → (B, K, 10, 64) float32
    """

    def __init__(
        self,
        d_model: int     = 120,
        hidden_size: int = 128,
        num_layers: int  = 2,
        pred_len: int    = 48,
        dropout: float   = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.embed    = PartitionEmbedding(d_model)
        self.lstm     = nn.LSTM(
            input_size  = d_model,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.head = nn.Linear(hidden_size, pred_len * SLOTS * NUM_PARTITIONS)

    def forward(self, seq_x: torch.Tensor) -> torch.Tensor:
        x           = self.embed(seq_x)         # (B, N, d_model)
        _, (h_n, _) = self.lstm(x)              # h_n: (layers, B, hidden)
        h_last      = h_n[-1]                   # (B, hidden)
        logits      = self.head(h_last)         # (B, K*10*64)
        B           = seq_x.shape[0]
        return logits.view(B, self.pred_len, SLOTS, NUM_PARTITIONS)


class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding; no learned parameters."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:, :length]                    # (1, L, d_model)


class WorkloadInformer(nn.Module):
    """
    Encoder-decoder Transformer for transaction-level partition ID prediction.

    Uses standard multi-head self/cross-attention (PyTorch built-in).
    Advantage over RNN/LSTM: full attention over all N encoder positions
    rather than only the final hidden state.

    Decoder input is constructed internally:
        last label_len tokens of seq_x  (real history)
        + pred_len sentinel placeholders (value 64, "unknown future")

    forward(seq_x) : (B, N, 10) int64 → (B, K, 10, 64) float32
    """

    def __init__(
        self,
        d_model: int    = 160,
        n_heads: int    = 8,
        e_layers: int   = 2,
        d_layers: int   = 1,
        d_ff: int       = 320,
        dropout: float  = 0.1,
        pred_len: int   = 48,
        label_len: int  = 48,
    ):
        super().__init__()
        assert d_model % SLOTS    == 0, \
            f"d_model ({d_model}) must be divisible by SLOTS ({SLOTS})"
        assert d_model % n_heads  == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.pred_len  = pred_len
        self.label_len = label_len

        self.enc_embedding = PartitionEmbedding(d_model)
        self.dec_embedding = PartitionEmbedding(d_model)
        self.pos_enc       = _SinusoidalPE(d_model)
        self.drop          = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=e_layers,
                                              enable_nested_tensor=False)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=d_layers)

        # Per-timestep projection: d_model → 10 classes × 64 partitions
        self.head = nn.Linear(d_model, SLOTS * NUM_PARTITIONS)

    def forward(self, seq_x: torch.Tensor) -> torch.Tensor:
        """
        seq_x : (B, N, SLOTS) int64
        → (B, pred_len, SLOTS, NUM_PARTITIONS) float32
        """
        B = seq_x.shape[0]

        # ── decoder input ──────────────────────────────────────────────────────
        label_part    = seq_x[:, -self.label_len:, :]                        # (B, label_len, 10)
        sentinel_part = torch.full(
            (B, self.pred_len, SLOTS), SENTINEL,
            dtype=torch.long, device=seq_x.device,
        )                                                                     # (B, pred_len, 10)
        dec_inp = torch.cat([label_part, sentinel_part], dim=1)              # (B, label_len+K, 10)

        # ── embed + positional encoding ────────────────────────────────────────
        enc_emb = self.drop(
            self.enc_embedding(seq_x)    + self.pos_enc(seq_x.shape[1])
        )                                                                     # (B, N, d_model)
        dec_emb = self.drop(
            self.dec_embedding(dec_inp)  + self.pos_enc(dec_inp.shape[1])
        )                                                                     # (B, label_len+K, d_model)

        # ── encode ─────────────────────────────────────────────────────────────
        memory = self.encoder(enc_emb)                                        # (B, N, d_model)

        # ── decode (causal self-attention) ─────────────────────────────────────
        tgt_len    = dec_inp.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=seq_x.device
        )                                                                     # (tgt_len, tgt_len)
        dec_out = self.decoder(
            dec_emb, memory,
            tgt_mask=causal_mask, tgt_is_causal=True,
        )                                                                     # (B, label_len+K, d_model)

        # ── project last pred_len steps ────────────────────────────────────────
        logits = self.head(dec_out[:, -self.pred_len:, :])                   # (B, K, SLOTS*64)
        return logits.view(B, self.pred_len, SLOTS, NUM_PARTITIONS)
