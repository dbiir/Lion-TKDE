# Transaction-Sequence Workload Redesign — Design Spec
**Date:** 2026-04-21
**Scope:** Replace the interval-aggregated workload design with a transaction-level sequence design. Each data point is one transaction represented as a sorted 10-dim partition ID vector. The task is: given N past transactions, predict K future transactions. Goal unchanged — prove Informer outperforms LSTM and RNN.

---

## 1. Goal

Generate synthetic OLTP transaction sequences that, when used to train and evaluate Informer vs LSTM vs RNN:

- Produce measurably higher **slot accuracy** and **set accuracy** for Informer than for LSTM/RNN
- Do so for principled reasons rooted in Transformer attention's known strengths (long-range dependency, cross-position attention)
- Support comparison across a **4×4 grid** of (N, K) settings
- Are reproducible via a fixed random seed

---

## 2. Data Schema

### CSV format (one row = one transaction)

```
date, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9
```

| Column | Type | Description |
|--------|------|-------------|
| `date` | ISO timestamp | Wall-clock time at 50 tx/s resolution (e.g. `2026-01-01 00:00:00.020`) |
| `p0`–`p9` | int | Sorted partition IDs in [0, 63]; unused slots filled with sentinel **64** |

**Example:** A transaction accessing partitions {5, 31, 42}:
```
2026-01-01 00:00:00.020, 5, 31, 42, 64, 64, 64, 64, 64, 64, 64
```

### Dataset parameters

| Parameter | Value |
|-----------|-------|
| Transactions per second | 50 |
| Duration | 2 hours |
| Total rows | 360,000 |
| Columns | 11 (date + p0..p9) |
| Partition IDs | 0–63 |
| Sentinel (padding) value | 64 |
| Seed | 42 (default) |

### Evaluation grid

N (input window) × K (prediction horizon):

|          | K=48 | K=96 | K=192 | K=336 |
|----------|------|------|-------|-------|
| **N=96** | ✓ | ✓ | ✓ | ✓ |
| **N=192** | ✓ | ✓ | ✓ | ✓ |
| **N=336** | ✓ | ✓ | ✓ | ✓ |
| **N=720** | ✓ | ✓ | ✓ | ✓ |

---

## 3. Model Architecture

### Input pipeline

1. Each transaction's 10 partition IDs are looked up in a shared partition embedding table: `Embedding(65, d_model // 10)` (65 entries: IDs 0–63 + sentinel 64)
2. The 10 slot embeddings are concatenated → one `d_model`-dim token per transaction
3. Sentinel slots (value 64) are **masked out of the loss** — cross-entropy is only computed over non-sentinel positions

### Seq2seq framing (matches existing Informer interface)

```
Encoder input:  N tokens                         (N past transactions)
Decoder input:  label_len tokens + K zero tokens  (label_len = N // 2)
Decoder output: K tokens
```

### Classification head (shared across all models)

```
Linear(d_model → 64 × 10)
→ reshape to (K, 10, 64)
→ softmax over dim=-1 (64 classes per slot)
```

At inference: `argmax` over 64 classes per slot → sort ascending → set sentinel slots to 64.

### Loss

```
CrossEntropy averaged over (K × non-sentinel slots)
```

### Models compared

All models use the same embedding layer and classification head. Only the sequence backbone differs.

| Model | Backbone | Complexity |
|-------|----------|------------|
| Informer | ProbSparse self-attention encoder + decoder | O(L log L) |
| LSTM | 2-layer LSTM | O(L) |
| RNN | 2-layer GRU | O(L) |

---

## 4. Workload Modes

All four modes generate 360,000 transactions at 50 tx/s (2 hours). Each transaction is a set of 1–10 partition IDs drawn from the mode's access probability distribution at that point in time.

### Mode A — `harmonic_superposition`

The access probability of each partition oscillates as a sum of four sinusoids:

| Component | Period (transactions) | Notes |
|-----------|-----------------------|-------|
| Fast | 96 | Within any N window |
| Medium | 960 | ~1–10× N window |
| Slow | 4,800 | ~7–50× N window |
| Seasonal | 19,200 | Spans full dataset; 26× longest N |

Each of 64 partitions has an independent random phase per component. The seasonal envelope (19,200 transactions ≈ 6.4 minutes) cannot be captured by LSTM hidden state; Informer's attention bridges it structurally.

### Mode B — `phase_cascade`

64 partitions divided into 8 groups of 8. Group 0 follows a clean sinusoidal hotspot cycle (period 120 transactions). Group i receives a delayed copy with lag `i × 80` transactions:

| Group | Lag (transactions) |
|-------|--------------------|
| 0 | 0 (source) |
| 1 | 80 |
| 2 | 160 |
| 3 | 240 |
| 4 | 320 |
| 5 | 400 |
| 6 | 480 |
| 7 | 560 |

Cumulative lag for Group 7 is 560 transactions — within the N=720 window, making the full cascade directly observable to the encoder. Informer's attention can directly correlate Group 0's current state (position t) with Group 7's state (position t−560) in a single forward pass. LSTM must propagate this information through 560 recurrent steps, causing signal decay that worsens at large K.

### Mode C — `frequency_modulated`

The hotspot rotation speed is itself modulated by a slow 8,000-transaction envelope:

```
P_inst(t) = P_mid + (P_max − P_mid) · sin(2π · t / P_envelope)

P_mid      = 400   transactions
P_max      = 1,600 transactions
P_envelope = 8,000 transactions
```

Instantaneous period oscillates between ~160 and 1,600 (clipped). To predict the next hotspot partition, the model must estimate its position within the slow envelope — a global context problem that local recurrence cannot solve.

### Mode D — `regime_with_precursors`

Five regimes with distinct hotspot partition sets and per-regime access lag structures. Regime transitions are preceded by three simultaneous leading signals appearing **200 transactions in advance**:

1. Rising access entropy (monotonic rise for 50+ consecutive transactions)
2. Declining hotspot concentration (monotonic fall for 50+ consecutive transactions)
3. Spiking partition spread (above μ + 2σ for 30+ consecutive transactions)

All three must co-occur; when they do, a transition fires exactly 200 transactions later. The 200-transaction lead time means the transition falls within the K=192 and K=336 prediction windows — models that detect the precursor (onset up to 250 transactions before the transition) will correctly forecast the regime shift; those that miss it will predict continuation of the current regime. LSTM/RNN detect local trends but miss the multi-feature co-occurrence across a 200-transaction span; Informer's attention correlates all three signals simultaneously.

---

## 5. Evaluation Metrics

All metrics computed over non-sentinel slots only.

| Metric | Definition |
|--------|-----------|
| **Slot Accuracy** | Fraction of (K × non-sentinel) slots predicted correctly (exact partition ID match) |
| **Set Accuracy** | Fraction of K transactions where the full predicted partition set exactly matches ground truth |
| **Top-3 Accuracy** | Fraction of slots where ground truth is in the top-3 predicted classes |
| **Cross-Entropy Loss** | Mean CE over non-sentinel slots (training objective, reported for comparability) |

All four metrics are reported for every cell in the 4×4 (N, K) grid, for each of the three models, across all four workload modes.

**Baseline:** A frequency-prior predictor (always predicts the top-10 most frequent partition IDs in the training set) is included as a sanity-check floor.

---

## 6. Files Changed

| File | Change |
|------|--------|
| `brain/modelgen/workload_generator.py` | Full rewrite: new transaction-level generators for modes A–D, new schema (date + p0..p9), 50 tx/s × 2h output |
| `brain/modelgen/workload_dataset.py` | Rewrite: load 11-column CSVs, partition embedding, sentinel masking, sliding-window dataset for (N, K) pairs |
| `brain/modelgen/evaluate.py` | Rewrite: slot accuracy, set accuracy, top-3 accuracy, CE loss; 4×4 grid evaluation; frequency-prior baseline |
| `brain/modelgen/tests/test_workload_generator.py` | Rewrite: tests for new schema, sentinel padding, mode-specific structural properties |
| `brain/modelgen/RUNNING.md` | Update CLI examples and mode descriptions |

**Not changed:** transformer backbone (`transformer/`), `LSTM.py`, `RNN.py`, `example.py`.

---

## 7. Verification Criteria

- Each CSV has exactly 360,000 rows and 11 columns
- `p0..p9` values are all in [0, 64] (sentinel inclusive)
- Within each row, non-sentinel values are strictly ascending
- No NaN or Inf values
- Mode B: cross-correlation between Group 0 and Group 7 access indicators peaks at lag 560 ± 5
- Mode C: instantaneous period recoverable from phase accumulator log
- Mode D: precursor triplet appears before every regime transition with lead time 200 ± 5 transactions

---

## 8. Open Question

If the four workload modes do not produce a measurable Informer advantage after initial training runs, the modes will be revised to amplify one or more of: longer cascade lag, stronger precursor signal, or deeper frequency modulation. The architecture and schema are fixed regardless.
