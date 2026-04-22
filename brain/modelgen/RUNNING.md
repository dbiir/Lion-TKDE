# Running the Transaction-Level Workload Generator

## Overview

Each CSV contains 360,000 rows (2 hours at 50 transactions/second).
Each row represents one transaction as a sorted list of up to 10 partition
IDs (0–63), with unused slots padded with sentinel value 64.

## CSV Format

```
date,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9
2026-01-01 00:00:00.000000000,3,17,42,64,64,64,64,64,64,64
```

## Generate All Modes

```bash
cd brain/modelgen
python workload_generator.py --mode all --out-dir data/
```

Writes four files to `data/`:
- `workload_harmonic_superposition.csv`
- `workload_phase_cascade.csv`
- `workload_frequency_modulated.csv`
- `workload_regime_with_precursors.csv`

## Generate a Single Mode

```bash
python workload_generator.py --mode A --out-dir data/
```

Options: `--mode A|B|C|D|all`, `--n-transactions N` (default 360000), `--seed S` (default 42).

## Workload Modes

| Mode | Name | Informer Advantage |
|------|------|--------------------|
| A | harmonic_superposition | Seasonal envelope (period 19,200 tx) requires global context beyond LSTM reach |
| B | phase_cascade | 560-step cascade lag within N=720 window; LSTM signal decays over distance |
| C | frequency_modulated | 8,000-tx modulation envelope; global position estimate needed |
| D | regime_with_precursors | 200-tx multi-feature precursor requires co-attention across long span |

## Evaluation Grid

N ∈ {96, 192, 336, 720} (encoder window) × K ∈ {48, 96, 192, 336} (prediction horizon).

## Train RNN, LSTM, and Informer Baselines

Each run trains **RNN (GRU), LSTM, and Informer** on every requested mode and
prints a side-by-side comparison table of slot accuracy, set accuracy, top-3
accuracy, and CE loss.

Quick demo (first 5,000 rows of each CSV, ~5 min):

```bash
cd brain/modelgen
python train_baselines.py
```

Full dataset (360,000 rows, several hours, ~8.6 GB RAM peak):

```bash
python train_baselines.py --n-rows 0
```

Single mode:

```bash
python train_baselines.py --modes A
```

| Flag | Default | Description |
|------|---------|-------------|
| `--modes` | `A B C D` | Modes to train |
| `--n-rows` | `5000` | Rows per CSV (0 = all 360 k) |
| `--seq-len` | `96` | Encoder window N |
| `--pred-len` | `48` | Prediction horizon K |
| `--epochs` | `10` | Max training epochs |
| `--patience` | `3` | Early-stopping patience |
| `--out-dir` | `checkpoints/` | Where to save `.pt` checkpoints |
| `--data-dir` | `data/` | Directory containing workload CSVs |

If `--n-rows` is too small for the requested window sizes the script auto-scales
`--seq-len` and `--pred-len` and prints a warning. The comparison table header
shows the effective values used.

### Model Architectures

| Model | Type | Key difference |
|-------|------|----------------|
| RNN | 2-layer GRU | Encodes seq into final hidden state only |
| LSTM | 2-layer LSTM | Same encoder-only design as RNN |
| Informer | Encoder-decoder Transformer | Full attention over all N positions; cross-attention in decoder lets each future step attend globally |

All three share the same `PartitionEmbedding` input layer (categorical
embedding per partition slot) and `masked_ce_loss` (cross-entropy ignoring
sentinel slots). The Informer's decoder uses the last `seq_len/2` real tokens
as context, then appends `pred_len` sentinel placeholders for autoregressive
prediction.

### Why Informer should outperform RNN/LSTM

RNN/LSTM compress the entire history into a single hidden state vector before
predicting. Long-range dependencies (e.g. the 19,200-tx seasonal envelope in
mode A or the 560-step cascade lag in mode B) suffer from vanishing gradients.
The Informer's self-attention can directly connect any two positions in the
encoder, and cross-attention lets each decoder step query the full encoder
output — the theoretical advantage the workload modes are designed to expose.

## Running Tests

```bash
cd brain/modelgen
python -m pytest tests/test_workload_generator.py tests/test_models.py -v
```

50 tests total (38 workload-generator + 12 model tests).
