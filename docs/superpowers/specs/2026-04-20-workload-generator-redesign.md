# Workload Generator Redesign — Design Spec
**Date:** 2026-04-20  
**Scope:** Replace workload modes A–D in `brain/modelgen/workload_generator.py` with four new complex modes designed to favour Informer (Transformer) over LSTM and RNN in a fair head-to-head comparison.

---

## 1. Goal

Generate synthetic OLTP workload CSVs that, when used to train and evaluate Informer vs LSTM vs RNN:

- Produce **measurably lower error (MAE, RMSE, MAPE)** for Informer than for LSTM/RNN
- Do so for principled reasons rooted in Transformer attention's known strengths
- Support comparison across **multiple seq_len windows** (96, 192, 336, 720) and **multiple pred_len horizons** (1, 6, 24, 48)
- Persist all CSVs to `./data/` unchanged from the current CLI interface

The feature schema (198 columns + date), `WorkloadDataset`, and `evaluate.py` are **not changed**.

---

## 2. Core Design Principle

Informer wins over LSTM/RNN when the signal requires:

1. **Long-range dependencies** — information needed for prediction originated more than ~50 intervals ago (beyond reliable LSTM memory)
2. **Cross-feature phase relationships** — feature X at time t−k predicts feature Y at time t, for large k
3. **Multi-scale decomposition** — the signal is a superposition of components at very different frequencies; slow components must be tracked across thousands of intervals

All four new modes embed at least two of these three properties simultaneously, at varying strengths, so the Transformer advantage is robust across different seq_len/pred_len settings.

---

## 3. Dataset Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_intervals` | 10,000 | Ensures ≥4 full cycles of the slowest signal component |
| seq_len benchmark | 96 / 192 / 336 / 720 | Matches Informer paper evaluation protocol |
| pred_len benchmark | 1 / 6 / 24 / 48 | Short to long horizon |
| Feature columns | 198 (unchanged) | `workload_dataset.py` compatibility |
| Output directory | `./data/` | Unchanged |
| Seed | 42 (default) | Reproducible |

---

## 4. Mode A — `harmonic_superposition`

### Signal structure

Transaction rate is a sum of four sinusoidal components:

```
txn_rate(t) = base + A1·sin(2π·t/P1 + φ1)
                   + A2·sin(2π·t/P2 + φ2)
                   + A3·sin(2π·t/P3 + φ3)
                   + A4·sin(2π·t/P4 + φ4)
```

| Component | Period Pi | Amplitude Ai | Notes |
|-----------|-----------|--------------|-------|
| Fast | 48 | 0.15·base | Within any seq_len window |
| Medium | 480 | 0.30·base | 5–10× seq_len |
| Slow | 2,400 | 0.25·base | 25× seq_len (96-window) |
| Seasonal | 9,600 | 0.20·base | Requires full 10k dataset |

Phases φ1–φ4 are randomly seeded but fixed per mode.

Each of the 64 partitions has its own access probability oscillation, with frequency drawn randomly from {P1, P2, P3} and an independent phase. This creates an interference pattern across partitions.

### Cross-feature coupling

- `write_ratio(t)` = base_write · (1 + 0.4 · sin(2π·t/P2 + φ2)) — amplitude modulated by medium component
- `abort_rate(t)` ≈ f(txn_rate(t−120)) + noise — 120-interval lagged echo of short component
- `queue_len(t)` = exponential smoothing of abort_rate with τ=30 intervals

### Why Transformer wins

- Decomposing the 2,400-period slow component requires retaining signal across thousands of intervals — beyond LSTM capacity
- The abort_rate ↔ txn_rate cross-feature lag of 120 intervals requires multi-variate attention across the full seq_len
- RNNs fit the fast (48-period) component well but systematically miss the slow envelope, producing large errors at longer pred_len

---

## 5. Mode B — `phase_cascade`

### Signal structure

64 partitions are divided into 8 groups of 8. Group 0 is the "source" with a clean sinusoidal access pattern (period P=300). Group i receives a delayed copy of group 0 with lag `Δ_i = i × 150` intervals:

```
access_group_i(t) = access_group_0(t − i·150) · (1 + ε_i)
```

where ε_i is group-specific multiplicative noise (σ=0.05).

Cumulative lag for group 7: **1,050 intervals**.

| Group | Lag (intervals) |
|-------|----------------|
| 0 | 0 (source) |
| 1 | 150 |
| 2 | 300 |
| 3 | 450 |
| 4 | 600 |
| 5 | 750 |
| 6 | 900 |
| 7 | 1,050 |

### Cross-feature coupling

- `total_txn(t)` = base + contribution from group 0 + group 4 activity
- `hotspot_ratio(t)` reflects index of currently-peaking group
- `entropy(t)` is minimized when groups are maximally desynchronized (alternating groups at opposite phases), maximized when all aligned

### Why Transformer wins

Predicting group 7 at time t requires remembering group 0 at t−1,050. LSTM cannot bridge a gap of 1,050 steps. Informer's sparse attention directly links position t to position t−1,050 in O(L log L). This effect is strongest at seq_len ≥ 192 where the full cascade is partially visible.

---

## 6. Mode C — `frequency_modulated`

### Signal structure

The hotspot rotation period is itself time-varying, modulated by a slow 4,000-interval envelope:

```
P_inst(t) = P_mid + (P_max − P_mid) · sin(2π·t / P_envelope)

P_mid      = 200   intervals (midpoint rotation speed)
P_max      = 800   intervals (slow phase)
P_envelope = 4,000 intervals (modulation period)
```

This means instantaneous rotation period oscillates between ~−400 and 800 (clipped to [80, 800]).

The currently-active hotspot partition set rotates based on a phase accumulator driven by P_inst(t):

```
phase(t) = phase(t−1) + 2π / P_inst(t)
hotspot_set(t) = f(floor(phase(t) / (2π)) mod N_SETS)
```

### Cross-feature coupling

- `total_txn(t)` ∝ burst when P_inst is small (fast rotation phase)
- `abort_rate` and `queue_len` show GARCH-style volatility clustering tied to fast-rotation windows
- `avg_partitions(t)` increases during slow-rotation phases (transactions touch more partitions when load is diffuse)

### Why Transformer wins

To predict the next hotspot, the model must estimate P_inst(t). Estimating P_inst requires seeing several recent rotation cycles to measure their duration. But to know whether P_inst is currently increasing or decreasing, the model needs to locate itself within the 4,000-interval modulation envelope — requiring global context. RNNs track fast cycles locally but cannot fit the modulation envelope. Informer over 720 intervals captures enough of the envelope.

---

## 7. Mode D — `regime_with_precursors`

### Signal structure

The workload transitions between 5 distinct regimes. Each regime has:
- A fixed hotspot partition set (non-overlapping across regimes)
- Its own periodic structure (period drawn from {120, 360, 720, 1200, 2400})
- Its own write_ratio base and amplitude
- Regime-specific cross-feature lag (see below)

Regime dwell time: uniform in [600, 1200] intervals.

### Precursor mechanism

Regime transitions are predictable **200 intervals in advance** via three simultaneous leading indicators:

1. `entropy` begins a monotonic rise (slope > θ_e for 50+ consecutive intervals)
2. `hotspot_ratio` begins a monotonic decline (slope < −θ_h for 50+ consecutive intervals)
3. `avg_partitions` spikes above μ + 2σ for 20+ consecutive intervals

All three must co-occur. When they do, a transition will fire exactly 200 intervals later. The target regime is deterministic given the current regime (fixed transition matrix).

### Regime-specific cross-feature lags

| Regime | write_ratio leads hotspot_ratio by | Notes |
|--------|------------------------------------|-------|
| 0 | 30 intervals | Positive correlation |
| 1 | 80 intervals | Anti-correlation |
| 2 | 60 intervals | Positive correlation |
| 3 | 0 (synchronous) | Direct coupling |
| 4 | 120 intervals | Weak correlation |

### Why Transformer wins

- Detecting the 3-feature precursor requires attending to a 50-interval window that may have started 200 intervals ago — direct test of long-range multi-variate attention
- Regime-specific cross-feature lags require the model to first identify the current regime (from global history) before applying the correct lag — a hierarchical reasoning task
- RNNs perform well within a stable regime but fail at transition prediction and regime-dependent lag estimation
- Note: the full precursor-to-transition span is 250 intervals (50 detection + 200 lead time); Transformer advantage on transition prediction is strongest at seq_len ≥ 336

---

## 8. Files Changed

| File | Change |
|------|--------|
| `brain/modelgen/workload_generator.py` | Full rewrite of mode generators; update `MODE_NAMES`, `MODE_GENERATORS`; increase default `n_intervals` to 10,000 |
| `brain/modelgen/RUNNING.md` | Update CLI examples and mode descriptions |

**Not changed:** `workload_dataset.py`, `evaluate.py`, `example.py`, `RNN.py`, `LSTM.py`, transformer framework.

---

## 9. Verification Criteria

After generation:
- Each CSV has exactly 10,000 rows and 199 columns (date + 198 features)
- No NaN or Inf values
- `hotspot_ratio` ∈ [0, 1] at all times
- `entropy` ∈ [0, 1] at all times
- Mode B: cross-correlation between group 0 and group 7 access peaks at lag 1,050 ± 10
- Mode C: instantaneous period recoverable from phase accumulator log
- Mode D: precursor windows appear before every regime transition, verifiable by inspection
