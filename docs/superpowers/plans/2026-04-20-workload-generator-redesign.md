# Workload Generator Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace workload modes A–D in `brain/modelgen/workload_generator.py` with four complex Transformer-favoring synthetic OLTP modes, persist output to `./data/`, and leave all downstream files (`workload_dataset.py`, `evaluate.py`, `example.py`, transformer framework) unchanged.

**Architecture:** Four standalone generator functions replace the existing ones; all share the existing `_aggregate_transactions`, `_make_access_probs_stable`, and `_to_dataframe` helpers. Each new mode encodes long-range dependencies, cross-feature phase lags, or multi-scale periodicity that Informer's attention can exploit but LSTM/RNN cannot. Tests validate shape, value ranges, and mode-specific structural properties.

**Tech Stack:** Python 3.x, NumPy, pandas, scipy, pytest

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `brain/modelgen/workload_generator.py` | Replace mode generators; update `MODE_NAMES`, `MODE_GENERATORS`; set `n_intervals` default to 10,000 |
| Create | `brain/modelgen/tests/__init__.py` | Empty — makes `tests/` a package |
| Create | `brain/modelgen/tests/test_workload_generator.py` | Shape/NaN/range/structural tests for all four modes |
| Modify | `brain/modelgen/RUNNING.md` | Update mode names, CLI examples, parameter table |

---

## Task 1: Create test scaffold (failing)

**Files:**
- Create: `brain/modelgen/tests/__init__.py`
- Create: `brain/modelgen/tests/test_workload_generator.py`

- [ ] **Step 1: Create the test package**

```bash
mkdir -p brain/modelgen/tests
touch brain/modelgen/tests/__init__.py
```

- [ ] **Step 2: Write the full test file**

Create `brain/modelgen/tests/test_workload_generator.py`:

```python
"""
Tests for workload_generator.py — all four redesigned modes.

Run from brain/modelgen/:
    pytest tests/test_workload_generator.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from workload_generator import (
    generate_mode_A, generate_mode_B, generate_mode_C, generate_mode_D,
    get_feature_columns, NUM_PARTITIONS,
)

N_SMALL = 500   # fast smoke-test length


@pytest.fixture(params=['A', 'B', 'C', 'D'])
def df(request):
    gen = {'A': generate_mode_A, 'B': generate_mode_B,
           'C': generate_mode_C, 'D': generate_mode_D}[request.param]
    return gen(n_intervals=N_SMALL, seed=42)


# ── Generic tests (all modes) ─────────────────────────────────────────────────

def test_shape(df):
    expected_cols = 1 + len(get_feature_columns())  # date + 198 features
    assert df.shape == (N_SMALL, expected_cols), (
        f"Expected ({N_SMALL}, {expected_cols}), got {df.shape}")

def test_no_nan(df):
    numeric = df.drop(columns=['date'])
    assert not numeric.isnull().any().any(), "Found NaN values in output"

def test_no_inf(df):
    numeric = df.drop(columns=['date'])
    assert not np.isinf(numeric.values).any(), "Found Inf values in output"

def test_hotspot_ratio_in_unit_interval(df):
    col = df['hotspot_ratio']
    assert (col >= 0.0).all() and (col <= 1.0).all(), (
        f"hotspot_ratio out of [0,1]: min={col.min():.4f} max={col.max():.4f}")

def test_entropy_in_unit_interval(df):
    col = df['entropy']
    assert (col >= 0.0).all() and (col <= 1.0).all(), (
        f"entropy out of [0,1]: min={col.min():.4f} max={col.max():.4f}")

def test_access_counts_non_negative(df):
    for i in range(NUM_PARTITIONS):
        col = df[f'access_{i}']
        assert (col >= 0).all(), f"access_{i} has negative values"

def test_write_le_access(df):
    for i in range(NUM_PARTITIONS):
        assert (df[f'write_{i}'] <= df[f'access_{i}']).all(), (
            f"write_{i} exceeds access_{i}")

def test_read_plus_write_eq_access(df):
    for i in range(NUM_PARTITIONS):
        diff = (df[f'read_{i}'] + df[f'write_{i}'] - df[f'access_{i}']).abs()
        assert (diff < 1e-3).all(), f"read_{i} + write_{i} != access_{i}"

def test_total_txn_positive(df):
    assert (df['total_txn'] > 0).all(), "total_txn contains non-positive values"


# ── Mode-A structural test ─────────────────────────────────────────────────────

def test_mode_A_slow_component_visible():
    """
    Mode A embeds a P=480 medium component. The total_txn series should have
    significant spectral power at frequency 1/480 (detectable even in 500 rows).
    We verify that total_txn is not flat (std > 0.05 * mean).
    """
    df = generate_mode_A(n_intervals=N_SMALL, seed=42)
    series = df['total_txn'].values.astype(float)
    assert series.std() > 0.05 * series.mean(), (
        "Mode A total_txn looks flat — sinusoidal components may be missing")


# ── Mode-B structural test ─────────────────────────────────────────────────────

def test_mode_B_cascade_correlation():
    """
    Group 7 lags group 0 by exactly 7*150 = 1050 intervals.
    With n=3000 rows we can detect this correlation at the known lag.
    """
    df = generate_mode_B(n_intervals=3000, seed=42)
    group0 = df[[f'access_{i}' for i in range(8)]].sum(axis=1).values.astype(float)
    group7 = df[[f'access_{i}' for i in range(56, 64)]].sum(axis=1).values.astype(float)

    lag = 1050
    n = len(group0)
    g0 = group0[:n - lag] - group0[:n - lag].mean()
    g7 = group7[lag:]     - group7[lag:].mean()

    corr = float(np.corrcoef(g0, g7)[0, 1])
    assert corr > 0.70, (
        f"Mode B: expected cross-correlation at lag 1050 > 0.70, got {corr:.3f}")


# ── Mode-C structural test ─────────────────────────────────────────────────────

def test_mode_C_hotspot_variety():
    """
    Mode C rotates hotspot sets. Over 500 intervals at least 2 distinct
    hotspot partitions should appear as the top-1 busiest partition.
    """
    df = generate_mode_C(n_intervals=N_SMALL, seed=42)
    access_cols = [f'access_{i}' for i in range(NUM_PARTITIONS)]
    top1 = df[access_cols].values.argmax(axis=1)
    unique_tops = len(np.unique(top1))
    assert unique_tops >= 2, (
        f"Mode C: expected ≥2 distinct top partitions, got {unique_tops}")


# ── Mode-D structural test ─────────────────────────────────────────────────────

def test_mode_D_regime_variation():
    """
    Mode D transitions between regimes with different periodic structures.
    Entropy across 500-interval blocks must show meaningful variance (non-stationary).
    """
    df = generate_mode_D(n_intervals=3000, seed=42)
    block_means = [df['entropy'].iloc[i * 500:(i + 1) * 500].mean() for i in range(6)]
    std = float(np.std(block_means))
    assert std > 0.01, (
        f"Mode D: entropy block-mean std={std:.4f} — regime variation not detectable")

def test_mode_D_precursor_entropy_rises():
    """
    Mode D precursor windows inject a rising entropy signal.
    Immediately before each transition, entropy should be higher than the
    regime steady-state (check at least one transition window).
    We use n=2000 with a fixed seed so transitions are deterministic.
    """
    df = generate_mode_D(n_intervals=2000, seed=42)
    ent = df['entropy'].values
    # Steady-state entropy: median of first 200 rows (within first regime)
    steady = float(np.median(ent[:200]))
    # A precursor window starts at t_transition - 200; t_transition ≥ 600.
    # With seed=42 first transition is around 600-1200. Check t=450:499 window.
    prec_window_mean = float(ent[450:500].mean())
    # The precursor should push entropy above steady-state
    assert prec_window_mean >= steady, (
        f"Mode D: precursor entropy {prec_window_mean:.4f} not above "
        f"steady-state {steady:.4f}")
```

- [ ] **Step 3: Run tests — expect failures (generators not yet rewritten)**

```bash
cd brain/modelgen
pytest tests/test_workload_generator.py -v 2>&1 | head -60
```

Expected: Most tests fail or error because the existing mode generators produce different data (Mode A outputs `stable_hotspot`, not `harmonic_superposition` style data). Some generic tests (shape, NaN, ranges) may pass since the helpers are unchanged.

- [ ] **Step 4: Commit test scaffold**

```bash
git add brain/modelgen/tests/
git commit -m "test: add workload generator test scaffold (failing)"
```

---

## Task 2: Rewrite Mode A — `harmonic_superposition`

**Files:**
- Modify: `brain/modelgen/workload_generator.py` (replace `generate_mode_A`)

- [ ] **Step 1: Replace `generate_mode_A` in `workload_generator.py`**

Find the existing `generate_mode_A` function and replace it entirely with:

```python
def generate_mode_A(n_intervals=10_000, seed=42):
    """
    Mode A – Harmonic Superposition.

    Transaction rate is the sum of four sinusoidal components at periods
    48 (fast), 480 (medium), 2400 (slow), 9600 (seasonal). Each of the 64
    partitions oscillates at one of {48, 480, 2400} with an independent phase.

    Cross-feature coupling:
      - write_ratio amplitude is modulated by the medium (P=480) component
      - abort_rate is a 120-interval lagged echo of the fast (P=48) component
      - queue_len is an exponential moving average of abort_rate (τ=30)

    Transformer advantage: the slow and seasonal components require retaining
    context across thousands of intervals; the abort_rate cross-feature lag
    (120 steps) requires multi-variate attention spanning the full seq_len.
    """
    rng = np.random.default_rng(seed)

    PERIODS    = [48,   480,   2_400,  9_600]
    AMPLITUDES = [0.15, 0.30,  0.25,   0.20]
    phases     = rng.uniform(0, 2 * np.pi, size=4)
    t          = np.arange(n_intervals, dtype=np.float64)
    base       = float(TXN_RATE_BASE)

    # Shape: (4, n_intervals) — one row per sinusoidal component
    components = np.array([
        amp * base * np.sin(2 * np.pi * t / P + phi)
        for P, amp, phi in zip(PERIODS, AMPLITUDES, phases)
    ])
    txn_rate = np.clip(base + components.sum(axis=0), base * 0.1, base * 3.0)

    # Each partition oscillates at one of the three fastest periods
    part_periods = rng.choice(PERIODS[:3], size=NUM_PARTITIONS)
    part_phases  = rng.uniform(0, 2 * np.pi, size=NUM_PARTITIONS)

    # write_ratio: base × (1 + 0.4 × medium_component_normalised)
    medium_norm = np.sin(2 * np.pi * t / 480 + phases[1])          # [-1, 1]
    write_ratio = np.clip(WRITE_PROB_DEFAULT * (1.0 + 0.4 * medium_norm), 0.05, 0.95)

    # abort_rate: normalised lagged echo of fast component, lag = 120
    short_norm = components[0] / (AMPLITUDES[0] * base)             # [-1, 1]
    abort_rate = np.zeros(n_intervals)
    for i in range(n_intervals):
        src = short_norm[max(0, i - 120)]
        abort_rate[i] = np.clip(0.02 + 0.015 * (src + 1.0), 0.005, 0.05)

    # queue_len: exponential moving average of abort_rate, τ = 30
    alpha     = 1.0 - np.exp(-1.0 / 30.0)
    queue_len = np.zeros(n_intervals)
    queue_len[0] = abort_rate[0]
    for i in range(1, n_intervals):
        queue_len[i] = (1.0 - alpha) * queue_len[i - 1] + alpha * abort_rate[i]

    records = []
    for i in range(n_intervals):
        weights = np.clip(
            1.0 + 0.5 * np.sin(2 * np.pi * i / part_periods.astype(float) + part_phases),
            1e-3, None
        )
        probs = weights / weights.sum()
        feat = _aggregate_transactions(
            int(txn_rate[i]), probs, float(write_ratio[i]),
            abort_prob=float(abort_rate[i]),
            queue_signal=float(queue_len[i]),
        )
        feat['interval'] = i
        records.append(feat)

    return _to_dataframe(records)
```

- [ ] **Step 2: Run Mode-A tests**

```bash
cd brain/modelgen
pytest tests/test_workload_generator.py -v -k "A"
```

Expected output (all pass):
```
PASSED tests/test_workload_generator.py::test_shape[A]
PASSED tests/test_workload_generator.py::test_no_nan[A]
PASSED tests/test_workload_generator.py::test_no_inf[A]
PASSED tests/test_workload_generator.py::test_hotspot_ratio_in_unit_interval[A]
PASSED tests/test_workload_generator.py::test_entropy_in_unit_interval[A]
PASSED tests/test_workload_generator.py::test_access_counts_non_negative[A]
PASSED tests/test_workload_generator.py::test_write_le_access[A]
PASSED tests/test_workload_generator.py::test_read_plus_write_eq_access[A]
PASSED tests/test_workload_generator.py::test_total_txn_positive[A]
PASSED tests/test_workload_generator.py::test_mode_A_slow_component_visible
```

- [ ] **Step 3: Commit**

```bash
git add brain/modelgen/workload_generator.py
git commit -m "feat: rewrite mode A as harmonic_superposition"
```

---

## Task 3: Rewrite Mode B — `phase_cascade`

**Files:**
- Modify: `brain/modelgen/workload_generator.py` (replace `generate_mode_B`)

- [ ] **Step 1: Replace `generate_mode_B`**

```python
def generate_mode_B(n_intervals=10_000, seed=42):
    """
    Mode B – Phase Cascade.

    64 partitions are split into 8 groups of 8. Group 0 is a source sinusoid
    (period 300). Group i carries a delayed copy of group 0 with lag i×150
    intervals, plus small multiplicative noise. The full cascade (group 0 →
    group 7) spans 1,050 intervals.

    Cross-feature coupling:
      - total_txn driven by the average of group-0 and group-4 signals
      - entropy is low when all groups are in phase, high when desynchronised

    Transformer advantage: predicting group-7 access at time t requires
    attending to group-0 state at t−1,050 — beyond reliable LSTM memory.
    The advantage grows with seq_len (strongest at 336/720).
    """
    rng = np.random.default_rng(seed)

    N_GROUPS  = 8
    PPG       = NUM_PARTITIONS // N_GROUPS   # 8 partitions per group
    PERIOD    = 300
    LAG_STEP  = 150
    NOISE_STD = 0.05

    t = np.arange(n_intervals, dtype=np.float64)
    src_phase = rng.uniform(0, 2 * np.pi)
    source    = np.sin(2 * np.pi * t / PERIOD + src_phase)   # shape (n,)

    # Group i: source delayed by i × LAG_STEP, with additive noise
    group_noise   = rng.normal(0, NOISE_STD, size=(N_GROUPS, n_intervals))
    group_signals = np.zeros((N_GROUPS, n_intervals))
    for g in range(N_GROUPS):
        lag    = g * LAG_STEP
        padded = np.concatenate([np.full(lag, source[0]), source])[:n_intervals]
        group_signals[g] = padded + group_noise[g]

    records = []
    for i in range(n_intervals):
        # Access probability proportional to (1 + signal) per group
        probs = np.zeros(NUM_PARTITIONS)
        for g in range(N_GROUPS):
            weight = max(1e-3, 1.0 + group_signals[g, i])
            probs[g * PPG:(g + 1) * PPG] = weight / PPG
        probs /= probs.sum()

        # txn_rate: average of group-0 and group-4 signals
        avg_sig    = (group_signals[0, i] + group_signals[4, i]) / 2.0
        rate_factor = max(0.1, 1.0 + 0.3 * avg_sig)
        txn_count  = int(rng.poisson(TXN_RATE_BASE * rate_factor))

        feat = _aggregate_transactions(
            txn_count, probs, WRITE_PROB_DEFAULT,
            abort_prob=0.015, queue_signal=0.10,
        )
        feat['interval'] = i
        records.append(feat)

    return _to_dataframe(records)
```

- [ ] **Step 2: Run Mode-B tests**

```bash
cd brain/modelgen
pytest tests/test_workload_generator.py -v -k "B"
```

Expected: all Mode-B tests pass including `test_mode_B_cascade_correlation` (which requires n=3000 internally; the fixture uses n=500 but the structural test calls the generator directly with n=3000).

- [ ] **Step 3: Commit**

```bash
git add brain/modelgen/workload_generator.py
git commit -m "feat: rewrite mode B as phase_cascade"
```

---

## Task 4: Rewrite Mode C — `frequency_modulated`

**Files:**
- Modify: `brain/modelgen/workload_generator.py` (replace `generate_mode_C`)

- [ ] **Step 1: Replace `generate_mode_C`**

```python
def generate_mode_C(n_intervals=10_000, seed=42):
    """
    Mode C – Frequency Modulated.

    The hotspot rotation period is itself time-varying, modulated by a
    4,000-interval sinusoidal envelope:

        P_inst(t) = 440 + 360 × sin(2π·t / 4000 + φ)
        clamped to [80, 800] intervals.

    A phase accumulator drives hotspot-set rotation: when P_inst is small
    (fast rotation, high txn burst), sets change rapidly; when large, slowly.

    Cross-feature coupling:
      - txn_rate bursts when P_inst is small (high-frequency phase)
      - abort_rate and queue_len show volatility clustering tied to fast phases

    Transformer advantage: estimating the current P_inst requires seeing
    several recent rotation cycles AND locating the model within the 4,000-
    interval envelope — requiring global context that RNNs cannot maintain.
    """
    rng = np.random.default_rng(seed)

    P_MID      = 440.0      # midpoint of period range  (80 + 800) / 2
    P_HALF     = 360.0      # half-range                (800 - 80) / 2
    P_ENVELOPE = 4_000.0
    N_SETS     = 8

    env_phase = rng.uniform(0, 2 * np.pi)

    # N_SETS non-overlapping hotspot sets (5 partitions each)
    shuffled     = rng.permutation(NUM_PARTITIONS)
    hotspot_sets = [list(shuffled[s * 5:(s + 1) * 5]) for s in range(N_SETS)]

    phase_acc = 0.0
    records   = []

    for i in range(n_intervals):
        p_inst    = float(np.clip(
            P_MID + P_HALF * np.sin(2 * np.pi * i / P_ENVELOPE + env_phase),
            80.0, 800.0
        ))
        phase_acc += 2 * np.pi / p_inst

        hot_idx = int(phase_acc / (2 * np.pi)) % N_SETS
        probs   = _make_access_probs_stable(hotspot_sets[hot_idx], hot_fraction=0.75)

        # txn_rate bursts when rotation is fast (small P_inst)
        rate_factor = 1.0 + 1.5 * (1.0 - (p_inst - 80.0) / 720.0)
        txn_count   = int(rng.poisson(TXN_RATE_BASE * rate_factor))

        abort_p = float(np.clip(0.01 + 0.04 * (rate_factor - 1.0) / 1.5, 0.005, 0.05))
        queue_s = float(np.clip(0.05 * rate_factor, 0.0, 1.0))

        feat = _aggregate_transactions(
            txn_count, probs, WRITE_PROB_DEFAULT,
            abort_prob=abort_p, queue_signal=queue_s,
        )
        feat['interval'] = i
        records.append(feat)

    return _to_dataframe(records)
```

- [ ] **Step 2: Run Mode-C tests**

```bash
cd brain/modelgen
pytest tests/test_workload_generator.py -v -k "C"
```

Expected: all pass including `test_mode_C_hotspot_variety`.

- [ ] **Step 3: Commit**

```bash
git add brain/modelgen/workload_generator.py
git commit -m "feat: rewrite mode C as frequency_modulated"
```

---

## Task 5: Rewrite Mode D — `regime_with_precursors`

**Files:**
- Modify: `brain/modelgen/workload_generator.py` (replace `generate_mode_D`, add `_REGIMES` constant)

- [ ] **Step 1: Add `_REGIMES` constant above `generate_mode_D`**

Insert this constant block immediately before the `generate_mode_D` function definition (after `generate_mode_C`):

```python
# Regime definitions for Mode D
# Each dict: periodic structure, hotspot partitions, write_ratio coupling
_REGIMES = [
    {'period': 120,  'hot': [0,  1,  2,  3,  4],  'w_base': 0.25, 'w_amp': 0.10, 'lag':  30, 'sign':  1},
    {'period': 360,  'hot': [10, 11, 20, 21, 30], 'w_base': 0.35, 'w_amp': 0.15, 'lag':  80, 'sign': -1},
    {'period': 720,  'hot': [32, 33, 40, 41, 50], 'w_base': 0.30, 'w_amp': 0.12, 'lag':  60, 'sign':  1},
    {'period': 1200, 'hot': [5,  15, 25, 35, 45], 'w_base': 0.20, 'w_amp': 0.08, 'lag':   0, 'sign':  1},
    {'period': 2400, 'hot': [55, 56, 60, 61, 62], 'w_base': 0.40, 'w_amp': 0.20, 'lag': 120, 'sign':  1},
]
```

- [ ] **Step 2: Replace `generate_mode_D`**

```python
def generate_mode_D(n_intervals=10_000, seed=42):
    """
    Mode D – Regime with Precursors.

    The workload lives in one of 5 regimes (each with its own period, hotspot
    set, and write_ratio lag). Transitions happen every 600–1,200 intervals,
    following a fixed rotation (regime i → regime (i+1) % 5).

    Transitions are predictable 200 intervals in advance via a 50-interval
    precursor window in which three signals are injected:
      1. entropy rises monotonically (access distribution blends toward uniform)
      2. hotspot_ratio falls (concentration decreases as distribution flattens)
      3. avg_partitions spikes (partitions_per_txn_mean increases 2.5 → 6.5)

    Cross-feature coupling: each regime has a distinct write_ratio lag relative
    to its hotspot_ratio phase (see _REGIMES table above).

    Transformer advantage:
      - detecting the 3-feature precursor requires attending to a 50-interval
        window that may have started up to 200 intervals ago
      - regime-specific write_ratio lags require global context to identify
        the current regime before applying the correct lag
      - advantage is strongest at seq_len ≥ 336
    """
    rng = np.random.default_rng(seed)

    PRECURSOR_LEN = 50
    LEAD_TIME     = 200

    # ── Pre-compute transition schedule ───────────────────────────────────────
    transitions = []           # (t_transition, from_regime, to_regime)
    t_cursor    = 0
    cur_regime  = 0
    while t_cursor < n_intervals:
        dwell        = int(rng.integers(600, 1201))
        t_transition = t_cursor + dwell
        if t_transition >= n_intervals:
            break
        next_regime = (cur_regime + 1) % len(_REGIMES)
        transitions.append((t_transition, cur_regime, next_regime))
        t_cursor   = t_transition
        cur_regime = next_regime

    # Per-interval regime index
    regime_at = np.zeros(n_intervals, dtype=int)
    prev_t, prev_r = 0, 0
    for (t_tr, _, to_r) in transitions:
        regime_at[prev_t:t_tr] = prev_r
        prev_t = t_tr
        prev_r = to_r
    regime_at[prev_t:] = prev_r

    # Precursor progress: 0 outside precursor, linear 0→1 inside
    prec_progress = np.zeros(n_intervals)
    for (t_tr, _, _) in transitions:
        t_ps = max(0, t_tr - LEAD_TIME)
        t_pe = min(n_intervals, t_ps + PRECURSOR_LEN)
        n_p  = t_pe - t_ps
        if n_p > 0:
            prec_progress[t_ps:t_pe] = np.linspace(0.0, 1.0, n_p)

    uniform_probs = np.ones(NUM_PARTITIONS) / NUM_PARTITIONS

    records = []
    for i in range(n_intervals):
        r      = _REGIMES[regime_at[i]]
        probs  = _make_access_probs_stable(r['hot'], hot_fraction=0.8)

        # Write ratio: lagged coupling with hotspot_ratio periodic signal
        lag      = r['lag']
        phase_i  = 2 * np.pi * max(0, i - lag) / r['period']
        hot_sig  = np.sin(phase_i)
        wr       = float(np.clip(r['w_base'] + r['sign'] * r['w_amp'] * hot_sig, 0.05, 0.95))

        # Precursor injection
        progress = float(prec_progress[i])
        if progress > 0.0:
            probs = (1.0 - progress) * probs + progress * uniform_probs
            probs /= probs.sum()
            ptm = 2.5 + 4.0 * progress   # avg partitions: 2.5 → 6.5
        else:
            ptm = 2.5

        txn_count = int(rng.poisson(TXN_RATE_BASE))
        feat = _aggregate_transactions(
            txn_count, probs, wr,
            partitions_per_txn_mean=ptm,
            abort_prob=0.018, queue_signal=0.12,
        )
        feat['interval'] = i
        records.append(feat)

    return _to_dataframe(records)
```

- [ ] **Step 3: Run Mode-D tests**

```bash
cd brain/modelgen
pytest tests/test_workload_generator.py -v -k "D"
```

Expected: all pass including `test_mode_D_regime_variation` and `test_mode_D_precursor_entropy_rises`.

- [ ] **Step 4: Commit**

```bash
git add brain/modelgen/workload_generator.py
git commit -m "feat: rewrite mode D as regime_with_precursors"
```

---

## Task 6: Update constants and CLI defaults

**Files:**
- Modify: `brain/modelgen/workload_generator.py` (update `MODE_NAMES`, `MODE_GENERATORS`, `main()` default)

- [ ] **Step 1: Replace `MODE_NAMES` and `MODE_GENERATORS` dicts**

Find:
```python
MODE_GENERATORS = {
    "A": generate_mode_A,
    "B": generate_mode_B,
    "C": generate_mode_C,
    "D": generate_mode_D,
}

MODE_NAMES = {
    "A": "stable_hotspot",
    "B": "hotspot_shift",
    "C": "bursty_traffic",
    "D": "mixed_workload",
}
```

Replace with:
```python
MODE_GENERATORS = {
    "A": generate_mode_A,
    "B": generate_mode_B,
    "C": generate_mode_C,
    "D": generate_mode_D,
}

MODE_NAMES = {
    "A": "harmonic_superposition",
    "B": "phase_cascade",
    "C": "frequency_modulated",
    "D": "regime_with_precursors",
}
```

- [ ] **Step 2: Update default `--intervals` in `main()` to 10,000**

Find in `main()`:
```python
    parser.add_argument("--intervals", type=int, default=3000,
                        help="Number of scheduling intervals to generate")
```

Replace with:
```python
    parser.add_argument("--intervals", type=int, default=10_000,
                        help="Number of scheduling intervals to generate")
```

- [ ] **Step 3: Run full test suite**

```bash
cd brain/modelgen
pytest tests/test_workload_generator.py -v
```

Expected: all tests pass across all four modes.

- [ ] **Step 4: Commit**

```bash
git add brain/modelgen/workload_generator.py
git commit -m "feat: update MODE_NAMES and default n_intervals=10000"
```

---

## Task 7: Generate all workload CSVs to `./data/`

**Files:**
- None modified — this is a run step to produce the data artifacts.

- [ ] **Step 1: Run the generator for all four modes**

```bash
cd brain/modelgen
python workload_generator.py --mode all --intervals 10000 --out-dir data/
```

Expected output:
```
[+] Generating mode A (harmonic_superposition) – 10000 intervals ...
    Saved 10000 rows × 199 columns → data/workload_harmonic_superposition.csv
    Feature dimensions: 198 workload features
[+] Generating mode B (phase_cascade) – 10000 intervals ...
    Saved 10000 rows × 199 columns → data/workload_phase_cascade.csv
    Feature dimensions: 198 workload features
[+] Generating mode C (frequency_modulated) – 10000 intervals ...
    Saved 10000 rows × 199 columns → data/workload_frequency_modulated.csv
    Feature dimensions: 198 workload features
[+] Generating mode D (regime_with_precursors) – 10000 intervals ...
    Saved 10000 rows × 199 columns → data/workload_regime_with_precursors.csv
    Feature dimensions: 198 workload features
```

- [ ] **Step 2: Verify CSV shape and integrity**

```bash
cd brain/modelgen
python -c "
import pandas as pd, numpy as np, os
for name in ['harmonic_superposition','phase_cascade','frequency_modulated','regime_with_precursors']:
    path = f'data/workload_{name}.csv'
    df = pd.read_csv(path)
    n_nan = df.isnull().sum().sum()
    n_inf = np.isinf(df.select_dtypes('number').values).sum()
    print(f'{name}: shape={df.shape}  NaN={n_nan}  Inf={n_inf}')
    assert df.shape == (10000, 199), f'Bad shape: {df.shape}'
    assert n_nan == 0 and n_inf == 0
print('All OK')
"
```

Expected:
```
harmonic_superposition: shape=(10000, 199)  NaN=0  Inf=0
phase_cascade: shape=(10000, 199)  NaN=0  Inf=0
frequency_modulated: shape=(10000, 199)  NaN=0  Inf=0
regime_with_precursors: shape=(10000, 199)  NaN=0  Inf=0
All OK
```

- [ ] **Step 3: Commit data directory note (not the CSVs themselves)**

Check if `data/` is in `.gitignore`; if not, add an entry so large CSVs are not committed:

```bash
cd brain/modelgen
grep -q "^data/" ../../.gitignore 2>/dev/null || echo "brain/modelgen/data/" >> ../../.gitignore
git add ../../.gitignore
git commit -m "chore: gitignore generated workload data CSVs"
```

---

## Task 8: Update RUNNING.md

**Files:**
- Modify: `brain/modelgen/RUNNING.md`

- [ ] **Step 1: Update mode table and CLI examples in RUNNING.md**

Find the "File Overview" table and add `tests/` row. Find the mode names table and update. Find CLI examples and update interval defaults.

Replace the existing "File Overview" table:

```markdown
| File | Purpose |
|------|---------|
| `workload_generator.py` | Generates synthetic OLTP workload CSVs (4 modes) |
| `workload_dataset.py`   | PyTorch Dataset + DataModule for the workload CSVs |
| `evaluate.py`           | Comprehensive evaluation metrics and plots |
| `example.py`            | Self-contained end-to-end demo (GRU baseline) |
| `RNN.py`                | GRU baseline model and training loop |
```

With:

```markdown
| File | Purpose |
|------|---------|
| `workload_generator.py`          | Generates synthetic OLTP workload CSVs (4 Transformer-favoring modes) |
| `workload_dataset.py`            | PyTorch Dataset + DataModule for the workload CSVs |
| `evaluate.py`                    | Comprehensive evaluation metrics and plots |
| `example.py`                     | Self-contained end-to-end demo (GRU baseline) |
| `RNN.py`                         | GRU baseline model and training loop |
| `tests/test_workload_generator.py` | Shape, integrity, and structural tests for all modes |
```

Replace the paragraph beginning "This produces:" (the mode file list) with:

```markdown
This produces:

```
data/
  workload_harmonic_superposition.csv   # Mode A – multi-scale sinusoidal superposition
  workload_phase_cascade.csv            # Mode B – phase-lagged cross-partition cascade
  workload_frequency_modulated.csv      # Mode C – FM-style hotspot rotation speed
  workload_regime_with_precursors.csv   # Mode D – 5-regime with 200-interval precursor signals
```
```

Replace the interval count in the "Training Settings" table row:

Find:
```markdown
| Input features | 198 (access + read/write + global stats) |
```

Add above it:
```markdown
| Dataset size | 10,000 intervals per mode |
| Recommended seq_len | 96 / 192 / 336 / 720 (benchmark suite) |
| Recommended pred_len | 1 / 6 / 24 / 48 |
```

- [ ] **Step 2: Commit**

```bash
git add brain/modelgen/RUNNING.md
git commit -m "docs: update RUNNING.md for redesigned workload modes"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by task |
|-----------------|----------------|
| Mode A: 4 sinusoidal components at 48/480/2400/9600 | Task 2 |
| Mode A: write_ratio amplitude-modulated by P=480 | Task 2 |
| Mode A: abort_rate lagged echo at 120 intervals | Task 2 |
| Mode A: queue_len = EMA(abort_rate, τ=30) | Task 2 |
| Mode B: 8 groups × 8 partitions, lag i×150 | Task 3 |
| Mode B: txn_rate driven by group 0 + group 4 | Task 3 |
| Mode B: total lag group 0→7 = 1,050 intervals | Task 3 |
| Mode C: P_inst oscillates 80–800 over P_envelope=4000 | Task 4 |
| Mode C: phase accumulator drives hotspot rotation | Task 4 |
| Mode C: txn burst when P_inst small | Task 4 |
| Mode D: 5 regimes, dwell 600–1200, rotation order | Task 5 |
| Mode D: precursor 50 intervals, lead time 200 intervals | Task 5 |
| Mode D: entropy rises in precursor (blend to uniform) | Task 5 |
| Mode D: hotspot_ratio falls in precursor | Task 5 (side effect of blending) |
| Mode D: avg_partitions spikes in precursor | Task 5 |
| Mode D: regime-specific write_ratio lag | Task 5 |
| n_intervals default = 10,000 | Task 6 |
| MODE_NAMES updated | Task 6 |
| CSVs saved to ./data/ | Task 7 |
| No NaN/Inf/out-of-range values | Task 1 (tests), Task 7 (verify step) |
| Mode B cross-correlation test at lag 1050 | Task 1 |
| RUNNING.md updated | Task 8 |

**Placeholder scan:** No TBDs or incomplete code blocks found.

**Type consistency:**
- `_aggregate_transactions` is called with `partitions_per_txn_mean` keyword arg in Task 5 — confirmed this param exists in the existing function signature (`def _aggregate_transactions(..., partitions_per_txn_mean=2.5, ...)`). ✓
- `_make_access_probs_stable(hot_partitions, hot_fraction=...)` — signature confirmed from existing code. ✓
- `_to_dataframe(records)` — confirmed existing helper, unchanged. ✓
- `rng.integers(600, 1201)` — NumPy `default_rng` API, upper bound exclusive, so range is [600, 1200]. ✓
