# Transaction-Sequence Workload Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the interval-aggregated workload design with a transaction-level CSV (date + p0..p9) and rewrite the dataset loader and evaluator to support a 4×4 (N, K) classification benchmark.

**Architecture:** `workload_generator.py` emits 360,000-row CSVs where each row is one transaction (sorted partition IDs 0–63, sentinel-padded to 10 slots). `workload_dataset.py` returns integer tensors for sliding-window (N, K) pairs. `evaluate.py` computes slot accuracy, set accuracy, top-3 accuracy, and CE loss across a 4×4 grid.

**Tech Stack:** NumPy, Pandas, PyTorch, pytest, scipy (cross-correlation tests only)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `brain/modelgen/workload_generator.py` | Rewrite | Transaction-level generators for modes A–D; 360k-row CSV; CLI |
| `brain/modelgen/tests/test_workload_generator.py` | Rewrite | TDD: schema tests (all modes) + structural tests per mode |
| `brain/modelgen/workload_dataset.py` | Rewrite | PyTorch Dataset for (N, K) sliding windows; integer tensors; 70/10/20 split |
| `brain/modelgen/evaluate.py` | Rewrite | Classification metrics + 4×4 grid evaluation + frequency-prior baseline |
| `brain/modelgen/RUNNING.md` | Update | CLI examples and mode descriptions |

---

## Task 1: Rewrite workload_generator.py skeleton + schema tests

**Files:**
- Rewrite: `brain/modelgen/workload_generator.py`
- Rewrite: `brain/modelgen/tests/test_workload_generator.py`

- [ ] **Step 1.1: Write failing schema tests**

Replace `brain/modelgen/tests/test_workload_generator.py` entirely:

```python
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from workload_generator import generate_mode, SENTINEL, SLOTS, NUM_PARTITIONS

MODES = ['A', 'B', 'C', 'D']
SMALL_N = 500


@pytest.mark.parametrize('mode', MODES)
def test_schema_shape(mode):
    df = generate_mode(mode, n=SMALL_N, seed=42)
    assert df.shape == (SMALL_N, 11), f"Expected ({SMALL_N}, 11), got {df.shape}"


@pytest.mark.parametrize('mode', MODES)
def test_no_nan(mode):
    df = generate_mode(mode, n=SMALL_N, seed=42)
    assert not df.isnull().any().any()


@pytest.mark.parametrize('mode', MODES)
def test_partition_ids_in_range(mode):
    df = generate_mode(mode, n=SMALL_N, seed=42)
    cols = [f'p{i}' for i in range(SLOTS)]
    vals = df[cols].values
    assert ((vals >= 0) & (vals <= SENTINEL)).all()


@pytest.mark.parametrize('mode', MODES)
def test_sorted_ascending(mode):
    df = generate_mode(mode, n=SMALL_N, seed=42)
    cols = [f'p{i}' for i in range(SLOTS)]
    for row in df[cols].values:
        non_s = row[row != SENTINEL]
        if len(non_s) > 1:
            assert np.all(non_s[:-1] <= non_s[1:]), f"Not sorted: {row}"


@pytest.mark.parametrize('mode', MODES)
def test_sentinel_contiguous(mode):
    """Once sentinel appears, all remaining slots must be sentinel."""
    df = generate_mode(mode, n=SMALL_N, seed=42)
    cols = [f'p{i}' for i in range(SLOTS)]
    for row in df[cols].values:
        sentinel_seen = False
        for v in row:
            if sentinel_seen:
                assert v == SENTINEL, f"Non-sentinel after sentinel: {row}"
            if v == SENTINEL:
                sentinel_seen = True


@pytest.mark.parametrize('mode', MODES)
def test_at_least_one_real_partition(mode):
    df = generate_mode(mode, n=SMALL_N, seed=42)
    assert (df['p0'] != SENTINEL).all(), "Some transactions access no partitions"


@pytest.mark.parametrize('mode', MODES)
def test_date_column_is_datetime(mode):
    df = generate_mode(mode, n=SMALL_N, seed=42)
    assert pd.api.types.is_datetime64_any_dtype(df['date'])


@pytest.mark.parametrize('mode', MODES)
def test_date_monotonic(mode):
    df = generate_mode(mode, n=SMALL_N, seed=42)
    assert df['date'].is_monotonic_increasing
```

- [ ] **Step 1.2: Run tests to verify they fail with ImportError**

```bash
cd /Users/andrew/2025/lion-tkde/brain/modelgen
/Users/andrew/2025/lion-tkde/venv/bin/python -m pytest tests/test_workload_generator.py::test_schema_shape -v 2>&1 | head -20
```

Expected: `ERROR` — `ImportError` or `ModuleNotFoundError` (generate_mode does not exist yet).

- [ ] **Step 1.3: Write the skeleton workload_generator.py**

Replace `brain/modelgen/workload_generator.py` entirely:

```python
"""
Transaction-Level Workload Generator for OLTP Prediction

Each CSV row represents one transaction: up to 10 partition IDs sorted
ascending, padded with SENTINEL (64) for unused slots.

CSV format:  date, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9

Workload Modes:
  A - harmonic_superposition  : multi-frequency sinusoidal access patterns
  B - phase_cascade           : cascaded delay chain across 8 partition groups
  C - frequency_modulated     : hotspot rotation speed modulated by slow envelope
  D - regime_with_precursors  : regime shifts preceded by 3-signal co-occurring precursor

Usage:
  python workload_generator.py --mode A --out-dir data/
  python workload_generator.py --mode all --out-dir data/
"""

import argparse
import os
import numpy as np
import pandas as pd

NUM_PARTITIONS = 64
SENTINEL       = 64          # padding value for unused partition slots
SLOTS          = 10          # max partitions per transaction
TXN_PER_SECOND = 50
DURATION_HOURS = 2
N_TRANSACTIONS = TXN_PER_SECOND * 3600 * DURATION_HOURS   # 360_000
START_TIME     = pd.Timestamp('2026-01-01 00:00:00')

MODE_NAMES = {
    'A': 'harmonic_superposition',
    'B': 'phase_cascade',
    'C': 'frequency_modulated',
    'D': 'regime_with_precursors',
}


def get_feature_columns():
    """Return the 10 partition slot column names p0..p9."""
    return [f'p{i}' for i in range(SLOTS)]


def _make_timestamps(n):
    """Return n Timestamps at TXN_PER_SECOND Hz starting from START_TIME."""
    freq = pd.Timedelta(seconds=1.0 / TXN_PER_SECOND)
    return pd.date_range(start=START_TIME, periods=n, freq=freq)


def _sample_txn(probs, k, rng):
    """
    Sample one transaction: choose k distinct partition IDs, sort, sentinel-pad.

    Parameters
    ----------
    probs : (NUM_PARTITIONS,) probability vector (must sum to 1)
    k     : int, number of partitions to access (1 ≤ k ≤ SLOTS)
    rng   : np.random.Generator

    Returns
    -------
    row : (SLOTS,) int32 array, sorted partition IDs then SENTINEL padding
    """
    chosen = rng.choice(NUM_PARTITIONS, size=k, replace=False, p=probs)
    chosen.sort()
    row = np.full(SLOTS, SENTINEL, dtype=np.int32)
    row[:k] = chosen
    return row


def _n_partitions(rng):
    """Sample number of partitions per transaction: Poisson(3) clipped to [1, SLOTS]."""
    return int(np.clip(rng.poisson(3), 1, SLOTS))


def _build_df(rows, n):
    """Stack list of (SLOTS,) arrays into a DataFrame with a leading date column."""
    df = pd.DataFrame(np.stack(rows), columns=get_feature_columns())
    df.insert(0, 'date', _make_timestamps(n))
    return df


# ── Mode stubs (implemented in Tasks 2–5) ────────────────────────────────────

def _generate_A(n, seed, return_metadata):
    raise NotImplementedError("Mode A not yet implemented")

def _generate_B(n, seed, return_metadata):
    raise NotImplementedError("Mode B not yet implemented")

def _generate_C(n, seed, return_metadata):
    raise NotImplementedError("Mode C not yet implemented")

def _generate_D(n, seed, return_metadata):
    raise NotImplementedError("Mode D not yet implemented")


MODE_GENERATORS = {
    'A': _generate_A,
    'B': _generate_B,
    'C': _generate_C,
    'D': _generate_D,
}


def generate_mode(mode, n=N_TRANSACTIONS, seed=42, return_metadata=False):
    """
    Generate n transactions for the given mode.

    Parameters
    ----------
    mode            : str, one of 'A', 'B', 'C', 'D'
    n               : int, number of transactions
    seed            : int, random seed
    return_metadata : bool, if True return (df, dict) with mode-specific metadata

    Returns
    -------
    pd.DataFrame  or  (pd.DataFrame, dict)
    """
    if mode not in MODE_GENERATORS:
        raise ValueError(f"Unknown mode {mode!r}. Choose from {list(MODE_GENERATORS)}")
    return MODE_GENERATORS[mode](n=n, seed=seed, return_metadata=return_metadata)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="Generate synthetic OLTP transaction workloads")
    parser.add_argument('--mode', default='all',
                        choices=['A', 'B', 'C', 'D', 'all'])
    parser.add_argument('--n-transactions', type=int, default=N_TRANSACTIONS,
                        help=f'Number of transactions (default {N_TRANSACTIONS})')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default='data')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    modes = list(MODE_GENERATORS) if args.mode == 'all' else [args.mode]
    for m in modes:
        df = generate_mode(m, n=args.n_transactions, seed=args.seed)
        fname = f"workload_{MODE_NAMES[m]}.csv"
        path = os.path.join(args.out_dir, fname)
        df.to_csv(path, index=False)
        print(f"Wrote {len(df):,} transactions to {path}")


if __name__ == '__main__':
    _cli()
```

- [ ] **Step 1.4: Run schema tests — expect NotImplementedError, not ImportError**

```bash
cd /Users/andrew/2025/lion-tkde/brain/modelgen
/Users/andrew/2025/lion-tkde/venv/bin/python -m pytest tests/test_workload_generator.py::test_schema_shape -v 2>&1 | head -20
```

Expected: `FAILED` with `NotImplementedError: Mode A not yet implemented` (import succeeds).

- [ ] **Step 1.5: Commit**

```bash
cd /Users/andrew/2025/lion-tkde
git add brain/modelgen/workload_generator.py brain/modelgen/tests/test_workload_generator.py
git commit -m "feat: skeleton transaction-level workload_generator with schema tests"
```

---

## Task 2: Implement Mode A — harmonic_superposition

**Files:**
- Modify: `brain/modelgen/workload_generator.py` (replace `_generate_A` stub)
- Modify: `brain/modelgen/tests/test_workload_generator.py` (append structural test)

- [ ] **Step 2.1: Append Mode A structural test to test file**

Add to the end of `brain/modelgen/tests/test_workload_generator.py`:

```python
def test_mode_a_harmonic_structure():
    """Per-partition access indicator should show a spectral peak at period 96."""
    n = 5000
    df = generate_mode('A', n=n, seed=42)
    cols = [f'p{i}' for i in range(SLOTS)]
    # Build access indicator for partition 0
    access = np.zeros(n)
    for col in cols:
        access += (df[col].values == 0).astype(float)
    access -= access.mean()
    fft_mag = np.abs(np.fft.rfft(access))
    # Bin index for period 96 (frequency = n/96 in the rfft output)
    idx = int(round(n / 96))
    # Should be a local maximum relative to its neighbours
    assert fft_mag[idx] > fft_mag[idx - 1], \
        f"No left-side peak at period 96 (bin {idx})"
    assert fft_mag[idx] > fft_mag[idx + 1], \
        f"No right-side peak at period 96 (bin {idx})"
```

- [ ] **Step 2.2: Run to verify it fails**

```bash
cd /Users/andrew/2025/lion-tkde/brain/modelgen
/Users/andrew/2025/lion-tkde/venv/bin/python -m pytest tests/test_workload_generator.py::test_mode_a_harmonic_structure -v
```

Expected: `FAILED` — `NotImplementedError`.

- [ ] **Step 2.3: Replace `_generate_A` stub in workload_generator.py**

```python
def _generate_A(n, seed, return_metadata):
    """
    Mode A — harmonic_superposition

    Access probability of each partition oscillates as a sum of 4 sinusoids:
      periods     = [96, 960, 4800, 19200]
      amplitudes  = [0.15, 0.30, 0.25, 0.20]  (as multiples of base prob)

    Each partition has an independent random phase per component.
    """
    rng        = np.random.default_rng(seed)
    periods    = np.array([96, 960, 4800, 19200], dtype=float)
    amplitudes = np.array([0.15, 0.30, 0.25, 0.20])
    base       = 1.0 / NUM_PARTITIONS

    # phases[p, j] = random phase for partition p and sinusoidal component j
    phases = rng.uniform(0, 2 * np.pi, size=(NUM_PARTITIONS, len(periods)))

    rows = []
    for t in range(n):
        probs = np.ones(NUM_PARTITIONS) * base
        for j, (P, A) in enumerate(zip(periods, amplitudes)):
            probs += base * A * np.sin(2 * np.pi * t / P + phases[:, j])
        probs = np.maximum(probs, 1e-9)
        probs /= probs.sum()
        rows.append(_sample_txn(probs, _n_partitions(rng), rng))

    df = _build_df(rows, n)
    if return_metadata:
        return df, {'periods': periods.tolist(), 'phases': phases}
    return df
```

- [ ] **Step 2.4: Run all Mode A tests**

```bash
cd /Users/andrew/2025/lion-tkde/brain/modelgen
/Users/andrew/2025/lion-tkde/venv/bin/python -m pytest tests/test_workload_generator.py \
  -k "A] or test_mode_a" -v
```

Expected: all 9 tests (`test_schema_shape[A]`, `test_no_nan[A]`, `test_sorted_ascending[A]`, `test_sentinel_contiguous[A]`, `test_partition_ids_in_range[A]`, `test_at_least_one_real_partition[A]`, `test_date_column_is_datetime[A]`, `test_date_monotonic[A]`, `test_mode_a_harmonic_structure`) **PASS**.

- [ ] **Step 2.5: Commit**

```bash
cd /Users/andrew/2025/lion-tkde
git add brain/modelgen/workload_generator.py brain/modelgen/tests/test_workload_generator.py
git commit -m "feat: implement Mode A harmonic_superposition"
```

---

## Task 3: Implement Mode B — phase_cascade

**Files:**
- Modify: `brain/modelgen/workload_generator.py` (replace `_generate_B` stub)
- Modify: `brain/modelgen/tests/test_workload_generator.py` (append structural test)

- [ ] **Step 3.1: Append Mode B cross-correlation test**

```python
def test_mode_b_phase_cascade():
    """
    Group 7 access pattern is a delayed copy of Group 0 with lag 560.
    Cross-correlation should peak near lag 560 in a local search window.
    """
    from scipy.signal import correlate, correlation_lags
    n = 5000
    df = generate_mode('B', n=n, seed=42)
    cols = [f'p{i}' for i in range(SLOTS)]

    group0_ids = set(range(0, 8))
    group7_ids = set(range(56, 64))

    g0 = np.zeros(n, dtype=float)
    g7 = np.zeros(n, dtype=float)
    for col in cols:
        g0 += df[col].isin(group0_ids).astype(float)
        g7 += df[col].isin(group7_ids).astype(float)
    g0 -= g0.mean()
    g7 -= g7.mean()

    # correlate(g7, g0): peak at lag L means g7(t) ≈ g0(t - L)
    corr = correlate(g7, g0, mode='full')
    lags = correlation_lags(n, n, mode='full')

    TARGET = 560
    HALF_PERIOD = 60   # half of group-0 period (120), narrows search without aliasing
    mask = (lags >= TARGET - HALF_PERIOD) & (lags <= TARGET + HALF_PERIOD)
    peak_lag = int(lags[mask][np.argmax(corr[mask])])
    assert abs(peak_lag - TARGET) <= 5, \
        f"Expected local cross-corr peak at lag {TARGET}, got {peak_lag}"
```

- [ ] **Step 3.2: Run to verify it fails**

```bash
cd /Users/andrew/2025/lion-tkde/brain/modelgen
/Users/andrew/2025/lion-tkde/venv/bin/python -m pytest tests/test_workload_generator.py::test_mode_b_phase_cascade -v
```

Expected: `FAILED` — `NotImplementedError`.

- [ ] **Step 3.3: Replace `_generate_B` stub**

```python
def _generate_B(n, seed, return_metadata):
    """
    Mode B — phase_cascade

    64 partitions in 8 groups of 8.
    Group 0 has sinusoidal activity at period 120 transactions.
    Group i receives a delayed copy of Group 0 with lag i × 80 transactions.
    Max lag (Group 7) = 560, within the N=720 encoder window.
    """
    rng        = np.random.default_rng(seed)
    N_GROUPS   = 8
    GROUP_SIZE = NUM_PARTITIONS // N_GROUPS   # 8
    PERIOD     = 120
    LAG        = 80
    BASE       = 0.1 / NUM_PARTITIONS
    HOT        = 0.9 / GROUP_SIZE

    groups = [list(range(i * GROUP_SIZE, (i + 1) * GROUP_SIZE))
              for i in range(N_GROUPS)]

    rows = []
    for t in range(n):
        probs = np.full(NUM_PARTITIONS, BASE)
        for i, grp in enumerate(groups):
            activity = 0.5 + 0.5 * np.sin(2 * np.pi * (t - i * LAG) / PERIOD)
            for p in grp:
                probs[p] = BASE + activity * (HOT - BASE)
        probs = np.maximum(probs, 1e-9)
        probs /= probs.sum()
        rows.append(_sample_txn(probs, _n_partitions(rng), rng))

    df = _build_df(rows, n)
    if return_metadata:
        return df, {'groups': groups, 'period': PERIOD, 'lag': LAG}
    return df
```

- [ ] **Step 3.4: Run all Mode B tests**

```bash
cd /Users/andrew/2025/lion-tkde/brain/modelgen
/Users/andrew/2025/lion-tkde/venv/bin/python -m pytest tests/test_workload_generator.py \
  -k "B] or test_mode_b" -v
```

Expected: all 9 tests PASS.

- [ ] **Step 3.5: Commit**

```bash
cd /Users/andrew/2025/lion-tkde
git add brain/modelgen/workload_generator.py brain/modelgen/tests/test_workload_generator.py
git commit -m "feat: implement Mode B phase_cascade"
```

---

## Task 4: Implement Mode C — frequency_modulated

**Files:**
- Modify: `brain/modelgen/workload_generator.py` (replace `_generate_C` stub)
- Modify: `brain/modelgen/tests/test_workload_generator.py` (append structural test)

- [ ] **Step 4.1: Append Mode C period-variation test**

```python
def test_mode_c_period_variation():
    """
    Inter-switch intervals between hotspot sets must vary significantly,
    reflecting the slow 8000-transaction modulation envelope.
    """
    n = 5000
    df = generate_mode('C', n=n, seed=42)
    cols = [f'p{i}' for i in range(SLOTS)]

    # 8 hotspot sets, each covering 8 consecutive partitions
    hotspot_sets = [set(range(i * 8, (i + 1) * 8)) for i in range(8)]

    def dominant_set(row):
        vals = set(int(v) for v in row if v != SENTINEL)
        counts = [len(vals & s) for s in hotspot_sets]
        return int(np.argmax(counts))

    active = np.array([dominant_set(row) for row in df[cols].values])
    switches = np.where(np.diff(active) != 0)[0]
    assert len(switches) >= 3, f"Too few hotspot switches: {len(switches)}"
    inter = np.diff(switches)
    assert inter.std() > 10, \
        f"Switching period appears constant (std={inter.std():.2f}); modulation not working"
    assert inter.min() >= 40, \
        f"Switching too fast (min={inter.min()}); P_inst floor not respected"
```

- [ ] **Step 4.2: Run to verify it fails**

```bash
cd /Users/andrew/2025/lion-tkde/brain/modelgen
/Users/andrew/2025/lion-tkde/venv/bin/python -m pytest tests/test_workload_generator.py::test_mode_c_period_variation -v
```

Expected: `FAILED` — `NotImplementedError`.

- [ ] **Step 4.3: Replace `_generate_C` stub**

```python
def _generate_C(n, seed, return_metadata):
    """
    Mode C — frequency_modulated

    Hotspot rotation speed is modulated by a slow 8000-transaction envelope:
      P_inst(t) = 400 + 1200 * sin(2π * t / 8000),  clipped to [160, 1600]

    Phase accumulator drives set selection:
      phase += 2π / P_inst(t)
      active_set = floor(phase / 2π) % N_SETS
    """
    rng      = np.random.default_rng(seed)
    N_SETS   = 8
    GRP_SIZE = NUM_PARTITIONS // N_SETS   # 8
    P_MID    = 400.0
    P_AMP    = 1200.0
    P_ENV    = 8000.0
    P_MIN    = 160.0
    P_MAX    = 1600.0
    HOT      = 0.9 / GRP_SIZE
    BASE     = 0.1 / NUM_PARTITIONS

    sets = [list(range(i * GRP_SIZE, (i + 1) * GRP_SIZE)) for i in range(N_SETS)]

    phase_acc = 0.0
    phase_log = []
    rows = []

    for t in range(n):
        P_inst = P_MID + P_AMP * np.sin(2 * np.pi * t / P_ENV)
        P_inst = float(np.clip(P_inst, P_MIN, P_MAX))
        phase_acc += 2 * np.pi / P_inst
        phase_log.append(phase_acc)

        active_idx = int(phase_acc / (2 * np.pi)) % N_SETS
        probs = np.full(NUM_PARTITIONS, BASE)
        for p in sets[active_idx]:
            probs[p] = HOT
        probs /= probs.sum()

        rows.append(_sample_txn(probs, _n_partitions(rng), rng))

    df = _build_df(rows, n)
    if return_metadata:
        return df, {'phase_log': phase_log, 'P_mid': P_MID,
                    'P_amp': P_AMP, 'P_env': P_ENV}
    return df
```

- [ ] **Step 4.4: Run all Mode C tests**

```bash
cd /Users/andrew/2025/lion-tkde/brain/modelgen
/Users/andrew/2025/lion-tkde/venv/bin/python -m pytest tests/test_workload_generator.py \
  -k "C] or test_mode_c" -v
```

Expected: all 9 tests PASS.

- [ ] **Step 4.5: Commit**

```bash
cd /Users/andrew/2025/lion-tkde
git add brain/modelgen/workload_generator.py brain/modelgen/tests/test_workload_generator.py
git commit -m "feat: implement Mode C frequency_modulated"
```

---

## Task 5: Implement Mode D — regime_with_precursors

**Files:**
- Modify: `brain/modelgen/workload_generator.py` (replace `_generate_D` stub)
- Modify: `brain/modelgen/tests/test_workload_generator.py` (append structural tests)

- [ ] **Step 5.1: Append Mode D precursor and regime-variation tests**

```python
def test_mode_d_precursor_lead_time():
    """Each transition must be preceded by a precursor exactly 200 transactions earlier."""
    n = 5000
    df, meta = generate_mode('D', n=n, seed=42, return_metadata=True)
    transitions     = meta['transitions']       # list of t where regime changes
    precursor_starts = meta['precursor_starts'] # list of t where precursor begins
    assert len(transitions) >= 2, \
        f"Need ≥2 transitions in {n} transactions, got {len(transitions)}"
    # zip truncates to shorter list (last precursor may have no transition if near end)
    for t_trans, t_pre in zip(transitions, precursor_starts):
        diff = t_trans - t_pre
        assert diff == 200, \
            f"Precursor lead time should be 200, got {diff} (trans={t_trans}, pre={t_pre})"


def test_mode_d_regime_variation():
    """Dominant partition ranges should differ between first and last quarter."""
    n = 5000
    df = generate_mode('D', n=n, seed=42)
    cols = [f'p{i}' for i in range(SLOTS)]
    fq = df.iloc[:n // 4][cols].values.flatten()
    lq = df.iloc[3 * n // 4:][cols].values.flatten()
    fq_parts = set(fq[fq != SENTINEL])
    lq_parts = set(lq[lq != SENTINEL])
    overlap = len(fq_parts & lq_parts) / max(len(fq_parts | lq_parts), 1)
    assert overlap < 0.9, \
        f"First and last quarters share too many partitions (overlap={overlap:.2f})"
```

- [ ] **Step 5.2: Run to verify they fail**

```bash
cd /Users/andrew/2025/lion-tkde/brain/modelgen
/Users/andrew/2025/lion-tkde/venv/bin/python -m pytest tests/test_workload_generator.py \
  -k "test_mode_d" -v
```

Expected: `FAILED` — `NotImplementedError`.

- [ ] **Step 5.3: Replace `_generate_D` stub**

```python
def _generate_D(n, seed, return_metadata):
    """
    Mode D — regime_with_precursors

    5 regimes, each owning ~12-13 non-overlapping partitions.
    Regime dwell: uniform [600, 1200] transactions.
    Regime transitions are preceded by a 200-transaction precursor:
      - Current-regime partition probability fades linearly from HOT to BASE
      - Next-regime partition probability rises linearly from BASE to HOT
      - Transactions touch more partitions (k biased upward)
    """
    rng          = np.random.default_rng(seed)
    N_REGIMES    = 5
    PRECURSOR    = 200
    DWELL_MIN    = 600
    DWELL_MAX    = 1200
    HOT          = 0.85
    BASE_OTHER   = 0.01 / NUM_PARTITIONS

    # Partition ownership: 5 groups covering all 64 partitions
    sizes      = [13, 13, 13, 13, 12]
    boundaries = np.cumsum([0] + sizes)
    r_parts    = [list(range(boundaries[i], boundaries[i + 1]))
                  for i in range(N_REGIMES)]

    rows             = []
    transitions      = []
    precursor_starts = []

    regime = 0
    dwell  = int(rng.integers(DWELL_MIN, DWELL_MAX + 1))
    age    = 0   # incremented at top of loop

    for t in range(n):
        age += 1
        next_regime       = (regime + 1) % N_REGIMES
        pre_start_age     = dwell - PRECURSOR
        in_precursor      = age >= pre_start_age
        progress          = max(0.0, (age - pre_start_age) / PRECURSOR)  # 0 → 1

        # Log precursor start (only if the transition will occur within the sequence)
        if age == pre_start_age and t + PRECURSOR < n:
            precursor_starts.append(t)

        # Build access probabilities: current regime fades, next rises
        probs = np.full(NUM_PARTITIONS, BASE_OTHER)
        for p in r_parts[regime]:
            probs[p] = BASE_OTHER + (1 - progress) * (
                HOT / len(r_parts[regime]) - BASE_OTHER)
        for p in r_parts[next_regime]:
            probs[p] = BASE_OTHER + progress * (
                HOT / len(r_parts[next_regime]) - BASE_OTHER)
        probs = np.maximum(probs, 1e-9)
        probs /= probs.sum()

        # During precursor, transactions touch more partitions
        k = (int(np.clip(rng.poisson(4) + 1, 2, SLOTS))
             if in_precursor else _n_partitions(rng))
        rows.append(_sample_txn(probs, k, rng))

        # Regime transition
        if age >= dwell:
            transitions.append(t)
            regime = next_regime
            dwell  = int(rng.integers(DWELL_MIN, DWELL_MAX + 1))
            age    = 0

    df = _build_df(rows, n)
    if return_metadata:
        meta = {
            'transitions':      transitions,
            'precursor_starts': precursor_starts,
            'regime_partitions': r_parts,
        }
        return df, meta
    return df
```

- [ ] **Step 5.4: Run all Mode D tests**

```bash
cd /Users/andrew/2025/lion-tkde/brain/modelgen
/Users/andrew/2025/lion-tkde/venv/bin/python -m pytest tests/test_workload_generator.py \
  -k "D] or test_mode_d" -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5.5: Run the full test suite**

```bash
cd /Users/andrew/2025/lion-tkde/brain/modelgen
/Users/andrew/2025/lion-tkde/venv/bin/python -m pytest tests/test_workload_generator.py -v
```

Expected: **all tests PASS** (8 parametrized schema tests × 4 modes + 4 structural tests = 36 total).

- [ ] **Step 5.6: Commit**

```bash
cd /Users/andrew/2025/lion-tkde
git add brain/modelgen/workload_generator.py brain/modelgen/tests/test_workload_generator.py
git commit -m "feat: implement Mode D regime_with_precursors"
```

---

## Task 6: Rewrite workload_dataset.py

**Files:**
- Rewrite: `brain/modelgen/workload_dataset.py`

- [ ] **Step 6.1: Replace workload_dataset.py**

```python
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

SLOTS          = 10
SENTINEL       = 64
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
        self.csv_path   = csv_path
        self.seq_len    = seq_len
        self.label_len  = label_len
        self.pred_len   = pred_len
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _ds(self, flag):
        return WorkloadDataset(self.csv_path, flag=flag, seq_len=self.seq_len,
                               label_len=self.label_len, pred_len=self.pred_len)

    def train_loader(self):
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
```

- [ ] **Step 6.2: Smoke-test the dataset**

```bash
cd /Users/andrew/2025/lion-tkde/brain/modelgen
/Users/andrew/2025/lion-tkde/venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, '.')
import tempfile, os, torch
from workload_generator import generate_mode
from workload_dataset import WorkloadDataset

df = generate_mode('A', n=2000, seed=42)
with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
    df.to_csv(f, index=False); path = f.name

ds = WorkloadDataset(path, flag='train', seq_len=96, label_len=48, pred_len=48)
print(f"Train samples: {len(ds)}")
sx, sy, sxm, sym = ds[0]
assert sx.shape  == torch.Size([96, 10]),  sx.shape
assert sy.shape  == torch.Size([96, 10]),  sy.shape   # label_len(48)+pred_len(48)
assert sxm.shape == torch.Size([96,  2]),  sxm.shape
assert sym.shape == torch.Size([96,  2]),  sym.shape
assert sx.dtype  == torch.int64,           sx.dtype
assert sxm.dtype == torch.float32,         sxm.dtype
os.unlink(path)
print("workload_dataset smoke test: OK")
EOF
```

Expected output:
```
Train samples: <positive integer>
workload_dataset smoke test: OK
```

- [ ] **Step 6.3: Commit**

```bash
cd /Users/andrew/2025/lion-tkde
git add brain/modelgen/workload_dataset.py
git commit -m "feat: rewrite workload_dataset for transaction-level integer tensors"
```

---

## Task 7: Rewrite evaluate.py

**Files:**
- Rewrite: `brain/modelgen/evaluate.py`

- [ ] **Step 7.1: Replace evaluate.py**

```python
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

SENTINEL      = 64
SLOTS         = 10
N_PARTITIONS  = 64


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
    correct = sum(
        set(pred[k][pred[k] != SENTINEL]) == set(target[k][target[k] != SENTINEL])
        for k in range(K)
    )
    return correct / K


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
    mask         = (target != SENTINEL).flatten()
    flat_logits  = torch.tensor(logits.reshape(-1, N_PARTITIONS)[mask], dtype=torch.float32)
    flat_target  = torch.tensor(target.flatten()[mask], dtype=torch.long)
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
    predict : callable  (target: np.ndarray (K,10)) → pred: np.ndarray (K,10)
    """
    flat   = train_targets.flatten()
    flat   = flat[flat != SENTINEL]
    ids, counts = np.unique(flat, return_counts=True)
    top10  = np.sort(ids[np.argsort(-counts)[:SLOTS]])
    row    = np.full(SLOTS, SENTINEL, dtype=np.int32)
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
            L   = label_len_fn(N)
            dm  = WorkloadDataModule(csv_path, seq_len=N, label_len=L,
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

            results[(N, K)] = {
                'slot_acc': float(np.nanmean([
                    slot_accuracy(pred_ids[i], targets_arr[i])
                    for i in range(len(pred_ids))])),
                'set_acc':  float(np.nanmean([
                    set_accuracy(pred_ids[i], targets_arr[i])
                    for i in range(len(pred_ids))])),
                'top3_acc': float(np.nanmean([
                    top3_accuracy(logits_arr[i], targets_arr[i])
                    for i in range(len(logits_arr))])),
                'ce_loss':  cross_entropy_loss(
                    logits_arr.reshape(-1, SLOTS, N_PARTITIONS),
                    targets_arr.reshape(-1, SLOTS)),
            }
    return results
```

- [ ] **Step 7.2: Smoke-test evaluate.py**

```bash
cd /Users/andrew/2025/lion-tkde/brain/modelgen
/Users/andrew/2025/lion-tkde/venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, '.')
import numpy as np
from evaluate import (slot_accuracy, set_accuracy, top3_accuracy,
                      cross_entropy_loss, frequency_prior_baseline, SENTINEL, SLOTS)

K = 10
perfect = np.array([[1, 5, 10, 64, 64, 64, 64, 64, 64, 64]] * K)
wrong   = np.array([[2, 6, 11, 64, 64, 64, 64, 64, 64, 64]] * K)

assert slot_accuracy(perfect, perfect) == 1.0
assert slot_accuracy(wrong,   perfect) == 0.0
assert set_accuracy(perfect,  perfect) == 1.0
assert set_accuracy(wrong,    perfect) == 0.0

logits  = np.random.randn(K, SLOTS, 64)
logits[:, :3, 1] = 100.0   # force partition 1 to rank 1 for first 3 slots
target  = np.full((K, SLOTS), SENTINEL, dtype=np.int32)
target[:, :3] = 1
assert top3_accuracy(logits, target) == 1.0

train  = np.array([[3, 7, SENTINEL] + [SENTINEL] * 7] * 50, dtype=np.int32)
pred_fn = frequency_prior_baseline(train)
out    = pred_fn(perfect)
assert out.shape == perfect.shape

print("evaluate.py smoke test: OK")
EOF
```

Expected: `evaluate.py smoke test: OK`

- [ ] **Step 7.3: Commit**

```bash
cd /Users/andrew/2025/lion-tkde
git add brain/modelgen/evaluate.py
git commit -m "feat: rewrite evaluate.py for classification metrics and 4x4 grid"
```

---

## Task 8: Update RUNNING.md, generate datasets, integrity check

**Files:**
- Update: `brain/modelgen/RUNNING.md`

- [ ] **Step 8.1: Replace RUNNING.md content**

```markdown
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

## Running Tests

```bash
cd brain/modelgen
python -m pytest tests/test_workload_generator.py -v
```
```

- [ ] **Step 8.2: Generate all four datasets**

```bash
cd /Users/andrew/2025/lion-tkde/brain/modelgen
/Users/andrew/2025/lion-tkde/venv/bin/python workload_generator.py --mode all --out-dir data/
```

Expected (one line per mode, ~several minutes total):
```
Wrote 360,000 transactions to data/workload_harmonic_superposition.csv
Wrote 360,000 transactions to data/workload_phase_cascade.csv
Wrote 360,000 transactions to data/workload_frequency_modulated.csv
Wrote 360,000 transactions to data/workload_regime_with_precursors.csv
```

- [ ] **Step 8.3: Run integrity check**

```bash
cd /Users/andrew/2025/lion-tkde
/Users/andrew/2025/lion-tkde/venv/bin/python - <<'EOF'
import pandas as pd, numpy as np, pathlib, sys
SENTINEL = 64
data_dir = pathlib.Path("brain/modelgen/data")
fnames   = [
    "workload_harmonic_superposition.csv",
    "workload_phase_cascade.csv",
    "workload_frequency_modulated.csv",
    "workload_regime_with_precursors.csv",
]
errors = []
for fname in fnames:
    df   = pd.read_csv(data_dir / fname, parse_dates=['date'])
    cols = [f'p{i}' for i in range(10)]
    if df.shape != (360_000, 11):
        errors.append(f"{fname}: shape {df.shape}")
    if df.isnull().any().any():
        errors.append(f"{fname}: NaN values")
    vals = df[cols].values
    if not ((vals >= 0) & (vals <= SENTINEL)).all():
        errors.append(f"{fname}: values out of range")
    for row in vals[:1000]:
        non_s = row[row != SENTINEL]
        if len(non_s) > 1 and not (non_s[:-1] <= non_s[1:]).all():
            errors.append(f"{fname}: not sorted"); break
    print(f"OK  {fname}: {len(df):,} rows")
if errors:
    print("ERRORS:", errors); sys.exit(1)
print("All integrity checks passed.")
EOF
```

Expected:
```
OK  workload_harmonic_superposition.csv: 360,000 rows
OK  workload_phase_cascade.csv: 360,000 rows
OK  workload_frequency_modulated.csv: 360,000 rows
OK  workload_regime_with_precursors.csv: 360,000 rows
All integrity checks passed.
```

- [ ] **Step 8.4: Run full test suite**

```bash
cd /Users/andrew/2025/lion-tkde/brain/modelgen
/Users/andrew/2025/lion-tkde/venv/bin/python -m pytest tests/test_workload_generator.py -v
```

Expected: all 36 tests PASS.

- [ ] **Step 8.5: Commit RUNNING.md**

```bash
cd /Users/andrew/2025/lion-tkde
git add -f brain/modelgen/RUNNING.md
git commit -m "docs: update RUNNING.md for transaction-level workload design"
```
