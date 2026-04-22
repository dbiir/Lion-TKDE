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


def _build_df(rows):
    """Stack list of (SLOTS,) arrays into a DataFrame with a leading date column."""
    n = len(rows)
    df = pd.DataFrame(np.stack(rows), columns=get_feature_columns())
    df.insert(0, 'date', _make_timestamps(n))
    return df


# ── Mode stubs (implemented in Tasks 2–5) ────────────────────────────────────

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

    # Vectorize probability computation: shape (n, NUM_PARTITIONS)
    t_arr = np.arange(n)
    all_probs = np.ones((n, NUM_PARTITIONS)) * base
    for j in range(len(periods)):
        angle = 2 * np.pi * t_arr[:, None] / periods[j] + phases[None, :, j]
        all_probs += base * amplitudes[j] * np.sin(angle)
    all_probs = np.maximum(all_probs, 1e-9)  # guard against future amplitude changes
    all_probs /= all_probs.sum(axis=1, keepdims=True)

    rows = []
    for t in range(n):
        rows.append(_sample_txn(all_probs[t], _n_partitions(rng), rng))

    df = _build_df(rows)
    if return_metadata:
        return df, {'periods': periods.tolist(), 'phases': phases}
    return df

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

    df = _build_df(rows)
    if return_metadata:
        return df, {'groups': groups, 'period': PERIOD, 'lag': LAG}
    return df

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
    HOT      = 1.0 / GRP_SIZE
    NEAR_ZERO = 1e-9   # non-hotspot partitions are effectively inaccessible

    sets = [list(range(i * GRP_SIZE, (i + 1) * GRP_SIZE)) for i in range(N_SETS)]

    phase_acc = 0.0
    phase_log = [] if return_metadata else None
    rows = []

    for t in range(n):
        P_inst = P_MID + P_AMP * np.sin(2 * np.pi * t / P_ENV)
        P_inst = float(np.clip(P_inst, P_MIN, P_MAX))
        phase_acc += 2 * np.pi / P_inst
        if return_metadata:
            phase_log.append(phase_acc)

        active_idx = int(phase_acc / (2 * np.pi)) % N_SETS
        probs = np.full(NUM_PARTITIONS, NEAR_ZERO)
        for p in sets[active_idx]:
            probs[p] = HOT
        probs /= probs.sum()

        rows.append(_sample_txn(probs, _n_partitions(rng), rng))

    df = _build_df(rows)
    if return_metadata:
        return df, {'phase_log': phase_log, 'P_mid': P_MID,
                    'P_amp': P_AMP, 'P_env': P_ENV}
    return df

def _generate_D(n, seed, return_metadata):
    """
    Mode D — regime_with_precursors

    5 regimes, each owning ~12-13 non-overlapping partitions.
    Regime dwell: uniform [600, 1200] transactions.
    Regime transitions are preceded by a 200-transaction precursor driven by
    a single `progress` variable (0 → 1) that:
      - Fades current-regime probability linearly from HOT_FRAC to near-zero
      - Rises next-regime probability linearly from near-zero to HOT_FRAC
      - Biases k upward (Poisson(4)+1 vs baseline Poisson(3))

    This produces three correlated leading signals detectable by the model:
      1. Rising access entropy (distribution flattens as two regimes mix)
      2. Declining hotspot concentration (probability mass disperses)
      3. Spiking partition spread (k upward bias increases distinct IDs per transaction)

    All three are driven by the same underlying mechanism and are therefore
    fully correlated; they co-occur by construction.
    """
    rng          = np.random.default_rng(seed)
    N_REGIMES    = 5
    PRECURSOR    = 200
    DWELL_MIN    = 600
    DWELL_MAX    = 1200
    assert DWELL_MIN > PRECURSOR, \
        f"DWELL_MIN ({DWELL_MIN}) must exceed PRECURSOR ({PRECURSOR}) to prevent precursor bleed"
    HOT_FRAC     = 0.85   # total probability mass on the current-regime partitions
    NEAR_ZERO    = 1e-9   # background probability for inactive partitions

    # Partition ownership: 5 groups covering all 64 partitions
    sizes      = [13, 13, 13, 13, 12]
    boundaries = np.cumsum([0] + sizes)
    r_parts    = [list(range(int(boundaries[i]), int(boundaries[i + 1])))
                  for i in range(N_REGIMES)]

    rows             = []
    transitions      = []
    precursor_starts = []

    regime = 0
    dwell  = int(rng.integers(DWELL_MIN, DWELL_MAX + 1))
    age    = 0

    for t in range(n):
        age += 1
        next_regime   = (regime + 1) % N_REGIMES
        pre_start_age = dwell - PRECURSOR
        in_precursor  = age >= pre_start_age
        # progress goes 0→1 over the 200-transaction precursor window
        progress      = max(0.0, min(1.0, (age - pre_start_age) / PRECURSOR))

        # Log precursor start (only if the transition will occur within the sequence)
        if age == pre_start_age and t + PRECURSOR < n:
            precursor_starts.append(t)

        # Build access probabilities: current regime fades, next rises
        probs = np.full(NUM_PARTITIONS, NEAR_ZERO)
        cur_hot = HOT_FRAC * (1 - progress) / len(r_parts[regime])
        nxt_hot = HOT_FRAC * progress       / len(r_parts[next_regime])
        for p in r_parts[regime]:
            probs[p] = max(cur_hot, NEAR_ZERO)
        for p in r_parts[next_regime]:
            probs[p] += max(nxt_hot, NEAR_ZERO)
        probs /= probs.sum()

        # During precursor, transactions touch more partitions
        if in_precursor:
            k = int(np.clip(rng.poisson(4) + 1, 2, SLOTS))
        else:
            k = _n_partitions(rng)
        rows.append(_sample_txn(probs, k, rng))

        # Regime transition at end of dwell
        if age >= dwell:
            transitions.append(t)
            regime = next_regime
            dwell  = int(rng.integers(DWELL_MIN, DWELL_MAX + 1))
            age    = 0

    df = _build_df(rows)
    if return_metadata:
        meta = {
            'transitions':       transitions,
            'precursor_starts':  precursor_starts,
            'regime_partitions': r_parts,
        }
        return df, meta
    return df


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
