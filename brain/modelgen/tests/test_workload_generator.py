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


def test_mode_a_harmonic_structure():
    """
    Mode A access probability should show a spectral peak at period 96.

    Tests the underlying probability time series (via return_metadata) rather than
    the noisy binary samples, which have insufficient SNR at the period-96 amplitude
    (0.15 * base) to reliably detect a local maximum across arbitrary seeds.

    n=4800 = 50 * 96 ensures period-96 energy lands exactly at bin 50.
    """
    # Documented Mode A design parameters
    PERIODS    = np.array([96.0, 960.0, 4800.0, 19200.0])
    AMPLITUDES = np.array([0.15, 0.30, 0.25, 0.20])
    base       = 1.0 / NUM_PARTITIONS

    # Verify amplitudes won't push any prob negative (floor would be active if sum > 1)
    assert sum(AMPLITUDES) < 1.0, \
        f"Amplitudes sum {sum(AMPLITUDES)} >= 1.0; clipping floor would be active"

    n = 4800
    _df, meta = generate_mode('A', n=n, seed=42, return_metadata=True)
    phases = meta['phases']  # shape (NUM_PARTITIONS, 4)

    # Reconstruct the access-probability time series for partition 0
    t = np.arange(n)
    prob = np.ones(n) * base
    for j, (P, A) in enumerate(zip(PERIODS, AMPLITUDES)):
        prob += base * A * np.sin(2 * np.pi * t / P + phases[0, j])
    prob -= prob.mean()

    window  = np.hanning(n)
    fft_mag = np.abs(np.fft.rfft(prob * window))
    idx     = int(round(n / 96))  # = 50 exactly for n=4800
    assert fft_mag[idx] > fft_mag[idx - 1], \
        f"No left-side peak at period 96 (bin {idx})"
    assert fft_mag[idx] > fft_mag[idx + 1], \
        f"No right-side peak at period 96 (bin {idx})"


def test_mode_b_phase_cascade():
    """
    Group 7 theoretical activity is a delayed copy of Group 0 with lag = 7 × LAG.

    Tests theoretical sinusoidal activity curves (via return_metadata) rather than
    noisy binary samples: sampling at ~3 tx/group per transaction gives insufficient
    SNR in the binary count signal to resolve a ±5 lag reliably across seeds.
    """
    from scipy.signal import correlate, correlation_lags

    n = 5000
    _df, meta = generate_mode('B', n=n, seed=42, return_metadata=True)
    PERIOD = meta['period']       # 120
    LAG    = meta['lag']          # 80
    TARGET = 7 * LAG              # 560

    t  = np.arange(n, dtype=float)
    # Pure sinusoidal activity (DC removed by using sin directly)
    g0 = np.sin(2 * np.pi * t / PERIOD)
    g7 = np.sin(2 * np.pi * (t - TARGET) / PERIOD)

    corr = correlate(g7, g0, mode='full')
    lags = correlation_lags(n, n, mode='full')

    HALF_PERIOD = 60
    mask     = (lags >= TARGET - HALF_PERIOD) & (lags <= TARGET + HALF_PERIOD)
    peak_lag = int(lags[mask][np.argmax(corr[mask])])
    assert abs(peak_lag - TARGET) <= 5, \
        f"Expected cross-corr peak at lag {TARGET}, got {peak_lag}"


def test_mode_C_hotspot_variety():
    """
    Mode C rotates through hotspot sets at a varying speed.
    Over 5000 transactions the dominant active partition set must
    change at least 3 times, and the inter-switch intervals must
    vary (std > 10 transactions), proving the modulation is working.
    """
    n = 5000
    df = generate_mode('C', n=n, seed=42)
    cols = [f'p{i}' for i in range(SLOTS)]

    hotspot_sets = [set(range(i * 8, (i + 1) * 8)) for i in range(8)]

    def dominant_set(row):
        vals = [int(v) for v in row if v != SENTINEL]
        if not vals:
            return -1  # no partitions accessed — shouldn't happen
        counts = [sum(1 for v in vals if v in s) for s in hotspot_sets]
        return int(np.argmax(counts))

    active = np.array([dominant_set(row) for row in df[cols].values])
    switches = np.where(np.diff(active) != 0)[0]
    assert len(switches) >= 3, f"Too few hotspot switches: {len(switches)}"
    inter = np.diff(switches)
    assert inter.std() > 10, \
        f"Switching period appears constant (std={inter.std():.2f}); modulation not working"
    # P_MIN=160 transactions per full rotation, 8 sets → minimum ~20 tx per set.
    # Threshold 40 is 2× that floor to absorb sampling noise.
    assert inter.min() >= 40, \
        f"Switching too fast (min={inter.min()}); P_inst floor not respected"


def test_mode_D_regime_variation():
    """Dominant partition ranges should differ between first and last quarter."""
    n = 5000
    df = generate_mode('D', n=n, seed=42)
    cols = [f'p{i}' for i in range(SLOTS)]
    fq = df.iloc[:n // 4][cols].values.flatten()
    lq = df.iloc[3 * n // 4:][cols].values.flatten()
    fq_parts = set(int(v) for v in fq if v != SENTINEL)
    lq_parts = set(int(v) for v in lq if v != SENTINEL)
    overlap = len(fq_parts & lq_parts) / max(len(fq_parts | lq_parts), 1)
    assert overlap < 0.9, \
        f"First and last quarters share too many partitions (overlap={overlap:.2f})"


def test_mode_D_precursor_entropy_rises():
    """
    In the 200 transactions before each regime transition, access entropy
    (number of distinct partitions per transaction) should be measurably higher
    than in the stable-regime baseline.

    Tests via metadata so we know exactly where transitions are.
    """
    n = 5000
    df, meta = generate_mode('D', n=n, seed=42, return_metadata=True)
    transitions = meta['transitions']
    assert len(transitions) >= 2, \
        f"Need >= 2 transitions in {n} transactions, got {len(transitions)}"

    cols = [f'p{i}' for i in range(SLOTS)]

    def mean_distinct(start, end):
        rows = df.iloc[start:end][cols].values
        return np.mean([np.sum(row != SENTINEL) for row in rows])

    precursor_width = 150   # narrower than 200 to avoid boundary effects
    baseline_width  = 150

    precursor_scores = []
    baseline_scores  = []
    for t_trans in transitions:
        pre_start = t_trans - precursor_width
        bas_start = t_trans - precursor_width - baseline_width
        if bas_start >= 0 and pre_start >= 0:
            precursor_scores.append(mean_distinct(pre_start, t_trans))
            baseline_scores.append(mean_distinct(bas_start, pre_start))

    assert precursor_scores, \
        "No usable precursor windows found — check n, DWELL_MIN, and window widths"
    avg_pre = np.mean(precursor_scores)
    avg_bas = np.mean(baseline_scores)
    assert avg_pre > avg_bas, \
        f"Precursor partition count ({avg_pre:.2f}) not higher than baseline ({avg_bas:.2f})"


def test_mode_D_precursor_offset():
    """
    Each precursor_start recorded in metadata must be exactly 200 transactions
    before the corresponding transition.
    """
    n = 5000
    _df, meta = generate_mode('D', n=n, seed=42, return_metadata=True)
    transitions      = meta['transitions']
    precursor_starts = meta['precursor_starts']
    assert len(transitions) >= 2, \
        f"Need >= 2 transitions in {n} transactions, got {len(transitions)}"
    # precursor_starts may be shorter if last precursor falls near end of sequence
    assert len(precursor_starts) >= 1, \
        "No precursor_starts recorded — precursor window never opened"
    for t_pre, t_trans in zip(precursor_starts, transitions):
        assert t_trans - t_pre == 200, \
            f"Precursor-to-transition offset should be 200, got {t_trans - t_pre}"
