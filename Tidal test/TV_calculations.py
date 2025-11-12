"""
TV_fromlog.py  (no argparse)

How to use:
1) Set FILE below to your .log path.
2) Optionally tweak DT/DEADBAND/TAIL_IGNORE and filter/baseline params.
3) Run:  python TV_fromlog.py
"""

from __future__ import annotations
from collections import deque
from typing import List, Iterable, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt

# USER SETTINGS 
#EDIT THIS TO THE LOG FILE
FILE = r"d:\Users\Tejaswini\Desktop\neurosyn\live plotting\New method\realtime_all\tidal1.log"

# Sampling period (seconds). 0.005 = 200 Hz
DT = 0.005

# Segmentation params
DEADBAND     = 100   # min segment length (samples)
TAIL_IGNORE  = 150   # ignore last N samples when segmenting

# Filtering + baseline params
TRI_WINDOW   = 20    # triangular half-window; FIR length = 2*TRI_WINDOW - 1 (>=2)
INIT_MEAN_N  = 150   # samples for initial mean removal
END_MEAN_N   = 150   # samples used to estimate linear drift toward tail

# Plot?
PLOT = True

#Coefficients
# 4-term basis coefficients (pull=inhale, push=exhale)
pull_coefficients = np.array([ 0.334646, -0.001808, -0.526989,  0.498103], dtype=float)  # inhale/pull
push_coefficients = np.array([ 1.21753e-01,  2.10000e-05,  7.26680e-02, -1.59643e-01], dtype=float)  # exhale/push
#for device = f039

#Parsing
def parse_log_file(file_path: str) -> np.ndarray:
    """
    Read one .log, extract pressure samples in Pa.
    - Skip only the first all-zero frame
    - Convert 24-bit counts to Pa
    """
    pressures_pa = []
    skipped_first_null = False
    with open(file_path, "r", errors="ignore") as f:
        for line in f:
            a = line.find('['); b = line.find(']')
            if a == -1 or b == -1 or b <= a + 1:
                continue
            parts = line[a+1:b].split(':')
            try:
                arr = [int(x, 16) for x in parts]
            except ValueError:
                continue

            # quick frame sanity
            if len(arr) < 120:
                continue
            if arr[0] != 83 or arr[1] != 72 or arr[119] != 70:
                continue

            # skip only the first all-zero payload
            if not skipped_first_null and all(v == 0 for v in arr):
                skipped_first_null = True
                continue

            OS_dig = 2**23
            FSS_inH2O = 120.0
            for i in range(7, 107, 5):
                if i + 4 >= len(arr):
                    break
                b2 = arr[i+2] & 0xFF
                b1 = arr[i+3] & 0xFF
                b0 = arr[i+4] & 0xFF
                decimal_count = (b2 << 16) | (b1 << 8) | b0
                pressure_inH2O = 1.25 * ((decimal_count - OS_dig) / (2**24)) * FSS_inH2O
                pressure_pa = pressure_inH2O * 249.089
                pressures_pa.append(pressure_pa)
    return np.asarray(pressures_pa, dtype=np.float64)

#Filtering & baseline/drift 
def triangular_weights(window: int) -> np.ndarray:
    if window < 2:
        raise ValueError("TRI_WINDOW must be >= 2")
    up = np.arange(1, window + 1, dtype=np.float64)
    down = np.arange(window - 1, 0, -1, dtype=np.float64)
    w = np.concatenate([up, down])
    w /= w.sum()
    return w

def streaming_fir(x: Iterable[float], weights: np.ndarray) -> np.ndarray:
    L = len(weights)
    buf = deque(maxlen=L)
    out: List[float] = []
    for v in x:
        buf.append(v)
        if len(buf) == L:
            out.append(float(np.dot(weights, np.fromiter(buf, dtype=np.float64))))
    return np.asarray(out, dtype=np.float64)

def remove_initial_mean(x: np.ndarray, n_init: int) -> tuple[np.ndarray, float]:
    if len(x) < n_init:
        m = float(np.mean(x)) if len(x) else 0.0
        return x - m, m
    m = float(np.mean(x[:n_init]))
    return x - m, m

def remove_linear_tail_drift(x: np.ndarray, n_tail: int) -> tuple[np.ndarray, float]:
    if len(x) < n_tail or n_tail <= 0:
        return x, 0.0
    tail_mean = float(np.mean(x[-n_tail:]))
    N = len(x)
    slope = tail_mean / N
    idx = np.arange(N, dtype=np.float64)
    return x - slope * idx, tail_mean

def preprocess_one(p_raw: np.ndarray) -> np.ndarray:
    """
    Filter + baseline (start mean) + linear drift correction.
    Returns filtered, zero-centered signal.
    """
    if p_raw.size == 0:
        return p_raw
    p0, _ = remove_initial_mean(p_raw, INIT_MEAN_N)
    pc, _ = remove_linear_tail_drift(p0, END_MEAN_N)
    w = triangular_weights(TRI_WINDOW)
    pf = streaming_fir(pc, w)
    return pf

#Segmentation logic
def detect_segments(x: np.ndarray, deadband: int = DEADBAND, tail_ignore: int = TAIL_IGNORE
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect inhale/exhale segments as continuous runs of same sign in x.
    - deadband: minimum segment length (samples)
    - tail_ignore: ignore last N samples
    Returns (starts, ends) indices within x.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    n_eff = max(0, n - int(tail_ignore))
    if n_eff == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    x_eff = x[:n_eff]
    sign = np.where(x_eff >= 0, 1, -1)

    # boundaries where sign flips
    change_idx = np.flatnonzero(sign[1:] != sign[:-1]) + 1
    starts = np.concatenate(([0], change_idx))
    ends   = np.concatenate((change_idx - 1, [n_eff - 1]))

    # keep only segments with enough length and amplitude
    keep_mask = []
    for s, e in zip(starts, ends):
        length_ok = (e - s + 1) >= deadband
        amp_ok    = np.mean(np.abs(x_eff[s:e+1])) > 10  # ~10 Pa threshold
        keep_mask.append(length_ok and amp_ok)
    keep_mask = np.asarray(keep_mask, dtype=bool)

    return starts[keep_mask], ends[keep_mask]

#Basis (4 terms)
def basis_from_filtered(pf: np.ndarray) -> np.ndarray:
    """
    Build 4-term basis ON ALREADY-FILTERED pressure:
      [ s*sqrt(|p|), p, s*|p|^(1/3), s ]
    """
    p = np.asarray(pf, dtype=float)
    a = np.abs(p)
    s = np.sign(p)
    return np.column_stack([
        s * np.sqrt(a),
        p,
        s * np.cbrt(a),
        s,
    ])

# Main compute
def compute_segment_volumes(p_corr: np.ndarray, dt: float,
                            deadband: int, tail_ignore: int
                           ) -> tuple[list[float], list[float], list[tuple[int,int,float,str]]]:
    """
    Returns (inhale_vols_L, exhale_vols_L, segments_debug)
      - inhale = pull (negative pressure segment) -> use pull_coefficients
      - exhale = push (positive pressure segment) -> use push_coefficients
    segments_debug: list of (s, e, vol_L, direction)
    """
    inhales, exhales, dbg = [], [], []
    if p_corr.size == 0:
        return inhales, exhales, dbg

    starts, ends = detect_segments(p_corr, deadband=deadband, tail_ignore=tail_ignore)
    if len(starts) == 0:
        return inhales, exhales, dbg

    Phi_all = basis_from_filtered(p_corr)

    for s, e in zip(starts, ends):
        seg = p_corr[s:e+1]
        seg_sign = 1 if np.median(seg) >= 0 else -1  # robust direction
        coeffs = push_coefficients if seg_sign > 0 else pull_coefficients
        # integrate volume over segment
        V_seg = dt * float(np.sum(Phi_all[s:e+1] @ coeffs))
        direction = "exhale/push" if seg_sign > 0 else "inhale/pull"

        # store absolute magnitudes for tidal volume averaging
        if seg_sign > 0:
            exhales.append(abs(V_seg))
        else:
            inhales.append(abs(V_seg))
        dbg.append((s, e, V_seg, direction))
    return inhales, exhales, dbg

def tidy_stats(values: list[float]) -> tuple[float,float,int]:
    if len(values) == 0:
        return float("nan"), float("nan"), 0
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0), int(len(arr))

def run_one(file_path: str):
    if not os.path.isfile(file_path):
        print(f"ERROR: file not found: {file_path}")
        return

    # 1) load & preprocess
    p_raw  = parse_log_file(file_path)
    p_corr = preprocess_one(p_raw)

    # 2) per-segment volumes with direction-based coeffs
    inh, exh, dbg = compute_segment_volumes(
        p_corr, dt=DT, deadband=DEADBAND, tail_ignore=TAIL_IGNORE
    )

    mean_inh, std_inh, n_inh = tidy_stats(inh)
    mean_exh, std_exh, n_exh = tidy_stats(exh)

    # 3) final tidal volume: average of inhale mean and exhale mean
    if np.isfinite(mean_inh) and np.isfinite(mean_exh):
        TV = 0.5 * (mean_inh + mean_exh)
        tv_note = "TV = average(mean_inhale, mean_exhale)"
    elif np.isfinite(mean_inh):
        TV = mean_inh
        tv_note = "Only inhale segments found; TV = mean_inhale"
    elif np.isfinite(mean_exh):
        TV = mean_exh
        tv_note = "Only exhale segments found; TV = mean_exhale"
    else:
        TV = float("nan")
        tv_note = "No valid segments found"

    # 4) report
    print("\n--- Tidal Volume Report ---")
    print(f"File: {file_path}")
    print(f"Samples (raw): {len(p_raw)}, Samples (filtered): {len(p_corr)}")
    print(f"Segments found: inhales={n_inh}, exhales={n_exh}")
    if np.isfinite(mean_inh):
        print(f"Inhale peaks:  mean={mean_inh:.3f} L, std={std_inh:.3f} L, n={n_inh}")
    else:
        print("Inhale peaks:  none")
    if np.isfinite(mean_exh):
        print(f"Exhale peaks:  mean={mean_exh:.3f} L, std={std_exh:.3f} L, n={n_exh}")
    else:
        print("Exhale peaks:  none")
    print(f"\nFinal TV: {TV:.3f} L  ({tv_note})")
    true_volume = 3
    error = (true_volume - abs(TV))*100/true_volume
    print(f"Percentage error compared to true volume of {true_volume} L: {error:.2f} %")

    # 5) plot with shaded inhale/exhale regions
    if PLOT and len(p_corr) > 0:
        t = np.arange(len(p_corr)) * DT
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(t, p_corr, lw=1, label="filtered pressure", color="steelblue")

        for s, e, vol, direction in dbg:
            tt = t[s:e+1]
            yy = p_corr[s:e+1]

            if "inhale" in direction:  # pull
                shade_color = "#ffb6c1"  # light pink
                line_color = "deeppink"
            else:  # exhale / push
                shade_color = "#d8b4fe"  # light purple (lavender)
                line_color = "purple"

            # shaded region under curve
            ax.fill_between(tt, yy, 0, color=shade_color, alpha=0.4)
            # segment line
            ax.plot(tt, yy, lw=2, color=line_color, label=f"{direction}: {vol:.3f} L")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pressure (Pa) [zero-centered]")
        ax.set_title(f"Tidal Volume from {os.path.basename(file_path)}")
        # remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ------------------------------ ENTRY POINT ------------------------------- #
if __name__ == "__main__":
    run_one(FILE)

