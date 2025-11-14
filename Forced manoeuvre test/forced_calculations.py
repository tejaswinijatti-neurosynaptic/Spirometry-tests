"""
TV_fromlog.py  (no argparse)

How to use:
1) Set FILE below to your .log path.
2) Optionally tweak DT/DEADBAND/TAIL_IGNORE and filter/baseline params.
3) Run:  python TV_fromlog.py

This version FINISHES the pipeline by:
- Detecting inhale/exhale segments on filtered pressure
- Converting pressure -> flow using your 4-term model (separate push/pull coeffs)
- Integrating flow to volume
- Plotting Flow–Volume loops (per-segment and combined)
- Plotting time-series (pressure, flow, volume) with segment overlays
"""

from __future__ import annotations
from collections import deque
from typing import List, Iterable, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt

# USER SETTINGS 
# EDIT THIS TO THE LOG FILE
FILE = r"d:\Users\Tejaswini\Desktop\neurosyn\live plotting\New method\realtime_all\f084_forced2.log"

# Sampling period (seconds). 0.005 = 200 Hz
DT = 0.005

# Segmentation params
DEADBAND     = 100   # min segment length (samples)
TAIL_IGNORE  = 150   # ignore last N samples when segmenting

# Filtering + baseline params
TRI_WINDOW   = 20    # triangular half-window; FIR length = 2*TRI_WINDOW - 1 (>=2)
INIT_MEAN_N  = 150   # samples for initial mean removal
END_MEAN_N   = 150   # samples used to estimate linear drift toward tail

# FEV1 onset detection (within exhale segment)
THRESH_ON_FRAC = 0.05   # start when flow exceeds 5% of PEF
THRESH_ON_ABS  = 0.10   # and at least 0.10 L/s absolute

# Plot?
PLOT = True
SAVE_FIGS = True
OUTDIR = "plots"

#Coefficients
# 4-term basis coefficients (pull=inhale, push=exhale)
pull_coefficients = np.array([ 0.176201, -0.000513, -0.111826,  0.045253, -0.045253], dtype=float)  # inhale/pull
push_coefficients = np.array([-0.498273,  0.004855,  1.690918, -0.943883, -0.943883], dtype=float)  # exhale/push

# Parsing
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

# Filtering & baseline/drift 
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
    Note: FIR shortens the array by (2*TRI_WINDOW - 2) samples.
    """
    if p_raw.size == 0:
        return p_raw
    p0, _ = remove_initial_mean(p_raw, INIT_MEAN_N)
    pc, _ = remove_linear_tail_drift(p0, END_MEAN_N)
    w = triangular_weights(TRI_WINDOW)
    pf = streaming_fir(pc, w)
    return pf

# Basis (4 terms) and model
def basis_from_filtered(pf: np.ndarray) -> np.ndarray:
    """
    Build 4-term basis ON ALREADY-FILTERED pressure:
      [ s*sqrt(|p|), p, s*|p|^(1/3), s ]
    """
    p = np.asarray(pf, dtype=float)
    a = np.abs(p)
    s = np.sign(p)
    return np.column_stack([
        s * np.sqrt(a),    # turbulent (Bernoulli) component
        p,                # laminar component
        s * a ** (1/3),    # mid-range correction
        s,
        np.ones_like(p),
    ])

def pressure_to_flow_segments(pf: np.ndarray,
                              starts: list[int],
                              ends: list[int],
                              min_peak_pa: float = 200.0) -> np.ndarray:
    """
    Convert filtered pressure to flow using per-segment coefficients.
    - Only compute flow for segments whose peak absolute pressure >= min_peak_pa.
      Segments below deadband are treated as artifacts and left as zeros.
    - Uses segment-mean sign to choose push vs pull model (same as realtime).
    Returns flow array (same length as pf).
    """
    flow = np.zeros_like(pf, dtype=float)
    Phi = basis_from_filtered(pf)

    for s_idx, e_idx in zip(starts, ends):
        # if segment is too small/inactive (peak below deadband) -> skip it
        seg_abs_peak = float(np.nanmax(np.abs(pf[s_idx:e_idx+1]))) if e_idx >= s_idx else 0.0
        if seg_abs_peak < min_peak_pa:
            # keep zeros (ignore artifact)
            # you can optionally print a debug message here
            # print(f"[skip] segment {s_idx}-{e_idx} peak {seg_abs_peak:.1f} < deadband {min_peak_pa}")
            continue

        seg_mean = float(np.mean(pf[s_idx:e_idx+1]))
        coeffs = push_coefficients if seg_mean > 0 else pull_coefficients
        flow[s_idx:e_idx+1] = Phi[s_idx:e_idx+1] @ coeffs

    return flow


def integrate_flow_to_volume(flow: np.ndarray, dt: float) -> np.ndarray:
    """
    Simple rectangular numerical integration to get volume [L] if flow is in [L/s].
    """
    vol = np.cumsum(flow) * dt
    return vol

# Segmentation logic
# Robust segment detection for pf (filtered pressure) -> returns lists of start,end sample indices
def detect_segments_from_pf(
    pf: np.ndarray,
    dt: float,
    start_thr: float = 4,      # LOWER threshold to *detect* start (Pa) — you asked for lower detection
    end_thr: float | None = None,
    start_hold: int = 3,
    end_hold: int = 10,
    min_seg_sec: float = 0.02,
    merge_gap_sec: float = 0.12,
) -> tuple[list[int], list[int]]:
    """
    Robust segment detection with separate start/end thresholds.
    - start_thr: threshold used to *start* a segment (lower so we capture lead-in)
    - end_thr: threshold used to *end* a segment (defaults to 0.6 * start_thr)
    - start_hold: consecutive samples above start_thr to confirm start
    - end_hold: consecutive samples below end_thr to confirm end
    - min_seg_sec: discard segments shorter than this (seconds)
    - merge_gap_sec: merge segments separated by a short gap (seconds)
    """
    n = pf.size
    if n == 0:
        return [], []

    if end_thr is None:
        end_thr = start_thr * 0.6

    abs_pf = np.abs(pf)
    is_start_above = abs_pf >= start_thr
    is_end_below = abs_pf <= end_thr

    starts = []
    ends = []

    i = 0
    while i < n:
        if is_start_above[i]:
            # check for start_hold consecutive trues
            run = 1
            j = i + 1
            while j < n and is_start_above[j] and run < start_hold:
                run += 1
                j += 1

            if run >= start_hold:
                start_idx = i
                # find end using end_thr + end_hold
                k = j
                quiet = 0
                while k < n:
                    if is_end_below[k]:
                        quiet += 1
                    else:
                        quiet = 0
                    k += 1
                    if quiet >= end_hold:
                        end_idx = k - end_hold - 1
                        break
                else:
                    end_idx = n - 1

                starts.append(int(start_idx))
                ends.append(int(end_idx))
                i = end_idx + 1
                continue
            else:
                i = j
                continue
        i += 1

    # Merge close segments if gap <= merge_gap_sec
    merged_starts = []
    merged_ends = []
    if starts:
        cur_s, cur_e = starts[0], ends[0]
        for s, e in zip(starts[1:], ends[1:]):
            gap = (s - cur_e - 1) * dt
            if gap <= merge_gap_sec:
                cur_e = e
            else:
                merged_starts.append(cur_s)
                merged_ends.append(cur_e)
                cur_s, cur_e = s, e
        merged_starts.append(cur_s)
        merged_ends.append(cur_e)

    # Enforce minimum length
    min_len_samples = max(1, int(round(min_seg_sec / dt)))
    final_starts = []
    final_ends = []
    for s, e in zip(merged_starts, merged_ends):
        if (e - s + 1) >= min_len_samples:
            final_starts.append(int(s))
            final_ends.append(int(e))

    return final_starts, final_ends

# Plotting helpers
def plot_time_series(t: np.ndarray, pf: np.ndarray, flow: np.ndarray, vol: np.ndarray,
                     starts: np.ndarray, ends: np.ndarray, title_prefix: str = ""):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Pressure
    axes[0].plot(t, pf, linewidth=1)
    axes[0].set_ylabel("Pressure (Pa) [filtered]")
    axes[0].grid(True, alpha=0.3)

    # Volume
    axes[1].plot(t, -vol, linewidth=1)
    axes[1].set_ylabel("Volume")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    # segments overlay
    for s, e in zip(starts, ends):
        axes[0].axvspan(t[s], t[e], alpha=0.15)
        axes[1].axvspan(t[s], t[e], alpha=0.10)

    fig.suptitle(f"Pressure/Flow/Volume with segments")
    fig.tight_layout()
    return fig


def plot_flow_volume(flow: np.ndarray, vol: np.ndarray,
                      starts: np.ndarray, ends: np.ndarray,
                      title_prefix: str = ""):

    # Combined loop (entire series, unnormalized)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    ax2.plot(vol, flow, linewidth=0.8)
    ax2.set_xlabel("Volume")
    ax2.set_ylabel("Flow")
    ax2.set_title(f"Flow–Volume Loop (combined)")
    ax2.grid(True, alpha=0.3)

    return fig2

# FVC/FEV1 metrics
def compute_exhale_metrics(flow: np.ndarray, vol: np.ndarray, starts: np.ndarray, ends: np.ndarray, dt: float):
    """Compute FVC, FEV1, FEV1% from the *primary forced exhale* segment.
    Selection: among segments with positive mean flow, choose the one with
    the largest exhaled volume (delta volume).
    Start time (t=0) for FEV1 is detected when flow rises above
    max(THRESH_ON_FRAC * PEF, THRESH_ON_ABS) to avoid quiescent lead-in.
    Returns dict with keys: FVC, FEV1, FEV1/FVC, seg_index, Exlen, ErrNum,
    and indices s_best, e_best, s_on.
    ErrNum: 0 ok, 7 = FEV1 unavailable (segment < 1s after onset), 9 = no exhale segment.
    """
    exhale_candidates = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        seg_flow = flow[s:e+1]
        if np.nanmean(seg_flow) <= 0:
            continue
        v_seg = vol[s:e+1] - vol[s]
        fvc_i = float(np.nanmax(v_seg) - np.nanmin(v_seg))
        exhale_candidates.append((i, s, e, fvc_i))

    if not exhale_candidates:
        return dict(valid=False,
                message="No valid exhale detected. Please blow forcefully into the device.")

    # choose segment with largest FVC
    i_best, s_best, e_best, _ = max(exhale_candidates, key=lambda t: t[3])

    # Onset detection based on fraction of PEF
    seg_flow = flow[s_best:e_best+1]
    pef = float(np.nanmax(seg_flow)) if seg_flow.size else 0.0
    thresh = max(THRESH_ON_FRAC * pef, THRESH_ON_ABS)
    above = np.flatnonzero(seg_flow >= thresh)
    if above.size:
        s_on = s_best + int(above[0])
    else:
        s_on = s_best  # fallback

    v_on = vol[s_on:e_best+1] - vol[s_on]
    Exlen = int(e_best - s_on + 1)
    FVC = float(v_on[-1]) if v_on.size else -1.0

    idx_1s = int(round(1.0 / dt))
    if Exlen > idx_1s and v_on.size > idx_1s:
        FEV1 = float(v_on[idx_1s])
        err = 0
    else:
        FEV1 = -1.0
        err = 7

    if FVC > 0 and FEV1 > 0:
        FEV1_FVC = float((FEV1 / FVC) * 100.0)
    else:
        FEV1_FVC = -1.0
        if err == 0:
            err = 8

    return dict(FVC=FVC, FEV1=FEV1, FEV1_FVC=FEV1_FVC,
                seg_index=i_best, Exlen=Exlen, ErrNum=err,
                s_best=s_best, e_best=e_best, s_on=s_on)


def compute_additional_metrics(flow: np.ndarray, vol: np.ndarray, starts: np.ndarray, ends: np.ndarray, dt: float,
                                base_metrics: dict):
    out = dict(FEF25=-1.0, FEF50=-1.0, FEF75=-1.0, PEF=-1.0, FEF25_75=-1.0,
               FET=-1.0, PIF=-1.0, TLC=-1.0, RV=-1.0, VC=-1.0, FIVC=-1.0)

    if base_metrics.get('seg_index', -1) < 0:
        return out

    s_best = int(base_metrics['s_best']); e_best = int(base_metrics['e_best']); s_on = int(base_metrics['s_on'])
    if s_on < 0:
        return out

    # Exhale arrays from onset (volume zeroed at onset for FVC/FEF calc)
    ex_flow = flow[s_on:e_best+1]
    ex_vol  = vol[s_on:e_best+1] - vol[s_on]
    if ex_flow.size < 2:
        return out

    FVC = float(ex_vol[-1])

    # PEF
    out['PEF'] = float(np.nanmax(ex_flow))

    # Helper: first index where ex_vol crosses target volume
    def first_cross_idx(target):
        v = ex_vol
        for i in range(0, len(v)-1):
            if v[i] <= target and v[i+1] > target:
                return i
        return -1

    # FEF25/50/75
    v25, v50, v75 = 0.25 * FVC, 0.50 * FVC, 0.75 * FVC
    i25, i50, i75 = first_cross_idx(v25), first_cross_idx(v50), first_cross_idx(v75)
    if i25 >= 0: out['FEF25'] = float(ex_flow[i25])
    if i50 >= 0: out['FEF50'] = float(ex_flow[i50])
    if i75 >= 0: out['FEF75'] = float(ex_flow[i75])

    # FEF25-75 (mean)
    if i25 >= 0 and i75 > i25:
        slab = ex_flow[i25:i75+1]
        out['FEF25_75'] = float(np.nanmean(slab)) if slab.size else -1.0

    # FET
    out['FET'] = float((e_best - s_on + 1) * dt)

    # ---- Find the next inhale segment (if any) ----
    next_inhale_idx = -1
    for j, (ss, ee) in enumerate(zip(starts, ends)):
        if ss > e_best and np.nanmean(flow[ss:ee+1]) < 0:
            next_inhale_idx = j
            s_in, e_in = ss, ee
            break

    # PIF (if inhale present)
    if next_inhale_idx >= 0:
        in_flow = flow[s_in:e_in+1]
        out['PIF'] = float(-np.nanmin(in_flow))  # report magnitude as positive

    # ==== NEW: TLC/RV/VC from loop extremes ====
    # Window for min/max volume: from exhalation onset to end of next inhale (if present)
    win_end = e_in if next_inhale_idx >= 0 else e_best
    vseg = vol[s_on:win_end+1]
    if vseg.size:
        v_min = float(np.nanmin(vseg))  # far-left end (RV, possibly negative)
        v_max = float(np.nanmax(vseg))  # far-right end (TLC)

        out['RV']  = abs(v_min)         # per your rule: abs(RV)
        out['TLC'] = v_max+(v_max+abs(v_min))              # far-right end
        out['VC']  = out['TLC'] - out['RV']  # VC = TLC - abs(RV)

    #print(f"printing v_min and v_max{v_min, v_max, (v_max+abs(v_min)) }")
    # FIVC (keep as before or recompute if you prefer a different definition)
    if next_inhale_idx >= 0:
        # magnitude of inhaled volume in that next inhale segment
        in_flow = flow[s_in:e_in+1]
        in_vol = -(np.cumsum(in_flow) * dt)
        out['FIVC'] = float(in_vol[-1]) if in_vol.size else -1.0

    return out

PRESS_THR_PA = 3   # pressure threshold for segmentation (Pa)
START_HOLD   = 10      # samples above threshold to start segment    
END_HOLD     = 10     # samples below threshold to end segment
# Main
# Main
def main():
    if not os.path.exists(FILE):
        raise FileNotFoundError(f"Log file not found: {FILE}")

    # 1) Parse
    p_raw = parse_log_file(FILE)
    if p_raw.size == 0:
        raise RuntimeError("No pressure samples decoded from the log.")

    # 2) Filter + baseline
    pf = preprocess_one(p_raw)

    # Build time vector matching filtered signal length
    t = np.arange(pf.size) * DT

    # 3) Detect segments on filtered pressure
    # pf is your filtered pressure array, DT is sample interval
    # detection with LOWER start threshold (capture lead-in)
    START_DETECT_THR = 3          # lower detection threshold in Pa (tune as needed)
    END_DETECT_THR   = START_DETECT_THR * 0.6

    # deadband for exhale/artifact filtering (Pa)
    SEGMENT_PEAK_DEADBAND_PA = 150.0   # set to 200 or 300 as you prefer

    starts, ends = detect_segments_from_pf(
        pf,
        dt=DT,
        start_thr=START_DETECT_THR,
        end_thr=END_DETECT_THR,
        start_hold=START_HOLD,
        end_hold=END_HOLD,
        min_seg_sec=0.02,
        merge_gap_sec=0.12,
    )
    print(f"Detected {len(starts)} raw segments: {list(zip(starts, ends))}")

    # now per-segment pressure->flow but only for segments that exceed deadband
    flow = pressure_to_flow_segments(pf, starts, ends, min_peak_pa=SEGMENT_PEAK_DEADBAND_PA)

    # optional: compute baseline from quiet region then subtract if you do that later
    flow_baseline = float(np.median(flow[:2000])) if flow.size >= 2000 else float(np.median(flow))
    flow_baseline_removed = flow - flow_baseline
    print(f"Flow baseline removed: {flow_baseline:.6f} L/s")


    # 5) Integrate to Volume
    vol = integrate_flow_to_volume(flow, DT)

    # 6) Plot
    if PLOT:
        os.makedirs(OUTDIR, exist_ok=True)

        # Compute metrics once
        metrics = compute_exhale_metrics(flow, vol, starts, ends, DT)
        extra = compute_additional_metrics(flow, vol, starts, ends, DT, metrics)

        # Create 1x3 grid (2 plots + 1 for metrics)
        fig, axs = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True) # Changed 1x2 to 1x3, increased size
        fig.suptitle("Spirometry Analysis (Plots and Metrics)", fontsize=14, y=1.05)

        # Define the three axes
        ax_fv      = axs[0]  # LEFT plot: Flow–Volume Loop
        ax_evs     = axs[1]  # MIDDLE plot: Exhaled Volume vs Time
        ax_metrics = axs[2]  # RIGHT slot: Metrics display


        # --- Flow–Volume Loop (LEFT: ax_fv) ---
        # Build named segment records (human numbering: Segment 1, Segment 2, ...)
        segments = []   # clean list start
        FLOW_MIN_THRESHOLD = 0.2   # L/s  (your requirement)
        for idx, (s, e) in enumerate(zip(starts, ends), start=1):
            seg_flow = flow[s:e+1]
            seg_vol  = vol[s:e+1]
            if seg_flow.size == 0:
                continue

            seg_peak_flow = np.nanmax(np.abs(seg_flow))
            if seg_peak_flow < FLOW_MIN_THRESHOLD:
                # skip small baby-breath artifacts
                continue
            mean_flow = float(np.nanmean(seg_flow)) if seg_flow.size else 0.0
            seg_name = f"Segment {idx}"
            segments.append({
                "name": seg_name,
                "start": int(s),
                "end": int(e),
                "mean_flow": mean_flow,
                "flow": seg_flow,
                "vol": seg_vol,
            })

        # Debug print of detected segments
        print("Segments summary:")
        for seg in segments:
            print(f"  {seg['name']}: {seg['start']}..{seg['end']}, mean_flow={seg['mean_flow']:.4f}")

        # Choose which human-numbered segments to plot (2nd and 3rd)
        wanted_one_based = [2, 3]
        wanted_zero_based = [n - 1 for n in wanted_one_based]

        # ---------------- Assemble selected segments but zero FLOW at each segment start (x unchanged) ------------
        sel_v = []
        sel_f = []

        # tuning: how much pre-start to use to estimate baseline flow (ms -> samples)
        PRE_START_BASELINE_MS = 50
        pre_samples = max(1, int(round((PRE_START_BASELINE_MS / 1000.0) / DT)))

        for i in wanted_zero_based:
            if 0 <= i < len(segments):
                seg = segments[i]
                s = seg["start"]
                e = seg["end"]

                seg_v = seg["vol"].astype(float)   # keep absolute volume (no x-zeroing)
                seg_f = seg["flow"].astype(float)  # original flow

                # estimate baseline from a short window BEFORE the segment start if available,
                # otherwise use the first few samples of the segment
                if s - pre_samples >= 0:
                    baseline_win = flow[s - pre_samples : s]
                    if baseline_win.size:
                        baseline_flow = float(np.nanmedian(baseline_win))
                    else:
                        baseline_flow = float(np.nanmedian(seg_f[:min(3, seg_f.size)]))
                else:
                    baseline_flow = float(np.nanmedian(seg_f[:min(3, seg_f.size)]))

                # subtract that baseline so flow near the start is referenced to quiet
                seg_f_plot = seg_f - baseline_flow

                # enforce exact zero at the first plotted sample (guarantees segment starts at 0 L/s)
                # (this only affects the plotted copy; underlying seg_f stays unchanged)
                if seg_f_plot.size:
                    seg_f_plot = seg_f_plot - float(seg_f_plot[0])

                # append to plotting arrays (keep NaN separators so segments are not joined)
                sel_v.extend(seg_v.tolist())
                sel_f.extend(seg_f_plot.tolist())
                sel_v.append(np.nan); sel_f.append(np.nan)
            else:
                print(f"[warning] requested segment {i+1} not available (found {len(segments)})")

        # ---------------- Plot: Flow–Volume Loop (LEFT: ax_fv) ------------
        ax_fv.cla()
        ax_fv.set_xlabel("Volume (L)", fontsize=10)
        ax_fv.set_ylabel("Flow (L/s)", fontsize=10)
        ax_fv.set_title("Flow–Volume Loop — selected segments", fontsize=11)
        ax_fv.grid(True, alpha=0.3)

        if len(sel_v):
            ax_fv.plot(sel_v, sel_f, linewidth=1.2, label="Segments")
            # annotate segment starts using original absolute coords
            for j, n in enumerate(wanted_zero_based):
                if 0 <= n < len(segments):
                    seg = segments[n]
                    # show marker at real absolute volume and the zeroed flow (which is 0)
                    x0 = float(seg['vol'][0])
                    y0 = 0.0
                    ax_fv.scatter(x0, y0, s=30, zorder=5)
                    ax_fv.annotate(seg['name'], xy=(x0, y0),
                                  xytext=(6 + 16*(j%2), 6 + 8*(j%2)), textcoords='offset points', fontsize=8)
            ax_fv.legend(loc='upper right', fontsize=8)
        else:
            ax_fv.text(0.5, 0.5, "No selected segments to plot", ha='center', va='center', transform=ax_fv.transAxes)

        # optional: keep aspect so loop looks sane
        try:
            ax_fv.set_aspect(0.5, adjustable='box')
        except Exception:
            pass

        # --- Exhaled Volume vs Time (MIDDLE: ax_evs) ---
        if metrics["s_best"] >= 0 and metrics["e_best"] >= 0:
            s_on = metrics["s_on"]
            e_best = metrics["e_best"]
            t_exhale = t[s_on:e_best+1] - t[s_on]
            vol_exhale = vol[s_on:e_best+1] - vol[s_on]
            # Force non-decreasing (prevents tiny negative tails)
            vol_exhale = np.maximum.accumulate(vol_exhale)

            ax_evs.plot(t_exhale, (vol_exhale), linewidth=1.5)
            ax_evs.set_xlabel("Time (s)", fontsize=10)
            ax_evs.set_ylabel("Volume (L)", fontsize=10)
            ax_evs.set_title("Exhaled Volume vs Time (Main Exhale)", fontsize=11)

            idx_1s = int(round(1.0 / DT))
            if len(t_exhale) > idx_1s:
                ax_evs.axvline(t_exhale[idx_1s], color='r', linestyle='--', alpha=0.6, label="1s (FEV1)")
            ax_evs.axvline(t_exhale[-1], color='g', linestyle='--', alpha=0.6, label="End of Exhale (FVC)")
            ax_evs.legend(loc="best", fontsize=9)
            ax_evs.grid(True, alpha=0.3)
        else:
            ax_evs.text(0.5, 0.5, "No exhale segment detected", ha="center", va="center", fontsize=10)

        # --- Metrics Display (RIGHT: ax_metrics) ---
        ax_metrics.axis('off') # Hide axes for a clean text box
        
        # Prepare text content
        # FVC/FEV1 Block
        text_lines = [f"File: {os.path.basename(FILE)}"]
        text_lines.append("\n-- Main Metrics --")
        text_lines.append(f"FVC = {metrics['FVC']:.3f} L")
        if metrics["ErrNum"] == 9:
            text_lines.append("FEV1 = -1 (No Exhale)")
            text_lines.append("FEV1% = -1")
        else:
            if metrics["ErrNum"] == 7:
                text_lines.append("FEV1 = -1 (Seg < 1s)")
            else:
                text_lines.append(f"FEV1 = {metrics['FEV1']:.3f} L")
            if metrics["FEV1_FVC"] > 0:
                text_lines.append(f"FEV1% = {metrics['FEV1_FVC']:.1f} %")
            else:
                text_lines.append("FEV1% = -1 (Invalid)")
        
        # FEFs/PEF Block
        text_lines.append("\n-- Flow Rates --")
        text_lines.append(f"PEF = {extra.get('PEF', -1):.3f} L/s")
        text_lines.append(f"PIF = {extra.get('PIF', -1):.3f} L/s")
        text_lines.append(f"FEF25 = {extra.get('FEF25', -1):.3f} L/s")
        text_lines.append(f"FEF50 = {extra.get('FEF50', -1):.3f} L/s")
        text_lines.append(f"FEF75 = {extra.get('FEF75', -1):.3f} L/s")
        text_lines.append(f"FEF25-75 = {extra.get('FEF25_75', -1):.3f} L/s")
        text_lines.append(f"FET = {extra.get('FET', -1):.2f} s")
        
        # Volume Capacity Block
        text_lines.append("\n-- Capacity --")
        text_lines.append(f"VC = {extra.get('VC', -1):.3f} L")
        text_lines.append(f"TLC = {extra.get('TLC', -1):.3f} L")
        text_lines.append(f"RV = {extra.get('RV', -1):.3f} L")
        text_lines.append(f"FIVC = {extra.get('FIVC', -1):.3f} L")
        
        # Display the text
        text_content = "\n".join(text_lines).replace("-1.000", "-1").replace("-1.0", "-1")
        ax_metrics.text(0.05, 0.95, text_content, 
                        transform=ax_metrics.transAxes, 
                        fontsize=10, 
                        verticalalignment='top',
                        family='monospace')


        # Final layout tweaks
        try:
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        except Exception:
            plt.tight_layout()

        # Save figure with tight bounding box so PNG doesn't crop awkwardly
        if SAVE_FIGS:
            figpath = os.path.join(OUTDIR, "combined_plots_with_metrics.png")
            fig.savefig(figpath, dpi=150, bbox_inches='tight')

        plt.show() # Keep the first show call
        
        # --- End of Plotting Block ---


    # 6.5) Metrics (FVC, FEV1, FEV1%)
    # KEEPING THIS FOR CONSOLE OUTPUT/DEBUG
    metrics = compute_exhale_metrics(flow, vol, starts, ends, DT)
    FVC = metrics["FVC"]; FEV1 = metrics["FEV1"]; FEV1_FVC = metrics["FEV1_FVC"]

    # Additional parameters
    extra = compute_additional_metrics(flow, vol, starts, ends, DT, metrics)
    # Print block
    print(f"FVC = {FVC:.6f}")
    if metrics["ErrNum"] == 9:
        print("FEV1 = -1  (no exhale segment)")
        print("FEV1/FVC = -1  (no exhale segment)")
    else:
        if metrics["ErrNum"] == 7:
            print("FEV1 = -1  (segment shorter than 1s)")
        else:
            print(f"FEV1 = {FEV1:.6f}")
        if FEV1_FVC > 0:
            print(f"FEV1/FVC = {FEV1_FVC:.3f} %")
        else:
            print("FEV1/FVC = -1  (FEV1/FVC invalid)")

    # FEFs and PEF
    if extra:
        if extra.get('FEF25') != -1: print(f"FEF25 = {extra['FEF25']:.6f}")
        else: print("FEF25 = -1")
        if extra.get('FEF50') != -1: print(f"FEF50 = {extra['FEF50']:.6f}")
        else: print("FEF50 = -1")
        if extra.get('FEF75') != -1: print(f"FEF75 = {extra['FEF75']:.6f}")
        else: print("FEF75 = -1")
        if extra.get('FEF25_75') != -1: print(f"FEF25_75 = {extra['FEF25_75']:.6f}")
        else: print("FEF25_75 = -1")
        if extra.get('PEF') != -1: print(f"PEF = {extra['PEF']:.6f}")
        else: print("PEF = -1")
        if extra.get('FET') != -1: print(f"FET = {extra['FET']:.3f} s")
        else: print("FET = -1")
        if extra.get('PIF') != -1: print(f"PIF = {extra['PIF']:.6f}")
        else: print("PIF = -1")
        if extra.get('TLC') != -1: print(f"TLC = {extra['TLC']:.6f}")
        else: print("TLC = -1")
        if extra.get('RV') != -1: print(f"RV = {extra['RV']:.6f}")
        else: print("RV = -1")
        if extra.get('VC') != -1: print(f"VC = {extra['VC']:.6f}")
        else: print("VC = -1")
        if extra.get('FIVC') != -1: print(f"FIVC = {extra['FIVC']:.6f}")
        else: print("FIVC = -1")

    # 7) Print simple summary
    #print(f"Decoded samples (raw): {p_raw.size}")
    #print(f"Filtered samples     : {pf.size}")
    print(f"Segments detected    : {len(starts)}")
    for i, (s, e) in enumerate(zip(starts, ends), 1):
        seg = slice(s, e+1)
        seg_type = "EXHALE/push" if np.mean(pf[seg]) > 0 else "INHALE/pull"
        print(f"  seg#{i:02d}: {seg_type:12s}  len={e-s+1} samples, duration={(e-s+1)*DT:.3f}s")

if __name__ == "__main__":
    main()
