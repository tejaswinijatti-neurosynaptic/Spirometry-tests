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
FILE = r"d:\Users\Tejaswini\Desktop\neurosyn\live plotting\New method\realtime_all\forced1.log"

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

# Coefficients (device f039)
# 4-term basis coefficients (pull=inhale, push=exhale)
pull_coefficients = np.array([ 0.334646, -0.001808, -0.526989,  0.498103], dtype=float)  # inhale/pull
push_coefficients = np.array([ 1.21753e-01,  2.10000e-05,  7.26680e-02, -1.59643e-01], dtype=float)  # exhale/push

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
        s * np.sqrt(a),
        p,
        s * np.cbrt(a),
        s,
    ])

def pressure_to_flow_segments(pf: np.ndarray,
                              starts: np.ndarray,
                              ends: np.ndarray) -> np.ndarray:
    """
    Convert filtered pressure to flow using per-segment coefficients:
    - Use sign of mean(p) within segment to choose push vs pull model
    Returns flow array (same length as pf-len) with zeros outside FIR-valid region.
    Units: whatever your model yields (typically L/s if trained that way).
    """
    flow = np.zeros_like(pf, dtype=float)
    Phi = basis_from_filtered(pf)
    for s_idx, e_idx in zip(starts, ends):
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
    Returns dict with keys: FVC, FEV1, FEV1_Percentage, seg_index, Exlen, ErrNum,
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
        return dict(FVC=-1.0, FEV1=-1.0, FEV1_Percentage=-1.0, seg_index=-1, Exlen=0, ErrNum=9,
                    s_best=-1, e_best=-1, s_on=-1)

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
        FEV1_Percentage = float((FEV1 / FVC) * 100.0)
    else:
        FEV1_Percentage = -1.0
        if err == 0:
            err = 8

    return dict(FVC=FVC, FEV1=FEV1, FEV1_Percentage=FEV1_Percentage,
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

    """
    # choose segment with largest FVC
    i_best, s_best, e_best, FVC = max(exhale_candidates, key=lambda t: t[3])
    v_best = vol[s_best:e_best+1] - vol[s_best]
    Exlen = int(e_best - s_best + 1)

    idx_1s = int(round(1.0 / dt))
    if Exlen > idx_1s:
        FEV1 = float(v_best[idx_1s])
        err = 0
    else:
        FEV1 = -1.0
        err = 7

    if FVC > 0 and FEV1 > 0:
        FEV1_Percentage = float((FEV1 / FVC) * 100.0)
    else:
        FEV1_Percentage = -1.0
        if err == 0:  # set a different error only if not already set
            err = 8  # FEV1 faulty due to non-positive FVC/FEV1

    return dict(FVC=FVC, FEV1=FEV1, FEV1_Percentage=FEV1_Percentage,
                seg_index=i_best, Exlen=Exlen, ErrNum=err)
    """
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
    starts, ends = detect_segments(pf, DEADBAND, TAIL_IGNORE)

    # 4) Pressure -> Flow (per segment model selection)
    flow = pressure_to_flow_segments(pf, starts, ends)

    # 5) Integrate to Volume
    vol = integrate_flow_to_volume(flow, DT)

    # 6) Plot
    if PLOT:
        os.makedirs(OUTDIR, exist_ok=True)

        # Compute metrics once (no changes to logic)
        metrics = compute_exhale_metrics(flow, vol, starts, ends, DT)
        extra = compute_additional_metrics(flow, vol, starts, ends, DT, metrics)

        # Create 2x2 grid of equal-size subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle("Spirometry Analysis – Combined 2×2 Layout", fontsize=14)

        ax00 = axs[0, 0]  # Pressure vs Time
        ax01 = axs[0, 1]  # Flow–Volume Loop
        ax10 = axs[1, 0]  # Volume vs Time (Full Trace)
        ax11 = axs[1, 1]  # Exhaled Volume vs Time (Main Exhale)

        # --- 1️⃣ Pressure vs Time ---
        ax00.plot(t, pf, linewidth=1)
        ax00.set_ylabel("Pressure (Pa) [filtered]")
        ax00.set_title("Filtered Pressure vs Time")
        for s, e in zip(starts, ends):
            ax00.axvspan(t[s], t[e], alpha=0.15)
        ax00.grid(True, alpha=0.3)
        try:
            ax00.set_box_aspect(1)
        except Exception:
            pass

        # --- 2️⃣ Flow–Volume Loop (keep your original flow–volume plot) ---
        ax01.plot(vol, flow, linewidth=0.8)
        ax01.set_xlabel("Volume (L)")
        ax01.set_ylabel("Flow (L/s)")
        ax01.set_title("Flow–Volume Loop (Combined)")
        ax01.grid(True, alpha=0.3)
        try:
            ax01.set_box_aspect(1)
        except Exception:
            pass

        # --- 3️⃣ Volume vs Time (full trace) ---
        ax10.plot(t, vol, linewidth=1)
        ax10.set_ylabel("Volume (L)")
        ax10.set_xlabel("Time (s)")
        ax10.set_title("Volume vs Time (Full Trace)")
        for s, e in zip(starts, ends):
            ax10.axvspan(t[s], t[e], alpha=0.10)
        ax10.grid(True, alpha=0.3)
        try:
            ax10.set_box_aspect(1)
        except Exception:
            pass

        # --- 4️⃣ Exhaled Volume vs Time (main exhale) ---
        if metrics["s_best"] >= 0 and metrics["e_best"] >= 0:
            s_on = metrics["s_on"]
            e_best = metrics["e_best"]
            t_exhale = t[s_on:e_best+1] - t[s_on]
            vol_exhale = vol[s_on:e_best+1] - vol[s_on]

            ax11.plot(t_exhale, vol_exhale, linewidth=1.5)
            ax11.set_xlabel("Time (s)")
            ax11.set_ylabel("Volume (L)")
            ax11.set_title("Exhaled Volume vs Time (Main Exhale)")

            idx_1s = int(round(1.0 / DT))
            if len(t_exhale) > idx_1s:
                ax11.axvline(t_exhale[idx_1s], color='r', linestyle='--', alpha=0.6, label="1s (FEV1)")
            ax11.axvline(t_exhale[-1], color='g', linestyle='--', alpha=0.6, label="End of Exhale (FVC)")
            ax11.legend(loc="best")
            ax11.grid(True, alpha=0.3)
        else:
            ax11.text(0.5, 0.5, "No exhale segment detected", ha="center", va="center")
        try:
            ax11.set_box_aspect(1)
        except Exception:
            pass

        # --- Layout and save ---
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if SAVE_FIGS:
            fig.savefig(os.path.join(OUTDIR, "combined_2x2_all_plots.png"), dpi=150)

        plt.show()



    # 6.5) Metrics (FVC, FEV1, FEV1%)
    metrics = compute_exhale_metrics(flow, vol, starts, ends, DT)
    FVC = metrics["FVC"]; FEV1 = metrics["FEV1"]; FEV1_Percentage = metrics["FEV1_Percentage"]

    # Additional parameters
    extra = compute_additional_metrics(flow, vol, starts, ends, DT, metrics)
    # Print block
    print(f"FVC = {FVC:.6f}")
    if metrics["ErrNum"] == 9:
        print("FEV1 = -1  (no exhale segment)")
        print("FEV1_Percentage = -1  (no exhale segment)")
    else:
        if metrics["ErrNum"] == 7:
            print("FEV1 = -1  (segment shorter than 1s)")
        else:
            print(f"FEV1 = {FEV1:.6f}")
        if FEV1_Percentage > 0:
            print(f"FEV1_Percentage = {FEV1_Percentage:.3f} %")
        else:
            print("FEV1_Percentage = -1  (FEV1/FVC invalid)")

    # FEFs and PEF
    if extra:
        if extra['FEF25'] != -1: print(f"FEF25 = {extra['FEF25']:.6f}")
        else: print("FEF25 = -1")
        if extra['FEF50'] != -1: print(f"FEF50 = {extra['FEF50']:.6f}")
        else: print("FEF50 = -1")
        if extra['FEF75'] != -1: print(f"FEF75 = {extra['FEF75']:.6f}")
        else: print("FEF75 = -1")
        if extra['FEF25_75'] != -1: print(f"FEF25_75 = {extra['FEF25_75']:.6f}")
        else: print("FEF25_75 = -1")
        if extra['PEF'] != -1: print(f"PEF = {extra['PEF']:.6f}")
        else: print("PEF = -1")
        if extra['FET'] != -1: print(f"FET = {extra['FET']:.3f} s")
        else: print("FET = -1")
        if extra['PIF'] != -1: print(f"PIF = {extra['PIF']:.6f}")
        else: print("PIF = -1")
        if extra['TLC'] != -1: print(f"TLC = {extra['TLC']:.6f}")
        else: print("TLC = -1")
        if extra['RV'] != -1: print(f"RV = {extra['RV']:.6f}")
        else: print("RV = -1")
        if extra['VC'] != -1: print(f"VC = {extra['VC']:.6f}")
        else: print("VC = -1")
        if extra['FIVC'] != -1: print(f"FIVC = {extra['FIVC']:.6f}")
        else: print("FIVC = -1")

    # 7) Print simple summary
    print(f"Decoded samples (raw): {p_raw.size}")
    print(f"Filtered samples     : {pf.size}")
    print(f"Segments detected    : {len(starts)}")
    for i, (s, e) in enumerate(zip(starts, ends), 1):
        seg = slice(s, e+1)
        seg_type = "EXHALE/push" if np.mean(pf[seg]) > 0 else "INHALE/pull"
        print(f"  seg#{i:02d}: {seg_type:12s}  len={e-s+1} samples, duration={(e-s+1)*DT:.3f}s")

if __name__ == "__main__":
    main()
