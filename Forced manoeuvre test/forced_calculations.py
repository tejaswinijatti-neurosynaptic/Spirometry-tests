"""
TV_fromlog.py

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
from GLI_2012_referencevalues_inputchangeinside import (equations, fev1_males, fev1_females, fvc_females, fvc_males, fev1fvc_males, fev1fvc_females,
                                                        fef2575_females, fef2575_males, fef75_females, fef75_males)
import json
import sys

# USER SETTINGS 
# EDIT THIS TO THE LOG FILE
FILES = [r"d:\Users\Tejaswini\Desktop\neurosyn\live plotting\New method\realtime_all\f078.log",
r"d:\Users\Tejaswini\Desktop\neurosyn\live plotting\New method\realtime_all\f066_1.log",
r"d:\Users\Tejaswini\Desktop\neurosyn\live plotting\New method\realtime_all\un02_1.log"
]


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
def load_coeffs(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Look in the 'models' subdirectory
    file_path = os.path.join(script_dir, "models", filename)
    
    if not os.path.exists(file_path):
        print(f"\nCRITICAL ERROR: Could not find '{filename}' in 'models' folder.")
        print(f"Path searched: {file_path}")
        sys.exit(1)
        
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {filename}")
        return np.array(data["coeffs"], dtype=float)
    except Exception as e:
        print(f"Error reading JSON {filename}: {e}")
        sys.exit(1)

print("--- Loading Coefficients ---")
pull_coefficients = load_coeffs("coeffs_pull.json")
push_coefficients = load_coeffs("coeffs_push.json")

# Parsing
def parse_log_file(file_path: str) -> np.ndarray:
    pressures_pa = []
    frame_count = 0
    
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
            
            # frame sanity check
            if len(arr) < 120:
                continue
            if arr[0] != 83 or arr[1] != 72 or arr[119] != 70:
                continue
            
            frame_count += 1
            
            # In first 5 frames, skip "Frame A" type (has zeros at bytes 7-13)
            if frame_count <= 5:
                # Frame A signature: bytes 7-13 are all zeros
                if all(arr[i] == 0 for i in range(7, 14)):
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

# Basisand model
def basis_from_filtered(pf: np.ndarray) -> np.ndarray:
    """
    9-term basis matching the training script:
    [p, s*sqrt(|p|), |p|, p*s*sqrt(|p|), p^2, 1, s*log(|p|+1), p^3, abs_deriv]
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
        flow[s_idx:e_idx+1] = Phi[s_idx:e_idx+1] @ coeffs[:5]# segment-level features
        seg = pf[s_idx:e_idx+1]
        
        # build contribution from full model
        flow[s_idx:e_idx+1] = Phi[s_idx:e_idx+1] @ coeffs

    return flow

def integrate_flow_to_volume(flow: np.ndarray, dt: float) -> np.ndarray:
    """
    Simple rectangular numerical integration to get volume [L] if flow is in [L/s].
    """
    vol = np.cumsum(flow) * dt
    return vol

deadband = 100
threshold = 5
start_threshold = 1
release_threshold = 1

# Segmentation logic
# Robust segment detection for pf (filtered pressure) -> returns lists of start,end sample indices
def detect_segments(
    x,
    deadband=100,
    tail_ignore=100,
    start_threshold=1,
    release_threshold=1,
    mean_amp_threshold=None,
):
    x = np.asarray(x, dtype=float)
    n = len(x)
    n_eff = max(0, n - tail_ignore)
    if n_eff <= 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)

    x = x[:n_eff]

    amp_threshold = mean_amp_threshold if mean_amp_threshold is not None else threshold

    kept_starts, kept_ends = [], []

    i = 0
    while i < n_eff:
        while i < n_eff and np.abs(x[i]) < start_threshold:
            i += 1
        if i >= n_eff:
            break

        s = i

        while i < n_eff and np.abs(x[i]) >= release_threshold:
            i += 1
        e = i - 1

        if e >= s:
            length_ok = (e - s + 1) >= deadband
            amp_ok = np.mean(np.abs(x[s:e+1])) > amp_threshold
            if length_ok and amp_ok:
                kept_starts.append(s)
                kept_ends.append(e)

    return np.asarray(kept_starts, dtype=int), np.asarray(kept_ends, dtype=int)

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
    ax2.set_xticks(np.arange(-6, 6, 1))
    ax2.set_yticks(np.arange(-10, 14, 2))
    ax2.set_xlim([-6, 6])
    ax2.set_ylim([-10, 14])
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
        return dict(
            FVC=np.nan,
            FEV1=np.nan,
            FEV1_FVC=np.nan,
            seg_index=-1,
            Exlen=0,
            ErrNum=9,
            s_best=-1,
            e_best=-1,
            s_on=-1,
            valid=False,
            message="No valid exhale detected. Please blow forcefully into the device."
        )

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
    FVC = float(v_on[-1]) if v_on.size else np.nan

    idx_1s = int(round(1.0 / dt))
    if Exlen > idx_1s and v_on.size > idx_1s:
        FEV1 = float(v_on[idx_1s])
        err = 0
    else:
        FEV1 = np.nan
        err = 7

    if FVC > 0 and FEV1 > 0:
        FEV1_FVC = float((FEV1 / FVC) * 100.0)
    else:
        FEV1_FVC = np.nan
        if err == 0:
            err = 8

    return dict(FVC=FVC, FEV1=FEV1, FEV1_FVC=FEV1_FVC,
                seg_index=i_best, Exlen=Exlen, ErrNum=err,
                s_best=s_best, e_best=e_best, s_on=s_on)


def compute_additional_metrics(flow: np.ndarray, vol: np.ndarray, starts: np.ndarray, ends: np.ndarray, dt: float,
                                base_metrics: dict):
    out = dict(FEF25=np.nan , FEF50=np.nan, FEF75=np.nan, PEF=np.nan, FEF25_75=np.nan,
               FET=np.nan, PIF=np.nan, TLC=np.nan, RV=np.nan, VC=np.nan, FIVC=np.nan, BEV=np.nan)
    
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

    #BEV
    # Work on the volume–time curve of this exhalation
    bev = np.nan
    if FVC>0 and ex_vol.size>1:
        n = ex_vol.size
        t_rel = np.arange(n, dtype=np.float64) * dt # t = 0 at onset, dt per sample
        
        # Preferred window: 5–25% of FVC (by exhaled volume)
        v_exh = ex_vol  # already volume-from-onset
        low = 0.05 * FVC
        high = 0.25 * FVC
        mask = (v_exh >= low) & (v_exh <= high)

        # Fallback: first 0.15 s if 5–25% window is too small
        if np.count_nonzero(mask) < 3:
            mask = t_rel <= 0.15  # 150 ms

        if np.count_nonzero(mask) >= 3:
            t_sel = t_rel[mask]
            v_sel = v_exh[mask]
            
            # Fit line: v ≈ a*t + b
            a, b = np.polyfit(t_sel, v_sel, 1)

            bev = float(abs(b))  # L, magnitude

    out['BEV'] = bev

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
        out['FEF25_75'] = float(np.nanmean(slab)) if slab.size else np.nan

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

    #   NEW: TLC/RV/VC from loop extremes  
    # Window for min/max volume: from exhalation onset to end of next inhale (if present)
    win_end = e_in if next_inhale_idx >= 0 else e_best
    vseg = vol[s_on:win_end+1]
    if vseg.size:
        v_min = float(np.nanmin(vseg))  # most negative (RV side)
        v_max = float(np.nanmax(vseg))  # most positive (TLC side)

        out['RV']  = abs(v_min)     # report RV as positive magnitude
        out['TLC'] = v_max          # far-right end
        out['VC']  = out['TLC'] - out['RV']  # VC = TLC - RV

    # FIVC (keep as before or recompute if you prefer a different definition)
    if next_inhale_idx >= 0:
        # magnitude of inhaled volume in that next inhale segment
        in_flow = flow[s_in:e_in+1]
        in_vol = -(np.cumsum(in_flow) * dt)
        out['FIVC'] = float(in_vol[-1]) if in_vol.size else np.nan

    return out

# ----------------- PATIENT DATA (EDIT WHEN NEEDED) ----------------- #
sex = "male"        
age = 30
height = 170
ethnicity = "others"

# ------------- FETCH PREDICTED + SD VALUES FROM GLI ---------------- #
if sex.upper() == "male":
    fev1_ref     = equations(age, height, ethnicity, fev1_males(age, height, ethnicity))
    fvc_ref      = equations(age, height, ethnicity, fvc_males(age, height, ethnicity))
    fev1fvc_ref  = equations(age, height, ethnicity, fev1fvc_males(age, height, ethnicity))
    fef2575_ref  = equations(age, height, ethnicity, fef2575_males(age, height, ethnicity))
    fef75_ref    = equations(age, height, ethnicity, fef75_males(age, height, ethnicity))
else:
    fev1_ref     = equations(age, height, ethnicity, fev1_females(age, height, ethnicity))
    fvc_ref      = equations(age, height, ethnicity, fvc_females(age, height, ethnicity))
    fev1fvc_ref  = equations(age, height, ethnicity, fev1fvc_females(age, height, ethnicity))
    fef2575_ref  = equations(age, height, ethnicity, fef2575_females(age, height, ethnicity))
    fef75_ref    = equations(age, height, ethnicity, fef75_females(age, height, ethnicity))

# predicted values
FEV1_pred     = fev1_ref["M"]
FVC_pred      = fvc_ref["M"]
FEV1FVC_pred  = fev1fvc_ref["M"]
FEF2575_pred  = fef2575_ref["M"]
FEF75_pred    = fef75_ref["M"]  

def gli_z(measured, L, M, S):
    """
    Official GLI 2012 Z-score formula using LMS model.
    """
    if measured is None or measured <= 0 or M <= 0 or S <= 0:
        return float("nan")
    if abs(L) < 1e-6:   # when L is ~0 → log-normal simplify
        return np.log(measured / M) / S
    return ((measured / M) ** L - 1) / (L * S)

PRESS_THR_PA = 3   # pressure threshold for segmentation (Pa)
START_HOLD   = 10      # samples above threshold to start segment    
END_HOLD     = 10     # samples below threshold to end segment
# Main
# Main
def main():
    # --- CONFIG ---
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Check files
    existing_files = [f for f in FILES if os.path.exists(f)]
    if not existing_files:
        raise FileNotFoundError("None of the provided FILES exist.")

    # --- SETUP PLOT (Once) ---
    if PLOT:
        # WINDOW 1: Flow-Volume Loop
        fig1, ax_fv = plt.subplots(figsize=(8, 8), constrained_layout=True)
        fig1.canvas.manager.set_window_title("Flow-Volume Loop") # Set Window Title
        fig1.suptitle("Analysis: Flow-Volume Loop", fontsize=16, weight='bold')

        # WINDOW 2: Volume-Time Graph
        fig2, ax_evs = plt.subplots(figsize=(10, 7), constrained_layout=True)
        fig2.canvas.manager.set_window_title("Volume-Time Graph") # Set Window Title
        fig2.suptitle("Analysis: Volume vs Time", fontsize=16, weight='bold')

        # WINDOW 3: Metrics Table
        # Made this window wider (12) to accommodate the table columns comfortably
        fig3, ax_metrics = plt.subplots(figsize=(12, 6), constrained_layout=True)
        fig3.canvas.manager.set_window_title("Metrics Summary") # Set Window Title
        fig3.suptitle("Metrics Summary", fontsize=16, weight='bold')
        
        # Ensure the table axis has no borders/ticks
        ax_metrics.axis('off')

        # --- STYLING FIGURE 1 (Flow-Volume) ---
        ax_fv.set_xlabel("Volume (L)", fontsize=10)
        ax_fv.set_ylabel("Flow (L/s)", fontsize=10)
        ax_fv.set_title("Flow–Volume Loop", fontsize=12)
        ax_fv.set_xlim([-1, 10]) 
        ax_fv.set_ylim([-10, 15]) 
        ax_fv.set_xticks(np.arange(-1, 11, 1))
        ax_fv.set_yticks(np.arange(-10, 16, 2))
        ax_fv.axhline(0, color='k', linewidth=1.2) 
        ax_fv.axvline(0, color='k', linewidth=1.2)
        ax_fv.grid(True, which='both', linestyle='-', alpha=0.3)

        # --- STYLING FIGURE 1 (Metrics Panel) ---
        ax_metrics.axis('off')
        ax_metrics.set_title("", fontsize=12)

        # --- STYLING FIGURE 2 (Volume-Time) ---
        ax_evs.set_xlabel("Time (s)", fontsize=10)
        ax_evs.set_ylabel("Volume (L)", fontsize=10)
        ax_evs.set_title("Volume vs Time", fontsize=12)
        ax_evs.set_xlim([-1, 15])
        ax_evs.set_ylim([-1, 9])
        ax_evs.set_xticks(np.arange(-1, 16, 1))
        ax_evs.set_yticks(np.arange(-1, 10, 1))
        ax_evs.axhline(0, color='k', linewidth=1.2)
        ax_evs.axvline(0, color='k', linewidth=1.2)
        ax_evs.grid(True, which='both', linestyle='-', alpha=0.3)

    table_data = []
    column_headers = []

    # --- LOOP THROUGH FILES ---
    for i, file_path in enumerate(FILES):
        trial_label = f"Trial {i+1}"
        color = colors[i % len(colors)]
        
        # TERMINAL LOGGING
        print(f" PROCESSING {trial_label}: {os.path.basename(file_path)}")

        if not os.path.exists(file_path):
            print(f"  [Skipped] File not found.")
            table_data.append({})
            column_headers.append(f"{trial_label}\n(Missing)")
            continue

        # 1) Parse
        p_raw = parse_log_file(file_path)
        if p_raw.size == 0:
            print("  [Skipped] No data found.")
            continue

        # 2) Filter + Baseline
        pf = preprocess_one(p_raw)
        t = np.arange(pf.size) * DT

        # 3) Detect Segments
        SEGMENT_PEAK_DEADBAND_PA = 150.0
        tail_ignore=100
        starts, ends = detect_segments(
    pf,
    deadband=100,
    tail_ignore=150,
    start_threshold=1,
    release_threshold=1,
    mean_amp_threshold=5,
)

        # 4) Pressure -> Flow
        flow = pressure_to_flow_segments(pf, starts, ends, min_peak_pa=SEGMENT_PEAK_DEADBAND_PA)
        flow_baseline = float(np.median(flow[:2000])) if flow.size >= 2000 else float(np.median(flow))

        # 5) Integrate to Volume
        vol = integrate_flow_to_volume(flow, DT)

        # 6) Compute Metrics
        metrics = compute_exhale_metrics(flow, vol, starts, ends, DT)
        extra = compute_additional_metrics(flow, vol, starts, ends, DT, metrics)
        
        # Merge into one dict BEFORE appending
        full_stats = {**metrics, **extra}
        
        column_headers.append(trial_label)

        # ---------------------- Measured values ---------------------- #
        meas_FEV1     = metrics["FEV1"]
        meas_FVC      = metrics["FVC"]
        meas_ratio    = metrics["FEV1_FVC"]/100 if metrics["FEV1_FVC"]>0 else float("nan") # convert %→fraction
        meas_FEF2575  = extra["FEF25_75"]
        meas_FEF75    = extra["FEF75"]

        # ---------------------- GLI Z-scores ------------------------- #
        FEV1_z     = gli_z(meas_FEV1,    fev1_ref["L"], fev1_ref["M"], fev1_ref["S"])
        FVC_z      = gli_z(meas_FVC,     fvc_ref["L"],  fvc_ref["M"],  fvc_ref["S"])
        FEV1FVC_z  = gli_z(meas_ratio,   fev1fvc_ref["L"], fev1fvc_ref["M"], fev1fvc_ref["S"])
        FEF2575_z  = gli_z(meas_FEF2575, fef2575_ref["L"], fef2575_ref["M"], fef2575_ref["S"])
        FEF75_z    = gli_z(meas_FEF75,   fef75_ref["L"],  fef75_ref["M"],  fef75_ref["S"])

        # ---------------------- push to results ---------------------- #
        # Update the full_stats dict that will be appended
        full_stats.update({
            "FEV1_pred": FEV1_pred,
            "FVC_pred": FVC_pred,
            "FEV1FVC_pred": FEV1FVC_pred * 100, # Store as percentage (e.g. 80.5) to match measured
            "FEF25_75_pred": FEF2575_pred,
            "FEF75_pred": FEF75_pred,

            "FEV1_z": FEV1_z,
            "FVC_z": FVC_z,
            "FEV1FVC_z": FEV1FVC_z,
            "FEF25_75_z": FEF2575_z,
            "FEF75_z": FEF75_z
        })

        # Append AFTER all updates
        table_data.append(full_stats)

        # --- PLOTTING (Flow–Volume Loop: Exhale + immediate Inhale) ---
        if PLOT and metrics["s_best"] >= 0:

            # Forced exhale indices
            s_on = metrics["s_on"]
            e_ex = metrics["e_best"]

            # Find the inhale segment immediately AFTER the forced exhale
            s_in, e_in = None, None
            for ss, ee in zip(starts, ends):
                if ss > e_ex and np.nanmean(flow[ss:ee+1]) < 0:
                    s_in, e_in = ss, ee
                    break

            # SINGLE volume reference (this is the key fix)
            v0 = vol[s_on]

            # Exhale limb
            fv_vol_ex = vol[s_on:e_ex+1] - v0
            fv_flow_ex = flow[s_on:e_ex+1]

            # --- TRIM unstable zero-crossing region (BACKTRACK METHOD) ---
            # Strategy: Find the PEAK of the exhale, then walk BACKWARDS 
            # until flow drops below a cutoff. This deletes the "hesitation loop".
            if len(fv_flow_ex) > 0:
                # 1. Find Peak
                idx_peak = np.argmax(fv_flow_ex)
                
                # 2. Threshold (10% of Peak)
                cutoff_flow = 0.10 * fv_flow_ex[idx_peak]
                
                # 3. Walk Backward
                start_trim_idx = 0
                for k in range(idx_peak, -1, -1):
                    if fv_flow_ex[k] < cutoff_flow:
                        start_trim_idx = k + 1
                        break
                
                # 4. Slice (Cut the garbage)
                fv_flow_ex = fv_flow_ex[start_trim_idx:]
                fv_vol_ex = fv_vol_ex[start_trim_idx:]
                
                # 5. RE-ZERO & ANCHOR
                if len(fv_vol_ex) > 0:
                    # A. Calculate a tiny volume offset for a natural slope 
                    # (Triangle area: 0.5 * flow * dt)
                    # This prevents the "Vertical Wall" look.
                    vol_offset = (fv_flow_ex[0] * DT) / 2.0
                    
                    # B. Shift the main curve so it starts slightly after 0
                    fv_vol_ex = (fv_vol_ex - fv_vol_ex[0]) + vol_offset
                    
                    # C. Prepend (0,0) to force the line to start at origin
                    fv_flow_ex = np.insert(fv_flow_ex, 0, 0.0)
                    fv_vol_ex  = np.insert(fv_vol_ex, 0, 0.0)

            if s_in is not None:
                # Inhale limb (continuous)
                fv_vol_in = vol[s_in:e_in+1] - v0
                fv_flow_in = flow[s_in:e_in+1]

                fv_vol = np.concatenate([fv_vol_ex, fv_vol_in])
                fv_flow = np.concatenate([fv_flow_ex, fv_flow_in])
            else:
                # Fallback: exhale only
                fv_vol = fv_vol_ex
                fv_flow = fv_flow_ex

            ax_fv.plot(
                fv_vol,
                fv_flow,
                linewidth=1.8,
                color=color,
                label=trial_label
            )

            # B. Prepare Volume-Time
            if metrics["s_best"] >= 0 and metrics["e_best"] >= 0:
                s_on = metrics["s_on"]
                e_best = metrics["e_best"]
                t_exhale = t[s_on:e_best+1] - t[s_on]
                vol_exhale = vol[s_on:e_best+1] - vol[s_on]
                vol_exhale = np.maximum.accumulate(vol_exhale)
                
                ax_evs.plot(t_exhale, vol_exhale, linewidth=1.5, color=color, label=trial_label)

    #        ATS/ERS BEST TRIAL LOGIC        #
    # Best FEV1 trial
    best_FEV1_idx = max(range(len(table_data)), key=lambda i: table_data[i].get("FEV1", -np.inf))

    # Best FVC trial
    best_FVC_idx  = max(range(len(table_data)), key=lambda i: table_data[i].get("FVC", -np.inf))

    # Best FLOW trial → highest combined effort (FEV1 + FVC)
    best_FLOW_idx = max(range(len(table_data)), 
                        key=lambda i: (table_data[i].get("FEV1",0) + table_data[i].get("FVC",0)))

    # Final values based on ATS/ERS
    final_FEV1 = table_data[best_FEV1_idx]["FEV1"]
    final_FVC  = table_data[best_FVC_idx]["FVC"]
    final_ratio = (final_FEV1 / final_FVC * 100) if final_FVC>0 else np.nan

    # FLOW metrics taken from best FLOW trial
    FLOW_best = table_data[best_FLOW_idx]
    final_PEF      = FLOW_best.get("PEF", np.nan)
    final_FEF2575  = FLOW_best.get("FEF25_75", np.nan)
    final_FEF75    = FLOW_best.get("FEF75", np.nan)

    #ZScore + %Pred using correct selected values
    final_FEV1_z = gli_z(final_FEV1, fev1_ref["L"], fev1_ref["M"], fev1_ref["S"])
    final_FVC_z  = gli_z(final_FVC,  fvc_ref["L"],  fvc_ref["M"],  fvc_ref["S"])
    final_ratio_z = gli_z(final_ratio/100, fev1fvc_ref["L"], fev1fvc_ref["M"], fev1fvc_ref["S"])
    # ADD THESE LINES to define the missing variables
    final_FEF2575_z = gli_z(final_FEF2575, fef2575_ref["L"], fef2575_ref["M"], fef2575_ref["S"])
    final_FEF75_z   = gli_z(final_FEF75,   fef75_ref["L"],  fef75_ref["M"],  fef75_ref["S"])
    final_PEF_pct      = (final_PEF/FEF75_pred*100) if FEF75_pred>0 else np.nan
    final_FEV1_pct     = (final_FEV1/FEV1_pred*100)
    final_FVC_pct      = (final_FVC/FVC_pred*100)
    final_FEF2575_pct  = (final_FEF2575/FEF2575_pred*100) if FEF2575_pred>0 else np.nan

    print(f"Best FEV1 → Trial {best_FEV1_idx+1}: {final_FEV1:.2f} L")
    print(f"Best FVC  → Trial {best_FVC_idx+1}: {final_FVC:.2f} L")
    print(f"Best Flow Trial (for other parameters) → Trial {best_FLOW_idx+1}")

    # === [MISSING CODE] Construct the dictionaries for the table loop ===
    final_vals = {
        "FEV1": final_FEV1,
        "FVC": final_FVC,
        "FEV1_FVC": final_ratio,
        "PEF": final_PEF,
        "FEF25_75": final_FEF2575,
        "FEF75": final_FEF75,
        # For other metrics, pull directly from the best flow trial
        "FET": FLOW_best.get("FET", np.nan),
        "TLC": FLOW_best.get("TLC", np.nan),
        "RV": FLOW_best.get("RV", np.nan),
        "VC": FLOW_best.get("VC", np.nan),
        "BEV": FLOW_best.get("BEV", np.nan),
        "FEF25": FLOW_best.get("FEF25", np.nan),
        "FEF50": FLOW_best.get("FEF50", np.nan),
        "FIVC": FLOW_best.get("FIVC", np.nan),
        "PIF": FLOW_best.get("PIF", np.nan),
    }

    final_z = {
        "FEV1": final_FEV1_z,
        "FVC": final_FVC_z,
        "FEV1_FVC": final_ratio_z,
        "FEF25_75": final_FEF2575_z,
        "FEF75": final_FEF75_z,
    }
    
    # --- FINAL TABLE GENERATION ---
    if PLOT and len(table_data) > 0:
        ax_fv.legend(loc='upper right', fontsize=9)
        ax_evs.legend(loc='lower right', fontsize=9)

        # --- IDENTIFY BEST TRIAL INDICES ---
        if table_data:
            # Index of trial with best FEV1
            idx_best_fev1 = max(range(len(table_data)), key=lambda i: table_data[i].get("FEV1", -np.inf))
            
            # Index of trial with best FVC
            idx_best_fvc = max(range(len(table_data)), key=lambda i: table_data[i].get("FVC", -np.inf))
            
            # Index of Best Flow Trial (Sum of FEV1 + FVC)
            idx_best_flow = max(range(len(table_data)), 
                                key=lambda i: (table_data[i].get("FEV1", 0) or 0) + (table_data[i].get("FVC", 0) or 0))
        else:
            idx_best_fev1 = idx_best_fvc = idx_best_flow = -1
            
        # 2. Define Rows with mapping to (Metric Key, Pred Key, Z Key)
        # Format: (Display Label, Metric_Key, Pred_Key, Z_Key, Format_String)
        rows_config = [
            ("FVC (L)",      "FVC",      "FVC_pred",      "FVC_z",      "{:.2f}"),
            ("FEV1 (L)",     "FEV1",     "FEV1_pred",     "FEV1_z",     "{:.2f}"),
            ("FEV1/FVC (%)", "FEV1_FVC", "FEV1FVC_pred",  "FEV1FVC_z",  "{:.1f}"), # Ratio is %
            ("FEF75 (L/s)",  "FEF75",    "FEF75_pred",    "FEF75_z",    "{:.2f}"),
            ("FEF25-75",     "FEF25_75", "FEF25_75_pred", "FEF25_75_z", "{:.2f}"),
            # Non-Zscore rows
            ("PEF (L/s)",    "PEF",      None,            None,         "{:.2f}"),
            ("PIF (L/s)",    "PIF",      None,            None,         "{:.2f}"),
            ("FET (s)",      "FET",      None,            None,         "{:.2f}"),
            ("TLC (L)",      "TLC",      None,            None,         "{:.2f}"),
            ("RV (L)",       "RV",       None,            None,         "{:.2f}"),
            ("VC (L)",       "VC",       None,            None,         "{:.2f}"),
            ("BEV (L)",      "BEV",      None,            None,         "{:.2f}"),
            ("FEF25 (L/s)",   "FEF25",    None,            None,         "{:.2f}"),
            ("FEF50 (L/s)",   "FEF50",    None,            None,         "{:.2f}"),
            ("FIVC (L)",      "FIVC",     None,            None,         "{:.2f}"),
        ]

        #        TABLE BUILD SECTION       == #

        cell_text = []
        row_labels = []


        # Column headers = Trials + 3 Summary Columns
        trial_headers = [f"Trial {i+1}" for i in range(len(table_data))]
        final_cols = trial_headers + ["Pred", "Best Values", "%Pred", "Z Score"]


        # Calculate the column index for "Best Values"
        # Index = Number of Trials (0 to N-1) + 1 (Pred column)
        best_val_col_idx = len(table_data) + 1 

        red_cells = [] 
        for r_idx, (label, data_key, pred_key, z_key, fmt) in enumerate(rows_config, start=1):
            row_labels.append(label)
            current_row = []

            # Determine which trial gets the star/color
            if data_key == "FEV1":
                target_idx = idx_best_fev1
            elif data_key == "FVC":
                target_idx = idx_best_fvc
            elif data_key == "FEV1_FVC":
                target_idx = -1 
            else:
                target_idx = idx_best_flow

            # A. Trial Data
            for i, d in enumerate(table_data):
                val = d.get(data_key, np.nan)
                if isinstance(val, (int, float)) and not np.isnan(val):
                    val_str = fmt.format(val)
                    # Check if this is the chosen best trial
                    if i == target_idx:
                        # Store (row, col) to color it later
                        red_cells.append((r_idx, i)) 
                    current_row.append(val_str)
                else:
                    current_row.append("-")

            # B. Predicted
            pred_val = table_data[0].get(pred_key, np.nan) if (table_data and pred_key) else np.nan
            current_row.append(fmt.format(pred_val) if not np.isnan(pred_val) else "-")

            # C. Best Values (ATS Selection)
            best_val = final_vals.get(data_key, np.nan)
            
            # --- NEW: Mark the "Best Values" column red as well ---
            if not np.isnan(best_val):
                 red_cells.append((r_idx, best_val_col_idx))
            # ----------------------------------------------------
            
            current_row.append(fmt.format(best_val) if not np.isnan(best_val) else "-")

            # D. % Predicted
            if not np.isnan(best_val) and not np.isnan(pred_val) and pred_val != 0:
                pct = (best_val / pred_val) * 100
                current_row.append(f"{pct:.1f}")
            else:
                current_row.append("-")

            # E. Z-Score
            z_val = final_z.get(data_key, np.nan)
            current_row.append(f"{z_val:.2f}" if not np.isnan(z_val) else "-")

            cell_text.append(current_row)

        # 3. Create Table
        final_cols = column_headers + ["Pred", "Best Values", "%Pred", "z-score"]
        
        the_table = ax_metrics.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=final_cols,
            loc='center',
            cellLoc='center',
            edges='horizontal'
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        the_table.scale(1, 1.5)

        #Apply red color to best values
        for (r, c) in red_cells:
            # Check if cell exists (safety)
            if (r, c) in the_table.get_celld():
                cell = the_table[r, c]
                cell.get_text().set_color('red')
                cell.get_text().set_weight('bold') # Optional: make it bold too

        plt.show()
        plt.show()

if __name__ == "__main__":
    main()


