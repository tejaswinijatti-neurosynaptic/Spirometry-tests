"""
This version FINISHES the pipeline by:
- Detecting inhale/exhale segments on filtered pressure
- Converting pressure -> flow using your 4-term model
- Integrating flow to volume
- Plotting Flow–Volume loops and Time-series
- Displaying a metric table with GLI Z-scores
- Performing QC checks and displaying messages
"""

from __future__ import annotations
from collections import deque
from typing import List, Iterable, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from scipy.signal import find_peaks

#  IMPORT GLI REFERENCES 
# Ensure 'GLI_2012_referencevalues_inputchangeinside.py' is in the same folder
try:
    from GLI_2012_referencevalues_inputchangeinside import (
        equations, fev1_males, fev1_females, fvc_females, fvc_males, 
        fev1fvc_males, fev1fvc_females, fef2575_females, fef2575_males, 
        fef75_females, fef75_males
    )
except ImportError:
    print("CRITICAL ERROR: 'GLI_2012_referencevalues_inputchangeinside.py' not found.")
    sys.exit(1)

#  USER SETTINGS 
FILES = [r"d:\Users\Tejaswini\Desktop\neurosyn\live plotting\New method\realtime_all\qc_check_logs\trial.log",
         ]

# Sampling period (seconds). 0.005 = 200 Hz
DT = 0.005

# Segmentation params
DEADBAND     = 100   # min segment length (samples)
TAIL_IGNORE  = 150   # ignore last N samples when segmenting

# Filtering + baseline params
TRI_WINDOW   = 20    # triangular half-window
INIT_MEAN_N  = 150   # samples for initial mean removal
END_MEAN_N   = 150   # samples used to estimate linear drift toward tail

# FEV1 onset detection
THRESH_ON_FRAC = 0.05   
THRESH_ON_ABS  = 0.10   

# Plot settings
PLOT = True
SAVE_FIGS = True
OUTDIR = "plots"

QC_MESSAGES = {
    "SLOW_START": {
        "operator": "Slow or hesitant start of forced exhalation",
        "patient":  "Please blow out harder and faster at the start"
    },
    "COUGH": {
        "operator": "Cough detected during forced exhalation",
        "patient":  "Please avoid coughing during the test"
    },
    "VARIABLE_FLOW": {
        "operator": "Inconsistent or stop–start effort",
        "patient":  "Please blow out smoothly without stopping"
    },
    "EARLY_TERMINATION": {
        "operator": "Exhalation ended too early",
        "patient":  "Please keep blowing until told to stop"
    },
    "EXTRA_BREATH": {
        "operator": "Unexpected number of breaths detected",
        "patient":  "Please perform a single inhale followed by a single forced exhale and a single inhale"
    },
}
ALLOW_QC_OVERRIDE = True   # operator mode

#  COEFFICIENT LOADING 
def load_coeffs(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "models", filename)
    
    if not os.path.exists(file_path):
        # Fallback to current dir if models dir doesn't exist
        file_path_alt = os.path.join(script_dir, filename)
        if os.path.exists(file_path_alt):
             file_path = file_path_alt
        else:
            print(f"\nCRITICAL ERROR: Could not find '{filename}' in 'models' folder.")
            sys.exit(1)
        
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {filename}")
        return np.array(data["coeffs"], dtype=float)
    except Exception as e:
        print(f"Error reading JSON {filename}: {e}")
        sys.exit(1)

print("Loading Coefficients...")
pull_coefficients = load_coeffs("coeffs_pull.json")
push_coefficients = load_coeffs("coeffs_push.json")

#  PARSING 
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
            
            # In first 5 frames, skip "Frame A" type
            if frame_count <= 5:
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

#  FILTERING 
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
    if p_raw.size == 0:
        return p_raw
    p0, _ = remove_initial_mean(p_raw, INIT_MEAN_N)
    pc, _ = remove_linear_tail_drift(p0, END_MEAN_N)
    w = triangular_weights(TRI_WINDOW)
    pf = streaming_fir(pc, w)
    return pf

#  BASIS AND MODEL 
def basis_from_filtered(pf: np.ndarray) -> np.ndarray:
    p = np.asarray(pf, dtype=float)
    a = np.abs(p)
    s = np.sign(p)
    return np.column_stack([
        s * np.sqrt(a),    # turbulent
        p,                 # laminar
        s * a ** (1/3),    # mid-range
        s,
        np.ones_like(p),
    ])

def pressure_to_flow_segments(pf: np.ndarray,
                              starts: list[int],
                              ends: list[int],
                              min_peak_pa: float = 200.0) -> np.ndarray:
    flow = np.zeros_like(pf, dtype=float)
    Phi = basis_from_filtered(pf)

    for s_idx, e_idx in zip(starts, ends):
        seg_abs_peak = float(np.nanmax(np.abs(pf[s_idx:e_idx+1]))) if e_idx >= s_idx else 0.0
        if seg_abs_peak < min_peak_pa:
            continue

        seg_mean = float(np.mean(pf[s_idx:e_idx+1]))
        coeffs = push_coefficients if seg_mean > 0 else pull_coefficients
        flow[s_idx:e_idx+1] = Phi[s_idx:e_idx+1] @ coeffs

    return flow

def integrate_flow_to_volume(flow: np.ndarray, dt: float) -> np.ndarray:
    vol = np.cumsum(flow) * dt
    return vol

#  SEGMENTATION 
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
    amp_threshold = mean_amp_threshold if mean_amp_threshold is not None else 5 # Default 5 if None

    kept_starts, kept_ends = [], []

    i = 0
    while i < n_eff:
        # Find start
        while i < n_eff and np.abs(x[i]) < start_threshold:
            i += 1
        if i >= n_eff:
            break
        s = i

        # Find end
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

def find_next_inhale(starts, ends, flow, after_idx):
    """
    Returns (s_in, e_in) of the first inhale segment after after_idx.
    If none found, returns (None, None)
    """
    for s, e in zip(starts, ends):
        if s > after_idx:
            if np.nanmean(flow[s:e+1]) < 0:
                return s, e
    return None, None

#  METRICS CALCULATIONS 
def compute_exhale_metrics(flow: np.ndarray, vol: np.ndarray, starts: np.ndarray, ends: np.ndarray, dt: float):
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
            FVC=np.nan, FEV1=np.nan, FEV1_FVC=np.nan, seg_index=-1, Exlen=0, ErrNum=9,
            s_best=-1, e_best=-1, s_on=-1, valid=False,
            message="No valid exhale detected."
        )

    i_best, s_best, e_best, _ = max(exhale_candidates, key=lambda t: t[3])

    seg_flow = flow[s_best:e_best+1]
    pef = float(np.nanmax(seg_flow)) if seg_flow.size else 0.0
    thresh = max(THRESH_ON_FRAC * pef, THRESH_ON_ABS)
    above = np.flatnonzero(seg_flow >= thresh)
    
    if above.size:
        s_on = s_best + int(above[0])
    else:
        s_on = s_best

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

    ex_flow = flow[s_on:e_best+1]
    ex_vol  = vol[s_on:e_best+1] - vol[s_on]
    if ex_flow.size < 2:
        return out

    FVC = float(ex_vol[-1])

    # BEV
    bev = np.nan
    if FVC>0 and ex_vol.size>1:
        n = ex_vol.size
        t_rel = np.arange(n, dtype=np.float64) * dt
        low, high = 0.05 * FVC, 0.25 * FVC
        mask = (ex_vol >= low) & (ex_vol <= high)
        if np.count_nonzero(mask) < 3:
            mask = t_rel <= 0.15

        if np.count_nonzero(mask) >= 3:
            t_sel = t_rel[mask]
            v_sel = ex_vol[mask]
            _, b = np.polyfit(t_sel, v_sel, 1)
            bev = float(abs(b))
    out['BEV'] = bev

    out['PEF'] = float(np.nanmax(ex_flow))

    def first_cross_idx(target):
        for i in range(len(ex_vol)-1):
            if ex_vol[i] <= target and ex_vol[i+1] > target:
                return i
        return -1

    v25, v50, v75 = 0.25 * FVC, 0.50 * FVC, 0.75 * FVC
    i25, i50, i75 = first_cross_idx(v25), first_cross_idx(v50), first_cross_idx(v75)
    
    if i25 >= 0: out['FEF25'] = float(ex_flow[i25])
    if i50 >= 0: out['FEF50'] = float(ex_flow[i50])
    if i75 >= 0: out['FEF75'] = float(ex_flow[i75])

    if i25 >= 0 and i75 > i25:
        slab = ex_flow[i25:i75+1]
        out['FEF25_75'] = float(np.nanmean(slab)) if slab.size else np.nan

    out['FET'] = float((e_best - s_on + 1) * dt)

    next_inhale_idx = -1
    s_in, e_in = -1, -1
    for j, (ss, ee) in enumerate(zip(starts, ends)):
        if ss > e_best and np.nanmean(flow[ss:ee+1]) < 0:
            next_inhale_idx = j
            s_in, e_in = ss, ee
            break

    if next_inhale_idx >= 0:
        in_flow = flow[s_in:e_in+1]
        out['PIF'] = float(-np.nanmin(in_flow))

    win_end = e_in if next_inhale_idx >= 0 else e_best
    vseg = vol[s_on:win_end+1]
    if vseg.size:
        v_min = float(np.nanmin(vseg))
        v_max = float(np.nanmax(vseg))
        out['RV']  = abs(v_min)
        out['TLC'] = v_max
        out['VC']  = out['TLC'] - out['RV']

    if next_inhale_idx >= 0:
        in_flow = flow[s_in:e_in+1]
        in_vol = -(np.cumsum(in_flow) * dt)
        out['FIVC'] = float(in_vol[-1]) if in_vol.size else np.nan

    return out

#  PATIENT DATA 
sex = "female"        
age = 25
height = 155
ethnicity = "South East Asian" 

#  GLI PREDICTIONS 
if sex.upper() == "MALE":
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

FEV1_pred     = fev1_ref["M"]
FVC_pred      = fvc_ref["M"]
FEV1FVC_pred  = fev1fvc_ref["M"]
FEF2575_pred  = fef2575_ref["M"]
FEF75_pred    = fef75_ref["M"]  

#  QC FUNCTIONS 
def check_extra_breaths(flow, starts, ends):
    if len(starts) != 3:
        return True, {"segments": len(starts), "reason": "Expected 3 breaths"}
    signs = [np.sign(np.nanmean(flow[s:e+1])) for s, e in zip(starts, ends)]
    if not (signs[0] < 0 and signs[1] > 0 and signs[2] < 0):
        return True, {"pattern": signs, "reason": "Invalid breath order"}
    return False, {"pattern": signs}

def check_variable_flow(flow, metrics, dt):
    s_on = metrics.get("s_on", -1)
    e_ex = metrics.get("e_exhale", -1)
    if s_on < 0 or e_ex <= s_on: return False, {}

    ex_flow = flow[s_on:e_ex+1]
    if ex_flow.size < 20: return False, {}

    pef_idx = int(np.argmax(ex_flow))
    pef = ex_flow[pef_idx]
    if pef <= 0: return False, {}

    post = ex_flow[pef_idx+1:]
    if post.size < 20: return False, {}

    peaks, props = find_peaks(post, height=0.25 * pef, distance=int(0.06 / dt))
    meaningful_peaks = 0
    LOOKBACK = int(0.30 / dt)
    for i, p in enumerate(peaks):
        left = max(0, p - LOOKBACK)
        dip = np.min(post[left:p+1])
        if (props["peak_heights"][i] - dip) > 0.20 * pef:
            meaningful_peaks += 1

    pause_restart = False
    MIN_PAUSE = int(0.10 / dt)
    i = 0
    while i < post.size - MIN_PAUSE:
        if np.all(post[i:i+MIN_PAUSE] < 0.12):
            if np.any(post[i+MIN_PAUSE:] > 0.35):
                pause_restart = True; break
            i += MIN_PAUSE
        else: i += 1

    late_fail = False
    win = max(3, int(0.04 / dt))
    smooth = np.convolve(post, np.ones(win)/win, mode="same")
    dflow = np.diff(smooth)
    i = 0
    while i < dflow.size:
        if dflow[i] > 0.20:
            j = i
            while j < dflow.size and dflow[j] > 0.20: j += 1
            if (j - i) >= int(0.08 / dt):
                late_fail = True; break
            i = j
        else: i += 1

    return (meaningful_peaks >= 1 or pause_restart or late_fail), {
        "meaningful_peaks": meaningful_peaks, "pause_restart": pause_restart, "late_oscillation": late_fail
    }

def check_slow_start_bev_beta(flow, vol, dt, metrics):
    s_on = metrics.get("s_on", -1); e_ex = metrics.get("e_exhale", -1)
    if s_on < 0 or e_ex <= s_on: return False, {}
    
    ex_flow = flow[s_on:e_ex+1]
    ex_vol  = vol[s_on:e_ex+1] - vol[s_on]
    if ex_flow.size < 5: return False, {}

    bev = metrics.get("BEV", np.nan)
    FVC = ex_vol[-1]
    bev_fail = (not np.isnan(bev) and FVC > 0 and bev > max(0.15, 0.05 * FVC))

    pef_idx = int(np.argmax(ex_flow))
    pef = ex_flow[pef_idx]; vol_pef = ex_vol[pef_idx]
    beta = np.degrees(np.arctan(pef / vol_pef)) if (pef > 0 and vol_pef > 0) else np.nan
    beta_fail = beta <70 if not np.isnan(beta) else False

    return (bev_fail and beta_fail), {"BEV": bev, "beta_deg": beta}
    #return (bev_fail), {"BEV": bev, "beta_deg": beta}

def check_cough(flow, metrics, dt):
    s_on = metrics.get("s_on", -1)
    e_ex = metrics.get("e_exhale", -1)
    if s_on < 0 or e_ex <= s_on:
        return False, {}

    ex_flow = flow[s_on:e_ex+1]
    if ex_flow.size < int(0.5 / dt):   # need at least 0.5 s
        return False, {}

    #  parameters 
    IGNORE_INITIAL = int(0.20 / dt)     # ignore first 200 ms
    MIN_DIP_FRAC   = 0.30               # dip must fall below 30% of PEF
    SPIKE_FRAC     = 0.25               # spike must exceed 25% PEF
    MIN_SPIKES     = 2                  # suppress single-spike false positives
    CLUSTER_WIN    = int(0.03 / dt)     # 30 ms spike clustering

    #  derivatives 
    dflow = np.diff(ex_flow) / dt
    dflow[:IGNORE_INITIAL] = 0.0

    pef_idx = int(np.argmax(ex_flow))
    pef = ex_flow[pef_idx]
    if pef <= 0:
        return False, {}

    # adaptive spike threshold
    spike_thresh = max(150.0, SPIKE_FRAC * pef / dt)
    spike_idxs = np.where(dflow > spike_thresh)[0]

    cough_events = 0
    valid_spikes = []

    for idx in spike_idxs:
        # must be post-PEF
        if idx <= pef_idx:
            continue

        # check for flow dip before spike
        lookback = max(0, idx - int(0.05 / dt))
        pre_min = np.min(ex_flow[lookback:idx+1])
        if pre_min > MIN_DIP_FRAC * pef:
            continue

        valid_spikes.append(idx)

    # cluster spikes into cough events
    i = 0
    while i < len(valid_spikes):
        j = i + 1
        while j < len(valid_spikes) and valid_spikes[j] - valid_spikes[i] <= CLUSTER_WIN:
            j += 1
        cough_events += 1
        i = j

    return cough_events >= MIN_SPIKES, {
        "cough_events": cough_events,
        "valid_spikes": len(valid_spikes)
    }

def check_early_termination(flow, vol, metrics, dt):
    if metrics.get("slow_start_failed", False):
        return False, {}
    s_on = metrics.get("s_on", -1); e_ex = metrics.get("e_exhale", -1)
    if s_on < 0 or e_ex <= s_on: return False, {}
    
    FET = (e_ex - s_on + 1) * dt
    ex_vol = vol[s_on:e_ex+1] - vol[s_on]
    plateau = False
    if ex_vol.size >= int(1.0/dt):
        tail = ex_vol[-int(1.0/dt):]
        if (np.max(tail) - np.min(tail)) < 0.025: plateau = True
            
    return (FET < 5 or not plateau), {"FET": FET, "plateau": plateau}

QC_CHECKS = [
    ("EXTRA_BREATH", check_extra_breaths),
    ("SLOW_START",   check_slow_start_bev_beta),
    ("COUGH",        check_cough),
    ("VARIABLE_FLOW",check_variable_flow),
    ("EARLY_TERMINATION", check_early_termination),
]

def qc_spirometry(flow, vol, dt, metrics, extra, starts, ends):
    failures = []
    for code, fn in QC_CHECKS:
        if code == "EXTRA_BREATH": fail, details = fn(flow, starts, ends)
        elif code == "SLOW_START":
            fail, details = fn(flow, vol, dt, {**metrics, **extra})
            if fail:
                metrics["slow_start_failed"] = True
        elif code == "EARLY_TERMINATION": fail, details = fn(flow, vol, metrics, dt)
        else: fail, details = fn(flow, metrics, dt)
        
        if fail:
            failures.append({
                "code": code, "details": details,
                "operator_msg": QC_MESSAGES[code]["operator"],
                "patient_msg":  QC_MESSAGES[code]["patient"]
            })

    if failures:
        return {"status": "FAIL", "codes": [f["code"] for f in failures], "failures": failures}
    return {"status": "PASS", "codes": [], "failures": []}

def gli_z(measured, L, M, S):
    if measured is None or measured <= 0 or M <= 0 or S <= 0:
        return float("nan")
    if abs(L) < 1e-6:
        return np.log(measured / M) / S
    return ((measured / M) ** L - 1) / (L * S)

#  MAIN 
def main():
    colors = ['#79dbf9', '#002a5c', '#fc5863']
    
    if PLOT:
        fig1, ax_fv = plt.subplots(figsize=(8, 8), constrained_layout=True)
        fig1.canvas.manager.set_window_title("Flow-Volume Loop")
        fig1.suptitle("Analysis: Flow-Volume Loop", fontsize=16, weight='bold')
        
        fig2, ax_evs = plt.subplots(figsize=(10, 7), constrained_layout=True)
        fig2.canvas.manager.set_window_title("Volume-Time Graph")
        fig2.suptitle("Analysis: Volume vs Time", fontsize=16, weight='bold')

        fig3, ax_metrics = plt.subplots(figsize=(12, 6), constrained_layout=True)
        fig3.canvas.manager.set_window_title("Metrics Summary")
        fig3.suptitle("Metrics Summary", fontsize=16, weight='bold')
        ax_metrics.axis('off')

        # STYLING FIGURE 1 (Flow-Volume)
        ax_fv.set_xlabel("Volume (L)", fontsize=10)
        ax_fv.set_ylabel("Flow (L/s)", fontsize=10)
        ax_fv.set_title("Flow–Volume Loop", fontsize=12)
        ax_fv.set_xlim([-1, 10]) 
        ax_fv.set_ylim([-10, 15]) 
        ax_fv.set_xticks(np.arange(-1, 11, 1))
        ax_fv.set_yticks(np.arange(-10, 16, 2))
        ax_fv.axhline(0, color='k', linewidth=1.2) 
        ax_fv.axvline(0, color='k', linewidth=1.2)
        ax_fv.grid(True, which='both', linestyle='-', alpha=0.1)

        # STYLING FIGURE 1 (Metrics Panel)
        ax_metrics.axis('off')
        ax_metrics.set_title("", fontsize=12)

        # STYLING FIGURE 2 (Volume-Time) 
        ax_evs.set_xlabel("Time (s)", fontsize=10)
        ax_evs.set_ylabel("Volume (L)", fontsize=10)
        ax_evs.set_title("Volume vs Time", fontsize=12)
        ax_evs.set_xlim([-1, 15])
        ax_evs.set_ylim([-1, 9])
        ax_evs.set_xticks(np.arange(-1, 16, 1))
        ax_evs.set_yticks(np.arange(-1, 10, 1))
        ax_evs.axhline(0, color='k', linewidth=1.2)
        ax_evs.axvline(0, color='k', linewidth=1.2)
        ax_evs.grid(True, which='both', linestyle='-', alpha=0.1)

    table_data = []

    for i, file_path in enumerate(FILES):
        trial_label = f"Trial {i+1}"
        color = colors[i % len(colors)]
        print(f" PROCESSING {trial_label}: {os.path.basename(file_path)}")

        if not os.path.exists(file_path):
            print(f"  [Skipped] File not found.")
            continue

        p_raw = parse_log_file(file_path)
        if p_raw.size == 0:
            print("  [Skipped] No data found.")
            continue

        pf = preprocess_one(p_raw)
        t = np.arange(pf.size) * DT

        #  FIX: INDENTATION CORRECTED HERE 
        starts, ends = detect_segments(
            pf,
            deadband=100,
            tail_ignore=150,
            start_threshold=10,
            release_threshold=2,
            mean_amp_threshold=15,
        )

        flow = pressure_to_flow_segments(pf, starts, ends, min_peak_pa=10.0)
        vol = integrate_flow_to_volume(flow, DT)
        metrics = compute_exhale_metrics(flow, vol, starts, ends, DT)

        s_best = metrics["s_best"]
        e_best = metrics["e_best"]     # ← ORIGINAL segment end
        s_on   = metrics["s_on"]

        # NOW extend exhale
        s_in, e_in = find_next_inhale(starts, ends, flow, e_best)

        if s_in is not None:
            e_exhale = s_in - 1
        else:
            e_exhale = len(flow) - 1

        # STORE IT
        metrics["e_exhale"] = e_exhale
        metrics["s_in"] = s_in

        extra = compute_additional_metrics(flow, vol, starts, ends, DT, metrics)
        qc = qc_spirometry(flow, vol, DT, metrics, extra, starts, ends)

        if qc["status"] == "FAIL":
            print(f"  QC FAIL: {qc['codes']}")

        full_stats = {**metrics, **extra}
        full_stats.update({
            "QC_STATUS": qc["status"],
            "QC_REASON": qc["codes"],
            "QC_PATIENT_MSG": qc.get("patient_msg", ""),
            "FEV1_pred": FEV1_pred, "FVC_pred": FVC_pred, "FEV1FVC_pred": FEV1FVC_pred * 100,
            "FEF25_75_pred": FEF2575_pred, "FEF75_pred": FEF75_pred,
            "FEV1_z": gli_z(metrics["FEV1"], fev1_ref["L"], fev1_ref["M"], fev1_ref["S"]),
            "FVC_z": gli_z(metrics["FVC"], fvc_ref["L"], fvc_ref["M"], fvc_ref["S"]),
            "FEV1FVC_z": gli_z(metrics["FEV1_FVC"]/100 if metrics["FEV1_FVC"] else 0, fev1fvc_ref["L"], fev1fvc_ref["M"], fev1fvc_ref["S"]),
            "FEF25_75_z": gli_z(extra["FEF25_75"], fef2575_ref["L"], fef2575_ref["M"], fef2575_ref["S"]),
            "FEF75_z": gli_z(extra["FEF75"], fef75_ref["L"], fef75_ref["M"], fef75_ref["S"])
        })
        table_data.append(full_stats)

        if PLOT:
            s_on = metrics.get("s_on", -1); e_ex = metrics.get("e_exhale", -1)
            if s_on >= 0 and e_ex >= 0:
                v0 = vol[s_on]
                fv_vol_ex = vol[s_on:e_ex+1] - v0
                fv_flow_ex = flow[s_on:e_ex+1]

                if len(fv_flow_ex) > 0:
                    idx_peak = np.argmax(fv_flow_ex)
                    cutoff_flow = 0.10 * fv_flow_ex[idx_peak]
                    start_trim_idx = 0
                    for k in range(idx_peak, -1, -1):
                        if fv_flow_ex[k] < cutoff_flow:
                            start_trim_idx = k + 1; break
                    fv_flow_ex = fv_flow_ex[start_trim_idx:]
                    fv_vol_ex = fv_vol_ex[start_trim_idx:]
                    if len(fv_vol_ex) > 0:
                         vol_offset = (fv_flow_ex[0] * DT) / 2.0
                         fv_vol_ex = (fv_vol_ex - fv_vol_ex[0]) + vol_offset
                         fv_flow_ex = np.insert(fv_flow_ex, 0, 0.0)
                         fv_vol_ex  = np.insert(fv_vol_ex, 0, 0.0)

                s_in, e_in = None, None
                for ss, ee in zip(starts, ends):
                    if ss > e_ex and np.nanmean(flow[ss:ee+1]) < 0:
                        s_in, e_in = ss, ee; break
                
                if s_in:
                    fv_vol = np.concatenate([fv_vol_ex, vol[s_in:e_in+1] - v0])
                    fv_flow = np.concatenate([fv_flow_ex, flow[s_in:e_in+1]])
                else:
                    fv_vol, fv_flow = fv_vol_ex, fv_flow_ex
                
                qc_failed = (qc["status"] == "FAIL")
                ax_fv.plot(fv_vol, fv_flow, ls="--" if qc_failed else "-", alpha=0.7 if qc_failed else 1.0, 
                           color=color, label=f"{trial_label} {qc['codes'] if qc_failed else ''}")
                
                t_exhale = t[s_on:e_ex+1] - t[s_on]
                vol_exhale = np.maximum.accumulate(vol[s_on:e_ex+1] - vol[s_on])
                ax_evs.plot(t_exhale, vol_exhale, lw=1.5, color=color, label=trial_label)

    if not table_data:
        print("No valid spirometry data processed.")
        return

    # Select best trials (ATS/ERS)
    best_FEV1_trial = max(table_data, key=lambda d: d.get("FEV1", -np.inf))
    best_FVC_trial  = max(table_data, key=lambda d: d.get("FVC", -np.inf))
    best_FLOW_trial = max(table_data, key=lambda d: (d.get("FEV1", 0) or 0) + (d.get("FVC", 0) or 0))

    final_FEV1 = best_FEV1_trial["FEV1"]
    final_FVC  = best_FVC_trial["FVC"]
    final_ratio = (final_FEV1 / final_FVC * 100) if final_FVC>0 else np.nan

    FLOW_best = best_FLOW_trial
    final_vals = {
        "FEV1": final_FEV1, "FVC": final_FVC, "FEV1_FVC": final_ratio,
        "PEF": FLOW_best.get("PEF", np.nan), "FEF25_75": FLOW_best.get("FEF25_75", np.nan),
        "FEF75": FLOW_best.get("FEF75", np.nan), "FET": FLOW_best.get("FET", np.nan),
        "TLC": FLOW_best.get("TLC", np.nan), "RV": FLOW_best.get("RV", np.nan),
        "VC": FLOW_best.get("VC", np.nan), "BEV": FLOW_best.get("BEV", np.nan),
        "FEF25": FLOW_best.get("FEF25", np.nan), "FEF50": FLOW_best.get("FEF50", np.nan),
        "FIVC": FLOW_best.get("FIVC", np.nan), "PIF": FLOW_best.get("PIF", np.nan),
    }

    #  FIX: ALIGN Z-KEYS WITH TABLE CONFIG 
    final_z = {
        "FEV1": gli_z(final_FEV1, fev1_ref["L"], fev1_ref["M"], fev1_ref["S"]),
        "FVC":  gli_z(final_FVC, fvc_ref["L"], fvc_ref["M"], fvc_ref["S"]),
        "FEV1_FVC": gli_z(final_ratio/100, fev1fvc_ref["L"], fev1fvc_ref["M"], fev1fvc_ref["S"]),
        "FEF25_75": gli_z(final_vals["FEF25_75"], fef2575_ref["L"], fef2575_ref["M"], fef2575_ref["S"]),
        "FEF75":    gli_z(final_vals["FEF75"], fef75_ref["L"], fef75_ref["M"], fef75_ref["S"])
    }

    if PLOT:
        ax_fv.legend(loc='upper right', fontsize=9)
        ax_evs.legend(loc='upper right', fontsize=9)
        
        # Table Configuration
        idx_best_fev1 = max(range(len(table_data)), key=lambda i: table_data[i].get("FEV1", -np.inf))
        idx_best_fvc = max(range(len(table_data)), key=lambda i: table_data[i].get("FVC", -np.inf))
        idx_best_flow = max(range(len(table_data)), key=lambda i: (table_data[i].get("FEV1", 0) or 0) + (table_data[i].get("FVC", 0) or 0))

        rows_config = [
            ("FVC (L)",      "FVC",      "FVC_pred",      "FVC_z",      "{:.2f}"),
            ("FEV1 (L)",     "FEV1",     "FEV1_pred",     "FEV1_z",     "{:.2f}"),
            ("FEV1/FVC (%)", "FEV1_FVC", "FEV1FVC_pred",  "FEV1FVC_z",  "{:.1f}"),
            ("FEF75 (L/s)",  "FEF75",    "FEF75_pred",    "FEF75_z",    "{:.2f}"),
            ("FEF25-75",     "FEF25_75", "FEF25_75_pred", "FEF25_75_z", "{:.2f}"),
            ("PEF (L/s)",    "PEF",      None,            None,         "{:.2f}"),
            ("PIF (L/s)",    "PIF",      None,            None,         "{:.2f}"),
            ("FET (s)",      "FET",      None,            None,         "{:.2f}"),
            ("TLC (L)",      "TLC",      None,            None,         "{:.2f}"),
            ("RV (L)",       "RV",       None,            None,         "{:.2f}"),
            ("VC (L)",       "VC",       None,            None,         "{:.2f}"),
            ("BEV (L)",      "BEV",      None,            None,         "{:.2f}"),
            ("FEF25 (L/s)",  "FEF25",    None,            None,         "{:.2f}"),
            ("FEF50 (L/s)",  "FEF50",    None,            None,         "{:.2f}"),
            ("FIVC (L)",     "FIVC",     None,            None,         "{:.2f}"),
        ]

        cell_text = []
        row_labels = [r[0] for r in rows_config]
        final_cols = [f"Trial {i+1}" for i in range(len(table_data))] + ["Pred", "Best Values", "%Pred", "Z Score"]
        
        red_cells = [] 
        for r_idx, (label, data_key, pred_key, z_key, fmt) in enumerate(rows_config, start=1):
            current_row = []
            
            # Target index for coloring best trial
            if data_key == "FEV1": target_idx = idx_best_fev1
            elif data_key == "FVC": target_idx = idx_best_fvc
            elif data_key == "FEV1_FVC": target_idx = -1
            else: target_idx = idx_best_flow

            # 1. Trial Data
            for i, d in enumerate(table_data):
                val = d.get(data_key, np.nan)
                if isinstance(val, (int, float)) and not np.isnan(val):
                    current_row.append(fmt.format(val))
                    if i == target_idx: red_cells.append((r_idx, i)) 
                else: current_row.append("-")

            # 2. Predicted
            pred_val = table_data[0].get(pred_key, np.nan) if (table_data and pred_key) else np.nan
            current_row.append(fmt.format(pred_val) if not np.isnan(pred_val) else "-")

            # 3. Best Value
            best_val = final_vals.get(data_key, np.nan)
            if not np.isnan(best_val): red_cells.append((r_idx, len(table_data) + 1))
            current_row.append(fmt.format(best_val) if not np.isnan(best_val) else "-")

            # 4. % Pred
            if not np.isnan(best_val) and not np.isnan(pred_val) and pred_val != 0:
                current_row.append(f"{(best_val/pred_val)*100:.1f}")
            else: current_row.append("-")

            # 5. Z-Score
            z_val = final_z.get(data_key, np.nan)
            current_row.append(f"{z_val:.2f}" if not np.isnan(z_val) else "-")

            cell_text.append(current_row)

        the_table = ax_metrics.table(cellText=cell_text, rowLabels=row_labels, colLabels=final_cols,
                                     loc='center', cellLoc='center', edges='horizontal')
        the_table.auto_set_font_size(False); the_table.set_fontsize(9); the_table.scale(1, 1.5)

        for (r, c) in red_cells:
            if (r, c) in the_table.get_celld():
                the_table[r, c].get_text().set_color('red')
                the_table[r, c].get_text().set_weight('bold')

        plt.show()

if __name__ == "__main__":
    main()
