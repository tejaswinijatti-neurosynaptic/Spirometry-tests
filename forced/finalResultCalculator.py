from __future__ import annotations
import asyncio
from collections import deque
from typing import List, Iterable
import os
import numpy as np
import json
import sys
from scipy.signal import find_peaks
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

#  IMPORT GLI REFERENCES 
# Ensure 'GLI_2012_referencevalues.py' is in the same folder
try:
    from forced.GLI_2012_referencevalues import (
        equations, fev1_males, fev1_females, fvc_females, fvc_males, 
        fev1fvc_males, fev1fvc_females, fef2575_females, fef2575_males, 
        fef75_females, fef75_males
    )
except ImportError:
    logger.error("CRITICAL ERROR: 'GLI_2012_referencevalues.py' not found.")
    sys.exit(1)

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

_SEND_ASYNC = None
DEBUGGING = False

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

def resource_path(relative_path):
    """
    Get absolute path to resource, works for:
    - normal Python
    - PyInstaller onefile
    - PyInstaller onedir
    """
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


def load_coeffs(filename):
    file_path = resource_path(os.path.join("models", filename))

    if not os.path.exists(file_path):
        logger.info(f"\nCRITICAL ERROR: Could not find '{filename}' in 'models' folder.")
        logger.info(f"Path searched: {file_path}")
        sys.exit(1)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"[Realtime] Loaded {filename}")
        return np.array(data["coeffs"], dtype=float)

    except Exception as e:
        logger.info(f"Error reading JSON {filename}: {e}")
        sys.exit(1)

logger.info("Loading Coefficients...")
pull_coefficients = load_coeffs("coeffs_pull.json")
push_coefficients = load_coeffs("coeffs_push.json")

def set_sender(send_async_fn):
    """
    send_async_fn must be an async function: await send_async_fn("msgType~payload")
    """
    global _SEND_ASYNC
    _SEND_ASYNC = send_async_fn

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


async def calculateFinalResult(p_raw):
    table_data = []
    if True:
        if p_raw.size == 0:
            logger.info("  [Skipped] No data found.")
            return
        
        pf = preprocess_one(p_raw)
        t = np.arange(pf.size) * DT

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
        e_best = metrics["e_best"]
        s_on   = metrics["s_on"]

        # Extend exhale
        s_in, e_in = find_next_inhale(starts, ends, flow, e_best)
        if s_in is not None:
            e_exhale = s_in - 1
        else:
            e_exhale = len(flow) - 1

        metrics["e_exhale"] = e_exhale
        metrics["s_in"] = s_in

        extra = compute_additional_metrics(flow, vol, starts, ends, DT, metrics)
        qc = qc_spirometry(flow, vol, DT, metrics, extra, starts, ends)

        if qc["status"] == "FAIL":
            logger.info(f"  QC FAIL: {qc['codes']}")

            # ADD QC STATUS SEND
            if _SEND_ASYNC is not None:
                loop = asyncio.get_running_loop()
        
                qc_status = {
                    "status": "FAIL" if qc["status"] == "FAIL" else "PASS",
                    "codes": [str(code) for code in qc['codes']],  # Convert to strings
                    "message": f"QC {'FAIL' if qc['status'] == 'FAIL' else 'PASS'}: {qc['codes']}"
                }
    
                enc_qc = json.dumps(qc_status, separators=(',', ':'))
                loop.create_task(_SEND_ASYNC(f"dataFromLib~108~spiroQCStatus~{enc_qc}"))
        else:
            if _SEND_ASYNC is not None:
                loop = asyncio.get_running_loop()
        
                qc_status = {
                    "status":  "PASS"
                }
    
                enc_qc = json.dumps(qc_status, separators=(',', ':'))
                loop.create_task(_SEND_ASYNC(f"dataFromLib~108~spiroQCStatus~{enc_qc}"))

        # Store stats for the table later
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

        #  PREPARE DATA BUFFERS FOR CSV & PLOTTING
        # Initialize lists with None (empty) for the whole duration
        csv_flow_fv = [None] * len(t)
        csv_vol_fv  = [None] * len(t)
        csv_vol_vt  = [None] * len(t)

        s_on = metrics.get("s_on", -1)
        e_ex = metrics.get("e_exhale", -1)

        if s_on >= 0 and e_ex >= 0:
            # 1. PREPARE FV LOOP DATA (Exhale + Inhale) 
            v0 = vol[s_on]
            raw_vol_ex = vol[s_on:e_ex+1] - v0
            raw_flow_ex = flow[s_on:e_ex+1]

            # Apply Trimming (same as plot)
            if len(raw_flow_ex) > 0:
                idx_peak = np.argmax(raw_flow_ex)
                cutoff_flow = 0.10 * raw_flow_ex[idx_peak]
                start_trim_idx = 0
                for k in range(idx_peak, -1, -1):
                    if raw_flow_ex[k] < cutoff_flow:
                        start_trim_idx = k + 1; break
                
                trimmed_flow = raw_flow_ex[start_trim_idx:]
                trimmed_vol  = raw_vol_ex[start_trim_idx:]

                # Apply Offset
                if len(trimmed_vol) > 0:
                    vol_offset = (trimmed_flow[0] * DT) / 2.0
                    final_vol_ex = (trimmed_vol - trimmed_vol[0]) + vol_offset
                    final_flow_ex = trimmed_flow 

                    # Save Exhale to CSV buffers
                    abs_start = s_on + start_trim_idx
                    for k in range(len(final_vol_ex)):
                        if abs_start + k < len(t):
                            csv_flow_fv[abs_start + k] = final_flow_ex[k]
                            csv_vol_fv[abs_start + k]  = final_vol_ex[k]

            # Append Inhale (if exists)
            s_in, e_in = None, None
            for ss, ee in zip(starts, ends):
                if ss > e_ex and np.nanmean(flow[ss:ee+1]) < 0:
                    s_in, e_in = ss, ee; break
            
            if s_in:
                inhale_vol = vol[s_in:e_in+1] - v0
                inhale_flow = flow[s_in:e_in+1]
                for k in range(len(inhale_vol)):
                    if s_in + k < len(t):
                        csv_flow_fv[s_in + k] = inhale_flow[k]
                        csv_vol_fv[s_in + k]  = inhale_vol[k]

            #  2. PREPARE VOL-TIME DATA (Exhale Only) 
            # Strictly matches the "Best Trial" Volume-Time graph
            if len(raw_vol_ex) > 0:
                vt_data = np.maximum.accumulate(raw_vol_ex)
                for k in range(len(vt_data)):
                    if s_on + k < len(t):
                        csv_vol_vt[s_on + k] = vt_data[k]

        #  GENERATE SEPARATE CSV FOR THIS TRIAL
        # Logic: input "data/test1.log" -> output "data/test1.csv"
        # Instead of writing to file, collect as list of lists
        csv_data = []

        # Same header
        csv_data.append(["Time_s", "Pressure_Pa", "FV_Flow_Lps", "FV_Volume_L", "VT_Volume_Exhale_L"])

        # Same format helper
        def fmt(val, p=4): 
            return f"{val:.{p}f}" if val is not None else ""

        # Build rows exactly like CSV writer
        for idx in range(len(t)):
            csv_data.append([
                fmt(t[idx]),            # Continuous Time
                f"{pf[idx]:.2f}",       # Continuous Pressure
                fmt(csv_flow_fv[idx]),  # Sparse FV Flow
                fmt(csv_vol_fv[idx]),   # Sparse FV Volume
                fmt(csv_vol_vt[idx])    # Sparse VT Volume (Exhale only)
            ])

        logger.info(f"  [CSV Collection] {len(csv_data)} rows collected")
        
        if csv_data and _SEND_ASYNC is not None:
            loop = asyncio.get_running_loop()
            data_rows = csv_data[1:]

            # Time-Pressure (columns 0,1)
            t = np.array([float(r[0]) for r in data_rows], dtype=float)
            p = np.array([float(r[1]) for r in data_rows], dtype=float)

            # FIXED: Extract from FILTERED pairs only
            valid_rows = [r for r in data_rows if r[2] != "" and r[3] != ""]
            if valid_rows:
                flow_fv  = np.array([float(r[2]) for r in valid_rows])  # FV Flow (col 2)
                vol_fv   = np.array([float(r[3]) for r in valid_rows])  # FV Volume (col 3)
                #vt       = np.array([float(r[4]) for r in valid_rows])  # VT Volume (col 4)
                vt_vals = [float(r[4]) for r in data_rows if r[4] != ""]
                vt = max(vt_vals) if vt_vals else 0.0            

                payload = {
                    "x": flow_fv.tolist(),
                    "y": vol_fv.tolist(),
                    "z": [vt]
                }

                try:
                    loop.create_task(_SEND_ASYNC(
                    "dataFromLib~108~spiroForcedFinalFlowVolume~" + json.dumps(payload, separators=(",", ":")))
                    )
                except Exception as e:
                    logger.info("[encoder] ERROR sending x/y/z:", e)
                          
            # Time–Volume (FULL CURVE)
            # -------------------------------
            vt_series = [float(r[4]) for r in data_rows if r[4] != ""]

            time_series = t[:len(vt_series)].tolist()

            payload = {
                "x": time_series,
                "y": vt_series
            }

            try:
                loop.create_task(_SEND_ASYNC(
                    "dataFromLib~108~spiroForcedFinalTimeVolume~" +
                    json.dumps(payload, separators=(",", ":"))
                ))
            except Exception as e:
                logger.info("[encoder] ERROR sending x / y:", e)
            

    if not table_data:
        logger.info("No valid spirometry data processed.")
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
        "FEV1 (L)": final_FEV1, "FVC (L)": final_FVC, "FEV1/FVC (%)": final_ratio,
        "PEF (L/s)": FLOW_best.get("PEF", np.nan), "FEF25-75": FLOW_best.get("FEF25_75", np.nan),
        "FEF75 (L/s)": FLOW_best.get("FEF75", np.nan), "FET (s)": FLOW_best.get("FET", np.nan),
        "TLC (L)": FLOW_best.get("TLC", np.nan), "RV (L)": FLOW_best.get("RV", np.nan),
        "VC (L)": FLOW_best.get("VC", np.nan), "BEV (L)": FLOW_best.get("BEV", np.nan),
        "FEF25 (L/s)": FLOW_best.get("FEF25", np.nan), "FEF50 (L/s)": FLOW_best.get("FEF50", np.nan),
        "FIVC (L)": FLOW_best.get("FIVC", np.nan), "PIF (L/s)": FLOW_best.get("PIF", np.nan),
    }

    # Send results (same pattern as your streaming data)
    if _SEND_ASYNC is not None:
        # Clean NaN values for JSON
        final_vals_clean = {k: (None if np.isnan(v) else v) for k, v in final_vals.items()}
        
        payload = json.dumps(final_vals_clean, separators=(',', ':'))
        
        # Use your existing async dispatch (thread-safe)
        if 'loop' in locals():
            loop.create_task(_SEND_ASYNC(f"dataFromLib~108~spiroForcedResults~{payload}"))
        else:
            loop = asyncio.get_event_loop()
            loop.create_task(_SEND_ASYNC(f"dataFromLib~108~spiroForcedResults~{payload}"))
        
        logger.info("[SENT] Spirometry results:", final_vals_clean)
    else:
        logger.info("ERROR: _SEND_ASYNC not injected")