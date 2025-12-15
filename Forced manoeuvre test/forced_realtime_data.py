from __future__ import annotations
import asyncio
import websockets
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from typing import List, Iterable, Tuple
import re
import os, time, importlib.util
from datetime import datetime
import winsound
import sys
import json

last_data_time = None
end_of_test_reported = False
IDLE_TIMEOUT_SEC = 1.0  # seconds to wait before auto-close

# near other globals / top of file
SESSION_MAX_SEC = 20   # stop accepting data after this many seconds (from first sample)
client_should_stop = False   # set True to request websocket stop from the UI thread
ws_conn = None                # will hold websocket object reference (best-effort)
session_end_time = None       # computed once start_time is known

uri = "ws://localhost:8444/bleWS/"

# Shared queue for thread-safe communication between websocket and plot
pressure_queue = queue.Queue()

# Rolling buffer for the most recent pressure values
pressures_pa = deque(maxlen=6000)
baseline_pressure = 0

# --- NEW: JSON LOADER FUNCTION ---
def load_coeffs(filename):
    # Determines the path relative to THIS script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "models", filename)
    
    if not os.path.exists(file_path):
        print(f"\nCRITICAL ERROR: Could not find '{filename}' in 'models' folder.")
        print(f"Path searched: {file_path}")
        sys.exit(1)
        
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"[Realtime] Loaded {filename}")
        return np.array(data["coeffs"], dtype=float)
    except Exception as e:
        print(f"Error reading JSON {filename}: {e}")
        sys.exit(1)

# --- REPLACED HARDCODED VALUES WITH LOADER ---
print("--- Loading Coefficients ---")
pull_coefficients = load_coeffs("coeffs_pull.json")
push_coefficients = load_coeffs("coeffs_push.json")

# --- NEW: knobs ---
USE_DEADBAND = True
USE_STATE_MACHINE = True

# ==== Flow-Volume activation settings ====
DEAD_BAND_FLOW = 0.06   # L/s below this = idle
PRESS_THR_PA   = 15.0   # Pa threshold to trigger plotting
START_HOLD     = 3      # consecutive samples above threshold to start
END_HOLD       = 10      # consecutive samples below threshold to stop

# ====== State holders ======
active = False          # current state: True = ACTIVE, False = IDLE
start_cnt = 0
end_cnt = 0
V_state = 0.0
flow_last = 0.0
vol_last = 0.0

# per-maneuver buffers
curr_v = []
curr_f = []

# process all new samples each frame
last_pf_len = 0

# grace to bridge zero-crossing so inhale+exhale stay one curve
GRACE_SAMPLES = 40   # ~0.2 s at 200 Hz
below_cnt = 0

# Volume–Time state (independent of FV)
# ----- Volume–Time rolling history (no session resets) -----
VT_PRESS_THR   = 15.0          # use your PRESS_THR_PA or set here
VT_START_HOLD  = 3             # to enter active
VT_END_HOLD    = 10            # to leave active (prevents chatter)
vt_active      = False
vt_above_cnt   = 0
vt_quiet_cnt   = 0

vt_V = 0.0                     # running volume
vt_t_hist: list[float]  = []   # all times (rolling)
vt_vol_hist: list[float] = []  # all volumes (rolling)
vt_last_pf_len = 0             # processed length

DT = 1/200                      
# ==== Exhale timer (visual cue) ====
EXHALE_START_Pressure = 50       # L/s to *start* timer (exhale begins)
EXHALE_END_Pressure   = -50        # L/s below this we *consider* end-phase
EXHALE_END_HOLD_SAMPLES  = 50       # consecutive -ve pressure for end
EXHALE_START_HOLD_SAMPLES = 10      # consecutive +ve pressure for start
MIN_EXHALE_TIME_BEFORE_STOP = 1.0  # s: don't end timer before 1 second

exhale_timer_running     = False
exhale_timer_done        = False
exhale_end_hold_count    = 0 # counts negative pressure samples (for end)
exhale_start_walltime    = None
exhale_last_duration_sec = 0.0
exhale_start_press_count = 0 # counts positive pressure samples (for start)

def play_start_beep():
    winsound.Beep(1400, 250)  # short beep

def play_long_beep():
    winsound.Beep(2000, 1400)  # long beep

# ---- audio state for cues ----
SHORT_BEEP_INTERVAL = 1.0      # seconds between short beeps during exhale

beep_start_played      = False
long_beep_played       = False
seven_sec_beep_played   = False

# 🧮 Pressure conversion helper
import re

# configuration (tweak these)
INITIAL_WINDOW_FRAMES = 10   # only apply special skipping for the first N frames
MAX_SPECIAL_TO_SEPARATE = 5  # collect/skip at most M special frames at the start

def is_special_frame_bytes(b):
    """
    Heuristic to detect your Frame A vs Frame B.
    Observations from your examples:
      - header bytes [0..5] are common
      - Frame A has b[6] == 0x00 while Frame B had b[6] == 0x01
      - Frame A payload contains many zero bytes shortly after header in the sample region
    Heuristic used here:
      - require length >= 120 (your header guard)
      - b[6] == 0x00 AND the number of zero bytes in the payload region (7..106) is high
    This is intentionally conservative; adjust thresholds if you see false positives/negatives.
    """
    if len(b) < 120:
        return False
    # quick discriminant seen in examples
    if b[6] != 0x00:
        return False
    # count zeros in the main payload area (exclude header/trailer)
    payload = b[7:107]
    zero_count = sum(1 for x in payload if x == 0)
    # threshold: >60% zeros in payload (you can lower or raise this)
    return zero_count >= int(0.60 * len(payload))

import re

def decode_pressure_from_message(message):
    """
    Decodes pressure. 
    Includes a 'Startup Gate' that nukes the first 10 'Frame A' types
    (identified by byte[6] == 0x00).
    """
    # 1. Regex Parse
    m = re.search(r'\[([0-9A-Fa-f:]+)\]', message)
    if not m:
        return []

    try:
        b = [int(t, 16) & 0xFF for t in m.group(1).split(':')]
    except ValueError:
        return []

    # 2. Basic Sanity/Header Check
    # Must start with S (83), H (72) and end with F (70)
    # Frame A and B BOTH pass this, so we check this first to filter total garbage.
    if len(b) <= 119 or b[0] != 83 or b[1] != 72 or b[119] != 70:
        return []

    #3. STARTUP GATE
    # Initialize a static counter if it doesn't exist
    if not hasattr(decode_pressure_from_message, "_frame_a_skip_count"):
        decode_pressure_from_message._frame_a_skip_count = 0

    # DIFFERENTIATOR: Check byte 6-13. 
    # Frame A has 0x00. Frame B has 0x01.
    grabage_frame = (b[i] == 0 for i in range(7,14)) 

    # Logic: If it's Frame A, AND we haven't skipped 10 of them yet... YEET IT.
    if grabage_frame and decode_pressure_from_message._frame_a_skip_count < 5:
        decode_pressure_from_message._frame_a_skip_count += 1
        print(f"DEBUG: Ignored Frame #{decode_pressure_from_message._frame_a_skip_count}")
        return []

    # --- 4. DECODE ---
    # ADC/pressure conversion
    OS_dig = 2**23
    FSS_inH2O = 120.0
    INH2O_TO_PA = 249.089

    out = []
    i = 7  # skip header
    while i < 107 and (i + 4) < len(b):
        b2 = b[i+2] & 0xFF
        b1 = b[i+3] & 0xFF
        b0 = b[i+4] & 0xFF
        decimal_count = (b2 << 16) | (b1 << 8) | b0
        
        # Avoid division by zero if calc gets weird, though unlikely with these constants
        p_inH2O = 1.25 * ((decimal_count - OS_dig) / (2**24)) * FSS_inH2O
        out.append(p_inH2O * INH2O_TO_PA)
        i += 5

    return out

# small helper to inspect stored initial special frames later
def get_initial_special_frames():
    s = getattr(decode_pressure_from_message, "_state", None)
    if not s:
        return []
    return s["initial_special_frames"]


# 🔌 WebSocket listener running in a background thread
async def ws_listener():
    global ws_conn, client_should_stop
    try:
        async with websockets.connect(uri) as websocket:
            ws_conn = websocket
            print("WebSocket connection established.")
            await websocket.send("BleAppletInit")
            print("Sent: BleAppletInit")
            await asyncio.sleep(1)
            await websocket.send("startScanFromHtml~60")
            print("Sent: startScanFromHtml~60")

            with open(LOG_FILE, "a", buffering=1, encoding="utf-8") as f:
                async for message in websocket:
                    # check if main thread asked us to stop the session
                    if client_should_stop:
                        try:
                            # polite stop command (vendor specific — harmless if unsupported)
                            await websocket.send("stopScanFromHtml")
                        except Exception:
                            pass
                        print("[ws_listener] client requested stop — closing ws.")
                        break

                    # log everything exactly as received
                    f.write(message.strip() + "\n")

                    if message.startswith("spirodata~"):
                        data = decode_pressure_from_message(message)
                        if data:
                            for v in data:
                                pressure_queue.put(v)

    except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
        print(f"WebSocket connection failed: {e}")
    except Exception as e:
        print(f"An error occurred in the WebSocket listener: {e}")
    finally:
        ws_conn = None

LOG_FILE = r"d:\Users\Tejaswini\Desktop\neurosyn\live plotting\New method\realtime_all\trial.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def start_ws_thread():
    asyncio.run(ws_listener())

# Start websocket listener in background thread
thread = threading.Thread(target=start_ws_thread, daemon=True)
thread.start()

# ------------ Filtering + baseline (ONLY what you already use) ------------
TRI_WINDOW = 20                 # triangular half-window; FIR length = 2*TRI_WINDOW - 1 (>=2)
INIT_MEAN_N  = 200

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

def basis_from_filtered(pf: np.ndarray) -> np.ndarray:
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

# Per-sample model choice (no segments): push if pf>=0 else pull
def pressure_to_flow_per_sample(pf: np.ndarray) -> np.ndarray:
    Phi  = basis_from_filtered(pf)
    push = Phi @ push_coefficients
    pull = Phi @ pull_coefficients
    mask = (pf >= 0)
    return np.where(mask, push, pull)

def integrate_flow_to_volume(flow: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(flow) * dt

W_TRI = triangular_weights(TRI_WINDOW)

# Setup plots
#FIGURE + AXES LAYOUT 
fig = plt.figure(figsize=(11.8, 5.6))

# 4 columns: 3 plots + 1 narrow timer panel
gs = fig.add_gridspec(1, 4, width_ratios=[3, 3, 3, 1], wspace=0.35)

ax_p   = fig.add_subplot(gs[0, 0])  # Pressure–Time
ax_fv  = fig.add_subplot(gs[0, 1])  # Volume–Flow
ax_vt  = fig.add_subplot(gs[0, 2])  # Volume–Time
ax_tim = fig.add_subplot(gs[0, 3])  # Timer panel

# Timer panel styling (no ticks, just a box)
ax_tim.set_xticks([])
ax_tim.set_yticks([])
ax_tim.set_facecolor("#FCFCFC")
for spine in ax_tim.spines.values():
    spine.set_visible(True)

ax_tim.set_title("Timer", fontsize=11)

timer_text = ax_tim.text(
    0.5, 0.5, "0 s",
    ha="center",
    va="center",
    fontsize=16,
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5", edgecolor="black")
)

line_p,  = ax_p.plot([], [], lw=1)
line_fv, = ax_fv.plot([], [], lw=1.1, linestyle='-')  # add marker
line_vt, = ax_vt.plot([], [], lw=1.1, color='tab:orange')  # placeholder for VT plot
prev_active = False  # NEW: to detect transitions


# Left axes config (set limits *here*, as requested)
ax_p.set_xlabel("Time (s)")
ax_p.set_ylabel("Pressure (Pa)")
ax_p.set_title("Real-time Pressure (Pa)")
ax_p.grid(True, alpha=0.3)
ax_p.set_xlim(0, 20)
ax_p.set_ylim(-5000, 5000)

# Right axes config (Volume–Flow loop) — limits set directly here
ax_fv.set_xlabel("Volume (L)")
ax_fv.set_ylabel("Flow (L/s)")
ax_fv.set_title("Real-time Volume–Flow Loop")
ax_fv.grid(True, alpha=0.3)
ax_fv.set_xlim(-6, 6.0)
ax_fv.set_ylim(-12.0, 12.0)

# VT plot config (placeholder)
ax_vt.set_xlabel("Time (s)")
ax_vt.set_ylabel("Volume (L)")
ax_vt.set_title("Real-time Volume–Time")
ax_vt.grid(True, alpha=0.3)
ax_vt.set_xlim(0, 20)
ax_vt.set_ylim(-6, 6.0)

start_time = None      # will be set when the first sample arrives
MAX_RUNTIME = 20      # seconds to auto-stop the realtime plot

# 🔁 Animation update
def update(_):
    global last_data_time, end_of_test_reported, start_time, session_end_time, client_should_stop
    WINDOW_SEC = 26
    # ingest new pressure samples...
    got_new = False
    while True:
        try:
            val = pressure_queue.get_nowait()
        except queue.Empty:
            break
        pressures_pa.append(val)
        got_new = True

    if start_time is None and got_new:
        start_time = time.time()
        session_end_time = start_time + SESSION_MAX_SEC
        print(f"[Realtime] Session started. Will stop accepting data at {session_end_time} (epoch).")

    if not pressures_pa:
        return line_p, line_fv

    # raw rolling data and time
    y_raw = np.fromiter(pressures_pa, dtype=float)
    t = np.arange(y_raw.size, dtype=float) * DT

    # baseline and FIR
    y0, _ = remove_initial_mean(y_raw, INIT_MEAN_N)
    pf = streaming_fir(y0, W_TRI)
    if pf.size == 0:
        return line_p, line_fv

    # align time to filtered length
    t_f = t[-pf.size:]

    # --- LEFT: Pressure–Time
    line_p.set_data(t_f, pf)
    ax_p.set_xlim(0, 20)
    ax_p.set_title(f"Real-time Pressure (Pa) — samples: {len(pressures_pa)}")

    # --- auto-close once data stops coming ---
    global last_data_time, end_of_test_reported
    now = time.time()

    # mark time whenever new data arrives
    if got_new:
        last_data_time = now

    # if no new samples for >1 s, close the plot
    if (not got_new) and last_data_time and (now - last_data_time) > 1.0 and not end_of_test_reported:
        end_of_test_reported = True
        print("[Realtime] No new data — closing plot and running analysis…")

        # schedule the figure to close so plt.show() returns cleanly
                # stop the animation and close safely using a Python timer
        try:
            if 'ani' in globals() and hasattr(ani, 'event_source'):
                ani.event_source.stop()
        except Exception:
            pass

        def _safe_close():
            try:
                plt.close(fig)
            except Exception:
                pass

        threading.Timer(0.1, _safe_close).start()

        return line_p, line_fv, line_vt


    # --- RIGHT: Volume–Flow loop ---
    # Convert filtered pressure -> flow (per-sample push/pull), integrate -> volume
    global active, start_cnt, end_cnt, V_state, flow_last, vol_last
    global curr_v, curr_f, last_pf_len, below_cnt
    global exhale_timer_running, exhale_timer_done
    global exhale_start_walltime, exhale_last_duration_sec
    global exhale_start_press_count, exhale_end_hold_count
    global beep_start_played, long_beep_played, seven_sec_beep_played

    flow_all = pressure_to_flow_per_sample(pf)

    # process only the new samples since last frame
    start_idx = last_pf_len
    if start_idx < 0 or start_idx > pf.size:
        start_idx = 0

    for i in range(start_idx, pf.size):
        p_now = pf[i]
        f_now = flow_all[i]

        # ===== Exhale timer logic (pressure-based) =====
        # START: sustained positive pressure = exhale
        if not exhale_timer_running and not exhale_timer_done:
            if p_now >= EXHALE_START_Pressure:
                exhale_start_press_count += 1
                if exhale_start_press_count >= EXHALE_START_HOLD_SAMPLES:
                    exhale_timer_running     = True
                    exhale_start_walltime    = time.time()
                    exhale_last_duration_sec = 0.0
                    exhale_end_hold_count    = 0

                    # reset audio state for this maneuver
                    beep_start_played   = False
                    long_beep_played    = False
                    seven_sec_beep_played = False

                    # START CUE: play start beep once (non-blocking)(only once per exhale)
                    if not beep_start_played:
                        threading.Thread(target=play_start_beep, daemon=True).start()
                        beep_start_played = True

            else:
                # broke the positive run
                exhale_start_press_count = 0

        #END: after min exhale time, sustained negative pressure = inhale
        elif exhale_timer_running:
            now     = time.time()
            elapsed = now - exhale_start_walltime if exhale_start_walltime is not None else 0.0

            if elapsed >= MIN_EXHALE_TIME_BEFORE_STOP:
                if p_now <= EXHALE_END_Pressure:
                    exhale_end_hold_count += 1
                    if exhale_end_hold_count >= EXHALE_END_HOLD_SAMPLES:
                        exhale_timer_running     = False
                        exhale_timer_done        = True
                        exhale_last_duration_sec = elapsed
                else:
                    # not consistently inhale → reset end counter
                    exhale_end_hold_count = 0
            else:
                # during first 1s, don't let anything stop the timer
                exhale_end_hold_count = 0

        # ===== Existing Volume–Flow state machine =====
        if active:
            if abs(p_now) < PRESS_THR_PA:
                below_cnt += 1
            else:
                below_cnt = 0

            if below_cnt >= max(END_HOLD, GRACE_SAMPLES):
                active = False
                below_cnt = 0
                # do NOT clear buffers here
            else:
                V_state += f_now * DT
                flow_last = f_now
                vol_last  = V_state
                curr_v.append(vol_last)
                curr_f.append(flow_last)
        else:
            if abs(p_now) >= PRESS_THR_PA:
                start_cnt += 1
                if start_cnt >= START_HOLD:
                    active = True
                    start_cnt = 0
                    # reset timer per maneuver
                    exhale_timer_running     = False
                    exhale_timer_done        = False
                    exhale_start_walltime    = None
                    exhale_last_duration_sec = 0.0
                    exhale_end_hold_count    = 0
                    exhale_start_press_count = 0

                    # reset audio per maneuver
                    beep_start_played   = False
                    long_beep_played    = False
            else:
                start_cnt = 0


    # remember how far we processed
    last_pf_len = pf.size

    # draw ALL accumulated points (not just one)
    if curr_v:
        line_fv.set_data(np.asarray(curr_v), np.asarray(curr_f))
    else:
        line_fv.set_data([], [])

    #CUMULATIVE VOLUME–TIME PLOT UPDATE
    global vt_active, vt_above_cnt, vt_quiet_cnt, vt_V, vt_t_hist, vt_vol_hist, vt_last_pf_len

    i0 = vt_last_pf_len
    if i0 < 0 or i0 > pf.size:
        i0 = 0

    for i in range(i0, pf.size):
        p  = pf[i]
        f  = flow_all[i]
        ti = t_f[i]

        # hysteresis state (no buffer resets!)
        if vt_active:
            if abs(p) >= VT_PRESS_THR:
                vt_quiet_cnt = 0
                # integrate only when above threshold
                vt_V += -(f * DT)
            else:
                vt_quiet_cnt += 1
                # hold vt_V (no integration) while below threshold
                if vt_quiet_cnt >= VT_END_HOLD:
                    vt_active = False
                    vt_quiet_cnt = 0
            # append a point every sample so time advances flat while holding
            vt_t_hist.append(ti)
            vt_vol_hist.append(vt_V)

        else:
            # currently idle → look for start
            if abs(p) >= VT_PRESS_THR:
                vt_above_cnt += 1
                if vt_above_cnt >= VT_START_HOLD:
                    vt_active = True
                    vt_above_cnt = 0
                    # DO NOT reset vt_V or history here — we want all peaks to stay
                    # immediate integration for this sample since we just turned active
                    vt_V += f * DT
            else:
                vt_above_cnt = 0
            # append even while idle so plot shows flat zero (or last value)
            vt_t_hist.append(ti)
            vt_vol_hist.append(vt_V)

    vt_last_pf_len = pf.size

    # draw last 26 s of the rolling history
    win = int(26 / DT)
    if len(vt_t_hist) > 0:
        x = np.asarray(vt_t_hist[-win:])
        y = np.asarray(vt_vol_hist[-win:])
        line_vt.set_data(x, y)
    else:
        line_vt.set_data([], [])


        # --- session timeout handling ---
    if session_end_time is not None and not end_of_test_reported:
        now = time.time()
        if now >= session_end_time:
            end_of_test_reported = True
            print(f"[Realtime] Session time reached {SESSION_MAX_SEC}s — stopping data collection and closing plot.")
            client_should_stop = True

            try:
                if 'ani' in globals() and hasattr(ani, 'event_source'):
                    ani.event_source.stop()
            except Exception:
                pass

            def _safe_close():
                try:
                    plt.close(fig)
                except Exception:
                    pass

            threading.Timer(0.1, _safe_close).start()

    # ===== Update Timer UI (EXHALE TIMER ONLY) =====
    if exhale_timer_running and exhale_start_walltime is not None:
        elapsed_exhale = time.time() - exhale_start_walltime
        exhale_last_duration_sec = elapsed_exhale
        timer_text.set_text(f"{int(elapsed_exhale)} s")

        # === 7-second long beep ===
        if elapsed_exhale >= 7.0 and not seven_sec_beep_played:
            seven_sec_beep_played = True
            threading.Thread(target=play_long_beep, daemon=True).start()


    elif exhale_timer_done:
        # exhale finished → freeze at final duration
        timer_text.set_text(f"{int(exhale_last_duration_sec)} s")

        # --- FINAL LONG BEEP (once per maneuver) ---
        if not long_beep_played:
            threading.Thread(target=play_long_beep, daemon=True).start()
            long_beep_played = True

    else:
        # no exhale yet → show 0 s
        timer_text.set_text("0 s")

    return line_p, line_fv, line_vt

ani = FuncAnimation(fig, update, interval=50, blit=False)
plt.tight_layout()
plt.show()

# === run analysis after the realtime plot window is closed ===
try:
    import os, time, importlib.util
    from datetime import datetime

    # tiny flush cushion
    time.sleep(0.5)

    # load forced_calculations.py from same folder (exec as module)
    here = os.path.abspath(os.path.dirname(__file__))
    analysis_path = os.path.join(here, "forced_calculations_2plots_json.py") 
    spec = importlib.util.spec_from_file_location("forced_calculations_2plots_json", analysis_path)
    fa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fa)

    # route the same log into analysis
    fa.FILE = LOG_FILE  # <- reuse realtime log
    # timestamped output dir beside the log
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fa.OUTDIR = os.path.join(os.path.dirname(LOG_FILE), f"plots_{ts}")
    fa.SAVE_FIGS = True

    print(f"\n[fetch→analysis] Using log: {fa.FILE}")
    print(f"[fetch→analysis] Output dir: {fa.OUTDIR}")

    # run full analysis (prints metrics + shows plots)
    fa.main()
    print("[fetch→analysis] Analysis complete.")

except Exception as e:
    print(f"[fetch→analysis] Failed to run analysis: {e}")
