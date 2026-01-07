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
import csv  # <--- Added for CSV logging

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

#  NEW: JSON LOADER FUNCTION 
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

#  REPLACED HARDCODED VALUES WITH LOADER 
print(" Loading Coefficients ")
pull_coefficients = load_coeffs("coeffs_pull.json")
push_coefficients = load_coeffs("coeffs_push.json")

#  NEW: knobs 
USE_DEADBAND = True
USE_STATE_MACHINE = True

#  Flow-Volume activation settings 
DEAD_BAND_FLOW = 0.06   # L/s below this = idle
PRESS_THR_PA   = 15.0   # Pa threshold to trigger plotting
START_HOLD     = 3      # consecutive samples above threshold to start
END_HOLD       = 10      # consecutive samples below threshold to stop

#  State holders 
active = False          # current state: True = ACTIVE, False = IDLE
first_activation = True
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
#  Volume–Time rolling history (no session resets) 
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
#  Exhale timer (visual cue) 
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

#  audio state for cues 
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
    if len(b) < 120:
        return False
    if b[6] != 0x00:
        return False
    payload = b[7:107]
    zero_count = sum(1 for x in payload if x == 0)
    return zero_count >= int(0.60 * len(payload))

import re

def decode_pressure_from_message(message):
    m = re.search(r'\[([0-9A-Fa-f:]+)\]', message)
    if not m:
        return []

    try:
        b = [int(t, 16) & 0xFF for t in m.group(1).split(':')]
    except ValueError:
        return []

    if len(b) <= 119 or b[0] != 83 or b[1] != 72 or b[119] != 70:
        return []

    if not hasattr(decode_pressure_from_message, "_frame_a_skip_count"):
        decode_pressure_from_message._frame_a_skip_count = 0

    grabage_frame = (b[i] == 0 for i in range(7,14)) 

    if grabage_frame and decode_pressure_from_message._frame_a_skip_count < 5:
        decode_pressure_from_message._frame_a_skip_count += 1
        print(f"DEBUG: Ignored Frame #{decode_pressure_from_message._frame_a_skip_count}")
        return []

    OS_dig = 2**23
    FSS_inH2O = 120.0
    INH2O_TO_PA = 249.089

    out = []
    i = 7
    while i < 107 and (i + 4) < len(b):
        b2 = b[i+2] & 0xFF
        b1 = b[i+3] & 0xFF
        b0 = b[i+4] & 0xFF
        decimal_count = (b2 << 16) | (b1 << 8) | b0
        p_inH2O = 1.25 * ((decimal_count - OS_dig) / (2**24)) * FSS_inH2O
        out.append(p_inH2O * INH2O_TO_PA)
        i += 5

    return out

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
                    if client_should_stop:
                        try:
                            await websocket.send("stopScanFromHtml")
                        except Exception:
                            pass
                        print("[ws_listener] client requested stop — closing ws.")
                        break

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

LOG_FILE = r"d:\Users\Tejaswini\Desktop\neurosyn\live plotting\New method\realtime_all\qc_check_logs\forced_trial.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

print("Time(s) | Pressure(Pa) | FV_Flow(L/s) | FV_Vol(L) | VT_Vol(L)")

def start_ws_thread():
    asyncio.run(ws_listener())

# Start websocket listener in background thread
thread = threading.Thread(target=start_ws_thread, daemon=True)
thread.start()

#  Filtering + baseline (ONLY what you already use) 
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
    global last_data_time, end_of_test_reported, start_time, session_end_time, client_should_stop, first_activation
    global csv_writer, csv_file_handle 

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

    #  LEFT: Pressure–Time
    line_p.set_data(t_f, pf)
    ax_p.set_xlim(0, 20)
    ax_p.set_title(f"Real-time Pressure (Pa) — samples: {len(pressures_pa)}")

    #  auto-close once data stops coming 
    global last_data_time, end_of_test_reported
    now = time.time()

    # mark time whenever new data arrives
    if got_new:
        last_data_time = now

    if (not got_new) and last_data_time and (now - last_data_time) > 1.0 and not end_of_test_reported:
        end_of_test_reported = True
        print("[Realtime] No new data — closing plot and running analysis…")
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


    #  RIGHT: Volume–Flow loop 
    global active, start_cnt, end_cnt, V_state, flow_last, vol_last
    global curr_v, curr_f, last_pf_len, below_cnt
    global exhale_timer_running, exhale_timer_done
    global exhale_start_walltime, exhale_last_duration_sec
    global exhale_start_press_count, exhale_end_hold_count
    global beep_start_played, long_beep_played, seven_sec_beep_played

    flow_all = pressure_to_flow_per_sample(pf)
    if USE_DEADBAND:
        flow_all = np.where(np.abs(flow_all) < DEAD_BAND_FLOW, 0.0, flow_all)

    # process only the new samples since last frame
    start_idx = last_pf_len
    if start_idx < 0 or start_idx > pf.size:
        start_idx = 0

    # Buffer to hold Flow-Volume data specifically for logging
    # We want to know exactly what the loop decided for each sample in this chunk
    chunk_fv_data = [] # Stores tuple (Flow_Val, Vol_Val) or (None, None)

    for i in range(start_idx, pf.size):
        p_now = pf[i]
        f_now = flow_all[i]

        #  Exhale timer logic (pressure-based) 
        if not exhale_timer_running and not exhale_timer_done:
            if p_now >= EXHALE_START_Pressure:
                exhale_start_press_count += 1
                if exhale_start_press_count >= EXHALE_START_HOLD_SAMPLES:
                    exhale_timer_running     = True
                    
                    #NEW: PRINT TIMER START
                    print(f"\n[EVENT] Timer STARTED (Sustained Pressure > {EXHALE_START_Pressure})")

                    exhale_start_walltime    = time.time()
                    exhale_last_duration_sec = 0.0
                    exhale_end_hold_count    = 0
                    beep_start_played   = False
                    long_beep_played    = False
                    seven_sec_beep_played = False
                    
                    if not beep_start_played:
                        threading.Thread(target=play_start_beep, daemon=True).start()
                        beep_start_played = True
                        # NEW: PRINT FIRST BEEP
                        print("[EVENT]First Beep Triggered (Start)")

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
                        
                        # NEW: PRINT TIMER STOP
                        print(f"[EVENT] Timer STOPPED (Duration: {elapsed:.2f}s)")
                else:
                    exhale_end_hold_count = 0
            else:
                exhale_end_hold_count = 0

        #  Volume–Flow state machine 
        should_record_fv = False # Flag for logging

        if active:
            if abs(p_now) < PRESS_THR_PA:
                below_cnt += 1
            else:
                below_cnt = 0

            if below_cnt >= max(END_HOLD, GRACE_SAMPLES):
                active = False
                below_cnt = 0
            else:
                V_state += f_now * DT
                flow_last = f_now
                vol_last  = V_state
                curr_v.append(vol_last)
                curr_f.append(flow_last)
                should_record_fv = True # <- ACTIVE DATA
        else:
            if abs(p_now) >= PRESS_THR_PA:
                start_cnt += 1
                if start_cnt >= START_HOLD:
                    active = True
                    start_cnt = 0
                    if first_activation:
                        V_state = 0.0
                        curr_v.append(0.0)
                        curr_f.append(0.0)
                        first_activation = False
                    else:
                        curr_v.append(V_state) 
                        curr_f.append(0.0)
                    
                    exhale_timer_running     = False
                    exhale_timer_done        = False
                    exhale_start_walltime    = None
                    exhale_last_duration_sec = 0.0
                    exhale_end_hold_count    = 0
                    exhale_start_press_count = 0
                    beep_start_played   = False
                    long_beep_played    = False
                    
                    should_record_fv = True # <--- START TRIGGERED
            else:
                start_cnt = 0

        # Store result for this sample for CSV logging
        if should_record_fv:
            # If active, store actual values
            # (Use '0.0' for flow if we just triggered start, effectively)
            val_f = flow_last if active else 0.0 
            val_v = vol_last if active else V_state
            chunk_fv_data.append( (val_f, val_v) )
        else:
            # If inactive, store None implies empty cell in CSV
            chunk_fv_data.append( (None, None) )


    last_pf_len = pf.size

    if curr_v:
        line_fv.set_data(np.asarray(curr_v), np.asarray(curr_f))
    else:
        line_fv.set_data([], [])

    #CUMULATIVE VOLUME–TIME PLOT UPDATE + CSV WRITING
    global vt_active, vt_above_cnt, vt_quiet_cnt, vt_V, vt_t_hist, vt_vol_hist, vt_last_pf_len

    i0 = vt_last_pf_len
    if i0 < 0 or i0 > pf.size:
        i0 = 0
    
    csv_rows = []

    for i in range(i0, pf.size):
        p  = pf[i]
        f  = flow_all[i]
        ti = t_f[i]

        # VT Logic
        if vt_active:
            if abs(p) >= VT_PRESS_THR:
                vt_quiet_cnt = 0
                vt_V += -(f * DT)
            else:
                vt_quiet_cnt += 1
                if vt_quiet_cnt >= VT_END_HOLD:
                    vt_active = False
                    vt_quiet_cnt = 0
            vt_t_hist.append(ti)
            vt_vol_hist.append(vt_V)

        else:
            if abs(p) >= VT_PRESS_THR:
                vt_above_cnt += 1
                if vt_above_cnt >= VT_START_HOLD:
                    vt_active = True
                    vt_above_cnt = 0
                    vt_V += f * DT
            else:
                vt_above_cnt = 0
            vt_t_hist.append(ti)
            vt_vol_hist.append(vt_V)

        # We need the FV data from the first loop.
        # Calculate relative index:
        rel_idx = i - i0
        
        # Get FV values if they exist for this sample
        fv_f_str = ""
        fv_v_str = ""
        
        if rel_idx < len(chunk_fv_data):
            f_val, v_val = chunk_fv_data[rel_idx]
            if f_val is not None:
                fv_f_str = f"{f_val:.4f}"
                fv_v_str = f"{v_val:.4f}"
        
        # Row format: Time, Pressure, FV_Flow, FV_Volume, VT_Volume
        csv_rows.append([
            f"{ti:.4f}",
            f"{p:.2f}",
            fv_f_str,    # Only filled if active
            fv_v_str,    # Only filled if active
            f"{vt_V:.4f}" # Always filled
        ])

    vt_last_pf_len = pf.size
    
    # Print to Terminal
    if csv_rows:
        for row in csv_rows:
            # row is a list: [Time, Pressure, FV_Flow, FV_Volume, VT_Volume]
            # We join them with commas or pipes for readability
            print(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}")

    # draw last 26 s of the rolling history
    win = int(26 / DT)
    if len(vt_t_hist) > 0:
        x = np.asarray(vt_t_hist[-win:])
        y = np.asarray(vt_vol_hist[-win:])
        line_vt.set_data(x, y)
    else:
        line_vt.set_data([], [])


    #  session timeout handling 
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

    #  Update Timer UI 
    if exhale_timer_running and exhale_start_walltime is not None:
        elapsed_exhale = time.time() - exhale_start_walltime
        exhale_last_duration_sec = elapsed_exhale
        timer_text.set_text(f"{int(elapsed_exhale)} s")
        
        # 7-second beep
        if elapsed_exhale >= 7.0 and not seven_sec_beep_played:
            seven_sec_beep_played = True
            threading.Thread(target=play_long_beep, daemon=True).start()
            #  NEW: PRINT 7s BEEP 
            print("[EVENT]Second Beep Triggered (7s Mark)")

    elif exhale_timer_done:
        timer_text.set_text(f"{int(exhale_last_duration_sec)} s")
        
        # Final beep (End of maneuver)
        if not long_beep_played:
            threading.Thread(target=play_long_beep, daemon=True).start()
            long_beep_played = True
            # NEW: PRINT END BEEP
            print("[EVENT] ♫ Final Beep Triggered (Maneuver Complete)")
    else:
        timer_text.set_text("0 s")

    return line_p, line_fv, line_vt

ani = FuncAnimation(fig, update, interval=50, blit=False)
plt.tight_layout()
plt.show()

#  run analysis after the realtime plot window is closed 
try:
    import os, time, importlib.util
    from datetime import datetime

    # tiny flush cushion
    time.sleep(0.5)

    # load forced_calculations.py from same folder (exec as module)
    here = os.path.abspath(os.path.dirname(__file__))
    analysis_path = os.path.join(here, "forced_calculations.py") 
    spec = importlib.util.spec_from_file_location("forced_calculations", analysis_path)
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
