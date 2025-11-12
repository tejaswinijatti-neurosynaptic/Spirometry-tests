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
import os
import re
import os, time, importlib.util
from datetime import datetime

last_data_time = None
end_of_test_reported = False
IDLE_TIMEOUT_SEC = 1.0  # seconds to wait before auto-close

uri = "ws://localhost:8444/bleWS/"

# Shared queue for thread-safe communication between websocket and plot
pressure_queue = queue.Queue()

# Rolling buffer for the most recent pressure values
pressures_pa = deque(maxlen=6000)
baseline_pressure = 0

# Coefficients (device f039)
# 4-term basis coefficients (pull=inhale, push=exhale)
pull_coefficients = np.array([ 0.334646, -0.001808, -0.526989,  0.498103], dtype=float)  # inhale/pull
push_coefficients = np.array([ 1.21753e-01,  2.10000e-05,  7.26680e-02, -1.59643e-01], dtype=float)  # exhale/push

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



# 🧮 Pressure conversion helper
def decode_pressure_from_message(message):
    """
    Parse 'spirodata~[AA:BB:...:ZZ]' into pressures (Pa).
    Frame-level gating baked in:
      - Skip the first *all-zero* frame entirely.
      - Skip the very next frame (second) as well.
      - Then decode normally.
    """
    m = re.search(r'\[([0-9A-Fa-f:]+)\]', message)
    if not m:
        return []

    # Parse bytes
    try:
        b = [int(t, 16) & 0xFF for t in m.group(1).split(':')]
    except ValueError:
        return []

    # Header guard + bounds: require S(83), H(72), F(70) at [0],[1],[119]
    if len(b) <= 119 or b[0] != 83 or b[1] != 72 or b[119] != 70:
        return []

    # --- Frame-level skip logic (kept inside the function) ---
    state = getattr(decode_pressure_from_message, "_skip_state", 0)  # 0: wait null, 1: skip next, 2: normal
    is_null = all(v == 0 for v in b)

    if state == 0:
        if is_null:
            decode_pressure_from_message._skip_state = 1  # next frame will be skipped
            return []
        # no null seen yet → pass through
    elif state == 1:
        decode_pressure_from_message._skip_state = 2  # done skipping
        return []
    # state == 2 → normal

    # ADC/pressure conversion
    OS_dig = 2**23
    FSS_inH2O = 120.0
    INH2O_TO_PA = 249.089

    out = []
    i = 7  # skip 7-byte header
    while i < 107 and (i + 4) < len(b):
        b2 = b[i+2] & 0xFF
        b1 = b[i+3] & 0xFF
        b0 = b[i+4] & 0xFF
        decimal_count = (b2 << 16) | (b1 << 8) | b0
        p_inH2O = 1.25 * ((decimal_count - OS_dig) / (2**24)) * FSS_inH2O
        out.append(p_inH2O * INH2O_TO_PA)
        i += 5

    # IMPORTANT: do NOT drop first two samples here—frames are already gated
    return out

# 🔌 WebSocket listener running in a background thread
async def ws_listener():
    try:
        async with websockets.connect(uri) as websocket:
            print("WebSocket connection established.")
            await websocket.send("BleAppletInit")
            print("Sent: BleAppletInit")
            await asyncio.sleep(1)
            await websocket.send("startScanFromHtml~60")
            print("Sent: startScanFromHtml~60")

            # open the file once, keep it open for the session
            with open(LOG_FILE, "a", buffering=1, encoding="utf-8") as f:
                async for message in websocket:
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

LOG_FILE = r"d:\Users\Tejaswini\Desktop\neurosyn\live plotting\New method\realtime_all\forced1.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def start_ws_thread():
    asyncio.run(ws_listener())

# Start websocket listener in background thread
thread = threading.Thread(target=start_ws_thread, daemon=True)
thread.start()

# ------------ Filtering + baseline (ONLY what you already use) ------------
DT = 0.005                      # 200 Hz
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
    """
    4-term basis on filtered pressure:
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

# 🎨 Setup plots: LEFT = Pressure–Time, RIGHT = Volume–Flow
fig, (ax_p, ax_fv, ax_vt) = plt.subplots(1, 3, figsize=(11.8, 5.6))
line_p,  = ax_p.plot([], [], lw=1)
line_fv, = ax_fv.plot([], [], lw=1.1, linestyle='-')  # add marker
line_vt, = ax_vt.plot([], [], lw=1.1, color='tab:orange')  # placeholder for VT plot
prev_active = False  # NEW: to detect transitions


# Left axes config (set limits *here*, as requested)
ax_p.set_xlabel("Time (s)")
ax_p.set_ylabel("Pressure (Pa)")
ax_p.set_title("Real-time Pressure (Pa)")
ax_p.grid(True, alpha=0.3)
ax_p.set_xlim(0, 26)
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
ax_vt.set_xlim(0, 26)
ax_vt.set_ylim(-6, 6.0)

# 🔁 Animation update
def update(_):
    global last_data_time, end_of_test_reported  #declare both
    WINDOW_SEC = 26  # seconds shown on x-axis
    # ingest new pressure samples
    got_new = False
    while True:
        try:
            val = pressure_queue.get_nowait()
        except queue.Empty:
            break
        pressures_pa.append(val)
        got_new = True

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

    # Sliding x-window for pressure
    if t_f[-1] > 26:
        ax_p.set_xlim(t_f[-1] - 26, t_f[-1])
    else:
        ax_p.set_xlim(0, 26)

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
        timer = fig.canvas.new_timer(interval=100)  # close in 0.1 s
        timer.add_callback(lambda: plt.close(fig))
        timer.start()

    # --- RIGHT: Volume–Flow loop ---
    # Convert filtered pressure -> flow (per-sample push/pull), integrate -> volume
    global active, start_cnt, end_cnt, V_state, flow_last, vol_last
    global curr_v, curr_f, last_pf_len, below_cnt

    flow_all = pressure_to_flow_per_sample(pf)

    # process only the new samples since last frame
    start_idx = last_pf_len
    if start_idx < 0 or start_idx > pf.size:
        start_idx = 0

    for i in range(start_idx, pf.size):
        p_now = pf[i]
        f_now = flow_all[i]

        # activation using hysteresis
        if active:
            if abs(p_now) < PRESS_THR_PA:
                below_cnt += 1
            else:
                below_cnt = 0

            # allow short dropouts (zero-crossing) without ending
            if below_cnt >= max(END_HOLD, GRACE_SAMPLES):
                active = False
                below_cnt = 0
                # do NOT clear buffers here; we keep the loop on-screen
            else:
                # integrate and append
                V_state += f_now * DT
                flow_last = f_now
                vol_last  = V_state
                curr_v.append(vol_last)
                curr_f.append(flow_last)

        else:
            # idle → look for start hold
            if abs(p_now) >= PRESS_THR_PA:
                start_cnt += 1
                if start_cnt >= START_HOLD:
                    active = True
                    start_cnt = 0
                    # starting a new maneuver: if previous loop should persist, keep buffers;
                    # if you want a fresh curve each time, uncomment the next two lines:
                    # curr_v, curr_f = [], []
                    # V_state = vol_last  # or 0.0 if you prefer absolute volume from zero
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
