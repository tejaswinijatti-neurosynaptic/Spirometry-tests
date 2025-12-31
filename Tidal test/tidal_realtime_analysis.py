from __future__ import annotations
import asyncio
import websockets
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time
from typing import List, Iterable, Tuple
import os
import re
from datetime import datetime, timezone
from pathlib import Path
import json

# Sampling period (seconds). 0.005 = 200 Hz
DT = 0.005
TEST_DURATION = 20.0  # < NEW: Auto-stop after 20 seconds

#  GLOBAL STATE 
last_data_time = None
start_data_time = None  #Tracks when the first sample arrived
end_of_test_reported = False
IDLE_TIMEOUT_SEC = 1.0

uri = "ws://localhost:8444/bleWS/"

# Shared queue for thread-safe communication between websocket and plot
pressure_queue = queue.Queue()

# Rolling buffer for the most recent pressure values
pressures_pa = deque(maxlen=6000)
baseline_pressure = 0


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

    #  Frame-level skip logic (kept inside the function) 
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
    OS = 2**23
    FSS_inH2O = 120
    INH2O_TO_PA = 249.089

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

                    # optional: your existing logic continues
                    if message.startswith("spirodata~"):
                        print(f"Received: {message}")
                        data = decode_pressure_from_message(message)
                        if data:
                            for v in data:
                                pressure_queue.put(v)

    except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
        print(f"WebSocket connection failed: {e}")
    except Exception as e:
        print(f"An error occurred in the WebSocket listener: {e}")

LOG_FILE = r"d:\Users\Tejaswini\Desktop\neurosyn\live plotting\New method\realtime_all\tidal_test.log"  # ← change path if you want
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def start_ws_thread():
    asyncio.run(ws_listener())

# Start websocket listener in background thread
thread = threading.Thread(target=start_ws_thread, daemon=True)
thread.start()

# 🎨 Setup plot
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=1)
ax.set_xlabel("Time(s)")
ax.set_ylabel("Pressure (Pa)")
ax.set_title("Real-time Pressure (Pa)")
ax.grid(True, alpha=0.3)

# Fix the axis ranges here:
YMIN, YMAX = -5000, 5000
ax.set_xlim(0, 20)
ax.set_ylim(YMIN, YMAX)

# Sampling period (seconds). 0.005 = 200 Hz
DT = 0.005

# Segmentation params
DEADBAND     = 100   # min segment length (samples)
TAIL_IGNORE  = 150   # ignore last N samples when segmenting

# Filtering + baseline params
TRI_WINDOW   = 20    # triangular half-window; FIR length = 2*TRI_WINDOW - 1 (>=2)
INIT_MEAN_N  = 150   # samples for initial mean removal
END_MEAN_N   = 150   # samples used to estimate linear drift toward tail


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

last_smoothed = []
end_of_test_reported = False
last_data_time = None
# If no new samples arrive for this many seconds, treat as end-of-test
IDLE_TIMEOUT_SEC = 1.0
# Require at least this many samples before reporting TV on idle
MIN_SAMPLES_FOR_TV = 100

# add once near your other globals (after triangular_weights is defined)
W_TRI = triangular_weights(TRI_WINDOW)

# 🔁 Animation update function
def update(_):
    global last_data_time, end_of_test_reported, start_data_time
    WINDOW_SEC = 26 
    
    # Get new decoded pressures from queue
    got_new = False
    now = time.time()
    
    while True:
        try:
            val = pressure_queue.get_nowait()
        except queue.Empty:
            break
        
        pressures_pa.append(val)
        got_new = True
        last_data_time = now
        
        # Start the 20s timer when the FIRST packet arrives
        if start_data_time is None:
            start_data_time = now
            print(f"[Realtime] Data flow started. Timer set for {TEST_DURATION} seconds.")

    if not pressures_pa:
        return line,

    # ... [Keep existing filtering/plotting logic: y_raw, y0, yf, line.set_data] ...
    y_raw = np.fromiter(pressures_pa, dtype=float)
    x = np.arange(y_raw.size, dtype=float) * DT
    y0, _ = remove_initial_mean(y_raw, INIT_MEAN_N)
    yf = streaming_fir(y0, W_TRI)
    if yf.size == 0: return line,
    x_f = x[-yf.size:]
    line.set_data(x_f, yf)
    
    # Dynamic sliding window
    if x_f[-1] > WINDOW_SEC:
        ax.set_xlim(x_f[-1] - WINDOW_SEC, x_f[-1])
    else:
        ax.set_xlim(0, WINDOW_SEC)
        
    # Update Title with Timer Countdown
    if start_data_time:
        elapsed = now - start_data_time
        remaining = max(0, TEST_DURATION - elapsed)
        ax.set_title(f"Real-time Pressure (Pa) — Time Remaining: {remaining:.1f}s")
    else:
        ax.set_title("Real-time Pressure (Pa) — Waiting for data...")

    # --- AUTO-CLOSE LOGIC ---
    should_close = False
    close_reason = ""

    # Condition A: Idle Timeout (No data for 1s)
    if (not got_new) and last_data_time and (now - last_data_time) > IDLE_TIMEOUT_SEC:
        should_close = True
        close_reason = "Idle timeout (no data)"

    # Condition B: 20-Second Timer Limit (NEW)
    if start_data_time and (now - start_data_time) > TEST_DURATION:
        should_close = True
        close_reason = f"Test duration limit ({TEST_DURATION}s) reached"

    if should_close and not end_of_test_reported:
        end_of_test_reported = True
        print(f"[Realtime] {close_reason} — closing plot and running analysis…")
        timer = fig.canvas.new_timer(interval=100)
        timer.add_callback(lambda: plt.close(fig))
        timer.start()

    return line,

# 🎬 Start animation (updates every 50 ms)
ani = FuncAnimation(fig, update, interval=50, blit=False)

plt.show()

# === run tidal calculations after realtime window closes ===
try:
    import importlib.util
    from datetime import datetime
    import time

    # tiny flush so file writer has time to finish
    time.sleep(0.5)

    here = os.path.abspath(os.path.dirname(__file__))
    tv_path = os.path.join(here, "TV_calculations.py")
    spec = importlib.util.spec_from_file_location("TV_calculations", tv_path)
    tv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tv)

    # If TV_calculations has FILE variable, make sure it uses our LOG_FILE
    try:
        tv.FILE = LOG_FILE
    except Exception:
        pass

    print(f"[tidal→TV] Running TV_calculations on: {LOG_FILE}")
    # TV_calculations.run_one expects a file path
    if hasattr(tv, "run_one"):
        tv.run_one(LOG_FILE)
    else:
        # fallback: if module exposes a different API, try calling main-like entry
        if hasattr(tv, "main"):
            tv.main()
        else:
            print("[tidal→TV] TV_calculations: no run_one() or main() function found.")

except Exception as e:
    print(f"[tidal→TV] Failed to run TV_calculations: {e}")
