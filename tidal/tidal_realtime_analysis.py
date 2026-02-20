from __future__ import annotations
import asyncio
import importlib
import json
import websockets
import threading
import queue
import numpy as np

from logging_config import setup_logging

_SEND_ASYNC = None
DEBUGGING = False
if DEBUGGING:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
from collections import deque
import time
from typing import List, Iterable, Tuple
import re
from pathlib import Path
import json
import os

from tidal import TV_calculations
import logging

logger = logging.getLogger(__name__)

#  CONFIG 
# Sampling period (seconds). 0.005 = 200 Hz
DT = 0.005
TEST_DURATION = 20.0  # Auto-stop after 20 seconds
IDLE_TIMEOUT_SEC = 1.0

# Only one log file now (Raw Hex)
LOG_FILE = Path("realtime_all") / "tidal_test.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

#  GLOBAL STATE 
last_data_time = None
start_data_time = None 
end_of_test_reported = False

uri = "ws://localhost:8444/bleWS/"

# Shared queue
pressure_queue = queue.Queue()
all_samples_pa = deque()

# Rolling buffer for plotting
pressures_pa = deque(maxlen=6000)
baseline_pressure = 0

stop_event = threading.Event()

frame_queue = queue.Queue()

def set_sender(send_async_fn):
    """
    send_async_fn must be an async function: await send_async_fn("msgType~payload")
    """
    global _SEND_ASYNC
    _SEND_ASYNC = send_async_fn

async def clearGlobalReferences():
    global _fir_buf, _mean_buf, _mean_ready, _mean_value, _sample_counter, all_samples_pa
    
    _fir_buf.clear()
    _mean_buf.clear()
    _mean_ready = False
    _mean_value = 0.0
    _sample_counter = 0
    all_samples_pa.clear()

async def RelealTimeAnalysis(message):
        if message:     
            with open(LOG_FILE, "a", buffering=1, encoding="utf-8") as tv:
                # log everything exactly as received
                tv.write(message.strip() + "\n")   
            
            data = decode_pressure_from_message(message)
            if data and len(data) == 20:
                # collect for final TV
                all_samples_pa.extend(data)

                x, y = process_frame_xy(data)

                if x is not None and _SEND_ASYNC is not None:
                    payload = {
                        "x": x.tolist(),
                        "y": y.tolist()
                    }

                    try:
                        await _SEND_ASYNC(
                            "dataFromLib~107~" + json.dumps(payload, separators=(",", ":"))
                        )
                    except Exception as e:
                        logger.info("[encoder] ERROR sending x/y:", e)
                        

def triangular_weights(window: int) -> np.ndarray:
    up = np.arange(1, window + 1, dtype=float)
    down = np.arange(window - 1, 0, -1, dtype=float)
    w = np.concatenate([up, down])
    return w / w.sum()

TRI_WINDOW = 20
INIT_MEAN_N = 150

W_TRI = triangular_weights(TRI_WINDOW)

_mean_buf = []
_mean_ready = False
_mean_value = 0.0
_fir_buf = deque(maxlen=len(W_TRI))

def process_frame_xy(frame, *, global_time=False):
    if not frame:
        return None, None

    global _mean_ready, _mean_value

    y_raw = np.fromiter(frame, dtype=float)

    # ---- streaming-equivalent remove_initial_mean ----
    if not _mean_ready:
        _mean_buf.extend(y_raw.tolist())
        if len(_mean_buf) < INIT_MEAN_N:
            return None, None
        _mean_value = float(np.mean(_mean_buf[:INIT_MEAN_N]))
        _mean_ready = True

    y0 = y_raw - _mean_value

    # ---- TRUE streaming FIR (equivalent to update) ----
    yf_list = []
    for v in y0:
        _fir_buf.append(v)
        if len(_fir_buf) == len(W_TRI):
            yf_list.append(
                float(np.dot(W_TRI, np.fromiter(_fir_buf, dtype=float)))
            )

    if not yf_list:
        return None, None

    yf = np.asarray(yf_list, dtype=float)

    # ---- x axis (same logic as update) ----
    x = np.arange(yf.size, dtype=float) * DT

    return x, yf

async def callCalculateTV():
    snapshot = list(all_samples_pa)
    logger.info("Collected samples:", len(snapshot))
    arr = np.asarray(snapshot, dtype=np.float64)
    await TV_calculations.calculateFinalTV(arr)
    all_samples_pa.clear()

ENCODE_EVERY_SEC = 0.05  # 50ms
TRI_WINDOW = 20                       # => FIR length L = 2*TRI_WINDOW - 1 = 39
SEND_RAW_UNTIL_FIR_READY = True       # send raw samples during warm-up

def triangular_weights(window: int) -> np.ndarray:
    if window < 2:
        raise ValueError("TRI_WINDOW must be >= 2")
    up = np.arange(1, window + 1, dtype=np.float64)
    down = np.arange(window - 1, 0, -1, dtype=np.float64)
    w = np.concatenate([up, down])
    w /= w.sum()
    return w

W_TRI = triangular_weights(TRI_WINDOW)
L = len(W_TRI)                  # 39
_fir_buf = deque(maxlen=L)      # FIR state buffer (persistent)

def fir_process_frame(frame_samples: List[float], weights: np.ndarray, buf: deque) -> List[float]:
    """
    Streaming FIR: for each new sample, once buf is full, produce one output.
    Returns the outputs generated by THIS frame.
    """
    out = []
    for v in frame_samples:
        buf.append(v)
        if len(buf) == len(weights):
            out.append(float(np.dot(weights, np.fromiter(buf, dtype=np.float64))))
    return out

async def encoder_loop_deprecated():
    while True:
        try:
            frame = frame_queue.get_nowait()
            
            if len(frame) != 20:
                continue

            frame = [int(v) & 0xFFFFFF for v in frame]
            
            if frame:  # Only send non-empty
                await _SEND_ASYNC(f"dataFromLib~107~{json.dumps(frame)}")
        except queue.Empty:
            await asyncio.sleep(0.05)
        except Exception as e:
            logger.info("[encoder] ERROR:", e)
            await asyncio.sleep(0.05)

  
# Pressure conversion helper
def decode_pressure_from_message(message):
    """
    Parse 'spirodata~[AA:BB:...:ZZ]' into pressures (Pa).
    Synced with forced_calculations.
    """
    m = re.search(r'\[([0-9A-Fa-f:]+)\]', message)
    if not m:
        return []

    try:
        b = [int(t, 16) & 0xFF for t in m.group(1).split(':')]
    except ValueError:
        return []

    if len(b) < 120:
        return []

    if b[0] != 83 or b[1] != 72 or b[119] != 70:
        return []

    # Frame Gating Logic 
    if not hasattr(decode_pressure_from_message, "_frame_count"):
        decode_pressure_from_message._frame_count = 0
    
    decode_pressure_from_message._frame_count += 1
    count = decode_pressure_from_message._frame_count

    if count <= 5:
        if all(v == 0 for v in b[7:14]):
            return []

    OS_dig = 2**23
    FSS_inH2O = 120.0
    INH2O_TO_PA = 249.089

    out = []
    for i in range(7, 107, 5):
        if (i + 4) >= len(b):
            break
        
        b2 = b[i+2] & 0xFF
        b1 = b[i+3] & 0xFF
        b0 = b[i+4] & 0xFF
        
        decimal_count = (b2 << 16) | (b1 << 8) | b0            

        p_inH2O = 1.25 * ((decimal_count - OS_dig) / (2**24)) * FSS_inH2O
        out.append(p_inH2O * INH2O_TO_PA)

    return out

# WebSocket listener running in a background thread
async def ws_listener():
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("WebSocket connection established.")
            await websocket.send("BleAppletInit")
            logger.info("Sent: BleAppletInit")
            await asyncio.sleep(1)
            await websocket.send("startScanFromHtml~60")
            logger.info("Sent: startScanFromHtml~60")
            
            # TRACK TOTAL SAMPLES FOR RELATIVE TIME
            total_sample_count = 0 

            # Open raw log for appending
            with open(LOG_FILE, "a", buffering=1, encoding="utf-8") as f_raw:
                async for message in websocket:
                    if stop_event.is_set(): 
                        break
                    f_raw.write(message.strip() + "\n")

                    if message.startswith("spirodata~"):
                        # Decode
                        data = decode_pressure_from_message(message)                             

                        if data:
                            frame_queue.put(data)

                            for val in data:
                                # INCREMENT COUNTER
                                total_sample_count += 1
                                
                                # CALCULATE RELATIVE TIME (0.005, 0.010, etc.)
                                rel_time = total_sample_count * DT
                                
                                # logger.info TO TERMINAL
                                logger.info(f"[Time: {rel_time:.3f}s] {val:8.2f} Pa")

                                # Send to plot
                                pressure_queue.put(val)

    except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
        logger.info(f"WebSocket connection failed: {e}")
    except Exception as e:
        logger.info(f"An error occurred in the WebSocket listener: {e}")

def start_ws_thread():
    asyncio.run(ws_listener())

if DEBUGGING:
    # Start websocket listener in background thread
    setup_logging()
    logger = logging.getLogger(__name__)
    
    ws_thread = threading.Thread(target=start_ws_thread, daemon=True) # Renamed to ws_thread
    ws_thread.start()

    # Setup plot
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=1)
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Pressure (Pa)")
    ax.set_title("Real-time Pressure (Pa)")
    ax.grid(True, alpha=0.3)

    YMIN, YMAX = -5000, 5000
    ax.set_xlim(0, 20)
    ax.set_ylim(YMIN, YMAX)

# Segmentation params
TRI_WINDOW   = 20    
INIT_MEAN_N  = 150   
END_MEAN_N   = 150   

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

W_TRI = triangular_weights(TRI_WINDOW)

# Animation update function
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
        
        if start_data_time is None:
            start_data_time = now
            logger.info(f"[Realtime] Data flow started. Timer set for {TEST_DURATION} seconds.")

    if not pressures_pa:
        return line,

    y_raw = np.fromiter(pressures_pa, dtype=float)
    x = np.arange(y_raw.size, dtype=float) * DT
    y0, _ = remove_initial_mean(y_raw, INIT_MEAN_N)
    yf = streaming_fir(y0, W_TRI)
    if yf.size == 0: return line,
    x_f = x[-yf.size:]
    
    line.set_data(x_f, yf)
            
    if start_data_time:
        ax.set_title(f"Real-time Pressure (Pa)")
    else:
        ax.set_title("Real-time Pressure (Pa) — Waiting for data...")

    #  AUTO-CLOSE LOGIC
    should_close = False
    close_reason = ""

    if (not got_new) and last_data_time and (now - last_data_time) > IDLE_TIMEOUT_SEC:
        should_close = True
        close_reason = "Idle timeout (no data)"

    if start_data_time and (now - start_data_time) > TEST_DURATION:
        should_close = True
        close_reason = f"Test duration limit ({TEST_DURATION}s) reached"

    if should_close and not end_of_test_reported:
        end_of_test_reported = True
        logger.info(f"[Realtime] {close_reason} — closing plot and running analysis…")
        timer = fig.canvas.new_timer(interval=100)
        timer.add_callback(lambda: plt.close(fig))
        timer.start()

    return line,


if DEBUGGING:
    #Start animation
    ani = FuncAnimation(fig, update, interval=50, blit=False)
    plt.show()

    # Signal the thread to stop and wait for it
    logger.info("[Realtime] Plot closed. Waiting for data stream to finish...")
    stop_event.set()
    if 'ws_thread' in globals() and ws_thread.is_alive():
        ws_thread.join(timeout=2.0) # Wait up to 2 seconds for it to finish logger.infoing

    #run tidal calculations after realtime window closes
    try:
        time.sleep(0.5)
        here = os.path.abspath(os.path.dirname(__file__))
        tv_path = os.path.join(here, "TV_calculations.py")
        
        if not os.path.exists(tv_path):
            logger.info(f"[tidal→TV] Error: '{tv_path}' not found. Cannot run analysis.")
        else:
            spec = importlib.util.spec_from_file_location("TV_calculations", tv_path)
            tv = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tv)

            try:
                tv.FILE = LOG_FILE
            except Exception:
                pass

            logger.info(f"[tidal→TV] Running TV_calculations on: {LOG_FILE}")
            if hasattr(tv, "run_one"):
                tv.run_one(LOG_FILE)
            elif hasattr(tv, "main"):
                tv.main()
            else:
                logger.info("[tidal→TV] TV_calculations: no run_one() or main() function found.")

    except Exception as e:
        logger.info(f"[tidal→TV] Failed to run TV_calculations: {e}")
