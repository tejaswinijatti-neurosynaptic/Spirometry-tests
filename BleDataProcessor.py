#!/usr/bin/env python3
from __future__ import annotations

import sys
import asyncio
import websockets
import threading
import ctypes
from ctypes import wintypes
import logging

def ensure_single_instance(name="ReMeDi_BleDataProcessor"):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    mutex = kernel32.CreateMutexW(
        None,
        wintypes.BOOL(True),
        name
    )

    ERROR_ALREADY_EXISTS = 183
    if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        # Another instance is already running
        sys.exit(0)

    return mutex  # keep reference alive

# ----------------------------
# Paths / imports
# ----------------------------
from logging_config import setup_logging
from tidal import TV_calculations, tidal_realtime_analysis
from forced import finalResultCalculator, forced_realtime_data
from UrineTest import OR_App

URI = "ws://localhost:8444/bleDataProcessor/"
DEBUG = False

class BleApplet:
    """BLEApplet with RPC methods from all imported modules"""

    def __init__(self):
        try:
            # TV methods
            if "TV_calculations" in globals():
                for name in dir(TV_calculations):
                    if not name.startswith("_") and callable(getattr(TV_calculations, name)):
                        setattr(self, f"tv_{name}", getattr(TV_calculations, name))

            # Tidal methods
            if "tidal_realtime_analysis" in globals():
                for name in dir(tidal_realtime_analysis):
                    if not name.startswith("_") and callable(getattr(tidal_realtime_analysis, name)):
                        setattr(self, f"tidal_{name}", getattr(tidal_realtime_analysis, name))

            # Forced methods
            if 'finalResultCalculator' in globals():
                for name in dir(finalResultCalculator):
                    if not name.startswith('_') and callable(getattr(finalResultCalculator, name)):
                        setattr(self, f'forced_{name}', getattr(finalResultCalculator, name))

            # Forced realtime methods
            if 'forced_realtime_data' in globals():
                for name in dir(forced_realtime_data):
                    if not name.startswith('_') and callable(getattr(forced_realtime_data, name)):
                        setattr(self, f'realtime_{name}', getattr(forced_realtime_data, name))

            if 'OR_App' in globals():
                for name in dir(OR_App):
                    if not name.startswith('_') and callable(getattr(OR_App, name)):
                        setattr(self, f'OR_{name}', getattr(OR_App, name))
        except Exception as e:
            logger.error(f"Warning: Could not load module methods: {e}")


class BleWebSocketHandler:
    def __init__(self):
        self.remote = None
        self.ble_applet = BleApplet()
        self.connected = False
  
    async def on_connect(self, websocket):
        self.remote = websocket
        self.connected = True
        logger.info("**Connected")
        self.ble_applet = BleApplet()
        logger.info("BleApplet instance created with all module methods")

        # Inject sender into tidal module (CRITICAL)        
        if 'tidal_realtime_analysis' in globals():
            tidal_realtime_analysis.set_sender(self.send_msg)
        else:
            logger.warning("WARNING: tidal_realtime_analysis not loaded")

        if 'TV_calculations' in globals():
            TV_calculations.set_sender(self.send_msg)

        if 'forced_realtime_data' in globals():
            forced_realtime_data.set_sender(self.send_msg)

        if 'finalResultCalculator' in globals():
            finalResultCalculator.set_sender(self.send_msg)

        if 'OR_App' in globals():
            OR_App.set_sender(self.send_msg)

    async def on_message(self, message: str):
        # Log raw incoming messages too (date-wise file in Logs/)
        # (keep it separate from self.log(...) formatting)
        logger.info(message.rstrip())

        if DEBUG:
            logger.info(f"Received: {message}")

        await self.call_jar_method(message)

    async def on_close(self, code, reason):
        self.connected = False
        self.remote = None
        logger.info(f"**Closed: {code} {reason}")

    async def send_msg(self, message: str):
        if self.remote and self.connected:
            await self.remote.send(message)
        else:
            logger.info("ERROR: not connected, cannot send")

    async def call_jar_method(self, msg: str):
        parts = msg.split("~")
        method_name = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        try:
            if hasattr(self.ble_applet, method_name):
                method = getattr(self.ble_applet, method_name)

                if len(args) == 0:
                    return await method() if asyncio.iscoroutinefunction(method) else method()
                if len(args) == 1:
                    return await method(args[0]) if asyncio.iscoroutinefunction(method) else method(args[0])
                if len(args) == 2:
                    return await method(args[0], args[1]) if asyncio.iscoroutinefunction(method) else method(args[0], args[1])
                if len(args) == 3:
                    return await method(args[0], args[1], args[2]) if asyncio.iscoroutinefunction(method) else method(args[0], args[1], args[2])
                if len(args) >= 4:
                    return await method(args[0], args[1], args[2], args[3]) if asyncio.iscoroutinefunction(method) else method(args[0], args[1], args[2], args[3])
            else:
                logger.error(f"ERROR: method not found: {method_name}")

        except Exception as e:
            logger.error(f"ERROR calling {method_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def ws_listener(self):
        while True:
            try:
                logger.info(f"Connecting to {URI} ...")
                async with websockets.connect(URI) as websocket:
                    await self.on_connect(websocket)
                    async for message in websocket:
                        await self.on_message(message)
            except Exception as e:
                logger.error(f"WebSocket loop error: {e}")
                await asyncio.sleep(2)

    def start_thread(self):
        def run():
            asyncio.run(self.ws_listener())

        threading.Thread(target=run, daemon=True).start()
        logger.info("WebSocket thread started")


if __name__ == "__main__":

    _mutex = ensure_single_instance()
    
    setup_logging()
    logger = logging.getLogger(__name__)

    handler = BleWebSocketHandler()
    handler.start_thread()

    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")