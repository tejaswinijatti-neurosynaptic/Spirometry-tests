Tidal Volume Real-Time & Analysis:
This repository provides tools for real-time tidal breathing analysis using a pressure-based spirometer.
It captures live breathing signals, visualizes pressure data in real time, and automatically computes tidal volume (TV) and per-breath statistics once the recording ends.

📦 Repository Contents
File	Description:
tidal_realtime_analysis.py:	Handles live data capture and visualization, then triggers TV_calculations.py automatically for tidal volume computation.
TV_calculations.py:	Performs offline tidal volume analysis from a saved .log file and generates summary statistics and plots.
SDK/:	Contains the device SDK required to enable BLE/WebSocket communication with the spirometer. The SDK must be running before you start the real-time analysis.

⚙️ Setup Instructions
1. Clone or Download the Repository
git clone https://github.com/your-org/tidal-analysis.git
cd tidal-analysis

2. Install Dependencies
pip install numpy matplotlib websockets asyncio

3. Start the SDK
Before starting any Python scripts, launch the SDK provided in this repository.
It establishes the BLE connection and serves the WebSocket endpoint (ws://localhost:8444/bleWS/) used for real-time data streaming.
Keep the SDK running in the background while testing.

🧩 File Workflow
Run Only the Real-Time Script
You do not need to run TV_calculations.py manually.

This will:
    Connect to the spirometer through the SDK.
    Stream real-time pressure data and save it to a .log file.
    Display a live Pressure vs Time plot.
    Automatically call TV_calculations.py after the data stream ends (≈1 s of inactivity).

NOTE: Keep tidal_realtime_analysis.py and TV_calculations.py in the same folder, since the real-time script imports the analysis module dynamically.

📁 Recommended Folder Structure
tidal-analysis/
│
├── SDK/                         # BLE communication SDK (run first)
│   ├── start_sdk.exe
│
├── tidal_realtime_analysis.py    # Run this script for realtime plotting + analysis
├── TV_calculations.py            # Automatically executed after test
├── README.md
│
└── logs/
    ├── tidal_test_001.log
    ├── tidal_test_002.log
    └── plots_20251112_145955/

🧠 Important Notes
>Start the SDK first, then run tidal_realtime_analysis.py.
>Always save each reading to a new .log file. If you reuse the same log file, the data will append and corrupt your results.
>Both scripts must remain in the same folder for automatic execution.
>The test ends automatically after roughly 1 second of no new data; analysis begins immediately.

📊 Output
During Real-Time Capture:
A live Pressure vs Time plot displays your breathing pattern.

After Capture Ends:
>The analysis script (TV_calculations.py) runs automatically and:
>Segments inhalations and exhalations.
>Computes per-breath tidal volumes.
>Prints averages, standard deviations, and total sample count.
>Generates a color-coded plot highlighting inhale and exhale regions.
>
Example console output:

  --- Tidal Volume Report ---
  File: tidal_test_001.log
  Samples (raw): 23841, Samples (filtered): 23682
  Segments found: inhales=8, exhales=8
  Inhale peaks:  mean=0.521 L, std=0.034 L, n=8
  Exhale peaks:  mean=0.528 L, std=0.031 L, n=8
  Final TV: 0.525 L  (TV = average(mean_inhale, mean_exhale))

Plots show:
Filtered pressure waveform
Inhale regions shaded pink
Exhale regions shaded lavender
Volume annotations for each breath segment

🧪 Quick Start Recap

1. Run the SDK.
2. Start:
      python tidal_realtime_analysis.py
3. Breathe normally through the spirometer.
4. Wait ~1 s after the last breath — the window closes and prints the TV report.
Check saved .log and plots in your logs/ folder.

