🫁 Spirometer Real-Time & Forced Calculations

This repository provides a complete toolkit for real-time spirometry testing and post-processing analysis using pressure-based flow sensors.
It includes scripts to collect, visualize, and calculate parameters such as FVC, FEV₁, PEF, and FEF25–75 automatically after each test.

📦 Repository Contents
File	Description
forced_realtime_data.py: Handles live data capture, real-time plotting, and automatically triggers full analysis after each test.
forced_calculations.py:	Performs complete offline calculations and generates plots + lung metrics from a saved .log file.
SDK:	Contains the device SDK required for BLE/WebSocket communication with the spirometer. Make sure it’s running before you start real-time capture.

⚙️ Setup Instructions
1. Clone or Download the Repository
    git clone https://github.com/your-org/spirometry-analysis.git
    cd spirometry-analysis

2. Install Dependencies
pip install numpy matplotlib websockets asyncio

3. Start the SDK
Before running any Python script, start the SDK uploaded in this repo.
The SDK is responsible for initializing BLE communication with your spirometer device and exposing the WebSocket endpoint (ws://localhost:8444/bleWS/) that the realtime script connects to.

Keep the SDK running in the background.

🧩 File Workflow
>Run Only the Real-Time Script
>You do not need to run forced_calculations.py manually.
>Simply run the realtime script, and it will:
>Connect to the spirometer via SDK.
>Stream and save pressure readings to a .log file.
>Automatically trigger the calculations script after test completion.
NOTE: Make sure both forced_realtime_data.py and forced_calculations.py are kept in the same folder — the realtime script dynamically imports and executes the calculations module after test completion.

📁Folder Structure
A clean setup should look like this:

spirometry-analysis/
│
├── SDK/                       # BLE communication SDK (run this first)
│   ├── start_sdk.exe          # or similar entrypoint
│
├── forced_realtime_data.py    # Run this script for realtime + analysis
├── forced_calculations.py     # Called automatically by the realtime script
├── README.md
│
└── logs/
    ├── forced_test_001.log
    ├── forced_test_002.log
    └── plots_20251112_123456/

📊 Output
    During Realtime Capture:
      You’ll see three live plots:
          Pressure vs Time
          Flow–Volume Loop
          Volume vs Time
    After Test Completion:
          Analysis runs automatically.
          Metrics printed in terminal:
              FVC = 3.512 L
              FEV1 = 2.905 L
              FEV1_Percentage = 82.7 %
              PEF = 7.25 L/s
              .
              .
              .
    Figures saved to:  logs/plots_<timestamp>/

🧪 Quick Start Summary
1. Start the SDK (keeps BLE connection alive).
2. Run:
    python forced_realtime_data.py
3. Breathe through the spirometer. Wait for ~1s after the test; the window will close and show your analysis results.

