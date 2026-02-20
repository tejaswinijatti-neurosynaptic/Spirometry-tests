import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import tempfile

LOGS_DIR = Path(tempfile.gettempdir()) / "ReMeDi_Logs" / "BleDataProcessor" / "Logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def setup_logging():
    log_file = LOGS_DIR / "ble.log"

    handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=10,
        encoding="utf-8"
    )

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        "%Y/%m/%d %H:%M:%S"
    )

    handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler, logging.StreamHandler(sys.stdout)]
    )
