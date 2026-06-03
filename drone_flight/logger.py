import logging
import csv
import os
import time
from datetime import datetime

os.makedirs("logs", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"logs/{timestamp}.log"
CSV_FILE = f"logs/{timestamp}_telemetry.csv"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("drone")

_csv_file = open(CSV_FILE, "w", newline="")
_csv_writer = csv.writer(_csv_file)
_csv_writer.writerow([
    "time_s", "phase", "throttle_pct", "alt_m", "climb_mps",
    "roll_deg", "pitch_deg"
])


def log_telemetry(phase, throttle, alt=None, climb=None, roll=None, pitch=None):
    _csv_writer.writerow([
        round(time.time(), 3),
        phase,
        throttle,
        round(alt, 3) if alt is not None else "",
        round(climb, 3) if climb is not None else "",
        round(roll, 2) if roll is not None else "",
        round(pitch, 2) if pitch is not None else "",
    ])
    _csv_file.flush()


def close_logger():
    _csv_file.close()
