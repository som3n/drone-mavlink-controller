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
    "roll_deg", "pitch_deg", "battery_h", "comm_h", "stability_h",
    "sensor_h", "overall_h", "anomaly_score", "risk_score",
    "pred_alt_1s", "pred_alt_3s", "pred_alt_5s", "pred_volt_rem"
])


def log_telemetry(phase, throttle, alt=None, climb=None, roll=None, pitch=None,
                  health=None, anomaly_score=None, risk_score=None,
                  pred_alt_1s=None, pred_alt_3s=None, pred_alt_5s=None,
                  pred_volt_rem=None):
    h = health if health is not None else {}
    _csv_writer.writerow([
        round(time.time(), 3),
        phase,
        throttle,
        round(alt, 3) if alt is not None else "",
        round(climb, 3) if climb is not None else "",
        round(roll, 2) if roll is not None else "",
        round(pitch, 2) if pitch is not None else "",
        round(h.get("battery", 100.0), 1),
        round(h.get("communication", 100.0), 1),
        round(h.get("stability", 100.0), 1),
        round(h.get("sensor", 100.0), 1),
        round(h.get("overall", 100.0), 1),
        round(anomaly_score, 1) if anomaly_score is not None else 0.0,
        round(risk_score, 1) if risk_score is not None else 0.0,
        round(pred_alt_1s, 3) if pred_alt_1s is not None else "",
        round(pred_alt_3s, 3) if pred_alt_3s is not None else "",
        round(pred_alt_5s, 3) if pred_alt_5s is not None else "",
        round(pred_volt_rem, 3) if pred_volt_rem is not None else "",
    ])
    _csv_file.flush()


def close_logger():
    _csv_file.close()
