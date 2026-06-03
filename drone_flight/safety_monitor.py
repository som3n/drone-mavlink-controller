import time
import threading
from drone_flight.logger import log
from drone_flight import telemetry as telem

abort_mission = threading.Event()
abort_reason = ""


def safety_monitor(master, cfg, flight_start_time):
    global abort_reason
    safety = cfg["safety"]
    log.info("Safety monitor active.")

    while not abort_mission.is_set():
        now = time.time()

        # Telemetry timeout
        last_time = telem.get_last_msg_time()
        if (now - last_time) > safety["telemetry_timeout_sec"]:
            log.error(f"ABORT — telemetry lost. Last msg was {now - last_time:.2f}s ago")
            abort_reason = "telemetry lost"
            abort_mission.set()
            return

        # Max flight time
        if (now - flight_start_time) > safety["max_flight_sec"]:
            log.error("ABORT — max flight time exceeded.")
            abort_reason = "max flight time exceeded"
            abort_mission.set()
            return

        # IMU spike
        ax, ay, az = telem.get_raw_imu()
        if (ax > safety["crash_accel_g"] or
                ay > safety["crash_accel_g"] or
                az > safety["crash_accel_g"]):
            log.error(f"ABORT — IMU spike ax={ax:.1f}g ay={ay:.1f}g az={az:.1f}g")
            abort_reason = f"IMU spike ax={ax:.1f}g ay={ay:.1f}g az={az:.1f}g"
            abort_mission.set()
            return

        # Extreme attitude
        roll, pitch = telem.get_attitude()
        if roll is not None and pitch is not None:
            if abs(roll) > safety["crash_roll_deg"] or abs(pitch) > safety["crash_pitch_deg"]:
                log.error(f"ABORT — attitude roll={roll:.1f}° pitch={pitch:.1f}°")
                abort_reason = f"critical attitude roll={roll:.1f}° pitch={pitch:.1f}°"
                abort_mission.set()
                return

        # Low battery check (protect battery health, skip if USB only / 0V)
        volts = telem.get_battery_voltage()
        if volts is not None and volts > 1.0:
            min_voltage = safety.get(
                "battery_min_voltage",
                cfg["flight"].get("battery_min_voltage", 10.5)
            )
            if volts < min_voltage:
                log.error(f"ABORT — battery critical {volts:.2f}V (threshold: {min_voltage}V)")
                abort_reason = f"battery critical {volts:.2f}V"
                abort_mission.set()
                return

        time.sleep(0.05)

    log.info("Safety monitor exited.")
