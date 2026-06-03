import time
import threading
from drone_flight.logger import log
from drone_flight import telemetry as telem

abort_mission = threading.Event()
abort_reason = ""


def safety_monitor(master, cfg, flight_start_time, dt=None, hm=None):
    global abort_reason
    safety = cfg["safety"]
    log.info("Safety monitor active.")
    att_err_consecutive_ticks = 0

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

        bench_test = cfg.get("bench_test", True)
        roll, pitch = telem.get_attitude()

        if not bench_test:
            # 1. IMU spike (Impact > 4g)
            ax, ay, az = telem.get_raw_imu()
            if (ax > safety["crash_accel_g"] or
                    ay > safety["crash_accel_g"] or
                    az > safety["crash_accel_g"]):
                log.error(f"ABORT — IMU spike ax={ax:.1f}g ay={ay:.1f}g az={az:.1f}g")
                abort_reason = f"IMU spike ax={ax:.1f}g ay={ay:.1f}g az={az:.1f}g"
                abort_mission.set()
                return

            # 2. Extreme attitude / Inverted vessel check
            if roll is not None and pitch is not None:
                if (abs(roll) > safety["crash_roll_deg"] or
                        abs(pitch) > safety["crash_pitch_deg"] or
                        abs(roll) > 90.0 or abs(pitch) > 90.0):
                    log.error(f"ABORT — attitude roll={roll:.1f}° pitch={pitch:.1f}°")
                    abort_reason = f"critical attitude roll={roll:.1f}° pitch={pitch:.1f}°"
                    abort_mission.set()
                    return

            # 3. Attitude Error check (> 45 deg for > 2 seconds)
            target_roll, target_pitch = telem.get_target_attitude()
            if roll is not None and pitch is not None:
                roll_err = target_roll - roll
                pitch_err = target_pitch - pitch
                max_err = max(abs(roll_err), abs(pitch_err))
                if max_err > 45.0:
                    att_err_consecutive_ticks += 1
                else:
                    att_err_consecutive_ticks = 0

                if att_err_consecutive_ticks >= 40:  # 2s at 20Hz (0.05s sleep)
                    log.error(
                        f"ABORT — Attitude error > 45 deg for more than 2s "
                        f"(error={max_err:.1f}°)"
                    )
                    abort_reason = f"critical attitude error {max_err:.1f}°"
                    abort_mission.set()
                    return

            # 4. Critical Stability Score check
            with telem.state.lock:
                stability_score = telem.state.stability_score
            if stability_score >= 80.0:
                log.error(f"ABORT — Stability score went critical: {stability_score:.1f}")
                abort_reason = f"critical stability score {stability_score:.1f}"
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

        # Predictive checks if digital twin and health monitor are provided
        if not bench_test and dt is not None and hm is not None:
            # 1. Predictive battery exhaustion check
            if volts is not None and volts > 1.0:
                bat_state = dt.predict_battery_state()
                rem_flight_time = bat_state["remaining_flight_time_sec"]
                if rem_flight_time < 15.0:
                    log.error(
                        f"PREDICTIVE ABORT — Battery expected to deplete "
                        f"to cutoff in {rem_flight_time:.1f}s"
                    )
                    abort_reason = "predictive battery critical"
                    abort_mission.set()
                    return

            # 2. Predictive stability failure check (attitude drift / oscillation growth)
            stability_trends = dt.evaluate_stability_trends()
            if stability_trends["oscillation_growing"]:
                risk = dt.get_flight_risk_score()
                if risk > 80.0:
                    log.error("PREDICTIVE ABORT — Growing oscillations and high flight risk")
                    abort_reason = "predictive stability failure"
                    abort_mission.set()
                    return

            # 3. Statistical Anomaly Detection check
            anomalies = hm.detect_anomalies(dt)
            if anomalies["is_anomaly"] and anomalies["anomaly_score"] >= 80.0:
                reasons_str = ", ".join(anomalies["reasons"])
                log.error(f"PREDICTIVE ABORT — Critical statistical anomalies: {reasons_str}")
                abort_reason = f"anomaly: {reasons_str}"
                abort_mission.set()
                return

        time.sleep(0.05)

    log.info("Safety monitor exited.")
