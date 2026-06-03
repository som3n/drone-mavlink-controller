import time
import platform
import threading
import yaml
import json
import os
from datetime import datetime

from pymavlink import mavutil
from drone_flight.logger import log, log_telemetry, close_logger
from drone_flight import telemetry as telem
from drone_flight.safety_monitor import safety_monitor, abort_mission
from drone_flight.digital_twin import DigitalTwin
from drone_flight.health_monitor import HealthMonitor, std_dev


def load_config(path="config/bench_test.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def check_abort(phase):
    if abort_mission.is_set():
        from drone_flight.safety_monitor import abort_reason
        log.error(f"Abort detected during: {phase} | Reason: {abort_reason}")
        return True
    return False


def abort_and_land(master, reason, bench=True):
    from drone_flight.safety_monitor import abort_reason
    full_reason = f"{reason} | {abort_reason}" if abort_reason else reason
    log.error(f"EMERGENCY RECOVERY TRIGGERED — {full_reason}")
    reason_lower = full_reason.lower()

    if ("attitude" in reason_lower or "crash" in reason_lower or
            "imu" in reason_lower or "user interrupt" in reason_lower):
        log.warning("CRITICAL SAFETY FAULT — DISARMING MOTORS IMMEDIATELY!")
        telem.send_rc_throttle(master, 0, "emergency")
        telem.disarm(master)
        return

    log.warning("MAJOR FAULT — INITIALIZING CONTROLLED AUTONOMOUS LANDING...")
    try:
        master.mav.rc_channels_override_send(
            master.target_system, master.target_component,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        telem.set_mode(master, "LAND")

        land_start = time.time()
        timeout_limit = 5.0 if bench else 30.0
        while time.time() - land_start < timeout_limit:
            with telem.state.lock:
                is_armed = telem.state.armed
            if not is_armed:
                log.info("Ground confirmed: Disarmed successfully.")
                return
            time.sleep(0.2)

        log.warning("Landing timeout exceeded — force-disarming!")
        telem.disarm(master)
    except Exception as e:
        log.error(f"Controlled landing error: {e}. Falling back to manual ramp.")
        for pct in range(int(telem.current_throttle), 0, -5):
            telem.send_rc_throttle(master, pct, "emergency")
            time.sleep(0.3)
        telem.send_rc_throttle(master, 0, "emergency")
        telem.disarm(master)


def update_and_log_framework(phase, throttle, dt, hm):
    alt, climb = telem.get_vfr_hud()
    roll, pitch = telem.get_attitude()
    xacc, yacc, zacc = telem.get_raw_imu()
    volts = telem.get_battery_voltage()

    dt.update(alt, climb, volts, roll, pitch, xacc, yacc, zacc)

    health = hm.calculate_health_scores(dt)
    anomalies = hm.detect_anomalies(dt)
    anomaly_score = anomalies["anomaly_score"]
    risk_score = dt.get_flight_risk_score()

    pred_alt_1s = dt.predict_future_altitude(1.0)
    pred_alt_3s = dt.predict_future_altitude(3.0)
    pred_alt_5s = dt.predict_future_altitude(5.0)

    bat_state = dt.predict_battery_state()
    pred_volt_rem = bat_state["remaining_voltage"]

    # Read Attitude Recovery state variables
    with telem.state.lock:
        target_roll = telem.state.target_roll
        target_pitch = telem.state.target_pitch
        roll_err = telem.state.roll_err
        pitch_err = telem.state.pitch_err
        roll_rate = telem.state.rollspeed
        pitch_rate = telem.state.pitchspeed
        roll_corr = telem.state.roll_corr
        pitch_corr = telem.state.pitch_corr
        stability_score = telem.state.stability_score
        recovery_state = telem.state.recovery_state

    log_telemetry(
        phase, throttle, alt, climb, roll, pitch,
        health=health, anomaly_score=anomaly_score, risk_score=risk_score,
        pred_alt_1s=pred_alt_1s, pred_alt_3s=pred_alt_3s, pred_alt_5s=pred_alt_5s,
        pred_volt_rem=pred_volt_rem,
        target_roll=target_roll, roll_err=roll_err,
        target_pitch=target_pitch, pitch_err=pitch_err,
        roll_rate=roll_rate, pitch_rate=pitch_rate,
        roll_corr=roll_corr, pitch_corr=pitch_corr,
        stability_score=stability_score, recovery_state=recovery_state
    )


def set_parameter(master, name, value, param_type=mavutil.mavlink.MAV_PARAM_TYPE_REAL32):
    log.info(f"Setting parameter {name} to {value}...")
    param_name = name.encode('utf-8')
    try:
        master.mav.param_set_send(
            master.target_system,
            master.target_component,
            param_name,
            float(value),
            param_type
        )
    except Exception as e:
        log.warning(f"Error setting parameter {name}: {e}")


def load_learned_params():
    path = "config/learned_params.json"
    if os.path.exists(path):
        try:
            with open(path) as f:
                params = json.load(f)
                log.info(f"Loaded learned flight parameters: {params}")
                return params
        except Exception as e:
            log.warning(f"Error loading learned parameters: {e}")
    return {}


def save_learned_params(liftoff_th, hover_th, landing_th):
    os.makedirs("config", exist_ok=True)
    path = "config/learned_params.json"
    params = {
        "liftoff_throttle": int(liftoff_th),
        "hover_throttle": int(hover_th),
        "landing_throttle": int(landing_th)
    }
    try:
        with open(path, "w") as f:
            json.dump(params, f, indent=4)
        log.info(f"Saved learned flight parameters to {path}: {params}")
    except Exception as e:
        log.warning(f"Error saving learned parameters: {e}")


def generate_post_flight_report(flight_start, dt, hover_altitudes_list):
    duration = time.time() - flight_start
    altitudes = dt.history["alt"]
    climbs = dt.history["climb"]
    rolls = dt.history["roll"]
    pitches = dt.history["pitch"]
    voltages = dt.history["voltage"]

    max_alt = max(altitudes) if altitudes else 0.0
    avg_climb = sum(climbs) / len(climbs) if climbs else 0.0
    max_roll = max(abs(r) for r in rolls) if rolls else 0.0
    max_pitch = max(abs(p) for p in pitches) if pitches else 0.0

    hover_stability = std_dev(hover_altitudes_list) if hover_altitudes_list else 0.0

    bat_used = 0.0
    if voltages and voltages[0] > 1.0:
        bat_used = max(0.0, voltages[0] - voltages[-1])

    report = (
        f"==================================================\n"
        f"                FLIGHT ANALYTICS REPORT           \n"
        f"==================================================\n"
        f"Flight Duration:         {duration:.1f}s\n"
        f"Maximum Altitude:        {max_alt:.2f}m\n"
        f"Average Climb Rate:      {avg_climb:.2f} m/s\n"
        f"Hover Stability (StdDev): {hover_stability:.3f}m\n"
        f"Maximum Roll:            {max_roll:.1f}°\n"
        f"Maximum Pitch:           {max_pitch:.1f}°\n"
        f"Battery Consumed:        {bat_used:.2f}V\n"
        f"==================================================\n"
    )

    log.info("\n" + report)

    try:
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"logs/{timestamp}_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        log.info(f"Saved flight analytics report to {report_path}")
    except Exception as e:
        log.warning(f"Error saving flight analytics report: {e}")


def calculate_attitude_recovery(target_roll, target_pitch, dt):
    roll, pitch = telem.get_attitude()
    roll_rate, pitch_rate = telem.get_attitude_rates()

    if roll is None or pitch is None or roll_rate is None or pitch_rate is None:
        return 1500, 1500, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "NORMAL"

    # 1. Error calculation
    roll_err = target_roll - roll
    pitch_err = target_pitch - pitch
    max_err = max(abs(roll_err), abs(pitch_err))

    # 2. Stability Zones Evaluation
    if max_err <= 5.0:
        kp_factor = 0.0
        kd_factor = 0.0
        recovery_state = "NORMAL"
    elif max_err <= 10.0:
        kp_factor = 0.5
        kd_factor = 0.5
        recovery_state = "MINOR CORRECTION"
    elif max_err <= 20.0:
        kp_factor = 1.0
        kd_factor = 1.0
        recovery_state = "ACTIVE RECOVERY"
    elif max_err <= 35.0:
        kp_factor = 1.5
        kd_factor = 1.0
        recovery_state = "HIGH RECOVERY"
    else:
        kp_factor = 2.0
        kd_factor = 1.0
        recovery_state = "RECOVERY MODE"

    # PD Controller Gains
    KP = 8.0 * kp_factor
    KD = 1.5 * kd_factor

    # PD correction formulas (damping positive rates)
    roll_corr = KP * roll_err - KD * roll_rate
    pitch_corr = KP * pitch_err - KD * pitch_rate

    # Clamp corrections to safe range [-200, 200]
    roll_corr = max(-200.0, min(200.0, roll_corr))
    pitch_corr = max(-200.0, min(200.0, pitch_corr))

    # Add corrections to neutral (1500)
    roll_cmd = int(1500 + roll_corr)
    pitch_cmd = int(1500 + pitch_corr)

    # 3. Stability Score Calculation (0 to 100)
    if max_err <= 5.0:
        att_score = 0.0
    else:
        att_score = min(30.0, (max_err - 5.0) / 30.0 * 30.0)

    rolls = dt.history["roll"]
    pitches = dt.history["pitch"]

    def calculate_variance(data):
        if len(data) < 2:
            return 0.0
        avg = sum(data) / len(data)
        return sum((x - avg) ** 2 for x in data) / len(data)

    roll_var = calculate_variance(rolls)
    pitch_var = calculate_variance(pitches)

    roll_var_score = min(20.0, (roll_var / 5.0) * 20.0)
    pitch_var_score = min(20.0, (pitch_var / 5.0) * 20.0)

    max_rate = max(abs(roll_rate), abs(pitch_rate))
    if max_rate <= 10.0:
        rate_score = 0.0
    else:
        rate_score = min(30.0, (max_rate - 10.0) / 90.0 * 30.0)

    stability_score = att_score + roll_var_score + pitch_var_score + rate_score
    stability_score = min(100.0, max(0.0, stability_score))

    # Update state variables
    with telem.state.lock:
        telem.state.stability_score = stability_score
        telem.state.recovery_state = recovery_state

    return (
        roll_cmd, pitch_cmd,
        roll_err, pitch_err,
        roll_rate, pitch_rate,
        roll_corr, pitch_corr,
        stability_score, recovery_state
    )


def send_throttle_safe(master, target_pct, phase, dt):
    target_roll, target_pitch = telem.get_target_attitude()

    (
        roll_cmd, pitch_cmd,
        roll_err, pitch_err,
        roll_rate, pitch_rate,
        roll_corr, pitch_corr,
        stability_score, recovery_state
    ) = calculate_attitude_recovery(target_roll, target_pitch, dt)

    with telem.state.lock:
        telem.state.roll_err = roll_err
        telem.state.pitch_err = pitch_err
        telem.state.roll_corr = roll_corr
        telem.state.pitch_corr = pitch_corr

    # Log warnings for significant errors (ACTIVE RECOVERY or worse)
    if recovery_state in ["ACTIVE RECOVERY", "HIGH RECOVERY", "RECOVERY MODE"]:
        log.warning(
            f"  [ATTITUDE WARNING] State: {recovery_state} | "
            f"Roll Error: {roll_err:.1f}° Pitch Error: {pitch_err:.1f}° | "
            f"Corrections: Roll={roll_cmd} Pitch={pitch_cmd}"
        )

    telem.send_rc_throttle(master, target_pct, phase, roll=roll_cmd, pitch=pitch_cmd)


def run_flight(config_path="config/bench_test.yaml"):
    cfg = load_config(config_path)
    fc = cfg["flight"]
    bench = cfg["bench_test"]

    dt = DigitalTwin(update_rate_hz=10)
    hm = HealthMonitor()

    # Load learned parameters
    learned = load_learned_params()
    learned_hover = learned.get("hover_throttle")
    if learned_hover is not None:
        fc["known_hover_throttle_pct"] = learned_hover
        log.info(f"Overriding known hover throttle with learned value: {learned_hover}%")

    log.info(f"Platform: {platform.system()} | Mode: {'BENCH TEST' if bench else 'REAL FLIGHT'}")

    port = cfg["port"]["windows"] if platform.system() == "Windows" else cfg["port"]["linux"]
    baud = cfg["port"]["baud"]
    log.info(f"Connecting: {port} @ {baud}")

    master = mavutil.mavlink_connection(port, baud=baud)
    telem.current_throttle = 0.0

    try:
        log.info("Waiting for heartbeat...")
        hb = master.wait_heartbeat(timeout=10.0)
        if hb is None:
            log.error("Timeout waiting for heartbeat. Check port connection and power.")
            return
        log.info(f"Heartbeat OK — system {master.target_system}")

        # Request data streams
        log.info("Requesting data streams...")
        master.mav.request_data_stream_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
        )
        master.mav.request_data_stream_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 20, 1  # ATTITUDE at 20Hz
        )
        master.mav.request_data_stream_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION, 10, 1  # VFR_HUD / GLOBAL_POSITION_INT
        )
        time.sleep(1)

        # Disable radio failsafes for testing
        set_parameter(master, "FS_THR_ENABLE", 0)
        set_parameter(master, "FS_OPTIONS", 0)
        if bench:
            set_parameter(master, "FS_CRASH_CHECK", 0)
        else:
            set_parameter(master, "FS_CRASH_CHECK", 1)

        # Set fake GPS Global Origin and Home Position to initialize EKF origin and zero altitude
        log.info("Initializing EKF origin and home position...")
        lat = int(12.9716 * 1e7)
        lon = int(77.5946 * 1e7)
        origin_alt = 500.0  # 500m default origin

        master.mav.set_gps_global_origin_send(
            master.target_system,
            lat,
            lon,
            int(origin_alt * 1000)
        )
        time.sleep(0.5)

        # Wait and read EKF altitude to zero the home relative altitude
        ekf_alt = None
        for _ in range(20):
            msg = master.recv_match(blocking=True, timeout=0.1)
            if msg and msg.get_type() == 'GLOBAL_POSITION_INT':
                ekf_alt = msg.alt / 1000.0
                break

        if ekf_alt is None:
            # Fallback to VFR_HUD
            for _ in range(20):
                msg = master.recv_match(type='VFR_HUD', blocking=True, timeout=0.1)
                if msg:
                    ekf_alt = msg.alt
                    break

        if ekf_alt is None:
            log.warning("Could not read EKF altitude — using default origin altitude.")
            ekf_alt = origin_alt

        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_HOME,
            0,    # confirmation
            0,    # param 1: 0 = use specified lat/lon/alt
            0,    # param 2: unused
            0,    # param 3: unused
            0,    # param 4: unused
            12.9716,  # param 5: latitude
            77.5946,  # param 6: longitude
            ekf_alt   # param 7: altitude
        )
        log.info(f"EKF origin and home initialized (zero altitude: {ekf_alt:.2f}m).")
        time.sleep(0.5)

        # Check battery voltage if connected (both in bench and real flight)
        volts = telem.get_battery_voltage(master)
        if volts is not None and volts > 1.0:
            min_volts = fc.get("battery_min_voltage", 10.5)
            if volts < min_volts:
                log.error(f"Battery check failed: {volts:.2f}V (requires {min_volts}V)")
                return
            log.info(f"Battery: {volts:.2f}V OK")
        else:
            log.info("No battery detected (or USB powered only) — skipping startup battery check.")

        telem.set_mode(master, "STABILIZE")
        telem.arm(master, bench)

        # Start telemetry reader thread
        stop_reader = threading.Event()
        reader_thread = threading.Thread(
            target=telem.start_telemetry_reader,
            args=(master, stop_reader),
            daemon=True
        )
        reader_thread.start()

        flight_start = time.time()

        monitor = threading.Thread(
            target=safety_monitor,
            args=(master, cfg, flight_start, dt, hm),
            daemon=True
        )
        monitor.start()

        # Initialize target attitude
        telem.set_target_attitude(0.0, 0.0)

        # ── PHASE 1: Spin-up ──────────────────────────────────
        log.info("PHASE: Spin-up 10% x 5s")
        for _ in range(50):
            if check_abort("spinup"):
                abort_and_land(master, "spinup", bench)
                return
            send_throttle_safe(master, 10, "spinup", dt)
            update_and_log_framework("spinup", 10, dt, hm)
            time.sleep(0.1)

        # ── PHASE 2: Pre-lift stabilize ───────────────────────
        for pct, hold in [(15, 2.0), (20, 3.0)]:
            log.info(f"PHASE: Stabilize {pct}% x {hold}s")
            for _ in range(int(hold / 0.1)):
                if check_abort("stabilize"):
                    abort_and_land(master, "stabilize", bench)
                    return
                send_throttle_safe(master, pct, "stabilize", dt)
                update_and_log_framework("stabilize", pct, dt, hm)
                time.sleep(0.1)

        # ── PHASE 3: Takeoff ramp ─────────────────────────────
        log.info("PHASE: Takeoff ramp")
        hover_throttle = None
        liftoff_counter = 0

        for pct, hold in cfg["takeoff_ramp"]:
            log.info(f"  Ramp {pct}% x {hold}s")
            for _ in range(int(hold / 0.1)):
                if check_abort("ramp"):
                    abort_and_land(master, "ramp", bench)
                    return
                send_throttle_safe(master, pct, "ramp", dt)
                update_and_log_framework("ramp", pct, dt, hm)
                time.sleep(0.1)
                if hover_throttle is None and pct >= 30:
                    alt, climb = telem.get_vfr_hud()
                    if alt is not None and climb is not None:
                        if alt > fc["liftoff_alt_m"] and climb > fc["liftoff_climb_rate"]:
                            liftoff_counter += 1
                        else:
                            liftoff_counter = 0

                        if liftoff_counter >= 10:  # 1.0s persistence check
                            hover_throttle = pct
                            log.info(
                                f"LIFTOFF DETECTED at {pct}% | "
                                f"alt={alt:.2f}m climb={climb:.2f}m/s"
                            )
                            break
            if hover_throttle is not None:
                break

        if hover_throttle is None:
            hover_throttle = fc["known_hover_throttle_pct"]
            log.warning(f"Liftoff not detected — using hover throttle {hover_throttle}%")

        # ── PHASE 4: Hover ────────────────────────────────────
        target_alt = fc.get("target_altitude_m", fc.get("liftoff_alt_m", 0.10) + 0.15)
        kp = fc.get("hover_kp", 15.0)
        log.info(
            f"PHASE: Hover (Target: {target_alt:.2f}m) "
            f"using base {hover_throttle}% x {fc['hover_hold_sec']}s"
        )
        hover_throttles_list = []
        hover_altitudes_list = []

        for _ in range(int(fc["hover_hold_sec"] / 0.1)):
            if check_abort("hover"):
                abort_and_land(master, "hover", bench)
                return

            alt, climb = telem.get_vfr_hud()
            error = target_alt - (alt if alt is not None else 0.0)
            cmd_throttle = hover_throttle + kp * error
            cmd_throttle = int(max(15.0, min(60.0, cmd_throttle)))

            hover_throttles_list.append(cmd_throttle)
            if alt is not None:
                hover_altitudes_list.append(alt)

            send_throttle_safe(master, cmd_throttle, "hover", dt)
            update_and_log_framework("hover", cmd_throttle, dt, hm)
            time.sleep(0.1)

        # ── PHASE 5: Land Mode ────────────────────────────────
        log.info("PHASE: Initializing autonomous LAND mode...")
        # Clear RC overrides to return control to the autopilot
        master.mav.rc_channels_override_send(
            master.target_system, master.target_component,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        telem.set_mode(master, "LAND")

        log.info("Waiting for landing confirmation (disarm)...")
        land_start = time.time()
        while True:
            if check_abort("landing"):
                abort_and_land(master, "landing", bench)
                return

            update_and_log_framework("land", 0, dt, hm)

            # Check if disarmed
            with telem.state.lock:
                is_armed = telem.state.armed

            if not is_armed:
                log.info("Ground confirmed: Disarmed successfully.")
                break

            # Extra safety timeout: 5s for bench test (no propellers to trigger
            # landing detection), 30s for real flight
            timeout_limit = 5.0 if bench else 30.0
            if time.time() - land_start > timeout_limit:
                if bench:
                    log.info("Bench test landing timeout completed — disarming motors.")
                else:
                    log.warning("Landing timeout exceeded — force-disarming!")
                telem.disarm(master)
                break

            time.sleep(0.1)

        log.info("Flight sequence complete.")

        # Generate post-flight report
        generate_post_flight_report(flight_start, dt, hover_altitudes_list)

        # Save learned parameters
        if hover_throttle is not None and hover_throttles_list:
            learned_hover = int(sum(hover_throttles_list) / len(hover_throttles_list))
            save_learned_params(hover_throttle, learned_hover, fc.get("land_final_throttle", 10))

    except KeyboardInterrupt:
        log.warning("Ctrl+C — emergency landing.")
        abort_mission.set()
        abort_and_land(master, "user interrupt", bench)

    except Exception as e:
        log.error(f"Exception: {e}")
        abort_mission.set()
        abort_and_land(master, str(e), bench)
        raise

    finally:
        # Signal safety monitor thread to exit
        abort_mission.set()
        if 'stop_reader' in locals():
            stop_reader.set()
        if 'reader_thread' in locals():
            reader_thread.join(timeout=1.0)
        close_logger()


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/bench_test.yaml"
    run_flight(config_file)
