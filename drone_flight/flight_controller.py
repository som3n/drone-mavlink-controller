import time
import platform
import threading
import yaml
import json
import os
import sys
from datetime import datetime

# Add the project root directory to sys.path to allow running the script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pymavlink import mavutil
from drone_flight.logger import log, log_telemetry, close_logger
from drone_flight import telemetry as telem
from drone_flight.safety_monitor import safety_monitor, abort_mission
from drone_flight.digital_twin import DigitalTwin
from drone_flight.health_monitor import HealthMonitor, std_dev
from drone_flight.recovery_learning import load_controller_learning, evaluate_recovery_performance


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
        att_err_score = telem.state.attitude_error_score
        flight_risk = telem.state.flight_risk
        pred_roll_err_1s = getattr(telem.state, "pred_roll_err_1s", 0.0)
        pred_pitch_err_1s = getattr(telem.state, "pred_pitch_err_1s", 0.0)
        pred_stability_2s = getattr(telem.state, "pred_stability_2s", 0.0)
        pred_success_prob = getattr(telem.state, "pred_success_prob", 100.0)
        auth_factor = getattr(telem.state, "auth_factor", 0.0)
        kp_scaled = getattr(telem.state, "kp_scaled", 8.0)
        kd_scaled = getattr(telem.state, "kd_scaled", 1.5)

    risk_score = dt.get_flight_risk_score(att_err_score, stability_score)

    log_telemetry(
        phase, throttle, alt, climb, roll, pitch,
        health=health, anomaly_score=anomaly_score, risk_score=risk_score,
        pred_alt_1s=pred_alt_1s, pred_alt_3s=pred_alt_3s, pred_alt_5s=pred_alt_5s,
        pred_volt_rem=pred_volt_rem,
        target_roll=target_roll, roll_err=roll_err,
        target_pitch=target_pitch, pitch_err=pitch_err,
        roll_rate=roll_rate, pitch_rate=pitch_rate,
        roll_corr=roll_corr, pitch_corr=pitch_corr,
        stability_score=stability_score, recovery_state=recovery_state,
        att_err_score=att_err_score, flight_risk=flight_risk,
        pred_roll_err_1s=pred_roll_err_1s, pred_pitch_err_1s=pred_pitch_err_1s,
        pred_stability_2s=pred_stability_2s, pred_success_prob=pred_success_prob,
        auth_factor=auth_factor, kp_scaled=kp_scaled, kd_scaled=kd_scaled
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

    with telem.state.lock:
        summaries = list(telem.state.recovery_analytics_summary)
        active_event = telem.state.active_recovery_event

    if active_event is not None:
        event_duration = time.time() - active_event["start_time"]
        summary = (
            f"Recovery Event #{active_event['event_number']} | "
            f"Peak Roll Error: {active_event['peak_roll_err']:.1f}° | "
            f"Peak Pitch Error: {active_event['peak_pitch_err']:.1f}° | "
            f"Recovery Duration: {event_duration:.2f}s | "
            f"Result: FAILED (End of Flight)"
        )
        summaries.append(summary)

    recovery_section = ""
    if summaries:
        recovery_section += "Recovery Events Summary:\n"
        for line in summaries:
            recovery_section += f"  - {line}\n"
    else:
        recovery_section += "Recovery Events Summary: None\n"

    learning_data = load_controller_learning()
    learning_report = (
        f"==================================================\n"
        f"SELF-LEARNING CONTROLLER REPORT\n"
        f"==================================================\n"
        f"Zone 2\n"
        f"KP: {learning_data['zone2']['kp']:.3f}\n"
        f"KD: {learning_data['zone2']['kd']:.3f}\n"
        f"Samples: {learning_data['zone2']['samples']}\n\n"
        f"Zone 3\n"
        f"KP: {learning_data['zone3']['kp']:.3f}\n"
        f"KD: {learning_data['zone3']['kd']:.3f}\n"
        f"Samples: {learning_data['zone3']['samples']}\n\n"
        f"Zone 4\n"
        f"KP: {learning_data['zone4']['kp']:.3f}\n"
        f"KD: {learning_data['zone4']['kd']:.3f}\n"
        f"Samples: {learning_data['zone4']['samples']}\n\n"
        f"Zone 5\n"
        f"KP: {learning_data['zone5']['kp']:.3f}\n"
        f"KD: {learning_data['zone5']['kd']:.3f}\n"
        f"Samples: {learning_data['zone5']['samples']}\n"
    )

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
        f"--------------------------------------------------\n"
        f"{recovery_section}"
        f"{learning_report}"
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


def calculate_attitude_recovery(target_roll, target_pitch, dt, bench=False):
    roll, pitch = telem.get_attitude()
    roll_rate, pitch_rate = telem.get_attitude_rates()

    if roll is None or pitch is None or roll_rate is None or pitch_rate is None:
        return 1500, 1500, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "NORMAL"

    # 1. Error calculation
    roll_err = target_roll - roll
    pitch_err = target_pitch - pitch
    max_err = max(abs(roll_err), abs(pitch_err))

    # 2. Separate Attitude Error Score (0-100)
    attitude_error = abs(roll_err) + abs(pitch_err)
    att_err_score = min(100.0, (attitude_error / 45.0) * 100.0)

    # 3. Stability Score Calculation (0 to 100)
    rolls = dt.history["roll"]
    pitches = dt.history["pitch"]

    def calculate_variance(data):
        if len(data) < 2:
            return 0.0
        avg = sum(data) / len(data)
        return sum((x - avg) ** 2 for x in data) / len(data)

    roll_var = calculate_variance(rolls)
    pitch_var = calculate_variance(pitches)

    roll_var_score = min(30.0, (roll_var / 5.0) * 30.0)
    pitch_var_score = min(30.0, (pitch_var / 5.0) * 30.0)

    max_rate = max(abs(roll_rate), abs(pitch_rate))
    rate_score = min(40.0, (max_rate / 90.0) * 40.0)

    stability_score = roll_var_score + pitch_var_score + rate_score
    stability_score = min(100.0, max(0.0, stability_score))

    # 4. Composite Flight Risk Engine
    flight_risk = (att_err_score * 0.4) + (stability_score * 0.6)
    flight_risk = min(100.0, max(0.0, flight_risk))

    # 5. Digital Twin Predictive Recovery
    pred = dt.predict_recovery_outcome(
        roll_err, pitch_err, roll_rate, pitch_rate, stability_score
    )
    pred_roll_err_1s = pred["pred_roll_err_1s"]
    pred_pitch_err_1s = pred["pred_pitch_err_1s"]
    pred_stability_2s = pred["pred_stability_2s"]
    pred_success_prob = pred["recovery_success_prob"]

    # Load learned controller gains from VehicleState cache
    with telem.state.lock:
        learning_data = telem.state.controller_learning
    if learning_data is None:
        learning_data = load_controller_learning()
        with telem.state.lock:
            telem.state.controller_learning = learning_data

    # 6. Adaptive Gain Scheduling & Authority Levels
    if max_err <= 5.0:
        auth_factor = 0.0
        KP = 0.0
        KD = 0.0
        recovery_state = "NORMAL"
        event_zone = 1
    elif max_err <= 10.0:
        auth_factor = 0.25
        KP = learning_data["zone2"]["kp"]
        KD = learning_data["zone2"]["kd"]
        recovery_state = "MINOR CORRECTION"
        event_zone = 2
    elif max_err <= 20.0:
        auth_factor = 0.50
        KP = learning_data["zone3"]["kp"]
        KD = learning_data["zone3"]["kd"]
        recovery_state = "ACTIVE RECOVERY"
        event_zone = 3
    elif max_err <= 35.0:
        auth_factor = 0.75
        KP = learning_data["zone4"]["kp"]
        KD = learning_data["zone4"]["kd"]
        recovery_state = "HIGH RECOVERY"
        event_zone = 4
    else:
        auth_factor = 1.00
        KP = learning_data["zone5"]["kp"]
        KD = learning_data["zone5"]["kd"]
        recovery_state = "RECOVERY MODE"
        event_zone = 5

    # Predictive boost if success probability is below 50%
    if pred_success_prob < 50.0 and max_err > 5.0:
        KP *= 1.2
        KD *= 1.1

    # 7. Recovery Authority Manager Scaling
    roll_output = KP * roll_err - KD * roll_rate
    pitch_output = KP * pitch_err - KD * pitch_rate

    roll_corr = roll_output * auth_factor
    pitch_corr = pitch_output * auth_factor

    # Clamp corrections to safe range [-200, 200]
    roll_corr = max(-200.0, min(200.0, roll_corr))
    pitch_corr = max(-200.0, min(200.0, pitch_corr))

    # 8. Recovery Output Rate Limiter (±20 max change per iteration)
    with telem.state.lock:
        prev_roll_cmd = telem.state.prev_roll_cmd
        prev_pitch_cmd = telem.state.prev_pitch_cmd

    desired_roll_pwm = int(1500 + roll_corr)
    desired_pitch_pwm = int(1500 + pitch_corr)

    roll_delta = desired_roll_pwm - prev_roll_cmd
    roll_delta = max(-20.0, min(20.0, roll_delta))
    roll_cmd = int(prev_roll_cmd + roll_delta)

    pitch_delta = desired_pitch_pwm - prev_pitch_cmd
    pitch_delta = max(-20.0, min(20.0, pitch_delta))
    pitch_cmd = int(prev_pitch_cmd + pitch_delta)

    # 9. Recovery Performance Analytics Tracking & Gain Adaptation
    now_time = time.time()
    with telem.state.lock:
        active_event = telem.state.active_recovery_event

        if max_err > 5.0:
            if active_event is None:
                # Start new event
                active_event = {
                    "start_time": now_time,
                    "max_zone": event_zone,
                    "peak_roll_err": abs(roll_err),
                    "peak_pitch_err": abs(pitch_err),
                    "event_number": telem.state.recovery_events_count + 1,
                    "has_crossed_target": False,
                    "overshoot": 0.0,
                    "stability_scores": [stability_score],
                    "authority_factors": [auth_factor],
                    "stable_ticks": 0
                }
                telem.state.recovery_events_count += 1
                telem.state.active_recovery_event = active_event
                log.info(
                    f"  [RECOVERY ANALYTICS] Started Recovery Event "
                    f"#{active_event['event_number']} for Zone {event_zone}"
                )
            else:
                active_event["max_zone"] = max(active_event["max_zone"], event_zone)
                active_event["peak_roll_err"] = max(
                    active_event["peak_roll_err"], abs(roll_err)
                )
                active_event["peak_pitch_err"] = max(
                    active_event["peak_pitch_err"], abs(pitch_err)
                )
                active_event["stability_scores"].append(stability_score)
                active_event["authority_factors"].append(auth_factor)
                active_event["stable_ticks"] = 0  # reset stable window

                # Track overshoot if target has been crossed
                if not active_event["has_crossed_target"]:
                    if abs(roll_err) <= 5.0 and abs(pitch_err) <= 5.0:
                        active_event["has_crossed_target"] = True
                else:
                    curr_err = max(abs(roll_err), abs(pitch_err))
                    active_event["overshoot"] = max(active_event["overshoot"], curr_err)
        else:
            if active_event is not None:
                active_event["stable_ticks"] += 1
                active_event["stability_scores"].append(stability_score)
                active_event["authority_factors"].append(auth_factor)

                # Track overshoot if target crossed
                if not active_event["has_crossed_target"]:
                    active_event["has_crossed_target"] = True
                else:
                    curr_err = max(abs(roll_err), abs(pitch_err))
                    active_event["overshoot"] = max(active_event["overshoot"], curr_err)

                # Event ends successfully when attitude remains in Zone 1 for 5 ticks (0.5s)
                if active_event["stable_ticks"] >= 5:
                    duration = now_time - active_event["start_time"]
                    avg_stability = sum(active_event["stability_scores"]) / len(active_event["stability_scores"])
                    avg_auth = sum(active_event["authority_factors"]) / len(active_event["authority_factors"])

                    event_data = {
                        "zone": active_event["max_zone"],
                        "duration": duration,
                        "peak_roll_error": active_event["peak_roll_err"],
                        "peak_pitch_error": active_event["peak_pitch_err"],
                        "overshoot": active_event["overshoot"],
                        "stability_score": avg_stability,
                        "authority_factor": avg_auth,
                        "success": True,
                        "recoverable": not bench
                    }

                    # Trigger learning evaluation and update gains in cache
                    res = evaluate_recovery_performance(event_data)
                    telem.state.controller_learning = load_controller_learning()

                    if not event_data["recoverable"]:
                        result_str = "BENCH_VALIDATION"
                        learning_info = " | Learning: SKIPPED"
                    else:
                        result_str = f"SUCCESS (Score: {res['recovery_score']:.1f}, {res['quality']})"
                        learning_info = ""

                    summary = (
                        f"Recovery Event #{active_event['event_number']} | "
                        f"Peak Roll Error: {event_data['peak_roll_error']:.1f}° | "
                        f"Peak Pitch Error: {event_data['peak_pitch_error']:.1f}° | "
                        f"Overshoot: {event_data['overshoot']:.1f}° | "
                        f"Duration: {duration:.2f}s | "
                        f"Result: {result_str}{learning_info}"
                    )
                    telem.state.recovery_analytics_summary.append(summary)
                    log.info(f"  [RECOVERY ANALYTICS] {summary}")
                    telem.state.active_recovery_event = None
                    active_event = None

        # Check for event timeout (FAILED if duration exceeds 5.0 seconds)
        if active_event is not None:
            duration = now_time - active_event["start_time"]
            if duration > 5.0:
                avg_stability = sum(active_event["stability_scores"]) / len(active_event["stability_scores"])
                avg_auth = sum(active_event["authority_factors"]) / len(active_event["authority_factors"])

                event_data = {
                    "zone": active_event["max_zone"],
                    "duration": duration,
                    "peak_roll_error": active_event["peak_roll_err"],
                    "peak_pitch_error": active_event["peak_pitch_err"],
                    "overshoot": active_event["overshoot"],
                    "stability_score": avg_stability,
                    "authority_factor": avg_auth,
                    "success": False,
                    "recoverable": not bench
                }

                res = evaluate_recovery_performance(event_data)
                telem.state.controller_learning = load_controller_learning()

                if not event_data["recoverable"]:
                    result_str = "BENCH_VALIDATION"
                    learning_info = " | Learning: SKIPPED"
                else:
                    result_str = f"FAILED (Score: {res['recovery_score']:.1f})"
                    learning_info = ""

                summary = (
                    f"Recovery Event #{active_event['event_number']} | "
                    f"Peak Roll Error: {event_data['peak_roll_error']:.1f}° | "
                    f"Peak Pitch Error: {event_data['peak_pitch_error']:.1f}° | "
                    f"Overshoot: {event_data['overshoot']:.1f}° | "
                    f"Duration: {duration:.2f}s | "
                    f"Result: {result_str}{learning_info}"
                )
                telem.state.recovery_analytics_summary.append(summary)
                log.warning(f"  [RECOVERY ANALYTICS] {summary}")
                telem.state.active_recovery_event = None

    # Update state variables
    with telem.state.lock:
        telem.state.stability_score = stability_score
        telem.state.attitude_error_score = att_err_score
        telem.state.flight_risk = flight_risk
        telem.state.recovery_state = recovery_state
        telem.state.prev_roll_cmd = roll_cmd
        telem.state.prev_pitch_cmd = pitch_cmd
        # Save prediction variables for logging
        telem.state.pred_roll_err_1s = pred_roll_err_1s
        telem.state.pred_pitch_err_1s = pred_pitch_err_1s
        telem.state.pred_stability_2s = pred_stability_2s
        telem.state.pred_success_prob = pred_success_prob
        telem.state.auth_factor = auth_factor
        telem.state.kp_scaled = KP
        telem.state.kd_scaled = KD

    return (
        roll_cmd, pitch_cmd,
        roll_err, pitch_err,
        roll_rate, pitch_rate,
        roll_corr, pitch_corr,
        stability_score, recovery_state
    )


def send_throttle_safe(master, target_pct, phase, dt, bench=False):
    # 1. Throttle Output Rate Limiter (Smooth Ramping)
    current = telem.current_throttle
    max_throttle_step = 1.5  # Max change ±1.5% per iteration (15% per second)
    delta = target_pct - current
    delta = max(-max_throttle_step, min(max_throttle_step, delta))
    actual_pct = current + delta

    target_roll, target_pitch = telem.get_target_attitude()

    (
        roll_cmd, pitch_cmd,
        roll_err, pitch_err,
        roll_rate, pitch_rate,
        roll_corr, pitch_corr,
        stability_score, recovery_state
    ) = calculate_attitude_recovery(target_roll, target_pitch, dt, bench=bench)

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

    # Use the smooth actual_pct instead of raw target_pct
    telem.send_rc_throttle(master, actual_pct, phase, roll=roll_cmd, pitch=pitch_cmd)


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

    # Load learned controller gains
    learning_data = load_controller_learning()
    with telem.state.lock:
        telem.state.controller_learning = learning_data

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
            send_throttle_safe(master, 10, "spinup", dt, bench=bench)
            update_and_log_framework("spinup", 10, dt, hm)
            time.sleep(0.1)

        # ── PHASE 2: Pre-lift stabilize ───────────────────────
        for pct, hold in [(15, 2.0), (20, 3.0)]:
            log.info(f"PHASE: Stabilize {pct}% x {hold}s")
            for _ in range(int(hold / 0.1)):
                if check_abort("stabilize"):
                    abort_and_land(master, "stabilize", bench)
                    return
                send_throttle_safe(master, pct, "stabilize", dt, bench=bench)
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
                send_throttle_safe(master, pct, "ramp", dt, bench=bench)
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

            send_throttle_safe(master, cmd_throttle, "hover", dt, bench=bench)
            update_and_log_framework("hover", cmd_throttle, dt, hm)
            time.sleep(0.1)

        # Check if recovery is active before transitioning to Land Mode
        with telem.state.lock:
            active_recovery = telem.state.active_recovery_event is not None

        if active_recovery:
            log.warning("PHASE transition delayed: active attitude recovery in progress. Holding hover altitude...")
            recovery_wait_start = time.time()
            while True:
                with telem.state.lock:
                    active_recovery = telem.state.active_recovery_event is not None

                if not active_recovery:
                    log.info("Attitude recovery complete. Proceeding to Land Mode.")
                    break

                if time.time() - recovery_wait_start > 5.0:
                    log.warning("Attitude recovery wait timed out. Proceeding to Land Mode.")
                    break

                if check_abort("recovery_wait"):
                    abort_and_land(master, "recovery_wait", bench)
                    return

                # Continue sending recovery commands during wait
                alt, climb = telem.get_vfr_hud()
                error = target_alt - (alt if alt is not None else 0.0)
                cmd_throttle = hover_throttle + kp * error
                cmd_throttle = int(max(15.0, min(60.0, cmd_throttle)))

                send_throttle_safe(master, cmd_throttle, "recovery_hold", dt, bench=bench)
                update_and_log_framework("recovery_hold", cmd_throttle, dt, hm)
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
    import os
    import sys
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if len(sys.argv) > 1:
        # User specified custom config path, resolve it relative to current CWD
        config_file = os.path.abspath(sys.argv[1])
    else:
        # Use default config file under the project root
        config_file = os.path.join(project_root, "config", "bench_test.yaml")
        
    # Change working directory to project root to keep logs/ and config/ references consistent
    os.chdir(project_root)
    
    run_flight(config_file)
