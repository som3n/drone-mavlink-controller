import time
import platform
import threading
import yaml

from pymavlink import mavutil
from drone_flight.logger import log, log_telemetry, close_logger
from drone_flight import telemetry as telem
from drone_flight.safety_monitor import safety_monitor, abort_mission


def load_config(path="config/bench_test.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def check_abort(phase):
    if abort_mission.is_set():
        from drone_flight.safety_monitor import abort_reason
        log.error(f"Abort detected during: {phase} | Reason: {abort_reason}")
        return True
    return False


def abort_and_land(master, reason):
    from drone_flight.safety_monitor import abort_reason
    full_reason = f"{reason} | {abort_reason}" if abort_reason else reason
    log.error(f"EMERGENCY LAND — {full_reason}")
    reason_lower = full_reason.lower()
    if "attitude" in reason_lower or "crash" in reason_lower or "imu" in reason_lower:
        log.warning("CRITICAL ATTITUDE/IMU SAFETY FAULT — DISARMING MOTORS IMMEDIATELY!")
        telem.send_rc_throttle(master, 0, "emergency")
        telem.disarm(master)
        return

    # Normal landing sequence for other reasons (e.g. telemetry timeout, user exit)
    for pct in range(int(telem.current_throttle), 0, -5):
        telem.send_rc_throttle(master, pct, "emergency")
        time.sleep(0.3)
    telem.send_rc_throttle(master, 0, "emergency")
    telem.disarm(master)


def send_throttle_safe(master, target_pct, phase):
    # Retrieve current attitude
    roll, pitch = telem.get_attitude()

    roll_cmd = 1500
    pitch_cmd = 1500

    # Proportional correction to actively level the drone if it tilts
    if roll is not None and pitch is not None:
        max_tilt = max(abs(roll), abs(pitch))
        # Only apply active leveling if the tilt is significant (>= 5.0 degrees)
        # to prevent motor unbalancing on the ground/bench during flat idle
        if max_tilt >= 5.0:
            # P-controller to counter tilt (gain Kp = 8.0 PWM units per degree)
            Kp = 8.0
            roll_cmd = 1500 - (Kp * roll)
            pitch_cmd = 1500 - (Kp * pitch)

            # Clamp overrides to safe range [1300, 1700] to prevent extreme inputs
            roll_cmd = max(1300, min(1700, int(roll_cmd)))
            pitch_cmd = max(1300, min(1700, int(pitch_cmd)))

            # Log a warning if the tilt is significant (>15 degrees)
            if max_tilt > 15.0:
                log.warning(f"  [TILT WARNING] roll={roll:.1f}° pitch={pitch:.1f}° | Applying correction: Roll={roll_cmd} Pitch={pitch_cmd}")

    telem.send_rc_throttle(master, target_pct, phase, roll=roll_cmd, pitch=pitch_cmd)


def run_flight(config_path="config/bench_test.yaml"):
    cfg = load_config(config_path)
    fc = cfg["flight"]
    bench = cfg["bench_test"]

    log.info(f"Platform: {platform.system()} | Mode: {'BENCH TEST' if bench else 'REAL FLIGHT'}")

    port = cfg["port"]["windows"] if platform.system() == "Windows" else cfg["port"]["linux"]
    baud = cfg["port"]["baud"]
    log.info(f"Connecting: {port} @ {baud}")

    master = mavutil.mavlink_connection(port, baud=baud)
    telem.current_throttle = 0.0

    try:
        log.info("Waiting for heartbeat...")
        master.wait_heartbeat()
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
            mavutil.mavlink.MAV_DATA_STREAM_POSITION, 10, 1  # VFR_HUD
        )
        time.sleep(1)

        # Check battery voltage if connected (both in bench and real flight)
        volts = telem.get_battery_voltage(master)
        if volts is not None and volts > 1.0:
            min_volts = fc.get("battery_min_voltage", 10.5)
            if volts < min_volts:
                log.error(f"Battery check failed: {volts:.2f}V (requires {min_volts}V)"); return
            log.info(f"Battery: {volts:.2f}V OK")
        else:
            log.info("No battery detected (or USB powered only) — skipping startup battery check.")

        telem.set_mode(master, "STABILIZE")
        telem.arm(master, bench)

        # Start telemetry reader thread
        stop_reader = threading.Event()
        reader_thread = threading.Thread(target=telem.start_telemetry_reader, args=(master, stop_reader), daemon=True)
        reader_thread.start()

        flight_start = time.time()

        monitor = threading.Thread(target=safety_monitor, args=(master, cfg, flight_start), daemon=True)
        monitor.start()

        # ── PHASE 1: Spin-up ──────────────────────────────────
        log.info("PHASE: Spin-up 10% x 5s")
        for _ in range(50):
            if check_abort("spinup"):
                abort_and_land(master, "spinup"); return
            send_throttle_safe(master, 10, "spinup")
            alt, climb = telem.get_vfr_hud()
            roll, pitch = telem.get_attitude()
            log_telemetry("spinup", 10, alt, climb, roll, pitch)
            time.sleep(0.1)

        # ── PHASE 2: Pre-lift stabilize ───────────────────────
        for pct, hold in [(15, 2.0), (20, 3.0)]:
            log.info(f"PHASE: Stabilize {pct}% x {hold}s")
            for _ in range(int(hold / 0.1)):
                if check_abort("stabilize"):
                    abort_and_land(master, "stabilize"); return
                send_throttle_safe(master, pct, "stabilize")
                alt, climb = telem.get_vfr_hud()
                roll, pitch = telem.get_attitude()
                log_telemetry("stabilize", pct, alt, climb, roll, pitch)
                time.sleep(0.1)

        # ── PHASE 3: Takeoff ramp ─────────────────────────────
        log.info("PHASE: Takeoff ramp")
        hover_throttle = None
        for pct, hold in cfg["takeoff_ramp"]:
            log.info(f"  Ramp {pct}% x {hold}s")
            for _ in range(int(hold / 0.1)):
                if check_abort("ramp"):
                    abort_and_land(master, "ramp"); return
                send_throttle_safe(master, pct, "ramp")
                alt, climb = telem.get_vfr_hud()
                roll, pitch = telem.get_attitude()
                log_telemetry("ramp", pct, alt, climb, roll, pitch)
                time.sleep(0.1)
                if hover_throttle is None and pct >= 30:
                    if alt is not None and climb is not None:
                        log.debug(f"  alt={alt:.3f}m climb={climb:.3f}m/s @ {pct}%")
                        if alt > fc["liftoff_alt_m"] and climb > fc["liftoff_climb_rate"]:
                            hover_throttle = pct
                            log.info(f"LIFTOFF DETECTED at {pct}% | alt={alt:.2f}m climb={climb:.2f}m/s")

        if hover_throttle is None:
            hover_throttle = fc["known_hover_throttle_pct"]
            log.warning(f"Liftoff not detected — using known hover throttle {hover_throttle}%")

        # ── PHASE 4: Hover ────────────────────────────────────
        log.info(f"PHASE: Hover {hover_throttle}% x {fc['hover_hold_sec']}s")
        for _ in range(int(fc["hover_hold_sec"] / 0.1)):
            if check_abort("hover"):
                abort_and_land(master, "hover"); return
            send_throttle_safe(master, hover_throttle, "hover")
            alt, climb = telem.get_vfr_hud()
            roll, pitch = telem.get_attitude()
            log_telemetry("hover", hover_throttle, alt, climb, roll, pitch)
            time.sleep(0.1)

        # ── PHASE 5: Descent ──────────────────────────────────
        log.info("PHASE: Descent")
        step = 1
        while True:
            if check_abort("descent"):
                abort_and_land(master, "descent"); return
            throttle = max(hover_throttle - (fc["descent_step_pct"] * step), 10)
            log.info(f"  Descent step {step}: {throttle}%")
            for _ in range(int(fc["descent_step_hold_sec"] / 0.1)):
                if check_abort("descent hold"):
                    abort_and_land(master, "descent hold"); return
                send_throttle_safe(master, throttle, "descent")
                alt, climb = telem.get_vfr_hud()
                roll, pitch = telem.get_attitude()
                log_telemetry("descent", throttle, alt, climb, roll, pitch)
                time.sleep(0.1)
            alt, _ = telem.get_vfr_hud()
            if (alt is not None and alt < fc["land_alt_threshold_m"]) or throttle <= 10:
                break
            step += 1

        # ── PHASE 6: Land ─────────────────────────────────────
        log.info("PHASE: Landing")
        for _ in range(50):
            if check_abort("land confirm"):
                abort_and_land(master, "land confirm"); return
            alt, climb = telem.get_vfr_hud()
            roll, pitch = telem.get_attitude()
            log_telemetry("land", fc["land_final_throttle"], alt, climb, roll, pitch)
            if alt is not None and climb is not None:
                if alt < fc["land_alt_threshold_m"] and abs(climb) < 0.05:
                    log.info(f"Ground confirmed: alt={alt:.2f}m climb={climb:.2f}m/s")
                    break
            time.sleep(0.1)

        telem.send_rc_throttle(master, fc["land_final_throttle"], "land")
        time.sleep(fc["land_final_hold_sec"])
        telem.send_rc_throttle(master, 0, "land")
        telem.disarm(master)
        log.info("Flight sequence complete.")

    except KeyboardInterrupt:
        log.warning("Ctrl+C — emergency landing.")
        abort_mission.set()
        abort_and_land(master, "user interrupt")

    except Exception as e:
        log.error(f"Exception: {e}")
        abort_mission.set()
        abort_and_land(master, str(e))
        raise

    finally:
        if 'stop_reader' in locals():
            stop_reader.set()
        if 'reader_thread' in locals():
            reader_thread.join(timeout=1.0)
        close_logger()


if __name__ == "__main__":
    run_flight()