import time
import platform
import threading
import yaml
import math

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
    if roll is not None and pitch is not None and not (math.isnan(roll) or math.isnan(pitch)):
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
                log.warning(
                    f"  [TILT WARNING] roll={roll:.1f}° pitch={pitch:.1f}° "
                    f"| Applying correction: Roll={roll_cmd} Pitch={pitch_cmd}"
                )

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
            args=(master, cfg, flight_start),
            daemon=True
        )
        monitor.start()

        # ── PHASE 1: Spin-up ──────────────────────────────────
        log.info("PHASE: Spin-up 10% x 5s")
        for _ in range(50):
            if check_abort("spinup"):
                abort_and_land(master, "spinup")
                return
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
                    abort_and_land(master, "stabilize")
                    return
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
                    abort_and_land(master, "ramp")
                    return
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
                            log.info(
                                f"LIFTOFF DETECTED at {pct}% | "
                                f"alt={alt:.2f}m climb={climb:.2f}m/s"
                            )

        if hover_throttle is None:
            hover_throttle = fc["known_hover_throttle_pct"]
            log.warning(f"Liftoff not detected — using known hover throttle {hover_throttle}%")

        # ── PHASE 4: Hover ────────────────────────────────────
        log.info(f"PHASE: Hover {hover_throttle}% x {fc['hover_hold_sec']}s")
        for _ in range(int(fc["hover_hold_sec"] / 0.1)):
            if check_abort("hover"):
                abort_and_land(master, "hover")
                return
            send_throttle_safe(master, hover_throttle, "hover")
            alt, climb = telem.get_vfr_hud()
            roll, pitch = telem.get_attitude()
            log_telemetry("hover", hover_throttle, alt, climb, roll, pitch)
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
                abort_and_land(master, "landing")
                return

            # Read and log telemetry
            alt, climb = telem.get_vfr_hud()
            roll, pitch = telem.get_attitude()
            log_telemetry("land", 0, alt, climb, roll, pitch)

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
