import time
import math
import threading
from pymavlink import mavutil
from drone_flight.logger import log, log_telemetry

current_throttle = 0.0

class VehicleState:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_msg_time = time.time()
        self.armed = False
        self.mode_id = None
        self.alt = 0.0
        self.climb = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.voltage_battery = 0.0
        self.xacc = 0.0
        self.yacc = 0.0
        self.zacc = 0.0
        self.heartbeat_received = False

state = VehicleState()


RC3_MIN = 1100
RC3_MAX = 1900

def pct_to_pwm(pct: float) -> int:
    return int(RC3_MIN + (max(0.0, min(100.0, pct)) / 100.0) * (RC3_MAX - RC3_MIN))


def send_rc_throttle(master, pct: float, phase="unknown", roll=1500, pitch=1500, yaw=1500):
    global current_throttle
    current_throttle = pct
    # Override all 4 primary channels (roll, pitch, throttle, yaw) to prevent RC failsafe
    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        int(roll),
        int(pitch),
        pct_to_pwm(pct),
        int(yaw),
        65535, 65535, 65535, 65535
    )


def telemetry_reader_loop(master, stop_event):
    global state
    log.info("Telemetry reader thread started.")
    while not stop_event.is_set():
        try:
            msg = master.recv_match(blocking=True, timeout=0.1)
            if msg is None:
                continue

            msg_type = msg.get_type()
            now = time.time()

            with state.lock:
                state.last_msg_time = now

                if msg_type == 'HEARTBEAT':
                    state.heartbeat_received = True
                    state.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    state.mode_id = msg.custom_mode

                elif msg_type == 'VFR_HUD':
                    state.alt = msg.alt
                    state.climb = msg.climb

                elif msg_type == 'ATTITUDE':
                    state.roll = math.degrees(msg.roll)
                    state.pitch = math.degrees(msg.pitch)

                elif msg_type == 'SYS_STATUS':
                    state.voltage_battery = msg.voltage_battery / 1000.0

                elif msg_type == 'RAW_IMU':
                    state.xacc = abs(msg.xacc) / 1000.0
                    state.yacc = abs(msg.yacc) / 1000.0
                    state.zacc = abs(msg.zacc) / 1000.0

                elif msg_type == 'STATUSTEXT':
                    log.warning(f"DRONE MSG: {msg.text}")

        except Exception as e:
            if not stop_event.is_set():
                log.error(f"Telemetry reader error: {e}")
            time.sleep(0.5)
    log.info("Telemetry reader thread exited.")


def start_telemetry_reader(master, stop_event):
    with state.lock:
        state.last_msg_time = time.time()
    telemetry_reader_loop(master, stop_event)


def get_last_msg_time():
    with state.lock:
        return state.last_msg_time


def get_battery_voltage(master=None, timeout=5):
    with state.lock:
        if state.voltage_battery > 0:
            return state.voltage_battery
    # Fallback if reader thread not running yet
    if master is not None:
        msg = master.recv_match(type='SYS_STATUS', blocking=True, timeout=timeout)
        if msg and msg.voltage_battery > 0:
            return msg.voltage_battery / 1000.0
    return None


def get_ekf_status(master, timeout=5):
    return master.recv_match(type='EKF_STATUS_REPORT', blocking=True, timeout=timeout)


def get_vfr_hud(master=None):
    with state.lock:
        return state.alt, state.climb


def get_attitude(master=None):
    with state.lock:
        return state.roll, state.pitch


def get_raw_imu():
    with state.lock:
        return state.xacc, state.yacc, state.zacc


def set_mode(master, mode_name: str):
    mode_id = master.mode_mapping().get(mode_name)
    if mode_id is None:
        raise ValueError(f"Unknown mode: {mode_name}")
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    for _ in range(20):
        ack = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if ack and ack.custom_mode == mode_id:
            log.info(f"Mode {mode_name} confirmed.")
            return
    log.warning(f"Mode {mode_name} not confirmed — continuing.")


def arm(master, bench_test: bool):
    if bench_test:
        log.info("Force-arming (bench test)...")
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 21196, 0, 0, 0, 0, 0
        )
        time.sleep(2)
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
    else:
        log.info("Arming...")
        master.arducopter_arm()

    log.info("Waiting for armed state...")
    time.sleep(2)
    for _ in range(30):
        hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if hb and (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            log.info("Armed successfully.")
            return
        time.sleep(0.2)
    log.warning("Arm confirmation not received — continuing anyway.")


def disarm(master):
    log.info("Disarming...")
    try:
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        time.sleep(1)
        log.info("Disarm command sent.")
    except Exception as e:
        log.warning(f"Disarm error: {e}")