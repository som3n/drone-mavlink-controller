from drone_flight import telemetry as telem


def test_vehicle_state_initialization():
    state = telem.state
    with state.lock:
        state.armed = False
        state.alt = 0.0
        state.alt_offset = None
        state.climb = 0.0
        state.roll = 0.0
        state.pitch = 0.0
        state.voltage_battery = 0.0
        state.xacc = 0.0
        state.yacc = 0.0
        state.zacc = 0.0
    assert state.armed is False
    assert state.alt == 0.0
    assert state.alt_offset is None
    assert state.climb == 0.0
    assert state.roll == 0.0
    assert state.pitch == 0.0
    assert state.voltage_battery == 0.0
    assert state.xacc == 0.0
    assert state.yacc == 0.0
    assert state.zacc == 0.0


def test_getters_under_state_updates():
    state = telem.state

    with state.lock:
        state.alt = 10.5
        state.climb = 0.5
        state.roll = 2.3
        state.pitch = -1.4
        state.voltage_battery = 15.2
        state.xacc = 1.0
        state.yacc = 2.0
        state.zacc = 9.8
        state.last_msg_time = 1234567.89

    alt, climb = telem.get_vfr_hud()
    assert alt == 10.5
    assert climb == 0.5

    roll, pitch = telem.get_attitude()
    assert roll == 2.3
    assert pitch == -1.4

    assert telem.get_battery_voltage() == 15.2
    assert telem.get_last_msg_time() == 1234567.89

    x, y, z = telem.get_raw_imu()
    assert x == 1.0
    assert y == 2.0
    assert z == 9.8


def test_telemetry_reader_loop_global_position_int():
    class MockMessage:
        def __init__(self, type_name, **kwargs):
            self._type = type_name
            for k, v in kwargs.items():
                setattr(self, k, v)

        def get_type(self):
            return self._type

    import threading
    stop_event = threading.Event()

    class MockMaster:
        def __init__(self):
            self.messages = [
                # First sets offset to 1.5m
                MockMessage('GLOBAL_POSITION_INT', relative_alt=1500),
                # Updates relative alt to 2.5 - 1.5 = 1.0m
                MockMessage('GLOBAL_POSITION_INT', relative_alt=2500),
                MockMessage('VFR_HUD', climb=0.25)
            ]

        def recv_match(self, blocking=True, timeout=0.1):
            if self.messages:
                return self.messages.pop(0)
            stop_event.set()
            return None

    # Reset state to default values before test
    with telem.state.lock:
        telem.state.alt = 0.0
        telem.state.alt_offset = None
        telem.state.climb = 0.0

    master = MockMaster()
    telem.telemetry_reader_loop(master, stop_event)

    alt, climb = telem.get_vfr_hud()
    assert alt == 1.0
    assert climb == 0.25


def test_telemetry_reader_loop_consecutive_errors():
    import threading
    stop_event = threading.Event()

    class FailingMaster:
        def recv_match(self, blocking=True, timeout=0.1):
            raise IOError("Connection lost")

    from drone_flight.safety_monitor import abort_mission
    abort_mission.clear()

    master = FailingMaster()
    telem.telemetry_reader_loop(master, stop_event)

    assert abort_mission.is_set()
