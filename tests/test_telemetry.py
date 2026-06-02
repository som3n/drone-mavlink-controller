import time
from drone_flight import telemetry as telem

def test_vehicle_state_initialization():
    state = telem.state
    assert state.armed is False
    assert state.alt == 0.0
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
