import time
from drone_flight.digital_twin import DigitalTwin


def test_altitude_prediction():
    dt = DigitalTwin(buffer_duration_sec=2.0, update_rate_hz=10)
    # Update state: current altitude 10.0m, climb rate 2.0m/s
    dt.update(alt=10.0, climb=2.0, voltage=0.0, roll=0.0, pitch=0.0,
              xacc=0.0, yacc=0.0, zacc=1.0)

    # 1s future altitude prediction: 10 + 2*1 = 12m
    assert dt.predict_future_altitude(1.0) == 12.0
    # 3s future altitude prediction: 10 + 2*3 = 16m
    assert dt.predict_future_altitude(3.0) == 16.0


def test_battery_prediction_usb():
    dt = DigitalTwin()
    # USB-only mode (voltage <= 1.0)
    dt.update(alt=0.0, climb=0.0, voltage=0.5, roll=0.0, pitch=0.0,
              xacc=0.0, yacc=0.0, zacc=1.0)
    res = dt.predict_battery_state()
    assert res["voltage_drop_rate"] == 0.0
    assert res["remaining_flight_time_sec"] == 999.0 * 60.0


def test_battery_prediction_discharging():
    dt = DigitalTwin()
    t_start = time.time()

    # Simulate voltage drop from 12.5V down to 11.5V over 10 seconds
    for i in range(11):
        t = t_start + i * 1.0
        v = 12.5 - i * 0.1  # drop rate 0.1 V/s
        dt.history["time"].append(t)
        dt.history["voltage"].append(v)
        dt.history["alt"].append(0.0)
        dt.history["climb"].append(0.0)
        dt.history["roll"].append(0.0)
        dt.history["pitch"].append(0.0)
        dt.history["xacc"].append(0.0)
        dt.history["yacc"].append(0.0)
        dt.history["zacc"].append(1.0)

    res = dt.predict_battery_state()
    # Drop rate should be ~0.1 Volts/sec
    assert abs(res["voltage_drop_rate"] - 0.1) < 1e-3
    # Critical voltage is 10.5V, current is 11.5V, so remaining time is (11.5 - 10.5) / 0.1 = 10s
    assert abs(res["remaining_flight_time_sec"] - 10.0) < 0.2


def test_stability_evaluation():
    dt = DigitalTwin(buffer_duration_sec=2.0, update_rate_hz=5)

    # 1. Constant stable attitude
    for _ in range(10):
        dt.update(alt=0.0, climb=0.0, voltage=0.0, roll=0.0, pitch=0.0,
                  xacc=0.0, yacc=0.0, zacc=1.0)
    res = dt.evaluate_stability_trends()
    assert res["drift_deg"] == 0.0
    assert res["oscillation_growing"] is False

    # 2. Growing oscillations
    # First half: stable
    for _ in range(5):
        dt.update(alt=0.0, climb=0.0, voltage=0.0, roll=1.0, pitch=-1.0,
                  xacc=0.0, yacc=0.0, zacc=1.0)
    # Second half: wild oscillations
    for i in range(5):
        roll_val = 15.0 if i % 2 == 0 else -15.0
        dt.update(alt=0.0, climb=0.0, voltage=0.0, roll=roll_val, pitch=-roll_val,
                  xacc=0.0, yacc=0.0, zacc=1.0)

    res = dt.evaluate_stability_trends()
    assert res["oscillation_growing"] is True


def test_risk_score():
    dt = DigitalTwin()
    # Safe condition: 10 * 0.4 + 20 * 0.6 = 16.0
    risk = dt.get_flight_risk_score(attitude_error_score=10.0, stability_score=20.0)
    assert abs(risk - 16.0) < 1e-3

    # Critical condition: 80 * 0.4 + 90 * 0.6 = 86.0
    risk = dt.get_flight_risk_score(attitude_error_score=80.0, stability_score=90.0)
    assert abs(risk - 86.0) < 1e-3
