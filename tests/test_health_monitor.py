import time
from drone_flight.digital_twin import DigitalTwin
from drone_flight.health_monitor import HealthMonitor


def test_health_monitor_usb_mode():
    dt = DigitalTwin()
    hm = HealthMonitor()

    # USB only telemetry (voltage = 0.5V <= 1.0V)
    dt.update(alt=0.0, climb=0.0, voltage=0.5, roll=0.0, pitch=0.0,
              xacc=0.0, yacc=0.0, zacc=1.0)

    scores = hm.calculate_health_scores(dt)
    # Battery health must be 100% on USB mode as requested
    assert scores["battery"] == 100.0
    # Everything is stable, so overall health should be high
    assert scores["overall"] >= 95.0


def test_health_monitor_battery_depleting():
    dt = DigitalTwin()
    hm = HealthMonitor()
    t_start = time.time()

    # Active battery discharging (voltage drops below critical 10.5V)
    dt.crit_voltage = 10.5
    for i in range(10):
        dt.history["time"].append(t_start + i * 1.0)
        dt.history["voltage"].append(11.0 - i * 0.1)  # drops from 11.0 to 10.1V
        dt.history["alt"].append(0.0)
        dt.history["climb"].append(0.0)
        dt.history["roll"].append(0.0)
        dt.history["pitch"].append(0.0)
        dt.history["xacc"].append(0.0)
        dt.history["yacc"].append(0.0)
        dt.history["zacc"].append(1.0)

    scores = hm.calculate_health_scores(dt)
    assert scores["battery"] == 0.0
    # Battery failure drops overall score
    assert scores["overall"] < 80.0


def test_communication_health_latency():
    dt = DigitalTwin()
    hm = HealthMonitor()

    # Force a latency of 3.0 seconds
    now = time.time()
    dt.history["time"].append(now - 3.0)
    dt.history["voltage"].append(0.5)
    dt.history["alt"].append(0.0)
    dt.history["climb"].append(0.0)
    dt.history["roll"].append(0.0)
    dt.history["pitch"].append(0.0)
    dt.history["xacc"].append(0.0)
    dt.history["yacc"].append(0.0)
    dt.history["zacc"].append(1.0)

    scores = hm.calculate_health_scores(dt)
    # Communication health should be reduced significantly
    assert scores["communication"] < 60.0


def test_anomaly_detection_attitude_oscillations():
    dt = DigitalTwin(buffer_duration_sec=2.0, update_rate_hz=5)
    hm = HealthMonitor()

    # High attitude oscillations
    for i in range(10):
        roll_val = 20.0 if i % 2 == 0 else -20.0
        dt.update(alt=0.0, climb=0.0, voltage=0.5, roll=roll_val, pitch=0.0,
                  xacc=0.0, yacc=0.0, zacc=1.0)

    anomalies = hm.detect_anomalies(dt)
    assert anomalies["is_anomaly"] is True
    assert "high attitude oscillation" in anomalies["reasons"]
    assert anomalies["anomaly_score"] >= 40.0
