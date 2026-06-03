import os
import pytest
from drone_flight.recovery_learning import (
    load_controller_learning,
    save_controller_learning,
    evaluate_recovery_performance,
    LEARNING_FILE,
    DATASET_FILE
)


@pytest.fixture(autouse=True)
def clean_database():
    """Ensure a clean/fresh database file for every test."""
    if os.path.exists(LEARNING_FILE):
        try:
            os.remove(LEARNING_FILE)
        except Exception:
            pass
    if os.path.exists(DATASET_FILE):
        try:
            os.remove(DATASET_FILE)
        except Exception:
            pass
    yield
    if os.path.exists(LEARNING_FILE):
        try:
            os.remove(LEARNING_FILE)
        except Exception:
            pass
    if os.path.exists(DATASET_FILE):
        try:
            os.remove(DATASET_FILE)
        except Exception:
            pass


def test_load_controller_learning_initial():
    data = load_controller_learning()
    assert "zone2" in data
    assert data["zone3"]["kp"] == 8.0
    assert data["zone4"]["kd"] == 2.5
    assert data["zone5"]["samples"] == 0


def test_save_controller_learning():
    data = load_controller_learning()
    data["zone3"]["kp"] = 9.5
    data["zone3"]["kd"] = 1.8
    data["zone3"]["samples"] = 4
    save_controller_learning(data)

    new_data = load_controller_learning()
    assert new_data["zone3"]["kp"] == 9.5
    assert new_data["zone3"]["kd"] == 1.8
    assert new_data["zone3"]["samples"] == 4


def test_slow_recovery_rule():
    # Rule 1: duration > 2.5s -> KP += 0.10
    event_data = {
        "zone": 3,
        "duration": 3.0,
        "overshoot": 1.0,
        "stability_score": 10.0,
        "authority_factor": 0.5,
        "success": True
    }

    res = evaluate_recovery_performance(event_data)
    assert abs(res["kp"] - 8.10) < 1e-3
    assert abs(res["kd"] - 1.5) < 1e-3

    # Check database persistence
    data = load_controller_learning()
    assert data["zone3"]["samples"] == 1
    assert abs(data["zone3"]["kp"] - 8.10) < 1e-3


def test_overshoot_rule():
    # Rule 2: overshoot > 5.0 -> KD += 0.05
    event_data = {
        "zone": 3,
        "duration": 0.8,
        "overshoot": 6.0,
        "stability_score": 10.0,
        "authority_factor": 0.5,
        "success": True
    }

    res = evaluate_recovery_performance(event_data)
    assert abs(res["kp"] - 8.0) < 1e-3
    assert abs(res["kd"] - 1.55) < 1e-3

    # Check database persistence
    data = load_controller_learning()
    assert data["zone3"]["samples"] == 1
    assert abs(data["zone3"]["kd"] - 1.55) < 1e-3


def test_oscillation_rule():
    # Rule 3: stability_score > 40 -> KP -= 0.05, KD += 0.05
    event_data = {
        "zone": 3,
        "duration": 0.8,
        "overshoot": 1.0,
        "stability_score": 45.0,
        "authority_factor": 0.5,
        "success": True
    }

    res = evaluate_recovery_performance(event_data)
    assert abs(res["kp"] - 7.95) < 1e-3
    assert abs(res["kd"] - 1.55) < 1e-3


def test_excellent_recovery_no_change():
    # Rule 4: excellent recovery -> no change
    event_data = {
        "zone": 3,
        "duration": 0.5,
        "overshoot": 1.5,
        "stability_score": 12.0,
        "authority_factor": 0.5,
        "success": True
    }

    res = evaluate_recovery_performance(event_data)
    assert abs(res["kp"] - 8.0) < 1e-3
    assert abs(res["kd"] - 1.5) < 1e-3


def test_gain_clamping():
    # Test min limits clamping (e.g. Zone 2)
    # Repeatedly trigger oscillation to reduce KP below KP_MIN (2.0)
    event_data = {
        "zone": 2,
        "duration": 0.5,
        "overshoot": 1.0,
        "stability_score": 90.0,
        "authority_factor": 0.25,
        "success": True
    }

    # Initial kp is 5.0. 90 triggers Rule 3 (KP -= 0.05, KD += 0.05).
    # We run it 70 times to drive KP down. It should clamp at 2.0.
    for _ in range(70):
        res = evaluate_recovery_performance(event_data)

    assert res["kp"] == 2.0
    assert res["kd"] == 4.5  # 1.0 + 70 * 0.05 = 4.5

    # Check max limits clamping
    # Run slow recovery to drive KP up above KP_MAX (25.0)
    slow_event = {
        "zone": 5,
        "duration": 4.0,
        "overshoot": 1.0,
        "stability_score": 10.0,
        "authority_factor": 1.0,
        "success": True
    }

    # Initial kp is 16.0. 4.0s triggers Rule 1 (KP += 0.10).
    # Run 100 times to exceed 25.0.
    for _ in range(100):
        res = evaluate_recovery_performance(slow_event)

    assert res["kp"] == 25.0


def test_bench_validation_no_learning():
    # If recoverable is False, learning must be skipped and gains unmodified
    event_data = {
        "zone": 3,
        "duration": 4.0,  # exceeds 2.5s -> would trigger Rule 1 (KP += 0.10)
        "overshoot": 6.0,  # exceeds 5.0 -> would trigger Rule 2 (KD += 0.05)
        "stability_score": 45.0,  # exceeds 40 -> would trigger Rule 3 (KP -= 0.05, KD += 0.05)
        "authority_factor": 0.5,
        "success": False,
        "recoverable": False
    }

    # Initial gains for Zone 3 are KP=8.0, KD=1.5
    res = evaluate_recovery_performance(event_data)
    assert res["kp"] == 8.0
    assert res["kd"] == 1.5
    assert res["learning_status"] == "SKIPPED"

    # Database values must remain unmodified
    data = load_controller_learning()
    assert data["zone3"]["kp"] == 8.0
    assert data["zone3"]["kd"] == 1.5
    assert data["zone3"]["samples"] == 0

    # Entry must be logged to CSV with success = "BENCH_VALIDATION"
    assert os.path.exists(DATASET_FILE)
    with open(DATASET_FILE, "r") as f:
        lines = f.readlines()
        # header + 1 row
        assert len(lines) >= 2
        last_row = lines[-1].strip().split(",")
        assert last_row[1] == "3"
        assert last_row[2] == "8.0"
        assert last_row[3] == "1.5"
        assert last_row[8] == "BENCH_VALIDATION"
