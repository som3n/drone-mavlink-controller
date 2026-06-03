from drone_flight.telemetry import pct_to_pwm


def test_zero():
    assert pct_to_pwm(0) == 1100


def test_full():
    assert pct_to_pwm(100) == 1900


def test_half():
    assert pct_to_pwm(50) == 1500


def test_clamp_low():
    assert pct_to_pwm(-10) == 1100


def test_clamp_high():
    assert pct_to_pwm(110) == 1900
