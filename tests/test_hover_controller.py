# Unit test for Altitude-based Hover P-Controller

def calculate_hover_cmd(alt, hover_throttle, target_alt, kp):
    error = target_alt - (alt if alt is not None else 0.0)
    cmd_throttle = hover_throttle + kp * error
    cmd_throttle = int(max(15.0, min(60.0, cmd_throttle)))
    return cmd_throttle


def test_hover_controller_at_target():
    # At target altitude, error = 0, commanded throttle = hover throttle base
    cmd = calculate_hover_cmd(alt=0.25, hover_throttle=30, target_alt=0.25, kp=15.0)
    assert cmd == 30


def test_hover_controller_too_low():
    # Below target, error > 0, commanded throttle increases
    cmd = calculate_hover_cmd(alt=0.15, hover_throttle=30, target_alt=0.25, kp=15.0)
    # error = 0.10, delta = 0.10 * 15 = 1.5% -> cmd should be 31 or 32
    assert cmd == 31


def test_hover_controller_too_high():
    # Above target, error < 0, commanded throttle decreases
    cmd = calculate_hover_cmd(alt=0.35, hover_throttle=30, target_alt=0.25, kp=15.0)
    # error = -0.10, delta = -1.5% -> cmd should be 28
    assert cmd == 28


def test_hover_controller_clamp_low():
    # Extremely high altitude error should clamp to 15%
    cmd = calculate_hover_cmd(alt=2.0, hover_throttle=30, target_alt=0.25, kp=15.0)
    assert cmd == 15


def test_hover_controller_clamp_high():
    # Extremely low altitude error should clamp to 60%
    cmd = calculate_hover_cmd(alt=0.0, hover_throttle=50, target_alt=1.0, kp=20.0)
    assert cmd == 60
