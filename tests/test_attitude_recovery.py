# Unit tests for PD Attitude Correction & Recovery Controller

def calculate_recovery_command(target, actual, rate, kp_base, kd_base):
    error = target - actual
    max_err = abs(error)

    # Zone-based gains
    if max_err <= 5.0:
        kp_factor, kd_factor = 0.0, 0.0
        state = "NORMAL"
    elif max_err <= 10.0:
        kp_factor, kd_factor = 0.5, 0.5
        state = "MINOR CORRECTION"
    elif max_err <= 20.0:
        kp_factor, kd_factor = 1.0, 1.0
        state = "ACTIVE RECOVERY"
    elif max_err <= 35.0:
        kp_factor, kd_factor = 1.5, 1.0
        state = "HIGH RECOVERY"
    else:
        kp_factor, kd_factor = 2.0, 1.0
        state = "RECOVERY MODE"

    KP = kp_base * kp_factor
    KD = kd_base * kd_factor

    corr = KP * error - KD * rate
    corr = max(-200.0, min(200.0, corr))
    cmd = int(1500 + corr)

    return cmd, state, error, corr


def calculate_test_stability_score(max_err, roll_var, pitch_var, max_rate):
    # Component A: Error (max 30)
    if max_err <= 5.0:
        att_score = 0.0
    else:
        att_score = min(30.0, (max_err - 5.0) / 30.0 * 30.0)

    # Component B: Variance (max 20 each)
    roll_var_score = min(20.0, (roll_var / 5.0) * 20.0)
    pitch_var_score = min(20.0, (pitch_var / 5.0) * 20.0)

    # Component C: Rates (max 30)
    if max_rate <= 10.0:
        rate_score = 0.0
    else:
        rate_score = min(30.0, (max_rate - 10.0) / 90.0 * 30.0)

    score = att_score + roll_var_score + pitch_var_score + rate_score
    return min(100.0, max(0.0, score))


def test_zone_1_stable():
    cmd, state, err, corr = calculate_recovery_command(
        target=0.0, actual=4.0, rate=5.0, kp_base=8.0, kd_base=1.5
    )
    assert state == "NORMAL"
    assert corr == 0.0
    assert cmd == 1500


def test_zone_3_active_recovery():
    # Target 15 deg, Actual 0 deg -> error = 15, Zone 3
    cmd, state, err, corr = calculate_recovery_command(
        target=15.0, actual=0.0, rate=2.0, kp_base=8.0, kd_base=1.5
    )
    assert state == "ACTIVE RECOVERY"
    # KP = 8 * 1.0 = 8.0, KD = 1.5 * 1.0 = 1.5
    # corr = 8.0 * 15 - 1.5 * 2.0 = 120 - 3 = 117
    assert abs(corr - 117.0) < 1e-3
    assert cmd == 1617


def test_zone_5_recovery_mode_clamp():
    # Extremely large error should clamp to 200/1700
    cmd, state, err, corr = calculate_recovery_command(
        target=50.0, actual=0.0, rate=0.0, kp_base=8.0, kd_base=1.5
    )
    assert state == "RECOVERY MODE"
    assert corr == 200.0
    assert cmd == 1700


def test_stability_score_stable():
    # Perfectly stable parameters should output 0 score
    score = calculate_test_stability_score(max_err=0.0, roll_var=0.0, pitch_var=0.0, max_rate=0.0)
    assert score == 0.0


def test_stability_score_critical():
    # High error, high rate, and high variance should trigger critical score (>80)
    score = calculate_test_stability_score(
        max_err=40.0, roll_var=6.0, pitch_var=6.0, max_rate=110.0
    )
    assert score >= 80.0


def test_authority_manager_and_gain_scheduling():
    from drone_flight.flight_controller import calculate_attitude_recovery
    from drone_flight.digital_twin import DigitalTwin
    from drone_flight import telemetry as telem

    dt = DigitalTwin()
    telem.set_target_attitude(0.0, 0.0)

    # Mock attitude state in Zone 1 (Error <= 5.0) -> auth = 0.0
    with telem.state.lock:
        telem.state.roll = 4.0
        telem.state.pitch = 0.0
        telem.state.rollspeed = 2.0
        telem.state.pitchspeed = 0.0
        telem.state.prev_roll_cmd = 1500
        telem.state.prev_pitch_cmd = 1500

    roll_cmd, _, _, _, _, _, roll_corr, _, _, rec_state = calculate_attitude_recovery(0.0, 0.0, dt)
    assert rec_state == "NORMAL"
    assert roll_corr == 0.0
    assert roll_cmd == 1500


def test_rate_limiter():
    from drone_flight.flight_controller import calculate_attitude_recovery
    from drone_flight.digital_twin import DigitalTwin
    from drone_flight import telemetry as telem

    dt = DigitalTwin()

    # Previous command is 1500. Large error demands desired_roll_pwm = 1300
    # The rate limiter clamps change to -20, outputting 1480.
    with telem.state.lock:
        telem.state.roll = 50.0
        telem.state.pitch = 0.0
        telem.state.rollspeed = 0.0
        telem.state.pitchspeed = 0.0
        telem.state.prev_roll_cmd = 1500
        telem.state.prev_pitch_cmd = 1500

    roll_cmd, _, _, _, _, _, _, _, _, _ = calculate_attitude_recovery(0.0, 0.0, dt)
    assert roll_cmd == 1480


def test_throttle_rate_limiter():
    from drone_flight.flight_controller import send_throttle_safe
    from drone_flight.digital_twin import DigitalTwin
    from drone_flight import telemetry as telem

    dt = DigitalTwin()

    class DummyMaster:
        instance = None

        def __init__(self):
            self.target_system = 1
            self.target_component = 1
            self.sent_throttle = None
            DummyMaster.instance = self

        class mav:
            @staticmethod
            def rc_channels_override_send(sys, comp, c1, c2, throttle_pwm, *args):
                DummyMaster.instance.sent_throttle = throttle_pwm

    master = DummyMaster()
    telem.current_throttle = 10.0
    telem.state.prev_roll_cmd = 1500
    telem.state.prev_pitch_cmd = 1500

    # Demand 20.0% throttle. Change 10.0% -> 20.0% is +10.0%.
    # Clamps to +1.5%, giving actual throttle of 11.5%.
    send_throttle_safe(master, 20.0, "test", dt)

    assert abs(telem.current_throttle - 11.5) < 1e-3
    assert master.sent_throttle == 1192
