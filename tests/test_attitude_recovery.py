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
