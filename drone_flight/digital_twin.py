import time
import math


# Simple linear regression utility for trend analysis
def linear_fit(x, y):
    n = len(x)
    if n < 2:
        return 0.0, 0.0  # slope, intercept
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xx = sum(val * val for val in x)
    sum_xy = sum(val_x * val_y for val_x, val_y in zip(x, y))
    denom = (n * sum_xx - sum_x * sum_x)
    if abs(denom) < 1e-6:
        return 0.0, 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


class DigitalTwin:
    def __init__(self, buffer_duration_sec=5.0, update_rate_hz=10):
        self.buffer_size = int(buffer_duration_sec * update_rate_hz)
        self.history = {
            "time": [],
            "alt": [],
            "climb": [],
            "voltage": [],
            "roll": [],
            "pitch": [],
            "xacc": [],
            "yacc": [],
            "zacc": [],
        }
        self.crit_voltage = 10.5

    def update(self, alt, climb, voltage, roll, pitch, xacc, yacc, zacc, last_msg_time=None):
        now = time.time()
        self.history["time"].append(now)
        self.history["alt"].append(alt if alt is not None else 0.0)
        self.history["climb"].append(climb if climb is not None else 0.0)
        self.history["voltage"].append(voltage if voltage is not None else 0.0)
        self.history["roll"].append(roll if roll is not None else 0.0)
        self.history["pitch"].append(pitch if pitch is not None else 0.0)
        self.history["xacc"].append(xacc if xacc is not None else 0.0)
        self.history["yacc"].append(yacc if yacc is not None else 0.0)
        self.history["zacc"].append(zacc if zacc is not None else 0.0)

        # Keep buffer within size limits
        for key in self.history:
            if len(self.history[key]) > self.buffer_size:
                self.history[key].pop(0)

    def predict_future_altitude(self, seconds):
        if not self.history["alt"]:
            return 0.0
        current_alt = self.history["alt"][-1]
        current_climb = self.history["climb"][-1]
        # Linear projection based on vertical velocity
        predicted_alt = current_alt + (current_climb * seconds)
        return max(0.0, predicted_alt)

    def predict_battery_state(self, crit_voltage_override=None):
        v_crit = crit_voltage_override if crit_voltage_override is not None else self.crit_voltage
        times = self.history["time"]
        voltages = self.history["voltage"]

        if not voltages or voltages[-1] <= 1.0:
            # USB only / No battery connected
            return {
                "remaining_voltage": 0.0,
                "remaining_flight_time_sec": 999.0 * 60.0,
                "voltage_drop_rate": 0.0,
            }

        v_now = voltages[-1]
        if len(voltages) < 5:
            # Not enough data for trend yet
            return {
                "remaining_voltage": v_now,
                "remaining_flight_time_sec": 120.0,  # Safe default limit (2 mins)
                "voltage_drop_rate": 0.0,
            }

        # Fit a line to estimate voltage drop rate (slope)
        t_normalized = [t - times[0] for t in times]
        slope, _ = linear_fit(t_normalized, voltages)

        # If voltage is dropping (negative slope)
        if slope < -1e-5:
            drop_rate = -slope  # Volts per second
            time_to_crit = (v_now - v_crit) / drop_rate
            return {
                "remaining_voltage": max(0.0, v_now),
                "remaining_flight_time_sec": max(0.0, time_to_crit),
                "voltage_drop_rate": drop_rate,
            }
        else:
            # Voltage stable or rising (e.g. resting battery / noise)
            return {
                "remaining_voltage": v_now,
                "remaining_flight_time_sec": 120.0,  # Default safety boundary
                "voltage_drop_rate": 0.0,
            }

    def evaluate_stability_trends(self):
        rolls = self.history["roll"]
        pitches = self.history["pitch"]
        n = len(rolls)
        if n < 6:
            return {"drift_deg": 0.0, "oscillation_growing": False}

        # Measure drift as absolute average attitude offset in current half
        half = n // 2
        recent_roll_avg = sum(rolls[half:]) / len(rolls[half:])
        recent_pitch_avg = sum(pitches[half:]) / len(pitches[half:])
        drift = math.sqrt(recent_roll_avg**2 + recent_pitch_avg**2)

        # Oscillation growth: check if variance in second half is larger than first half
        def variance(data):
            avg = sum(data) / len(data)
            return sum((x - avg) ** 2 for x in data) / len(data)

        var1 = variance(rolls[:half]) + variance(pitches[:half])
        var2 = variance(rolls[half:]) + variance(pitches[half:])

        osc_growing = (var2 > var1 * 1.5) and (var2 > 5.0)  # Grow by >50% and variance >5.0 deg^2

        return {
            "drift_deg": drift,
            "oscillation_growing": osc_growing,
        }

    def get_flight_risk_score(self, current_time_sec=None):
        # 1. Telemetry latency risk
        now = time.time()
        last_msg = self.history["time"][-1] if self.history["time"] else now
        latency = now - last_msg
        if latency <= 1.0:
            latency_risk = 0.0
        elif latency >= 5.0:
            latency_risk = 100.0
        else:
            latency_risk = (latency - 1.0) / 4.0 * 100.0

        # 2. Battery risk
        bat_state = self.predict_battery_state()
        v_now = self.history["voltage"][-1] if self.history["voltage"] else 0.0
        if v_now <= 1.0:
            # USB only - no battery risk
            bat_risk = 0.0
        else:
            rem_sec = bat_state["remaining_flight_time_sec"]
            if rem_sec <= 0.0:
                bat_risk = 100.0
            elif rem_sec <= 20.0:
                bat_risk = 80.0 + (20.0 - rem_sec) / 20.0 * 20.0
            elif rem_sec <= 60.0:
                bat_risk = 30.0 + (60.0 - rem_sec) / 40.0 * 50.0
            else:
                bat_risk = max(0.0, 30.0 - (rem_sec - 60.0) / 120.0 * 30.0)

        # 3. Attitude risk
        roll = abs(self.history["roll"][-1]) if self.history["roll"] else 0.0
        pitch = abs(self.history["pitch"][-1]) if self.history["pitch"] else 0.0
        max_tilt = max(roll, pitch)
        if max_tilt < 15.0:
            att_risk = 0.0
        elif max_tilt >= 60.0:
            att_risk = 100.0
        elif max_tilt >= 40.0:
            att_risk = 50.0 + (max_tilt - 40.0) / 20.0 * 50.0
        else:
            att_risk = (max_tilt - 15.0) / 25.0 * 50.0

        # 4. Instability risk
        trends = self.evaluate_stability_trends()
        instability_risk = 0.0
        if trends["oscillation_growing"]:
            instability_risk += 40.0
        instability_risk += min(40.0, trends["drift_deg"] / 20.0 * 40.0)

        # 5. G-force vibration risk
        xacc = self.history["xacc"][-1] if self.history["xacc"] else 0.0
        yacc = self.history["yacc"][-1] if self.history["yacc"] else 0.0
        zacc = self.history["zacc"][-1] if self.history["zacc"] else 0.0
        max_g = max(xacc, yacc, zacc)
        if max_g < 2.0:
            vib_risk = 0.0
        elif max_g >= 4.0:
            vib_risk = 100.0
        else:
            vib_risk = (max_g - 2.0) / 2.0 * 100.0

        # Overall risk: take maximum of critical risks, or weighted average of general risks
        max_critical_risk = max(latency_risk, att_risk, vib_risk)
        weighted_general_risk = (bat_risk * 0.4) + (instability_risk * 0.6)

        overall_risk = max(max_critical_risk, weighted_general_risk)
        return min(100.0, max(0.0, overall_risk))
