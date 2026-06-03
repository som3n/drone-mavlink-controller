import math
import time


def mean(data):
    if not data:
        return 0.0
    return sum(data) / len(data)


def std_dev(data):
    n = len(data)
    if n < 2:
        return 0.0
    avg = sum(data) / n
    variance = sum((x - avg) ** 2 for x in data) / n
    return math.sqrt(variance)


class HealthMonitor:
    def __init__(self):
        pass

    def calculate_health_scores(self, dt_instance):
        history = dt_instance.history

        # 1. Battery Health
        voltages = history["voltage"]
        if not voltages or voltages[-1] <= 1.0:
            # USB only / No battery connected: report 100% to avoid dragging down overall score
            battery_health = 100.0
        else:
            v_now = voltages[-1]
            # Base health on how close voltage is to safety cutoff
            crit_v = dt_instance.crit_voltage
            max_v = 12.6  # 3S nominal full charge is ~12.6V
            if v_now <= crit_v:
                battery_health = 0.0
            elif v_now >= max_v:
                battery_health = 100.0
            else:
                battery_health = (v_now - crit_v) / (max_v - crit_v) * 100.0

            # Deduct health for high voltage sags/discharge rate
            bat_state = dt_instance.predict_battery_state()
            drop_rate = bat_state["voltage_drop_rate"]
            # 0.05 V/s is very fast discharge (drains 3S in ~3 mins)
            drop_penalty = min(50.0, (drop_rate / 0.05) * 50.0)
            battery_health = max(0.0, battery_health - drop_penalty)

        # 2. Communication Health
        times = history["time"]
        now = time.time()
        last_msg_time = times[-1] if times else now
        latency = now - last_msg_time

        if latency <= 0.5:
            comm_health = 100.0
        elif latency >= 5.0:
            comm_health = 0.0
        else:
            # Scale down linearly from 100% (0.5s) to 0% (5.0s)
            comm_health = 100.0 - ((latency - 0.5) / 4.5 * 100.0)

        # 3. Stability Health
        rolls = history["roll"]
        pitches = history["pitch"]
        climbs = history["climb"]

        if len(rolls) < 5:
            stability_health = 100.0
        else:
            roll_std = std_dev(rolls)
            pitch_std = std_dev(pitches)
            climb_std = std_dev(climbs)

            # Define penalty metrics (std dev limits: 10 deg for attitude, 0.5 m/s for climb rate)
            att_penalty = (roll_std + pitch_std) / 20.0 * 50.0
            climb_penalty = climb_std / 0.5 * 50.0

            stability_health = max(0.0, 100.0 - att_penalty - climb_penalty)

        # 4. Sensor Health
        xacc = history["xacc"]
        yacc = history["yacc"]
        zacc = history["zacc"]

        if len(xacc) < 5:
            sensor_health = 100.0
        else:
            # High standard deviation of acceleration = high vibration
            x_vib = std_dev(xacc)
            y_vib = std_dev(yacc)
            z_vib = std_dev(zacc)
            avg_vib = (x_vib + y_vib + z_vib) / 3.0

            # Penalty scales up: 0.5g vibration averages start reducing sensor health
            vib_penalty = min(60.0, (avg_vib / 0.5) * 60.0)
            sensor_health = max(40.0, 100.0 - vib_penalty)  # Floor at 40%

        # 5. Overall Health Score (Weighted average)
        overall_health = (
            (battery_health * 0.25) +
            (comm_health * 0.25) +
            (stability_health * 0.25) +
            (sensor_health * 0.25)
        )

        return {
            "battery": battery_health,
            "communication": comm_health,
            "stability": stability_health,
            "sensor": sensor_health,
            "overall": overall_health,
        }

    def detect_anomalies(self, dt_instance):
        history = dt_instance.history
        rolls = history["roll"]
        pitches = history["pitch"]
        climbs = history["climb"]
        zacc = history["zacc"]

        if len(rolls) < 5:
            return {"anomaly_score": 0.0, "is_anomaly": False}

        # Calculate metrics
        roll_std = std_dev(rolls)
        pitch_std = std_dev(pitches)
        climb_std = std_dev(climbs)
        zacc_val = zacc[-1] if zacc else 0.0

        # Anomaly scoring triggers
        anomaly_reasons = []
        score = 0.0

        # 1. Stability anomalies (large oscillations)
        if roll_std > 12.0 or pitch_std > 12.0:
            score += 40.0
            anomaly_reasons.append("high attitude oscillation")

        # 2. Vertical rate anomalies (rapid vertical oscillation)
        if climb_std > 0.4:
            score += 30.0
            anomaly_reasons.append("high climb rate oscillation")

        # 3. Acceleration anomalies (impact/vibration spikes)
        if zacc_val > 3.0:
            score += 30.0
            anomaly_reasons.append("high vertical G force spike")

        # 4. Telemetry latency anomalies
        times = history["time"]
        now = time.time()
        last_msg_time = times[-1] if times else now
        latency = now - last_msg_time
        if latency > 2.0:
            score += 20.0
            anomaly_reasons.append("telemetry latency spike")

        # Ensure we cap the score
        anomaly_score = min(100.0, score)
        is_anomaly = anomaly_score >= 40.0

        return {
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "reasons": anomaly_reasons,
        }
