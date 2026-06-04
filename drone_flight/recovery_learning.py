import os
import json
import csv
from datetime import datetime
from drone_flight.logger import log

LEARNING_FILE = "config/controller_learning.json"
DATASET_FILE = "logs/recovery_learning.csv"

DEFAULT_LEARNING = {
    "zone2": {"kp": 5.0, "kd": 1.0, "samples": 0},
    "zone3": {"kp": 8.0, "kd": 1.5, "samples": 0},
    "zone4": {"kp": 12.0, "kd": 2.5, "samples": 0},
    "zone5": {"kp": 16.0, "kd": 3.5, "samples": 0}
}

KP_MIN, KP_MAX = 2.0, 25.0
KD_MIN, KD_MAX = 0.5, 8.0


def load_controller_learning():
    """Load controller learning data, initializing it if missing."""
    if os.path.exists(LEARNING_FILE):
        try:
            with open(LEARNING_FILE) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Error loading controller learning file: {e}")
    # Initialize directory and write defaults
    os.makedirs(os.path.dirname(LEARNING_FILE), exist_ok=True)
    try:
        with open(LEARNING_FILE, "w") as f:
            json.dump(DEFAULT_LEARNING, f, indent=4)
        log.info(f"Initialized controller learning database at {LEARNING_FILE}")
    except Exception as e:
        log.warning(f"Error writing default controller learning database: {e}")
    return {
        "zone2": dict(DEFAULT_LEARNING["zone2"]),
        "zone3": dict(DEFAULT_LEARNING["zone3"]),
        "zone4": dict(DEFAULT_LEARNING["zone4"]),
        "zone5": dict(DEFAULT_LEARNING["zone5"]),
    }


def save_controller_learning(data):
    """Save controller learning data to json database."""
    try:
        os.makedirs(os.path.dirname(LEARNING_FILE), exist_ok=True)
        with open(LEARNING_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        log.error(f"Failed to save controller learning database: {e}")


def evaluate_recovery_performance(event_data):
    """Evaluate performance metrics of a completed recovery event, apply adaptation rules,

    clamp gains within safe limits, and persist adjustments.
    """
    zone = event_data.get("zone", 3)
    duration = event_data.get("duration", 0.0)
    overshoot = event_data.get("overshoot", 0.0)
    stability_score = event_data.get("stability_score", 0.0)
    auth_factor = event_data.get("authority_factor", 0.0)
    success = event_data.get("success", False)
    recoverable = event_data.get("recoverable", True)

    # 1. Quality scoring formula (all metrics normalized to 0-100 scale)
    duration_norm = min(100.0, (duration / 5.0) * 100.0)
    overshoot_norm = min(100.0, (overshoot / 15.0) * 100.0)
    stability_norm = min(100.0, stability_score)

    recovery_score = (duration_norm * 0.4) + (overshoot_norm * 0.3) + (stability_norm * 0.3)

    if not success:
        quality = "Failed"
    elif recovery_score <= 20.0:
        quality = "Excellent"
    elif recovery_score <= 40.0:
        quality = "Good"
    elif recovery_score <= 65.0:
        quality = "Fair"
    else:
        quality = "Poor"

    log.info(
        f"  [RECOVERY LEARNING] Quality: {quality} (Score: {recovery_score:.2f}) | "
        f"Dur: {duration:.2f}s Overshoot: {overshoot:.1f}° Stability: {stability_score:.1f} | "
        f"Recoverable: {recoverable}"
    )

    # Load learning database
    learning_data = load_controller_learning()
    zone_key = f"zone{max(2, min(5, zone))}"

    kp_old, kd_old = 8.0, 1.5
    kp_new, kd_new = 8.0, 1.5

    if zone_key in learning_data:
        kp_old = learning_data[zone_key]["kp"]
        kd_old = learning_data[zone_key]["kd"]
        kp_new, kd_new = kp_old, kd_old

        if recoverable:
            # Rule-based adaptation rules
            # Excellent quality recovery — skip all gain changes
            if quality != "Excellent":
                # Rule 1: Slow Recovery
                if duration > 2.5:
                    kp_new += 0.10
                    log.info("  [LEARNING RULE] Duration > 2.5s -> KP += 0.10")

                # Rule 2: Excessive Overshoot
                if overshoot > 5.0:
                    kd_new += 0.05
                    log.info("  [LEARNING RULE] Overshoot > 5.0° -> KD += 0.05")

                # Rule 3: Oscillation Detected
                if stability_score > 40.0:
                    kp_new -= 0.05
                    kd_new += 0.05
                    log.info(
                        "  [LEARNING RULE] Stability > 40 -> KP -= 0.05, KD += 0.05"
                    )

            # 2. Gain safety limits enforcement
            kp_new = round(max(KP_MIN, min(KP_MAX, kp_new)), 3)
            kd_new = round(max(KD_MIN, min(KD_MAX, kd_new)), 3)

            # Update database values (only increment samples when gains actually changed)
            gains_changed = (kp_new != kp_old or kd_new != kd_old)
            learning_data[zone_key]["kp"] = kp_new
            learning_data[zone_key]["kd"] = kd_new
            if gains_changed:
                learning_data[zone_key]["samples"] += 1
            save_controller_learning(learning_data)

            if gains_changed:
                log.info(
                    f"  [LEARNING UPDATE] Optimized {zone_key} gains: "
                    f"KP: {kp_old} -> {kp_new} | KD: {kd_old} -> {kd_new}"
                )
        else:
            log.info("  [RECOVERY LEARNING] Bench test mode: skipping parameter adaptation updates.")

    # Determine learning status
    if not recoverable:
        learning_status = "SKIPPED"
    elif quality == "Excellent":
        learning_status = "EXCELLENT_NO_CHANGE"
    elif kp_new != kp_old or kd_new != kd_old:
        learning_status = "LEARNED"
    else:
        learning_status = "NO_CHANGE"

    # 3. CSV Dataset logging for future research
    try:
        os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)
        write_header = not os.path.exists(DATASET_FILE)
        with open(DATASET_FILE, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow([
                    "timestamp", "zone", "kp", "kd", "duration", "overshoot",
                    "stability_score", "authority_factor", "success",
                    "recoverable", "quality", "recovery_score", "learning_status"
                ])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                zone,
                kp_new,
                kd_new,
                round(duration, 3),
                round(overshoot, 2),
                round(stability_score, 2),
                round(auth_factor, 2),
                "BENCH_VALIDATION" if not recoverable else success,
                recoverable,
                quality,
                round(recovery_score, 2),
                learning_status
            ])
    except Exception as e:
        log.warning(f"Failed to log recovery learning event to CSV: {e}")

    return {
        "recovery_score": recovery_score,
        "quality": quality,
        "kp": kp_new,
        "kd": kd_new,
        "learning_status": learning_status
    }
