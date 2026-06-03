import os
import json
import tempfile


def load_params_from_path(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_params_to_path(path, liftoff_th, hover_th, landing_th):
    params = {
        "liftoff_throttle": int(liftoff_th),
        "hover_throttle": int(hover_th),
        "landing_throttle": int(landing_th)
    }
    with open(path, "w") as f:
        json.dump(params, f, indent=4)


def test_save_and_load_parameters():
    # Use temporary file to verify loading and saving learned parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        param_path = os.path.join(tmpdir, "learned_params.json")

        # Loading non-existent file returns empty dictionary
        assert load_params_from_path(param_path) == {}

        # Save learned parameters
        save_params_to_path(param_path, 33, 35, 10)

        # Load parameters back and verify values
        loaded = load_params_from_path(param_path)
        assert loaded["liftoff_throttle"] == 33
        assert loaded["hover_throttle"] == 35
        assert loaded["landing_throttle"] == 10
