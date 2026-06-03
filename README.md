# Autonomous Drone MAVLink Controller

A robust, thread-safe, and highly stable Python-based flight controller designed to command ArduPilot/MAVLink-compatible drones. It features active attitude corrections, autonomous landing, battery protection, hardware failsafes, and EKF origin initialization optimized for indoor and bench testing.

---

## 🛠️ Technology Stack

- **Core Language**: Python 3.11+
- **MAVLink API**: `pymavlink`
- **Testing**: `pytest`
- **Configuration**: YAML (`pyyaml`)
- **Logging**: Console stream & CSV-based telemetry logs

---

## 📂 Project Structure

```
├── config/
│   ├── bench_test.yaml       # Bench test profile (bench_test: true, 5s land timeout)
│   └── real_flight.yaml      # Real flight profile (bench_test: false, standard arming)
├── drone_flight/
│   ├── __init__.py
│   ├── flight_controller.py  # Main flight sequencer and state machine
│   ├── telemetry.py          # Thread-safe telemetry daemon reader and RC commands
│   ├── safety_monitor.py     # Background safety monitoring thread
│   └── logger.py             # Logging utilities (CSV & console logger)
├── logs/                     # Telemetry records (.log and telemetry .csv output)
├── tests/
│   ├── test_pct_to_pwm.py    # Unit tests for throttle PWM remapping
│   └── test_telemetry.py     # Mock MAVLink telemetry reader unit tests
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd drone-mavlink-controller
   ```

2. **Set up the virtual environment**:
   Make sure you have virtual environment active. To initialize:
   ```bash
   python -m venv drone_env
   .\drone_env\Scripts\activate     # Windows
   source drone_env/bin/activate    # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration Files

The project uses YAML profiles in the `config/` directory.

- **`bench_test.yaml`** (used for safe bench testing with propellers **OFF**):
  - `bench_test: true`: Enables bypass modes for pre-arm checks and forces arming without GPS.
  - Landing timeout is automatically capped at `5.0` seconds since there is no physical landing impact to trigger automatic touchdown disarms.
- **`real_flight.yaml`** (used for real flights):
  - `bench_test: false`: Standard flight controller arming checks (`arducopter_arm()`) are enforced.
  - Dynamic landing checks are monitored until the drone disarms automatically upon landing, with a `30.0` second backup safety timeout.

---

## ✈️ Running the Flight Controller

Activate your virtual environment and execute the script:

### 1. Run Bench Test (Default)
```bash
python drone_flight/flight_controller.py
```

### 2. Run Real Flight Profile
```bash
python drone_flight/flight_controller.py config/real_flight.yaml
```

*Note: Telemetry log files and CSV graphs are automatically created under `logs/` directory upon start.*

---

## 🛡️ Safety Features & Failsafes

The flight controller prioritizes vehicle health and pilot safety with multiple active layers:

### 1. Active Proportional Tilt Correction
During flight/takeoff phases, if the vehicle tilts significant angles ($>= 5.0^\circ$ deadband), a proportional controller (P-gain = `8.0` PWM units per degree) applies corrective overrides (`roll` and `pitch` command updates) clamped inside a safe range `[1300, 1700]` to level the drone. It ignores minor tilts under $5^\circ$ to prevent unbalancing on flat ground/bench.

### 2. EKF Origin & Dynamic Home Alignment
Indoors or on the bench (where GPS lock is unavailable), the EKF fails to align vertical references, causing the relative altitude to drift wildly ($>30\text{ cm}$). The script dynamically sets the EKF GPS Global Origin and resets the Home Position to the exact current barometric reference. This zero-calibrates relative altitude to exactly `0.0m` (noise floor limited to $<2\text{ cm}$).

### 3. Altitude Normalization
The first MAVLink position package processed establishes an `alt_offset`. All subsequent altitude metrics are offset-zeroed and non-negatively clamped (`max(0.0, raw - offset)`), guaranteeing that height coordinates start at exactly `0.0m`.

### 4. Continuous Safety Monitor
A background `safety_monitor` daemon actively polls the telemetry at 20Hz:
- **IMU Spikes**: Immediate safety abort if accelerometer G-forces exceed limit (`crash_accel_g: 4.0`).
- **Extreme Attitude**: Safety abort if the drone tilts beyond safe limits (`crash_roll_deg: 60`, `crash_pitch_deg: 60`).
- **Battery Health**: Shuts down/aborts if battery drops below threshold (e.g. `10.5V` for 3S). It automatically bypasses this check if USB-only power is detected ($< 1.0\text{V}$).
- **Heartbeat Timeout**: Aborts flight if connection is lost (`telemetry_timeout_sec: 5.0`).
- **Hardware Disconnect**: Telemetry thread aborts and immediately disarms the motors if 5 consecutive read errors occur (e.g. USB/serial unplugged).

### 5. Autonomous Landing Transition
When executing landing phases, the script clears all RC overrides to release manual channel bounds and commands native `LAND` mode, transferring landing control to ArduPilot.

---

## 🧪 Unit Testing

We maintain a comprehensive unit test suite utilizing mock MAVLink interfaces to test packet decoding, throttle remap math, and error thresholds.

To execute tests:
```bash
.\drone_env\Scripts\python.exe -m pytest
```

**Test Modules**:
- `tests/test_pct_to_pwm.py`: Verifies PWM calibrations are scaled within mapped receiver ranges (`1100-1900`).
- `tests/test_telemetry.py`: Tests thread safety, message handlers (`GLOBAL_POSITION_INT`, `VFR_HUD`), offset calculation, and connection failure aborts.
