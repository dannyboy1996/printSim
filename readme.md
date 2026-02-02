# PrintSim

**PrintSim** is a high-fidelity 3D printer sound emulator. It transforms G-code into a realistic `.wav` audio file by simulating the physics of a 3D printer's motion controller and its mechanical components.

Unlike simple G-code-to-audio converters, **PrintSim** utilizes a full motion planning engine to simulate acceleration, deceleration, and junction deviation, ensuring the sounds you hear match the rhythmic reality of a physical machine.

---

### ✨ Features

*   **Realistic Motion Planning**: Powered by `pyGCodeDecode` to simulate Marlin-style motion profiles, including look-ahead and junction deviation.
*   **Physics-Based Synthesis**:
    *   **Stepper Motors**: Phase-accurate modeling of X, Y, Z, and Extruder motors with velocity-dependent pitch and harmonics.
    *   **Stereo Panning**: Dynamic X-axis panning based on the simulated toolhead position.
    *   **Fan Simulation**: Multi-fan environment (PSU, Mainboard, Hotend, and Part Cooling) with realistic spin-up/spin-down curves.
    *   **Frame Resonance**: Statefull filtering to emulate the acoustic ringing and "body" of a printer frame.
*   **Configurable Hardware**: Adjust acceleration, max speeds, and jerk via a simple YAML preset file.

---

### 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/printsim.git
    cd printsim
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

### 🚀 Usage

Run the script by passing your G-code file as an argument:

```bash
python printsim.py sample.gcode
```

The script will generate a `.wav` file in the same directory (e.g., `sample.wav`).

---

### ⚙️ Configuration

You can customize the simulated printer's behavior in `presets.yaml`. This allows you to match the simulation to your specific printer's limits (e.g., Ender 3, Prusa MK3, etc.):

```yaml
default_printer:
  p_vel: 60      # Default travel speed
  p_acc: 1500    # Acceleration (mm/s^2)
  jerk: 10       # Junction deviation / Jerk
  vX: 300        # Max X speed
  vY: 300        # Max Y speed
  # ... and more
```

---

### 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.