import argparse
import math
import soundfile as sf
import numpy as np
import io
import os
from tqdm import tqdm
from scipy.signal import butter, lfilter
from pyGCodeDecode import gcode_interpreter
from pathlib import Path

# --- CONSTANTS & CONFIG ---
SAMPLE_RATE = 44100
PRESETS_FILE = "presets.yaml"
PRINTER_NAME = "default_printer"

# --- REINSTATED SOUND ENGINE COMPONENTS ---

class ResonanceFilter:
    """A bank of resonant filters to simulate the printer's frame ringing."""
    def __init__(self, sample_rate=44100):
        params = [(85, 30, 0.6), (120, 40, 0.4), (250, 20, 0.2)]
        self.filters = [self._create_filter(freq, Q, gain, sample_rate) for freq, Q, gain in params]

    def _create_filter(self, freq, Q, gain, fs):
        nyquist = 0.5 * fs
        low = max(1, freq - freq / (2 * Q))
        high = min(nyquist - 1, freq + freq / (2 * Q))
        if low >= high: high = low + 1
        b, a = butter(2, [low, high], btype='band', fs=fs)
        return {'b': b, 'a': a, 'gain': gain, 'z': np.zeros(max(len(a), len(b)) - 1)}

    def process(self, signal):
        output = np.zeros_like(signal)
        for f in self.filters:
            y, f['z'] = lfilter(f['b'], f['a'], signal, zi=f['z'])
            output += y * f['gain']
        return output

class Fan:
    """Simulates a fan based on mathematical principles."""
    def __init__(self, sample_rate, ramp_time=1.5, num_blades=7, max_rpm=4000, vol=1.0, hum_to_noise_ratio=0.3):
        self.sample_rate = sample_rate
        self.ramp_time = ramp_time
        self.num_blades = num_blades
        self.max_rpm = max_rpm
        self.vol = vol
        self.hum_to_noise_ratio = hum_to_noise_ratio
        self._noise_z_fallback = np.zeros(3)
        self.current_speed = 0.0
        self.target_speed = 0.0
        self.phase_accumulator = 0.0

    def set_speed(self, speed_percent):
        self.target_speed = np.clip(speed_percent, 0.0, 1.0)

    def generate_audio(self, num_samples):
        if self.current_speed == 0 and self.target_speed == 0:
            return np.zeros(num_samples, dtype=np.float32)

        ramp_amount_per_second = 1.0 / self.ramp_time if self.ramp_time > 0 else float('inf')
        max_ramp_change = ramp_amount_per_second * (num_samples / self.sample_rate)
        start_speed = self.current_speed
        end_speed = self.current_speed + np.clip(self.target_speed - self.current_speed, -max_ramp_change, max_ramp_change)
        speed_profile = np.linspace(start_speed, end_speed, num_samples)
        self.current_speed = end_speed

        volume_profile = speed_profile * self.vol
        rpm_profile = self.max_rpm * speed_profile

        white_noise = np.random.randn(num_samples).astype(np.float32)
        b = [0.0499, -0.0959, 0.0506, -0.0044]
        a = [1, -2.4949, 2.0172, -0.5221]
        airflow_noise, self._noise_z_fallback = lfilter(b, a, white_noise, zi=self._noise_z_fallback)

        blade_freq_profile = (rpm_profile / 60.0) * self.num_blades
        phase_increments = 2 * np.pi * blade_freq_profile / self.sample_rate
        phase = self.phase_accumulator + np.cumsum(phase_increments)
        self.phase_accumulator = phase[-1] % (2 * np.pi)

        harmonics_gain = 0.4 + speed_profile * 0.6
        fan_hum = np.sin(phase) * 0.6 + np.sin(phase * 2) * 0.25 * harmonics_gain
        final_noise = (airflow_noise * (1.0 - self.hum_to_noise_ratio) + fan_hum * self.hum_to_noise_ratio) * volume_profile
        return final_noise.astype(np.float32)

def generate_stepper_waveform(phase):
    return ((np.sin(phase) + 0.2 * np.sin(phase * 2)) / 1.2 + 0.5 * np.sin(phase * 8))

def generate_extruder_waveform(phase):
    return (np.sin(phase) + 0.5 * np.sin(phase * 3) + 0.2 * np.sin(phase * 5)) / 1.7

# --- MAIN ENGINE ---

def gcode_to_audio(gcode_file, output_file, printer_name=PRINTER_NAME):
    print(f"Step 1: Planning motion with pyGCodeDecode using printer: {printer_name}...")
    
    clean_gcode_path = gcode_file + ".tmp"
    fan_events = []
    
    with open(gcode_file, 'r') as f:
        lines = f.readlines()
        
    with open(clean_gcode_path, 'w') as f:
        for i, line in enumerate(lines):
            l = line.strip().upper()
            if l.startswith('G28'):
                f.write("G92 X110 Y110 Z125 E0\n")
                f.write("G1 X0 Y0 Z0 F3000\n")
            elif l.startswith('G4'):
                continue
            elif l.startswith('M106'):
                s = 255
                for p in l.split():
                    if p.startswith('S'): s = float(p[1:])
                fan_events.append({'line': i+1, 'speed': s / 255.0})
                f.write(line)
            elif l.startswith('M107'):
                fan_events.append({'line': i+1, 'speed': 0.0})
                f.write(line)
            elif l.startswith('G0') or l.startswith('G1'):
                # Filter out moves that are ONLY extruder (retractions/primes)
                # These cause silent gaps in the audio.
                parts = l.split()
                has_xyz = any(p[0] in 'XYZ' for p in parts)
                has_e = any(p.startswith('E') for p in parts)
                if has_e and not has_xyz:
                    # Preserve feedrate if it was set in this line
                    f_part = next((p for p in parts if p.startswith('F')), None)
                    if f_part:
                        f.write(f"{parts[0]} {f_part}\n")
                    continue
                f.write(line)
            else:
                f.write(line)

    try:
        setup = gcode_interpreter.setup(presets_file=PRESETS_FILE, printer=printer_name)
        sim = gcode_interpreter.simulation(gcode_path=clean_gcode_path, initial_machine_setup=setup)
    finally:
        if os.path.exists(clean_gcode_path): os.remove(clean_gcode_path)

    total_duration = sim.blocklist[-1].get_segments()[-1].t_end
    total_samples = int(total_duration * SAMPLE_RATE)
    final_audio = np.zeros((total_samples, 2), dtype=np.float32)
    
    # Setup Fans
    psu_fan = Fan(SAMPLE_RATE, vol=0.15, max_rpm=2000)
    hotend_fan = Fan(SAMPLE_RATE, vol=0.20, max_rpm=4000)
    part_cooling_fan = Fan(SAMPLE_RATE, vol=0.8, max_rpm=6000)
    psu_fan.set_speed(1.0)
    hotend_fan.set_speed(1.0)

    resonance_model = ResonanceFilter(SAMPLE_RATE)
    motor_vol, extruder_vol = 0.55, 0.45
    last_phases = {'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'E': 0.0}
    fan_idx = 0

    print("Step 2: Synthesizing audio...")
    for block in tqdm(sim.blocklist, desc="Processing Blocks"):
        line_num = block.state_B.line_number
        while fan_idx < len(fan_events) and fan_events[fan_idx]['line'] <= line_num:
            part_cooling_fan.set_speed(fan_events[fan_idx]['speed'])
            fan_idx += 1

        for seg in block.get_segments():
            t_start, t_end = float(seg.t_begin), float(seg.t_end)
            s_start, s_end = int(t_start * SAMPLE_RATE), int(t_end * SAMPLE_RATE)
            num_samples = s_end - s_start
            if num_samples <= 0: continue

            v_begin = seg.vel_begin.get_vec(withExtrusion=True)
            v_end = seg.vel_end.get_vec(withExtrusion=True)
            pan = np.clip(seg.get_position(t_start).x / 220.0, 0.1, 0.9)

            seg_mono = np.zeros(num_samples, dtype=np.float32)
            fans_audio = psu_fan.generate_audio(num_samples) + \
                         hotend_fan.generate_audio(num_samples) + \
                         part_cooling_fan.generate_audio(num_samples)

            for axis_idx, axis_key in enumerate(['X', 'Y', 'Z', 'E']):
                v0, v1 = v_begin[axis_idx], v_end[axis_idx]
                if v0 == 0 and v1 == 0: continue
                
                v_profile = np.abs(np.linspace(v0, v1, num_samples))
                base_freq = {'X': 70, 'Y': 75, 'Z': 35, 'E': 150}[axis_key]
                freq_profile = base_freq + v_profile * 10
                phases = last_phases[axis_key] + np.cumsum(2 * np.pi * freq_profile / SAMPLE_RATE)
                last_phases[axis_key] = phases[-1] % (2 * np.pi)
                
                if axis_key == 'E':
                    wav = generate_extruder_waveform(phases) * extruder_vol
                else:
                    wav = generate_stepper_waveform(phases) * motor_vol
                
                if axis_key == 'X':
                    final_audio[s_start:s_end, 0] += wav * (1 - pan)
                    final_audio[s_start:s_end, 1] += wav * pan
                else:
                    seg_mono += wav
            
            final_audio[s_start:s_end, 0] += seg_mono + fans_audio
            final_audio[s_start:s_end, 1] += seg_mono + fans_audio

    print("Step 3: Post-processing...")
    final_audio[:, 0] += resonance_model.process(final_audio[:, 0])
    final_audio[:, 1] += resonance_model.process(final_audio[:, 1])

    max_val = np.max(np.abs(final_audio))
    if max_val > 0: final_audio /= max_val
    sf.write(output_file, final_audio, SAMPLE_RATE)
    print(f"Done: {output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert G-code to printer sound (Motion Test).")

    parser.add_argument("gcode", help="Input G-code file")

    parser.add_argument("--printer", default=PRINTER_NAME, help=f"Printer preset from {PRESETS_FILE} (default: {PRINTER_NAME})")

    args = parser.parse_args()

    gcode_to_audio(args.gcode, str(Path(args.gcode).with_suffix(".wav")), printer_name=args.printer)
