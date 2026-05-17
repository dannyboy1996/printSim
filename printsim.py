import argparse
import math
import soundfile as sf
import numpy as np
import io
import os
from tqdm import tqdm
from scipy.signal import butter, lfilter, resample_poly
from pathlib import Path
import yaml
from klipper_planner import KlipperPlanner

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

def gcode_to_audio(gcode_file, output_file, printer_name=PRINTER_NAME, force_corexy=False):
    print(f"Step 1: Planning motion with KlipperPlanner using printer: {printer_name}...")
    
    # Load printer preset
    with open(PRESETS_FILE, 'r') as f:
        presets = yaml.safe_load(f)
    preset = presets.get(printer_name, presets['default_printer'])
    
    planner = KlipperPlanner(
        max_velocity=preset.get('vX', 300), 
        max_accel=preset.get('p_acc', 1500),
        scv=preset.get('jerk', 10) # Approx jerk as SCV
    )
    
    # Pre-parse only to replace G28 (homing) with explicit position commands
    clean_gcode_path = gcode_file + ".tmp"

    with open(gcode_file, 'r') as f:
        lines = f.readlines()

    with open(clean_gcode_path, 'w') as f:
        for line in lines:
            stripped = line.split(';')[0].strip()
            if not stripped:
                continue
            if stripped.upper().startswith('G28'):
                f.write(f"G92 X{preset.get('X',110)} Y{preset.get('Y',110)} Z{preset.get('Z',125)} E0\n")
                f.write(f"G1 X0 Y0 Z0 F3000\n")
            else:
                f.write(stripped + '\n')

    moves, fan_events = planner.parse_gcode(clean_gcode_path)
    if os.path.exists(clean_gcode_path): os.remove(clean_gcode_path)

    if not moves:
        print("No moves found in G-code.")
        return

    total_duration = sum(m.accel_t + m.cruise_t + m.decel_t
                         for m in moves if m.is_kinematic_move)
    
    total_samples = int(total_duration * SAMPLE_RATE) + 100 # bit of buffer
    final_audio = np.zeros((total_samples, 2), dtype=np.float32)
    
    # Setup Fans
    psu_fan = Fan(SAMPLE_RATE, vol=0.15, max_rpm=2000)
    hotend_fan = Fan(SAMPLE_RATE, vol=0.20, max_rpm=4000)
    part_cooling_fan = Fan(SAMPLE_RATE, vol=0.8, max_rpm=6000)
    psu_fan.set_speed(1.0)
    hotend_fan.set_speed(1.0)

    resonance_model = ResonanceFilter(SAMPLE_RATE)
    kinematics = 'corexy' if force_corexy else preset.get('kinematics', 'cartesian')
    motor_vol, extruder_vol = 0.55, 0.45
    last_phases = {'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'E': 0.0,
                   'A': 0.0, 'B': 0.0, 'TA': 0.0, 'TB': 0.0, 'TC': 0.0}
    fan_event_idx = 0

    # Precompute delta tower geometry
    if kinematics == 'delta':
        _r   = preset.get('tower_radius', 100.0)
        _arm2 = preset.get('arm_length', 220.0) ** 2
        delta_towers = [
            ('TA', _r * math.cos(math.radians(210)), _r * math.sin(math.radians(210))),
            ('TB', _r * math.cos(math.radians(330)), _r * math.sin(math.radians(330))),
            ('TC', _r * math.cos(math.radians(90)),  _r * math.sin(math.radians(90))),
        ]
    else:
        delta_towers = None
        _arm2 = None
    
    curr_t = 0.0
    curr_sample = 0
    print("Step 2: Synthesizing audio...")
    for m in tqdm(moves, desc="Processing Moves"):
        move_duration = m.accel_t + m.cruise_t + m.decel_t

        while fan_event_idx < len(fan_events) and fan_events[fan_event_idx]['time'] <= curr_t + move_duration:
            part_cooling_fan.set_speed(fan_events[fan_event_idx]['speed'])
            fan_event_idx += 1

        if not m.is_kinematic_move:
            continue

        phases_data = [
            (m.accel_t, m.start_v, m.cruise_v),
            (m.cruise_t, m.cruise_v, m.cruise_v),
            (m.decel_t, m.cruise_v, m.end_v)
        ]

        move_start_pos = np.array(m.start_pos)
        axes_r = np.array(m.axes_r)
        dist_traveled = 0.0

        if kinematics == 'corexy':
            motor_ratios = [
                ('A', abs(float(axes_r[0]) + float(axes_r[1]))),
                ('B', abs(float(axes_r[0]) - float(axes_r[1]))),
                ('Z', abs(float(axes_r[2]))),
                ('E', abs(float(axes_r[3]))),
            ]
        elif kinematics == 'delta':
            motor_ratios = None  # towers handled separately (position-dependent)
        else:
            motor_ratios = [
                ('X', abs(float(axes_r[0]))),
                ('Y', abs(float(axes_r[1]))),
                ('Z', abs(float(axes_r[2]))),
                ('E', abs(float(axes_r[3]))),
            ]

        for duration, v0, v1 in phases_data:
            if duration <= 1e-6: continue

            num_samples = int(duration * SAMPLE_RATE)
            if num_samples <= 0:
                curr_t += duration
                continue

            s_start = curr_sample
            s_end = s_start + num_samples
            
            if s_end > final_audio.shape[0]:
                s_end = final_audio.shape[0]
                num_samples = s_end - s_start
                if num_samples <= 0: break

            t_profile = np.linspace(0, duration, num_samples)
            v_profile = v0 + (v1 - v0) * (t_profile / duration)
            
            # Cumulative distance in this phase
            d_in_phase = np.cumsum(v_profile / SAMPLE_RATE)
            
            # Current X position for panning
            phase_start_dist = dist_traveled
            x_positions = move_start_pos[0] + axes_r[0] * (phase_start_dist + d_in_phase)
            pan = np.clip(x_positions / 220.0, 0.1, 0.9)
            
            seg_mono = np.zeros(num_samples, dtype=np.float32)
            fans_audio = psu_fan.generate_audio(num_samples) + \
                         hotend_fan.generate_audio(num_samples) + \
                         part_cooling_fan.generate_audio(num_samples)

            OS = 4
            OS_SR = SAMPLE_RATE * OS
            base_freqs = {'X': 70, 'Y': 75, 'Z': 35, 'A': 70, 'B': 75, 'E': 150}
            v_profile_os = np.linspace(v0, v1, num_samples * OS, dtype=np.float32)

            if kinematics == 'delta':
                # Tower velocities are position-dependent — compute at normal rate
                # then interpolate up before synthesis
                x_pos = move_start_pos[0] + axes_r[0] * (dist_traveled + d_in_phase)
                y_pos = move_start_pos[1] + axes_r[1] * (dist_traveled + d_in_phase)
                vx = v_profile * float(axes_r[0])
                vy = v_profile * float(axes_r[1])
                vz = v_profile * float(axes_r[2])
                t_os = np.arange(num_samples * OS) * (1.0 / OS)
                t_normal = np.arange(num_samples, dtype=np.float64)
                for tower_key, tx, ty in delta_towers:
                    dx = x_pos - tx
                    dy = y_pos - ty
                    d_vert = np.sqrt(np.maximum(_arm2 - dx**2 - dy**2, 1.0))
                    tower_v = np.abs((-dx * vx - dy * vy) / d_vert + vz)
                    tower_v_os = np.interp(t_os, t_normal, tower_v).astype(np.float32)
                    freq_os = 70.0 + tower_v_os * 10
                    phases_os = last_phases[tower_key] + np.cumsum(2 * np.pi * freq_os / OS_SR)
                    last_phases[tower_key] = float(phases_os[-1]) % (2 * np.pi)
                    wav_os = generate_stepper_waveform(phases_os) * motor_vol
                    seg_mono += resample_poly(wav_os, 1, OS).astype(np.float32)[:num_samples]
                # Extruder (independent of tower geometry)
                e_r = abs(float(axes_r[3]))
                if e_r > 1e-6:
                    e_v_os = v_profile_os * e_r
                    freq_os = 150.0 + e_v_os * 10
                    phases_os = last_phases['E'] + np.cumsum(2 * np.pi * freq_os / OS_SR)
                    last_phases['E'] = float(phases_os[-1]) % (2 * np.pi)
                    wav_os = generate_extruder_waveform(phases_os) * extruder_vol
                    seg_mono += resample_poly(wav_os, 1, OS).astype(np.float32)[:num_samples]
            else:
                for motor_key, motor_r in motor_ratios:
                    motor_v_os = v_profile_os * motor_r
                    if np.max(motor_v_os) < 1e-6: continue
                    freq_os = base_freqs[motor_key] + motor_v_os * 10
                    phases_os = last_phases[motor_key] + np.cumsum(2 * np.pi * freq_os / OS_SR)
                    last_phases[motor_key] = float(phases_os[-1]) % (2 * np.pi)
                    if motor_key == 'E':
                        wav_os = generate_extruder_waveform(phases_os) * extruder_vol
                    else:
                        wav_os = generate_stepper_waveform(phases_os) * motor_vol
                    wav = resample_poly(wav_os, 1, OS).astype(np.float32)[:num_samples]
                    if motor_key == 'X':
                        final_audio[s_start:s_end, 0] += wav * (1 - pan)
                        final_audio[s_start:s_end, 1] += wav * pan
                    else:
                        seg_mono += wav
            
            final_audio[s_start:s_end, 0] += seg_mono + fans_audio
            final_audio[s_start:s_end, 1] += seg_mono + fans_audio
            
            curr_t += duration
            curr_sample += num_samples
            dist_traveled += (v0 + v1) * 0.5 * duration

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
    parser.add_argument("--corexy", action="store_true", help="Force CoreXY kinematics regardless of preset")

    args = parser.parse_args()

    gcode_to_audio(args.gcode, str(Path(args.gcode).with_suffix(".wav")), printer_name=args.printer, force_corexy=args.corexy)
