import argparse
import math
import soundfile as sf
import numpy as np
import os
from tqdm import tqdm
from scipy.signal import butter
from pathlib import Path
import yaml
from klipper_planner import KlipperPlanner
import sound_kernels as sk

# --- CONSTANTS & CONFIG ---
SAMPLE_RATE = 44100
PRESETS_FILE = "presets.yaml"
PRINTER_NAME = "default_printer"

# Frame ringing: (centre frequency, Q, gain)
RESONANCE_MODES = [(85, 30, 0.6), (120, 40, 0.4), (250, 20, 0.2)]

<<<<<<< HEAD
# Airflow noise shaping filter (pink-ish), shared by every fan.
FAN_NOISE_B = np.array([0.0499, -0.0959, 0.0506, -0.0044])
FAN_NOISE_A = np.array([1.0, -2.4949, 2.0172, -0.5221])

# Motor voice base frequencies (Hz at zero velocity)
BASE_FREQS = {'X': 70.0, 'Y': 75.0, 'Z': 35.0, 'A': 70.0, 'B': 75.0, 'E': 150.0}

PAN_WIDTH_MM = 220.0


class FanSpec:
    """Parameters of one fan. The audio itself is rendered by sound_kernels."""

    def __init__(self, vol=1.0, max_rpm=4000, ramp_time=1.5, num_blades=7,
                 hum_to_noise_ratio=0.3, initial_speed=0.0, events=None):
        self.vol = vol
        self.max_rpm = max_rpm
=======
class Fan:
    """Simulates a fan based on mathematical principles."""
    def __init__(self, sample_rate, ramp_time=1.5, num_blades=7, max_rpm=4000, vol=1.0, hum_to_noise_ratio=0.3):
        self.sample_rate = sample_rate
>>>>>>> 94a1576671dc886884280d66625d7dcc0ce2ab18
        self.ramp_time = ramp_time
        self.num_blades = num_blades
        self.hum_to_noise_ratio = hum_to_noise_ratio
        self.initial_speed = initial_speed
        # (sample index, speed) changes; empty for fans that just run flat out
        self.ev_sample, self.ev_speed = events or (np.zeros(0, dtype=np.int64),
                                                   np.zeros(0))

    def params(self):
        return [self.initial_speed, self.ramp_time, self.num_blades,
                self.max_rpm, self.vol, self.hum_to_noise_ratio]


def render_fans(fans, out, sample_rate=SAMPLE_RATE, block=1 << 20, rng=None):
    """Add every fan to both channels of ``out``, one block at a time.

    White noise comes from NumPy's vectorized generator (much faster than
    drawing normals one at a time inside the kernel) and the fans themselves
    are rendered in parallel. Working in blocks keeps the noise and mix buffers
    cache-sized instead of allocating another full-length track.
    """
    rng = rng or np.random.default_rng()
    n_fan = len(fans)
    params = np.array([f.params() for f in fans], dtype=np.float64)
    ev_sample = np.concatenate([f.ev_sample for f in fans]).astype(np.int64)
    ev_speed = np.concatenate([f.ev_speed for f in fans]).astype(np.float64)
    counts = np.array([len(f.ev_sample) for f in fans], dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    states = np.zeros((n_fan, sk.FAN_STATE_SIZE))
    noise = np.empty((n_fan, block))
    scratch = np.empty((n_fan, block))

    total = out.shape[1]
    for i in range(0, total, block):
        m = min(block, total - i)
        # always fill the whole buffer so the kernel keeps one array type;
        # the tail block just ignores the samples past m
        rng.standard_normal((n_fan, block), out=noise)
        sk.synth_fans_block(params, ev_sample, ev_speed, starts, counts,
                            FAN_NOISE_B, FAN_NOISE_A, sk.SIN_LUT,
                            float(sample_rate), m, noise, i, states, scratch,
                            out, i)


def resonance_coefficients(sample_rate=SAMPLE_RATE):
    """Band-pass bank simulating the printer frame ringing."""
    nyquist = 0.5 * sample_rate
    b_all, a_all, gains = [], [], []
    for freq, Q, gain in RESONANCE_MODES:
        low = max(1, freq - freq / (2 * Q))
        high = min(nyquist - 1, freq + freq / (2 * Q))
        if low >= high:
            high = low + 1
        b, a = butter(2, [low, high], btype='band', fs=sample_rate)
        b_all.append(b)
        a_all.append(a)
        gains.append(gain)
    return (np.array(b_all), np.array(a_all), np.array(gains))


# --- MOTION LAYOUT ---

def _plan_segments(moves, fan_events, kinematics, total_samples):
    """Flatten the planned moves into flat per-segment arrays for the kernels.

    One segment is one constant-acceleration phase (accel / cruise / decel) of
    one move. Everything the synthesis kernels need is packed into parallel
    NumPy arrays so the whole file can be rendered without returning to Python.

    Returns (segments dict, fan event sample indices, fan event speeds).
    """
    max_seg = 3 * len(moves)
    ns_arr = np.zeros(max_seg, dtype=np.int64)
    start_arr = np.zeros(max_seg, dtype=np.int64)
    v0_arr = np.zeros(max_seg)
    v1_arr = np.zeros(max_seg)
    ratio_arr = np.zeros((max_seg, 4))
    x0_arr = np.zeros(max_seg)
    y0_arr = np.zeros(max_seg)
    arx_arr = np.zeros(max_seg)
    ary_arr = np.zeros(max_seg)
    arz_arr = np.zeros(max_seg)
    are_arr = np.zeros(max_seg)
    d0_arr = np.zeros(max_seg)

    fan_ev_sample, fan_ev_speed = [], []

    n_seg = 0
    curr_t = 0.0
    curr_sample = 0
    fan_event_idx = 0
    n_fan_events = len(fan_events)
    out_of_room = False

    for m in tqdm(moves, desc="Laying out moves"):
        move_duration = m.accel_t + m.cruise_t + m.decel_t

        while (fan_event_idx < n_fan_events
               and fan_events[fan_event_idx]['time'] <= curr_t + move_duration):
            fan_ev_sample.append(curr_sample)
            fan_ev_speed.append(float(np.clip(
                fan_events[fan_event_idx]['speed'], 0.0, 1.0)))
            fan_event_idx += 1

        if not m.is_kinematic_move or out_of_room:
            continue

        axes_r = m.axes_r
        rx, ry, rz, re = (float(axes_r[0]), float(axes_r[1]),
                          float(axes_r[2]), float(axes_r[3]))

        if kinematics == 'corexy':
            ratios = (abs(rx + ry), abs(rx - ry), abs(rz), abs(re))
        else:
            ratios = (abs(rx), abs(ry), abs(rz), abs(re))

        dist_traveled = 0.0
        for duration, v0, v1 in ((m.accel_t, m.start_v, m.cruise_v),
                                 (m.cruise_t, m.cruise_v, m.cruise_v),
                                 (m.decel_t, m.cruise_v, m.end_v)):
            if duration <= 1e-6:
                continue
            num_samples = int(duration * SAMPLE_RATE)
            if num_samples <= 0:
                curr_t += duration
                continue
            if curr_sample + num_samples > total_samples:
                num_samples = total_samples - curr_sample
                if num_samples <= 0:
                    out_of_room = True
                    break

            ns_arr[n_seg] = num_samples
            start_arr[n_seg] = curr_sample
            v0_arr[n_seg] = v0
            v1_arr[n_seg] = v1
            ratio_arr[n_seg, 0] = ratios[0]
            ratio_arr[n_seg, 1] = ratios[1]
            ratio_arr[n_seg, 2] = ratios[2]
            ratio_arr[n_seg, 3] = ratios[3]
            x0_arr[n_seg] = m.start_pos[0]
            y0_arr[n_seg] = m.start_pos[1]
            arx_arr[n_seg] = rx
            ary_arr[n_seg] = ry
            arz_arr[n_seg] = rz
            are_arr[n_seg] = abs(re)
            d0_arr[n_seg] = dist_traveled
            n_seg += 1

            curr_t += duration
            curr_sample += num_samples
            dist_traveled += (v0 + v1) * 0.5 * duration

    segs = {
        'n': n_seg,
        'ns': ns_arr[:n_seg], 'start': start_arr[:n_seg],
        'v0': v0_arr[:n_seg], 'v1': v1_arr[:n_seg],
        'ratio': np.ascontiguousarray(ratio_arr[:n_seg]),
        'x0': x0_arr[:n_seg], 'y0': y0_arr[:n_seg],
        'arx': arx_arr[:n_seg], 'ary': ary_arr[:n_seg],
        'arz': arz_arr[:n_seg], 'are': are_arr[:n_seg],
        'd0': d0_arr[:n_seg],
    }
    return (segs,
            np.array(fan_ev_sample, dtype=np.int64),
            np.array(fan_ev_speed, dtype=np.float64))


def _batch_bounds(seg_ns, target):
    """Split the segments into batches of roughly ``target`` output samples.

    Batches bound the size of the per-voice scratch buffer; a batch is always
    at least one segment long, however big that segment is.
    """
    bounds = []
    lo = 0
    span = 0
    for i, ns in enumerate(seg_ns):
        span += int(ns)
        if span >= target:
            bounds.append((lo, i + 1, span))
            lo, span = i + 1, 0
    if lo < len(seg_ns):
        bounds.append((lo, len(seg_ns), span))
    return bounds


def _render_motors(segs, kinematics, preset, out, motor_vol, extruder_vol,
                   batch_samples=1 << 20):
    """Run the synthesis kernels over the segment arrays.

    Work is done in batches: each batch renders its voices in parallel into a
    scratch buffer, then mixes them down. Phase and decimation-filter state
    live in arrays that persist across batches, so the result is identical to
    rendering the whole file in one go.
    """
    n_seg = segs['n']
    if n_seg == 0:
        return

    n_voice = 4
    if kinematics == 'delta':
        radius = preset.get('tower_radius', 100.0)
        arm2 = float(preset.get('arm_length', 220.0)) ** 2
        angles = (210.0, 330.0, 90.0)
        tower_x = np.array([radius * math.cos(math.radians(a)) for a in angles])
        tower_y = np.array([radius * math.sin(math.radians(a)) for a in angles])
        pan_voice = -1          # towers are not panned, as before
    else:
        keys = ('A', 'B', 'Z', 'E') if kinematics == 'corexy' else ('X', 'Y', 'Z', 'E')
        base_freq = np.array([BASE_FREQS[k] for k in keys])
        is_extruder = np.array([k == 'E' for k in keys])
        pan_voice = keys.index('X') if 'X' in keys else -1

    phase = np.zeros(n_voice)
    hist = np.zeros((n_voice, sk.NHIST))

    bounds = _batch_bounds(segs['ns'], batch_samples)
    vbuf = np.empty((n_voice, max(b[2] for b in bounds)), dtype=np.float32)
    ns_all = segs['ns']

    for lo, hi, span in tqdm(bounds, desc="Synthesizing audio"):
        sl = slice(lo, hi)
        # offsets of each segment inside the scratch buffer
        off = np.concatenate(([0], np.cumsum(ns_all[sl])[:-1])).astype(np.int64)
        if kinematics == 'delta':
            sk.synth_delta(ns_all[sl], off, segs['v0'][sl], segs['v1'][sl],
                           segs['x0'][sl], segs['y0'][sl],
                           segs['arx'][sl], segs['ary'][sl],
                           segs['arz'][sl], segs['are'][sl], segs['d0'][sl],
                           tower_x, tower_y, arm2,
                           phase, hist, sk.DECIM_FIR,
                           sk.STEPPER_LUT, sk.EXTRUDER_LUT,
                           motor_vol, extruder_vol, float(SAMPLE_RATE), vbuf)
        else:
            sk.synth_motors(ns_all[sl], off, segs['v0'][sl], segs['v1'][sl],
                            np.ascontiguousarray(segs['ratio'][sl]),
                            base_freq, is_extruder,
                            phase, hist, sk.DECIM_FIR,
                            sk.STEPPER_LUT, sk.EXTRUDER_LUT,
                            motor_vol, extruder_vol, float(SAMPLE_RATE), vbuf)

        sk.mix_voices(vbuf, n_voice, ns_all[sl], off, segs['start'][sl],
                      segs['v0'][sl], segs['v1'][sl], segs['x0'][sl],
                      segs['arx'][sl], segs['d0'][sl], pan_voice,
                      PAN_WIDTH_MM, float(SAMPLE_RATE), out)


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

<<<<<<< HEAD
    total_samples = int(total_duration * SAMPLE_RATE) + 100 # bit of buffer
=======
>>>>>>> 94a1576671dc886884280d66625d7dcc0ce2ab18
    kinematics = 'corexy' if force_corexy else preset.get('kinematics', 'cartesian')
    motor_vol, extruder_vol = 0.55, 0.45

    print("Step 2: Laying out motion segments...")
    segs, fan_ev_sample, fan_ev_speed = _plan_segments(
        moves, fan_events, kinematics, total_samples)

    # Channels are kept as rows of a (2, N) buffer: each row is contiguous for
    # the kernels, the two channels can be filtered in parallel, and the file
    # is interleaved in blocks at the end instead of copying the whole mix.
    out = np.zeros((2, total_samples), dtype=np.float32)

    print("Step 3: Synthesizing audio...")
    _render_motors(segs, kinematics, preset, out, motor_vol, extruder_vol)

    print("Step 4: Fans...")
    render_fans([
        FanSpec(vol=0.15, max_rpm=2000, initial_speed=1.0),   # PSU
        FanSpec(vol=0.20, max_rpm=4000, initial_speed=1.0),   # hotend
        FanSpec(vol=0.8, max_rpm=6000,                        # part cooling
                events=(fan_ev_sample, fan_ev_speed)),
    ], out)

    print("Step 5: Post-processing...")
    b, a, gains = resonance_coefficients(SAMPLE_RATE)
    sk.apply_resonance(out, b, a, gains)

    peak = sk.peak_abs(out)
    gain = np.float32(1.0 / peak) if peak > 0 else np.float32(1.0)

<<<<<<< HEAD
    # Normalize and interleave in blocks straight into the file, so a long
    # print never needs a second full-length copy of the mix in RAM.
    block = 1 << 20
    # subtype left to soundfile's default for the container (PCM_16 for .wav),
    # matching what sf.write() produced before
    with sf.SoundFile(output_file, 'w', SAMPLE_RATE, 2) as f:
        buf = np.empty((block, 2), dtype=np.float32)
        for i in range(0, total_samples, block):
            j = min(i + block, total_samples)
            n = j - i
            buf[:n, 0] = out[0, i:j]
            buf[:n, 1] = out[1, i:j]
            buf[:n] *= gain
            f.write(buf[:n])
=======
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

    max_val = np.max(np.abs(final_audio))
    if max_val > 0: final_audio /= max_val
    sf.write(output_file, final_audio, SAMPLE_RATE)
>>>>>>> 94a1576671dc886884280d66625d7dcc0ce2ab18
    print(f"Done: {output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert G-code to printer sound (Motion Test).")

    parser.add_argument("gcode", help="Input G-code file")
    parser.add_argument("--printer", default=PRINTER_NAME, help=f"Printer preset from {PRESETS_FILE} (default: {PRINTER_NAME})")
    parser.add_argument("--corexy", action="store_true", help="Force CoreXY kinematics regardless of preset")

    args = parser.parse_args()

    gcode_to_audio(args.gcode, str(Path(args.gcode).with_suffix(".wav")), printer_name=args.printer, force_corexy=args.corexy)
