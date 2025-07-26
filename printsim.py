import argparse
import math
import wave
import numpy as np
from gcodeparser import GcodeParser
from tqdm import tqdm
from scipy.signal import butter, lfilter

# --- PRINTER PHYSICS & FIRMWARE CONSTANTS ---
ACCELERATION = 1500  # mm/s^2. A realistic value for a modern printer.
HOMING_FEED_RATE = 3000
LEVELING_XY_FEED_RATE = 6000
LEVELING_Z_PROBE_FEED_RATE = 400
PROBE_POINTS = [ (30, 30), (190, 30), (190, 190), (30, 190) ]
BED_SIZE = {'X': 220, 'Y': 220, 'Z': 250}
PROBE_Z_HEIGHT, PROBE_DEPTH = 5, 5
GCODE_DURATIONS = { ('M', 109): 15.0, ('M', 190): 20.0 } # Heatup times

# --- SOUND ENGINE & RESONANCE MODEL ---
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

# --- FAN SIMULATION MODEL (MODIFIED FOR PHASE CONTINUITY) ---
class Fan:
    """Simulates a single fan with realistic spin-up and spin-down."""
    def __init__(self, sample_rate, ramp_time=1.5, base_blade_freq=80, vol=1.0, base_pitch=1.0):
        self.sample_rate = sample_rate
        self.ramp_time = ramp_time
        self.base_blade_freq = base_blade_freq
        self.base_pitch = base_pitch
        self.vol = vol
        self.current_speed = 0.0
        self.target_speed = 0.0
        self._noise_z = [0.0]  # State for the brown noise filter
        self.phase_accumulator = 0.0  # ADDED: To maintain phase continuity for the hum

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
        
        white_noise = np.random.randn(num_samples).astype(np.float32)
        avg_alpha = np.mean(0.995 - (speed_profile * 0.1))
        brown_noise, self._noise_z = lfilter([1 - avg_alpha], [1, -avg_alpha], white_noise, zi=self._noise_z)

        # --- MODIFIED PHASE CALCULATION ---
        blade_freq_profile = self.base_blade_freq * (0.5 + speed_profile * 0.5)
        phase_increments = 2 * np.pi * blade_freq_profile / self.sample_rate
        phase = self.phase_accumulator + np.cumsum(phase_increments)
        fan_hum = 0.2 * np.sin(phase)
        self.phase_accumulator = phase[-1] % (2 * np.pi)
        # --- END MODIFICATION ---
        
        final_noise = (brown_noise * 0.8 + fan_hum) * volume_profile
        return final_noise.astype(np.float32)

# --- HIGH-FIDELITY SOUND GENERATION ---
def generate_stepper_waveform(phase):
    return ((np.sin(phase) + 0.2 * np.sin(phase * 2)) / 1.2 + 0.5 * np.sin(phase * 8))

def generate_extruder_waveform(phase):
    return (np.sin(phase) + 0.5 * np.sin(phase * 3) + 0.2 * np.sin(phase * 5)) / 1.7

# --- PHYSICS-BASED DURATION CALCULATION ---
def calculate_move_duration(distance, feed_rate_mms):
    if distance <= 0 or feed_rate_mms <= 0: return 0.0
    time_to_accel = feed_rate_mms / ACCELERATION
    dist_to_accel = 0.5 * ACCELERATION * (time_to_accel ** 2)
    if distance < (2 * dist_to_accel):
        return math.sqrt(distance / ACCELERATION) * 2
    else:
        time_cruise = (distance - (2 * dist_to_accel)) / feed_rate_mms
        return (2 * time_to_accel) + time_cruise

# --- Main Program ---
def gcode_to_audio(gcode_file, output_file, sample_rate=44100):
    try:
        with open(gcode_file, 'r') as f: gcode = f.read()
    except FileNotFoundError: print(f"Error: The file '{gcode_file}' was not found."); return

    parser = GcodeParser(gcode)
    lines = parser.lines
    events = []
    last_pos = {'X': BED_SIZE['X']/2, 'Y': BED_SIZE['Y']/2, 'Z': BED_SIZE['Z']/2, 'E': 0}
    total_duration = 0
    rapid_feed_rate, last_feed_rate = 3000, 1500
    first_heat_time = -1

    print("Pass 1/2: Planning motion and events...")
    def add_event(type, time, duration, details):
        events.append({'type': type, 'start_time': time, 'duration': duration, 'details': details})

    for line in lines:
        if line.command == ('M', 106):
            add_event('fan_speed', total_duration, 0, {'speed': line.params.get('S', 255) / 255.0})
        elif line.command == ('M', 107):
            add_event('fan_speed', total_duration, 0, {'speed': 0.0})
        elif (line.command in [('M', 104), ('M', 109)]) and first_heat_time < 0 and line.params.get('S', 0) > 0:
            first_heat_time = total_duration
        
        elif line.command == ('G', 28):
            current_pos_for_event = last_pos.copy()
            for axis in ['X', 'Y', 'Z']:
                if last_pos[axis] != 0:
                    duration = calculate_move_duration(abs(last_pos[axis]), HOMING_FEED_RATE / 60.0)
                    add_event('move', total_duration, duration, {'feed_rate':HOMING_FEED_RATE, 'delta':{axis: -last_pos[axis]}, 'start_pos': current_pos_for_event, 'resets_phase': True})
                    total_duration += duration
                    last_pos[axis] = 0
        elif line.command == ('G', 29):
            for px, py in PROBE_POINTS:
                current_pos_for_event = last_pos.copy()
                dist = math.sqrt((px-last_pos['X'])**2 + (py-last_pos['Y'])**2)
                travel_dur = calculate_move_duration(dist, LEVELING_XY_FEED_RATE/60.0)
                if travel_dur > 0:
                    add_event('move', total_duration, travel_dur, {'feed_rate':LEVELING_XY_FEED_RATE, 'delta':{'X':px-last_pos['X'],'Y':py-last_pos['Y']}, 'start_pos': current_pos_for_event})
                    total_duration += travel_dur
                    last_pos.update({'X':px, 'Y':py})
                current_pos_for_event = last_pos.copy()
                probe_dur = calculate_move_duration(PROBE_DEPTH, LEVELING_Z_PROBE_FEED_RATE/60.0) * 2
                add_event('move', total_duration, probe_dur, {'feed_rate':LEVELING_Z_PROBE_FEED_RATE, 'delta':{'Z':-PROBE_DEPTH}, 'start_pos': current_pos_for_event})
                total_duration += probe_dur
        elif line.command in [('G', 0), ('G', 1)]:
            current_pos_for_event = last_pos.copy()
            new_pos = {axis: line.params.get(axis, last_pos[axis]) for axis in 'XYZE'}
            delta = {axis: new_pos[axis] - last_pos[axis] for axis in 'XYZE'}
            feed_rate = line.params.get('F', rapid_feed_rate if line.command==('G',0) else last_feed_rate)
            distance = math.sqrt(delta['X']**2 + delta['Y']**2 + delta['Z']**2)
            move_duration = calculate_move_duration(distance, feed_rate / 60.0)
            if move_duration > 0:
                is_travel = delta.get('E', 0) == 0
                add_event('move', total_duration, move_duration, {'feed_rate': feed_rate, 'delta': delta, 'start_pos': current_pos_for_event, 'resets_phase': is_travel})
                total_duration += move_duration
                last_pos = new_pos
                if line.params.get('F') is not None: last_feed_rate = feed_rate
        else:
            duration = GCODE_DURATIONS.get(line.command, 0)
            if duration > 0: total_duration += duration

    print(f"Analysis Complete. Realistic Duration: {total_duration/3600:.2f} hours")
    if total_duration <= 0: print("No audible events found."); return

    print("Pass 2/2: Synthesizing audio...")
    total_samples = int(total_duration * sample_rate)
    final_audio = np.zeros((total_samples, 2), dtype=np.float32)
    
    resonance_model = ResonanceFilter(sample_rate)
    motor_vol, extruder_vol, master_gain = 0.9, 0.55, 1.6

    psu_fan = Fan(sample_rate, ramp_time=3.0, base_blade_freq=70, vol=0.15, base_pitch=0.8)
    mainboard_fan = Fan(sample_rate, ramp_time=3.0, base_blade_freq=90, vol=0.12, base_pitch=1.0)
    hotend_fan = Fan(sample_rate, ramp_time=2.0, base_blade_freq=110, vol=0.20, base_pitch=1.2)
    part_cooling_fan = Fan(sample_rate, ramp_time=1.5, base_blade_freq=100, vol=0.25, base_pitch=1.1)
    psu_fan.set_speed(1.0)
    mainboard_fan.set_speed(1.0)
    hotend_fan_start_time = first_heat_time if first_heat_time >= 0 else total_duration
    
    events.sort(key=lambda x: x['start_time'])
    
    print("Pre-rendering motor sounds with dedicated panning...")
    last_phases = {'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'E': 0.0}

    for event in tqdm(events, unit='evt'):
        if event['type'] != 'move': continue
        
        start_sample = int(event['start_time'] * sample_rate)
        num_samples_seg = int(event['duration'] * sample_rate)
        end_sample = start_sample + num_samples_seg
        if num_samples_seg <= 0: continue

        details, delta, duration = event['details'], event['details']['delta'], event['duration']
        
        if details.get('resets_phase', False):
            last_phases = {k: 0.0 for k in last_phases}

        feed_rate = details['feed_rate']
        
        time_to_accel = (feed_rate/60.0) / ACCELERATION if ACCELERATION > 0 else 0
        v_profile = np.ones(num_samples_seg) * (feed_rate/60.0)
        accel_samples = int(time_to_accel * sample_rate)
        if accel_samples > 0 and accel_samples*2 < num_samples_seg:
            v_profile[:accel_samples] = np.linspace(0, feed_rate/60.0, accel_samples)
            v_profile[-accel_samples:] = np.linspace(feed_rate/60.0, 0, accel_samples)
        
        total_dist_sq = sum(d**2 for d in delta.values() if d is not None)
        total_dist = math.sqrt(total_dist_sq) if total_dist_sq > 0 else 0
        
        def get_phase(axis_delta, base_freq, initial_phase):
            if total_dist == 0 or axis_delta is None: return np.full(num_samples_seg, initial_phase)
            axis_v_profile = v_profile * (abs(axis_delta) / total_dist)
            return initial_phase + np.cumsum(2 * np.pi * (base_freq + axis_v_profile * 10) / sample_rate)

        phase_y = get_phase(delta.get('Y'), 75, last_phases['Y'])
        raw_y = generate_stepper_waveform(phase_y) * motor_vol if delta.get('Y') else np.zeros(num_samples_seg)
        last_phases['Y'] = phase_y[-1] % (2 * np.pi)
        
        phase_z = get_phase(delta.get('Z'), 35, last_phases['Z'])
        raw_z = generate_stepper_waveform(phase_z) * motor_vol if delta.get('Z') else np.zeros(num_samples_seg)
        last_phases['Z'] = phase_z[-1] % (2 * np.pi)

        phase_e = get_phase(delta.get('E'), 150, last_phases['E'])
        raw_e = generate_extruder_waveform(phase_e) * extruder_vol if delta.get('E') else np.zeros(num_samples_seg)
        last_phases['E'] = phase_e[-1] % (2 * np.pi)

        centered_mono = raw_y + raw_z + raw_e
        final_audio[start_sample:end_sample, 0] += centered_mono
        final_audio[start_sample:end_sample, 1] += centered_mono

        if delta.get('X'):
            phase_x = get_phase(delta.get('X'), 70, last_phases['X'])
            raw_x = generate_stepper_waveform(phase_x) * motor_vol
            last_phases['X'] = phase_x[-1] % (2 * np.pi)

            start_pos_x = details['start_pos']['X']
            end_pos_x = start_pos_x + delta.get('X', 0)
            start_pan = np.clip(start_pos_x / BED_SIZE['X'], 0.1, 0.9)
            end_pan = np.clip(end_pos_x / BED_SIZE['X'], 0.1, 0.9)
            pan_profile = np.linspace(start_pan, end_pan, num_samples_seg)

            final_audio[start_sample:end_sample, 0] += raw_x * (1 - pan_profile)
            final_audio[start_sample:end_sample, 1] += raw_x * pan_profile
    
    # --- MODIFIED: MEMORY-EFFICIENT MIXING LOOP ---
    print("Synthesizing fans, applying resonance, and mixing...")
    event_idx = 0
    chunk_size = 4096 # Process in chunks to save memory
    with tqdm(total=total_samples, unit='samp') as pbar:
        for current_sample in range(0, total_samples, chunk_size):
            end_sample = min(current_sample + chunk_size, total_samples)
            num_samples_in_chunk = end_sample - current_sample
            
            # Get a VIEW of the motor audio for this chunk. This does not copy memory.
            chunk_slice = final_audio[current_sample:end_sample]
            
            # Check for fan events that occur within this time window
            while event_idx < len(events) and int(events[event_idx]['start_time'] * sample_rate) < end_sample:
                event = events[event_idx]
                if event['type'] == 'fan_speed':
                    part_cooling_fan.set_speed(event['details']['speed'])
                event_idx += 1
            
            # Check if hotend fan should turn on
            if current_sample / sample_rate >= hotend_fan_start_time and hotend_fan.target_speed == 0:
                hotend_fan.set_speed(1.0)

            # Generate fan audio for just this chunk (phase is maintained by the Fan class)
            mixed_fans = (psu_fan.generate_audio(num_samples_in_chunk) + 
                          mainboard_fan.generate_audio(num_samples_in_chunk) + 
                          hotend_fan.generate_audio(num_samples_in_chunk) + 
                          part_cooling_fan.generate_audio(num_samples_in_chunk))
            
            # Add fans to the existing motor sounds IN-PLACE
            chunk_slice[:, 0] += mixed_fans
            chunk_slice[:, 1] += mixed_fans
            
            # Apply resonance filter IN-PLACE on the slice.
            # The filter is stateful, so it works correctly across chunks.
            # This avoids creating a massive new array for the filtered output.
            chunk_slice[:, 0] += resonance_model.process(chunk_slice[:, 0])
            chunk_slice[:, 1] += resonance_model.process(chunk_slice[:, 1])

            pbar.update(num_samples_in_chunk)

    print("Mastering audio and saving...")
    final_audio *= master_gain
    np.clip(final_audio, -1.0, 1.0, out=final_audio)
    
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((final_audio * 32767).astype(np.int16).tobytes())
    print("Conversion complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a G-code file to a highly realistic WAV audio file.")
    parser.add_argument("gcode_file", help="The input G-code file.")
    args = parser.parse_args()
    if not args.gcode_file.lower().endswith(('.gcode', '.gco', '.g')):
        print("Warning: Input file does not have a common G-code extension.")
    output_filename = '.'.join(args.gcode_file.rsplit('.', 1)[:-1]) + '.wav'
    gcode_to_audio(args.gcode_file, output_filename)