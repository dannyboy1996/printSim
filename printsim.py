import argparse
import math
import soundfile as sf
import numpy as np
import os
from tqdm import tqdm
from scipy.signal import butter, firwin
from pathlib import Path
import yaml
from klipper_planner import KlipperPlanner
from numba import njit, prange

# --- CONSTANTS & CONFIG ---
SAMPLE_RATE = 44100
PRESETS_FILE = "presets.yaml"
PRINTER_NAME = "default_printer"

# Frame ringing: (centre frequency, Q, gain)
RESONANCE_MODES = [(85, 30, 0.6), (120, 40, 0.4), (250, 20, 0.2)]

# Airflow noise shaping filter (pink-ish), shared by every fan.
FAN_NOISE_B = np.array([0.0499, -0.0959, 0.0506, -0.0044])
FAN_NOISE_A = np.array([1.0, -2.4949, 2.0172, -0.5221])

# Motor voice base frequencies (Hz at zero velocity)
BASE_FREQS = {'X': 70.0, 'Y': 75.0, 'Z': 35.0, 'A': 70.0, 'B': 75.0, 'E': 150.0}

PAN_WIDTH_MM = 220.0


# --- JIT SYNTHESIS KERNELS ---
#
# Everything below is compiled by numba the first time it runs (and cached on
# disk afterwards, so only the very first run pays for it). The kernels do the
# per-sample work the engine used to do with per-segment NumPy/scipy calls:
# a print is thousands of short segments, and the fixed cost of those calls
# dominated the actual arithmetic.

TWO_PI = 2.0 * math.pi
OS = 4              # motor synthesis oversampling factor
LUT_SIZE = 8192     # waveform table resolution (+1 guard point for interp)
CHUNK = 4096        # output samples rendered per inner block


def _make_lut(fn):
    """One period of a waveform, sampled for table lookup."""
    return fn(np.linspace(0.0, TWO_PI, LUT_SIZE + 1))


SIN_LUT = _make_lut(np.sin)
STEPPER_LUT = _make_lut(
    lambda p: (np.sin(p) + 0.2 * np.sin(p * 2)) / 1.2 + 0.5 * np.sin(p * 8))
EXTRUDER_LUT = _make_lut(
    lambda p: (np.sin(p) + 0.5 * np.sin(p * 3) + 0.2 * np.sin(p * 5)) / 1.7)

# Anti-alias filter for the OS -> 1 decimation. Same design that
# resample_poly(x, 1, OS) uses internally, so the decimated signal matches the
# pre-JIT engine; the difference is that here it runs as one continuous stream
# per voice instead of being restarted (and zero-padded) on every segment.
DECIM_FIR = firwin(2 * 10 * OS + 1, 1.0 / OS, window=('kaiser', 5.0))
NHIST = DECIM_FIR.size - 1   # oversampled input history carried per voice
FAN_STATE_SIZE = 8           # see synth_fans_block for the layout


@njit(cache=True, fastmath=True, inline='always')
def _lut_at(lut, phase):
    """Linearly interpolated table lookup; ``phase`` is radians in [0, 2pi)."""
    x = phase * (LUT_SIZE / TWO_PI)
    i = int(x)
    if i < 0:
        i = 0
    elif i >= LUT_SIZE:
        i = LUT_SIZE - 1
    f = x - i
    return lut[i] + f * (lut[i + 1] - lut[i])


@njit(cache=True, fastmath=True, inline='always')
def _wrap(ph):
    """Fold a phase back into [0, 2pi).

    A full fold, not a single subtraction: a delta tower whose effector sits
    outside the arm reach can ask for a frequency above the oversampled rate,
    and a phase step bigger than 2pi would otherwise run away and walk the
    table lookup off the end of the array.
    """
    if ph >= TWO_PI:
        ph -= TWO_PI * math.floor(ph * (1.0 / TWO_PI))
    return ph


@njit(cache=True, fastmath=True, inline='always')
def _dist_at(m, v0, dv, sample_rate):
    """Distance covered through sample ``m`` of a linear velocity ramp.

    Closed form of cumsum(v / sample_rate) for v[k] = v0 + dv * k.
    """
    return ((m + 1.0) * v0 + dv * m * (m + 1.0) * 0.5) / sample_rate


@njit(cache=True, fastmath=True)
def _decimate(buf, nout, fir, dst, dst_off):
    """Filter ``buf`` (NHIST history + nout*OS new samples) down by OS.

    The FIR is linear phase, so taps are applied in ascending order over
    contiguous memory. Output lags the input by half the filter (10 samples,
    0.2 ms), the same for every voice, so the mix stays aligned.
    """
    ntap = fir.size
    for m in range(nout):
        base = NHIST + m * OS + OS - ntap
        acc = 0.0
        for k in range(ntap):
            acc += fir[k] * buf[base + k]
        dst[dst_off + m] = np.float32(acc)


@njit(cache=True, fastmath=True)
def _silence(buf, hist, v, n, fir, dst, off):
    """Fill a segment where this voice is not moving.

    The filter still has to ring out from whatever came before; once its
    history is empty the rest of the segment is just zeros.
    """
    ringing = False
    for k in range(NHIST):
        if hist[v, k] != 0.0:
            ringing = True
            break
    m0 = 0
    if ringing:
        m0 = NHIST // OS + 1
        if m0 > n:
            m0 = n
        for k in range(NHIST):
            buf[k] = hist[v, k]
        for j in range(m0 * OS):
            buf[NHIST + j] = 0.0
        _decimate(buf, m0, fir, dst, off)
        for k in range(NHIST):
            hist[v, k] = buf[m0 * OS + k]
    for m in range(m0, n):
        dst[off + m] = np.float32(0.0)


@njit(cache=True, parallel=True, fastmath=True)
def synth_motors(ns, off, v0, v1, ratio, base_freq, is_extruder,
                 phase, hist, fir, stepper_lut, extruder_lut,
                 motor_vol, extruder_vol, sample_rate, vbuf):
    """Render one batch of segments for every motor voice.

    Voices are independent (own phase, own filter history), so they run in
    parallel; segments within a voice stay in order because the phase carries.
    """
    n_seg = ns.size
    n_voice = base_freq.size
    os_sr = sample_rate * OS
    for v in prange(n_voice):
        lut = extruder_lut if is_extruder[v] else stepper_lut
        vol = extruder_vol if is_extruder[v] else motor_vol
        bf = base_freq[v]
        buf = np.empty(NHIST + CHUNK * OS)
        ph = phase[v]
        dst = vbuf[v]
        for s in range(n_seg):
            n = ns[s]
            o = off[s]
            r = ratio[s, v]
            a0 = v0[s] * r
            a1 = v1[s] * r
            if max(abs(a0), abs(a1)) < 1e-6:
                _silence(buf, hist, v, n, fir, dst, o)
                continue

            nos = n * OS
            dv = (a1 - a0) / (nos - 1) if nos > 1 else 0.0
            m0 = 0
            while m0 < n:
                c = min(CHUNK, n - m0)
                for k in range(NHIST):
                    buf[k] = hist[v, k]
                for j in range(c * OS):
                    vel = a0 + dv * (m0 * OS + j)
                    ph = _wrap(ph + TWO_PI * (bf + vel * 10.0) / os_sr)
                    buf[NHIST + j] = _lut_at(lut, ph) * vol
                _decimate(buf, c, fir, dst, o + m0)
                for k in range(NHIST):
                    hist[v, k] = buf[c * OS + k]
                m0 += c
        phase[v] = ph


@njit(cache=True, fastmath=True)
def _tower_v(m, v0, dv, d0, x0, y0, arx, ary, arz, tx, ty, arm2, sample_rate):
    """Carriage speed of one delta tower at output sample ``m``.

    Differentiating the delta kinematics: tower height is
    sqrt(arm^2 - dx^2 - dy^2) + z, so its rate depends on where the effector
    currently is, not just on how fast it is moving.
    """
    d = d0 + _dist_at(m, v0, dv, sample_rate)
    dx = x0 + arx * d - tx
    dy = y0 + ary * d - ty
    vert = arm2 - dx * dx - dy * dy
    if vert < 1.0:
        vert = 1.0
    vert = math.sqrt(vert)
    vm = v0 + dv * m
    return abs((-dx * (vm * arx) - dy * (vm * ary)) / vert + vm * arz)


@njit(cache=True, parallel=True, fastmath=True)
def synth_delta(ns, off, v0, v1, x0, y0, arx, ary, arz, are, d0,
                tower_x, tower_y, arm2, phase, hist, fir,
                stepper_lut, extruder_lut, motor_vol, extruder_vol,
                sample_rate, vbuf):
    """Delta form of synth_motors: voices 0-2 are towers, voice 3 the extruder."""
    n_seg = ns.size
    os_sr = sample_rate * OS
    for v in prange(4):
        buf = np.empty(NHIST + CHUNK * OS)
        ph = phase[v]
        dst = vbuf[v]
        is_tower = v < 3
        tx = tower_x[v] if is_tower else 0.0
        ty = tower_y[v] if is_tower else 0.0
        lut = stepper_lut if is_tower else extruder_lut
        vol = motor_vol if is_tower else extruder_vol
        bf = 70.0 if is_tower else 150.0

        for s in range(n_seg):
            n = ns[s]
            o = off[s]
            a0 = v0[s]
            a1 = v1[s]
            r = 1.0 if is_tower else are[s]
            if r < 1e-6 or (is_tower and max(abs(a0), abs(a1)) < 1e-6):
                _silence(buf, hist, v, n, fir, dst, o)
                continue

            # tower speed is evaluated at output rate and interpolated up;
            # the extruder just rides the velocity ramp
            dv_out = (a1 - a0) / (n - 1) if n > 1 else 0.0
            nos = n * OS
            dv_os = (a1 * r - a0 * r) / (nos - 1) if nos > 1 else 0.0

            m0 = 0
            while m0 < n:
                c = min(CHUNK, n - m0)
                for k in range(NHIST):
                    buf[k] = hist[v, k]
                if is_tower:
                    tv0 = _tower_v(m0, a0, dv_out, d0[s], x0[s], y0[s],
                                   arx[s], ary[s], arz[s], tx, ty, arm2,
                                   sample_rate)
                    for m in range(c):
                        nxt = m0 + m + 1
                        if nxt > n - 1:
                            nxt = n - 1
                        tv1 = _tower_v(nxt, a0, dv_out, d0[s], x0[s], y0[s],
                                       arx[s], ary[s], arz[s], tx, ty, arm2,
                                       sample_rate)
                        for j in range(OS):
                            tv = tv0 + (tv1 - tv0) * (j / OS)
                            ph = _wrap(ph + TWO_PI * (bf + tv * 10.0) / os_sr)
                            buf[NHIST + m * OS + j] = _lut_at(lut, ph) * vol
                        tv0 = tv1
                else:
                    for j in range(c * OS):
                        vel = a0 * r + dv_os * (m0 * OS + j)
                        ph = _wrap(ph + TWO_PI * (bf + vel * 10.0) / os_sr)
                        buf[NHIST + j] = _lut_at(lut, ph) * vol
                _decimate(buf, c, fir, dst, o + m0)
                for k in range(NHIST):
                    hist[v, k] = buf[c * OS + k]
                m0 += c
        phase[v] = ph


@njit(cache=True, parallel=True, fastmath=True)
def mix_voices(vbuf, n_voice, ns, off, start, v0, v1, x0, arx, d0,
               pan_voice, pan_width, sample_rate, out):
    """Sum the voices into the stereo buffer, panning one voice by X position.

    Segments own disjoint stretches of the output, so they mix in parallel.
    """
    for s in prange(ns.size):
        n = ns[s]
        o = off[s]
        st = start[s]
        a0 = v0[s]
        dv = (v1[s] - a0) / (n - 1) if n > 1 else 0.0
        for m in range(n):
            mono = 0.0
            for v in range(n_voice):
                if v != pan_voice:
                    mono += vbuf[v, o + m]
            if pan_voice >= 0:
                d = d0[s] + _dist_at(m, a0, dv, sample_rate)
                pan = (x0[s] + arx[s] * d) / pan_width
                if pan < 0.1:
                    pan = 0.1
                elif pan > 0.9:
                    pan = 0.9
                w = vbuf[pan_voice, o + m]
                out[0, st + m] += np.float32(mono + w * (1.0 - pan))
                out[1, st + m] += np.float32(mono + w * pan)
            else:
                out[0, st + m] += np.float32(mono)
                out[1, st + m] += np.float32(mono)


@njit(cache=True, parallel=True, fastmath=True)
def synth_fans_block(params, ev_sample, ev_speed, starts, counts, nb, na,
                     sin_lut, sample_rate, n, noise, global_start,
                     states, scratch, out, out_off):
    """Render one block of every fan and add it to both channels.

    Per-fan state row: [started, speed, target, phase, z0, z1, z2, event idx].
    Speed ramps one sample at a time towards the target, so a fan command lands
    on its own sample instead of being quantized to the block.
    """
    n_fan = params.shape[0]
    for f in prange(n_fan):
        ramp_time = params[f, 1]
        blades = params[f, 2]
        max_rpm = params[f, 3]
        vol = params[f, 4]
        hnr = params[f, 5]
        if states[f, 0] == 0.0:
            states[f, 0] = 1.0
            states[f, 2] = params[f, 0]   # initial speed is the first target
        cur = states[f, 1]
        tgt = states[f, 2]
        ph = states[f, 3]
        z0 = states[f, 4]
        z1 = states[f, 5]
        z2 = states[f, 6]
        ei = int(states[f, 7])
        e_lo = starts[f]
        e_n = counts[f]
        step = 1.0 / (ramp_time * sample_rate) if ramp_time > 0 else 2.0

        for i in range(n):
            g = global_start + i
            while ei < e_n and ev_sample[e_lo + ei] <= g:
                tgt = ev_speed[e_lo + ei]
                ei += 1
            d = tgt - cur
            if d > step:
                d = step
            elif d < -step:
                d = -step
            cur += d

            # airflow: white noise through the shared shaping filter
            w = noise[f, i]
            y = nb[0] * w + z0
            z0 = nb[1] * w - na[1] * y + z1
            z1 = nb[2] * w - na[2] * y + z2
            z2 = nb[3] * w - na[3] * y

            # blade passing tone plus its second harmonic
            ph = _wrap(ph + TWO_PI * (max_rpm * cur / 60.0) * blades / sample_rate)
            ph2 = _wrap(ph * 2.0)
            hum = (_lut_at(sin_lut, ph) * 0.6
                   + _lut_at(sin_lut, ph2) * 0.25 * (0.4 + cur * 0.6))

            scratch[f, i] = (y * (1.0 - hnr) + hum * hnr) * cur * vol

        states[f, 1] = cur
        states[f, 2] = tgt
        states[f, 3] = ph
        states[f, 4] = z0
        states[f, 5] = z1
        states[f, 6] = z2
        states[f, 7] = ei

    for i in prange(n):
        acc = 0.0
        for f in range(n_fan):
            acc += scratch[f, i]
        out[0, out_off + i] += np.float32(acc)
        out[1, out_off + i] += np.float32(acc)


@njit(cache=True, parallel=True, fastmath=True)
def apply_resonance(out, b, a, gains):
    """Add the frame ringing: a bank of band-passes fed by the dry mix."""
    n_mode = gains.size
    n = out.shape[1]
    for c in prange(out.shape[0]):
        z = np.zeros((n_mode, 4))
        for i in range(n):
            x = np.float64(out[c, i])
            acc = 0.0
            for k in range(n_mode):
                y = b[k, 0] * x + z[k, 0]
                z[k, 0] = b[k, 1] * x - a[k, 1] * y + z[k, 1]
                z[k, 1] = b[k, 2] * x - a[k, 2] * y + z[k, 2]
                z[k, 2] = b[k, 3] * x - a[k, 3] * y + z[k, 3]
                z[k, 3] = b[k, 4] * x - a[k, 4] * y
                acc += y * gains[k]
            out[c, i] = np.float32(x + acc)


@njit(cache=True, parallel=True)
def peak_abs(out):
    """Largest absolute sample in the mix, for normalization."""
    peak = 0.0
    for i in prange(out.shape[1]):
        for c in range(out.shape[0]):
            peak = max(peak, abs(out[c, i]))
    return peak

class FanSpec:
    """Parameters of one fan. The audio itself is rendered by synth_fans_block."""

    def __init__(self, vol=1.0, max_rpm=4000, ramp_time=1.5, num_blades=7,
                 hum_to_noise_ratio=0.3, initial_speed=0.0, events=None):
        self.vol = vol
        self.max_rpm = max_rpm
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
    states = np.zeros((n_fan, FAN_STATE_SIZE))
    noise = np.empty((n_fan, block))
    scratch = np.empty((n_fan, block))

    total = out.shape[1]
    for i in range(0, total, block):
        m = min(block, total - i)
        # always fill the whole buffer so the kernel keeps one array type;
        # the tail block just ignores the samples past m
        rng.standard_normal((n_fan, block), out=noise)
        synth_fans_block(params, ev_sample, ev_speed, starts, counts,
                            FAN_NOISE_B, FAN_NOISE_A, SIN_LUT,
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
    hist = np.zeros((n_voice, NHIST))

    bounds = _batch_bounds(segs['ns'], batch_samples)
    vbuf = np.empty((n_voice, max(b[2] for b in bounds)), dtype=np.float32)
    ns_all = segs['ns']

    for lo, hi, span in tqdm(bounds, desc="Synthesizing audio"):
        sl = slice(lo, hi)
        # offsets of each segment inside the scratch buffer
        off = np.concatenate(([0], np.cumsum(ns_all[sl])[:-1])).astype(np.int64)
        if kinematics == 'delta':
            synth_delta(ns_all[sl], off, segs['v0'][sl], segs['v1'][sl],
                           segs['x0'][sl], segs['y0'][sl],
                           segs['arx'][sl], segs['ary'][sl],
                           segs['arz'][sl], segs['are'][sl], segs['d0'][sl],
                           tower_x, tower_y, arm2,
                           phase, hist, DECIM_FIR,
                           STEPPER_LUT, EXTRUDER_LUT,
                           motor_vol, extruder_vol, float(SAMPLE_RATE), vbuf)
        else:
            synth_motors(ns_all[sl], off, segs['v0'][sl], segs['v1'][sl],
                            np.ascontiguousarray(segs['ratio'][sl]),
                            base_freq, is_extruder,
                            phase, hist, DECIM_FIR,
                            STEPPER_LUT, EXTRUDER_LUT,
                            motor_vol, extruder_vol, float(SAMPLE_RATE), vbuf)

        mix_voices(vbuf, n_voice, ns_all[sl], off, segs['start'][sl],
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

    total_samples = int(total_duration * SAMPLE_RATE) + 100 # bit of buffer
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
    apply_resonance(out, b, a, gains)

    peak = peak_abs(out)
    gain = np.float32(1.0 / peak) if peak > 0 else np.float32(1.0)

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
    print(f"Done: {output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert G-code to printer sound (Motion Test).")

    parser.add_argument("gcode", help="Input G-code file")
    parser.add_argument("--printer", default=PRINTER_NAME, help=f"Printer preset from {PRESETS_FILE} (default: {PRINTER_NAME})")
    parser.add_argument("--corexy", action="store_true", help="Force CoreXY kinematics regardless of preset")

    args = parser.parse_args()

    gcode_to_audio(args.gcode, str(Path(args.gcode).with_suffix(".wav")), printer_name=args.printer, force_corexy=args.corexy)
