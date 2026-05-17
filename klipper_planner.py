import math

class Move:
    def __init__(self, toolhead, start_pos, end_pos, speed):
        self.toolhead = toolhead
        self.start_pos = tuple(start_pos)
        self.end_pos = tuple(end_pos)
        self.accel = toolhead.max_accel
        self.junction_deviation = toolhead.junction_deviation
        
        velocity = min(speed, toolhead.max_velocity)
        self.is_kinematic_move = True
        self.axes_d = axes_d = [ep - sp for sp, ep in zip(start_pos, end_pos)]
        self.move_d = move_d = math.sqrt(sum([d*d for d in axes_d[:3]]))
        
        if move_d < .000000001:
            # Extrude only move
            self.end_pos = ((start_pos[0], start_pos[1], start_pos[2])
                            + self.end_pos[3:])
            axes_d[0] = axes_d[1] = axes_d[2] = 0.
            self.move_d = move_d = max([abs(ad) for ad in axes_d[3:]])
            inv_move_d = 0.
            if move_d:
                inv_move_d = 1. / move_d
            self.accel = 99999999.9
            velocity = speed
            self.is_kinematic_move = False
        else:
            inv_move_d = 1. / move_d
        
        self.axes_r = [d * inv_move_d for d in axes_d]
        self.min_move_t = move_d / velocity
        self.max_start_v2 = 0.
        self.max_cruise_v2 = velocity**2
        self.delta_v2 = 2.0 * move_d * self.accel
        self.next_junction_v2 = 999999999.9
        
        # Setup for minimum_cruise_ratio checks
        self.max_mcr_start_v2 = 0.
        self.mcr_delta_v2 = 2.0 * move_d * toolhead.mcr_pseudo_accel

    def limit_speed(self, speed, accel):
        speed2 = speed**2
        if speed2 < self.max_cruise_v2:
            self.max_cruise_v2 = speed2
            self.min_move_t = self.move_d / speed
        self.accel = min(self.accel, accel)
        self.delta_v2 = 2.0 * self.move_d * self.accel
        self.mcr_delta_v2 = min(self.mcr_delta_v2, self.delta_v2)

    def limit_next_junction_speed(self, speed):
        self.next_junction_v2 = min(self.next_junction_v2, speed**2)

    def calc_junction(self, prev_move):
        if not self.is_kinematic_move or not prev_move.is_kinematic_move:
            return
        
        max_start_v2 = min([self.max_cruise_v2,
                            prev_move.max_cruise_v2, prev_move.next_junction_v2,
                            prev_move.max_start_v2 + prev_move.delta_v2])
        
        axes_r = self.axes_r
        prev_axes_r = prev_move.axes_r
        junction_cos_theta = -(axes_r[0] * prev_axes_r[0]
                               + axes_r[1] * prev_axes_r[1]
                               + axes_r[2] * prev_axes_r[2])
        
        sin_theta_d2 = math.sqrt(max(0.5*(1.0-junction_cos_theta), 0.))
        cos_theta_d2 = math.sqrt(max(0.5*(1.0+junction_cos_theta), 0.))
        one_minus_sin_theta_d2 = 1. - sin_theta_d2
        
        if one_minus_sin_theta_d2 > 0. and cos_theta_d2 > 0.:
            R_jd = sin_theta_d2 / one_minus_sin_theta_d2
            move_jd_v2 = R_jd * self.junction_deviation * self.accel
            pmove_jd_v2 = R_jd * prev_move.junction_deviation * prev_move.accel
            quarter_tan_theta_d2 = .25 * sin_theta_d2 / cos_theta_d2
            move_centripetal_v2 = self.delta_v2 * quarter_tan_theta_d2
            pmove_centripetal_v2 = prev_move.delta_v2 * quarter_tan_theta_d2
            max_start_v2 = min(max_start_v2, move_jd_v2, pmove_jd_v2,
                               move_centripetal_v2, pmove_centripetal_v2)
        
        self.max_start_v2 = max_start_v2
        self.max_mcr_start_v2 = min(
            max_start_v2, prev_move.max_mcr_start_v2 + prev_move.mcr_delta_v2)

    def set_junction(self, start_v2, cruise_v2, end_v2):
        half_inv_accel = .5 / self.accel
        accel_d = (cruise_v2 - start_v2) * half_inv_accel
        decel_d = (cruise_v2 - end_v2) * half_inv_accel
        cruise_d = self.move_d - accel_d - decel_d
        
        self.start_v = start_v = math.sqrt(start_v2)
        self.cruise_v = cruise_v = math.sqrt(cruise_v2)
        self.end_v = end_v = math.sqrt(end_v2)
        
        self.accel_t = accel_d / ((start_v + cruise_v) * 0.5) if (start_v + cruise_v) > 0 else 0
        self.cruise_t = cruise_d / cruise_v if cruise_v > 0 else 0
        self.decel_t = decel_d / ((end_v + cruise_v) * 0.5) if (end_v + cruise_v) > 0 else 0

class LookAheadQueue:
    def __init__(self):
        self.queue = []
    
    def flush(self, lazy=False):
        queue = self.queue
        flush_count = len(queue)
        if not flush_count: return []
        
        junction_info = [None] * flush_count
        next_start_v2 = next_mcr_start_v2 = peak_cruise_v2 = 0.
        pending_cv2_assign = 0
        
        for i in range(flush_count-1, -1, -1):
            move = queue[i]
            reachable_start_v2 = next_start_v2 + move.delta_v2
            start_v2 = min(move.max_start_v2, reachable_start_v2)
            cruise_v2 = None
            pending_cv2_assign += 1
            reach_mcr_start_v2 = next_mcr_start_v2 + move.mcr_delta_v2
            mcr_start_v2 = min(move.max_mcr_start_v2, reach_mcr_start_v2)
            
            if mcr_start_v2 < reach_mcr_start_v2:
                peak_cruise_v2 = (mcr_start_v2 + reach_mcr_start_v2) * .5
                cruise_v2 = min((start_v2 + reachable_start_v2) * .5
                                , move.max_cruise_v2, peak_cruise_v2)
                pending_cv2_assign = 0
            
            junction_info[i] = (move, start_v2, cruise_v2, next_start_v2)
            next_start_v2 = start_v2
            next_mcr_start_v2 = mcr_start_v2
        
        prev_cruise_v2 = 0.
        for i in range(flush_count):
            move, start_v2, cruise_v2, next_start_v2 = junction_info[i]
            if cruise_v2 is None:
                cruise_v2 = min(prev_cruise_v2, start_v2)
            move.set_junction(min(start_v2, cruise_v2), cruise_v2
                              , min(next_start_v2, cruise_v2))
            prev_cruise_v2 = cruise_v2
        
        res = list(self.queue)
        self.queue = []
        return res

    def add_move(self, move):
        self.queue.append(move)
        if len(self.queue) > 1:
            move.calc_junction(self.queue[-2])

class ToolHead:
    def __init__(self, max_velocity=300, max_accel=3000, scv=5, mcr=0.5):
        self.max_velocity = max_velocity
        self.max_accel = max_accel
        self.square_corner_velocity = scv
        self.min_cruise_ratio = mcr
        self.commanded_pos = [0., 0., 0., 0.]
        self.lookahead = LookAheadQueue()
        self._calc_junction_deviation()
    
    def _calc_junction_deviation(self):
        scv2 = self.square_corner_velocity**2
        self.junction_deviation = scv2 * (math.sqrt(2.) - 1.) / self.max_accel
        self.mcr_pseudo_accel = self.max_accel * (1. - self.min_cruise_ratio)
    
    def move(self, newpos, speed):
        move = Move(self, self.commanded_pos, newpos, speed)
        if not move.move_d:
            return None
        self.commanded_pos[:] = move.end_pos
        self.lookahead.add_move(move)
        return move

    def flush(self):
        return self.lookahead.flush()

class KlipperPlanner:
    def __init__(self, max_velocity=300, max_accel=1500, scv=5, mcr=0.5):
        self.toolhead = ToolHead(max_velocity, max_accel, scv, mcr)
        self.pos = [0.0, 0.0, 0.0, 0.0]
        self.feedrate = 3000.0 / 60.0
        self.relative = False
        self.e_relative = False
        # Firmware retraction state
        self.retract_length = 0.5
        self.retract_speed = 25.0
        self.unretract_extra_length = 0.0
        self.unretract_speed = 25.0
    
    def parse_gcode(self, gcode_file):
        raw_fan_events = []  # (queue_size_at_event, speed)
        with open(gcode_file, 'r') as f:
            for line in f:
                line = line.split(';')[0].strip().upper()
                if not line: continue
                parts = line.split()
                if not parts: continue
                cmd = parts[0]

                if cmd == 'G90':
                    self.relative = False
                elif cmd == 'G91':
                    self.relative = True
                elif cmd == 'M82':
                    self.e_relative = False
                elif cmd == 'M83':
                    self.e_relative = True
                elif cmd == 'G92':
                    if len(parts) == 1:
                        self.pos = [0., 0., 0., 0.]
                    else:
                        for p in parts[1:]:
                            if len(p) >= 1 and p[0] in 'XYZE':
                                idx = 'XYZE'.find(p[0])
                                try:
                                    self.pos[idx] = float(p[1:]) if p[1:] else 0.
                                except ValueError:
                                    pass
                elif cmd == 'M204':
                    accel = None
                    p_accel = t_accel = None
                    for p in parts[1:]:
                        if p.startswith('S'):
                            try: accel = float(p[1:])
                            except ValueError: pass
                        elif p.startswith('P'):
                            try: p_accel = float(p[1:])
                            except ValueError: pass
                        elif p.startswith('T'):
                            try: t_accel = float(p[1:])
                            except ValueError: pass
                    if accel is not None:
                        self.toolhead.max_accel = accel
                    elif p_accel is not None or t_accel is not None:
                        vals = [v for v in [p_accel, t_accel] if v is not None]
                        self.toolhead.max_accel = min(vals)
                    self.toolhead._calc_junction_deviation()
                elif cmd == 'SET_VELOCITY_LIMIT':
                    for p in parts[1:]:
                        kv = p.split('=')
                        if len(kv) != 2: continue
                        key, val_str = kv
                        try:
                            val = float(val_str)
                            if key == 'VELOCITY': self.toolhead.max_velocity = val
                            elif key == 'ACCEL': self.toolhead.max_accel = val
                            elif key == 'SQUARE_CORNER_VELOCITY': self.toolhead.square_corner_velocity = val
                            elif key == 'MINIMUM_CRUISE_RATIO': self.toolhead.min_cruise_ratio = val
                        except ValueError: pass
                    self.toolhead._calc_junction_deviation()
                elif cmd == 'G10':
                    new_pos = list(self.pos)
                    new_pos[3] -= self.retract_length
                    self.toolhead.move(new_pos, self.retract_speed)
                    self.pos = new_pos
                elif cmd == 'G11':
                    new_pos = list(self.pos)
                    new_pos[3] += self.retract_length + self.unretract_extra_length
                    self.toolhead.move(new_pos, self.unretract_speed)
                    self.pos = new_pos
                elif cmd == 'M106':
                    s = 255.
                    for p in parts[1:]:
                        if p.startswith('S'):
                            try: s = float(p[1:])
                            except ValueError: pass
                    raw_fan_events.append((len(self.toolhead.lookahead.queue), s / 255.0))
                elif cmd == 'M107':
                    raw_fan_events.append((len(self.toolhead.lookahead.queue), 0.0))
                elif cmd in ['G0', 'G1', 'G2', 'G3']:
                    new_pos = list(self.pos)
                    arc_x, arc_y, arc_z = None, None, None
                    arc_i, arc_j = None, None
                    arc_r = None
                    arc_e = None
                    move_feedrate = self.feedrate

                    for p in parts[1:]:
                        if len(p) < 2: continue
                        char = p[0]
                        try:
                            val = float(p[1:])
                        except ValueError:
                            continue

                        if char == 'X': arc_x = val
                        elif char == 'Y': arc_y = val
                        elif char == 'Z': arc_z = val
                        elif char == 'I': arc_i = val
                        elif char == 'J': arc_j = val
                        elif char == 'R': arc_r = val
                        elif char == 'E': arc_e = val
                        elif char == 'F': move_feedrate = val / 60.0

                    if arc_x is not None: new_pos[0] = self.pos[0] + arc_x if self.relative else arc_x
                    if arc_y is not None: new_pos[1] = self.pos[1] + arc_y if self.relative else arc_y
                    if arc_z is not None: new_pos[2] = self.pos[2] + arc_z if self.relative else arc_z
                    if arc_e is not None: new_pos[3] = self.pos[3] + arc_e if self.e_relative else arc_e

                    if cmd in ['G0', 'G1']:
                        self.feedrate = move_feedrate
                        self.toolhead.move(new_pos, move_feedrate)
                        self.pos = new_pos
                    else:
                        self._add_arc_moves(cmd == 'G2', new_pos, arc_i, arc_j, arc_r, arc_e, move_feedrate)
                        self.pos = new_pos

        planned_moves = self.toolhead.flush()

        # Build cumulative time index so fan events get accurate timestamps
        cum_times = [0.0] * (len(planned_moves) + 1)
        for i, m in enumerate(planned_moves):
            cum_times[i + 1] = cum_times[i] + m.accel_t + m.cruise_t + m.decel_t

        fan_events = [
            {'time': cum_times[min(qs, len(planned_moves))], 'speed': speed}
            for qs, speed in raw_fan_events
        ]

        return planned_moves, fan_events

    def _add_arc_moves(self, is_clockwise, end_pos, offset_i, offset_j, radius, extruder_e, feedrate, segment_length=1.0):
        start_x, start_y, start_z, start_e = self.pos
        end_x, end_y, end_z, end_e = end_pos

        # Determine arc center (cx, cy)
        if offset_i is not None and offset_j is not None:
            cx = start_x + offset_i
            cy = start_y + offset_j
        elif radius is not None:
            print("WARNING: R parameter for arcs is not fully implemented. Using linear move for now.")
            self.toolhead.move(end_pos, feedrate)
            return
        else:
            print("WARNING: Arc command without I, J, or R offsets. Using linear move for now.")
            self.toolhead.move(end_pos, feedrate)
            return

        # Calculate initial and final angles
        start_angle = math.atan2(start_y - cy, start_x - cx)
        end_angle = math.atan2(end_y - cy, end_x - cx)

        # Ensure angles are positive
        if start_angle < 0: start_angle += 2 * math.pi
        if end_angle < 0: end_angle += 2 * math.pi

        # Calculate sweep angle
        sweep_angle = end_angle - start_angle
        if is_clockwise:
            if sweep_angle >= 0:
                sweep_angle -= 2 * math.pi
        else: # Counter-clockwise
            if sweep_angle <= 0:
                sweep_angle += 2 * math.pi
        
        # Calculate arc radius
        arc_radius = math.sqrt((start_x - cx)**2 + (start_y - cy)**2)
        
        arc_length = abs(sweep_angle * arc_radius)
        if arc_length < 1e-6:
            self.toolhead.move(end_pos, feedrate)
            return

        # Determine number of segments
        num_segments = max(2, int(arc_length / segment_length))
        
        prev_x, prev_y, prev_z, prev_e = start_x, start_y, start_z, start_e

        for i in range(1, num_segments + 1):
            ratio = i / num_segments
            angle = start_angle + sweep_angle * ratio
            
            # Calculate intermediate X, Y
            curr_x = cx + arc_radius * math.cos(angle)
            curr_y = cy + arc_radius * math.sin(angle)
            
            # Interpolate Z and E linearly
            curr_z = start_z + (end_z - start_z) * ratio
            curr_e = start_e + (extruder_e - start_e) * ratio if extruder_e is not None else start_e

            segment_end_pos = [curr_x, curr_y, curr_z, curr_e]
            self.toolhead.move(segment_end_pos, feedrate)
            prev_x, prev_y, prev_z, prev_e = curr_x, curr_y, curr_z, curr_e

