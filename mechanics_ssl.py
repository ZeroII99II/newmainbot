# mechanics_ssl.py — SSL mechanics detectors / telemetry (best-effort heuristics)
import math, numpy as np
from ones_profile import ONES
from boost_pathing import SMALL_PADS


def _hyp2(x, y):
    return float(math.hypot(x, y))


def _ang_norm(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _vec2(x, y):
    return np.array([float(x), float(y)], dtype=np.float32)


def _yaw(rot):
    return float(rot.yaw)


def _pitch(rot):
    return float(rot.pitch)


def _roll(rot):
    return float(rot.roll)


class SkillTelemetry:
    """
    Produces a dict of features indicating attempts / successes for SSL-relevant mechanics.
    Heuristic & tolerant: robust to missing fields; never crashes.
    """

    def __init__(self):
        # memory across ticks
        self._last_touch_me = False
        self._last_ball_loc = np.zeros(3, dtype=np.float32)
        self._last_my_on_wall = False
        self._last_on_ceiling = False
        self._air_chain_touch = 0
        self._ground_carry_ticks = 0
        self._dribble_ticks = 0
        self._recent_powerslide = 0
        self._recent_wavedash = 0
        self._recent_chain_dash = 0
        self._recent_stall = 0
        self._recent_fake = 0
        self._touch_quality_prev_speed = 0.0

    @staticmethod
    def _is_on_wall(loc, rot):
        # Rough wall test: near side walls/backboard and car not upright
        return (abs(loc.x) > 3000 or abs(loc.y) > 4800) and abs(_pitch(rot)) + abs(_roll(rot)) > 0.35

    @staticmethod
    def _is_on_ceiling(loc, has_wheel_contact):
        # Ceiling z ~ 2044; treat >1900 and wheel contact as ceiling
        try:
            return has_wheel_contact and loc.z > 1900
        except Exception:
            return loc.z > 1900

    @staticmethod
    def _ball_speed(ball):
        v = ball.physics.velocity
        return float(np.linalg.norm([v.x, v.y, v.z]))

    def update(self, packet, index):
        info = {}
        # accept pending external info from bot (e.g., exploit windows, finisher choice)
        try:
            pending = getattr(self, "_pending_info", {})
            if pending:
                info.update(pending)
                self._pending_info = {}
        except Exception:
            pass
        try:
            ball = packet.game_ball
            me = packet.game_cars[index]
            team = me.team
        except Exception:
            return {}

        # Basic vectors
        my_loc = me.physics.location
        my_rot = me.physics.rotation
        my_vel = me.physics.velocity
        ball_loc = ball.physics.location
        ball_vel = ball.physics.velocity
        my_xy = _vec2(my_loc.x, my_loc.y)
        ball_xy = _vec2(ball_loc.x, ball_loc.y)
        dist_xy = float(np.linalg.norm(ball_xy - my_xy))
        my_spd = float(np.linalg.norm([my_vel.x, my_vel.y, my_vel.z]))
        ball_spd = float(np.linalg.norm([ball_vel.x, ball_vel.y, ball_vel.z]))

        # --- Core mechanics ---
        # Fast aerial attempt: takeoff (no wheel contact) + strong vertical vel + heading toward ball
        try:
            airborne = (not me.has_wheel_contact) and my_loc.z > 150
        except Exception:
            airborne = my_loc.z > 150
        info["fast_aerial_attempt"] = 1.0 if (airborne and abs(my_vel.z) > 450 and dist_xy < 2300) else 0.0

        # Advanced recoveries: powerslide/wavedash/chain-dash proxies
        # We can't read inputs; use angular bursts + quick land-boost changes as hints
        ang = me.physics.angular_velocity
        ang_mag = float(np.linalg.norm([ang.x, ang.y, ang.z]))
        # count brief high-angle recoveries near ground
        if my_loc.z < 120 and (ang_mag > 3.8 or abs(_yaw(my_rot)) > 2.8):
            self._recent_powerslide = min(30, self._recent_powerslide + 1)
        else:
            self._recent_powerslide = max(0, self._recent_powerslide - 1)
        info["adv_recovery"] = 1.0 if self._recent_powerslide > 0 else 0.0

        # Wavedash/chain-dash proxies: jump impulses followed by ground speed spikes
        if my_loc.z < 40 and abs(my_vel.z) < 20 and my_spd > 1700:
            self._recent_wavedash = min(30, self._recent_wavedash + 1)
        else:
            self._recent_wavedash = max(0, self._recent_wavedash - 1)
        info["zap_chain_dash"] = 1.0 if self._recent_wavedash > 0 else 0.0

        # Wall play: touching ball near wall/backboard at height
        on_wall = self._is_on_wall(my_loc, my_rot)
        wall_play = on_wall and ball_loc.z > 200 and (abs(ball_loc.x) > 2800 or abs(ball_loc.y) > 4800)
        info["wall_play"] = 1.0 if wall_play else 0.0

        # Basic dribble & flicks: ball slow & near hood; then quick jump/dodge upward for power
        under_ball = (
            abs(ball_loc.x - my_loc.x) < 120
            and abs(ball_loc.y - my_loc.y) < 120
            and 60 < (ball_loc.z - my_loc.z) < 220
        )
        carrying = under_ball and ball_spd < 1200 and my_spd < 1700
        self._dribble_ticks = self._dribble_ticks + 1 if carrying else 0
        info["dribble_carry"] = 1.0 if self._dribble_ticks > 10 else 0.0  # ~> ~0.16s carry

        # Flick proxy: sudden positive ball dz after a short carry window
        dz = ball_vel.z
        info["flick_power"] = float(max(0.0, min(1.0, (dz - 550.0) / 900.0))) if self._dribble_ticks > 8 else 0.0

        # Shadow defense: align behind ball toward our goal, keeping spacing
        goal_y = -5120 if team == 0 else 5120
        to_goal = _vec2(0 - my_loc.x, goal_y - my_loc.y)
        to_ball = _vec2(ball_loc.x - my_loc.x, ball_loc.y - my_loc.y)
        dot = float(np.dot(to_goal, to_ball) / (np.linalg.norm(to_goal) * np.linalg.norm(to_ball) + 1e-6))
        shadow_angle = 0.5 * dot  # rough proxy
        shadow_dist_ok = 600 < dist_xy < 2200
        info["shadow_good"] = 1.0 if (shadow_dist_ok and shadow_angle > 0.5) else 0.0

        # Perfect first touches: first touch after a while that accelerates ball toward opponent goal with control (not blasted)
        try:
            lt = ball.latest_touch
            latest_touch_me = lt.player_index == index
        except Exception:
            latest_touch_me = False
        accel = max(0.0, ball_spd - self._ball_speed_prev if hasattr(self, "_ball_speed_prev") else 0.0)
        towards_opp = (ball_vel.y * (-1.0 if team == 0 else 1.0)) > 0
        controlled = ball_spd < 2500
        info["perfect_touch"] = 1.0 if (latest_touch_me and accel > 200 and towards_opp and controlled) else 0.0

        # Kickoff success proxy (used by director separately but we mirror here)
        info["kickoff_first_touch"] = (
            1.0
            if (
                latest_touch_me
                and packet.game_info.is_kickoff_pause == False
                and packet.game_info.seconds_elapsed < 3.0
            )
            else 0.0
        )

        # --- Advanced aerial mechanics ---
        # Air dribble: multiple midair touches within ~1.2s, z>600
        if airborne and ball_loc.z > 600 and latest_touch_me:
            self._air_chain_touch += 1
        elif not airborne:
            self._air_chain_touch = 0
        info["air_dribble_chain"] = float(min(1.0, self._air_chain_touch / 3.0))

        # Flip reset: midair contact near the car's wheels region & subsequent second dodge opportunity proxy
        # Heuristic: touch at z>900 with car pitched slightly up and vertical speed decreasing
        pitch = _pitch(my_rot)
        info["flip_reset_attempt"] = (
            1.0
            if (
                airborne and ball_loc.z > 900 and abs(pitch) < 0.6 and latest_touch_me and my_vel.z < 0
            )
            else 0.0
        )

        # Ceiling shot: was on ceiling then aerial touch soon after
        on_ceiling = self._is_on_ceiling(my_loc, getattr(me, "has_wheel_contact", False))
        info["ceiling_setup"] = (
            1.0 if (self._last_on_ceiling and airborne and ball_loc.z > 700) else 0.0
        )
        self._last_on_ceiling = on_ceiling

        # Musty & double tap proxies
        # Musty: backflip-ish contact (pitch negative) with ball above front after a small carry
        info["musty_attempt"] = (
            1.0
            if (
                latest_touch_me
                and self._dribble_ticks > 8
                and pitch < -0.4
                and ball_loc.z > my_loc.z + 120
            )
            else 0.0
        )
        # Double tap: backboard hit then next touch within ~1.2s
        backboard = (abs(ball_loc.y) > 5000 and ball_loc.z > 1200)
        if backboard:
            self._backboard_time = packet.game_info.seconds_elapsed
        last_bb = getattr(self, "_backboard_time", -9e9)
        info["double_tap_attempt"] = (
            1.0
            if (
                latest_touch_me
                and packet.game_info.seconds_elapsed - last_bb < 1.2
            )
            else 0.0
        )

        # Ultra-efficient reads: stalls (flip cancel midair) and deception proxies (approach then abort)
        # Stall proxy: airborne with low angular vel after flip (hard to observe; integrate loosely)
        angv = float(np.linalg.norm([ang.x, ang.y, ang.z]))
        info["stall_proxy"] = (
            1.0 if (airborne and angv < 1.0 and my_spd > 800 and ball_loc.z > 700) else 0.0
        )

        # Deception proxy: approach then sudden throttle-down causing opponent touch miss, followed by our possession
        # Without opponent state, use ball speed drop then our touch within 0.7s
        bs_drop = (
            self._ball_speed_prev - ball_spd if hasattr(self, "_ball_speed_prev") else 0.0
        )
        if bs_drop > 400 and latest_touch_me:
            self._recent_fake = min(30, self._recent_fake + 1)
        else:
            self._recent_fake = max(0, self._recent_fake - 1)
        info["deception_fake"] = 1.0 if self._recent_fake > 0 else 0.0

        # --- Recovery mastery detectors ---

        # Air-roll landing: recently airborne, then ground contact with nose-down pitch
        was_air = getattr(self, "_was_airborne", False)
        just_landed = (was_air and not airborne and my_loc.z < 70)
        air_roll_ok = just_landed and pitch > 0.15  # nose-down on landing
        info["air_roll_landing"] = 1.0 if air_roll_ok else 0.0
        self._was_airborne = airborne

        # Wave dash: ground contact after short hop + horizontal speed spike
        spd_xy = float(np.linalg.norm([my_vel.x, my_vel.y]))
        prev_spd_xy = getattr(self, "_prev_spd_xy", 0.0)
        self._prev_spd_xy = spd_xy
        wave_dash = (just_landed and (spd_xy - prev_spd_xy) > 300.0 and abs(my_vel.z) < 30)
        info["wave_dash_exec"] = 1.0 if wave_dash else 0.0

        # Half-flip proxy: quick 180 reversal + speed pickup within ~0.7s
        yaw = float(my_rot.yaw)
        prev_yaw = getattr(self, "_prev_yaw", yaw)
        d_yaw = abs(_ang_norm(yaw - prev_yaw))
        self._prev_yaw = yaw
        hf = (d_yaw > 2.5 and spd_xy > prev_spd_xy + 200.0)
        info["half_flip_exec"] = 1.0 if hf else 0.0

        # Ceiling reset usage: ceiling contact then stable aerial a moment later
        ceiling_touch = getattr(self, "_ceil_touch", False)
        now = packet.game_info.seconds_elapsed
        # consider contact if we were at z>1900 with wheel contact recently
        if self._is_on_ceiling(my_loc, getattr(me, "has_wheel_contact", False)):
            self._ceil_touch = True
            self._ceil_t = now
        elif getattr(self, "_ceil_touch", False) and now - getattr(self, "_ceil_t", 0.0) < 1.2 and airborne:
            info["ceiling_reset_exec"] = 1.0
            self._ceil_touch = False
        else:
            info["ceiling_reset_exec"] = 0.0

        # Goal ramp (inside net curve) pop/reset proxy: contact near y=±5120, x within 900, z increasing
        near_ramp = (abs(my_loc.y) > 4900 and abs(my_loc.x) < 900)
        ramp_pop = near_ramp and my_vel.z > 150
        info["net_ramp_reset_exec"] = 1.0 if ramp_pop else 0.0

        # Wall nose-down landing: on wall -> just grounded with positive pitch
        info["wall_nose_down"] = 1.0 if (self._last_my_on_wall and not on_wall and just_landed and pitch > 0.1) else 0.0
        self._last_my_on_wall = on_wall

        # Aggregate recovery score
        info["recovery_mastery"] = float(
            0.35 * info.get("air_roll_landing", 0.0) +
            0.30 * info.get("wave_dash_exec", 0.0) +
            0.25 * info.get("half_flip_exec", 0.0) +
            0.10 * info.get("wall_nose_down", 0.0)
        )

        # Small-pad pickup: if near any small-pad point this tick
        pads = SMALL_PADS
        p2 = np.array([my_loc.x, my_loc.y], dtype=np.float32)
        close = np.min(np.linalg.norm(pads - p2, axis=1))
        info["small_pad_pickup"] = 1.0 if close < ONES["pad_radius"] else 0.0

        # Low-50 proxy: under-ball, we jump, ball stays low but gains forward speed
        low50 = (under_ball and latest_touch_me and dz < 350 and abs(ball_vel.y) + abs(ball_vel.x) > 500)
        info["low50_success"] = 1.0 if low50 else 0.0

        # Air-dribble control: consecutive airborne self touches at z>600 with speed under control
        info["air_dribble_ctrl"] = float(min(1.0, info.get("air_dribble_chain", 0.0)))

        # Backboard save: our touch near own backboard at z>900 and ball exits toward corner (x moving outward)
        own_board = ( (team==0 and ball_loc.y < -4900) or (team==1 and ball_loc.y > 4900) )
        outward = abs(ball_vel.x) > 400
        backboard_save = (own_board and latest_touch_me and ball_loc.z > 900 and outward)
        info["backboard_save"] = 1.0 if backboard_save else 0.0

        # Corner clear quality after save: ball speed toward side + downward
        sideward = abs(ball_vel.x) > 700
        downward = ball_vel.z < -150
        info["corner_clear_quality"] = 1.0 if (backboard_save and sideward and downward) else 0.0

        # store for next tick
        self._ball_speed_prev = ball_spd
        self._last_touch_me = latest_touch_me
        # Exploit window mirror (from awareness)
        info["exploit_window"] = float(info.get("exploit_window", info.get("pressure_idx", 0.0) > 0.7))

        # Conversion attempt: if we had an exploit window and we produced a strong shot proxy
        # Shot proxy: ball velocity toward opponent goal increased after our (recent) interaction
        to_opp = 1.0 if ((team==0 and ball_loc.y>0) or (team==1 and ball_loc.y<0)) else 0.0
        speed_toward = (abs(ball_vel.y) + 0.001)
        info["conversion_attempt"] = 1.0 if (info.get("exploit_window",0.0) > 0.5 and to_opp and speed_toward > 1400) else 0.0

        # Success proxy: ball enters net soon after our attempt (coarse)
        # If you already track goals/own touches, replace with precise keys.
        goal_line = 5100 if team==0 else -5100
        scored = ( (team==0 and ball_loc.y > goal_line) or (team==1 and ball_loc.y < goal_line) )
        info["conversion_success"] = 1.0 if scored else 0.0

        # Finish variety (credit when finish choice changes)
        last_fin = getattr(self, "_last_finisher_label", "")
        cur_fin  = info.get("finisher_choice", "")
        info["finish_variety"] = 1.0 if (cur_fin and cur_fin != last_fin) else 0.0
        self._last_finisher_label = cur_fin

        return info

    # end update

