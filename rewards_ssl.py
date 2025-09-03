# rewards_ssl.py — SSL/pro shaping for mastery & reads
import numpy as np


DEFAULT_SSL_W = {
    # Core mastery
    "speedflip": 0.25,
    "adv_recovery": 0.25,
    "fast_aerial": 0.30,
    "wall_play": 0.20,
    "dribble_carry": 0.25,
    "flick_power": 0.30,
    "shadow": 0.35,
    "kickoff_win": 0.30,
    # Advanced aerials
    "air_dribble": 0.35,
    "flip_reset": 0.40,
    "ceiling_shot": 0.30,
    "musty": 0.25,
    "double_tap": 0.35,
    # Ultra-efficient reads / movement
    "zap_chain": 0.20,
    "stall": 0.15,
    "wavedash_flick": 0.20,
    "deception": 0.20,
    "perfect_touch": 0.40,
    # General play
    "ball_progress": 0.40,
    "touch_quality": 0.30,
    "boost_pos": 0.18,
    "boost_neg": 0.10,
    "overcommit": 0.28,
    "goal": 1.00,
    "concede": 1.00,
    "idle": 0.05,
    "kickoff": 0.22,
}


class SSLReward:
    def __init__(self, w=None):
        self.w = w or DEFAULT_SSL_W

    def __call__(self, info: dict) -> float:
        g = self.w
        r = 0.0
        # Core mastery
        r += g["speedflip"] * info.get("kickoff_first_touch", 0.0)
        r += g["adv_recovery"] * info.get("adv_recovery", 0.0)
        r += g["fast_aerial"] * info.get("fast_aerial_attempt", 0.0)
        r += g["wall_play"] * info.get("wall_play", 0.0)
        r += g["dribble_carry"] * info.get("dribble_carry", 0.0)
        r += g["flick_power"] * info.get("flick_power", 0.0)
        r += g["shadow"] * info.get("shadow_good", 0.0)
        r += g["kickoff_win"] * info.get("kickoff_score", 0.0)

        # Advanced aerials
        r += g["air_dribble"] * info.get("air_dribble_chain", 0.0)
        r += g["flip_reset"] * info.get("flip_reset_attempt", 0.0)
        r += g["ceiling_shot"] * info.get("ceiling_setup", 0.0)
        r += g["musty"] * info.get("musty_attempt", 0.0)
        r += g["double_tap"] * info.get("double_tap_attempt", 0.0)

        # Ultra-efficient reads
        r += g["zap_chain"] * info.get("zap_chain_dash", 0.0)
        r += g["stall"] * info.get("stall_proxy", 0.0)
        r += g["wavedash_flick"] * info.get("wavedash_flick", 0.0)
        r += g["deception"] * info.get("deception_fake", 0.0)
        r += g["perfect_touch"] * info.get("perfect_touch", 0.0)

        # General play
        r += g["ball_progress"] * info.get("ball_to_opp_goal_cos", 0.0)
        r += g["touch_quality"] * info.get("ball_speed_gain_norm", 0.0)
        r += g["boost_pos"] * info.get("boost_use_good", 0.0)
        r -= g["boost_neg"] * info.get("boost_waste", 0.0)
        r -= g["overcommit"] * info.get("last_man_break_flag", 0.0)
        r += g["kickoff"] * info.get("kickoff_score", 0.0)
        r += g["goal"] * info.get("scored", 0.0)
        r -= g["concede"] * info.get("conceded", 0.0)
        r -= g["idle"] * info.get("idle_ticks", 0.0)

        return float(max(-1.0, min(1.0, r)))

