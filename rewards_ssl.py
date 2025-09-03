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
    "air_dribble_ctrl": 0.45,
    "backboard_save": 0.60,
    "corner_clear": 0.35,
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
    "pressure_awareness": 0.25,
    "possession_awareness": 0.20,
    "recovery_ok": 0.20,
    "overcommit_flag_pen": 0.35,
    # Recovery mastery weights
    "recovery_mastery": 0.45,
    "air_roll_landing_w": 0.25,
    "wave_dash_w": 0.25,
    "half_flip_w": 0.25,
    "ceiling_reset_w": 0.15,
    "net_ramp_reset_w": 0.10,
    "wall_nose_down_w": 0.15,
    "small_pads": 0.20,
    "boost_delta": 0.15,
    "possession_time": 0.25,
    "low50": 0.30,
    "back_post_cover": 0.25,
    "demo_util": 0.25,
    "exploit_window": 0.25,
    "conversion_attempt": 0.35,
    "conversion_success": 1.50,
    "finish_variety": 0.10,
}


def _apply_stage_boosts(weights: dict, info: dict):
    # Optional stage-level multipliers injected from bot.py via info["_stage_reward_weights"]
    boosts = info.get("_stage_reward_weights", None)
    if not boosts:
        return weights
    w = dict(weights)
    for k, m in boosts.items():
        if k in w:
            w[k] = w[k] * float(m)
    return w


class SSLReward:
    def __init__(self, w=None):
        self.w = w or DEFAULT_SSL_W

    def __call__(self, info: dict) -> float:
        g = _apply_stage_boosts(self.w, info)
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

        # Aerial play mastery
        r += g["air_dribble_ctrl"] * info.get("air_dribble_ctrl", 0.0)
        r += g["backboard_save"]   * info.get("backboard_save", 0.0)
        r += g["corner_clear"]     * info.get("corner_clear_quality", 0.0)

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

        # Awareness taps
        r += g["pressure_awareness"]   * info.get("pressure_idx", 0.0)
        r += g["possession_awareness"] * info.get("possession_idx", 0.0)
        r += g["recovery_ok"]          * (1.0 if info.get("recovery_ok", False) else 0.0)
        r -= g["overcommit_flag_pen"]  * info.get("overcommit_flag", 0.0)

        # Recovery mastery
        r += g["recovery_mastery"]   * info.get("recovery_mastery", 0.0)
        r += g["air_roll_landing_w"] * info.get("air_roll_landing", 0.0)
        r += g["wave_dash_w"]        * info.get("wave_dash_exec", 0.0)
        r += g["half_flip_w"]        * info.get("half_flip_exec", 0.0)
        r += g["ceiling_reset_w"]    * info.get("ceiling_reset_exec", 0.0)
        r += g["net_ramp_reset_w"]   * info.get("net_ramp_reset_exec", 0.0)
        r += g["wall_nose_down_w"]   * info.get("wall_nose_down", 0.0)

        # 1s-specific
        r += g["small_pads"]       * info.get("small_pad_pickup", 0.0)
        r += g["boost_delta"]      * max(0.0, info.get("boost_delta_norm", 0.0))
        r += g["possession_time"]  * info.get("possession_ticks_norm", 0.0)
        r += g["low50"]            * info.get("low50_success", 0.0)
        r += g["back_post_cover"]  * info.get("back_post_ok", 0.0)
        r += g["demo_util"]        * info.get("demo_benefit", 0.0)

        # Exploit & conversion
        r += g["exploit_window"]      * info.get("exploit_window", 0.0)
        r += g["conversion_attempt"]  * info.get("conversion_attempt", 0.0)
        r += g["conversion_success"]  * info.get("conversion_success", 0.0)
        r += g["finish_variety"]      * info.get("finish_variety", 0.0)

        return float(max(-1.0, min(1.0, r)))

