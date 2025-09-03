# rewards_ssl.py — Bronze-only rewards
class SSLReward:
    def __init__(self):
        self.w = {
            # Positive
            "kickoff_first_touch": 0.60,
            "perfect_touch":       0.60,
            "ball_progress":       0.45,
            "small_pads":          0.35,
            "back_post_cover":     0.40,
            "corner_clear_success":0.75,
            # Negative
            "own_slot_time_pen":   0.35,
            "bad_center_touch_pen":0.90,
            "wasted_boost_pen":    0.25,
            "reverse_long_pen":    0.20,
        }

    def __call__(self, info: dict) -> float:
        g = self.w
        r = 0.0
        # Positives
        r += g["kickoff_first_touch"] * info.get("kickoff_first_touch", 0.0)
        r += g["perfect_touch"]       * info.get("perfect_touch", 0.0)
        r += g["ball_progress"]       * info.get("ball_progress", 0.0)
        r += g["small_pads"]          * info.get("small_pad_pickup", 0.0)
        r += g["back_post_cover"]     * info.get("back_post_ok", 0.0)
        r += g["corner_clear_success"]* info.get("corner_clear_success", 0.0)
        # Penalties
        r -= g["own_slot_time_pen"]   * info.get("own_slot_time", 0.0)
        r -= g["bad_center_touch_pen"]* info.get("bad_center_touch", 0.0)
        r -= g["wasted_boost_pen"]    * info.get("wasted_boost", 0.0)
        r -= g["reverse_long_pen"]    * info.get("reverse_ticks", 0.0)
        return float(r)
