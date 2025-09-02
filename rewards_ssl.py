import numpy as np

DEFAULT_SSL_W = {
  "ball_progress": 0.5, "touch_quality": 0.35, "aerial_ctrl": 0.25, "gta_trans": 0.15,
  "boost_pos": 0.15, "boost_neg": 0.10, "shadow": 0.15, "overcommit": 0.25,
  "kickoff": 0.25, "goal": 1.0, "concede": 1.0, "bad_touches": 0.15, "idle": 0.05
}

class SSLReward:
    def __init__(self, w=None):
        self.w = w or DEFAULT_SSL_W

    def __call__(self, info:dict) -> float:
        r  = 0.0
        g  = self.w
        r += g["ball_progress"] * info.get("ball_to_opp_goal_cos", 0.0)
        r += g["touch_quality"] * info.get("ball_speed_gain_norm", 0.0)
        r += g["aerial_ctrl"]  * info.get("aerial_alignment_cos", 0.0)
        r += g["gta_trans"]    * info.get("gta_transition_flag", 0.0)
        r += g["boost_pos"]    * info.get("boost_use_good", 0.0)
        r -= g["boost_neg"]    * info.get("boost_waste", 0.0)
        r += g["shadow"]       * info.get("shadow_angle_cos", 0.0)
        r -= g["overcommit"]   * info.get("last_man_break_flag", 0.0)
        r += g["kickoff"]      * info.get("kickoff_score", 0.0)
        r += g["goal"]         * info.get("scored", 0.0)
        r -= g["concede"]      * info.get("conceded", 0.0)
        r -= g["bad_touches"]  * info.get("own_goal_touch", 0.0)
        r -= g["idle"]         * info.get("idle_ticks", 0.0)
        return float(np.clip(r, -1.0, 1.0))
