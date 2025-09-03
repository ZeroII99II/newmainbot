# curriculum.py — Destroyer Academy staged training (Bronze→SSL)
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class StageSpec:
    name: str
    # (metric_key, target_value, progress_weight)
    goals: List[Tuple[str, float, float]]
    reward_boosts: Dict[str, float] = field(default_factory=dict)  # multiply reward weights for these keys
    drills: List[Tuple[str, float]] = field(default_factory=list)  # (drill_name, probability)
    intent_bias: Dict[str, float] = field(default_factory=dict)    # (intent -> bias), currently informational
    graduate_progress_threshold: float = 0.75
    sustain_secs: int = 90
    min_play_secs: int = 240

# Which metrics are "lower is better" (we invert for progress)
LOWER_IS_BETTER = {"idle_ticks", "overcommit_flag"}

STAGES: List[StageSpec] = [
    StageSpec(
        name="Bronze",
        goals=[
            ("small_pad_pickup", 6.0, 0.20),
            ("perfect_touch", 0.50, 0.20),
            ("shadow_good", 8.0, 0.25),
            ("idle_ticks", 0.00, 0.10),      # lower is better
            ("recovery_mastery", 0.20, 0.25),
        ],
        reward_boosts={"touch_quality":1.2,"ball_progress":1.1,"boost_pos":1.2,"idle":1.2},
        drills=[("dribble",0.3),("shadow",0.3),("wave_dash",0.2),("wall_land",0.2)],
        intent_bias={"CONTROL":+0.2,"SHADOW":+0.2,"BOOST":+0.1},
        graduate_progress_threshold=0.80, sustain_secs=60, min_play_secs=180
    ),
    StageSpec(
        name="Silver",
        goals=[
            ("kickoff_first_touch", 0.60, 0.20),
            ("adv_recovery", 4.0, 0.20),
            ("dribble_carry", 4.0, 0.20),
            ("back_post_ok", 6.0, 0.20),
            ("small_pad_pickup", 7.0, 0.20),
        ],
        reward_boosts={"kickoff":1.2,"recovery_mastery":1.2,"shadow":1.15},
        drills=[("shadow",0.25),("dribble",0.25),("wave_dash",0.25),("half_flip",0.25)],
        intent_bias={"SHADOW":+0.15,"DRIBBLE":+0.15,"BOOST":+0.1}
    ),
    StageSpec(
        name="Gold",
        goals=[
            ("fast_aerial_attempt", 2.0, 0.25),
            ("wall_play", 2.0, 0.20),
            ("backboard_save", 0.30, 0.20),
            ("perfect_touch", 0.80, 0.20),
            ("boost_delta_norm", 0.10, 0.15),
        ],
        reward_boosts={"fast_aerial":1.25,"wall_play":1.2,"backboard_save":1.25,"perfect_touch":1.2},
        drills=[("fast_aerial",0.3),("backboard_defense",0.3),("double_tap",0.2),("shadow",0.2)],
        intent_bias={"BACKBOARD_DEFEND":+0.2,"PRESS":+0.1}
    ),
    StageSpec(
        name="Platinum",
        goals=[
            ("speedflip", 0.90, 0.25),
            ("dribble_carry", 6.0, 0.20),
            ("flick_power", 0.40, 0.20),
            ("low50_success", 0.80, 0.20),
            ("recovery_mastery", 0.35, 0.15),
        ],
        reward_boosts={"kickoff_win":1.2,"flick_power":1.25,"low50":1.25},
        drills=[("dribble",0.25),("fast_aerial",0.2),("double_tap",0.2),("wave_dash",0.2),("half_flip",0.15)],
        intent_bias={"DRIBBLE":+0.15,"CONTROL":+0.15}
    ),
    StageSpec(
        name="Diamond",
        goals=[
            ("air_dribble_chain", 0.60, 0.25),
            ("double_tap_attempt", 0.60, 0.20),
            ("backboard_save", 0.60, 0.20),
            ("small_pad_pickup", 8.0, 0.15),
            ("deception_fake", 0.40, 0.20),
        ],
        reward_boosts={"air_dribble":1.25,"double_tap":1.25,"backboard_save":1.2,"deception":1.2},
        drills=[("wall_airdribble",0.35),("backboard_defense",0.3),("double_tap",0.2),("flip_reset",0.15)],
        intent_bias={"AIR_DRIBBLE":+0.2,"BACKBOARD_DEFEND":+0.2}
    ),
    StageSpec(
        name="Champion",
        goals=[
            ("air_dribble_ctrl", 0.70, 0.25),
            ("perfect_touch", 0.95, 0.20),
            ("zap_chain_dash", 0.80, 0.15),
            ("possession_ticks_norm", 0.70, 0.20),
            ("overcommit_flag", 0.10, 0.20),   # lower is better
        ],
        reward_boosts={"air_dribble_ctrl":1.3,"perfect_touch":1.2,"zap_chain":1.2,"possession_awareness":1.2,"overcommit":1.2},
        drills=[("wall_airdribble",0.3),("ceiling_reset",0.25),("backboard_defense",0.25),("net_ramp",0.2)],
        intent_bias={"CONTROL":+0.2,"PRESS":+0.1}
    ),
    StageSpec(
        name="Grand Champion",
        goals=[
            ("flip_reset_attempt", 0.60, 0.20),
            ("ceiling_setup", 0.60, 0.20),
            ("corner_clear_quality", 0.70, 0.20),
            ("boost_delta_norm", 0.25, 0.20),
            ("demo_benefit", 0.50, 0.20),
        ],
        reward_boosts={"flip_reset":1.25,"ceiling_shot":1.25,"corner_clear":1.25,"demo_util":1.2},
        drills=[("flip_reset",0.3),("ceiling_reset",0.3),("backboard_defense",0.2),("wall_airdribble",0.2)],
        intent_bias={"FAKE":+0.1,"STARVE":+0.1,"BUMP":+0.1}
    ),
    StageSpec(
        name="Supersonic Legend",
        goals=[
            ("air_dribble_ctrl", 0.90, 0.20),
            ("perfect_touch", 0.99, 0.20),
            ("recovery_mastery", 0.70, 0.20),
            ("pressure_idx", 0.85, 0.20),
            ("deception_fake", 0.80, 0.20),
        ],
        reward_boosts={"perfect_touch":1.3,"recovery_mastery":1.3,"pressure_awareness":1.2,"deception":1.2},
        drills=[("wall_airdribble",0.25),("double_tap",0.2),("flip_reset",0.2),("backboard_defense",0.2),("ceiling_reset",0.15)],
        intent_bias={"CONTROL":+0.15,"PRESS":+0.15,"CHALLENGE":+0.1},
        graduate_progress_threshold=1.00, sustain_secs=120, min_play_secs=600
    ),
]

def combine_reward_boosts(base: Dict[str, float], stage: StageSpec) -> Dict[str, float]:
    merged = dict(base)
    for k, m in stage.reward_boosts.items():
        merged[k] = merged.get(k, 1.0) * m
    return merged
