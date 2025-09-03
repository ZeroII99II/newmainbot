# ones_profile.py — knobs for 1v1 profile
ONES = {
    "target_min_boost": 40,        # prefer to play on >= 40 boost (Scrub Killa advice)
    "pad_radius": 220.0,           # proximity to count a small-pad pickup
    "low50_distance": 260.0,       # when under-ball & close, bias to low-50
    "back_post_buffer": 900.0,     # how far to favor back-post when defending
    "demo_allow": True,
    "demo_min_speed": 1800.0,      # need speed to threaten demo
    "tempo_slow_lead_secs": 60.0,  # last minute: prefer control when leading
    "tempo_slow_boost_edge": 20.0, # slow if we have +20 boost over opp
    "starve_when_edge": 1,         # enable boost-starve routine when we have pressure/edge
}
