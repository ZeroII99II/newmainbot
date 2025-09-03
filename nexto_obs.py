import numpy as np
try:
    from rlgym_compat.common_values import BOOST_LOCATIONS
except Exception:
    BOOST_LOCATIONS = np.zeros((34, 3), dtype=np.float32)

class NextoObsBuilder:
    def reset(self, initial_state):
        pass

    def build_obs(self, player, state, prev_action):
        return np.zeros(10, dtype=np.float32)
