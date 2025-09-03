from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
import numpy as np

try:
    from rlgym_compat.game_state import GameState
except ImportError:
    from rlgym_compat import GameState

from agent import Agent
from nexto_obs import NextoObsBuilder, BOOST_LOCATIONS
from bronze_heuristic import decide as bronze_decide


class Destroyer(BaseAgent):
    def initialize_agent(self):
        self.agent = Agent()
        self.obs_builder = NextoObsBuilder()
        self.game_state = None
        self.action = np.zeros(8, dtype=np.float32)
        self.beta = 0
        self.stochastic_kickoffs = False

    def get_output(self, packet):
        player = packet.game_cars[self.index]
        self.game_state = GameState(packet, boost_locs=BOOST_LOCATIONS)

        obs = self.obs_builder.build_obs(player, self.game_state, self.action)

        beta = self.beta
        if packet.game_info.is_match_ended:
            beta = 0
        if self.stochastic_kickoffs and packet.game_info.is_kickoff_pause:
            beta = 0.5

        model_out = None
        weights = None
        if self.agent is not None:
            act_out, weights = self.agent.act(obs, beta)
            if act_out is not None:
                model_out = act_out

        if model_out is None:
            self.action, intent = bronze_decide(packet, self.index)
            print(f"[Bronze] intent={intent}")
        else:
            self.action = model_out

        ctl = SimpleControllerState()
        ctl.steer = float(self.action[0])
        ctl.throttle = float(self.action[1])
        ctl.pitch = float(self.action[2])
        ctl.yaw = float(self.action[3])
        ctl.roll = float(self.action[4])
        ctl.jump = bool(self.action[5] > 0.5)
        ctl.boost = bool(self.action[6] > 0.5)
        ctl.handbrake = bool(self.action[7] > 0.5)
        return ctl


def create_agent(agent_name, team, index):
    return Destroyer(agent_name, team, index)
