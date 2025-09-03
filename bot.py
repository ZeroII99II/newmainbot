from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
import numpy as np

try:
    from rlgym_compat.game_state import GameState
except ImportError:
    from rlgym_compat import GameState

from agent import Agent
from nexto_obs import NextoObsBuilder, BOOST_LOCATIONS

def _clip(x, lo=-1, hi=1): 
    return float(max(lo, min(hi, x)))

def _steer_to(me, tx, ty, yaw_gain=2.2, d_gain=0.7):
    import math
    yaw = float(me.physics.rotation.yaw)
    yaw_rate = float(getattr(me.physics.angular_velocity, "z", 0.0))
    ang = math.atan2(ty - me.physics.location.y, tx - me.physics.location.x) - yaw
    while ang > math.pi: ang -= 2*math.pi
    while ang < -math.pi: ang += 2*math.pi
    steer = _clip(yaw_gain * ang - d_gain * yaw_rate, -1.0, 1.0)
    return steer, abs(ang)

def bronze_fallback(packet, index):
    me = packet.game_cars[index]
    ball = packet.game_ball
    a = np.zeros(8, dtype=np.float32)

    # Danger slot clear-to-corner
    own_y = -5120.0 if me.team == 0 else 5120.0
    if me.team == 0:
        y_min, y_max = own_y + 300.0, own_y + 2000.0
    else:
        y_min, y_max = own_y - 2000.0, own_y - 300.0
    dz = (abs(ball.physics.location.x) <= 1100.0 and y_min <= ball.physics.location.y <= y_max and ball.physics.location.z < 1100.0)
    if dz:
        cx = 3072.0 if ball.physics.location.x >= 0 else -3072.0
        cy = own_y + (500.0 if me.team == 0 else -500.0)
        steer, ang = _steer_to(me, cx, cy)
        a[0] = steer; a[1] = 1.0; a[6] = 1.0 if ang < 0.35 else 0.0
        return a

    # Kickoff: straight to center, front-flip when close
    if packet.game_info.is_kickoff_pause:
        steer, ang = _steer_to(me, 0.0, 0.0)
        a[0] = steer; a[1] = 1.0; a[6] = 1.0 if ang < 0.2 else 0.0
        db = ((me.physics.location.x - ball.physics.location.x)**2 + (me.physics.location.y - ball.physics.location.y)**2) ** 0.5
        if db < 450 and ang < 0.2:
            a[5] = 1.0; a[2] = 1.0
        return a

    # Otherwise: simple approach to ball with slight far-post bias
    gy = (5120.0 if me.team == 0 else -5120.0) * 0.92
    aim_x = float(np.clip(ball.physics.location.x * 0.7, -900, 900))
    steer, ang = _steer_to(me, aim_x, gy)
    a[0] = steer; a[1] = 0.9; a[6] = 0.3 if ang < 0.25 else 0.0
    return a


class Nexto(BaseAgent):
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

        act = None; weights = None
        try:
            act, weights = self.agent.act(obs, beta)
        except Exception:
            act = None

        if act is None:
            # No model / failed → Bronze fallback
            self.action = bronze_fallback(packet, self.index)
        else:
            self.action = act

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

# RLBot entry point
def create_agent(agent_name, team, index):
    return Nexto(agent_name, team, index)
