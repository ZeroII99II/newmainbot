import numpy as np
import torch
import time
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlgym_compat import GameState

from live_policy import LivePolicy
from ring_buffer import RingBuffer
from rewards_ssl import SSLReward, DEFAULT_SSL_W
from kickoff_detector import is_kickoff_pause, detect_spawn_side
from speedflip import run_speedflip, SpeedFlipParams
from necto_obs import NectoObsBuilder

ALLOWED_INDICES = {0, 1}

KICKOFF_CONTROLS = (
        11 * 4 * [SimpleControllerState(throttle=1, boost=True)]
        + 4 * 4 * [SimpleControllerState(throttle=1, boost=True, steer=-1)]
        + 2 * 4 * [SimpleControllerState(throttle=1, jump=True, boost=True)]
        + 1 * 4 * [SimpleControllerState(throttle=1, boost=True)]
        + 1 * 4 * [SimpleControllerState(throttle=1, yaw=0.8, pitch=-0.7, jump=True, boost=True)]
        + 13 * 4 * [SimpleControllerState(throttle=1, pitch=1, boost=True)]
        + 10 * 4 * [SimpleControllerState(throttle=1, roll=1, pitch=0.5)]
)

KICKOFF_NUMPY = np.array([
    [scs.throttle, scs.steer, scs.pitch, scs.yaw, scs.roll, scs.jump, scs.boost, scs.handbrake]
    for scs in KICKOFF_CONTROLS
])


class Necto(BaseAgent):
    def __init__(self, name, team, index, beta=1, render=False, hardcoded_kickoffs=True):
        if index not in ALLOWED_INDICES:
            print(f"[guard] refusing unexpected bot index={index}")
            raise SystemExit(0)
        super().__init__(name, team, index)

        self.obs_builder = None
        self.tick_skip = 8

        # Beta controls randomness:
        # 1=best action, 0.5=sampling from probability, 0=random, -1=worst action, or anywhere inbetween
        self.beta = beta
        self.render = render
        self.hardcoded_kickoffs = hardcoded_kickoffs

        self.policy = LivePolicy(path="necto-model.pt")
        self.buffer = RingBuffer(capacity=200_000, obs_dim=107, act_dim=8)
        self.ssl_reward = SSLReward(DEFAULT_SSL_W)
        self.kick_params = SpeedFlipParams()
        self.kick_start_time = None
        self.spawn_side = None

        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0
        self.kickoff_index = -1
        print('Necto Ready - Index:', index)
        print("Remember to run Necto at 120fps with vsync off! "
              "Stable 240/360 is second best if that's better for your eyes")
        print("Also check out the RLGym Twitch stream to watch live bot training and occasional showmatches!")

    def initialize_agent(self):
        # Initialize the rlgym GameState object now that the game is active and the info is available
        field_info = self.get_field_info()
        self.field_info = field_info
        self.obs_builder = NectoObsBuilder(field_info=field_info)
        self.game_state = GameState(field_info)
        self.ticks = self.tick_skip  # So we take an action the first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.update_action = True
        self.kickoff_index = -1

    def render_attention_weights(self, weights, obs):
        mean_weights = torch.mean(torch.stack(weights), dim=0).numpy()[0][0]

        top = sorted(range(len(mean_weights)), key=lambda i: mean_weights[i], reverse=True)
        top.remove(1)  # Self

        self.renderer.begin_rendering('attention_weights')

        invert = np.array([-1, -1, 1]) if self.team == 1 else np.ones(3)
        loc = obs[0][0, 0, 5:8] * 2300 * invert
        mx = mean_weights[~(np.arange(len(mean_weights)) == 1)].max()
        c = 1
        for i in top[:3]:
            weight = mean_weights[i] / mx
            # print(i, weight)
            dest = loc + obs[1][0, i, 5:8] * 2300 * invert
            color = self.renderer.create_color(255, round(255 * (1 - weight)), round(255),
                                               round(255 * (1 - weight)))
            self.renderer.draw_string_3d(dest, 2, 2, str(c), color)
            c += 1
            self.renderer.draw_line_3d(loc, dest, color)
        self.renderer.end_rendering()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        ticks_elapsed = round(delta * 120)
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

        if self.update_action == 1 and len(self.game_state.players) > self.index:
            self.update_action = 0

            player = self.game_state.players[self.index]
            teammates = [p for p in self.game_state.players if p.team_num == self.team and p != player]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]

            self.game_state.players = [player] + teammates + opponents

            obs_tuple = self.obs_builder.build_obs(player, self.game_state, self.action)
            obs_flat = self._flatten_obs(obs_tuple)

            self.action = self.policy.act(obs_flat)

            info = self._compute_info(packet)
            reward = self.ssl_reward(info)
            done = packet.game_info.is_match_ended
            self.buffer.push(obs_flat, self.action, reward, done)

        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_controls(self.action)
            self.update_action = 1

        ctl = self.controls
        if is_kickoff_pause(packet):
            if self.kick_start_time is None:
                self.kick_start_time = time.time()
                self.spawn_side = detect_spawn_side(packet.game_cars[self.index], self.field_info)
            tsk = time.time() - self.kick_start_time
            ctl = run_speedflip(packet, self.spawn_side, ctl, self.kick_params, tsk)
        else:
            self.kick_start_time = None

        return ctl

    def maybe_do_kickoff(self, packet, ticks_elapsed):
        if packet.game_info.is_kickoff_pause:
            if self.kickoff_index >= 0:
                self.kickoff_index += round(ticks_elapsed)
            elif self.kickoff_index == -1:
                is_kickoff_taker = False
                ball_pos = np.array([packet.game_ball.physics.location.x, packet.game_ball.physics.location.y])
                positions = np.array([[car.physics.location.x, car.physics.location.y]
                                      for car in packet.game_cars[:packet.num_cars]])
                distances = np.linalg.norm(positions - ball_pos, axis=1)
                if abs(distances.min() - distances[self.index]) <= 10:
                    is_kickoff_taker = True
                    indices = np.argsort(distances)
                    for index in indices:
                        if abs(distances[index] - distances[self.index]) <= 10 \
                                and packet.game_cars[index].team == self.team \
                                and index != self.index:
                            if self.team == 0:
                                is_left = positions[index, 0] < positions[self.index, 0]
                            else:
                                is_left = positions[index, 0] > positions[self.index, 0]
                            if not is_left:
                                is_kickoff_taker = False  # Left goes

                self.kickoff_index = 0 if is_kickoff_taker else -2

            if 0 <= self.kickoff_index < len(KICKOFF_NUMPY) \
                    and packet.game_ball.physics.location.y == 0:
                action = KICKOFF_NUMPY[self.kickoff_index]
                self.action = action
                self.update_controls(self.action)
        else:
            self.kickoff_index = -1

    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0

    def _flatten_obs(self, obs):
        if isinstance(obs, (list, tuple)):
            arr = np.concatenate([np.asarray(o).flatten() for o in obs])
        else:
            arr = np.asarray(obs).flatten()
        if arr.size < 107:
            arr = np.pad(arr, (0, 107 - arr.size))
        return arr[:107].astype(np.float32)

    def _compute_info(self, packet):
        ball = packet.game_ball
        vel = np.array([
            ball.physics.velocity.x,
            ball.physics.velocity.y,
            ball.physics.velocity.z,
        ])
        goal_dir = np.array([0, 1 if self.team == 0 else -1, 0], dtype=np.float32)
        speed = np.linalg.norm(vel) + 1e-6
        info = {
            "ball_to_opp_goal_cos": float(np.dot(vel, goal_dir) / speed),
            "ball_speed_gain_norm": float(speed / 6000.0),
            "aerial_alignment_cos": 0.0,
            "gta_transition_flag": 0.0,
            "boost_use_good": 0.0,
            "boost_waste": 0.0,
            "shadow_angle_cos": 0.0,
            "last_man_break_flag": 0.0,
            "kickoff_score": 0.0,
            "scored": 0.0,
            "conceded": 0.0,
            "own_goal_touch": 0.0,
            "idle_ticks": 0.0,
        }
        return info


def create_agent(name, team, index):
    return Necto(name, team, index)
