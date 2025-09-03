# drills_bronze.py — a few simple training spawns (requires enable_state_setting=True)
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator
import numpy as np

def _set(agent, ball_state=None, cars=None):
    if getattr(agent, "_state_setting_ok", False):
        agent.set_game_state(GameState(ball=ball_state, cars=cars or {}))

def inject_kickoff_basic(agent):
    t = agent.team
    ball = BallState(physics=Physics(location=Vector3(0,0,92)))
    car  = CarState(physics=Physics(location=Vector3(0, -2048 if t==0 else 2048, 17),
                                    rotation=Rotator(0,0,0),
                                    velocity=Vector3(0,0,0)), boost_amount=33)
    _set(agent, ball_state=ball, cars={agent.index: car})

def inject_shadow_lane(agent):
    t = agent.team
    yb = -2500 if t==0 else 2500
    yc = yb - (800 if t==0 else -800)
    ball = BallState(physics=Physics(location=Vector3(0, yb, 100)))
    car  = CarState(physics=Physics(location=Vector3(0, yc, 50)))
    _set(agent, ball_state=ball, cars={agent.index: car})

def inject_corner_push(agent):
    t = agent.team
    s  = -1 if np.random.rand()<0.5 else 1
    bx, by = 2500*s, -3500 if t==0 else 3500
    ball = BallState(physics=Physics(location=Vector3(bx, by, 100),
                                     velocity=Vector3(-400*s, 350*(1 if t==0 else -1), 0)))
    car  = CarState(physics=Physics(location=Vector3(bx-700*s, by-600*(1 if t==0 else -1), 50)), boost_amount=30)
    _set(agent, ball_state=ball, cars={agent.index: car})

def inject_box_clear(agent):
    t = agent.team
    own_y = -5120.0 if t==0 else 5120.0
    bx = float(np.random.uniform(-600, 600))
    by = own_y + (1200 if t==0 else -1200)
    bz = float(np.random.uniform(120, 240))
    cy = by + (-900 if t==0 else 900)
    cx = float(np.random.uniform(-900, 900))
    ball = BallState(physics=Physics(location=Vector3(bx, by, bz)))
    car  = CarState(physics=Physics(location=Vector3(cx, cy, 60)), boost_amount=50)
    _set(agent, ball_state=ball, cars={agent.index: car})
