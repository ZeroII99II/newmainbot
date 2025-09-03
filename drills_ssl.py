# drills_ssl.py — Bronze-only drills
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator
import numpy as np

def inject_kickoff_basic(agent):
    t = agent.team
    car_x, car_y = 0, -2048 if t==0 else 2048
    ball_state = BallState(physics=Physics(location=Vector3(0,0,92)))
    car_state = CarState(physics=Physics(location=Vector3(car_x, car_y, 17),
                                         rotation=Rotator(0, 0, 0),
                                         velocity=Vector3(0,0,0)),
                         boost_amount=33)
    if getattr(agent, "_state_setting_ok", False):
        agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state}))

def inject_shadow_lane(agent):
    t = agent.team
    yb = -2500 if t==0 else 2500
    yc = yb - (800 if t==0 else -800)
    ball_state = BallState(physics=Physics(location=Vector3(0, yb, 100), velocity=Vector3(0,0,0)))
    car_state  = CarState(physics=Physics(location=Vector3(0, yc, 50), rotation=Rotator(0,0,0)))
    if getattr(agent,"_state_setting_ok",False):
        agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state}))

def inject_corner_push(agent):
    t = agent.team
    s = -1 if np.random.rand()<0.5 else 1
    bx, by = 2500*s, -3500 if t==0 else 3500
    ball_state = BallState(physics=Physics(location=Vector3(bx, by, 100),
                                           velocity=Vector3(-400*s, 350*(1 if t==0 else -1), 0)))
    car_state  = CarState(physics=Physics(location=Vector3(bx-700*s, by-600*(1 if t==0 else -1), 50),
                                         rotation=Rotator(0,0,0), velocity=Vector3(0,0,0)), boost_amount=30)
    if getattr(agent,"_state_setting_ok",False):
        agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state}))

def inject_box_clear(agent):
    t = agent.team
    own_y = -5120.0 if t==0 else 5120.0
    bx = np.random.uniform(-600, 600)
    by = own_y + (1200 if t==0 else -1200)
    bz = np.random.uniform(120, 240)
    cy = by + (-900 if t==0 else 900)
    cx = np.random.uniform(-900, 900)
    car_state = CarState(physics=Physics(location=Vector3(cx, cy, 60),
                                         rotation=Rotator(0, 0, 0),
                                         velocity=Vector3(0,0,0)),
                         boost_amount=50)
    ball_state = BallState(physics=Physics(location=Vector3(bx, by, bz), velocity=Vector3(0,0,0)))
    if getattr(agent,"_state_setting_ok",False):
        agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state}))
