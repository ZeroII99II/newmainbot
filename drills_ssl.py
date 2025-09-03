# drills_ssl.py — curriculum state injectors for SSL mechanics
import numpy as np
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator
from mechanics_ssl import _vec2


def _rand(a, b):
    return float(a + np.random.rand() * (b - a))


def inject_fast_aerial(agent):
    # place ball high center, car somewhat back
    team = agent.team
    car_x = _rand(-1500, 1500)
    car_y = _rand(-2500, -1500) if team == 0 else _rand(1500, 2500)
    if team == 1:
        car_y *= -1
    car_z = _rand(50, 120)
    ball_x = _rand(-600, 600)
    ball_y = _rand(-600, 600)
    ball_z = _rand(900, 1400)
    bvx = _rand(-200, 200)
    bvy = _rand(-200, 200)
    bvz = _rand(150, 450)

    car_state = CarState(
        physics=Physics(
            location=Vector3(car_x, car_y, car_z),
            rotation=Rotator(0.0, 0.0, 0.0),
            velocity=Vector3(0, 0, 0),
        ),
        boost_amount=100,
    )
    ball_state = BallState(
        physics=Physics(
            location=Vector3(ball_x, ball_y, ball_z),
            velocity=Vector3(bvx, bvy, bvz),
        )
    )
    agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state}))


def inject_double_tap(agent):
    team = agent.team
    ball_x = _rand(-700, 700)
    ball_y = 5000 * (-1 if team == 0 else 1)
    ball_z = _rand(1200, 1600)
    bvy = -800 * (-1 if team == 0 else 1)  # bounce off backboard
    car_x = ball_x + _rand(-800, 800)
    car_y = ball_y - _rand(1000, 1600)
    car_z = _rand(50, 150)
    car_state = CarState(
        physics=Physics(
            location=Vector3(car_x, car_y, car_z),
            rotation=Rotator(0, 0, 0),
            velocity=Vector3(0, 0, 0),
        ),
        boost_amount=80,
    )
    ball_state = BallState(
        physics=Physics(
            location=Vector3(ball_x, ball_y, ball_z),
            velocity=Vector3(0, bvy, _rand(-150, 50)),
        )
    )
    agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state}))


def inject_flip_reset(agent):
    team = agent.team
    car_x = _rand(-1200, 1200)
    car_y = _rand(-1500, -400) if team == 0 else _rand(400, 1500)
    if team == 1:
        car_y *= -1
    car_z = _rand(400, 700)
    ball_x = car_x + _rand(-150, 150)
    ball_y = car_y + _rand(500, 900) * (-1 if team == 0 else 1)
    ball_z = _rand(900, 1200)
    car_state = CarState(
        physics=Physics(
            location=Vector3(car_x, car_y, car_z),
            rotation=Rotator(0, 0, 0),
            velocity=Vector3(0, 0, 0),
        ),
        boost_amount=100,
    )
    ball_state = BallState(
        physics=Physics(
            location=Vector3(ball_x, ball_y, ball_z),
            velocity=Vector3(_rand(-150, 150), _rand(-150, 150), _rand(-150, 50)),
        )
    )
    agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state}))


def inject_ceiling_shot(agent):
    team = agent.team
    car_x = _rand(-1200, 1200)
    car_y = _rand(-400, 400) * (-1 if team == 0 else 1)
    car_z = 1950  # near ceiling with contact
    ball_x = car_x + _rand(-500, 500)
    ball_y = car_y + _rand(800, 1400) * (-1 if team == 0 else 1)
    ball_z = _rand(800, 1200)
    car_state = CarState(
        physics=Physics(
            location=Vector3(car_x, car_y, car_z),
            rotation=Rotator(0.0, 0.0, 0.0),
            velocity=Vector3(0, 0, 0),
        ),
        boost_amount=60,
    )
    ball_state = BallState(
        physics=Physics(
            location=Vector3(ball_x, ball_y, ball_z),
            velocity=Vector3(_rand(-100, 100), _rand(-100, 100), _rand(-50, 150)),
        )
    )
    agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state}))


def inject_dribble_flick(agent):
    team = agent.team
    car_x = _rand(-800, 800)
    car_y = _rand(-1400, -800) if team == 0 else _rand(800, 1400)
    if team == 1:
        car_y *= -1
    car_z = _rand(40, 60)
    ball_x = car_x + _rand(-50, 50)
    ball_y = car_y + _rand(-50, 50)
    ball_z = _rand(70, 120)
    car_state = CarState(
        physics=Physics(
            location=Vector3(car_x, car_y, car_z),
            rotation=Rotator(0, 0, 0),
            velocity=Vector3(0, 0, 0),
        ),
        boost_amount=30,
    )
    ball_state = BallState(
        physics=Physics(
            location=Vector3(ball_x, ball_y, ball_z),
            velocity=Vector3(0, 0, 0),
        )
    )
    agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state}))


def inject_shadow_defense(agent):
    # place us between our goal and ball, at shadow distance
    team = agent.team
    goal_y = -5120 if team == 0 else 5120
    ball_x = _rand(-1500, 1500)
    ball_y = _rand(-1600, -800) if team == 0 else _rand(800, 1600)
    if team == 1:
        ball_y *= -1
    ball_z = _rand(80, 200)
    # our position ~1200 behind the ball toward our goal
    shadow_y = ball_y + (goal_y - ball_y) * 0.3
    car_x = ball_x + _rand(-200, 200)
    car_y = shadow_y
    car_z = _rand(40, 60)
    car_state = CarState(
        physics=Physics(
            location=Vector3(car_x, car_y, car_z),
            rotation=Rotator(0, 0, 0),
            velocity=Vector3(0, 0, 0),
        ),
        boost_amount=40,
    )
    ball_state = BallState(
        physics=Physics(
            location=Vector3(ball_x, ball_y, ball_z),
            velocity=Vector3(_rand(-200, 200), _rand(-100, 100), 0),
        )
    )
    agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state}))


