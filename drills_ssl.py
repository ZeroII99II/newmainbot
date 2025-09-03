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
    if getattr(agent, "_state_setting_ok", False):
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
    if getattr(agent, "_state_setting_ok", False):
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
    if getattr(agent, "_state_setting_ok", False):
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
    if getattr(agent, "_state_setting_ok", False):
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
    if getattr(agent, "_state_setting_ok", False):
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
    if getattr(agent, "_state_setting_ok", False):
        agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state}))


def inject_half_flip(agent):
    # Car facing away from ball, low speed
    team = agent.team
    car_x = _rand(-800, 800)
    car_y = _rand(-1200, -600) if team==0 else _rand(600, 1200)
    if team==1: car_y *= -1
    ball_x = car_x + _rand(-200, 200)
    ball_y = car_y + (800 * (-1 if team==0 else 1))
    car_state = CarState(physics=Physics(location=Vector3(car_x, car_y, 50),
                                         rotation=Rotator(0, _rand(-3.0, 3.0), 0),
                                         velocity=Vector3(0, 0, 0)), boost_amount=20)
    ball_state = BallState(physics=Physics(location=Vector3(ball_x, ball_y, 120)))
    if getattr(agent, "_state_setting_ok", False):
        agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state}))


def inject_wall_airdribble(agent):
    team = agent.team
    side = -1 if np.random.rand() < 0.5 else 1
    ball_x = 2200 * side
    ball_y = -1500 if team==0 else 1500
    car_x  = ball_x - 500 * side
    car_y  = ball_y - (450 * (-1 if team==0 else 1))
    car_z  = 50
    ball_z = 200
    car_state = CarState(physics=Physics(location=Vector3(car_x, car_y, car_z),
                                         rotation=Rotator(0, 0, 0),
                                         velocity=Vector3(0,0,0)),
                         boost_amount=100)
    ball_state = BallState(physics=Physics(location=Vector3(ball_x, ball_y, ball_z),
                                           velocity=Vector3(0, 0, 0)))
    if getattr(agent, "_state_setting_ok", False):
        agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state}))


def inject_backboard_defense(agent):
    team = agent.team
    # ball toward our backboard
    ball_x = np.random.uniform(-1200, 1200)
    ball_y = -4200 if team==0 else 4200
    ball_z = np.random.uniform(900, 1400)
    bvy = -700 if team==1 else 700  # moving toward our board
    car_x  = np.random.uniform(-1200, 1200)
    car_y  = -2400 if team==0 else 2400
    car_state = CarState(physics=Physics(location=Vector3(car_x, car_y, 50),
                                         rotation=Rotator(0, 0, 0),
                                         velocity=Vector3(0,0,0)),
                         boost_amount=60)
    ball_state = BallState(physics=Physics(location=Vector3(ball_x, ball_y, ball_z),
                                           velocity=Vector3(0, bvy, np.random.uniform(-50, 150))))
    if getattr(agent, "_state_setting_ok", False):
        agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state}))


def inject_wave_dash(agent):
    # Small hop then land — encourage wave dash
    team = agent.team
    car_x = _rand(-1000, 1000)
    car_y = _rand(-800, -200) if team==0 else _rand(200, 800)
    if team==1: car_y *= -1
    car_state = CarState(physics=Physics(location=Vector3(car_x, car_y, 70),
                                         velocity=Vector3(_rand(-200,200), _rand(-200,200), -200)))
    if getattr(agent, "_state_setting_ok", False):
        agent.set_game_state(GameState(cars={agent.index: car_state}))


def inject_wall_nose_down(agent):
    # Place on side wall and ask for nose-down landing
    team = agent.team
    x = 3000 if np.random.rand() < 0.5 else -3000
    y = _rand(-2500, 2500)
    car_state = CarState(physics=Physics(location=Vector3(x, y, 500),
                                         rotation=Rotator(_rand(0.3,0.6), 0.0, _rand(-1.2,1.2)),
                                         velocity=Vector3(0,0,-300)),
                         boost_amount=0.0)
    if getattr(agent, "_state_setting_ok", False):
        agent.set_game_state(GameState(cars={agent.index: car_state}))


def inject_ceiling_reset(agent):
    # Ceiling contact then drop
    team = agent.team
    x = _rand(-800, 800)
    y = _rand(-500, 500) * (-1 if team==0 else 1)
    car_state = CarState(physics=Physics(location=Vector3(x, y, 1980),
                                         rotation=Rotator(0.0, 0.0, 0.0),
                                         velocity=Vector3(0,0,0)),
                         boost_amount=20)
    if getattr(agent, "_state_setting_ok", False):
        agent.set_game_state(GameState(cars={agent.index: car_state}))


def inject_net_ramp_pop(agent):
    # Inside goal ramp, encourage ramp pop/reset
    team = agent.team
    y = -5050 if team==0 else 5050
    x = _rand(-700, 700)
    car_state = CarState(physics=Physics(location=Vector3(x, y, 120),
                                         rotation=Rotator(0.0, 0.0, 0.0),
                                         velocity=Vector3(0,0,50)),
                         boost_amount=0.0)
    if getattr(agent, "_state_setting_ok", False):
        agent.set_game_state(GameState(cars={agent.index: car_state}))


def inject_open_net(agent):
    team = agent.team
    # Put ball center-top of box; opponent out of lane
    ball_x = np.random.uniform(-600, 600)
    ball_y = 3600 if team==0 else -3600
    car_y  = ball_y - (900 if team==0 else -900)
    opp_y  = ball_y + (1600 if team==0 else -1600)
    opp_x  = 2200 if np.random.rand()<0.5 else -2200
    car_state_me = CarState(physics=Physics(location=Vector3(0, car_y, 50), rotation=Rotator(0, 0, 0), velocity=Vector3(0,0,0)), boost_amount=60)
    car_state_opp = CarState(physics=Physics(location=Vector3(opp_x, opp_y, 50), rotation=Rotator(0, 0, 0), velocity=Vector3(0,0,0)), boost_amount=10)
    ball_state = BallState(physics=Physics(location=Vector3(ball_x, ball_y, 160), velocity=Vector3(0,0,0)))
    if getattr(agent, "_state_setting_ok", False):
        agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state_me, 1-agent.index: car_state_opp}))


def inject_bad_recovery(agent):
    team = agent.team
    # Opponent airborne/low boost; ball bouncing in their half
    ball_x = np.random.uniform(-800, 800)
    ball_y = 2800 if team==0 else -2800
    ball_state = BallState(physics=Physics(location=Vector3(ball_x, ball_y, 300), velocity=Vector3(0,0,300)))
    opp_y  = ball_y + (900 if team==0 else -900)
    car_state_opp = CarState(physics=Physics(location=Vector3(ball_x+600, opp_y, 500), rotation=Rotator(0,0,0), velocity=Vector3(0,0,0)), boost_amount=0.0)
    car_state_me = CarState(physics=Physics(location=Vector3(ball_x-800, ball_y-800*(1 if team==0 else -1), 60), rotation=Rotator(0,0,0), velocity=Vector3(0,0,0)), boost_amount=50)
    if getattr(agent, "_state_setting_ok", False):
        agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state_me, 1-agent.index: car_state_opp}))


def inject_box_clear(agent):
    team = agent.team
    own_goal_y = -5120.0 if team == 0 else 5120.0
    # Ball centered in slot
    ball_x = np.random.uniform(-600, 600)
    ball_y = own_goal_y + (1200 if team == 0 else -1200)
    ball_z = np.random.uniform(120, 240)
    car_y  = ball_y + ( -900 if team == 0 else 900 )
    car_x  = np.random.uniform(-900, 900)
    car_state = CarState(physics=Physics(location=Vector3(car_x, car_y, 60),
                                         rotation=Rotator(0, 0, 0),
                                         velocity=Vector3(0,0,0)),
                         boost_amount=50)
    ball_state = BallState(physics=Physics(location=Vector3(ball_x, ball_y, ball_z),
                                           velocity=Vector3(0,0,0)))
    if getattr(agent, "_state_setting_ok", False):
        agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state}))


def inject_box_panic(agent):
    team = agent.team
    own_goal_y = -5120.0 if team == 0 else 5120.0
    ball_x = np.random.uniform(-400, 400)
    ball_y = own_goal_y + (900 if team == 0 else -900)
    ball_z = np.random.uniform(100, 300)
    bvy    = -800 if team == 1 else 800  # moving toward net
    car_y  = ball_y + ( -600 if team == 0 else 600 )
    car_x  = np.random.uniform(-600, 600)
    car_state = CarState(physics=Physics(location=Vector3(car_x, car_y, 60),
                                         rotation=Rotator(0, 0, 0),
                                         velocity=Vector3(0,0,0)),
                         boost_amount=30)
    ball_state = BallState(physics=Physics(location=Vector3(ball_x, ball_y, ball_z),
                                           velocity=Vector3(0, bvy, 0)))
    if getattr(agent, "_state_setting_ok", False):
        agent.set_game_state(GameState(ball=ball_state, cars={agent.index: car_state}))


