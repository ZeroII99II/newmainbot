# bronze_heuristic.py — minimal 1v1 Bronze: kickoff front-flip, shadow, control, clear-to-corner.
import math, numpy as np

def _clip(x, lo=-1.0, hi=1.0): return float(max(lo, min(hi, x)))

def _steer_to(me, tx, ty, yaw_gain=2.2, d_gain=0.7):
    yaw = float(me.physics.rotation.yaw)
    yaw_rate = float(getattr(me.physics.angular_velocity, "z", 0.0))
    ang = math.atan2(ty - me.physics.location.y, tx - me.physics.location.x) - yaw
    while ang > math.pi: ang -= 2*math.pi
    while ang < -math.pi: ang += 2*math.pi
    steer = _clip(yaw_gain * ang - d_gain * yaw_rate)
    return steer, abs(ang)

def _goal_y(team): return 5120.0 if team==0 else -5120.0

def decide(packet, index, intent_hint=None):
    me   = packet.game_cars[index]
    ball = packet.game_ball
    team = me.team

    action = np.zeros(8, dtype=np.float32)
    is_kickoff = bool(packet.game_info.is_kickoff_pause)

    # Danger slot in front of our net
    own_goal_y = -5120.0 if team==0 else 5120.0
    if team==0: y_min, y_max = own_goal_y+300, own_goal_y+2000
    else:       y_min, y_max = own_goal_y-2000, own_goal_y-300
    danger_zone = (abs(ball.physics.location.x) <= 1100 and y_min <= ball.physics.location.y <= y_max and ball.physics.location.z < 1100)

    if is_kickoff:
        steer, ang = _steer_to(me, 0.0, 0.0)
        action[0]=steer; action[1]=1.0; action[6]=1.0 if ang<0.2 else 0.0
        db = float(((me.physics.location.x - ball.physics.location.x)**2 + (me.physics.location.y - ball.physics.location.y)**2)**0.5)
        if db<450 and ang<0.2:
            action[5]=1.0; action[2]=1.0
        return action, "SHOOT"

    if danger_zone:
        cx = 3072.0 if ball.physics.location.x >= 0 else -3072.0
        cy = own_goal_y + (500.0 if team==0 else -500.0)
        steer, ang = _steer_to(me, cx, cy)
        action[0]=steer; action[1]=1.0; action[6]=1.0 if ang<0.35 else 0.0
        dx = me.physics.location.x - ball.physics.location.x
        dy = me.physics.location.y - ball.physics.location.y
        if ball.physics.location.z<200 and (dx*dx+dy*dy)**0.5<450 and ang<0.25:
            action[5]=1.0; action[2]=-0.2
        return action, "CLEAR_CORNER"

    # If we're behind ball (goal side), shadow; else challenge/control
    gy = _goal_y(team)
    bx, by, bz = ball.physics.location.x, ball.physics.location.y, ball.physics.location.z
    goal_side = (team==0 and me.physics.location.y < by) or (team==1 and me.physics.location.y > by)

    if goal_side:
        shy = by - (800 if team==0 else -800)
        steer, ang = _steer_to(me, bx, shy)
        action[0]=steer; action[1]=0.8; action[6]=0.0
        return action, "SHADOW"
    else:
        # Simple controlled push toward far post
        aim_x = float(np.clip(bx*0.7, -900, 900))
        steer, ang = _steer_to(me, aim_x, gy*0.92)
        action[0]=steer; action[1]=0.9; action[6]=0.3 if ang<0.25 else 0.0
        db = float(((me.physics.location.x - bx)**2 + (me.physics.location.y - by)**2)**0.5)
        if db<380 and abs(me.physics.location.x - bx)<180 and bz<180 and ang<0.2:
            action[5]=1.0; action[2]=0.6
        return action, "CONTROL"
