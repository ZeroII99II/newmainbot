import numpy as np, math

# Field and physics normalizers
MAX_X = 4096
MAX_Y = 5120
MAX_Z = 2000
MAX_VEL = 4000

# Keep function name/signature. Build 107-dim obs ~[-1,1]
def build_obs(packet, index) -> np.ndarray:
    try:
        ball = packet.game_ball
        me = packet.game_cars[index]
        opp = packet.game_cars[1 - index]
    except Exception:
        return np.zeros(107, dtype=np.float32)

    obs = np.zeros(107, dtype=np.float32)

    # Ball state
    obs[0:3] = [ball.physics.location.x / MAX_X,
                ball.physics.location.y / MAX_Y,
                ball.physics.location.z / MAX_Z]
    obs[3:6] = [ball.physics.velocity.x / MAX_VEL,
                ball.physics.velocity.y / MAX_VEL,
                ball.physics.velocity.z / MAX_VEL]

    # My car state
    obs[6:9] = [me.physics.location.x / MAX_X,
                me.physics.location.y / MAX_Y,
                me.physics.location.z / MAX_Z]
    obs[9:12] = [me.physics.velocity.x / MAX_VEL,
                 me.physics.velocity.y / MAX_VEL,
                 me.physics.velocity.z / MAX_VEL]
    r = me.physics.rotation
    obs[12:18] = [math.cos(r.yaw), math.sin(r.yaw),
                  math.cos(r.pitch), math.sin(r.pitch),
                  math.cos(r.roll), math.sin(r.roll)]
    obs[18] = getattr(me, "boost", 0.0) / 100.0

    # Relative ball position from me
    obs[19:22] = [(ball.physics.location.x - me.physics.location.x) / MAX_X,
                  (ball.physics.location.y - me.physics.location.y) / MAX_Y,
                  (ball.physics.location.z - me.physics.location.z) / MAX_Z]

    # Opponent state (limited)
    obs[22:25] = [opp.physics.location.x / MAX_X,
                  opp.physics.location.y / MAX_Y,
                  opp.physics.location.z / MAX_Z]
    obs[25:28] = [opp.physics.velocity.x / MAX_VEL,
                  opp.physics.velocity.y / MAX_VEL,
                  opp.physics.velocity.z / MAX_VEL]
    obs[28] = getattr(opp, "boost", 0.0) / 100.0

    # Remaining slots left as zeros
    return np.clip(obs, -1.0, 1.0).astype(np.float32)
