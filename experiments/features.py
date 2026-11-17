"""Features built from the tracking: ball motion, player facing, pressure on the ball, team shape.

All inputs are in metres and seconds (experiments/tracking.py); NaN means unseen. Nothing here
uses labels. These feed the viewer and are candidate evidence for later solutions.

The tracking has no height: x runs along the pitch and y across it. Whether the ball is in the
air can only be inferred; airborne_gap() uses the gap between where the two cameras' trackers put
the ball, which grows with height if each tracker projects the ball onto the grass.
"""

import numpy as np

# Params
FPS = 25
VEL_HALF = 2            # frames either side for ball velocity (0.16 s window)
EDGE_VEL = False        # True: backward velocity where the central one needs an unseen frame (cleaning loop)
RADIAL = False          # True: feature_cache adds radial_features (running at / away from the ball)
TOUCH_SPAN = 3          # frames before / after compared to spot a change of ball velocity
TOUCH_DV = 4.0          # m/s change in velocity that counts as a touch
TOUCH_M = 2.0           # a player this close to the ball gets the touch
ROLL_DECEL = 1.5        # m/s^2, a rolling ball slowing on grass, for the first second of a blackout
PHYSICS_FRAMES = FPS    # predict with physics for 1 s, then follow the players
FOLLOW_RATE = 0.15      # share of the gap to the players' target closed each frame
PLAYER_HALF = 3         # frames either side for player velocity
MOVING_MPS = 1.5        # above this a player is taken to face where they run
VIEW_HALF_DEG = 120     # 240 degree field of view: 120 either side of the heading
PRESS_NEAR, PRESS_FAR = 2.0, 5.0   # pressure zones around the ball carrier, metres
SPRINT_MPS = 5.5        # about 20 km/h, the usual high-intensity running threshold
CLOSING_M = 15.0        # players this close to the ball ...
CLOSING_MPS = 2.0       # ... running towards it this fast are closing it down
CARRIER_M = 1.5         # the ball carrier is the player within this of a slow ball
SLOW_MPS = 8.0
DUEL_BALL_M, DUEL_PAIR_M, DUEL_CLOSE_MPS = 10.0, 5.0, 3.0   # two opponents near the ball running at each other


def central_velocity(xy, half):
    """(N, ..., 2) positions -> velocity in m/s from frames t-half and t+half; NaN at gaps."""
    v = np.full(xy.shape, np.nan, dtype=np.float32)
    v[half:-half] = (xy[2 * half:] - xy[:-2 * half]) * FPS / (2 * half)
    return v


def ball_motion(ball):
    """Per frame: velocity (N, 2) m/s, speed, heading in degrees, acceleration magnitude m/s^2."""
    v = central_velocity(ball, VEL_HALF)
    if EDGE_VEL:
        # just before a gap the central difference needs unseen frames; look back over the same span
        back = np.full(ball.shape, np.nan, dtype=np.float32)
        back[2 * VEL_HALF:] = (ball[2 * VEL_HALF:] - ball[:-2 * VEL_HALF]) * FPS / (2 * VEL_HALF)
        v = np.where(np.isfinite(v), v, back)
    speed = np.hypot(v[:, 0], v[:, 1])
    heading = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    acc = np.full(len(ball), np.nan, dtype=np.float32)
    acc[VEL_HALF:-VEL_HALF] = np.hypot(*(v[2 * VEL_HALF:] - v[:-2 * VEL_HALF]).T) * FPS / (2 * VEL_HALF)
    return v, speed, heading, acc


def touches(ball, players, team):
    """Per frame: 0 none, 1 home touch, 2 away touch, 3 change with nobody within 2 m (bounce, post).

    A touch is a frame where the ball's velocity over the next 3 frames differs from the previous
    3 by more than 4 m/s, kept only at the local peak so one kick is one touch.
    """
    n, k = len(ball), TOUCH_SPAN
    dv = np.full(n, np.nan)
    before = (ball[k:-k] - ball[:-2 * k]) * FPS / k
    after = (ball[2 * k:] - ball[k:-k]) * FPS / k
    dv[k:-k] = np.hypot(*(after - before).T)
    out = np.zeros(n, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        cand = np.flatnonzero(dv > TOUCH_DV)
    for t in cand:
        lo, hi = max(0, t - k), min(n, t + k + 1)
        if dv[t] < np.nanmax(dv[lo:hi]):
            continue
        d = np.hypot(*(players[t, :, :2] - ball[t]).T)
        if np.all(np.isnan(d)) or np.nanmin(d) > TOUCH_M:
            out[t] = 3
        else:
            out[t] = 1 if team[np.nanargmin(d)] == 0 else 2
    return out


def predict_ball(ball, vel, players, pitch, blocks):
    """Ball position estimate everywhere in the halves: (pos (N, 2), source, radius m).

    source 0 = seen, 1 = physics (rolls on from the last seen velocity, slowing, for 1 s),
    2 = following the players (moves towards where the nearby players will be in 0.5 s).
    radius is a rough uncertainty that grows with time since the ball was seen.
    """
    n = len(ball)
    pos = np.full((n, 2), np.nan, dtype=np.float32)
    src = np.full(n, -1, dtype=np.int8)
    rad = np.full(n, np.nan, dtype=np.float32)
    pvel = central_velocity(players[:, :, :2], PLAYER_HALF)
    lo, hi = np.array([-2.0, -2.0]), np.array([pitch[0] + 2.0, pitch[1] + 2.0])
    for _, first, last in blocks:
        p = v = None
        age = 0
        for t in range(first, last + 1):
            if np.isfinite(ball[t, 0]):
                p = ball[t].astype(float)
                v = np.nan_to_num(vel[t]).astype(float)
                age = 0
                pos[t], src[t], rad[t] = p, 0, 0.0
                continue
            if p is None:
                continue
            age += 1
            if age <= PHYSICS_FRAMES:
                speed = np.hypot(*v)
                if speed > 0:
                    v = v * max(0.0, speed - ROLL_DECEL / FPS) / speed
                p = p + v / FPS
                src[t] = 1
            else:
                xy, pv = players[t, :, :2], pvel[t]
                ok = np.isfinite(xy[:, 0])
                if ok.any():
                    ahead = xy[ok] + np.nan_to_num(pv[ok]) * 0.5
                    d = np.hypot(*(xy[ok] - p).T)
                    nearest = np.argsort(d)[:4]
                    w = 1.0 / (1.0 + d[nearest])
                    target = (ahead[nearest] * w[:, None]).sum(0) / w.sum()
                    p = p + FOLLOW_RATE * (target - p)
                src[t] = 2
            p = np.clip(p, lo, hi)
            pos[t], rad[t] = p, min(25.0, 1.0 + 2.0 * age / FPS)
    return pos, src, rad


def airborne_gap(tball, bball_on_tactical):
    """Metres between where the two cameras' trackers put the ball; NaN unless both see it.

    If each tracker projects the ball onto the grass, a ball in the air lands in different places
    for the two cameras and the gap grows with its height. A large gap is a hint, not a height.
    """
    return np.hypot(*(tball - bball_on_tactical).T).astype(np.float32)


def player_headings(players, ball_pos):
    """(N, S) heading in degrees: direction of running when moving, else towards the ball.

    Where neither is known the last heading is kept. A track that swaps person scrambles it briefly.
    """
    xy = players[:, :, :2]
    v = central_velocity(xy, PLAYER_HALF)
    speed = np.hypot(v[..., 0], v[..., 1])
    run = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    to_ball = np.degrees(np.arctan2(ball_pos[:, None, 1] - xy[..., 1], ball_pos[:, None, 0] - xy[..., 0]))
    with np.errstate(invalid="ignore"):
        heading = np.where(speed > MOVING_MPS, run, to_ball)
    # carry the last known heading through gaps, per track
    idx = np.where(np.isfinite(heading), np.arange(len(heading))[:, None], -1)
    np.maximum.accumulate(idx, axis=0, out=idx)
    filled = np.take_along_axis(heading, np.maximum(idx, 0), axis=0)
    filled[idx < 0] = np.nan
    filled[~np.isfinite(xy[..., 0])] = np.nan
    return filled.astype(np.float32)


def in_view(heading_deg, from_xy, to_xy):
    """True where to_xy lies inside the 240 degree view of someone at from_xy facing heading_deg."""
    angle = np.degrees(np.arctan2(to_xy[..., 1] - from_xy[..., 1], to_xy[..., 0] - from_xy[..., 0]))
    diff = np.abs((angle - heading_deg + 180) % 360 - 180)
    return diff <= VIEW_HALF_DEG


def pressure(players, team, ball, ball_speed, headings):
    """Pressure on the ball carrier per frame.

    The carrier is the player within 1.5 m of a slow, seen ball. Returns a dict of arrays:
    carrier slot (-1 if none), opponents within 2 m and 5 m, opponents within 5 m in the carrier's
    blind spot, nearest opponent distance, closing speed of the nearest opponent (m/s, + = closing),
    and an index: sum over opponents within 5 m of (1 - d / 5), counted 1.5 times from the blind side.
    """
    n = len(ball)
    out = {k: np.full(n, -1 if k == "carrier" else np.nan, dtype=np.int16 if k == "carrier" else np.float32)
           for k in ("carrier", "near", "far", "blind", "nearest", "closing", "index")}
    xy = players[:, :, :2]
    pvel = central_velocity(xy, PLAYER_HALF)
    d_ball = np.hypot(xy[..., 0] - ball[:, None, 0], xy[..., 1] - ball[:, None, 1])
    with np.errstate(invalid="ignore"):
        has = np.isfinite(ball[:, 0]) & ~(ball_speed > SLOW_MPS) & (np.nanmin(np.where(np.isnan(d_ball), np.inf, d_ball), axis=1) <= CARRIER_M)
    for t in np.flatnonzero(has):
        c = int(np.nanargmin(d_ball[t]))
        opp = np.flatnonzero((team != team[c]) & np.isfinite(xy[t, :, 0]))
        out["carrier"][t] = c
        if not len(opp):
            for k in ("near", "far", "blind", "index"):
                out[k][t] = 0
            continue
        rel = xy[t, opp] - xy[t, c]
        d = np.hypot(*rel.T)
        blind = ~in_view(headings[t, c], xy[t, c], xy[t, opp]) if np.isfinite(headings[t, c]) else np.zeros(len(opp), bool)
        close = d <= PRESS_FAR
        out["near"][t] = int(np.sum(d <= PRESS_NEAR))
        out["far"][t] = int(np.sum(close))
        out["blind"][t] = int(np.sum(close & blind))
        out["index"][t] = float(np.sum((1 - d[close] / PRESS_FAR) * np.where(blind[close], 1.5, 1.0)))
        j = int(np.argmin(d))
        out["nearest"][t] = d[j]
        rv = np.nan_to_num(pvel[t, opp[j]]) - np.nan_to_num(pvel[t, c])
        out["closing"][t] = -float(np.dot(rv, rel[j]) / max(d[j], 1e-6))
    return out


def team_shape(players, team, gk, attacks_right, pitch_length):
    """Per frame per team, outfield players only: width, length (10th-90th percentile spread, so a
    stray track does not stretch it), centroid along the attack, and the defensive line's distance
    from the team's own goal line. attacks_right: (N,) True where home attacks towards x = length.
    """
    import warnings
    out = {}
    xy = players[:, :, :2]
    for name, side in (("H", 0), ("A", 1)):
        cols = (team == side) & ~gk
        x, y = xy[:, cols, 0], xy[:, cols, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out["width_" + name] = (np.nanpercentile(y, 90, axis=1) - np.nanpercentile(y, 10, axis=1)).astype(np.float32)
            out["length_" + name] = (np.nanpercentile(x, 90, axis=1) - np.nanpercentile(x, 10, axis=1)).astype(np.float32)
            right = attacks_right if side == 0 else ~attacks_right
            rear = np.where(right, np.nanpercentile(x, 5, axis=1), pitch_length - np.nanpercentile(x, 95, axis=1))
            out["line_" + name] = rear.astype(np.float32)
    return out


def team_movement(players, team, gk, attacks_right, ball_pos):
    """Per frame per team, outfield players: mean speed, mean forward speed (towards the opponent's
    goal; negative = recovery running), number sprinting, and number closing down the ball."""
    import warnings
    xy = players[:, :, :2]
    v = central_velocity(xy, PLAYER_HALF)
    out = {}
    for name, side in (("H", 0), ("A", 1)):
        cols = (team == side) & ~gk
        vx, vy = v[:, cols, 0], v[:, cols, 1]
        sign = np.where(attacks_right if side == 0 else ~attacks_right, 1.0, -1.0)[:, None]
        speed = np.hypot(vx, vy)
        rx, ry = ball_pos[:, None, 0] - xy[:, cols, 0], ball_pos[:, None, 1] - xy[:, cols, 1]
        dist = np.hypot(rx, ry)
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            toward = (vx * rx + vy * ry) / dist
            out["speed_" + name] = np.nanmean(speed, axis=1).astype(np.float32)
            out["forward_" + name] = np.nanmean(vx * sign, axis=1).astype(np.float32)
            out["sprint_" + name] = np.sum(speed > SPRINT_MPS, axis=1).astype(np.int8)
            out["closing_" + name] = np.sum((dist < CLOSING_M) & (toward > CLOSING_MPS), axis=1).astype(np.int8)
    return out


def radial_features(players, team, gk, ball_pos):
    """Per frame per team, outfield players: how fast they run at the ball (m/s, + = closing).

    near: the team's nearest player; near3: mean of its 3 nearest; away10: how many within 10 m run
    away from the ball faster than 1 m/s. Defenders close on the ball harder than attackers (nearest
    +1.8 vs +1.0 m/s) and attackers near the ball run away more, getting open (ball_runs_report.py).
    """
    import warnings
    xy = players[:, :, :2]
    v = central_velocity(xy, PLAYER_HALF)
    rel = ball_pos[:, None, :] - xy
    d = np.hypot(rel[..., 0], rel[..., 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (v[..., 0] * rel[..., 0] + v[..., 1] * rel[..., 1]) / np.maximum(d, 0.1)
    out = {}
    for name, side in (("H", 0), ("A", 1)):
        cols = np.flatnonzero((team == side) & ~gk)
        dd = np.where(np.isfinite(d[:, cols]) & np.isfinite(r[:, cols]), d[:, cols], np.inf)
        order = np.argsort(dd, axis=1)
        rr = np.take_along_axis(r[:, cols], order, axis=1)
        ok = np.take_along_axis(dd, order, axis=1) < np.inf
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            out["near_" + name] = np.where(ok[:, 0], rr[:, 0], np.nan).astype(np.float32)
            out["near3_" + name] = np.nanmean(np.where(ok[:, :3], rr[:, :3], np.nan), axis=1).astype(np.float32)
            out["away10_" + name] = np.sum((dd < 10) & (r[:, cols] < -1.0), axis=1).astype(np.int8)
    return out


def duels(pos, vel, team, ball):
    """Per frame: opposing pairs near the ball and near each other that close on each other fast."""
    ph, pa, vh, va = pos[:, team == 0], pos[:, team == 1], vel[:, team == 0], vel[:, team == 1]
    near_h = np.hypot(*(ph - ball[:, None]).transpose(2, 0, 1)) < DUEL_BALL_M
    near_a = np.hypot(*(pa - ball[:, None]).transpose(2, 0, 1)) < DUEL_BALL_M
    d = ph[:, :, None, :] - pa[:, None, :, :]
    gap = np.hypot(d[..., 0], d[..., 1])
    closing = -np.sum(d * (vh[:, :, None, :] - va[:, None, :, :]), axis=3) / np.where(gap > 1e-6, gap, np.nan)
    ok = near_h[:, :, None] & near_a[:, None, :] & (gap < DUEL_PAIR_M) & (closing > DUEL_CLOSE_MPS)
    return ok.sum(axis=(1, 2)).astype(float)


def zone_of(ball_x, attacks_right, pitch_length):
    """Third of the pitch from the attacking team's view: 1 own third (build-up), 2 middle
    (progression), 3 final third (near goal), 0 if the ball position is unknown."""
    progress = np.where(attacks_right, ball_x, pitch_length - ball_x)
    with np.errstate(invalid="ignore"):
        zone = 1 + (progress > pitch_length / 3).astype(np.int8) + (progress > 2 * pitch_length / 3).astype(np.int8)
    return np.where(np.isfinite(ball_x), zone, 0).astype(np.int8)


def _self_test():
    # a ball rolling right, kicked upward at frame 20 next to a home player
    n = 40
    ball = np.zeros((n, 2), np.float32)
    ball[:20, 0] = np.arange(20) * 0.4            # 10 m/s along x
    ball[20:, 0] = ball[19, 0]
    ball[20:, 1] = np.arange(n - 20) * 0.4        # then 10 m/s along y
    players = np.full((n, 2, 3), np.nan, np.float32)
    players[:, 0, :2] = [ball[20, 0] + 0.5, 0.0]  # home player at the kick
    players[:, 1, :2] = [50.0, 30.0]              # away player far away
    team = np.array([0, 1])
    t = touches(ball, players, team)
    assert np.flatnonzero(t).tolist() and set(t[np.flatnonzero(t)]) == {1}, t
    assert abs(int(np.flatnonzero(t)[0]) - 19) <= 1
    v, speed, heading, acc = ball_motion(ball)
    assert abs(speed[10] - 10) < 1e-3 and abs(heading[10]) < 1e-3 and abs(heading[30] - 90) < 1e-3
    # physics prediction rolls on along the last direction
    hidden = ball.copy()
    hidden[25:] = np.nan
    pos, src, rad = predict_ball(hidden, v, players, (105.0, 68.0), [(1, 0, n - 1)])
    assert src[26] == 1 and pos[26, 1] > pos[24, 1] and rad[26] > 0
    # the last seen frame before a gap has no central velocity; EDGE_VEL looks back instead
    assert np.isnan(ball_motion(hidden)[0][24]).all()
    global EDGE_VEL
    EDGE_VEL = True
    try:
        assert abs(ball_motion(hidden)[1][24] - 10) < 1e-3
    finally:
        EDGE_VEL = False
    # facing: a player running right faces 0 degrees; the ball behind them is out of view
    run = np.zeros((10, 1, 3), np.float32)
    run[:, 0, 0] = np.arange(10) * 0.2
    h = player_headings(run, np.full((10, 2), np.nan, np.float32))
    assert abs(h[5, 0]) < 1e-3
    assert in_view(np.float32(0), np.array([0, 0]), np.array([5, 1])) and not in_view(np.float32(0), np.array([0, 0]), np.array([-5, 0.1]))
    # a home side running right while attacking right moves forward; the away side runs back
    m = np.zeros((10, 2, 3), np.float32)
    m[:, 0, 0] = 40 + np.arange(10) * 0.2      # home, 5 m/s to the right
    m[:, 1, 0] = 60 + np.arange(10) * 0.2      # away, also to the right, which is towards its own goal
    mv = team_movement(m, np.array([0, 1]), np.array([False, False]), np.ones(10, bool), np.full((10, 2), [45.0, 0.0]))
    assert mv["forward_H"][5] > 4 and mv["forward_A"][5] < -4 and mv["closing_H"][5] == 1
    # radial: a home player running at a ball 10 m ahead at 5 m/s closes at +5; an away one running off at -5
    rp = np.full((10, 2, 3), np.nan, np.float32)
    rp[:, 0, 0] = 40 + np.arange(10) * 0.2
    rp[:, 1, 0] = 60 + np.arange(10) * 0.2
    rp[:, :, 1] = 30.0
    rad = radial_features(rp, np.array([0, 1]), np.array([False, False]), np.full((10, 2), [55.0, 30.0], np.float32))
    assert abs(rad["near_H"][5] - 5) < 1e-3 and abs(rad["near_A"][5] + 5) < 1e-3 and rad["away10_A"][5] == 1
    z = zone_of(np.array([10.0, 50.0, 90.0, np.nan]), np.array([True, True, True, True]), 105.0)
    assert z.tolist() == [1, 2, 3, 0]
    assert zone_of(np.array([10.0]), np.array([False]), 105.0).tolist() == [3]


if __name__ == "__main__":
    _self_test()
    print("features self-test passed")
