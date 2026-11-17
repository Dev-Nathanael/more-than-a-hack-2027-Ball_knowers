"""A double check that does not trust the ball's position.

    from double_check import comoving, movement_matrix, fit_movement_model

1. comoving: does the player nearest the ball move WITH it? A player who has the ball keeps it within
   reach and goes where it goes. A lofted ball projected onto the grass next to a player (the homography
   assumes the ball is on the ground) passes him at its own speed and direction: it is near him for a
   moment but does not move with him. Control should only count where the ball moves with its player.

2. movement_matrix + fit_movement_model: who has the ball, judged from the players alone. The team with
   the ball pushes forward, spreads out and sits higher; the team without it presses, runs back and
   closes up. A logistic regression learns that from frames we can trust (ball seen, clearly controlled,
   moving with its player) and gives P(home has the ball) on every frame, ball or no ball.

Self-consistency only: no ground truth is used to fit or to choose anything here.
"""

import warnings

import numpy as np

# Params
FPS = 25
COMOVE_HALF = 4          # frames either side: the ball must move with its player over about 0.3 s
COMOVE_M = 2.0           # it stays within reach (a dribble pushes the ball up to about 2 m ahead) ...
COMOVE_MPS = 3.0         # ... and its velocity stays within 3 m/s of the player's
SMOOTH = FPS             # movement features averaged over 1 s
SPRINT_MPS = 5.5
PRESS_M, PRESS_MPS = 5.0, 2.0   # a player within 5 m of an opponent closing on him at over 2 m/s presses
BLOCK = 60 * FPS         # alternate 60 s blocks for training and testing the movement model


def _moving_mean(x, w):
    x = np.asarray(x, np.float64)
    ok = np.isfinite(x)
    k = np.ones(w)
    num = np.convolve(np.where(ok, x, 0.0), k, "same")
    den = np.convolve(ok.astype(float), k, "same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def comoving(pos, vel, ball, ball_vel):
    """(T,) True where the player nearest the ball keeps it within COMOVE_M and its velocity within
    COMOVE_MPS of his own over +-COMOVE_HALF frames (the same player all along)."""
    n = len(ball)
    d = np.hypot(pos[..., 0] - ball[:, None, 0], pos[..., 1] - ball[:, None, 1])
    seen = np.isfinite(ball[:, 0]) & np.isfinite(d).any(axis=1)
    idx = np.zeros(n, int)
    idx[seen] = np.nanargmin(np.where(np.isfinite(d[seen]), d[seen], np.inf), axis=1)
    rows = np.arange(n)
    dist = np.where(seen, d[rows, idx], np.nan)
    rel = np.hypot(*(ball_vel - vel[rows, idx]).T)
    rel[~seen] = np.nan
    out = np.zeros(n, bool)
    for t in np.flatnonzero(seen):
        a, b = t - COMOVE_HALF, t + COMOVE_HALF + 1
        if a < 0 or b > n or not np.all(idx[a:b] == idx[t]):
            continue
        dd, rr = dist[a:b], rel[a:b]
        if np.all(np.isfinite(dd)) and np.all(np.isfinite(rr)) and dd.max() <= COMOVE_M and np.median(rr) <= COMOVE_MPS:
            out[t] = True
    return out


ACCEL_HALF = 12          # frames either side: change of velocity over about 1 s


def reacting_columns(vel, team):
    """[acceleration home - away, moving as a block home - away]. The team WITHOUT the ball reacts to every
    pass: its players keep changing velocity, and they shift together, all one way. Measured on labelled
    outside matches (idsse_explore.py): each tells who has the ball with AUC about 0.70, and with the other
    players-only measures lifts a left-out match from 0.867 to 0.890."""
    k = ACCEL_HALF
    dv = np.full(vel.shape, np.nan)
    dv[k:-k] = vel[2 * k:] - vel[:-2 * k]
    speed = np.hypot(vel[..., 0], vel[..., 1])
    accel, block = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # frames where a team has nobody tracked
        for s in (0, 1):
            accel[s] = np.nanmean(np.hypot(dv[:, team == s, 0], dv[:, team == s, 1]), axis=1)
            mean_v = np.nanmean(vel[:, team == s], axis=1)
            block[s] = np.hypot(mean_v[:, 0], mean_v[:, 1]) / np.nanmean(speed[:, team == s], axis=1)
    return [accel[0] - accel[1], block[0] - block[1]]


def movement_matrix(pos, vel, team, attacks_right, pitch_length, reacting=False):
    """(T, 8) features from the players only, all along HOME's attacking direction, smoothed over 1 s:
    forward speed home / away, territory, home block ahead of away, length and width difference,
    sprinters forward home minus away, pressers home minus away."""
    direction = np.where(attacks_right, 1.0, -1.0)
    cols = []
    fwd = {}
    for side, sign in ((0, 1.0), (1, -1.0)):
        fwd[side] = np.nanmean(vel[:, team == side, 0], axis=1) * direction * sign   # towards the goal it attacks
    x = pos[..., 0]
    cx = {s: np.nanmean(x[:, team == s], axis=1) for s in (0, 1)}
    length = {s: np.nanstd(x[:, team == s], axis=1) for s in (0, 1)}
    width = {s: np.nanstd(pos[:, team == s, 1], axis=1) for s in (0, 1)}
    speed = np.hypot(vel[..., 0], vel[..., 1])
    forward_v = vel[..., 0] * direction[:, None] * np.where(team == 0, 1.0, -1.0)[None, :]
    sprint = (speed > SPRINT_MPS) & (forward_v > 0)
    press_h, press_a = pressers(pos, vel, team)
    cols = [fwd[0], fwd[1], (np.nanmean(x, axis=1) - pitch_length / 2) * direction, (cx[0] - cx[1]) * direction,
            length[0] - length[1], width[0] - width[1],
            sprint[:, team == 0].sum(axis=1) - sprint[:, team == 1].sum(axis=1), press_h - press_a]
    if reacting:
        cols += reacting_columns(vel, team)
    return np.column_stack([_moving_mean(c, SMOOTH) for c in cols])


def pressers(pos, vel, team, chunk=10000):
    """(home pressers, away pressers) per frame: players within PRESS_M of an opponent and running at him
    faster than PRESS_MPS. Done in chunks: all home-away pairs at once would need gigabytes."""
    n = len(pos)
    ph_all, pa_all, vh_all, va_all = pos[:, team == 0], pos[:, team == 1], vel[:, team == 0], vel[:, team == 1]
    out_h, out_a = np.zeros(n), np.zeros(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        for a in range(0, n, chunk):
            ph, pa, vh, va = ph_all[a:a + chunk], pa_all[a:a + chunk], vh_all[a:a + chunk], va_all[a:a + chunk]
            gap = pa[:, None, :, :] - ph[:, :, None, :]                    # home -> away
            dist = np.hypot(gap[..., 0], gap[..., 1])
            unit = gap / np.where(dist > 1e-6, dist, np.nan)[..., None]
            near = dist < PRESS_M
            toward_a = np.sum(unit * vh[:, :, None, :], axis=3)           # home player running at the away one
            toward_h = -np.sum(unit * va[:, None, :, :], axis=3)          # away player running at the home one
            out_h[a:a + chunk] = np.any(near & (toward_a > PRESS_MPS), axis=2).sum(axis=1)
            out_a[a:a + chunk] = np.any(near & (toward_h > PRESS_MPS), axis=1).sum(axis=1)
    return out_h, out_a


def fit_movement_model(X, y, trusted):
    """(model, held-out accuracy, held-out AUC, n train, n test). y = 1 where home has the ball; trained on
    `trusted` frames of even 60 s blocks, tested on the odd ones."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    usable = trusted & np.all(np.isfinite(X), axis=1)
    test_block = (np.arange(len(X)) // BLOCK) % 2 == 1
    train, test = usable & ~test_block, usable & test_block
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X[train], y[train])
    p = model.predict_proba(X[test])[:, 1]
    return model, float(np.mean((p > 0.5) == y[test])), float(roc_auc_score(y[test], p)), int(train.sum()), int(test.sum())


def movement_opinion(model, X):
    """(T,) P(home has the ball) from movement alone; NaN where a feature is missing."""
    out = np.full(len(X), np.nan)
    ok = np.all(np.isfinite(X), axis=1)
    out[ok] = model.predict_proba(X[ok])[:, 1]
    return out


def _reacting_self_test():
    n = 100
    vel = np.zeros((n, 4, 2))
    vel[:, :2, 0] = 3.0                                     # home: both run the same way, steadily
    vel[:, 2, 0] = np.where(np.arange(n) % 50 < 25, 3.0, -3.0)   # away: one turns round every second ...
    vel[:, 3, 1] = 3.0                                      # ... the other runs across
    accel, block = reacting_columns(vel, np.array([0, 0, 1, 1]))
    assert np.nanmean(accel) < 0 and np.nanmean(block) > 0, "away accelerates more, home moves as a block"


def _self_test():
    _reacting_self_test()
    n = 40
    pos = np.zeros((n, 2, 2)); pos[:, 1] = [30.0, 30.0]
    pos[:, 0, 0] = np.arange(n) * 0.2                          # player 0 runs along x at 5 m/s
    vel = np.zeros((n, 2, 2)); vel[:, 0, 0] = 5.0
    ball = pos[:, 0].copy() + [0.5, 0.0]                       # dribbled half a metre ahead
    bvel = vel[:, 0].copy()
    assert comoving(pos, vel, ball, bvel)[10:30].all(), "a dribble moves with its player"
    ball2 = ball.copy(); ball2[:, 1] = 0.3                     # a ball crossing his path at 15 m/s sideways
    bvel2 = np.zeros((n, 2)); bvel2[:, 1] = 15.0
    assert not comoving(pos, vel, ball2, bvel2)[10:30].any(), "a ball passing at its own speed is not his"
    pp = np.array([[[0.0, 0.0], [3.0, 0.0]]]); vv = np.array([[[4.0, 0.0], [0.0, 0.0]]])
    h, a = pressers(pp, vv, np.array([0, 1]))
    assert h[0] == 1 and a[0] == 0, (h, a)                     # home runs at away from 3 m
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6000, 2)); y = (X[:, 0] + 0.3 * rng.normal(size=6000) > 0).astype(int)
    model, acc, auc, _, _ = fit_movement_model(X, y, np.ones(6000, bool))
    assert acc > 0.85 and auc > 0.9, (acc, auc)


if __name__ == "__main__":
    _self_test()
    print("double_check self-test passed")
