"""Which camera's ball is more believable where the two disagree?

The joined tracking (K7) joins the players only; the ball every solution reads is still the tactical
camera's. Both cameras give a ball, and where both see it they are over 2 m apart on a quarter of the
frames. There is no true ball position to compare with, so three checks that need none:

  1. a referee that saw neither ball: the players-only estimate of ball_from_players.py, learned on
     the outside matches. Where the two balls are far apart, which one is nearer to it?
  2. football: a real ball is at a player's feet or travelling between players. How far is each ball
     from the nearest player?
  3. physics: how often does each ball make a step no ball can make?

and whether the disagreement is the ball in the air (a camera can only say where the ball appears on
the grass, and a high ball appears at a different spot from each angle): fast balls should disagree more.

    python experiments/ball_cameras.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Params
FPS = 25
STEP = 5
GAP_BANDS_M = (0, 2, 5, 10, 1e9)
SPEED_BANDS = (0, 5, 12, 20, 1e9)       # m/s: at feet, a ground pass, a firm pass, a long ball or shot
MAX_BALL_MPS = 40.0


def band_names(edges, unit):
    return ["{}-{} {}".format(a, b, unit) if b < 1e8 else "over {} {}".format(a, unit) for a, b in zip(edges[:-1], edges[1:])]


def nearest_player_m(xy, ball):
    with np.errstate(invalid="ignore"):
        d = np.hypot(xy[..., 0] - ball[:, None, 0], xy[..., 1] - ball[:, None, 1])
    return np.min(np.where(np.isfinite(d), d, np.inf), axis=1)


def step_speed(ball):
    v = np.full(len(ball), np.nan)
    v[1:] = np.hypot(*(ball[1:] - ball[:-1]).T) * FPS
    return v


OUT_EXTRA_M = 1.0            # on top of the chain's 1 m: the two cameras' balls are 0.76 m apart (median)
OUT_MIN_FRAMES = FPS         # "clearly out, and out for some time": at least a second in a row


def clearly_out(ball, pitch, margin=None, min_frames=None):
    """(N,) True where a ball is at least `margin` past a line for `min_frames` in a row: clearly out, and out for
    some time (defaults: the chain's 1 m + OUT_EXTRA_M, and OUT_MIN_FRAMES)."""
    from cleaning import runs
    from control_hmm import OUT_MARGIN_M
    m = OUT_MARGIN_M + OUT_EXTRA_M if margin is None else margin
    k = OUT_MIN_FRAMES if min_frames is None else min_frames
    length, width = pitch
    with np.errstate(invalid="ignore"):
        out = (ball[:, 0] < -m) | (ball[:, 0] > length + m) | (ball[:, 1] < -m) | (ball[:, 1] > width + m)
    for a, z in runs(out):
        if z - a + 1 < k:
            out[a:z + 1] = False
    return out


def broadcast_ball_out(players, ball, team, pitch):
    """(N,) True where the broadcast camera, and only it, shows the ball clearly off the pitch for a while
    (our idea, 2026-11-18). The broadcast ball is not good enough to say who has the ball (filling the
    tactical gaps with it made more short possessions, solutions 35 and V10 / V11), but "the ball is two
    metres past the line and stays there" needs no accuracy: it is a reason for a dead ball. Only frames
    where the tactical camera has no ball are used; where it has one, the chain already reads it."""
    from camera_agreement import broadcast_on_tactical_frames
    from cleaning import runs
    from control_hmm import OUT_MARGIN_M
    bball = broadcast_on_tactical_frames(players, team, None)[1].astype(float)
    m = OUT_MARGIN_M + OUT_EXTRA_M
    length, width = pitch
    with np.errstate(invalid="ignore"):
        out = (bball[:, 0] < -m) | (bball[:, 0] > length + m) | (bball[:, 1] < -m) | (bball[:, 1] > width + m)
    out &= ~np.isfinite(ball[:, 0])
    for a, z in runs(out):
        if z - a + 1 < OUT_MIN_FRAMES:
            out[a:z + 1] = False
    return out


def report():
    import ball_from_players as B
    from camera_agreement import broadcast_on_tactical_frames
    from external import IDSSE_MATCHES
    from score import graded_mask
    from tracking import load, load_clean
    tp, _, team = load_clean("K7")
    tb = load("tactical")[1].astype(float)          # the raw tactical ball: K5 / K7 do not change it where it is seen
    _, bball, _, on = broadcast_on_tactical_frames(tp, team, None)
    bball = bball.astype(float)
    g = graded_mask()
    both = g & on & np.isfinite(tb[:, 0]) & np.isfinite(bball[:, 0])
    gap = np.hypot(*(tb - bball).T)
    print("{} graded frames where both cameras see the ball".format(int(both.sum())))

    print("\n3. physics: steps faster than {:.0f} m/s, per 1000 ball frames: tactical {:.1f}, broadcast {:.1f}".format(
        MAX_BALL_MPS, 1000 * np.nanmean(step_speed(tb)[g] > MAX_BALL_MPS), 1000 * np.nanmean(step_speed(bball)[g & on] > MAX_BALL_MPS)))

    f = np.flatnonzero(both)[::STEP]
    f = f[(f >= B.VEL_HALF) & (f < len(tb) - B.VEL_HALF)]
    model, _ = B.train(sorted(IDSSE_MATCHES))
    xy = tp[f][..., :2].astype(float)
    _, est, _ = B.estimate(model, B.frames_xy(tp, team, f), tp[f][..., :2])
    dt, db = np.hypot(*(tb[f] - est).T), np.hypot(*(bball[f] - est).T)
    nt, nb = nearest_player_m(xy, tb[f]), nearest_player_m(xy, bball[f])
    band = np.digitize(gap[f], GAP_BANDS_M[1:-1])
    print("\n1 and 2, by how far apart the two balls are ({} frames, every {}th):".format(len(f), STEP))
    print("| balls apart | share of frames | tactical ball nearer the referee | referee to tactical / broadcast (median) | nearest player to tactical / broadcast (median) | within 2 m of a player: tactical / broadcast |")
    print("| --- | --- | --- | --- | --- | --- |")
    for k, name in enumerate(band_names(GAP_BANDS_M, "m")):
        m = band == k
        if m.sum() < 20:
            continue
        print("| {} | {:.1f}% | {:.0f}% | {:.1f} m / {:.1f} m | {:.1f} m / {:.1f} m | {:.0f}% / {:.0f}% |".format(
            name, 100 * m.mean(), 100 * np.mean(dt[m] < db[m]), np.median(dt[m]), np.median(db[m]),
            np.median(nt[m]), np.median(nb[m]), 100 * np.mean(nt[m] < 2), 100 * np.mean(nb[m] < 2)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.nanmax(np.stack([step_speed(tb), step_speed(bball)]), axis=0)
    sb = np.digitize(speed[f], SPEED_BANDS[1:-1])
    print("\nis it the ball in the air? balls over 5 m apart, by the faster ball's speed:")
    print(" | ".join("{}: {:.0f}% of {}".format(n, 100 * np.mean(gap[f][sb == k] > 5), int((sb == k).sum()))
                     for k, n in enumerate(band_names(SPEED_BANDS, "m/s")) if (sb == k).sum() >= 20))

    only_t = g & on & np.isfinite(tb[:, 0]) & ~np.isfinite(bball[:, 0])
    only_b = g & on & ~np.isfinite(tb[:, 0]) & np.isfinite(bball[:, 0])
    for name, m, ball in (("tactical only", only_t, tb), ("broadcast only", only_b, bball)):
        ff = np.flatnonzero(m)[::STEP]
        ff = ff[(ff >= B.VEL_HALF) & (ff < len(tb) - B.VEL_HALF)]
        _, e, _ = B.estimate(model, B.frames_xy(tp, team, ff), tp[ff][..., :2])
        n = nearest_player_m(tp[ff][..., :2].astype(float), ball[ff])
        print("ball seen by the {} ({} frames): {:.1f} m from the referee (median), {:.1f} m from the nearest player, within 2 m of a player {:.0f}%".format(
            name, int(m.sum()), np.median(np.hypot(*(ball[ff] - e).T)), np.median(n), 100 * np.mean(n < 2)))


def _self_test():
    xy = np.array([[[0.0, 0.0], [10.0, 0.0]], [[0.0, 0.0], [np.nan, np.nan]]])
    assert np.allclose(nearest_player_m(xy, np.array([[9.0, 0.0], [3.0, 4.0]])), [1.0, 5.0])
    assert np.isclose(step_speed(np.array([[0.0, 0.0], [1.0, 0.0]]))[1], FPS)
    assert band_names((0, 2, 1e9), "m") == ["0-2 m", "over 2 m"]
    b = np.full((100, 2), 50.0)
    b[10:40, 1] = -2.5                   # 2.5 m past the touchline for 1.2 s
    b[60:70, 1] = -2.5                   # 0.4 s only
    b[80:, 1] = -1.5                     # not clearly out
    got = clearly_out(b, (105.0, 68.0))
    assert got[10:40].all() and got.sum() == 30
    import camera_agreement
    real = camera_agreement.broadcast_on_tactical_frames
    bb = np.full((100, 2), np.nan)
    bb[10:50] = [-3.0, 30.0]            # 3 m past the goal line for 1.6 s
    bb[60:70] = [-3.0, 30.0]            # only 0.4 s: not "for some time"
    bb[80:99] = [-1.5, 30.0]            # 1.5 m out: within what the two cameras disagree by
    tb = np.full((100, 2), np.nan)
    tb[45:50] = [50.0, 30.0]            # the tactical camera sees a ball here: its word counts, not the broadcast's
    camera_agreement.broadcast_on_tactical_frames = lambda *a: (None, bb, None, None)
    try:
        got = broadcast_ball_out(None, tb, None, (105.0, 68.0))
    finally:
        camera_agreement.broadcast_on_tactical_frames = real
    assert got[10:45].all() and not got[45:].any() and not got[:10].any()
    print("ball_cameras self-test passed")


if __name__ == "__main__":
    _self_test()
    report()
