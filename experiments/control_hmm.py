"""Experiment 5: hold possession through noise, switch at once on strong control, place it with hindsight.

Experiment 3 gave every frame with a visible ball the same weight, whoever was nearest and however
fast the ball was moving. That flips possession on a pass flying past an opponent and is slow to
accept a clean reception. Here a seen ball counts according to how much it looks like control:

  * control: a player within 1.5 m of a slow ball and no opponent within 3 m -- strong evidence
  * contested control: the same, but an opponent is also close                  -- weak evidence
  * near: a player within 3 m of a slow ball                                     -- very weak
  * a fast ball (a pass or shot in flight), or nobody near                      -- no evidence

The ATD's own ball-speed channel is mostly empty and never exceeds 7.3, too slow for a kick, so
ball speed is measured from the ball's positions. A ball seen past a line is evidence of a dead
ball, and a fast ball is evidence that play is alive.

Hindsight: between the last touch of one team and the first control by the other, a decoder has no
reason to prefer any frame for the switch. The ground truth keeps a loose ball with the last team,
so each change of team is moved to the first frame of the new team's own evidence.

    python experiments/control_hmm.py --tau 25 10 5 3 2
"""

import argparse
import os

import numpy as np

from tracking import GAME, GAMES, ROOT, half_video_blocks, load, team_features
from baseline import HOME, AWAY, write_submission
from blackout_hmm import (P_STOP, P_SWITCH, decode, dead_evidence, fit_shape_model,
                          nearest_team, shape_matrix)

# Params
CONTROL_M, CONTEST_M, NEAR_M = 1.5, 3.0, 3.0
FAST_MPS = 8.0              # faster than any player runs with the ball: a pass or a shot
OUT_MARGIN_M = 1.0          # ball this far past a line is out
L_CONTROL = np.log(0.95 / 0.05)
L_CONTESTED = np.log(0.70 / 0.30)
L_NEAR = np.log(0.60 / 0.40)
L_OUT, L_FAST = 3.0, -2.0   # dead-ball evidence from the ball itself


def team_distances(players, ball, team):
    """(nearest home player, nearest away player) distance to the ball per frame, NaN if unseen."""
    d = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    with np.errstate(invalid="ignore"):
        d_home = np.where(np.all(np.isnan(d[:, team == HOME]), axis=1), np.nan,
                          np.nanmin(np.where(np.isnan(d[:, team == HOME]), np.inf, d[:, team == HOME]), axis=1))
        d_away = np.where(np.all(np.isnan(d[:, team == AWAY]), axis=1), np.nan,
                          np.nanmin(np.where(np.isnan(d[:, team == AWAY]), np.inf, d[:, team == AWAY]), axis=1))
    return d_home, d_away


def ball_speed(ball):
    """m/s from the ball's position one frame either side; NaN unless both neighbours are seen."""
    v = np.full(len(ball), np.nan)
    step = np.hypot(ball[2:, 0] - ball[:-2, 0], ball[2:, 1] - ball[:-2, 1])
    v[1:-1] = step * 25 / 2
    return v


def control_evidence(d_home, d_away, speed, count_near=True):
    """Signed log-odds for home (+) vs away (-) from how much a seen ball looks controlled.

    count_near=False is the "chasing is not control" rule: a player 1.5-3 m from the ball, say
    running after a long kick, says nothing until the ball is within 1.5 m of them.
    """
    with np.errstate(invalid="ignore"):
        home_near = d_home < d_away
        d_near = np.where(home_near, d_home, d_away)
        d_far = np.where(home_near, d_away, d_home)
        slow = ~(speed > FAST_MPS)  # unknown speed counts as slow
        strength = np.select(
            [slow & (d_near <= CONTROL_M) & (d_far > CONTEST_M),
             slow & (d_near <= CONTROL_M),
             slow & (d_near <= NEAR_M)],
            [L_CONTROL, L_CONTESTED, L_NEAR if count_near else 0.0], default=0.0)
    strength = np.where(np.isfinite(d_near), strength, 0.0)
    return np.where(home_near, strength, -strength)


def ball_dead_evidence(ball, speed, pitch):
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        out = (x < -OUT_MARGIN_M) | (x > length + OUT_MARGIN_M) | (y < -OUT_MARGIN_M) | (y > width + OUT_MARGIN_M)
        fast = speed > FAST_MPS
    return np.where(out, L_OUT, 0.0) + np.where(fast, L_FAST, 0.0)


def place_with_hindsight(labels, team_llr, blocks, min_evidence=0.0):
    """Move each H <-> A change forward to the first frame where the new team has evidence of its own.

    With min_evidence = L_CONTESTED the change waits for the new team's first frame of control
    (ball within 1.5 m of one of its players), not merely the first frame it was nearer.
    min_evidence may also be a per-frame array, e.g. strict only where the ball is seen, since
    control cannot be judged during a blackout.
    """
    out = labels.copy()
    thresholds = np.broadcast_to(np.asarray(min_evidence, dtype=float), team_llr.shape)
    for _, first, last in blocks:
        seg = out[first:last + 1]  # a view
        llr = team_llr[first:last + 1]
        need = np.maximum(thresholds[first:last + 1] - 1e-9, 0)
        changes = np.flatnonzero((seg[1:] != seg[:-1]) & (seg[1:] != "D") & (seg[:-1] != "D")) + 1
        bounds = np.append(np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg))
        for t in changes:
            stop = bounds[np.searchsorted(bounds, t, side="right")]
            sign = 1.0 if seg[t] == "H" else -1.0
            own = np.flatnonzero(llr[t:stop] * sign > need[t:stop])
            if len(own) and own[0] > 0:
                seg[t:t + own[0]] = seg[t - 1]
    return out


def _self_test():
    labels = np.array(list("HHHHAAAAAA"))
    llr = np.array([1, 1, 0, 0, 0, 0, -1, -1, 0, 0], dtype=float)
    assert "".join(place_with_hindsight(labels, llr, [(1, 0, 9)])) == "HHHHHHAAAA"
    labels = np.array(list("HHDDAAA"))
    assert "".join(place_with_hindsight(labels, np.zeros(7), [(1, 0, 6)])) == "HHDDAAA", "D edges untouched"
    assert control_evidence(np.array([1.0]), np.array([5.0]), np.array([2.0]))[0] == L_CONTROL
    assert control_evidence(np.array([1.0]), np.array([5.0]), np.array([15.0]))[0] == 0.0, "a pass in flight"
    assert control_evidence(np.array([2.0]), np.array([1.0]), np.array([2.0]))[0] == -L_CONTESTED
    # chasing a long kick: nearest is 2.5 m away, so it counts only when near counts
    assert control_evidence(np.array([2.5]), np.array([9.0]), np.array([4.0]))[0] == L_NEAR
    assert control_evidence(np.array([2.5]), np.array([9.0]), np.array([4.0]), count_near=False)[0] == 0.0
    labels = np.array(list("HHAAAAAA"))
    weak_then_control = np.array([1, 1, -L_NEAR, -L_NEAR, -L_CONTROL, -L_CONTROL, 0, 0])
    assert "".join(place_with_hindsight(labels, weak_then_control, [(1, 0, 7)])) == "HHAAAAAA"
    assert "".join(place_with_hindsight(labels, weak_then_control, [(1, 0, 7)], L_CONTESTED)) == "HHHHAAAA"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    parser.add_argument("--tau", type=float, nargs="+", default=[5.0])
    args = parser.parse_args()
    _self_test()

    players, ball, team = load()
    n_frames, pitch = len(players), GAMES[GAME]["pitch"]
    blocks = half_video_blocks(GAME)
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    feat = team_features(players, team)
    owner, dist = nearest_team(players, ball, team)
    X = shape_matrix(feat, blocks, pitch[0])
    model, accuracy, base_rate, _, _ = fit_shape_model(X, owner, dist, graded)
    finite = np.all(np.isfinite(X), axis=1)
    shape_llr = np.zeros(n_frames)
    shape_llr[finite] = np.clip(model.decision_function(X[finite]), -3, 3)

    speed = ball_speed(ball)
    seen = np.isfinite(speed[graded])
    print("ball speed from positions, graded frames with a value: {:.1%}; median {:.1f} m/s, "
          "share faster than {:g} m/s {:.1%}".format(seen.mean(), np.nanmedian(speed[graded]), FAST_MPS,
                                                     np.nanmean(speed[graded] > FAST_MPS)))
    d_home, d_away = team_distances(players, ball, team)
    ctrl = control_evidence(d_home, d_away, speed)
    g = ctrl[graded]
    print("seen-ball frames by evidence: control {:.1%}, contested {:.1%}, near {:.1%}, none {:.1%}".format(
        np.mean(np.abs(g) == L_CONTROL), np.mean(np.abs(g) == L_CONTESTED), np.mean(np.abs(g) == L_NEAR),
        np.mean((g == 0) & np.isfinite(ball[graded, 0]))))

    blackout = owner < 0
    team_llr = np.where(blackout, shape_llr, ctrl)
    dead_llr = dead_evidence(feat["speed_all"], graded)[0] + ball_dead_evidence(ball, speed, pitch)
    print("the transition priors imply a mean possession spell of {:.1f} s".format(1 / (P_SWITCH + P_STOP) / 25))

    for tau in args.tau:
        raw = decode(team_llr, dead_llr, blocks, n_frames, True, tau)
        for name, labels in (("ctrl_nohind", raw), ("ctrl_hind", place_with_hindsight(raw, team_llr, blocks))):
            tag = "{}_tau{:g}".format(name, tau)
            write_submission(os.path.join(args.out, tag + ".csv"), labels)
            gl = labels[graded]
            spells = sum(int(np.sum((s[1:] != s[:-1]) & (s[1:] != "D"))) + int(s[0] != "D")
                         for s in (labels[first:last + 1] for _, first, last in blocks))
            print("{:<18} D {:.3f}  spells {:>4}  mean {:5.1f} s".format(
                tag, np.mean(gl == "D"), spells, np.sum(gl != "D") / max(spells, 1) / 25))
