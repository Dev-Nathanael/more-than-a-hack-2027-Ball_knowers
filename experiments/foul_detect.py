"""Likely foul moments: a challenge, then the whole field slows, and the ball did not go out.

    from foul_detect import detect_fouls
    fouls = detect_fouls(players, ball, team, ball_out, blocks)   # [(frame, speed drop m/s)]

A foul is a challenge (two opponents within CONTACT_M of each other near the ball) followed by the whistle,
after which everyone stops running: the players' mean speed over the next 1-3 s falls by at least DROP_MPS
and to SLOW_MPS or less. If the ball goes out around it, it is a ball out instead. One candidate per
SEPARATION (the strongest drop). Thresholds are set from the game (a LaLiga match has about 25 fouls, used
only as a sanity check on the count), not tuned on labelled data. Candidates are notes for the viewer.
"""

import sys

import numpy as np

# Params
FPS = 25
CONTACT_M = 1.0          # two opponents this close near the ball: a challenge
DROP_MPS = 0.5           # the whistle: the mean player speed falls at least this much ...
SLOW_MPS = 1.5           # ... to walking pace or below
BEFORE = FPS             # speed before = the second up to the challenge
AFTER = (FPS, 3 * FPS)   # speed after = 1 to 3 s after it
OUT_WINDOW = (-FPS, 3 * FPS)
SEPARATION = 10 * FPS


def closest_pair(pos, team, ball, near_m=10.0):
    """Per frame: distance (m) of the closest home-away pair with both players within near_m of the ball."""
    ph, pa = pos[:, team == 0], pos[:, team == 1]
    nh = np.hypot(*(ph - ball[:, None]).transpose(2, 0, 1)) < near_m
    na = np.hypot(*(pa - ball[:, None]).transpose(2, 0, 1)) < near_m
    d = np.hypot(*(ph[:, :, None, :] - pa[:, None, :, :]).transpose(3, 0, 1, 2))
    d = np.where(nh[:, :, None] & na[:, None, :], d, np.inf)
    out = d.min(axis=(1, 2))
    return np.where(np.isfinite(out), out, np.nan)


def foul_candidates(players, ball, team, ball_out, blocks):
    """detect_fouls from the tracking: players' mean speed and the closest opponent pair near the ball."""
    from features import central_velocity
    pos = players[..., :2]
    vel = np.stack([central_velocity(pos[:, i], 3) for i in range(pos.shape[1])], axis=1)
    mean_speed = np.nanmean(np.hypot(vel[..., 0], vel[..., 1]), axis=1)
    return detect_fouls(mean_speed, closest_pair(pos, team, ball), ball_out, blocks)


def detect_fouls(mean_speed, pair, ball_out, blocks):
    """[(frame, drop)] of likely fouls. mean_speed: players' mean speed per frame (m/s); pair: closest
    opponent pair near the ball (m, NaN when none); ball_out: ball past a line."""
    n = len(mean_speed)
    graded = np.zeros(n, bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    cand = []
    for c in np.flatnonzero(graded & (np.nan_to_num(pair, nan=99.0) <= CONTACT_M)):
        if c - BEFORE < 0 or c + AFTER[1] >= n or not graded[c + AFTER[1]]:
            continue
        before = np.nanmean(mean_speed[c - BEFORE:c + 1])
        after = np.nanmean(mean_speed[c + AFTER[0]:c + AFTER[1]])
        if not (np.isfinite(before) and np.isfinite(after)):
            continue
        if before - after >= DROP_MPS and after <= SLOW_MPS and not ball_out[max(c + OUT_WINDOW[0], 0):c + OUT_WINDOW[1]].any():
            cand.append((int(c), float(before - after)))
    out = []
    for c, drop in sorted(cand, key=lambda x: -x[1]):   # strongest first, one per SEPARATION
        if all(abs(c - k) >= SEPARATION for k, _ in out):
            out.append((c, drop))
    return sorted(out)


WHISTLE_DROP_MPS = 2.0    # a whistle with no challenge to see (offside, a foul away from the ball): the players'
WHISTLE_SLOW_MPS = 1.5    # mean speed falls this much, to walking pace. Set on the IDSSE matches (whistle_report)


def detect_whistles(mean_speed, blocks, drop=WHISTLE_DROP_MPS, slow=WHISTLE_SLOW_MPS):
    """[(frame, drop)] where play stops on a whistle, from the players alone: within about two seconds
    the mean player speed falls by at least `drop` to `slow` or below. No ball, no challenge and no tag is
    needed, which is what an offside looks like when the ball is out of the picture. Same windows as
    detect_fouls; one per SEPARATION, strongest first."""
    n = len(mean_speed)
    graded = np.zeros(n, bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(mean_speed))])
    cnt = np.concatenate([[0], np.cumsum(np.isfinite(mean_speed))])
    c = np.arange(BEFORE, n - AFTER[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        before = (csum[c + 1] - csum[c - BEFORE]) / (cnt[c + 1] - cnt[c - BEFORE])
        after = (csum[c + AFTER[1]] - csum[c + AFTER[0]]) / (cnt[c + AFTER[1]] - cnt[c + AFTER[0]])
    ok = graded[c] & graded[c + AFTER[1]] & np.isfinite(before) & np.isfinite(after) & (before - after >= drop) & (after <= slow)
    out = []
    for i in sorted(np.flatnonzero(ok), key=lambda i: -(before[i] - after[i])):
        if all(abs(c[i] - k) >= SEPARATION for k, _ in out):
            out.append((int(c[i]), float(before[i] - after[i])))
    return sorted(out)


def whistle_candidates(players, blocks):
    """detect_whistles from the tracking (the same mean speed foul_candidates uses)."""
    from features import central_velocity
    pos = players[..., :2]
    vel = np.stack([central_velocity(pos[:, i], 3) for i in range(pos.shape[1])], axis=1)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_speed = np.nanmean(np.hypot(vel[..., 0], vel[..., 1]), axis=1)
    return detect_whistles(mean_speed, blocks)


def whistle_report():
    """On the IDSSE matches: how often a sharp collective slowdown is the start of a real stoppage."""
    import warnings
    from external import IDSSE_MATCHES, load_idsse
    from features import central_velocity
    rows = {}
    starts_total = 0
    for m in sorted(IDSSE_MATCHES):
        players, ball, team, labels, pitch, blocks = load_idsse(m)
        pos = players[..., :2]
        vel = np.stack([central_velocity(pos[:, i], 3) for i in range(pos.shape[1])], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            speed = np.nanmean(np.hypot(vel[..., 0], vel[..., 1]), axis=1)
        dead = labels == "D"
        starts = np.flatnonzero(dead[1:] & ~dead[:-1]) + 1
        length, width = pitch
        with np.errstate(invalid="ignore"):
            out = (ball[:, 0] < 0) | (ball[:, 0] > length) | (ball[:, 1] < 0) | (ball[:, 1] > width)
        starts_total += len(starts)
        for drop in (1.0, 1.5, 2.0, 2.5, 3.0):
            found = [c for c, _ in detect_whistles(speed, blocks, drop)]
            hit = [any(-3 * FPS <= s - c <= 3 * FPS for s in starts) for c in found]
            in_dead = [dead[c:c + 3 * FPS].mean() > 0.5 for c in found]      # most of the next 3 s is a stoppage
            got = sum(any(-3 * FPS <= s - c <= 3 * FPS for c in found) for s in starts)
            r = rows.setdefault(drop, [0, 0, 0, 0])
            r[0] += len(found); r[1] += sum(hit); r[2] += sum(in_dead); r[3] += got
    print("seven outside matches, {} real stoppage starts".format(starts_total))
    print("| mean speed falls by at least | slowdowns found | a stoppage starts within 3 s | the next 3 s are mostly a stoppage | share of all stoppage starts found |")
    print("| --- | --- | --- | --- | --- |")
    for drop, (n, hit, dead_n, got) in rows.items():
        print("| {:.1f} m/s | {} | {:.0f}% | {:.0f}% | {:.0f}% |".format(drop, n, 100 * hit / max(n, 1), 100 * dead_n / max(n, 1), 100 * got / starts_total))


def _whistle_self_test():
    n = 1000
    speed = np.full(n, 4.0)
    speed[400:] = 1.0                                           # a sprint stops dead at frame 400
    speed[700:] = 3.5
    speed[720:] = 2.4                                           # a small dip later: not a whistle
    w = detect_whistles(speed, [(1, 0, n - 1)])
    assert len(w) == 1 and 370 <= w[0][0] <= 400 and w[0][1] >= 2.0, w
    assert detect_whistles(np.full(n, 1.0), [(1, 0, n - 1)]) == []


def _self_test():
    _whistle_self_test()
    pos = np.zeros((3, 2, 2)); pos[:, 1, 0] = [5.0, 0.5, 30.0]
    d = closest_pair(pos, np.array([0, 1]), np.zeros((3, 2)))
    assert d[0] == 5.0 and d[1] == 0.5 and np.isnan(d[2]), d   # third frame: away player too far from the ball
    n = 1000
    speed = np.full(n, 2.5); speed[310:] = 1.0                  # everyone slows after frame 300
    pair = np.full(n, 4.0); pair[300] = 0.6; pair[302] = 0.8; pair[700] = 0.5
    out = np.zeros(n, bool)
    f = detect_fouls(speed, pair, out, [(1, 0, n - 1)])
    assert [c for c, _ in f] == [300] or [c for c, _ in f] == [302], f   # one per stoppage; 700 has no slowdown
    out[320] = True
    assert detect_fouls(speed, pair, out, [(1, 0, n - 1)]) == [], "a ball going out is not a foul"


if __name__ == "__main__":
    _self_test()
    print("foul_detect self-test passed")
    if len(sys.argv) > 1 and sys.argv[1] == "whistle":
        whistle_report()
