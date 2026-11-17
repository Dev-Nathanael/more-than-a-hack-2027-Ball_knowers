"""Solution 32's rule: a sudden collective slowdown, with the ball unseen, is a whistle.

    from slowdown import slowdown_stoppages
    labels, found = slowdown_stoppages(labels, speed_all, ball_seen, ball_fast, blocks)

An offside, a foul the ball never left play for, a stop for an injury: the ball does not go out, so none of
the chain's reasons for a dead ball fires, and the ball is often unseen for most of it (93% of the offside at
frames 12,851-12,988). What every one of them has is the players: all twenty of them go from running to walking
within a second or two and stay there until the restart.

The cue is the mean speed of every tracked player (m/s, tracking.team_features "speed_all"):

  - it falls by at least DROP_MPS between the WINDOW before a frame and the WINDOW after it, and the WINDOW after
    is at walking pace (WALK_MPS or less): that frame is a slowdown;
  - the slow stretch it starts is MIN_STOP or longer after the cuts below (the shortest stoppage we allow,
    solution 24);
  - the stretch is cut at every kick or throw (the ball faster than KICK_MPS, FAST_PAD either side): a kick
    ends a stoppage or starts one (the pass behind an offside, the restart), so only the parts between kicks
    count, and each must last MIN_STOP too;
  - the ball is seen on at most SEEN_MAX of each part. Where the ball is seen, the chain's own evidence has
    had its say, and a slowdown in live play (a lull, a keeper holding the ball) must not override it. A ball
    that is mostly unseen cannot contradict the players.

A slowdown alone is a weak cue (on seven labelled outside matches 54% of them start a stoppage and 38% are live
play), which is why it is only allowed to act where nothing else can speak, and why it needs the stretch to last.
The settings are checked on those matches by experiments/idsse_slowdown.py, never on the public 5 minutes.
"""

import numpy as np

# Params
FPS = 25
SMOOTH = FPS | 1             # frames of the moving mean over the speed (odd, about 1 s), to take out step noise
WINDOW = 2 * FPS             # the speed before and after a frame is compared over 2 s
DROP_MPS = 1.0               # a slowdown: the mean speed falls by this much ...
WALK_MPS = 1.6               # ... to walking pace or less (the slow regime of dead_evidence is 1.03 m/s, play 2.18)
SETTLE = 2 * FPS             # the stretch is the slow run that is still going on this long after the onset
GAP = FPS                    # a bump of up to 1 s above walking pace inside a stretch does not end it
MIN_STOP = 3 * FPS           # the shortest stretch (solution 24: 2.2% of true stoppages are shorter than 3 s)
SEEN_MAX = 0.5               # the ball is seen on at most this share of the stretch
FAST_PAD = FPS // 2          # the ball is "in play" this long after a frame faster than a kick or throw (it is still
                             # moving, or the pass behind a whistle has not stopped rolling) ...
FAST_LEAD = 2                # ... and this long before it (speed comes from position steps, so a kick shows a frame
                             # early). On seven labelled matches a half second of lead made the stretch end 0.52 s
                             # before the restart on most stoppages, so it is kept to these 2 frames.


def runs(mask):
    """[(first, end)] of each True run."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask).astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def smoothed(v, k=SMOOTH):
    """Moving mean over the finite values; NaN where fewer than half the window is finite."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    kernel = np.ones(k)
    total = np.convolve(np.where(ok, v, 0.0), kernel, "same")
    count = np.convolve(ok.astype(float), kernel, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count < k / 2] = np.nan
    return out


def window_means(v, w=WINDOW):
    """(before, after): the mean of v over the w frames before each frame and the w frames from it on.
    NaN where that window runs off the end or holds a NaN."""
    n = len(v)
    finite = np.isfinite(v)
    c = np.r_[0.0, np.cumsum(np.where(finite, v, 0.0))]
    bad = np.r_[0, np.cumsum(~finite)]
    before = np.full(n, np.nan)
    after = np.full(n, np.nan)
    if n > w:
        i = np.arange(w, n)
        before[i] = np.where(bad[i] - bad[i - w] == 0, (c[i] - c[i - w]) / w, np.nan)
        i = np.arange(0, n - w)
        after[i] = np.where(bad[i + w] - bad[i] == 0, (c[i + w] - c[i]) / w, np.nan)
    return before, after


def bridge(mask, gap=GAP):
    """mask with every False run of at most gap frames, between two True frames, set True."""
    out = np.asarray(mask).copy()
    for a, b in runs(~out):
        if a > 0 and b < len(out) and b - a <= gap:
            out[a:b] = True
    return out


def kicked_mask(fast):
    """True on every frame within FAST_LEAD before or FAST_PAD after a frame where the ball is faster than a kick."""
    fast = np.asarray(fast, float)
    length = FAST_LEAD + FAST_PAD + 1
    reach = np.convolve(fast, np.ones(length))          # reach[i] counts fast frames in [i - length + 1, i]
    return reach[FAST_LEAD:FAST_LEAD + len(fast)] > 0   # frame t: fast within [t - FAST_PAD, t + FAST_LEAD]


def slowdown_onsets(v, drop=DROP_MPS, walk=WALK_MPS, w=WINDOW):
    """First frame of each run of frames where the speed v (already smoothed) falls by drop to walk or less."""
    before, after = window_means(v, w)
    with np.errstate(invalid="ignore"):
        hit = (before - after >= drop) & (after <= walk)
    return [a for a, _ in runs(hit)]


def slowdown_stoppages(labels, speed, seen, fast, blocks, drop=DROP_MPS, walk=WALK_MPS, min_stop=MIN_STOP,
                       seen_max=SEEN_MAX):
    """(labels, [(first, end)] of the stretches set dead).

    labels (T,) of H / A / D / X; speed (T,) mean speed of every tracked player, m/s; seen (T,) bool, the ball has
    a position; fast (T,) bool, the ball moves faster than a kick or a throw; blocks as games.half_video_blocks.
    Only frames in a graded block are touched. A stretch already entirely D changes nothing and is not reported.
    """
    out = labels.copy()
    found = []
    for _, first, last in blocks:
        sl = slice(first, last + 1)
        v = smoothed(speed[sl])
        n = len(v)
        with np.errstate(invalid="ignore"):
            below = bridge(v <= walk)                 # NaN compares False: an unseen team is never "slow"
        start = np.zeros(n, int)
        end = np.zeros(n, int)
        for a, b in runs(below):
            start[a:b] = a
            end[a:b] = b
        ball_seen = seen[sl]
        kicked = kicked_mask(fast[sl])
        done = 0                                      # onsets inside a stretch already handled are skipped
        for t in slowdown_onsets(v, drop, walk):
            p = t + SETTLE
            if t < done or p >= n or not below[p]:
                continue
            a, b = start[p], end[p]
            done = b
            for x, y in runs(~kicked[a:b]):           # a kick or throw ends or starts a stoppage: cut there
                x, y = a + x, a + y
                if y - x < min_stop or ball_seen[x:y].mean() > seen_max:
                    continue
                if (out[first + x:first + y] != "D").any():
                    out[first + x:first + y] = "D"
                    found.append((first + x, first + y))
    return out, found


def _self_test():
    n = 1000
    v = np.full(n, 5.0)
    v[400:700] = 0.5                                  # everyone walks from 400 to 700
    lab = np.full(n, "H")
    blocks = [(1, 0, n - 1)]
    none, every = np.zeros(n, bool), np.ones(n, bool)

    out, found = slowdown_stoppages(lab, v, none, none, blocks)
    assert len(found) == 1, found
    a, b = found[0]
    assert 395 <= a <= 420 and 685 <= b <= 700, found
    assert (out[a:b] == "D").all() and (out[:390] == "H").all() and (out[705:] == "H").all(), "only the stretch is dead"

    assert slowdown_stoppages(lab, v, every, none, blocks)[1] == [], "the ball is seen throughout: the chain decides"
    fast = none.copy()
    fast[420] = True                                  # the tail of the pass that led to the whistle
    found = slowdown_stoppages(lab, v, none, fast, blocks)[1]
    assert len(found) == 1 and found[0][0] == 420 + FAST_PAD + 1 and found[0][1] <= 700, found
    fast = none.copy()
    fast[690] = True                                  # the restart kick: the stretch runs up to it, not half a second short
    found = slowdown_stoppages(lab, v, none, fast, blocks)[1]
    assert len(found) == 1 and found[0][0] <= 410 and found[0][1] == 690 - FAST_LEAD, found
    assert kicked_mask(np.eye(1, 40, 20, dtype=bool)[0]).nonzero()[0].tolist() == list(range(20 - FAST_LEAD, 20 + FAST_PAD + 1))
    fast = none.copy()
    fast[500] = True                                  # a kick in the middle: both halves are still 3 s or more
    assert len(slowdown_stoppages(lab, v, none, fast, blocks)[1]) == 2
    short = np.full(n, 5.0)
    short[400:520] = 0.5
    fast = none.copy()
    fast[460] = True                                  # a kick in the middle of a 4.8 s stop: neither half is 3 s
    assert slowdown_stoppages(lab, short, none, fast, blocks)[1] == []
    short = np.full(n, 5.0)
    short[400:450] = 0.5
    assert slowdown_stoppages(lab, short, none, none, blocks)[1] == [], "under 3 s is not a stoppage"
    calm = np.full(n, 0.5)
    assert slowdown_stoppages(lab, calm, none, none, blocks)[1] == [], "no slowdown, no cue: slow all along is the chain's"
    gentle = np.full(n, 6.0)
    gentle[400:700] = 2.5
    assert slowdown_stoppages(lab, gentle, none, none, blocks)[1] == [], "a big drop that stays above walking pace"
    mild = np.full(n, 1.8)
    mild[400:700] = 1.0
    assert slowdown_stoppages(lab, mild, none, none, blocks)[1] == [], "a fall of 0.8 m/s is under the 1.0 needed"
    dead = lab.copy()
    dead[380:720] = "D"
    assert slowdown_stoppages(dead, v, none, none, blocks)[1] == [], "already dead: nothing to do"
    nan = v.copy()
    nan[395:705] = np.nan
    assert slowdown_stoppages(lab, nan, none, none, blocks)[1] == [], "no tracking is not slow players"
    assert bridge(np.array([1, 1, 0, 1, 0, 0, 0, 1], bool), gap=1).tolist() == [True, True, True, True, False, False, False, True]


if __name__ == "__main__":
    _self_test()
    print("slowdown self-test passed")
