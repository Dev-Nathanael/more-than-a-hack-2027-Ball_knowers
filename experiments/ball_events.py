"""Ball events from the shape of the ball's path: where was the ball played, and by which team?

    from ball_events import path_events, last_touch_team

The provider's ball track is smoothed, so a kick is not a spike in acceleration (features.touches finds
under one a minute). But between two touches a ball travels in a straight line and only slows down. So:

1. corners:   Ramer-Douglas-Peucker on each stretch of seen ball; a vertex is where the path bends by more
              than the tracking noise, however gently the smoothing rounds it.
2. speed-ups: the ball is faster than it was shortly before; a free ball never speeds up.
3. stops:     the ball starts moving after standing still (a placed ball being kicked).
An event belongs to the nearest player if one is within reach (wider than a foot: the homography puts a
ball in the air a few metres from where it is). Events with nobody near are bounces or tracking jumps.

All thresholds come from physics and tracking noise, none from the ground truth.
"""

import numpy as np

# Params
FPS = 25
RDP_M = 1.0              # path bends by more than this: above the tracking wobble (about 0.2-0.5 m)
SPEEDUP_MPS = 3.0        # a free ball only slows; a gain this large over 0.4 s is a kick
SPEEDUP_LAG = 5          # frames either side of the candidate over which speed is compared
STILL_MPS = 0.5          # slower than this the ball is standing (noise alone gives about 0.2-0.4 m/s)
REACH_M = 3.0            # the player who played it is within this (2 m reach + projection error)
REACH_HALF = 3           # frames either side: take each player's closest approach
MERGE = 8                # events closer than this (0.3 s) are one touch


def _rdp(xy, tol):
    """Indices of the vertices of the simplified path (first and last always kept)."""
    keep = np.zeros(len(xy), bool)
    keep[[0, -1]] = True
    stack = [(0, len(xy) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = xy[b] - xy[a]
        rel = xy[a + 1:b] - xy[a]
        norm = np.hypot(*seg)
        d = np.hypot(*rel.T) if norm < 1e-9 else np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / norm
        i = int(np.argmax(d))
        if d[i] > tol:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return np.flatnonzero(keep)


def _seen_runs(seen):
    edges = np.flatnonzero(np.diff(np.r_[0, seen.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def path_events(ball, pos, team):
    """(frames, team) of every ball event: team 0 home, 1 away, -1 nobody within reach.

    ball (T, 2), pos (T, S, 2) metres with NaN where unseen; team (S,) 0/1.
    """
    seen = np.isfinite(ball[:, 0])
    speed = np.full(len(ball), np.nan)
    speed[2:-2] = np.hypot(*(ball[4:] - ball[:-4]).T) * FPS / 4      # over 4 frames: steadier than 2
    cand = []
    for a, b in _seen_runs(seen):
        if b - a < 3:
            continue
        cand += list(a + _rdp(ball[a:b], RDP_M)[1:-1])
        s = speed[a:b]
        k = SPEEDUP_LAG
        if b - a > 2 * k + 2:
            gain = s[2 * k:] - s[:-2 * k]                          # speed k after minus k before, at index k..
            mid = np.arange(k, len(s) - k)
            with np.errstate(invalid="ignore"):
                up = gain > SPEEDUP_MPS
            for t in mid[up]:                                       # keep local maxima of the gain
                g = gain[max(t - k - k, 0):t - k + k + 1]
                if gain[t - k] >= np.nanmax(g):
                    cand.append(a + t)
    cand = np.unique(np.array(cand, dtype=int))
    if len(cand) == 0:
        return cand, np.zeros(0, int)
    groups = np.split(cand, np.flatnonzero(np.diff(cand) > MERGE) + 1)   # merge neighbours into one touch
    frames = np.array([g[len(g) // 2] for g in groups])
    who = np.full(len(frames), -1)
    for i, t in enumerate(frames):
        lo, hi = max(t - REACH_HALF, 0), min(t + REACH_HALF + 1, len(ball))
        d = np.hypot(pos[lo:hi, :, 0] - ball[lo:hi, None, 0], pos[lo:hi, :, 1] - ball[lo:hi, None, 1])
        d = np.where(np.isfinite(d), d, np.inf).min(axis=0)
        j = int(np.argmin(d))
        if d[j] <= REACH_M:
            who[i] = team[j]
    return frames, who


def last_touch_team(n, frames, who, seen=None, max_age=None):
    """(T,) team of the latest attributed event, -1 before the first. With seen / max_age the answer
    expires (-1) once the ball has been unseen since the event or the event is older than max_age frames:
    after a blackout nobody knows who played it last."""
    out = np.full(n, -1)
    ok = who >= 0
    out[frames[ok]] = who[ok]
    idx = np.maximum.accumulate(np.where(out >= 0, np.arange(n), -1))
    last = np.where(idx >= 0, out[np.maximum(idx, 0)], -1)
    if max_age is not None:
        last = np.where(np.arange(n) - idx > max_age, -1, last)
    if seen is not None:
        unseen_at = np.maximum.accumulate(np.where(~seen, np.arange(n), -1))
        last = np.where(unseen_at > idx, -1, last)
    return last


def heading_out(ball, blocks, pitch, horizon=2 * FPS, cap=15 * FPS):
    """(T,) True on unseen frames after the ball was last seen flying towards a line it would cross within
    `horizon` frames: it went out of play (and out of the camera's sight). Lasts until the ball is seen
    again, at most `cap` frames (a restart follows within about 15 s)."""
    length, width = pitch
    seen = np.isfinite(ball[:, 0])
    out = np.zeros(len(ball), bool)
    for _, first, last in blocks:
        starts = first + np.flatnonzero(seen[first:last] & ~seen[first + 1:last + 1]) + 1   # first unseen frame
        for t in starts:
            if t - 4 < first or not seen[t - 4:t].all():
                continue
            v = (ball[t - 1] - ball[t - 4]) / 3                                             # metres per frame
            steps = np.arange(1, horizon + 1)[:, None]
            path = ball[t - 1] + steps * v
            gone = (path[:, 0] < 0) | (path[:, 0] > length) | (path[:, 1] < 0) | (path[:, 1] > width)
            if not gone.any():
                continue
            a = t + int(np.argmax(gone))
            back = np.flatnonzero(seen[t:min(a + cap, last + 1)])
            b = t + back[0] if len(back) else min(a + cap, last + 1)
            out[a:b] = True
    return out


def event_distances(frames, ball, pos, team):
    """(nearest home, nearest away) distance to the ball at each event: closest approach over +-REACH_HALF."""
    d_home, d_away = np.full(len(frames), np.inf), np.full(len(frames), np.inf)
    for i, t in enumerate(frames):
        lo, hi = max(t - REACH_HALF, 0), min(t + REACH_HALF + 1, len(ball))
        d = np.hypot(pos[lo:hi, :, 0] - ball[lo:hi, None, 0], pos[lo:hi, :, 1] - ball[lo:hi, None, 1])
        d = np.where(np.isfinite(d), d, np.inf).min(axis=0)
        d_home[i], d_away[i] = d[team == 0].min(initial=np.inf), d[team == 1].min(initial=np.inf)
    return d_home, d_away


def drop_duel_touches(frames, who, d_home, d_away):
    """Events without the lone touches made in a duel.

    One touch by a team between two touches of the other team, with an opponent also within reach, is a
    challenge (a block, a toe on the ball, a header under pressure), not a possession: the ball goes
    straight back. A second touch by the same team, or a touch with no opponent near, is possession."""
    idx = np.flatnonzero(who >= 0)
    w = who[idx]
    single = np.ones(len(w), bool)
    single[1:] &= w[1:] != w[:-1]
    single[:-1] &= w[:-1] != w[1:]
    opp = np.where(who == 0, d_away, d_home)
    keep = np.ones(len(frames), bool)
    keep[idx[single & (opp[idx] <= REACH_M)]] = False
    return frames[keep], who[keep]


TOUCH_FEATURES = ("own distance", "opponent distance", "ball speed before", "ball speed after", "turn",
                  "next touch is a team-mate's", "seconds to next touch", "seconds since last touch")


def touch_features(frames, who, ball, pos, team):
    """(index of each change candidate, X (n, 8)): what an opponent's touch looks like.

    A change candidate is an attributed event by the team that did NOT play the ball last. Whether it starts
    a possession or is only a block or a deflection shows in: how close each team's nearest player is, how
    fast the ball was and is afterwards, how sharply it turned, and whether the same team plays it again."""
    idx = np.flatnonzero(who >= 0)
    f, w = frames[idx], who[idx]
    d_home, d_away = event_distances(f, ball, pos, team)
    n = len(ball)
    X, cand = [], []
    for i in range(1, len(f)):
        if w[i] == w[i - 1]:
            continue
        t = f[i]
        before, after = ball[max(t - 7, 0):t - 1], ball[t + 2:min(t + 8, n)]
        vb = (before[-1] - before[0]) / max(len(before) - 1, 1) * FPS if len(before) > 1 else np.full(2, np.nan)
        va = (after[-1] - after[0]) / max(len(after) - 1, 1) * FPS if len(after) > 1 else np.full(2, np.nan)
        sb, sa = np.hypot(*vb), np.hypot(*va)
        with np.errstate(invalid="ignore", divide="ignore"):
            turn = np.degrees(np.arccos(np.clip(np.dot(vb, va) / (sb * sa), -1, 1))) if sb > 0.5 and sa > 0.5 else 0.0
        nxt = i + 1 < len(f) and f[i + 1] - t <= 3 * FPS
        X.append([d_home[i] if w[i] == 0 else d_away[i], d_away[i] if w[i] == 0 else d_home[i], sb, sa, turn,
                  float(nxt and w[i + 1] == w[i]), min((f[i + 1] - t) / FPS, 3.0) if i + 1 < len(f) else 3.0,
                  min((t - f[i - 1]) / FPS, 5.0)])
        cand.append(idx[i])
    X = np.array(X, float).reshape(-1, len(TOUCH_FEATURES))
    X[:, :2] = np.minimum(X[:, :2], 10.0)                    # nobody within 10 m is all the same
    return np.array(cand, int), np.nan_to_num(X, nan=0.0)


# Fitted by idsse_touches.py on seven labelled outside matches (4,328 opponent touches, as delivered and with
# the ball hidden like ours). On a left-out match it is right about 86.4% of them (AUC 0.832); counting every
# touch is right about 81.3%, the duel rule about 66.8%, "two plays" (solution 22) about 61.4%.
TOUCH_MODEL = {"mean": [0.7136, 3.501, 6.5225, 10.1038, 54.4386, 0.5566, 1.7897, 2.287],
               "scale": [0.6331, 2.9148, 6.0927, 7.6518, 52.8558, 0.4968, 0.847, 1.5269],
               "coef": [-0.5289, 0.4661, -0.5746, 0.0545, 0.858, 0.2951, -0.1874, 0.3744], "intercept": 2.0504}


def touch_probability(X):
    """P(this opponent touch starts a possession) for touch_features rows."""
    m = TOUCH_MODEL
    z = ((X - np.array(m["mean"])) / np.array(m["scale"])) @ np.array(m["coef"]) + m["intercept"]
    return 1 / (1 + np.exp(-z))


def drop_unlikely_touches(frames, who, ball, pos, team, threshold=0.5):
    """Events without the opponent touches the learned model does not believe (blocks, deflections, a pass
    bending next to an opponent). Repeated until nothing more goes: dropping a touch changes what follows it."""
    for _ in range(10):
        cand, X = touch_features(frames, who, ball, pos, team)
        if len(cand) == 0:
            break
        bad = cand[touch_probability(X) < threshold]
        if len(bad) == 0:
            break
        keep = np.ones(len(frames), bool)
        keep[bad] = False
        frames, who = frames[keep], who[keep]
    return frames, who


SURE_P = 0.90            # = pipelines.SURE_P: the first pass is sure of a frame at 90%
SURE_HALF = 12           # frames (0.5 s) either side of the touch over which the first pass must stay sure


def drop_contradicted_touches(frames, who, post, sure=SURE_P, half=SURE_HALF):
    """Second cleaning, done after the match has been read once: (frames, who) without the touches the first
    pass contradicts.

    post (T, 3) is the first pass's P(home), P(away), P(dead). A touch credited to one team while the first
    pass is sure, for half a second either side, that the OTHER team has the ball is a data error, not a
    change of possession: the ball passed beside an opponent, or the track beside it carries the wrong team.
    A real change makes the first pass doubt, at least for a moment, so those touches stay."""
    other = np.nan_to_num(post[:, [1, 0]], nan=0.0)        # column 0: P(away), the "other team" for a home touch
    keep = np.ones(len(frames), bool)
    for i, (t, w) in enumerate(zip(frames, who)):
        if w >= 0 and other[max(t - half, 0):t + half + 1, w].min() >= sure:
            keep[i] = False
    return frames[keep], who[keep]


OUT_BACK = 10 * FPS       # the touch that put the ball out is at most this long before our stoppage starts
TAG_REACH = 10 * FPS      # = unified_hmm.REASON_TAG: a set-piece tag this close belongs to the stoppage
RESTART_LOOK = 2 * FPS    # where the ball is in the first 2 s of play says what kind of restart it was


def restart_back_check(labels, frames, who, blocks, beyond=None, out_tags=(), boundary=None, max_back=OUT_BACK):
    """The restart tells who played the ball out: (labels, possessions added).

    A throw-in, corner or goal kick goes to the team that did NOT touch the ball last, and after a goal the
    team that conceded kicks off. So when a stoppage is followed by team R, the last play before it was by the
    other team O. If the labels say R had the ball right up to the stoppage, look back (at most max_back):
      - beyond (T,) bool marks frames with the ball over a line, by any margin. If O touched the ball and it
        then went over a line with no touch by R in between, O's touch put it out: the frames from the touch to
        the line are O's and everything from the line on is dead. What "R touches" follow are the ball being
        fetched; the stoppage had simply been noticed late.
      - otherwise, if the latest touch before the stoppage is O's, the frames from it to the stoppage are O's.
      - otherwise R "touched" the ball after O's last touch and still R restarts. If the restart is a throw-in,
        corner or goal kick, R cannot have played it last, or O would restart. That kind of restart is known
        from where it is taken (boundary (T,) bool: the ball at a line or in a goal area, within RESTART_LOOK
        of the restart; in the match 21 of 25 tagged throw-ins restart within 3 m of a line, 6 of 47 untagged
        stoppages do) or from a Smart Tagging tag (out_tags: frames of such tags, within TAG_REACH). So those touches are not play: O has the ball from
        its touch to the first of them, and it is dead from there. After a foul this does not hold (the fouled
        team touched last and restarts), so without such a tag nothing changes.
    Independent facts must agree (the touch, the restart, and the line where seen); with only one, nothing
    changes. frames, who: every attributed ball event, before the duel rule."""
    out = labels.copy()
    added = 0
    ok = who >= 0
    f, w = frames[ok], who[ok]
    for _, first, last in blocks:
        seg = out[first:last + 1]
        cuts = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg)]
        for a, b in zip(cuts[1:-2], cuts[2:-1]):
            if seg[a] != "D" or seg[b] not in "HA" or seg[a - 1] != seg[b]:
                continue                                  # only: R ... stoppage ... R, the contradiction
            restart = 0 if seg[b] == "H" else 1
            other = "A" if restart == 0 else "H"
            stop, lo = first + a, max(first + a - max_back, first)
            near = np.flatnonzero((f >= lo) & (f < stop))
            theirs = near[w[near] != restart]             # touches by O in the window
            if len(theirs) == 0:
                continue
            i = theirs[-1]
            crossed = np.flatnonzero(beyond[f[i]:stop]) if beyond is not None else []
            if len(crossed):
                line = f[i] + crossed[0]
                if not np.any((f > f[i]) & (f < line)):   # nobody played it between O's touch and the line
                    seg[f[i] - first:line - first] = other
                    seg[line - first:a] = "D"
                    added += 1
                    continue
            if i == near[-1]:                             # O's touch is the latest before the stoppage
                seg[f[i] - first:a] = other
            elif (any(first + a - TAG_REACH <= t <= first + b + TAG_REACH for t in out_tags)
                  or (boundary is not None and boundary[first + b:first + b + RESTART_LOOK].any())):
                nxt = f[near[near > i][0]]                # R "touched" it after O did, yet R restarts from a ball out:
                seg[f[i] - first:nxt - first] = other     # those touches cannot be play (the ball being fetched, or a
                seg[nxt - first:a] = "D"                  # second ball), so play ended by the first of them
            else:
                continue
            added += 1
    return out, added


PASS_MAX = 4 * FPS          # a pass, a long ball included, is in flight for at most about this long
PASS_PAD = 3                # frames left out at each end: the kick and the first touch bend the path
PASS_STRAIGHT = 0.95        # straight-line distance over path length: 1 is a ruler, a deflection drops it
PASS_MPS = 5.0              # = unified_hmm.KICK_MPS: a ball whose typical speed in flight is this was kicked


PASS_RECENT = FPS            # a kick within this of the same team's own touch is that team's second play


def straight_flight(ball, a, z):
    """True if nothing played the ball between frames a and z: it is seen on every frame, was kicked (median
    speed PASS_MPS or more), travels in a straight line, and never speeds up again (a free ball only slows;
    a gain of SPEEDUP_MPS over SPEEDUP_LAG frames is a new kick). A rolling pass may slow to walking pace
    before it arrives, so no minimum speed is asked of every frame."""
    a, z = a + PASS_PAD, z - PASS_PAD
    if z - a < 3:
        return False
    p = ball[a:z + 1]
    if not np.isfinite(p[:, 0]).all():
        return False
    steps = np.hypot(*np.diff(p, axis=0).T)
    length = steps.sum()
    speed = steps * FPS
    gain = (speed[SPEEDUP_LAG:] - speed[:-SPEEDUP_LAG]).max() if len(speed) > SPEEDUP_LAG else 0.0
    return bool(length > 0 and np.hypot(*(p[-1] - p[0])) / length >= PASS_STRAIGHT
                and np.median(speed) >= PASS_MPS and gain < SPEEDUP_MPS)


def completed_passes(frames, who, ball, d_home=None, d_away=None):
    """(frames, who, protected) with completed passes recognised (our rule, 2026-11-18).

    A pass that is not intercepted belongs to the passing team from the kick to the reception. Nothing
    touched a ball that is seen throughout, stays faster than PASS_MPS and travels in a straight line.
      1. Whose play was it? The ball only changes velocity when somebody plays it, and that event is credited
         to the nearest player; with two players at the ball that can be the wrong one. So, looking back from
         the event: if a team played the ball under PASS_RECENT ago and still has a player within reach
         (d_home / d_away, from event_distances), the event is that team's next play. The man who has just
         received the ball is the one who plays it next; an opponent who really wins it will play it again,
         and those plays are his (one touch is not a possession, two are).
      2. Two consecutive plays by the same team joined by a straight flight are a completed pass. Both
         ends are protected (the kick is not a "duel touch" just because an opponent stood near), and any
         event of the other team in between is removed: the ball passed near him, he did not play it.
    A header or clearance that flies straight to the other team stays a change of team: only a play made
    within reach of a team that had just played the ball is re-credited. protected: bool per returned event."""
    who = who.copy()
    n = len(frames)
    if d_home is not None:
        own = lambda k, t: (d_home if t == 0 else d_away)[k]  # noqa: E731
        for k in range(1, n):
            t = who[k - 1]
            if t < 0 or who[k] < 0 or who[k] == t or frames[k] - frames[k - 1] > PASS_RECENT or not own(k, t) <= REACH_M:
                continue
            who[k] = t
    keep = np.ones(n, bool)
    protected = np.zeros(n, bool)
    for i in range(n):
        if who[i] < 0:
            continue
        later = np.flatnonzero((who[i + 1:] == who[i]) & (frames[i + 1:] - frames[i] <= PASS_MAX))
        if not len(later):
            continue
        j = i + 1 + later[0]
        if not straight_flight(ball, frames[i], frames[j]):
            continue
        protected[i] = protected[j] = True
        between = np.arange(i + 1, j)
        keep[between[who[between] != who[i]]] = False
    return frames[keep], who[keep], protected[keep]


def drop_doubtful_touches(frames, who, ball, pos, doubtful):
    """Ball events without the ones played by a player the tracking is unsure of. doubtful: (N, S) True where
    a player is seen by one camera only, or the two cameras put him on different teams. The player who played
    the ball is taken as the one nearest it at that frame. Such a touch is evidence for neither team: his
    position is unconfirmed, or his team is. Dropping it leaves the ball with whoever played it before: the last
    sure touch stands until the next sure one. Returns (frames, who)."""
    if not len(frames):
        return frames, who
    with np.errstate(invalid="ignore"):
        d = np.hypot(pos[frames, :, 0] - ball[frames, None, 0], pos[frames, :, 1] - ball[frames, None, 1])
    nearest = np.argmin(np.where(np.isfinite(d), d, np.inf), axis=1)
    keep = ~doubtful[frames, nearest]
    return frames[keep], who[keep]


def event_possession(labels, ball, pos, team, max_age=5 * FPS, learned=False, post=None, doubtful=None, passes=False):
    """labels with every live frame given to the team that last played the ball (duel touches dropped, or with
    learned=True the touches the learned model does not believe).

    Stoppages stay as they are. Where the last touch is unknown (ball unseen since, or older than max_age:
    no pass is in flight for 5 s) the label stays as it was. Returns (labels, info)."""
    seen = np.isfinite(ball[:, 0])
    frames, who = path_events(ball, pos, team)
    n_events = len(frames)
    if doubtful is not None:
        frames, who = drop_doubtful_touches(frames, who, ball, pos, doubtful)
    protected = np.zeros(len(frames), bool)
    if passes:                  # 39: a completed pass is the passing team's from the kick to the reception
        frames, who, protected = completed_passes(frames, who, ball, *event_distances(frames, ball, pos, team))
    d_home, d_away = event_distances(frames, ball, pos, team)
    f2, w2 = drop_unlikely_touches(frames, who, ball, pos, team) if learned else drop_duel_touches(frames, who, d_home, d_away)
    if protected.any():         # the two ends of a completed pass are never "duel touches"
        back = protected & ~np.isin(frames, f2)
        order = np.argsort(np.r_[f2, frames[back]], kind="stable")
        f2, w2 = np.r_[f2, frames[back]][order], np.r_[w2, who[back]][order]
    contradicted = 0
    if post is not None:
        n_before = len(f2)
        f2, w2 = drop_contradicted_touches(f2, w2, post)
        contradicted = n_before - len(f2)
    last = last_touch_team(len(ball), f2, w2, seen, max_age)
    out = labels.copy()
    live = ((labels == "H") | (labels == "A")) & (last >= 0)
    out[live] = np.where(last[live] == 0, "H", "A")
    info = {"events": len(frames), "attributed": int((who >= 0).sum()), "duel_touches": len(frames) - len(f2),
            "relabelled": int((out != labels).sum()), "frames": f2, "who": w2, "contradicted": contradicted,
            "raw_frames": frames, "raw_who": who, "doubtful_touches": n_events - len(frames),
            "completed_passes": int(protected.sum())}
    return out, info


def merge_untouched_spells(labels, frames, who, seen, blocks, max_len=3 * FPS):
    """A short spell (under max_len) with the ball seen throughout and no ball event of its own is not a
    possession: nobody of that team played the ball. It goes to its neighbour: the other team's spell
    beside it (before it if both sides are), or the stoppage if it sits between two stoppages."""
    out = labels.copy()
    merged = 0
    for _, first, last in blocks:
        seg = out[first:last + 1]
        edges = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg)]
        for a, b in zip(edges[1:-2], edges[2:-1]):
            me, before, after = seg[a], seg[a - 1], seg[b]
            if me not in "HA" or b - a >= max_len or not seen[first + a:first + b].all():
                continue
            own = 0 if me == "H" else 1
            if ((frames >= first + a) & (frames < first + b) & (who == own)).any():
                continue
            seg[a:b] = before if before in "HA" else (after if after in "HA" else "D")
            merged += 1
    return out, merged


def merge_lone_touch_spells(labels, frames, who, seen, blocks, max_len=3 * FPS, plays=2):
    """One touch is not a possession; two plays in a row by the same team are. A header won from a high
    ball, or a block, that goes straight back to the opponents is a contest inside THEIR possession, while
    a header followed by a team-mate's touch is a completed pass, however soon the ball is lost again.
    So a short spell (under max_len) between two spells of the other team, with the ball seen throughout
    and fewer than `plays` ball events of its own, goes back to that team. Spells next to a stoppage are
    left alone (a clearance into touch ends the possession for real), and so are spells with the ball
    missing, where a second touch could have gone unseen."""
    out = labels.copy()
    merged = 0
    for _, first, last in blocks:
        seg = out[first:last + 1]
        edges = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg)]
        for a, b in zip(edges[1:-2], edges[2:-1]):
            me, before, after = seg[a], seg[a - 1], seg[b]
            if me not in "HA" or before not in "HA" or before != after or b - a >= max_len:
                continue
            if not seen[first + a:first + b].all():
                continue
            own = 0 if me == "H" else 1
            if ((frames >= first + a) & (frames < first + b) & (who == own)).sum() >= plays:
                continue
            seg[a:b] = before
            merged += 1
    return out, merged


WHISTLE_NEAR = 3 * FPS       # a whistle belongs to a stoppage that starts within this of it


def merge_spells_before_whistle(labels, frames, who, whistles, blocks, max_len=3 * FPS, plays=2, protect=None):
    """The two-plays rule at a whistle. A team that gets the ball under max_len before play is stopped by a
    whistle, and makes fewer than `plays` plays with it, has not had a possession: the whistle was for
    something in the other team's possession (an offside flagged late, a foul as the ball was won). So a short
    spell between the other team's spell and a stoppage that starts at a whistle goes back to that team.
    whistles: frames of likely whistles (fouls and unseen whistles), not balls going out: a last touch that
    puts the ball out IS a possession (restart_back_check) and is left alone. Unlike merge_lone_touch_spells
    the ball need not be seen throughout: at a whistle it often is not. protect: (N,) True on frames another
    rule set on purpose (the restart back check); a spell touching them is never merged.
    Returns (labels, spells merged)."""
    out = labels.copy()
    merged = 0
    whistles = np.asarray(sorted(whistles), int)
    for _, first, last in blocks:
        seg = out[first:last + 1]
        edges = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg)]
        for a, b in zip(edges[1:-2], edges[2:-1]):
            me, before, after = seg[a], seg[a - 1], seg[b]
            if me not in "HA" or before not in "HA" or before == me or after != "D" or b - a >= max_len:
                continue
            if not len(whistles) or np.abs(whistles - (first + b)).min() > WHISTLE_NEAR:
                continue
            if protect is not None and protect[first + a:first + b].any():
                continue
            own = 0 if me == "H" else 1
            if ((frames >= first + a) & (frames < first + b) & (who == own)).sum() >= plays:
                continue
            seg[a:b] = before
            merged += 1
    return out, merged


def _before_whistle_self_test():
    lab = np.array(list("A" * 100 + "H" * 50 + "D" * 100 + "H" * 100 + "A" * 40 + "D" * 60))
    blocks = [(1, 0, len(lab) - 1)]
    none = np.zeros(0, int)
    got, n = merge_spells_before_whistle(lab, none, none, [140], blocks)
    assert n == 1 and "".join(got[:150]) == "A" * 150, "a short spell before a whistle goes back"
    assert merge_spells_before_whistle(lab, none, none, [], blocks)[1] == 0, "no whistle (the ball went out): left alone"
    two = np.array([110, 130])
    assert merge_spells_before_whistle(lab, two, np.array([0, 0]), [140], blocks)[1] == 0, "two plays are a possession"
    keep = np.zeros(len(lab), bool)
    keep[120] = True
    assert merge_spells_before_whistle(lab, none, none, [140], blocks, protect=keep)[1] == 0, "a protected spell stays"
    long = np.array(list("A" * 100 + "H" * 90 + "D" * 100))
    assert merge_spells_before_whistle(long, none, none, [185], [(1, 0, 289)])[1] == 0, "3 s or more is a possession"


def _lone_touch_self_test():
    lab = np.array(list("H" * 40 + "A" * 30 + "H" * 40))
    seen, blocks = np.ones(110, bool), [(1, 0, 109)]
    m, k = merge_lone_touch_spells(lab, np.array([40]), np.array([1]), seen, blocks)
    assert k == 1 and (m == "H").all(), "a lone header goes back to the team that had the ball"
    m, k = merge_lone_touch_spells(lab, np.array([40, 55]), np.array([1, 1]), seen, blocks)
    assert k == 0, "a header and a team-mate's touch is a possession"
    hidden = seen.copy(); hidden[50:55] = False
    assert merge_lone_touch_spells(lab, np.array([40]), np.array([1]), hidden, blocks)[1] == 0, "a touch may be unseen"
    out = np.array(list("H" * 40 + "A" * 30 + "D" * 40))
    assert merge_lone_touch_spells(out, np.array([40]), np.array([1]), seen, blocks)[1] == 0, "a clearance into touch stays"


def _touch_features_self_test():
    t = np.arange(100)
    ball = np.column_stack([np.minimum(t, 50) * 0.4, np.maximum(t - 50, 0) * 0.4])        # turns 90 degrees at 50
    pos = np.full((100, 2, 2), np.nan); pos[:, 0] = [0.0, 0.5]; pos[:, 1] = [20.5, 0.0]
    cand, X = touch_features(np.array([0, 50]), np.array([0, 1]), ball, pos, np.array([0, 1]))
    assert list(cand) == [1] and X.shape == (1, 8)
    assert abs(X[0, 4] - 90) < 1 and abs(X[0, 2] - 10) < 0.1 and X[0, 0] < 1 and X[0, 1] == 10.0, X


def _touch_model_self_test():
    # a clean take: toucher on the ball, nobody near, sharp turn, a team-mate plays it next
    take = np.array([[0.3, 8.0, 5.0, 12.0, 120.0, 1.0, 1.0, 3.0]])
    # a deflection: opponent as close as the toucher, fast ball barely turned, the other team plays it next
    deflection = np.array([[2.5, 0.5, 20.0, 15.0, 5.0, 0.0, 0.4, 0.5]])
    assert touch_probability(take)[0] > 0.9 and touch_probability(deflection)[0] < 0.5, (touch_probability(take), touch_probability(deflection))


def _contradicted_self_test():
    post = np.zeros((100, 3)); post[:, 0] = 0.95; post[:, 1] = 0.03          # the first pass: home, surely
    post[60:80, 0], post[60:80, 1] = 0.5, 0.45                                # ... except a doubtful second
    f, w = drop_contradicted_touches(np.array([20, 30, 70]), np.array([1, 0, 1]), post)
    assert list(f) == [30, 70] and list(w) == [0, 1], (f, w)                  # the away touch at 20 is a data error


def _out_touch_self_test():
    lab = np.array(list("A" * 100 + "D" * 100 + "A" * 100))             # away, stoppage, away restarts
    blocks = [(1, 0, 299)]
    out, k = restart_back_check(lab, np.array([20, 70]), np.array([1, 0]), blocks)   # home touched it last, at 70
    assert k == 1 and (out[70:100] == "H").all() and (out[:70] == "A").all() and (out[100:200] == "D").all()
    assert restart_back_check(lab, np.array([20, 70]), np.array([0, 1]), blocks)[1] == 0, "away touched last: no proof"
    home_restart = np.array(list("A" * 100 + "D" * 100 + "H" * 100))
    assert restart_back_check(home_restart, np.array([20, 70]), np.array([1, 0]), blocks)[1] == 0, "no contradiction"
    # the ball crossed the line at 60 after home's touch at 50; "away touches" at 75 and 90 are it being fetched
    beyond = np.zeros(300, bool); beyond[60:64] = True
    out, k = restart_back_check(lab, np.array([20, 50, 75, 90]), np.array([1, 0, 1, 1]), blocks, beyond)
    assert k == 1 and (out[50:60] == "H").all() and (out[60:200] == "D").all() and (out[:50] == "A").all()
    out, k = restart_back_check(lab, np.array([20, 50, 55, 90]), np.array([1, 0, 1, 1]), blocks, beyond)
    assert k == 0, "away played it between home's touch and the line, and nothing says the ball went out"
    out, k = restart_back_check(lab, np.array([20, 50, 55, 90]), np.array([1, 0, 1, 1]), blocks, beyond, out_tags=[195])
    assert k == 1 and (out[50:55] == "H").all() and (out[55:200] == "D").all(), "a corner for away: its touch at 55 was not play"
    at_line = np.zeros(300, bool); at_line[205] = True                  # no tag, but play restarts from the touchline
    assert restart_back_check(lab, np.array([20, 50, 55, 90]), np.array([1, 0, 1, 1]), blocks, beyond, boundary=at_line)[1] == 1


def _self_test():
    _out_touch_self_test()
    _contradicted_self_test()
    _touch_model_self_test()
    _touch_features_self_test()
    _lone_touch_self_test()
    # a pass along x at 10 m/s, received by an away player at frame 50 and played up the pitch
    t = np.arange(100)
    ball = np.column_stack([np.minimum(t, 50) * 0.4, np.maximum(t - 50, 0) * 0.4])
    ball += np.random.default_rng(0).normal(0, 0.05, ball.shape)       # wobble left after the provider's smoothing
    pos = np.full((100, 2, 2), np.nan)
    pos[:, 0] = [0.0, 0.5]                                             # home passer at the start
    pos[:, 1] = [20.5, 0.0]                                            # away receiver at the corner
    frames, who = path_events(ball, pos, np.array([0, 1]))
    near = np.abs(frames - 50) <= 3
    assert near.sum() == 1 and who[near][0] == 1, (frames, who)       # one event at the corner, away's
    assert not np.any((frames > 8) & (frames < 44)), frames            # none along the straight pass
    lt = last_touch_team(100, frames, who)
    assert lt[90] == 1
    seen = np.ones(100, bool); seen[70:80] = False
    lt = last_touch_team(100, frames, who, seen, max_age=200)
    assert lt[65] == 1 and lt[90] == -1, "a blackout makes the last touch unknown"
    b = np.full((60, 2), np.nan); b[:10] = np.column_stack([95 + np.arange(10) * 0.8, np.full(10, 30.0)])   # 20 m/s at the goal line
    ho = heading_out(b, [(1, 0, 59)], (105.0, 68.0))
    assert ho[13:25].all() and not ho[:10].any(), np.flatnonzero(ho)
    b[:10, 0] = 50 + np.arange(10) * 0.1                                 # slow, mid-pitch: just a blackout
    assert not heading_out(b, [(1, 0, 59)], (105.0, 68.0)).any()
    # H H A(duel) H: the lone contested away touch is dropped; an uncontested one stays
    fr, wh = np.array([10, 20, 30, 40]), np.array([0, 0, 1, 0])
    f2, w2 = drop_duel_touches(fr, wh, np.array([1.0, 1.0, 2.0, 1.0]), np.array([9.0, 9.0, 1.0, 9.0]))
    assert w2.tolist() == [0, 0, 0]
    f2, w2 = drop_duel_touches(fr, wh, np.array([1.0, 1.0, 9.0, 1.0]), np.array([9.0, 9.0, 1.0, 9.0]))
    assert w2.tolist() == [0, 0, 1, 0]
    lab = np.array(list("H" * 40 + "A" * 10 + "H" * 40))
    m, k = merge_untouched_spells(lab, np.array([5]), np.array([0]), np.ones(90, bool), [(1, 0, 89)])
    assert k == 1 and set(m) == {"H"}
    m, k = merge_untouched_spells(lab, np.array([45]), np.array([1]), np.ones(90, bool), [(1, 0, 89)])
    assert k == 0, "a spell with its own touch stays"
    lab = np.array(list("D" * 40 + "A" * 10 + "H" * 40))
    m, k = merge_untouched_spells(lab, np.array([60]), np.array([0]), np.ones(90, bool), [(1, 0, 89)])
    assert k == 1 and "".join(m) == "D" * 40 + "H" * 50, "after a stoppage it joins the team that did play"
    assert len(_rdp(np.column_stack([t, np.zeros(100)]).astype(float), 1.0)) == 2


def _pass_self_test():
    n = 200
    t = np.arange(n)
    ball = np.stack([10.0 + 0.5 * t, np.full(n, 30.0)], axis=1)                # 12.5 m/s along a straight line
    frames, who = np.array([20, 30, 80]), np.array([0, 1, 0])                  # home kicks, "away event", home receives
    f, w, p = completed_passes(frames, who, ball)
    assert f.tolist() == [20, 80] and w.tolist() == [0, 0] and p.all(), "the away event on a straight fast ball goes"
    bent = ball.copy()
    bent[50:, 1] = 30.0 + 0.5 * (t[50:] - 50)                                  # deflected at frame 50
    f, w, p = completed_passes(frames, who, bent)
    assert f.tolist() == [20, 30, 80] and not p.any(), "a deflected ball is not a completed pass"
    slow = np.stack([10.0 + 0.1 * t, np.full(n, 30.0)], axis=1)                # 2.5 m/s: carried, not passed
    assert not completed_passes(frames, who, slow)[2].any()
    hidden = ball.copy()
    hidden[40:45] = np.nan
    assert not completed_passes(frames, who, hidden)[2].any(), "an unseen stretch could hide a touch"
    assert not completed_passes(np.array([20, 80]), np.array([0, 1]), ball)[2].any(), "different teams: not a pass"
    rolling = np.stack([10.0 + np.cumsum(np.linspace(0.6, 0.1, n)), np.full(n, 30.0)], axis=1)   # 15 m/s slowing to 2.5
    assert completed_passes(np.array([5, 95]), np.array([0, 0]), rolling)[2].all(), "a pass that slows as it rolls is still a pass"
    kicked_on = rolling.copy()
    kicked_on[60:, 0] = kicked_on[59, 0] + 0.6 * (t[60:] - 59)                                    # sped up again at frame 60
    assert not completed_passes(np.array([5, 95]), np.array([0, 0]), kicked_on)[2].any(), "a second kick on the way is not one pass"
    # home touches at 20, the kick at 28 is credited to the away player standing nearer, home receives at 80
    fr, wh = np.array([20, 28, 80]), np.array([0, 1, 0])
    near, far = np.array([1.0, 1.5, 1.0]), np.array([9.0, 1.0, 9.0])
    f, w, pr = completed_passes(fr, wh, ball, near, far)
    assert w.tolist() == [0, 0, 0] and pr.tolist() == [False, True, True], "the kick is the home team's second play"
    f, w, pr = completed_passes(fr, wh, ball, np.array([1.0, 8.0, 1.0]), far)
    assert w.tolist() == [0, 0] and pr.all(), "not re-credited, but the ball flew straight from home to home past him"
    f, w, pr = completed_passes(np.array([0, 28, 80]), np.array([-1, 1, 0]), ball, near, far)
    assert w.tolist() == [-1, 1, 0], "a header that flies straight to the other team is still a change of team"


def _doubtful_self_test():
    ball = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    pos = np.array([[[10.5, 10.0], [40.0, 40.0]], [[50.0, 50.0], [20.5, 20.0]], [[30.5, 30.0], [31.5, 30.0]]])
    doubtful = np.array([[False, True], [False, True], [False, True]])
    f, w = drop_doubtful_touches(np.array([0, 1, 2]), np.array([0, 1, 0]), ball, pos, doubtful)
    assert f.tolist() == [0, 2] and w.tolist() == [0, 0], "only the touch by the doubtful player goes"
    assert len(drop_doubtful_touches(np.zeros(0, int), np.zeros(0, int), ball, pos, doubtful)[0]) == 0


if __name__ == "__main__":
    _doubtful_self_test()
    _before_whistle_self_test()
    _pass_self_test()
    _self_test()
    print("ball_events self-test passed")
