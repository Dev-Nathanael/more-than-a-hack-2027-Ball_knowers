"""One path per player through a half, by following each player from frame to frame (our idea).

The tracking does not say who is who and its tracks break every 35 s or so, but a player cannot
jump: the track nearest to where he was a moment ago is him. Starting from the first frame of a half
that shows 11 players of a team, each of the 11 is followed on the tracking joined from both cameras
(cleaning K7):

  - while his track continues, he stays on it;
  - when it ends he is "lost". If the broadcast camera was following him on a track of its own and that
    track runs on without a break, the tactical track it now sits on is him: the two cameras lose
    players at different moments, so one carries the identity across the other's break;
  - otherwise he takes over a track nobody is following that he could have reached (1 m plus 8 m/s for
    every second he has been lost);
  - if at that moment another lost teammate could have reached the same track, or he could have
    reached another unfollowed track, the pick is marked "unsure": from there the path may be
    another person's;
  - a gap of at most BRIDGE_FRAMES is drawn as a straight line and marked "bridged"; longer gaps stay empty.

Nothing here is used by any solution. Prints aggregates only; the paths are cached for the viewer.

    python experiments/player_paths.py        # both teams, first half
    python experiments/player_paths.py 2      # second half
"""
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Params
FPS = 25
TEAM = 11
SAME_TRACK_STEP_M = 1.5     # a track that moves less than this in one frame is still the same player
REACH_BASE_M = 1.0          # tracking noise
REACH_MPS = 8.0             # a player covers at most this while unseen (a sprint)
REACH_MAX_M = 40.0
BRIDGE_FRAMES = 2 * FPS     # gaps up to 2 s are drawn as a straight line
FILL_FRAMES = 10 * FPS      # gaps up to 10 s are filled from the team's shape; longer ones stay empty
SHAPE_MEMORY = 60 * FPS     # a player's place in the shape is his average over about the last minute
SHAPE_WARM = 10 * FPS       # ... and is used once he has been seen this long
SHAPE_AFTER = 10 * FPS      # ... for a player unseen at least this long. On IDSSE (real identities) the place in
                            # the shape names the right player 68% / 63% / 62% of the time after 10 / 30 / 60 s
                            # unseen, nearest-to-where-he-was 70% / 48% / 40%; at 5 s it is 71% against 86%
SHAPE_MIN_SPREAD_M = 3.0
SURE_RATIO = 2.0            # a pick is sure when the next best choice is this many times further (+ 1 m)
SEEN, BRIDGED, FILLED, LOST = 1, 2, 3, 0


def team_shape(xy):
    """Centre (N, 2) and spread (N, 2) of one team's tracks, frame by frame. NaN where fewer than 4 are seen.

    A player's place in the shape is (position - centre) / spread. Dividing by the spread is what makes
    the place hold whether the team is stretched out in attack or packed together in defence. Everybody
    counts, the goalkeeper too: leaving one track out makes the centre jump whenever that track changes."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre, spread = np.nanmean(xy, axis=1).astype(float), np.maximum(np.nanstd(xy, axis=1), SHAPE_MIN_SPREAD_M).astype(float)
    few = np.isfinite(xy[..., 0]).sum(axis=1) < 4
    centre[few], spread[few] = np.nan, np.nan
    return centre, spread


def window_shape(pts):
    """Centre and spread (n, 2) over a short window from the players seen on every frame of it, so the
    centre does not jump when somebody drops out. NaN with fewer than 4 such players."""
    pts = pts[:, np.isfinite(pts[..., 0]).all(axis=0)].astype(float)
    if pts.shape[1] < 4:
        return np.full((len(pts), 2), np.nan), np.full((len(pts), 2), np.nan)
    return pts.mean(axis=1), np.maximum(pts.std(axis=1), SHAPE_MIN_SPREAD_M)


def fill_gap(start, end, centre, spread):
    """Positions between two sightings (frames 1 .. n-2 of an n-frame window): the player keeps his
    place in the team's shape, sliding evenly from his place at the first sighting to his place at the
    second, so he moves with his team in between. A straight line where the shape is unknown."""
    n = len(centre)
    w = (np.arange(1, n - 1) / (n - 1))[:, None]
    if not np.isfinite(centre).all():
        return (1 - w) * start + w * end
    a, b = (start - centre[0]) / spread[0], (end - centre[-1]) / spread[-1]
    return centre[1:-1] + ((1 - w) * a + w * b) * spread[1:-1]


def unbroken_since(bxy):
    """(N, Sb) first frame of the unbroken stretch each broadcast track is in (-1 where unseen).
    A stretch breaks where the track is unseen or steps further than a player can."""
    seen = np.isfinite(bxy[..., 0])
    step = np.hypot(*np.moveaxis(np.diff(bxy, axis=0, prepend=bxy[:1]), -1, 0))
    start = seen & ~(np.roll(seen, 1, axis=0) & (step < SAME_TRACK_STEP_M))
    start[0] = seen[0]
    frame = np.arange(len(bxy))[:, None]
    return np.where(seen, np.maximum.accumulate(np.where(start, frame, -1), axis=0), -1)


def pair_with_broadcast(xy, bxy):
    """(N, S) the broadcast track paired with each track in each frame (-1 = none)."""
    from camera_agreement import match
    out = np.full(xy.shape[:2], -1, int)
    for f in np.flatnonzero(np.isfinite(bxy[..., 0]).any(axis=1)):
        ti, bi = np.flatnonzero(np.isfinite(xy[f, :, 0])), np.flatnonzero(np.isfinite(bxy[f, :, 0]))
        r, c, _ = match(bxy[f, bi], xy[f, ti])
        out[f, ti[c]] = bi[r]
    return out


def follow(xy, bid=None, since=None, shape=True):
    """xy: (N, S, 2) one team's tracks, NaN = unseen. bid (N, S) and since (N, Sb), from
    pair_with_broadcast and unbroken_since, let the broadcast carry a player across a break.
    shape: a lost player takes the free track nearest his usual place in the team's shape (not nearest
    where he was last seen), and gaps up to FILL_FRAMES are filled from the shape.
    Returns (path (N, 11, 2), state (N, 11), unsure (N, 11) bool: True from an unsure pick-up onwards,
    pickups, unsure_pickups, pickups the broadcast decided)."""
    n, s = xy.shape[:2]
    seen = np.isfinite(xy[..., 0])
    path = np.full((n, TEAM, 2), np.nan, np.float32)
    state = np.zeros((n, TEAM), np.uint8)
    unsure = np.zeros((n, TEAM), bool)
    full = np.flatnonzero(seen.sum(axis=1) == TEAM)
    if not len(full):
        return path, state, unsure, 0, 0, 0
    f0 = int(full[0])
    slot = np.flatnonzero(seen[f0]).copy()          # the track each player is on
    last = xy[f0, slot].copy()                      # where he was last seen
    age = np.ones(TEAM, int)                        # frames since then
    flag = np.zeros(TEAM, bool)
    path[f0], state[f0] = last, SEEN
    pickups = unsure_pickups = by_broadcast = 0
    b_track, b_frame = np.full(TEAM, -1), np.full(TEAM, -1)     # his broadcast track, and when it was last on him
    centre, spread = team_shape(xy)
    place, known = np.zeros((TEAM, 2)), np.zeros(TEAM, int)     # his usual place in the shape, frames it rests on
    for f in range(f0 + 1, n):
        got = np.zeros(TEAM, bool)
        taken = np.zeros(s, bool)
        for k in range(TEAM):                       # his own track goes on
            if age[k] == 1 and seen[f, slot[k]] and np.hypot(*(xy[f, slot[k]] - last[k])) < SAME_TRACK_STEP_M:
                got[k], taken[slot[k]] = True, True
        if bid is not None:                         # the broadcast track that was on him is now on another track
            for k in np.flatnonzero(~got & (b_track >= 0)):
                if 0 <= since[f, b_track[k]] <= b_frame[k]:
                    j = np.flatnonzero((bid[f] == b_track[k]) & seen[f] & ~taken)
                    if len(j):
                        slot[k], got[k], taken[j[0]] = j[0], True, True
                        pickups += 1
                        by_broadcast += 1
        rest, cand = np.flatnonzero(~got), np.flatnonzero(seen[f] & ~taken)
        if len(rest) and len(cand):                 # lost players take over unfollowed tracks they could have reached
            d = np.hypot(xy[f, cand, None, 0] - last[None, rest, 0], xy[f, cand, None, 1] - last[None, rest, 1]).T
            reach = np.minimum(REACH_BASE_M + REACH_MPS * age[rest] / FPS, REACH_MAX_M)[:, None]
            can = d <= reach
            cost = d.copy()
            if shape and np.isfinite(centre[f, 0]):     # how far each track is from each player's usual place
                usual = centre[f] + place[rest] * spread[f]
                by_shape = np.hypot(xy[f, cand, None, 0] - usual[None, :, 0], xy[f, cand, None, 1] - usual[None, :, 1]).T
                cost = np.where(((known[rest] >= SHAPE_WARM) & (age[rest] >= SHAPE_AFTER))[:, None], by_shape, d)
            cost = np.where(can, cost, 1e6)
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if not can[i, j]:
                    continue
                k = rest[i]
                pickups += 1
                other = min(np.delete(cost[i], j).min(initial=1e6), np.delete(cost[:, j], i).min(initial=1e6))
                if other < SURE_RATIO * cost[i, j] + 1.0:       # another track, or another player, fits nearly as well
                    unsure_pickups += 1
                    flag[k] = True
                slot[k], got[k] = cand[j], True
        last[got] = xy[f, slot[got]]
        if bid is not None:
            now = bid[f, slot]
            keep = got & (now >= 0)
            b_track[keep], b_frame[keep] = now[keep], f
        age = np.where(got, 1, age + 1)
        path[f, got], state[f, got] = last[got], SEEN
        if np.isfinite(centre[f, 0]):
            w = np.maximum(1.0 / (known[got] + 1), 1.0 / SHAPE_MEMORY)[:, None]
            place[got] += w * ((last[got] - centre[f]) / spread[f] - place[got])
            known[got] += 1
        unsure[f] = flag
    for k in range(TEAM):                           # gaps: his place in the shape carries him between two sightings
        idx = np.flatnonzero(state[:, k] == SEEN)
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= (FILL_FRAMES if shape else BRIDGE_FRAMES) + 1:
                others = np.where((state[a:b + 1] == SEEN)[..., None], path[a:b + 1], np.nan)
                others[:, k] = np.nan
                c, sp = window_shape(others) if shape else (np.full((b - a + 1, 2), np.nan),) * 2
                path[a + 1:b, k] = fill_gap(path[a, k], path[b, k], c, sp)
                state[a + 1:b, k] = BRIDGED if b - a <= BRIDGE_FRAMES + 1 else FILLED
    return path, state, unsure, pickups, unsure_pickups, by_broadcast


def idsse_check(match_id="J03WMX", seed=0):
    """On an outside match with real identities: is a player's place in the shape enough to tell who is
    who after everybody was unseen for a while, and how far off is a filled gap?"""
    from external import load_idsse
    players, _, team, _, _, blocks = load_idsse(match_id)
    rng = np.random.default_rng(seed)
    hits = {g: [0, 0, 0] for g in (5, 10, 30, 60)}          # by shape, by last position, players
    err = {g: [[], []] for g in (5, 10)}                    # straight line, shape
    for _, first, last in blocks:
        for side in (0, 1):
            xy = players[first:last + 1][:, team == side, :2].astype(float)
            centre, spread = team_shape(xy)
            norm = (xy - centre[:, None]) / spread[:, None]
            for f in range(120 * FPS, len(xy), 10 * FPS):
                for g in hits:
                    a = f - g * FPS
                    on = np.flatnonzero(np.isfinite(xy[f, :, 0]) & np.isfinite(xy[a, :, 0]) &
                                        (np.isfinite(norm[a - SHAPE_MEMORY:a, :, 0]).mean(axis=0) > 0.9))
                    if len(on) < 10 or not np.isfinite(centre[f, 0]):
                        continue
                    usual = centre[f] + np.nanmean(norm[a - SHAPE_MEMORY:a, on], axis=0) * spread[f]
                    for i, ref in enumerate((usual, xy[a, on])):
                        d = np.hypot(xy[f, on, None, 0] - ref[None, :, 0], xy[f, on, None, 1] - ref[None, :, 1])
                        r, c = linear_sum_assignment(d)
                        hits[g][i] += int(np.sum(r == c))
                    hits[g][2] += len(on)
                for g in err:                               # hide one player for g seconds and fill him back in
                    a, b = f - g * FPS, f
                    on = np.flatnonzero(np.isfinite(xy[a:b + 1, :, 0]).all(axis=0))
                    if len(on) < 10:
                        continue
                    k = rng.choice(on)
                    hidden = xy[a:b + 1].copy()
                    hidden[1:-1, k] = np.nan
                    hidden[:, k] = np.nan
                    c2, s2 = window_shape(hidden)
                    w = (np.arange(1, b - a) / (b - a))[:, None]
                    line = (1 - w) * xy[a, k] + w * xy[b, k]
                    for i, guess in enumerate((line, fill_gap(xy[a, k], xy[b, k], c2, s2))):
                        err[g][i].append(np.hypot(*(guess - xy[a + 1:b, k]).T).mean())
    print("{}: who is who after everyone was unseen, from 10+ players at a time".format(match_id))
    for g, (shape_hits, last_hits, total) in hits.items():
        print("  unseen {:>2} s: right by place in the shape {:.0f}%, by nearest to where he was {:.0f}%  ({} players)".format(
            g, 100 * shape_hits / max(total, 1), 100 * last_hits / max(total, 1), total))
    for g, (line, shaped) in err.items():
        print("  a {} s gap filled: straight line off by {:.2f} m, shape off by {:.2f} m (mean over the gap, {} gaps)".format(
            g, np.mean(line), np.mean(shaped), len(line)))


# ---------- grouping track pieces by place in the shape (the method in use) ----------
OUTFIELD = 10
PIECE_LONG = 60 * FPS       # pieces at least this long say where the ten places in the shape are
NEAR_ONLY_S = 10.0          # under this many seconds unseen, nearest-to-where-he-was decides (IDSSE: 97% right at
                            # 2 s, 86% at 5 s, 70% at 10 s; place in the shape 76% at any gap, the two together 79% at 10 s)
SHAPE_UNIT_M = 5.0          # 5 m off his usual place counts like being at the edge of his reach
UNSURE_MARGIN = 0.5         # a pick is unsure when another choice costs less than this much more
STAY_BONUS = 1.0            # a piece keeps its place unless another place fits it 5 m better
NEAR_WEIGHT = 1.0           # a piece starting where a player just vanished is worth as much as a 5 m better fit


def pieces_of(xy):
    """[(track, first, last)] unbroken pieces of every track: seen on every frame, no step a player cannot take."""
    from cleaning import runs
    out = []
    for k in range(xy.shape[1]):
        for a, z in runs(np.isfinite(xy[:, k, 0])):
            jump = np.flatnonzero(np.hypot(*np.diff(xy[a:z + 1, k], axis=0).T) >= SAME_TRACK_STEP_M)
            edges = [a] + [a + j + 1 for j in jump] + [z + 1]
            out += [(k, i, j - 1) for i, j in zip(edges[:-1], edges[1:])]
    return sorted(out, key=lambda t: (t[1], t[0]))


def assign_places(xy, bid=None, since=None, seed=0):
    """Ten outfield paths from one team's tracks (N, S, 2), goalkeeper tracks left out by the caller.

    The provider's unbroken track pieces behave like real players (measured: a piece's place across the
    pitch moves 0.37 team spreads from minute to minute, a real player's 0.45), so a piece is never cut.
    What has to be decided is which pieces belong together:
      1. every piece gets its average place in the team's shape;
      2. the long pieces say where the ten places are (k-means, seeded);
      3. every time a piece starts, all the pieces running at that moment are shared out over the ten
         places so that the total misfit is smallest. A piece keeps the place it has unless another fits
         clearly better; a piece that starts where a player vanished under 10 s ago leans to his place; a
         broadcast track that ran on unbroken from a player onto the new piece leans hard to his place.
    Every share-out is judged against the ten fixed places, not against what a path picked up before, so
    a wrong pick is put right at the next share-out instead of spoiling the rest of the half. A piece
    can therefore change place part-way ("moved"). A pick is "unsure" when another place, or another
    piece, fits nearly as well.
    Returns (path (N, 10, 2), state, unsure (N, 10), stats dict).
    """
    from sklearn.cluster import KMeans
    n = len(xy)
    centre, spread = team_shape(xy)
    typical = np.nanmedian(spread, axis=0)
    pieces = pieces_of(xy)
    norm = (xy - centre[:, None]) / spread[:, None]
    sig = np.array([np.nanmean(norm[a:z + 1, k], axis=0) for k, a, z in pieces])
    length = np.array([z - a + 1 for _, a, z in pieces])
    ok = np.isfinite(sig[:, 0])
    long = ok & (length >= PIECE_LONG)
    if long.sum() < OUTFIELD:
        long = ok
    places = KMeans(n_clusters=OUTFIELD, n_init=10, random_state=seed).fit(sig[long], sample_weight=length[long]).cluster_centers_
    path = np.full((n, OUTFIELD, 2), np.nan, np.float32)
    state = np.zeros((n, OUTFIELD), np.uint8)
    unsure = np.zeros((n, OUTFIELD), bool)
    holder = np.full(OUTFIELD, -1)                  # the piece on each place
    last_pos, last_frame = np.full((OUTFIELD, 2), np.nan), np.full(OUTFIELD, -1)
    b_track, b_frame = np.full(OUTFIELD, -1), np.full(OUTFIELD, -1)
    stats = {"pieces": len(pieces), "placed": 0, "unsure": 0, "by_broadcast": 0, "left_out_frames": 0, "by_nearness": 0, "moved": 0}
    starts = np.array([q[1] for q in pieces])
    events = np.unique(starts)
    active = []
    for e, f in enumerate(events):
        nxt = events[e + 1] if e + 1 < len(events) else n
        for r in np.flatnonzero(holder >= 0):       # pieces that ended before this moment free their place
            if pieces[holder[r]][2] < f:
                holder[r] = -1
        new = np.flatnonzero(starts == f).tolist()
        active = [q for q in active if pieces[q][2] >= f] + new
        cost = np.zeros((len(active), OUTFIELD))
        sure_link = np.zeros(cost.shape, bool)
        near_link = np.zeros(cost.shape, bool)
        for a, q in enumerate(active):
            k = pieces[q][0]
            cost[a] = np.hypot(*((sig[q] - places) * typical).T) / SHAPE_UNIT_M if ok[q] else 2.0
            for r in range(OUTFIELD):
                if holder[r] == q:
                    cost[a, r] -= STAY_BONUS
                elif q in new and holder[r] < 0 and last_frame[r] >= 0 and (f - last_frame[r]) / FPS < NEAR_ONLY_S:
                    gap = (f - last_frame[r]) / FPS
                    near = np.hypot(*(xy[f, k] - last_pos[r])) / (REACH_BASE_M + REACH_MPS / 2 * gap)
                    cost[a, r] += NEAR_WEIGHT * (near - 1.0)
                    near_link[a, r] = near < 1.0
                if q in new and bid is not None and b_track[r] >= 0 and bid[f, k] == b_track[r] and 0 <= since[f, b_track[r]] <= b_frame[r]:
                    cost[a, r] -= 3.0
                    sure_link[a, r] = True
        rows, cols = linear_sum_assignment(cost)
        was = holder.copy()
        holder[:] = -1
        for a, r in zip(rows, cols):
            q = active[a]
            k, first, last = pieces[q]
            end = min(last, nxt - 1)
            other = min(np.delete(cost[a], r).min(initial=1e3), np.delete(cost[:, r], a).min(initial=1e3))
            doubt = (not sure_link[a, r]) and other < cost[a, r] + UNSURE_MARGIN
            holder[r] = q
            path[f:end + 1, r], state[f:end + 1, r], unsure[f:end + 1, r] = xy[f:end + 1, k], SEEN, doubt
            if q in new:
                stats["placed"] += 1
                stats["unsure"] += int(doubt)
                stats["by_broadcast"] += int(sure_link[a, r])
                stats["by_nearness"] += int(near_link[a, r] and not sure_link[a, r])
            elif was[r] != q:
                stats["moved"] += 1
            last_pos[r], last_frame[r] = xy[end, k], end
            if bid is not None:
                on = np.flatnonzero(bid[f:end + 1, k] >= 0)
                if len(on):
                    b_track[r], b_frame[r] = bid[f + on[-1], k], f + on[-1]
        out = set(range(len(active))) - set(rows.tolist())
        stats["left_out_frames"] += int(sum(min(pieces[active[a]][2], nxt - 1) - f + 1 for a in out))
    for r in range(OUTFIELD):                       # gaps: his place in the shape carries him between two sightings
        idx = np.flatnonzero(state[:, r] == SEEN)
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= FILL_FRAMES + 1:
                others = np.where((state[a:b + 1] == SEEN)[..., None], path[a:b + 1], np.nan)
                others[:, r] = np.nan
                path[a + 1:b, r] = fill_gap(path[a, r], path[b, r], *window_shape(others))
                state[a + 1:b, r] = BRIDGED if b - a <= BRIDGE_FRAMES + 1 else FILLED
    return path, state, unsure, stats


def place_steadiness(path, state):
    """(how much a path's place across the pitch moves from minute to minute, how far apart the paths
    lie), both in team spreads. Real players (IDSSE): about 0.45 and 0.72."""
    import warnings
    xy = np.where((state != LOST)[..., None], path, np.nan).astype(float)
    centre, spread = team_shape(xy)
    win = 60 * FPS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pl = (xy[..., 1] - centre[:, None, 1]) / spread[:, None, 1]
        m = np.array([[np.nanmean(pl[i:i + win, k]) if np.isfinite(pl[i:i + win, k]).mean() > 0.8 else np.nan
                       for i in range(0, len(pl) - win + 1, win)] for k in range(pl.shape[1])])
        return float(np.nanmedian(np.nanstd(m, axis=1))), float(np.nanstd(np.nanmean(m, axis=1)))


def idsse_places(match_id="J03WMX", seed=0):
    """The whole method on an outside match with real identities: its tracks are cut into pieces like ours
    (a piece lasts 60 s on average, then the player is unseen for 4 s on average, and comes back on another
    track), the pieces are grouped again, and each path is checked against the player it should be."""
    from external import load_idsse
    players, _, team, _, pitch, blocks = load_idsse(match_id)
    rng = np.random.default_rng(seed)
    pure, placed, total, stead = [], 0, 0, []
    for _, first, last in blocks:
        for side in (0, 1):
            real = players[first:last + 1][:, team == side, :2].astype(float)
            mean_x = np.nanmean(real[..., 0], axis=0)
            real = np.delete(real, np.nanargmax(np.abs(mean_x - pitch[0] / 2)), axis=1)     # without the goalkeeper
            n, cols = real.shape[:2]
            cut = np.full((n, 2 * cols, 2), np.nan)
            who = np.full((n, 2 * cols), -1)
            for k in range(cols):
                f = 0
                while f < n:
                    z = min(n, f + 1 + int(rng.exponential(60 * FPS)))
                    free = np.flatnonzero((who[f:z] < 0).all(axis=0))
                    t = rng.choice(free)
                    seen = np.isfinite(real[f:z, k, 0])
                    cut[f:z, t][seen], who[f:z, t][seen] = real[f:z, k][seen], k
                    f = z + int(rng.exponential(4 * FPS))
            path, state, unsure, st = assign_places(cut, seed=seed)
            truth = np.full((n, OUTFIELD), -1)
            for r in range(OUTFIELD):
                for f in np.flatnonzero(state[:, r] == SEEN)[::5]:
                    hit = np.flatnonzero((cut[f, :, 0] == path[f, r, 0]) & (cut[f, :, 1] == path[f, r, 1]))
                    truth[f, r] = who[f, hit[0]] if len(hit) else -1
                t = truth[:, r][truth[:, r] >= 0]
                for w in range(0, n, 5 * 60 * FPS):         # is a path one player, five minutes at a time?
                    seg = truth[w:w + 5 * 60 * FPS, r]
                    seg = seg[seg >= 0]
                    if len(seg) > 50:
                        pure.append((np.bincount(seg).max(), len(seg)))
                if len(t):
                    placed += np.bincount(t).max()
                    total += len(t)
            stead.append(place_steadiness(path, state))
    pure = np.array(pure)
    print("{}: tracks cut into pieces and grouped again".format(match_id))
    print("  a path is the same player on {:.0f}% of its frames within any five minutes, {:.0f}% over a whole half".format(
        100 * pure[:, 0].sum() / pure[:, 1].sum(), 100 * placed / max(total, 1)))
    print("  place across the pitch moves {:.2f} from minute to minute, paths lie {:.2f} apart (real players: 0.45 and 0.72)".format(
        *np.mean(stead, axis=0)))


def cache_file(half):
    from tracking import CACHE_DIR, GAME
    return os.path.join(CACHE_DIR, "{}_player_paths_h{}.npz".format(GAME, half))


def report(half=1):
    from cleaning import runs
    from tracking import GAME, half_video_blocks, load_clean
    from camera_agreement import broadcast_on_tactical_frames
    from tracking import goalkeepers
    tp, _, team = load_clean("K7")
    gk = goalkeepers("tactical")
    _, first, last = [b for b in half_video_blocks(GAME) if b[0] == half][0]
    bxy, _, bteam, _ = broadcast_on_tactical_frames(tp, team, None)
    out = {"first": first}
    for side, name in ((0, "home"), (1, "away")):
        xy, b = tp[first:last + 1][:, (team == side) & ~gk, :2], bxy[first:last + 1][:, bteam == side]
        path, state, unsure, st = assign_places(xy, pair_with_broadcast(xy, b), unbroken_since(b))
        keeper = tp[first:last + 1][:, (team == side) & gk, :2]
        with np.errstate(invalid="ignore"):         # the goalkeeper is the track the data marks as one: column 0
            kp = np.nanmean(keeper, axis=1) if keeper.shape[1] else np.full((len(xy), 2), np.nan)
        path = np.concatenate([kp[:, None].astype(np.float32), path], axis=1)
        state = np.concatenate([np.isfinite(kp[:, :1]).astype(np.uint8), state], axis=1)
        unsure = np.concatenate([np.zeros((len(xy), 1), bool), unsure], axis=1)
        out.update({"path_" + name: path, "state_" + name: state, "unsure_" + name: unsure})
        s = state[:, 1:]
        print("\n{} half {}: {} frames, {} track pieces".format(name, half, len(s), st["pieces"]))
        print("  outfield: seen {:.1f}%, short gap (up to {} s) {:.1f}%, filled from the shape (up to {} s) {:.1f}%, empty {:.1f}%; goalkeeper seen {:.1f}%".format(
            100 * np.mean(s == SEEN), BRIDGE_FRAMES // FPS, 100 * np.mean(s == BRIDGED), FILL_FRAMES // FPS,
            100 * np.mean(s == FILLED), 100 * np.mean(s == LOST), 100 * np.mean(state[:, 0] == SEEN)))
        print("  known share per outfield player, best to worst: " + " ".join("{:.0f}%".format(100 * v) for v in sorted((s != LOST).mean(axis=0))[::-1]))
        print("  pieces placed: {}; by the broadcast track {:.0f}%, by nearness (unseen under {:.0f} s) {:.0f}%, by place in the shape {:.0f}%; unsure {:.0f}%".format(
            st["placed"], 100 * st["by_broadcast"] / max(st["placed"], 1), NEAR_ONLY_S, 100 * st["by_nearness"] / max(st["placed"], 1),
            100 * (st["placed"] - st["by_broadcast"] - st["by_nearness"]) / max(st["placed"], 1), 100 * st["unsure"] / max(st["placed"], 1)))
        print("  frames on an unsure piece: {:.1f}% of seen player-frames; track frames left out (no free place): {:.1f}%".format(
            100 * unsure[:, 1:].sum() / max((s == SEEN).sum(), 1), 100 * st["left_out_frames"] / max(np.isfinite(xy[..., 0]).sum(), 1)))
        print("  place across the pitch moves {:.2f} from minute to minute, paths lie {:.2f} apart (real players: 0.45 and 0.72)".format(
            *place_steadiness(path[:, 1:], s)))
    np.savez_compressed(cache_file(half), **out)
    print("\nsaved paths for the viewer")


def _self_test():
    n = 200
    xy = np.full((n, 13, 2), np.nan, np.float32)
    t = np.arange(n)
    for k in range(11):                             # 11 players walking in lanes 5 m apart
        xy[:, k, 0], xy[:, k, 1] = 10 + 0.1 * t, 5.0 * k
    xy[80:, 3] = np.nan                             # player 3's track ends, a new one starts 1 s later in his lane
    xy[105:, 11, 0], xy[105:, 11, 1] = 10 + 0.1 * t[105:], 15.0
    xy[150:160, 7] = np.nan                         # player 7 drops out for 10 frames on his own track
    path, state, unsure, pickups, bad, _ = follow(xy, shape=False)
    assert np.allclose(path[150, 3], [25.0, 15.0]) and np.allclose(path[199, 7], [29.9, 35.0])
    assert (state[80:105, 3] == BRIDGED).all() and np.allclose(path[90, 3], [19.0, 15.0], atol=0.2)
    assert (state[150:160, 7] == BRIDGED).all() and state[:80].min() == SEEN
    assert pickups == 2 and bad == 0 and not unsure.any()
    xy[105:, 12, 0], xy[105:, 12, 1] = 10 + 0.1 * t[105:], 17.0   # a second track he could have reached: unsure
    _, _, unsure, _, bad, _ = follow(xy, shape=False)
    assert bad == 1 and unsure[199, 3] and not unsure[99, 3] and not unsure[:, 0].any()
    # the broadcast follows player 3 without a break and sits on track 12 after the gap: sure, and track 12
    bxy = np.full((n, 2, 2), np.nan, np.float32)
    bxy[:, 0, 0], bxy[:, 0, 1] = 10 + 0.1 * t, 15.0 + 2.0 * np.clip((t - 80) / 25.0, 0, 1)
    bxy[60:, 1] = bxy[60:, 0] + [0.0, 20.0]
    since = unbroken_since(bxy)
    assert since[199, 0] == 0 and since[59, 1] == -1 and since[199, 1] == 60
    bid = pair_with_broadcast(xy, bxy)
    assert bid[50, 3] == 0 and bid[150, 12] == 0 and bid[170, 7] == 1
    path, _, unsure, _, bad, by_b = follow(xy, bid, since, shape=False)
    assert by_b == 2 and bad == 0 and not unsure.any() and np.allclose(path[150, 3], [25.0, 17.0])
    # shape: the lost player takes the track in his own lane, not the nearer one, and it is a sure pick
    xy[105:, 12, 1] = 13.0
    xy[105:, 11, 1] = 15.0
    xy[105:, 12, 0] = xy[105:, 11, 0] - 2.6
    global SHAPE_WARM
    global SHAPE_AFTER
    SHAPE_WARM, warm, SHAPE_AFTER, after = 50, SHAPE_WARM, 20, SHAPE_AFTER       # the toy match is 8 s long
    path, state, unsure, _, bad, _ = follow(xy)
    SHAPE_WARM, SHAPE_AFTER = warm, after
    assert np.allclose(path[150, 3], [25.0, 15.0]) and bad == 0 and (state[81:105, 3] == BRIDGED).all()
    c, sp = team_shape(xy[:50])
    assert np.isfinite(c).all() and sp.min() >= SHAPE_MIN_SPREAD_M
    mid = fill_gap(np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([[0.0, 0.0], [8.0, 0.0], [10.0, 0.0]]), np.ones((3, 2)))
    assert np.allclose(mid, [[8.0, 0.0]])     # carried along with the team, not left on the line between the ends
    # places: ten lanes cut into pieces on changing tracks come back as ten clean paths
    rng = np.random.default_rng(0)
    n = 3000
    lanes = np.stack([np.stack([40 + 10 * np.sin(np.arange(n) / 200.0) + 3 * (k % 3), 6.0 * k + 0 * np.arange(n)], axis=1) for k in range(10)], axis=1)
    cut = np.full((n, 20, 2), np.nan)
    for k in range(10):
        for a in range(0, n, 500):
            cut[a + 20 * (a > 0):a + 500, (k + 3 * (a // 500)) % 20 if (a // 500) % 2 else k] = lanes[a + 20 * (a > 0):a + 500, k]
    assert len(pieces_of(cut)) == 60
    path, state, unsure, st = assign_places(cut)
    for r in range(10):
        assert np.ptp(path[state[:, r] == SEEN, r, 1]) < 1e-3, "a path mixes lanes"
    assert st["placed"] == 60 and st["left_out_frames"] == 0 and (state != LOST).mean() > 0.99
    move, apart = place_steadiness(path, state)
    assert move < 0.05 and apart > 0.5
    print("player_paths self-test passed")


if __name__ == "__main__":
    _self_test()
    if len(sys.argv) > 1 and sys.argv[1] == "idsse":
        idsse_check(*sys.argv[2:3])
        idsse_places(*sys.argv[2:3])
        sys.exit()
    report(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
