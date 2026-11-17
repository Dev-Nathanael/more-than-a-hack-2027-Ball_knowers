"""Versions of the tracking clean-up, grown one step at a time by a self-improvement loop.

    from cleaning import VERSIONS, clean
    players, ball, summary = clean(players, ball, team, blocks, pitch, "V03")

Each version is the previous kept version plus one change, chosen by experiments/iterate.py's
ground-truth-free measurements (physics, a self-supervised ball test, the decoded match).
The reason for each change is written next to it and in docs/cleaning_iterations.md. A version
carries all its own parameters, so changing a later one never alters an earlier one, and every
version rebuilds byte for byte (nothing here is random or reads labels).

Steps, always in this order: orient, bounds, caps, keepers, relink, ghosts, extras, teamcheck,
broadcast, fill, smooth, speed. A step is off when its entry is missing or None. "feat" overrides parameters of experiments/features.py for that version.
"""

import copy
import hashlib
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Params
FPS = 25
MOVED_M = 0.01            # a position that moves more than 1 cm counts as moved in the summary
ORIENT_MIN_GAP_M = 10.0   # the two teams' rearmost players must differ by this to tell the direction
TEAM_MAX = 11

V01 = {"orient": False, "bounds": None, "caps": None, "ghosts": None, "fill": None, "smooth": None,
       "speed": "provider", "feat": {}}


def _then(prev, **changes):
    out = copy.deepcopy(prev)
    out.update(copy.deepcopy(changes))
    return out


VERSIONS = {}
VERSIONS["V01"] = V01                                           # the data every solution so far used
VERSIONS["V02"] = _then(V01, orient=True)                       # home attacks towards x = 105 in both halves
# kept: labels identical to V01, so no code depends on the direction of attack
VERSIONS["V03"] = _then(VERSIONS["V02"], bounds={"player_m": 8.0, "ball_m": 10.0})   # nothing real sits far off the pitch
# kept: 0.4% of ball positions removed, no players; decoded match unchanged (451 possessions, 23 under 1 s)
VERSIONS["V04"] = _then(VERSIONS["V03"], caps={"player_mps": 12.0, "ball_mps": 40.0, "jump_gap": 6})  # no impossible steps
# rejected: impossible steps gone, but one more possession under 1 s (24 vs 23); 6 blank ball frames
# after a jump likely turn a fast real moment into a short gap the decoder flickers across
VERSIONS["V05"] = _then(VERSIONS["V03"], caps={"player_mps": 12.0, "ball_mps": 40.0, "jump_gap": 6,
                                               "ball_jump_gap": 1})   # ball: blank only the jump frame
# rejected: labels identical to V04 (same 0.616% of ball removed), so every fast ball step is a
# one-frame spike, never a jump; the extra short possession comes from the spikes or the player cap
VERSIONS["V06"] = _then(VERSIONS["V03"], caps={"player_mps": 12.0, "ball_mps": None, "jump_gap": 6})  # players only
# kept: impossible player steps 0.49 -> 0 per 1000, short possessions still 23; so the ball spike cut
# is what added one, by leaving one-frame holes
VERSIONS["V07"] = _then(VERSIONS["V06"], caps={"player_mps": 12.0, "ball_mps": 40.0, "jump_gap": 6, "ball_jump_gap": 1},
                        fill={"ball": 2, "player": None})   # repair ball spikes: cut, then bridge holes of 1-2 frames
# rejected: still 24 short possessions, and bridging brings back fast steps; the decoder uses the
# spikes as evidence, so the ball is left as delivered from here on
VERSIONS["V08"] = _then(VERSIONS["V06"], ghosts={"near_m": 0.3, "frames": 25, "dv_mps": 0.3})  # 12.9% of frames show > 11 tracks
# rejected (no effect): output identical to V06. The extra tracks are not copies: 91-93% of crowded
# frames show exactly 12, the closest teammates stand 3-4 m apart and move; 15-17% have a track off
# the pitch (7-8% normally) and 36% one within 1.5 m of a line (20-24%): partly people at the touchline
VERSIONS["V09"] = _then(VERSIONS["V06"], extras={"off_m": 1.0})   # crowded team: drop tracks over 1 m off the pitch
# kept: crowded frames 12.8% -> 12.2%, 0.04% of player positions removed, decoded match unchanged
VERSIONS["V10"] = _then(VERSIONS["V09"], broadcast=True)   # ball gaps filled where the live broadcast sees it (9.2% of gaps)
# rejected: ball seen 59.1% -> 62.8%, but possessions 451 -> 478 and under 1 s 23 -> 26, fast ball
# steps 2.3 -> 2.7 per 1000: seams where the two cameras disagree, and brief broadcast sightings
VERSIONS["V11"] = _then(VERSIONS["V09"], broadcast={"min_run": 25})   # only broadcast sightings of 1 s or more
# rejected: possessions 475, under 1 s 25. The broadcast ball gives the decoder evidence inside former
# blackouts; whether the extra changes are real cannot be told without labels, so the rule says no
VERSIONS["V12"] = _then(VERSIONS["V09"], feat={"EDGE_VEL": True})   # predicted ball starts a gap with its real velocity
# kept: ball-test miss 2.43 -> 0.86 m at 0.5 s, 4.78 -> 3.04 m at 1 s, 9.92 -> 8.91 m at 3 s; labels unchanged
VERSIONS["V13"] = _then(VERSIONS["V12"], feat={"EDGE_VEL": True, "ROLL_DECEL": 3.0})   # a ground pass slows faster than 1.5 m/s^2
# kept: miss 0.86 -> 0.75 m at 0.5 s, 3.04 -> 2.70 m at 1 s, 8.91 -> 8.65 m at 3 s
VERSIONS["V14"] = _then(VERSIONS["V13"], feat={"EDGE_VEL": True, "ROLL_DECEL": 5.0})   # bracket: is it players stopping the ball?
# kept: miss 0.75 -> 0.69 m at 0.5 s, 2.70 -> 2.47 m at 1 s, 8.65 -> 8.43 m at 3 s (smaller steps now)
VERSIONS["V15"] = _then(VERSIONS["V14"], feat={"EDGE_VEL": True, "ROLL_DECEL": 5.0, "PHYSICS_FRAMES": 12})  # follow players after 0.5 s
# rejected: miss 2.47 -> 3.99 m at 1 s, 8.43 -> 9.22 m at 3 s; rolling physics beats following players early
VERSIONS["V16"] = _then(VERSIONS["V14"], feat={"EDGE_VEL": True, "ROLL_DECEL": 5.0, "PHYSICS_FRAMES": 50})  # physics for 2 s
# kept (marginal): only the 3 s miss moves, 8.43 -> 8.37 m; the predicted-ball tuning is done
VERSIONS["V17"] = _then(VERSIONS["V16"], fill={"ball": None, "player": 25})   # player gaps up to 1 s on a straight line
# rejected: it re-bridges the blanks the cap leaves after a track swap, bringing back impossible
# steps (0 -> 0.29 per 1000) for almost no gain (20.469 -> 20.475 players per frame)
VERSIONS["V18"] = _then(VERSIONS["V16"], smooth={"player": 7, "ball": None, "order": 2})   # light centred smoothing, players only
# kept: jitter 0.136 -> 0.118 cm per frame, decoded match unchanged; the provider already smooths,
# so stronger smoothing is not tried
VERSIONS["V19"] = _then(VERSIONS["V18"], speed="positions")   # player speed from cleaned positions, not the provider column
# rejected: dead share 38.2% -> 36.3%, 4.7% of labels change, possessions 465 and 26 under 1 s; the
# provider's speed and speed from positions disagree enough to move stoppages: a question for the bench
VERSIONS["V20"] = _then(VERSIONS["V18"], feat=dict(VERSIONS["V18"]["feat"], RADIAL=True))   # running at / away from the ball
VERSIONS["K1"] = _then(V01, keepers={"margin_m": 10.0})   # raw data minus keeper tracks deep in the other half
VERSIONS["K2"] = _then(VERSIONS["K1"], relink={"window": 25, "near_m": 2.0})   # stretches a wrong-team slot follows go back to the opponent
VERSIONS["K2W"] = _then(VERSIONS["K1"], relink={"window": 75, "near_m": 3.0})   # wider handover: 3 s, 3 m
VERSIONS["K3"] = _then(VERSIONS["K2"], bounds={"player_m": 8.0, "ball_m": 10.0},
                       caps={"player_mps": 12.0, "ball_mps": None, "jump_gap": 6}, extras={"off_m": 1.0})   # + the V loop's kept steps
# K2 / K2W / K3 (2026-11-16): re-link at 1 s / 2 m fixes 7% of the 12-against-10 frames, at 3 s / 3 m 16%; neither
# adds a possession under 1 s; the V loop's steps clear every impossible player step and far-off ball
VERSIONS["K4"] = _then(VERSIONS["K3"], relink={"window": 75, "near_m": 3.0})   # the submission's data from 2026-11-16
VERSIONS["K5"] = _then(VERSIONS["K4"], teamcheck={"match_m": 1.5, "min_frames": 25, "min_share": 0.5})   # broadcast names the wrong-team track
# K6 (2026-11-17): K5 deletes the misplaced keeper track, but on 1,229 graded frames that track is the player
# on the ball (the real keeper under the other keeper's slot). K6 keeps, as that half's keeper, whichever of the
# two keeper tracks stands nearer the goal over the stretch
VERSIONS["K6"] = _then(VERSIONS["K5"], keepers={"margin_m": 10.0, "nearer_goal": True})
# K7 (2026-11-17): both cameras joined player by player (fuse_cameras). Equal weight where both see a player,
# 3 to 1 for the broadcast in a bunch (another player within 1.5 m), players only the broadcast sees added
VERSIONS["K7"] = _then(VERSIONS["K5"], fuse={"gate_m": 3.0, "bunch_m": 1.5, "w_bunched": 0.75})
# K9 (2026-11-18): K7 + the broadcast ball where the tactical camera has none: sightings of 1 s or more (V11) that
# join on to the tactical ball within 3 m at each end they touch (a frame of the fastest ball is 1.6 m)
VERSIONS["K9"] = _then(VERSIONS["K7"], broadcast={"min_run": 25, "seam_m": 3.0})


def params_hash(version):
    """8 hex characters that change whenever anything in the version changes."""
    blob = json.dumps({"v": VERSIONS[version], "p": [FPS, ORIENT_MIN_GAP_M, TEAM_MAX]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def content_hash(arrays):
    """SHA-256 over names, dtypes, shapes and bytes of a dict of arrays.

    An .npz file stores the time it was written, so two identical runs give different file bytes;
    this hash compares what is inside instead.
    """
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update("{}|{}|{}|".format(key, a.dtype.str, a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# ---------- orientation ----------

def half_of_frame(n_frames, blocks):
    """(N,) 1 or 2: frames before the midpoint of half time belong to half 1 (as feature_cache does)."""
    split = (blocks[0][2] + blocks[1][1]) // 2
    return np.where(np.arange(n_frames) < split, blocks[0][0], blocks[1][0])


def flipped_halves(players, team, blocks):
    """{half: True if home attacks towards x = 0 there}. Raises when a half's direction is unclear.

    The side a team's rearmost player stands on is the side it defends (the goalkeeper)."""
    from tracking import team_features
    feat = team_features(players, team)
    out = {}
    for half, first, last in blocks:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rear = [np.nanmean(np.where(np.isfinite(feat[k][first:last + 1]), feat[k][first:last + 1], np.nan))
                    for k in ("min_x_H", "min_x_A")]
        if not np.all(np.isfinite(rear)) or abs(rear[0] - rear[1]) < ORIENT_MIN_GAP_M:
            raise ValueError("half {}: cannot tell which way the teams attack (rearmost players {} m)".format(
                half, np.round(rear, 1).tolist()))
        out[half] = bool(rear[0] > rear[1])
    return out


def rotate(xy, frames, pitch):
    """Turn the pitch by 180 degrees on the given frames: x -> length - x, y -> width - y.

    A rotation, not a mirror, so a team's left wing stays its left wing."""
    out = xy.copy()
    out[frames, ..., 0] = pitch[0] - xy[frames, ..., 0]
    out[frames, ..., 1] = pitch[1] - xy[frames, ..., 1]
    return out


def orient(players, ball, team, blocks, pitch):
    """(players, ball, flipped): every half turned so home attacks towards x = length."""
    flipped = flipped_halves(players, team, blocks)
    frames = np.isin(half_of_frame(len(ball), blocks), [h for h, f in flipped.items() if f])
    p = players.copy()
    p[..., :2] = rotate(players[..., :2], frames, pitch)
    return p, rotate(ball, frames, pitch), flipped


# ---------- physical limits ----------

def clip_bounds(xy, pitch, margin_m):
    """NaN where a position lies more than margin_m outside the pitch."""
    out = xy.copy()
    with np.errstate(invalid="ignore"):
        bad = ((xy[..., 0] < -margin_m) | (xy[..., 0] > pitch[0] + margin_m)
               | (xy[..., 1] < -margin_m) | (xy[..., 1] > pitch[1] + margin_m))
    out[bad] = np.nan
    return out


def step_speed(xy):
    """(N, ...) speed of the step into each frame, m/s; NaN where either end is unseen."""
    s = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    s[1:] = np.hypot(xy[1:, ..., 0] - xy[:-1, ..., 0], xy[1:, ..., 1] - xy[:-1, ..., 1]) * FPS
    return s


def cap_speed(xy, max_mps, jump_gap):
    """Cut what no player or ball can do.

    A lone spike (the step in and the step out both too fast) is one bad frame: blank it.
    A single jump (a track handed to another person) cannot be undone: blank jump_gap frames after
    it, so the track splits and no velocity window spans the two people.
    """
    out = xy.copy()
    with np.errstate(invalid="ignore"):
        fast = step_speed(out) > max_mps
        spike = np.zeros_like(fast)
        spike[:-1] = fast[:-1] & fast[1:]
        out[spike] = np.nan
        jump = step_speed(out) > max_mps
    blank = jump.copy()
    for k in range(1, jump_gap):
        blank[k:] |= jump[:-k]
    out[blank] = np.nan
    return out


# ---------- ghost tracks ----------

def runs(mask):
    """[(start, end_inclusive)] of True runs in a 1-D boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def drop_offpitch_extras(players, team, pitch, margin_m):
    """Where a team shows more than 11 tracks, its tracks more than margin_m outside the lines go
    (a substitute or a player coming on or off). Elsewhere a player may step off, e.g. for a throw-in."""
    out = players.copy()
    x, y = players[..., 0], players[..., 1]
    with np.errstate(invalid="ignore"):
        off = (x < -margin_m) | (x > pitch[0] + margin_m) | (y < -margin_m) | (y > pitch[1] + margin_m)
    for side in (0, 1):
        cols = team == side
        crowded = np.isfinite(x[:, cols]).sum(axis=1) > TEAM_MAX
        drop = off & cols[None, :] & crowded[:, None]
        out[drop] = np.nan
    return out


def keep_nearer_keeper(players, team, gk, home_defends_high, pitch, margin_m):
    """drop_misplaced_keepers, except that the misplaced track is not always the wrong one.

    A goalkeeper track deep in the half its team attacks is on that half's goalkeeper: the tracker has
    given him the other keeper's slot. The half's own keeper slot is then usually on someone else nearby.
    Of the two, the goalkeeper is the one nearer the goal. So over each stretch a keeper slot is misplaced,
    if it stands nearer the goal than the half's own keeper slot on most frames (or that slot is unseen),
    its positions move into that slot; either way the misplaced slot itself is blanked."""
    out = players.copy()
    half_x, width = pitch[0] / 2, pitch[1]
    slots = np.flatnonzero(gk)
    if len(slots) != 2 or team[slots[0]] == team[slots[1]]:
        return drop_misplaced_keepers(players, team, gk, home_defends_high, pitch, margin_m)
    for k, other in ((slots[0], slots[1]), (slots[1], slots[0])):
        own_high = home_defends_high if team[k] == 0 else ~home_defends_high
        with np.errstate(invalid="ignore"):
            depth = np.where(own_high, half_x - players[:, k, 0], players[:, k, 0] - half_x)
            wrong = depth > margin_m
        goal_x = np.where(own_high, 0.0, pitch[0])            # the goal of the half the slot is standing in
        with np.errstate(invalid="ignore"):
            d_k = np.hypot(players[:, k, 0] - goal_x, players[:, k, 1] - width / 2)
            d_o = np.hypot(players[:, other, 0] - goal_x, players[:, other, 1] - width / 2)
        nearer = wrong & ~(d_o <= d_k)                         # NaN d_o (slot unseen) counts as nearer
        edges = np.flatnonzero(np.diff(np.r_[0, wrong.astype(np.int8), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if nearer[a:b].mean() > 0.5:
                out[a:b, other] = players[a:b, k]
            out[a:b, k] = np.nan
    return out


def drop_misplaced_keepers(players, team, gk, home_defends_high, pitch, margin_m):
    """Blank a goalkeeper track where it stands more than margin_m inside the half its team attacks.

    A keeper guards his own goal; deep in the other half the tracker has put his slot on another
    player, usually the other keeper (2026-11-16: 18k graded frames, in stretches of up to 76 s).
    home_defends_high: (N,) True where the home team defends the goal at x = pitch length.
    """
    out = players.copy()
    half_x = pitch[0] / 2
    for k in np.flatnonzero(gk):
        own_high = home_defends_high if team[k] == 0 else ~home_defends_high
        with np.errstate(invalid="ignore"):
            depth = np.where(own_high, half_x - players[:, k, 0], players[:, k, 0] - half_x)
            out[depth > margin_m, k] = np.nan
    return out


def relink_cross_team(players, team, window, near_m):
    """Give a stretch back to the opponent it follows.

    The tracker loses a player and a new slot of the OTHER team picks him up: from that frame one team
    shows 12 tracks and the other 10. Where a track starts within near_m of an opponent's track that
    ended in the last `window` frames, and its team is one over while the other is one short, its
    positions move into the ended slot until the counts are back to normal (or it ends).
    Returns (players, frames moved).
    """
    out = players.copy()
    seen = np.isfinite(out[..., 0])
    count = {s: seen[:, team == s].sum(axis=1).astype(int) for s in (0, 1)}
    starts = []
    for k in range(len(team)):
        starts += [(a, k, z) for a, z in runs(seen[:, k]) if a > 0]
    moved = 0
    for a, k, z in sorted(starts):
        s, o = team[k], 1 - team[k]
        if not (np.isfinite(out[a, k, 0]) and count[s][a] > TEAM_MAX and count[o][a] < TEAM_MAX):
            continue
        best = None
        for j in np.flatnonzero(team == o):
            before = np.flatnonzero(np.isfinite(out[max(0, a - window):a, j, 0]))
            if not len(before) or np.isfinite(out[a, j, 0]):
                continue
            t = max(0, a - window) + before[-1]
            d = np.hypot(*(out[t, j, :2] - out[a, k, :2]))
            if d < near_m and (best is None or d < best[1]):
                best = (j, d)
        if best is None:
            continue
        e = a
        while e + 1 <= z and count[s][e + 1] > TEAM_MAX and count[o][e + 1] < TEAM_MAX:
            e += 1
        j = best[0]
        if np.isfinite(out[a:e + 1, j, 0]).any():
            continue
        out[a:e + 1, j] = out[a:e + 1, k]
        out[a:e + 1, k] = np.nan
        count[s][a:e + 1] -= 1
        count[o][a:e + 1] += 1
        moved += e - a + 1
    return out, moved


def broadcast_team_fix(players, team, bxy, bteam, match_m, min_frames, min_share):
    """Move a track to the other team where the broadcast tracker says it belongs there.

    Only in frames where the track's team shows more than 11 and the other fewer than 11. A track gets
    a vote in such a frame when the nearest broadcast player within match_m is of the other team and
    it is the only track of its team for which that holds. A stretch (track seen, counts 12 against
    10) with at least min_frames votes, and votes on at least min_share of its broadcast-covered
    frames, moves into a free slot of the other team.
    bxy: (N, Sb, 2) broadcast players on the tactical timeline, NaN where no live broadcast frame.
    Returns (players, frames moved).
    """
    out = players.copy()
    seen = np.isfinite(out[..., 0])
    count = {s: seen[:, team == s].sum(axis=1).astype(int) for s in (0, 1)}
    covered = np.isfinite(bxy[..., 0]).any(axis=1)
    moved = 0
    for s in (0, 1):
        o = 1 - s
        cols = np.flatnonzero(team == s)
        frames = np.flatnonzero((count[s] > TEAM_MAX) & (count[o] < TEAM_MAX) & covered)
        votes = np.zeros((len(out), len(cols)), bool)
        for i in range(0, len(frames), 2000):
            f = frames[i:i + 2000]
            t, b = out[f][:, cols, :2], bxy[f]
            d = np.hypot(t[:, :, None, 0] - b[:, None, :, 0], t[:, :, None, 1] - b[:, None, :, 1])
            d = np.where(np.isfinite(d), d, np.inf)
            j = np.argmin(d, axis=2)
            other = (np.take_along_axis(d, j[..., None], axis=2)[..., 0] < match_m) & (bteam[j] == o)
            votes[f] = other & (other.sum(axis=1, keepdims=True) == 1)
        for c, k in enumerate(cols):
            for a, z in runs(np.isfinite(out[:, k, 0]) & (count[s] > TEAM_MAX) & (count[o] < TEAM_MAX)):
                v, cov = votes[a:z + 1, c].sum(), covered[a:z + 1].sum()
                if v < min_frames or v < min_share * cov:
                    continue
                free = [j for j in np.flatnonzero(team == o) if not np.isfinite(out[a:z + 1, j, 0]).any()]
                if not free:
                    continue
                out[a:z + 1, free[0]] = out[a:z + 1, k]
                out[a:z + 1, k] = np.nan
                count[s][a:z + 1] -= 1
                count[o][a:z + 1] += 1
                moved += z - a + 1
    return out, moved


def fuse_cameras(players, team, bplayers, bteam, gate_m, bunch_m, w_bunched):
    """One set of players per frame from both cameras (our rule, 2026-11-17).

    Neither camera keeps one track per person for the whole match (a broadcast track's partner in the
    tactical data changes all the time), so the cameras are joined frame by frame, by position:
      - a player both cameras see (paired one to one within gate_m): the two positions are averaged with
        equal weight, or with w_bunched on the broadcast when another player is within bunch_m of him.
        Measured on this match: in a bunch the tactical tracking lacks 11.0% of the players the broadcast
        has, the broadcast lacks 6.4% of the tactical ones; for a free player it is 4.9% against 12.8%.
      - a player only the broadcast sees: added to an empty track of his team, if the team shows fewer
        than 11. He keeps that track while the broadcast keeps seeing him.
      - a player only the tactical camera sees, or a frame with no live broadcast view: unchanged.
    The team of a track is never changed here.
    bplayers: (N, Sb, 3) broadcast x / y / speed on the tactical timeline, NaN where no live frame.
    Returns (players, source) with source (N, S): 0 nobody, 1 tactical only, 2 broadcast only, 3 both.
    """
    from camera_agreement import match, nearest_other
    out = players.copy()
    seen = np.isfinite(players[..., 0])
    source = seen.astype(np.uint8)
    n = len(players)
    gap = np.full(seen.shape, n, np.int32)      # frames to the nearest frame the tactical data uses the track
    frame = np.arange(n)
    for s in range(seen.shape[1]):
        idx = np.flatnonzero(seen[:, s])
        if len(idx):
            k = np.searchsorted(idx, frame)
            gap[:, s] = np.minimum(np.abs(frame - idx[np.clip(k - 1, 0, len(idx) - 1)]),
                                   np.abs(idx[np.clip(k, 0, len(idx) - 1)] - frame))
    t_ok, b_ok = np.isin(team, (0, 1)), np.isin(bteam, (0, 1))
    held = {}                                   # broadcast track -> (tactical track, last frame it held it)
    for f in np.flatnonzero(np.isfinite(bplayers[..., 0]).any(axis=1)):
        bi = np.flatnonzero(b_ok & np.isfinite(bplayers[f, :, 0]))
        ti = np.flatnonzero(t_ok & seen[f])
        b, t = bplayers[f, bi, :2], players[f, ti, :2]
        r, c, _ = match(b, t, gate_m)
        if len(r):
            bunched = (nearest_other(b)[r] < bunch_m) | (nearest_other(t)[c] < bunch_m)
            w = np.where(bunched, w_bunched, 0.5)[:, None]
            out[f, ti[c], :2] = (1 - w) * t[c] + w * b[r]
            source[f, ti[c]] = 3
        count = {side: int(seen[f, team == side].sum()) for side in (0, 1)}
        for j in np.setdiff1d(np.arange(len(bi)), r):
            side = int(bteam[bi[j]])
            if count[side] >= TEAM_MAX:
                continue
            slot, last = held.get(int(bi[j]), (-1, -1))
            if last != f - 1 or source[f, slot]:
                free = np.flatnonzero((team == side) & (source[f] == 0))
                if not len(free):
                    continue
                slot = int(free[np.argmax(gap[f, free])])
            out[f, slot] = bplayers[f, bi[j]]
            source[f, slot] = 2
            held[int(bi[j])] = (slot, f)
            count[side] += 1
    return out, source


def camera_sources(fused, before):
    """source of fuse_cameras, read back from the tracking before and after it (same code as there)."""
    a, b = np.isfinite(fused[..., 0]), np.isfinite(before[..., 0])
    moved = a & b & np.any(fused[..., :2] != before[..., :2], axis=-1)
    return np.where(a & ~b, 2, np.where(moved, 3, a.astype(int))).astype(np.uint8)


def broadcast_players_on_tactical(players, team, broadcast, flipped, cols=2):
    """(N, Sb, 2) broadcast player positions on the tactical timeline (NaN unless a live, synced
    broadcast frame maps there), and the broadcast slots' teams."""
    from feature_cache import broadcast_live, turn_broadcast
    bp, bb, bteam = broadcast
    bp, _ = turn_broadcast(bp, bb, flipped)
    live, tframe = broadcast_live(bp, bteam, players, team)
    ok = live & (tframe >= 0)
    bxy = np.full((len(players), bp.shape[1], cols), np.nan, dtype=np.float32)
    bxy[tframe[ok]] = bp[ok, :, :cols]
    return bxy, bteam


def drop_ghosts(players, team, near_m, frames, dv_mps):
    """Remove duplicate tracks: a teammate copy within near_m, moving the same way, for `frames`
    frames, while the team shows more than 11 tracks. The track seen less overall loses the overlap."""
    from features import central_velocity
    out = players.copy()
    xy = players[..., :2]
    vel = central_velocity(xy, 3)
    seen = np.isfinite(xy[..., 0])
    lifetime = seen.sum(axis=0)
    for side in (0, 1):
        cols = np.flatnonzero(team == side)
        crowded = seen[:, cols].sum(axis=1) > TEAM_MAX
        if not crowded.any():
            continue
        rows = np.flatnonzero(crowded)
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                i, j = cols[a], cols[b]
                d = np.hypot(*(xy[rows, i] - xy[rows, j]).T)
                dv = np.hypot(*(vel[rows, i] - vel[rows, j]).T)
                close = np.zeros(len(players), bool)
                with np.errstate(invalid="ignore"):
                    # at a track's first and last frames velocity is unknown: judge on distance alone
                    close[rows] = (d < near_m) & ~(dv >= dv_mps)
                loser = j if lifetime[j] <= lifetime[i] else i
                for s, e in runs(close):
                    if e - s + 1 >= frames:
                        out[s:e + 1, loser] = np.nan
    return out


# ---------- gaps and smoothing ----------

def fill_short(xy, max_frames, blocks):
    """Fill NaN runs of at most max_frames on a straight line, only between two seen points in one half."""
    out = xy.copy()
    flat = out.reshape(len(out), -1, 2)
    for _, first, last in blocks:
        for k in range(flat.shape[1]):
            seg = flat[first:last + 1, k]
            for s, e in runs(~np.isfinite(seg[:, 0])):
                if s == 0 or e == len(seg) - 1 or e - s + 1 > max_frames:
                    continue
                w = (np.arange(1, e - s + 2) / (e - s + 2))[:, None]
                seg[s:e + 1] = seg[s - 1] + w * (seg[e + 1] - seg[s - 1])
    return out


def smooth_tracks(xy, window, order, blocks):
    """Savitzky-Golay (centred, so no time shift) per NaN-free run inside a half; short runs stay."""
    from scipy.signal import savgol_filter
    out = xy.copy()
    flat = out.reshape(len(out), -1, 2)
    for _, first, last in blocks:
        for k in range(flat.shape[1]):
            seg = flat[first:last + 1, k]
            for s, e in runs(np.isfinite(seg[:, 0])):
                if e - s + 1 >= window:
                    seg[s:e + 1] = savgol_filter(seg[s:e + 1], window, order, axis=0)
    return out


def speed_from_positions(players, half=3):
    """(N, S) speed in m/s from a central difference over `half` frames of the cleaned positions."""
    from features import central_velocity
    v = central_velocity(players[..., :2], half)
    return np.hypot(v[..., 0], v[..., 1]).astype(np.float32)


# ---------- one version ----------

def _changes(before, after):
    was, now = np.isfinite(before[..., 0]), np.isfinite(after[..., 0])
    both = was & now
    moved = np.zeros_like(both)
    moved[both] = np.hypot(*(after[both] - before[both]).T) > MOVED_M
    return {"removed": int(np.sum(was & ~now)), "added": int(np.sum(~was & now)), "moved": int(np.sum(moved))}


def broadcast_ball_fill(ball, players, team, blocks, pitch, broadcast, flipped, min_run=1, seam_m=None):
    """The tactical ball with its gaps filled from the broadcast camera, where a live (synced, not a
    replay) broadcast frame sees the ball for at least min_run frames in a row.
    seam_m: a filled stretch that touches a tactical sighting must start or end within this of it. The two
    cameras put a ball in flight at different spots (over 5 m apart on 48% of frames above 20 m/s), so a
    stretch that does not join on is the ball in the air, or another ball, and is left out.
    broadcast = (players, ball, team) as load("broadcast")."""
    from broadcast_gaps import broadcast_on_tactical
    from feature_cache import turn_broadcast
    bp, bb, bteam = broadcast
    bp, bb = turn_broadcast(bp, bb, flipped)
    bball = clip_bounds(broadcast_on_tactical(players, team, bp, bb, bteam)[0], pitch, 10.0)
    inside = np.zeros(len(ball), bool)
    for _, first, last in blocks:
        inside[first:last + 1] = True
    take = inside & ~np.isfinite(ball[:, 0]) & np.isfinite(bball[:, 0])
    if min_run > 1:
        for s, e in runs(take):
            if e - s + 1 < min_run:
                take[s:e + 1] = False
    if seam_m is not None:
        for s, e in runs(take):
            before = np.hypot(*(ball[s - 1] - bball[s])) if s > 0 and np.isfinite(ball[s - 1, 0]) else 0.0
            after = np.hypot(*(ball[e + 1] - bball[e])) if e + 1 < len(ball) and np.isfinite(ball[e + 1, 0]) else 0.0
            if max(before, after) > seam_m:
                take[s:e + 1] = False
    out = ball.copy()
    out[take] = bball[take]
    return out


def clean_cfg(players, ball, team, blocks, pitch, cfg, broadcast=None):
    """(players, ball, summary) for one configuration. Steps run in one fixed order; summary is counts only.
    broadcast (load("broadcast")) is needed only by versions with a "broadcast" step."""
    p, b = players.astype(np.float32, copy=True), ball.astype(np.float32, copy=True)
    summary = {"seen_players": int(np.isfinite(p[..., 0]).sum()), "seen_ball": int(np.isfinite(b[:, 0]).sum())}

    def on_xy(f):
        def g(a):
            out = a.copy()
            out[..., :2] = f(a[..., :2])
            out[~np.isfinite(out[..., 0]), 2] = np.nan
            return out
        return g

    def step(name, fp, fb):
        nonlocal p, b
        new_p, new_b = fp(p), fb(b)
        summary[name] = {"players": _changes(p[..., :2], new_p[..., :2]), "ball": _changes(b, new_b)}
        p, b = new_p, new_b

    keep = lambda a: a  # noqa: E731
    flipped = None
    if cfg["orient"]:
        p, b, flipped = orient(p, b, team, blocks, pitch)
        summary["flipped"] = {str(k): v for k, v in flipped.items()}
    if cfg["bounds"]:
        c = cfg["bounds"]
        step("bounds", on_xy(lambda xy: clip_bounds(xy, pitch, c["player_m"])), lambda a: clip_bounds(a, pitch, c["ball_m"]))
    if cfg["caps"]:
        c = cfg["caps"]
        step("caps", on_xy(lambda xy: cap_speed(xy, c["player_mps"], c["jump_gap"])),
             (lambda a: cap_speed(a, c["ball_mps"], c.get("ball_jump_gap", c["jump_gap"]))) if c.get("ball_mps") else keep)
    if cfg.get("keepers"):      # .get: versions made before this step existed keep their hash
        from tracking import goalkeepers
        if cfg["orient"]:       # turned so home attacks towards x = length: it defends x = 0 throughout
            high = np.zeros(len(p), bool)
        else:                   # flipped_halves: True where home attacks towards x = 0, so defends high x
            high = np.vectorize(flipped_halves(p, team, blocks).get)(half_of_frame(len(p), blocks)).astype(bool)
        fix = keep_nearer_keeper if cfg["keepers"].get("nearer_goal") else drop_misplaced_keepers
        step("keepers", lambda a: fix(a, team, goalkeepers("tactical"), high, pitch, cfg["keepers"]["margin_m"]), keep)
    if cfg.get("relink"):
        c = cfg["relink"]
        step("relink", lambda a: relink_cross_team(a, team, c["window"], c["near_m"])[0], keep)
    if cfg["ghosts"]:
        c = cfg["ghosts"]
        step("ghosts", lambda a: drop_ghosts(a, team, c["near_m"], c["frames"], c["dv_mps"]), keep)
    if cfg.get("extras"):
        step("extras", lambda a: drop_offpitch_extras(a, team, pitch, cfg["extras"]["off_m"]), keep)
    if cfg.get("teamcheck"):
        c = cfg["teamcheck"]
        bxy, bteam = broadcast_players_on_tactical(p, team, broadcast, flipped)
        step("teamcheck", lambda a: broadcast_team_fix(a, team, bxy, bteam, c["match_m"], c["min_frames"],
                                                       c["min_share"])[0], keep)
    if cfg.get("fuse"):
        c = cfg["fuse"]
        bxyz, bteam = broadcast_players_on_tactical(p, team, broadcast, flipped, cols=3)
        step("fuse", lambda a: fuse_cameras(a, team, bxyz, bteam, c["gate_m"], c["bunch_m"], c["w_bunched"])[0], keep)
    if cfg.get("broadcast"):    # .get: versions made before this step existed keep their hash
        min_run = cfg["broadcast"]["min_run"] if isinstance(cfg["broadcast"], dict) else 1
        seam = cfg["broadcast"].get("seam_m") if isinstance(cfg["broadcast"], dict) else None
        step("broadcast", keep, lambda a: broadcast_ball_fill(a, p, team, blocks, pitch, broadcast, flipped, min_run, seam))
    if cfg["fill"]:
        c = cfg["fill"]
        step("fill", on_xy(lambda xy: fill_short(xy, c["player"], blocks)) if c.get("player") else keep,
             (lambda a: fill_short(a, c["ball"], blocks)) if c.get("ball") else keep)
    if cfg["smooth"]:
        c = cfg["smooth"]
        step("smooth", on_xy(lambda xy: smooth_tracks(xy, c["player"], c["order"], blocks)) if c.get("player") else keep,
             (lambda a: smooth_tracks(a, c["ball"], c["order"], blocks)) if c.get("ball") else keep)
    if cfg["speed"] == "positions":
        p = p.copy()
        p[..., 2] = speed_from_positions(p)
        p[~np.isfinite(p[..., 0]), 2] = np.nan
    return p, b, summary


def clean(players, ball, team, blocks, pitch, version, broadcast=None):
    p, b, summary = clean_cfg(players, ball, team, blocks, pitch, VERSIONS[version], broadcast)
    summary.update(version=version, hash=params_hash(version))
    return p, b, summary


class feature_params:
    """with feature_params(VERSIONS[v]["feat"]): features.py runs with that version's overrides."""

    def __init__(self, overrides):
        import features
        self.module, self.overrides, self.saved = features, overrides, {}

    def __enter__(self):
        for k, v in self.overrides.items():
            if not hasattr(self.module, k):
                raise KeyError("features.py has no parameter " + k)
            self.saved[k] = getattr(self.module, k)
            setattr(self.module, k, v)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.module, k, v)


# ---------- self-test ----------

def _toy_match(n=600):
    """Two halves of 300 frames; 11 v 11 (plus one empty slot each) walking 1 m/s; home defends x = 0
    in half 1 and x = 105 in half 2."""
    rng = np.random.default_rng(0)
    team = np.array([0] * 12 + [1] * 12)
    players = np.full((n, 24, 3), np.nan, np.float32)
    blocks = [(1, 0, n // 2 - 1), (2, n // 2, n - 1)]
    for half, first, last in blocks:
        left = half == 1
        for k in range(11):
            players[first:last + 1, k, 0] = 5.0 + 5 * k if left else 100.0 - 5 * k
            players[first:last + 1, 12 + k, 0] = 100.0 - 5 * k if left else 5.0 + 5 * k
            players[first:last + 1, k, 1] = players[first:last + 1, 12 + k, 1] = 5.0 + 5 * k
    players[:, :, 1] += (np.arange(n) * 0.04)[:, None]
    players[:, :, :2] += rng.normal(0, 0.005, (n, 24, 2)).astype(np.float32)
    players[:, :, 2] = 1.0
    ball = np.stack([np.mod(52.5 + np.arange(n) * 0.2, 100.0), np.full(n, 34.0)], axis=1).astype(np.float32)
    return players, ball, team, blocks


def _self_test():
    pitch = (105.0, 68.0)
    players, ball, team, blocks = _toy_match()

    p, b, flipped = orient(players, ball, team, blocks, pitch)
    assert flipped == {1: False, 2: True}, flipped
    p2, _, f2 = orient(p, b, team, blocks, pitch)
    assert f2 == {1: False, 2: False} and np.array_equal(p2, p, equal_nan=True)
    assert np.allclose(p[400, 0, :2], [pitch[0] - players[400, 0, 0], pitch[1] - players[400, 0, 1]])
    amb = players.copy()
    amb[:, 12:, 0] = amb[:, :12, 0] + 4.0
    try:
        orient(amb, ball, team, blocks, pitch)
        raise AssertionError("orient guessed a direction it cannot know")
    except ValueError:
        pass

    xy = np.array([[112.0, 30.0]], np.float32)
    assert np.isfinite(clip_bounds(xy, pitch, 10.0)).all() and np.isnan(clip_bounds(xy, pitch, 5.0)).all()

    walk = np.stack([np.arange(40) * 0.04, np.zeros(40)], axis=1).astype(np.float32)
    spiky = walk.copy()
    spiky[10, 1] = 5.0
    out = cap_speed(spiky, 12.0, 6)
    assert np.isnan(out[10]).all() and np.isfinite(out[np.r_[0:10, 11:40]]).all()
    jumpy = walk.copy()
    jumpy[20:, 1] += 20.0
    out = cap_speed(jumpy, 12.0, 6)
    assert np.isnan(out[20:26]).all() and np.isfinite(out[:20]).all() and np.isfinite(out[26:]).all()
    lone = np.full((10, 2), np.nan, np.float32)
    lone[4] = [3.0, 3.0]
    assert np.array_equal(cap_speed(lone, 12.0, 6), lone, equal_nan=True)
    assert np.array_equal(fill_short(lone, 25, [(1, 0, 9)]), lone, equal_nan=True)

    wall = players.copy()
    wall[:, 1:4, 0] = wall[:, 0:1, 0] + 0.5 * np.arange(1, 4)
    wall[:, 1:4, 1] = wall[:, 0:1, 1]
    assert np.array_equal(drop_ghosts(wall, team, 0.3, 25, 0.3), wall, equal_nan=True)
    ghost = players.copy()
    ghost[100:150, 11, :2] = ghost[100:150, 0, :2] + 0.1
    ghost[100:150, 11, 2] = 1.0
    out = drop_ghosts(ghost, team, 0.3, 25, 0.3)
    assert np.isnan(out[100:150, 11, 0]).all() and np.isfinite(out[100:150, :11, 0]).sum(axis=1).min() == 11

    # keepers: a home keeper slot 20 m into the half home attacks is blanked, one at its own goal stays
    gk = np.zeros(len(team), bool)
    gk[0] = True
    kp = players[:10].copy()
    kp[:5, 0, 0] = pitch[0] / 2 + 20.0
    kp[5:, 0, 0] = 5.0
    out = drop_misplaced_keepers(kp, team, gk, np.zeros(10, bool), pitch, 10.0)
    assert np.isnan(out[:5, 0, 0]).all() and np.isfinite(out[5:, 0, 0]).all()
    out = drop_misplaced_keepers(kp, team, gk, np.ones(10, bool), pitch, 10.0)   # defending high x: the near-goal one is wrong
    assert np.isfinite(out[:5, 0, 0]).all() and np.isnan(out[5:, 0, 0]).all()

    # K6: the away keeper slot (12) stands at home's goal, nearer it than home's own keeper slot (0): it IS
    # home's keeper, so its positions move into slot 0; when it is the farther one it is just blanked
    gk2 = np.zeros(len(team), bool)
    gk2[[0, int(np.flatnonzero(team == 1)[0])]] = True
    a_k = int(np.flatnonzero(team == 1)[0])
    kp = players[:10].copy()
    kp[:, 0, :2] = [12.0, pitch[1] / 2]
    kp[:, a_k, :2] = [3.0, pitch[1] / 2]
    out = keep_nearer_keeper(kp, team, gk2, np.zeros(10, bool), pitch, 10.0)
    assert np.allclose(out[:, 0, :2], [3.0, pitch[1] / 2]) and np.isnan(out[:, a_k, 0]).all()
    kp[:, a_k, 0] = 20.0
    out = keep_nearer_keeper(kp, team, gk2, np.zeros(10, bool), pitch, 10.0)
    assert np.allclose(out[:, 0, 0], 12.0) and np.isnan(out[:, a_k, 0]).all()

    # relink: away slot 12 is lost at frame 100 and the spare home slot 11 picks him up at 103 (home 12,
    # away 10) until 150; the stretch goes back to slot 12. Picked up 3 m away: left alone.
    lost = players.copy()
    lost[103:151, 11, :] = lost[103:151, 12, :]
    lost[101:151, 12, :] = np.nan
    out, moved = relink_cross_team(lost, team, 25, 2.0)
    assert moved == 48 and np.isnan(out[103:151, 11, 0]).all(), moved
    assert np.allclose(out[103:151, 12, :2], players[103:151, 12, :2])
    far = lost.copy()
    far[103:151, 11, 0] += 3.0
    assert relink_cross_team(far, team, 25, 2.0)[1] == 0

    # teamcheck: away player 12 is tracked by the spare home slot 11 for frames 100-199 (home 12, away 10);
    # the broadcast sees him as away, so the stretch moves to a free away slot. No broadcast: untouched.
    wrong = players.copy()
    wrong[100:200, 11, :] = wrong[100:200, 12, :]
    wrong[100:200, 12, :] = np.nan
    bteam = np.array([0] * 11 + [1] * 11)
    bxy = np.concatenate([players[:, :11, :2], players[:, 12:23, :2]], axis=1)
    out, moved = broadcast_team_fix(wrong, team, bxy, bteam, 1.5, 25, 0.5)
    assert moved == 100 and np.isnan(out[100:200, 11, 0]).all(), moved
    assert np.isfinite(out[100:200, team == 1, 0]).sum(axis=1).min() == 11
    assert broadcast_team_fix(wrong, team, np.full_like(bxy, np.nan), bteam, 1.5, 25, 0.5)[1] == 0

    # extras: a 12th home track 3 m off the touchline goes; an 11-track team keeps its player off the line
    early = players[:40].copy()
    early[:, 11, :2] = [50.0, -3.0]
    early[:, 11, 2] = 1.0
    out = drop_offpitch_extras(early, team, pitch, 1.0)
    assert np.isnan(out[:, 11, 0]).all() and np.isfinite(out[:, :11, 0]).all()
    alone = players[:40].copy()
    alone[:, 0, :2] = [50.0, -3.0]
    assert np.array_equal(drop_offpitch_extras(alone, team, pitch, 1.0), alone, equal_nan=True)

    gappy = walk.copy()
    gappy[5:15] = np.nan
    gappy[20:33] = np.nan
    out = fill_short(gappy, 12, [(1, 0, 39)])
    assert np.allclose(out[5:15], walk[5:15], atol=1e-5) and np.isnan(out[20:33]).all()
    across = walk.copy()
    across[18:23] = np.nan
    assert np.isnan(fill_short(across, 12, [(1, 0, 19), (2, 20, 39)])[18:23]).all()
    start = walk.copy()
    start[:3] = np.nan
    assert np.isnan(fill_short(start, 12, [(1, 0, 39)])[:3]).all()

    rng = np.random.default_rng(1)
    noisy = walk + rng.normal(0, 0.2, walk.shape).astype(np.float32)
    jit = lambda a: np.std(np.diff(a[:, 1], 2))  # noqa: E731
    assert jit(smooth_tracks(noisy, 7, 2, [(1, 0, 39)])) < 0.5 * jit(noisy)
    stepped = np.zeros((100, 2), np.float32)
    stepped[50:, 0] = 10.0
    sm = smooth_tracks(stepped, 15, 2, [(1, 0, 99)])
    assert abs(sm[49, 0] + sm[50, 0] - 10.0) < 0.5 and sm[45, 0] < 5.0 < sm[54, 0]

    run = np.zeros((20, 1, 3), np.float32)
    run[:, 0, 0] = np.arange(20) * 0.2
    assert abs(speed_from_positions(run)[10, 0] - 5.0) < 1e-3

    full = _then(V01, orient=True, bounds={"player_m": 8.0, "ball_m": 10.0},
                 caps={"player_mps": 12.0, "ball_mps": 40.0, "jump_gap": 6},
                 ghosts={"near_m": 0.3, "frames": 25, "dv_mps": 0.3}, fill={"ball": 12, "player": 25},
                 smooth={"player": 7, "ball": 7, "order": 2}, speed="positions")
    p0, b0, _ = clean_cfg(players, ball, team, blocks, pitch, V01)
    assert np.array_equal(p0, players, equal_nan=True) and np.array_equal(b0, ball, equal_nan=True)
    x = clean_cfg(players, ball, team, blocks, pitch, full)
    y = clean_cfg(players, ball, team, blocks, pitch, full)
    assert content_hash({"p": x[0], "b": x[1]}) == content_hash({"p": y[0], "b": y[1]})
    assert len({params_hash(v) for v in VERSIONS}) == len(VERSIONS)

    import features
    before = features.FOLLOW_RATE
    with feature_params({"FOLLOW_RATE": 0.3}):
        assert features.FOLLOW_RATE == 0.3
    assert features.FOLLOW_RATE == before

    # fuse: a free player is averaged evenly, a bunched one 3 to 1 for the broadcast, a player only the
    # broadcast sees is added and keeps his track, and a full team takes nobody
    tm = np.array([0, 0, 0, 1, 1])
    tp = np.full((3, 5, 3), np.nan, np.float32)
    tp[:, 0, :2], tp[:, 1, :2], tp[:, 3, :2] = [10.0, 10.0], [30.0, 30.0], [31.0, 30.0]
    bt = np.array([0, 0, 1, 1])
    bp = np.full((3, 4, 3), np.nan, np.float32)
    bp[:2, 0, :2], bp[:2, 1, :2], bp[:2, 2, :2], bp[:2, 3, :2] = [10.4, 10.0], [30.4, 30.0], [31.0, 30.4], [50.0, 50.0]
    fused, src = fuse_cameras(tp, tm, bp, bt, 3.0, 1.5, 0.75)
    assert np.allclose(fused[0, 0, :2], [10.2, 10.0]) and np.allclose(fused[0, 1, :2], [30.3, 30.0])
    assert np.allclose(fused[:2, 4, :2], [50.0, 50.0]) and np.isnan(fused[2, 4, 0])
    assert src[0].tolist() == [3, 3, 0, 3, 2] and src[2].tolist() == [1, 1, 0, 1, 0]
    assert np.array_equal(camera_sources(fused, tp), src)
    # broadcast ball fill: a stretch that joins on is taken, one that lands 10 m from the tactical ball is not
    tb = np.full((120, 2), np.nan, np.float32)
    tb[:30], tb[90:] = [50.0, 30.0], [50.0, 30.0]
    bb = np.full((120, 2), np.nan, np.float32)
    bb[30:60] = [51.0, 30.0]
    bb[60:90] = [60.0, 30.0]
    import broadcast_gaps
    real = broadcast_gaps.broadcast_on_tactical
    broadcast_gaps.broadcast_on_tactical = lambda *a: (bb, None, None, None)
    try:
        got = broadcast_ball_fill(tb, None, None, [(1, 0, 119)], pitch, (None, bb, None), None, 25, 3.0)
        assert np.isfinite(got[:, 0]).sum() == 60, "the stretch 10 m off the tactical ball must be left out"
        bb[60:90] = [51.0, 30.0]
        assert np.isfinite(broadcast_ball_fill(tb, None, None, [(1, 0, 119)], pitch, (None, bb, None), None, 25, 3.0)[:, 0]).all()
    finally:
        broadcast_gaps.broadcast_on_tactical = real

    global TEAM_MAX
    TEAM_MAX, old = 1, TEAM_MAX
    assert not (fuse_cameras(tp, tm, bp, bt, 3.0, 1.5, 0.75)[1] == 2).any()
    TEAM_MAX = old


if __name__ == "__main__":
    _self_test()
    print("cleaning self-test passed")
