"""Where do the two cameras' tracking disagree?

The tactical and the broadcast tracking are two independent readings of the same players and ball.
On every graded tactical frame that a live, synced broadcast frame maps onto, this pairs the
players of the two cameras one to one (nearest first, any team) and counts what does not line up:

  - a broadcast player with no tactical player near him  (the tactical tracking lacks a player)
  - a tactical player inside the broadcast picture with no broadcast player near him
  - a pair that the two cameras put on different teams
  - how far apart a pair is, and whether that is a steady shift (the pitch mapping) or noise
  - the ball: seen by one camera only, or placed differently
  - the team of the player nearest the ball, camera by camera (what a possession model reads)

each split by how bunched the players are. Prints aggregates only.

    python experiments/camera_agreement.py                 # raw tactical (C0) and cleaned (K5)
    python experiments/camera_agreement.py 38640 38890     # the same, inside one tactical frame range
"""
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Params
GATE_M = 3.0            # two readings of one player are at most this far apart
VIEW_INSET_M = 3.0      # a tactical player this far inside the box around the broadcast players is "in the picture"
NEAR_BALL_M = 3.0       # a player this close to the ball can be the one on it
CROWD_BINS_M = (0.0, 1.5, 3.0, 1e9)   # distance to the nearest other player: bunched, close, free
BALL_FAR_M = (2.0, 5.0)


def match(b, t, gate=GATE_M):
    """One-to-one pairing of two point sets, least total distance, pairs over the gate dropped.
    Returns (rows of b, rows of t, distances)."""
    if not len(b) or not len(t):
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    d = np.hypot(b[:, None, 0] - t[None, :, 0], b[:, None, 1] - t[None, :, 1])
    r, c = linear_sum_assignment(np.where(d <= gate, d, 1e3))
    ok = d[r, c] <= gate
    return r[ok], c[ok], d[r, c][ok]


def nearest_other(xy):
    """Distance from each point to the nearest other point (inf for a lone point)."""
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1) if len(xy) else np.zeros(0)


def nearest_team(xy, team, ball):
    """Team of the player nearest the ball if within NEAR_BALL_M, else -1."""
    if not len(xy) or not np.isfinite(ball[0]):
        return -1
    d = np.hypot(xy[:, 0] - ball[0], xy[:, 1] - ball[1])
    i = int(np.argmin(d))
    return int(team[i]) if d[i] <= NEAR_BALL_M else -1


def broadcast_on_tactical_frames(tp, tteam, flipped):
    """Broadcast players (N, Sb, 2) and ball (N, 2) on the tactical timeline, turned like the tactical data."""
    from feature_cache import broadcast_live, turn_broadcast
    from tracking import load
    bp, bb, bteam = load("broadcast")
    bp, bb = turn_broadcast(bp, bb, flipped)
    live, tframe = broadcast_live(bp, bteam, tp, tteam)
    ok = live & (tframe >= 0)
    bxy = np.full((len(tp), bp.shape[1], 2), np.nan, np.float32)
    bball = np.full((len(tp), 2), np.nan, np.float32)
    on = np.zeros(len(tp), bool)
    bxy[tframe[ok]], bball[tframe[ok]], on[tframe[ok]] = bp[ok, :, :2], bb[ok], True
    return bxy, bball, bteam, on


def compare(tp, tb, tteam, bxy, bball, bteam, frames):
    """Counts over the given tactical frames. Everything returned is a total or a list of distances."""
    bins = len(CROWD_BINS_M) - 1
    out = {"frames": len(frames), "b_players": np.zeros(bins), "b_unmatched": np.zeros(bins),
           "pairs": np.zeros(bins), "team_differs": np.zeros(bins), "dist": [], "dx": [], "dy": [], "x": [],
           "t_in_view": 0, "t_unmatched": 0, "frames_b_extra": 0, "frames_team_differs": 0,
           "ball_both": 0, "ball_t_only": 0, "ball_b_only": 0, "ball_none": 0, "ball_dist": [],
           "frames_ball_b": 0, "lacks_near_ball": 0,
           "near_both": np.zeros(2), "near_differs": np.zeros(2), "near_t_only": 0, "near_b_only": 0}
    t_ok, b_ok = np.isin(tteam, (0, 1)), np.isin(bteam, (0, 1))
    for f in frames:
        b_sel = b_ok & np.isfinite(bxy[f, :, 0])
        t_sel = t_ok & np.isfinite(tp[f, :, 0])
        b, bt = bxy[f, b_sel], bteam[b_sel]
        t, tt = tp[f, t_sel, :2], tteam[t_sel]
        r, c, d = match(b, t)
        crowd = np.digitize(nearest_other(b), CROWD_BINS_M[1:-1])
        matched = np.zeros(len(b), bool)
        matched[r] = True
        differs = bt[r] != tt[c]
        np.add.at(out["b_players"], crowd, 1)
        np.add.at(out["b_unmatched"], crowd[~matched], 1)
        np.add.at(out["pairs"], crowd[r], 1)
        np.add.at(out["team_differs"], crowd[r][differs], 1)
        out["dist"].append(d)
        out["dx"].append(b[r, 0] - t[c, 0])
        out["dy"].append(b[r, 1] - t[c, 1])
        out["x"].append(t[c, 0])
        out["frames_b_extra"] += bool((~matched).any())
        out["frames_team_differs"] += bool(differs.any())
        if len(b) >= 4:     # the box around the broadcast players stands in for its picture
            lo, hi = b.min(axis=0) + VIEW_INSET_M, b.max(axis=0) - VIEW_INSET_M
            inside = np.all((t >= lo) & (t <= hi), axis=1)
            t_matched = np.zeros(len(t), bool)
            t_matched[c] = True
            out["t_in_view"] += int(inside.sum())
            out["t_unmatched"] += int((inside & ~t_matched).sum())
        ts, bs = np.isfinite(tb[f, 0]), np.isfinite(bball[f, 0])
        out["ball_both" if ts and bs else "ball_t_only" if ts else "ball_b_only" if bs else "ball_none"] += 1
        if ts and bs:
            out["ball_dist"].append(float(np.hypot(*(tb[f] - bball[f]))))
        if bs:
            out["frames_ball_b"] += 1
            near = np.hypot(b[:, 0] - bball[f, 0], b[:, 1] - bball[f, 1]) <= NEAR_BALL_M
            out["lacks_near_ball"] += bool((near & ~matched).any())
        nt, nb = nearest_team(t, tt, tb[f]), nearest_team(b, bt, bball[f])
        if nt >= 0 and nb >= 0:
            k = int(np.sum(np.hypot(t[:, 0] - tb[f, 0], t[:, 1] - tb[f, 1]) <= 5.0) >= 3)   # 3+ players within 5 m
            out["near_both"][k] += 1
            out["near_differs"][k] += nt != nb
        else:
            out["near_t_only"] += nt >= 0
            out["near_b_only"] += nb >= 0
    for k in ("dist", "dx", "dy", "x"):
        out[k] = np.concatenate(out[k]) if out[k] else np.zeros(0)
    out["ball_dist"] = np.array(out["ball_dist"])
    return out


def show(name, o, length):
    pct = lambda a, b: 100.0 * a / max(b, 1)  # noqa: E731
    labels = ["bunched (< {} m)".format(CROWD_BINS_M[1]), "close ({}-{} m)".format(*CROWD_BINS_M[1:3]), "free (> {} m)".format(CROWD_BINS_M[2])]
    print("\n== {}: {} tactical frames with a live broadcast view ==".format(name, o["frames"]))
    if not o["frames"]:
        return
    print("players the broadcast sees, by how near the next player is:")
    print("| | broadcast player-frames | no tactical player within {} m | paired, other team |".format(GATE_M))
    print("| --- | --- | --- | --- |")
    for i, lab in enumerate(labels):
        print("| {} | {:.0f} | {:.1f}% | {:.1f}% |".format(lab, o["b_players"][i], pct(o["b_unmatched"][i], o["b_players"][i]),
                                                       pct(o["team_differs"][i], o["pairs"][i])))
    print("| all | {:.0f} | {:.1f}% | {:.1f}% |".format(o["b_players"].sum(), pct(o["b_unmatched"].sum(), o["b_players"].sum()),
                                                    pct(o["team_differs"].sum(), o["pairs"].sum())))
    print("frames with at least one broadcast player the tactical lacks: {:.1f}%; with at least one pair on different teams: {:.1f}%".format(
        pct(o["frames_b_extra"], o["frames"]), pct(o["frames_team_differs"], o["frames"])))
    print("tactical players inside the broadcast picture with no broadcast player near: {:.1f}% of {}".format(
        pct(o["t_unmatched"], o["t_in_view"]), o["t_in_view"]))
    if len(o["dist"]):
        print("paired players are apart: median {:.2f} m, 90th pct {:.2f} m; steady shift x {:+.2f} m, y {:+.2f} m".format(
            np.median(o["dist"]), np.percentile(o["dist"], 90), o["dx"].mean(), o["dy"].mean()))
        third = np.digitize(o["x"], (length / 3, 2 * length / 3))
        print("  by third of the pitch (x): " + ", ".join(
            "median {:.2f} m, shift x {:+.2f}".format(np.median(o["dist"][third == k]), o["dx"][third == k].mean())
            for k in range(3) if np.any(third == k)))
    print("ball: both see it {:.1f}%, tactical only {:.1f}%, broadcast only {:.1f}%, neither {:.1f}%".format(
        *(pct(o[k], o["frames"]) for k in ("ball_both", "ball_t_only", "ball_b_only", "ball_none"))))
    if len(o["ball_dist"]):
        print("  where both see it they are apart: median {:.2f} m, 90th pct {:.2f} m; over {} m on {:.1f}%, over {} m on {:.1f}%".format(
            np.median(o["ball_dist"]), np.percentile(o["ball_dist"], 90), BALL_FAR_M[0], 100 * np.mean(o["ball_dist"] > BALL_FAR_M[0]),
            BALL_FAR_M[1], 100 * np.mean(o["ball_dist"] > BALL_FAR_M[1])))
    print("a broadcast player within {} m of the broadcast ball that the tactical lacks: {:.1f}% of {} frames".format(
        NEAR_BALL_M, pct(o["lacks_near_ball"], o["frames_ball_b"]), o["frames_ball_b"]))
    both = o["near_both"].sum()
    print("team of the player nearest the ball (within {} m, each camera with its own ball): both give one on {} frames;".format(NEAR_BALL_M, int(both)))
    print("  they name different teams on {:.1f}% ({:.1f}% with fewer than 3 players within 5 m, {:.1f}% with 3 or more)".format(
        pct(o["near_differs"].sum(), both), pct(o["near_differs"][0], o["near_both"][0]), pct(o["near_differs"][1], o["near_both"][1])))
    print("  only the tactical names one on {} frames, only the broadcast on {}".format(int(o["near_t_only"]), int(o["near_b_only"])))


def report(window=None):
    from cleaning import VERSIONS, flipped_halves
    from tracking import GAME, GAMES, half_video_blocks, load, load_clean
    blocks = half_video_blocks(GAME)
    length = GAMES[GAME]["pitch"][0]
    raw, _, team = load("tactical")
    for version in ("C0", "K5"):
        tp, tb, _ = load_clean(version)
        flipped = flipped_halves(raw, team, blocks) if version != "C0" and VERSIONS[version]["orient"] else None
        bxy, bball, bteam, on = broadcast_on_tactical_frames(tp, team, flipped)
        graded = np.zeros(len(tp), bool)
        for _, first, last in blocks:
            graded[first:last + 1] = True
        if window:
            graded &= (np.arange(len(tp)) >= window[0]) & (np.arange(len(tp)) <= window[1])
        show("{} tactical{}".format(version, " frames {}-{}".format(*window) if window else ""),
             compare(tp, tb, team, bxy, bball, bteam, np.flatnonzero(graded & on)), length)


def _self_test():
    t = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    b = np.array([[10.5, 0.0], [0.4, 0.0], [40.0, 0.0]])
    r, c, d = match(b, t)
    assert sorted(zip(r.tolist(), c.tolist())) == [(0, 1), (1, 0)] and np.allclose(sorted(d), [0.4, 0.5])
    assert len(match(np.zeros((0, 2)), t)[0]) == 0
    assert np.allclose(nearest_other(t), [10, 10, 10])
    assert nearest_team(t, np.array([0, 1, 0]), np.array([9.0, 0.0])) == 1
    assert nearest_team(t, np.array([0, 1, 0]), np.array([5.0, 0.0])) == -1
    # a tactical tracking that lacks the player on the ball: counted, and the cameras name different teams
    tp = np.full((1, 3, 3), np.nan, np.float32)
    tp[0, :2, :2] = [[0.0, 0.0], [1.0, 0.0]]
    bxy = np.array([[[0.0, 0.0], [1.0, 0.0], [0.5, 4.0]]], np.float32)
    tteam, bteam = np.array([0, 0, 1]), np.array([0, 0, 1])
    ball = np.array([[0.5, 3.5]], np.float32)
    o = compare(tp, ball, tteam, bxy, ball, bteam, np.array([0]))
    assert o["b_unmatched"].sum() == 1 and o["lacks_near_ball"] == 1 and o["team_differs"].sum() == 0
    assert o["near_b_only"] == 1 and o["near_both"].sum() == 0
    print("camera_agreement self-test passed")


if __name__ == "__main__":
    _self_test()
    report(tuple(int(a) for a in sys.argv[1:3]) if len(sys.argv) >= 3 else None)
