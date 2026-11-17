"""One team per track piece, by a vote of both cameras.

An unbroken track piece is one person (player_paths: a piece keeps a zone the way a real player
does), and one person cannot change team. The tactical tracking gives every track a team; the
broadcast tracking gives its own. Where the two cameras see the same player (paired one to one
within 3 m) the broadcast's team is a second opinion on the tactical one.

In a bunch the pairing itself can pick the wrong partner, so a single frame proves nothing there.
Over a whole piece it does: the frames where the player stands free (nobody else within FREE_M, so
the pairing cannot be wrong) are the votes.
  - the cameras agree on nearly all of them   -> the piece's team is right, and a disagreement in a
                                                 bunch is the pairing's mistake, not a wrong team;
  - they disagree on nearly all of them       -> the two cameras name different teams for this person;
  - they agree for a stretch, then disagree   -> the track jumped from one player to another as two
                                                 crossed (a swap), which an "unbroken" piece can hide.

    python experiments/piece_teams.py            # how many pieces of each kind, and how much play they touch
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Params
FPS = 25
FREE_M = 3.0                # nobody else within this: the pairing between the cameras cannot be mistaken
MIN_VOTES = 2 * FPS         # a piece needs this many free, paired frames (2 s) to be judged at all
AGREE, CONFLICT = 0.9, 0.1  # share of votes agreeing: at least AGREE = fine, at most CONFLICT = other team
NEAR_BALL_M = 3.0


def second_opinion(xy, team, bxy, bteam):
    """(N, S) the broadcast's team for each tactical track in each frame (-1 = not paired), and
    (N, S) True where the player stands free in both cameras' pictures."""
    from camera_agreement import match, nearest_other
    other = np.full(xy.shape[:2], -1, np.int8)
    free = np.zeros(xy.shape[:2], bool)
    t_ok, b_ok = np.isin(team, (0, 1)), np.isin(bteam, (0, 1))
    for f in np.flatnonzero(np.isfinite(bxy[..., 0]).any(axis=1)):
        ti = np.flatnonzero(t_ok & np.isfinite(xy[f, :, 0]))
        bi = np.flatnonzero(b_ok & np.isfinite(bxy[f, :, 0]))
        r, c, _ = match(bxy[f, bi], xy[f, ti])
        other[f, ti[c]] = bteam[bi[r]]
        free[f, ti[c]] = (nearest_other(xy[f, ti])[c] > FREE_M) & (nearest_other(bxy[f, bi])[r] > FREE_M)
    return other, free


def judge(pieces, team, other, free):
    """Per piece: (votes, share of them agreeing with the tactical team, first frame of a lasting
    disagreement or -1). kind is read from the share: fine / conflict / mixed / unjudged."""
    out = []
    for k, a, z in pieces:
        vote = free[a:z + 1, k] & (other[a:z + 1, k] >= 0)
        agree = other[a:z + 1, k][vote] == team[k]
        n = int(vote.sum())
        out.append((n, float(agree.mean()) if n else np.nan))
    return out


def kind(n, share):
    if n < MIN_VOTES:
        return "unjudged"
    return "fine" if share >= AGREE else "conflict" if share <= CONFLICT else "mixed"


SWAP_NEAR_M = 2.0            # two tracks can only trade players where they pass this close
SWAP_SLACK = 1 * FPS         # ... within this of the stretch between the last agreeing and first disagreeing vote


def wrong_stretches(pieces, team, other, free):
    """[(track, last agreeing vote frame or piece start, first frame of the stretch, last frame)]: inside a
    piece, a run of votes that nearly all name the other team (at least MIN_VOTES of them). The switch
    itself happened somewhere between the last agreeing vote and the first disagreeing one."""
    out = []
    for k, a, z in pieces:
        f = a + np.flatnonzero(free[a:z + 1, k] & (other[a:z + 1, k] >= 0))
        if len(f) < MIN_VOTES:
            continue
        bad = other[f, k] != team[k]
        edges = np.flatnonzero(np.diff(np.r_[False, bad, False].astype(int)))
        runs = [(i, j - 1) for i, j in zip(edges[::2], edges[1::2])]
        merged = []                                  # a few stray agreeing votes do not end a stretch
        for i, j in runs:
            if merged and bad[merged[-1][0]:j + 1].mean() >= 1 - CONFLICT:
                merged[-1] = (merged[-1][0], j)
            else:
                merged.append((i, j))
        for i, j in merged:
            if j - i + 1 >= MIN_VOTES and bad[i:j + 1].mean() >= 1 - CONFLICT:
                end = z if j == len(f) - 1 else int(f[j + 1]) - 1      # runs on to the next agreeing vote, or the piece's end
                out.append((k, int(f[i - 1]) if i > 0 else a, int(f[i]), end))
    return out


def swap_pairs(stretches, xy, team):
    """[(track a, track b, frame they traded, last frame)] two tracks of opposite teams whose wrong
    stretches begin together, at a moment the two pass within SWAP_NEAR_M: they traded players."""
    out, used = [], set()
    for i, (k, lo, hi, end) in enumerate(stretches):
        for j, (k2, lo2, hi2, end2) in enumerate(stretches):
            if j <= i or i in used or j in used or team[k] == team[k2]:
                continue
            a, z = max(lo, lo2) - SWAP_SLACK, min(hi, hi2) + SWAP_SLACK
            if a > z:
                continue
            a = max(a, 0)
            d = np.hypot(*(xy[a:z + 1, k] - xy[a:z + 1, k2]).T)
            if not np.isfinite(d).any() or np.nanmin(d) > SWAP_NEAR_M:
                continue
            out.append((k, k2, a + int(np.nanargmin(d)), min(end, end2)))
            used.update((i, j))
    return out


def unswap(players, team, bxy, bteam, blocks):
    """Give two tracks that traded players their own player back. Returns (players, pairs, stretches).
    Only a trade both cameras show is undone: each of the two tracks must be called the other team by the
    broadcast, on frames where its player stands free, from the moment the two crossed. A lone track the
    cameras disagree on is left alone: there is nothing to say which camera is right."""
    from player_paths import pieces_of
    xy = players[..., :2].astype(float)
    graded = np.zeros(len(xy), bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    pieces = pieces_of(np.where(graded[:, None, None], xy, np.nan))
    other, free = second_opinion(xy, team, bxy, bteam)
    stretches = wrong_stretches(pieces, team, other, free)
    pairs = swap_pairs(stretches, xy, team)
    out = players.copy()
    for k, k2, a, z in pairs:
        out[a:z + 1, k], out[a:z + 1, k2] = players[a:z + 1, k2], players[a:z + 1, k]
    return out, pairs, stretches


DISPUTED_BALL_M = 5.0        # a disputed player this close to the ball can be the one on it


def disputed_tracks(players, team, blocks):
    """(N, S) True on every stretch of track that the two cameras put on different teams (wrong_stretches)."""
    from camera_agreement import broadcast_on_tactical_frames
    from player_paths import pieces_of
    xy = players[..., :2].astype(float)
    bxy, _, bteam, _ = broadcast_on_tactical_frames(players, team, None)
    other, free = second_opinion(xy, team, bxy, bteam)
    graded = np.zeros(len(xy), bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    disputed = np.zeros(xy.shape[:2], bool)
    for k, lo, hi, end in wrong_stretches(pieces_of(np.where(graded[:, None, None], xy, np.nan)), team, other, free):
        disputed[lo:end + 1, k] = True
    return disputed


def disputed_near_ball(players, ball, team, blocks):
    """(N,) True where a player within DISPUTED_BALL_M of the ball is on a stretch of track that the two
    cameras put on different teams (wrong_stretches). Which camera is right cannot be told from the data,
    so the caller should not count that player's nearness as either team's control."""
    xy = players[..., :2].astype(float)
    disputed = disputed_tracks(players, team, blocks)
    with np.errstate(invalid="ignore"):
        d = np.hypot(xy[..., 0] - ball[:, None, 0], xy[..., 1] - ball[:, None, 1])
    return (disputed & (d <= DISPUTED_BALL_M)).any(axis=1)


def report():
    from camera_agreement import broadcast_on_tactical_frames
    from player_paths import pieces_of
    from score import graded_mask
    from tracking import load_clean
    tp, tb, team = load_clean("K5")
    xy = tp[..., :2].astype(float)
    bxy, _, bteam, on = broadcast_on_tactical_frames(tp, team, None)
    other, free = second_opinion(xy, team, bxy, bteam)
    g = graded_mask()
    pieces = [p for p in pieces_of(np.where(g[:, None, None], xy, np.nan))]
    verdict = judge(pieces, team, other, free)
    label = np.full(xy.shape[:2], -1, np.int8)      # per track and frame: which kind of piece it is on
    names = ["fine", "conflict", "mixed", "unjudged"]
    count = {n: 0 for n in names}
    for (k, a, z), (n, share) in zip(pieces, verdict):
        kd = kind(n, share)
        count[kd] += 1
        label[a:z + 1, k] = names.index(kd)
    seen = label >= 0
    total = seen.sum()
    print("{} track pieces in the graded halves ({} player-frames)".format(len(pieces), total))
    print("| kind of piece | pieces | share of player-frames |\n| --- | --- | --- |")
    for i, n in enumerate(names):
        print("| {} | {} | {:.1f}% |".format(n, count[n], 100 * (label == i).sum() / total))
    paired = seen & (other >= 0)
    differ = paired & (other != team[None, :])
    print("paired player-frames where the cameras name different teams: {:.1f}% of {}".format(100 * differ.sum() / paired.sum(), paired.sum()))
    print("  of those, on a piece that is fine: {:.0f}% (the pairing's mistake in a bunch), conflict {:.0f}%, mixed {:.0f}%, unjudged {:.0f}%".format(
        *(100 * (differ & (label == i)).sum() / differ.sum() for i in range(4))))
    # how much of the play it touches: the player nearest the ball
    d = np.hypot(xy[..., 0] - tb[:, None, 0], xy[..., 1] - tb[:, None, 1])
    near = np.argmin(np.where(np.isfinite(d), d, np.inf), axis=1)
    has = g & np.isfinite(tb[:, 0]) & (np.take_along_axis(np.where(np.isfinite(d), d, np.inf), near[:, None], axis=1)[:, 0] <= NEAR_BALL_M)
    lab = label[np.arange(len(xy)), near]
    print("frames with a player within {} m of the ball: {}; he is on a piece that is fine {:.1f}%, conflict {:.1f}%, mixed {:.1f}%, unjudged {:.1f}%".format(
        NEAR_BALL_M, int(has.sum()), *(100 * np.mean(lab[has] == i) for i in range(4))))
    stretches = wrong_stretches(pieces, team, other, free)
    pairs = swap_pairs(stretches, xy, team)
    in_pair = {k for p in pairs for k in p[:2]}
    frames = sum(z - hi + 1 for _, _, hi, z in stretches)
    print("stretches inside a piece where the broadcast names the other team: {} ({} player-frames, {:.1f}%)".format(
        len(stretches), frames, 100 * frames / total))
    print("  two opposite tracks that traded players as they crossed: {} pairs, {} player-frames ({:.1f}% of the stretches' frames)".format(
        len(pairs), 2 * sum(z - a + 1 for _, _, a, z in pairs), 100 * 2 * sum(z - a + 1 for _, _, a, z in pairs) / max(frames, 1)))
    print("  stretches with no partner (one camera is wrong, cannot say which): {}".format(len(stretches) - 2 * len(pairs)))
    lens = np.array([z - a + 1 for k, a, z in pieces])
    for i, n in enumerate(names[1:3], start=1):
        sel = np.array([kind(*v) == n for v in verdict])
        if sel.any():
            print("{} pieces last {:.0f} s (median), {:.0f} s (longest); votes per piece {:.0f} (median)".format(
                n, np.median(lens[sel]) / FPS, lens[sel].max() / FPS, np.median([v[0] for v, s in zip(verdict, sel) if s])))


def _self_test():
    team = np.array([0, 0, 1])
    xy = np.full((200, 3, 2), np.nan)
    xy[:, 0], xy[:, 1], xy[:, 2] = [10.0, 10.0], [40.0, 40.0], [70.0, 20.0]
    bxy = np.full((200, 3, 2), np.nan)
    bxy[:, 0], bxy[:, 1], bxy[:, 2] = [10.2, 10.0], [40.2, 40.0], [70.2, 20.0]
    bteam = np.array([0, 1, 1])                      # the broadcast calls track 1 an away player throughout
    other, free = second_opinion(xy, team, bxy, bteam)
    assert free.all() and (other[:, 1] == 1).all() and (other[:, 0] == 0).all()
    v = judge([(0, 0, 199), (1, 0, 199), (2, 0, 30)], team, other, free)
    assert [kind(*x) for x in v] == ["fine", "conflict", "unjudged"]
    other[100:, 0] = 1                               # track 0 agrees for 4 s, then not: a swap
    assert kind(*judge([(0, 0, 199)], team, other, free)[0]) == "mixed"
    # two tracks of opposite teams cross at frame 100 and trade players: found, and put right
    n = 400
    t = np.arange(n)
    p = np.full((n, 3, 3), np.nan)
    home = np.stack([20 + 0.1 * t, np.full(n, 30.0)], axis=1)       # the real home player runs right
    away = np.stack([60 - 0.1 * t, np.full(n, 30.5)], axis=1)       # the real away player runs left; they meet at t = 200
    p[:, 0, :2] = np.where((t < 200)[:, None], home, away)          # track 0 (home slot) leaves with the away player
    p[:, 2, :2] = np.where((t < 200)[:, None], away, home)
    b = np.full((n, 2, 2), np.nan)
    b[:, 0], b[:, 1] = home, away                                    # the broadcast follows both correctly
    fixed, pairs, st = unswap(p, np.array([0, 0, 1]), b, np.array([0, 1]), [(1, 0, n - 1)])
    assert len(pairs) == 1 and len(st) == 2 and abs(pairs[0][2] - 200) <= 2
    assert np.allclose(fixed[350, 0, :2], home[350]) and np.allclose(fixed[350, 2, :2], away[350])
    assert np.allclose(fixed[100, 0, :2], home[100])
    xy[:, 2] = [10.5, 10.0]                          # now two players stand together: no vote counts
    assert not second_opinion(xy, team, bxy, bteam)[1][:, 0].any()
    print("piece_teams self-test passed")


if __name__ == "__main__":
    _self_test()
    report()
