"""Can the broadcast tracking fill the tactical tracking's gaps? Aggregates only.

    python experiments/broadcast_gaps.py

The broadcast camera has its own tracker on its own timeline, synced to the tactical one by one
offset per half (feature_cache.OFFSETS). A broadcast frame counts as live when its visible players
sit on tactical players of the same team (median miss under 2 m), which rules out replays and
close-ups. This report answers, for the halves only:

  ball     of the frames where the tactical ball is missing, how many does a live broadcast frame
           see the ball on; split by gap length; how many gaps it touches and how many it covers
  trust    where both see the ball, how far apart the two positions are
  players  where a team shows fewer than 11 tactical tracks, how many live broadcast players stand
           more than 2 m from every tactical teammate (players the tactical tracker is missing)
"""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Params
NEW_PLAYER_M = 2.0        # a broadcast player this far from every tactical teammate is one the tactical data lacks
GAP_BINS_S = (0, 1, 5, 30, 60, 1e9)


def broadcast_on_tactical(tp, tteam, bp, bb, bteam):
    """Broadcast ball on the tactical timeline (N, 2), NaN unless a live broadcast frame sees it,
    plus (N,) True where a live broadcast frame maps onto that tactical frame. Both inputs must be
    in the same orientation."""
    from feature_cache import broadcast_live
    live, tframe = broadcast_live(bp, bteam, tp, tteam)
    ok = (tframe >= 0) & live
    ball = np.full((len(tp), 2), np.nan, dtype=np.float32)
    ball[tframe[ok]] = bb[ok]
    on = np.zeros(len(tp), bool)
    on[tframe[ok]] = True
    return ball, on, live, tframe


def _missing_players(tp, tteam, bp, bteam, live, tframe, graded):
    """(missing tactical player-frames, broadcast players that could fill them)."""
    missing = 0
    new = 0
    for side in (0, 1):
        n_t = np.isfinite(tp[:, tteam == side, 0]).sum(axis=1)
        missing += int(np.sum(np.clip(11 - n_t[graded], 0, None)))
    idx_all = np.flatnonzero((tframe >= 0) & live)
    idx_all = idx_all[graded[tframe[idx_all]]]
    for s in range(0, len(idx_all), 2000):
        idx = idx_all[s:s + 2000]
        t = tframe[idx]
        for side in (0, 1):
            bpl = bp[idx][:, bteam == side, :2]
            tpl = tp[t][:, tteam == side, :2]
            d = np.hypot(bpl[:, :, None, 0] - tpl[:, None, :, 0], bpl[:, :, None, 1] - tpl[:, None, :, 1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                nearest = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=2)
            far = np.isfinite(bpl[..., 0]) & (nearest > NEW_PLAYER_M)
            room = np.clip(11 - np.isfinite(tpl[..., 0]).sum(axis=1), 0, None)
            new += int(np.sum(np.minimum(far.sum(axis=1), room)))
    return missing, new


def report():
    from cleaning import runs
    from tracking import GAME, half_video_blocks, load
    blocks = half_video_blocks(GAME)
    tp, tb, tteam = load("tactical")
    bp, bb, bteam = load("broadcast")
    graded = np.zeros(len(tb), bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    bball, on, live, tframe = broadcast_on_tactical(tp, tteam, bp, bb, bteam)

    t_seen = np.isfinite(tb[:, 0])
    b_seen = np.isfinite(bball[:, 0])
    miss = graded & ~t_seen
    print("halves: {} frames; tactical ball missing on {} ({:.1f}%)".format(graded.sum(), miss.sum(), 100 * miss.mean() / graded.mean()))
    print("  a live broadcast frame exists on {:.1f}% of those; it sees the ball on {:.1f}% of them ({} frames, {:.1f} min)".format(
        100 * np.sum(miss & on) / miss.sum(), 100 * np.sum(miss & b_seen) / miss.sum(), np.sum(miss & b_seen),
        np.sum(miss & b_seen) / 25 / 60))
    print("  on frames where the tactical ball IS seen, broadcast is live {:.1f}% and sees the ball {:.1f}%".format(
        100 * np.sum(graded & t_seen & on) / np.sum(graded & t_seen), 100 * np.sum(graded & t_seen & b_seen) / np.sum(graded & t_seen)))

    print("\n| gap length | gaps | missing frames | frames broadcast fills | gaps touched | gaps fully covered |")
    print("| --- | --- | --- | --- | --- | --- |")
    gaps = []
    for _, first, last in blocks:
        for s, e in runs(~t_seen[first:last + 1]):
            gaps.append((first + s, first + e))
    for lo, hi in zip(GAP_BINS_S[:-1], GAP_BINS_S[1:]):
        sel = [(s, e) for s, e in gaps if lo * 25 <= e - s + 1 < hi * 25]
        if not sel:
            continue
        frames = sum(e - s + 1 for s, e in sel)
        filled = sum(int(b_seen[s:e + 1].sum()) for s, e in sel)
        touched = sum(1 for s, e in sel if b_seen[s:e + 1].any())
        full = sum(1 for s, e in sel if b_seen[s:e + 1].all())
        label = "{}-{} s".format(lo, hi) if hi < 1e9 else "over {} s".format(lo)
        print("| {} | {} | {} | {:.1f}% | {} | {} |".format(label, len(sel), frames, 100 * filled / frames, touched, full))

    both = graded & t_seen & b_seen
    d = np.hypot(*(tb[both] - bball[both]).T)
    print("\ntrust: where both see the ball ({} frames) the positions differ by median {:.2f} m, p75 {:.2f} m, p90 {:.2f} m".format(
        both.sum(), np.median(d), np.percentile(d, 75), np.percentile(d, 90)))
    missing, new = _missing_players(tp, tteam, bp, bteam, live, tframe, graded)
    print("players: {} tactical player-frames missing (teams under 11); live broadcast could add {} ({:.1f}%)".format(
        missing, new, 100 * new / max(1, missing)))


if __name__ == "__main__":
    report()
