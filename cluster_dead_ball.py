"""Do players gather into clusters during dead balls?  (free kick wall ~3-4, corner crowd ~4-5, goal kick none)

    python cluster_dead_ball.py          # needs cache/atd.npz, the GT file and the Smart Tagging file

Per frame, among tracked players (no identities needed), players are LINKED when within d metres; a cluster is
a connected group. Features:
  L3, L5, L8   size of the largest cluster of ANY players (both teams) at link distance 3, 5, 8 m
  T3, T5, T8   size of the largest cluster of a SINGLE team (the larger of the two teams): a wall / a team crowd
Section 1  public window: each feature for true D versus live (median, AUC; > 0.5 = larger in D), then per true D stretch
           against the 10 s before it.
Section 2  all set-piece tags, aligned to the start of the tag window (35 s; kickoffs skipped), by type, in 5 s bins,
           for L5 and T5, against the whole-match average. This checks the claim type by type.
Computed on every 5th frame. Descriptive only; four D stretches and 5-26 tags per type cannot tune anything.
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atd_explore import FPS, LENGTH, WIDTH, graded_mask, load_or_parse  # noqa: E402
from compare_gt import GT_PATH, load_gt, runs_of  # noqa: E402
from shape_features import auc  # noqa: E402
from tag_diagnostics import fmt_clock, load_events, prepare  # noqa: E402

LINKS = (3, 5, 8)
STEP = 5
CHUNK = 1500
TYPES = ("throw-in", "corner kick", "goal kick", "free kick")


def head(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


def largest_cluster(X, Y, d):
    """Largest connected group (link distance d) per frame; X, Y are (frames, players) in metres, NaN = missing."""
    ok = np.isfinite(X) & np.isfinite(Y)
    D = np.hypot(X[:, :, None] - X[:, None, :], Y[:, :, None] - Y[:, None, :])
    A = ((D <= d) & ok[:, :, None] & ok[:, None, :]).astype(np.float32)
    for _ in range(int(np.ceil(np.log2(max(2, X.shape[1])))) + 1):          # transitive closure by repeated squaring
        A = (np.matmul(A, A) > 0).astype(np.float32)
    return np.where(ok, A.sum(2), 0).max(1)


def features(P, team, frames):
    n = P.shape[0]
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in ["L%d" % d for d in LINKS] + ["T%d" % d for d in LINKS]}
    for a in range(0, len(frames), CHUNK):
        ch = frames[a:a + CHUNK]
        X, Y = P[ch][:, :, 0] * LENGTH, P[ch][:, :, 1] * WIDTH
        for d in LINKS:
            out["L%d" % d][ch] = largest_cluster(X, Y, d)
            out["T%d" % d][ch] = np.maximum(largest_cluster(X[:, team == 0], Y[:, team == 0], d),
                                            largest_cluster(X[:, team == 1], Y[:, team == 1], d))
    return out


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    P, B, team, *_ = load_or_parse(False)
    lab_all = load_gt(GT_PATH)
    v = np.flatnonzero(lab_all != "X")
    s, e = int(v[0]), int(v[-1]) + 1
    tags = prepare(load_events())
    tags = tags[tags.code_u == "SET PIECES"]
    g = graded_mask()
    pre, post = 10 * FPS, 45 * FPS
    need = set(range(s, e, STEP))
    for r in tags.itertuples():
        if r.type_l in TYPES:
            need.update(range(max(0, r.start_frame - pre), min(P.shape[0] - 1, r.start_frame + post), STEP))
    base = np.arange(0, P.shape[0], 25)                                          # one frame per second for the whole-match reference
    frames = np.array(sorted(need | set(base[g[base]].tolist())))
    F = features(P, team, frames)

    head("1. public window: cluster sizes in true D versus live frames (every %dth frame)" % STEP)
    idx = np.arange(s, e, STEP)
    lab = lab_all[idx].astype(str)
    dead, live = lab == "D", np.isin(lab, ["H", "A"])
    m = dead | live
    print("%-5s %12s %12s %8s" % ("", "median in D", "median live", "AUC"))
    for k in sorted(F):
        x = F[k][idx]
        print("%-5s %12.1f %12.1f %8.2f" % (k, np.nanmedian(x[dead]), np.nanmedian(x[live]), auc(x[m], dead[m])))
    print("\neach true D stretch: mean L5 / T5 in the 10 s before, during and after")
    for a1, b1, l in runs_of(lab_all, s, e):
        if l != "D":
            continue
        cells = []
        for lo, hi in ((a1 - pre, a1), (a1, b1), (b1, b1 + pre)):
            ii = np.arange(max(lo, s), min(hi, e))
            ii = ii[np.isin(ii, frames)]
            cells.append("L5 %.1f  T5 %.1f" % (np.nanmean(F["L5"][ii]), np.nanmean(F["T5"][ii])))
        print("  %s-%s  %4.1fs | before: %s | during: %s | after: %s" % (fmt_clock(a1)[2:9], fmt_clock(b1)[2:9], (b1 - a1) / FPS, *cells))

    head("2. set-piece tags aligned to the window start: mean largest cluster, 5 s bins (L5 = any players, T5 = one team)")
    gm = frames[g[frames]]
    ref = (np.nanmean(F["L5"][gm]), np.nanmean(F["T5"][gm]))
    offs = np.arange(-pre, post, 5 * FPS)
    for kind in TYPES:
        sub = tags[tags.type_l == kind]
        if sub.empty:
            continue
        print("\n%s (%d tags)   t(s):  %s" % (kind, len(sub), " ".join("%5.0f" % (o / FPS) for o in offs)))
        for key in ("L5", "T5"):
            row = []
            for o in offs:
                vals = []
                for r in sub.itertuples():
                    ii = np.arange(r.start_frame + o, r.start_frame + o + 5 * FPS)
                    ii = ii[np.isin(ii, frames)]
                    if len(ii):
                        vals.append(np.nanmean(F[key][ii]))
                row.append(np.nanmean(vals) if vals else np.nan)
            print("  %s %s  %s" % (key, " " * 14, " ".join("%5.1f" % x for x in row)))
    print("\nwhole-match average (graded frames): L5 %.1f, T5 %.1f" % ref)

    head("3. how often does a large cluster occur, by situation? (share of frames at or above a level)")
    N = P.shape[0]
    near = np.zeros(N, dtype=bool)                                    # inside any set-piece tag window, extended 15 s back
    cf = np.zeros(N, dtype=bool)                                      # inside a corner / free-kick window (-10 s .. +15 s)
    gt_ = np.zeros(N, dtype=bool)                                     # inside a goal-kick / throw-in window (-10 s .. +15 s)
    for r in tags.itertuples():
        near[max(0, r.start_frame - 15 * FPS):min(N, r.end_frame)] = True
        w = slice(max(0, r.start_frame - 10 * FPS), min(N, r.start_frame + 15 * FPS))
        if r.type_l in ("corner kick", "free kick"):
            cf[w] = True
        elif r.type_l in ("goal kick", "throw-in"):
            gt_[w] = True
    ref_frames = frames[g[frames]]
    groups = (("corner / free-kick windows", ref_frames[cf[ref_frames]]),
              ("goal-kick / throw-in windows", ref_frames[gt_[ref_frames] & ~cf[ref_frames]]),
              ("OUTSIDE every tag window", ref_frames[~near[ref_frames]]))
    pub = np.arange(s, e, STEP)
    pl = lab_all[pub].astype(str)
    print("%-26s %9s" % ("level", "") + "".join("%34s" % n_ for n_, _ in groups) + "%24s" % "public D | live")
    for key, levels in (("L5", (6, 8, 10)), ("T5", (4, 6))):
        for lv in levels:
            cells = []
            for _, fr in groups:
                x = F[key][fr]
                cells.append("%5.0f%% of %5d frames" % (100 * np.nanmean(x >= lv), len(fr)))
            x = F[key][pub]
            pd_, pl_ = np.nanmean(x[pl == "D"] >= lv), np.nanmean(x[np.isin(pl, ["H", "A"])] >= lv)
            print("%-26s %9s" % ("%s >= %d" % (key, lv), "") + "".join("%34s" % c for c in cells) + "%24s" % ("%.0f%% | %.0f%%" % (100 * pd_, 100 * pl_)))
    print("\nThe 'OUTSIDE' column is the false-alarm rate of the cluster cue over the whole match (one frame per second);"
          "\nthe tag-window columns say how often the cue is on where a set piece is expected.")
