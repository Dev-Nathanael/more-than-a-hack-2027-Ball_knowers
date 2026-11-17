"""Do shape features improve with short smoothing, and how late do they follow a possession change?

    python shape_smoothing.py          # needs cache/atd.npz, the GT file and the Smart Tagging file

Part 1: AUC of each kept feature (home-minus-away, oriented so that > 0.5 is the expected direction)
        at smoothing windows of 1 (raw), 5, 13, 25 and 50 frames, centred, never crossing a half.
        Public window (independent but 17 possessions) and whole-match tag intervals (noisy labels,
        possibly circular - see shape_features.py), the latter also for frames with no ball.
Part 2: for each direct H<->A change in the public ground truth, does the smoothed shape signal
        already point to the NEW team before the label changes ('lead'), flip after a delay ('lag', in
        seconds) or not within 10 s ('none')?  Descriptive only; 12 changes cannot tune anything.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atd_explore import FPS, load_or_parse  # noqa: E402
from compare_gt import GT_PATH, load_gt, runs_of  # noqa: E402
from shape_features import BLOCKS, auc, compute_all, get_dirs  # noqa: E402
from tag_diagnostics import load_events, prepare  # noqa: E402
from tag_module import build_table  # noqa: E402

WINDOWS = (1, 5, 13, 25, 50)        # frames
LAG_WINDOWS = (13, 25, 50)
MIN_COVERAGE = 0.5                  # a smoothed value needs this share of its window to be non-NaN
SUSTAIN = 25                        # frames the new sign must hold to count as a flip
SEARCH_S = 10                       # seconds after the label change to look for a flip
# orientation, from the previous run: the team WITHOUT the ball moves faster, so speed is inverted
KEPT = {"spread_y": +1, "iqr_y": +1, "length": +1, "surge": +1, "speed": -1}


def head(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


def smooth(v, w, min_cov=MIN_COVERAGE):
    """Centred moving mean over w frames, NaN-aware, computed inside each graded half only."""
    if w <= 1:
        return v.astype(np.float64)
    out = np.full(len(v), np.nan)
    h = w // 2
    for _, first, last in BLOCKS:
        seg = v[first:last + 1].astype(np.float64)
        fin = np.isfinite(seg)
        cs = np.concatenate(([0.0], np.cumsum(np.where(fin, seg, 0.0))))
        cn = np.concatenate(([0], np.cumsum(fin)))
        i = np.arange(len(seg))
        lo, hi = np.maximum(0, i - h), np.minimum(len(seg), i + h + 1)
        s, c, width = cs[hi] - cs[lo], cn[hi] - cn[lo], hi - lo
        out[first:last + 1] = np.where(c >= min_cov * width, s / np.maximum(c, 1), np.nan)
    return out


def oriented_auc(score, is_home, sign):
    a = auc(score, is_home)
    return a if sign > 0 else 1.0 - a


def fmt(x):
    return "  n/a" if not np.isfinite(x) else "%5.2f" % x


def part1(F, lab_all, B, table):
    s0 = np.flatnonzero(lab_all != "X")
    s, e = int(s0[0]), int(s0[-1]) + 1
    lab = lab_all[s:e].astype(str)
    pub = np.isin(lab, ["H", "A"])

    ps = np.full(len(B), "", dtype="<U1")
    for r in table[table.state.isin(["H", "A"]) & table.kind.isin(["possession", "transition"])].itertuples():
        ps[r.start_frame:r.end_frame] = r.state
    vis = np.isfinite(B[:, 0]) & np.isfinite(B[:, 1])
    samples = [("PUBLIC window (17 possessions)", None),
               ("WHOLE MATCH vs tag intervals, all frames", ps != ""),
               ("WHOLE MATCH vs tag intervals, ball missing", (ps != "") & ~vis)]
    sm = {(k, w): smooth(F[k], w) for k in KEPT for w in WINDOWS}
    for title, mask in samples:
        head("1. oriented AUC, %s" % title)
        print("%-9s" % "feature" + "".join("%9s" % ("raw" if w == 1 else "%df" % w) for w in WINDOWS))
        for k, sign in KEPT.items():
            cells = []
            for w in WINDOWS:
                v = sm[(k, w)]
                if mask is None:
                    cells.append(oriented_auc(v[s:e][pub], (lab == "H")[pub], sign))
                else:
                    cells.append(oriented_auc(v[mask], (ps == "H")[mask], sign))
            print("%-9s" % k + "".join("%9s" % fmt(c) for c in cells))
    return s, e


def part2(F, lab_all, s, e):
    head("2. lag of the smoothed shape signal at the direct H<->A changes in the public ground truth")
    runs = runs_of(lab_all, s, e)
    changes = [(a2, l2) for (a1, b1, l1), (a2, b2, l2) in zip(runs[:-1], runs[1:])
               if l1 in ("H", "A") and l2 in ("H", "A") and l1 != l2]
    print("%d direct changes. lead = already pointing to the new team in the second before the change; "
          "lag = seconds until the new sign holds for %d frames (median over those that flip); none = no flip within %d s"
          % (len(changes), SUSTAIN, SEARCH_S))
    print("%-9s %7s %6s %6s %6s %12s" % ("feature", "window", "lead", "lag", "none", "median lag s"))
    for k, sign in KEPT.items():
        for w in LAG_WINDOWS:
            o = sign * smooth(F[k], w)
            lead = none = 0
            lags = []
            for b, new in changes:
                want = 1 if new == "H" else -1
                pre = np.nanmedian(o[b - SUSTAIN:b]) if np.isfinite(o[b - SUSTAIN:b]).any() else np.nan
                if np.isfinite(pre) and np.sign(pre) == want:
                    lead += 1
                    continue
                seg = o[b:b + SEARCH_S * FPS + SUSTAIN]
                ok = (np.sign(seg) == want) & np.isfinite(seg)
                run = np.convolve(ok.astype(int), np.ones(SUSTAIN, dtype=int), "valid") == SUSTAIN
                hit = np.flatnonzero(run)
                if len(hit):
                    lags.append(hit[0] / float(FPS))
                else:
                    none += 1
            print("%-9s %5df %6d %6d %6d %12s" % (k, w, lead, len(lags), none, "%.1f" % np.median(lags) if lags else "-"))


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    P, B, team, *_ = load_or_parse(False)
    F = compute_all(P, team, get_dirs(P, team))
    lab_all = load_gt(GT_PATH)
    s, e = part1(F, lab_all, B, build_table(prepare(load_events())))
    part2(F, lab_all, s, e)
