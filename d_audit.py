"""Audit the dead-ball (D) runs of a submission file, with no labels.

    python d_audit.py                                    # reads evaluation/results/control_touch_v13.csv
    python d_audit.py --sub evaluation/results/mine.csv --top 20 --perms 500

1. The longest D runs: clock, duration, mean player speed, share of frames with the ball visible, the nearest
   set-piece / goal tag and where its window starts relative to the run. These are the intervals to look at in the
   video: if they show live play, the dead ball is over-predicted there.
2. A length histogram of all D runs: count, share of D time, ball-visible share and mean player speed per bin.
3. A permutation test of the alignment with the Smart Tagging set pieces. The tag windows (35 s, + 15 s look-back)
   cover about half of the match, so 'overlaps a tag window' is true about half the time by chance. Instead:
     recall     share of set-piece tags with >= 1 s of predicted D in [start - 5 s, start + 10 s]
     precision  share of D runs of <= 60 s with a set-piece tag start within 10 s of the run
   each against the same figure when all tag times are shifted by a random offset (kept inside the graded halves).
   If the real value is not above the shifted ones, the dead ball is not following the tags. Kickoff tags are excluded
   (the first-half kickoff is live play, the second-half one is covered by the pre-kickoff rule).
Note that the tags are incomplete (fouls and offsides are mostly untagged), so precision cannot reach 100%.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import games  # noqa: E402
from atd_explore import FPS, load_or_parse  # noqa: E402
from dead_ball_evidence import signals  # noqa: E402
from tag_diagnostics import fmt_clock, load_events, prepare  # noqa: E402

BINS = ((0, 5), (5, 10), (10, 20), (20, 30), (30, 60), (60, 1e9))


def head(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


def d_runs(labels, blocks):
    out = []
    for _, first, last in blocks:
        seg = (labels[first:last + 1] == "D").astype(np.int8)
        e = np.diff(np.concatenate(([0], seg, [0])))
        out += [(first + int(a), first + int(b)) for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))]
    return out


def stat_recall(D, starts):
    cs = np.concatenate(([0], np.cumsum(D)))
    hits = [(cs[min(len(D), f + 10 * FPS)] - cs[max(0, f - 5 * FPS)]) >= FPS for f in starts]
    return float(np.mean(hits)) if hits else np.nan


def stat_precision(runs, starts):
    st = np.sort(starts)
    ok = []
    for a, b in runs:
        if (b - a) <= 60 * FPS:
            i = np.searchsorted(st, a - 10 * FPS)
            ok.append(i < len(st) and st[i] <= b + 10 * FPS)
    return float(np.mean(ok)) if ok else np.nan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=os.path.join("evaluation", "results", "control_touch_v13.csv"))
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--sample-unbacked", type=int, default=0, help="list N randomly chosen tag-unbacked dead-ball runs (seed 0) to check in the video")
    a_ = ap.parse_args()
    np.seterr(invalid="ignore")
    labels = pd.read_csv(a_.sub)["label"].astype(str).to_numpy()
    P, B, team, *_ = load_or_parse(False)
    G = signals(P, B)
    blocks = games.half_video_blocks(games.GAME)
    runs = d_runs(labels, blocks)
    d = prepare(load_events())
    sp = d[d.code_u == "SET PIECES"]
    goals = d[d.code_u == "GOALS"]
    sp_nk = sp[~sp.type_l.str.startswith("kick off")]
    kick = sp[sp.type_l.str.startswith("kick off")]
    lens = np.array([(b - a) / FPS for a, b in runs])

    head("1. the %d longest dead-ball runs" % a_.top)
    print("%-3s %-10s %7s %9s %8s  %s" % ("#", "from", "length", "player m/s", "ball vis", "nearest set-piece / goal tag (tag start minus run start, s)"))
    for k, i in enumerate(np.argsort(-lens)[:a_.top]):
        a, b = runs[i]
        extra = ""
        for nm, tg in (("set piece", sp), ("goal", goals)):
            if len(tg):
                j = np.argmin(np.abs(tg.start_frame.to_numpy() - a))
                r = tg.iloc[j]
                if abs(r.start_frame - a) <= 120 * FPS:
                    extra += "%s%s: %s %+.0f s; " % (nm, "", (r.Type if isinstance(r.Type, str) else r.code_u.title()), (r.start_frame - a) / FPS)
        first_half_start = [f for _, f, _ in blocks][1]
        note = " [pre-kickoff period]" if a == first_half_start else ""
        print("%-3d %-10s %6.0fs %9.2f %7.0f%%  %s%s" % (k + 1, fmt_clock(a)[:10], (b - a) / FPS, np.nanmean(G["ps"][a:b]), 100 * G["vis"][a:b].mean(), extra or "none within 2 min", note))

    head("2. length histogram of the %d dead-ball runs" % len(runs))
    tot = lens.sum()
    print("%-10s %6s %12s %10s %12s" % ("length", "runs", "share of D", "ball vis", "player m/s"))
    for lo, hi in BINS:
        sel = [r for r, L in zip(runs, lens) if lo <= L < hi]
        if sel:
            fr = sum(b - a for a, b in sel)
            vis = np.mean(np.concatenate([G["vis"][a:b] for a, b in sel]))
            ps = np.nanmean(np.concatenate([G["ps"][a:b] for a, b in sel]))
            print("%-10s %6d %11.0f%% %9.0f%% %12.2f" % ("%g-%s s" % (lo, "end" if hi > 1e8 else "%g" % hi), len(sel), 100 * fr / FPS / tot, 100 * vis, ps))

    head("2b. tag-backed versus unbacked runs")
    sstart = np.sort(sp_nk.start_frame.to_numpy())
    gwin = [(g_.start_frame - 15 * FPS, g_.end_frame) for g_ in goals.itertuples()]
    pre_start = [f for _, f, _ in blocks][1]

    def backed(a, b):
        i = np.searchsorted(sstart, a - 10 * FPS)
        if i < len(sstart) and sstart[i] <= b + 10 * FPS:
            return True
        return a == pre_start or any(a < ge and b > gs for gs, ge in gwin)
    flags = np.array([backed(a, b) for a, b in runs])
    print("%-10s %6s %12s %12s %9s %9s %12s" % ("", "runs", "share of D", "median len", "ball vis", "player", "runs > 30 s"))
    for nm, sel in (("backed", flags), ("unbacked", ~flags)):
        idx = np.flatnonzero(sel)
        if not len(idx):
            continue
        fr = sum(runs[i][1] - runs[i][0] for i in idx)
        vis = np.mean(np.concatenate([G["vis"][runs[i][0]:runs[i][1]] for i in idx]))
        ps = np.nanmean(np.concatenate([G["ps"][runs[i][0]:runs[i][1]] for i in idx]))
        print("%-10s %6d %11.0f%% %11.1fs %8.0f%% %8.2f m/s %9d" % (nm, len(idx), 100 * fr / FPS / tot, np.median(lens[idx]), 100 * vis, ps, int((lens[idx] > 30).sum())))
    print("backed = a set-piece tag starts within 10 s of the run, or it overlaps a goal window (+15 s look-back), or it is the pre-kickoff period.")

    if a_.sample_unbacked:
        head("2c. a random sample of %d tag-unbacked dead-ball runs: is play really stopped from start to end?" % a_.sample_unbacked)
        ub = np.flatnonzero(~flags)
        pick = np.random.RandomState(0).permutation(ub)[:a_.sample_unbacked]
        print("%-3s %-10s %-10s %7s %11s %9s" % ("#", "from", "to", "length", "player m/s", "ball vis"))
        for k, i in enumerate(sorted(pick, key=lambda i: runs[i][0])):
            a, b = runs[i]
            print("%-3d %-10s %-10s %6.0fs %11.2f %8.0f%%" % (k + 1, fmt_clock(a)[:10], fmt_clock(b)[:10], (b - a) / FPS, np.nanmean(G["ps"][a:b]), 100 * G["vis"][a:b].mean()))
        print("For each: stopped for the WHOLE run (real), stopped for part of it (mixed), or live play with slow players (false)?")

    head("3. do the dead-ball runs follow the tags? observed against the same figure with the tag times shifted at random")
    Dm = (labels == "D").astype(np.int8)
    starts = sp_nk.start_frame.to_numpy()
    obs_r, obs_p = stat_recall(Dm, starts), stat_precision(runs, starts)
    graded = np.zeros(len(labels), dtype=bool)
    for _, f, l in blocks:
        graded[f:l + 1] = True
    rng = np.random.RandomState(0)
    nr, npr = [], []
    for _ in range(a_.perms):
        off = int(rng.choice([-1, 1]) * rng.randint(60 * FPS, 600 * FPS))
        sh = starts + off
        sh = sh[(sh >= 0) & (sh < len(labels))]
        sh = sh[graded[sh]]
        if len(sh) >= max(3, len(starts) // 2):
            nr.append(stat_recall(Dm, sh))
            npr.append(stat_precision(runs, sh))
    for nm, o, n in (("recall (tags with D near their start)", obs_r, nr), ("precision (D runs <= 60 s with a tag nearby)", obs_p, npr)):
        n = np.array(n)
        if not len(n):
            print("%-46s observed %5.1f%% | too few shifted tags fell inside the graded halves to build a null" % (nm, 100 * o))
            continue
        print("%-46s observed %5.1f%% | shifted tags: mean %5.1f%%, 5-95%% %5.1f-%5.1f%% | observed above %.0f%% of the shifts"
              % (nm, 100 * o, 100 * n.mean(), 100 * np.percentile(n, 5), 100 * np.percentile(n, 95), 100 * (n < o).mean()))
    print("\n%d set-piece tags (kickoffs excluded), %d D runs, %d of them <= 60 s." % (len(starts), len(runs), int((lens <= 60).sum())))
    print("If 'observed above' is near 100%, the dead ball follows the tags; near 50%, it does not. The tags are incomplete, so"
          "\nprecision is capped well below 100% by the untagged fouls and offsides.")