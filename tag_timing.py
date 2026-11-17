"""Dead-ball timing against the Smart Tagging set pieces.

    python tag_timing.py                                          # evaluation/results/control_touch_v17.csv, public truth
    python tag_timing.py --sub evaluation/results/mine.csv
    python tag_timing.py --list                                   # one block per set-piece tag outside the public window
    python tag_timing.py --sample 10 --seed 0 --csv cache/sample.csv   # seeded stratified sample to check in the video

REFERENCES. Real truth exists only in the public window, where just two set-piece tags fall (n = 2). Everywhere else the
reference is the submission's own D labels, which are MODEL OUTPUT: those numbers measure consistency between the tags and our
ATD-based dead ball, not accuracy. Each section says which reference it uses.

1  'If every set-piece tag window were D': precision / recall / F1
   1a  public window vs the true D (n = 2 tags); three window definitions, shown for information only
   1b  whole match vs the submission's D (consistency), by type
2  START: tag start versus the dead-ball start
   2a  public window: tag start minus true D start (n = 2)
   2b  whole match: start of the submission's D run nearest each tag, relative to the tag start; and the first time the ball is
       outside the pitch relative to the tag start (the physical 'ball went out' event), by type
3  END: how long the dead ball really lasts inside the fixed window, and three ATD end signals
   3a  D run end relative to the tag start and to the tag end (tests the 35 s assumption)
   3b  candidate end signals measured from the D run start: (i) the D run end (restart touch / control), (ii) players' speed
       recovering above RECOVER m/s for 1 s, (iii) the ball visible and faster than FAST m/s. Their spread around (i), and
       around the TRUE ends in the public window (n = 4)
Kickoff tags are excluded (the first-half kickoff is live play; the second half is covered by the pre-kickoff rule).

--list / --sample N (video calibration of the dead-ball timing; need only the submission, cache/atd.npz and the tags, no truth):
   For every set-piece tag outside the public window (kickoffs excluded; --include-public keeps the public ones) it prints the tag
   clock / frame / type / Team, our D run nearest the tag (start, end, length, offsets from the tag start), the first time the ball
   is outside the pitch, the time the players' speed recovers, and a LIVE-PLAY CHECK: the longest stretch INSIDE our D run, before
   the ball goes out (or before the tag start when it never does), where the ball is visible, inside the lines and moving faster than
   LIVE_MS. A stretch >= LIVE_MIN_S flags 'LIVE?': a possible mislabel of live play as D (a ball held at a free-kick spot is
   stationary and does not count; slow deliberate play can still trigger it, so it is a pointer for the video, not a verdict).
   --sample N is seeded and stratified 40% throw-ins / 20% each corner, goal kick, free kick (shortfalls are given to other types).
   Clocks are video clocks H:MM:SS.ss (same clock as the tactical video). --csv writes the same rows plus empty columns for the
   true start (ball out / whistle) and the restart kick that you read off the video.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import games  # noqa: E402
from atd_explore import FPS, LENGTH, WIDTH, load_or_parse  # noqa: E402
from compare_gt import GT_PATH, load_gt, runs_of  # noqa: E402
from d_audit import d_runs  # noqa: E402
from dead_ball_evidence import signals  # noqa: E402
from tag_diagnostics import fmt_clock, load_events, prepare  # noqa: E402

RECOVER, FAST = 1.6, 5.0          # m/s: player speed recovery level, ball speed for 'the ball is moving fast'
SEARCH_S = 20                      # seconds either side of a tag start in which its dead ball is looked for
TYPES = ("throw-in", "corner kick", "goal kick", "free kick")
PUBLIC_WINDOW = (11185, 18685)     # frames with public truth (first 5 minutes); tags overlapping it are already checked
LIVE_MS = 3.0                      # m/s: ball 'moving' for the live-play check (ball held at a restart spot is ~0)
LIVE_MIN_S = 5.0                   # a moving-ball stretch this long inside our D run flags 'LIVE?'
LIVE_BRIDGE_S = 1.0                # gaps up to this long inside a moving-ball stretch are bridged
QUOTA = (("throw-in", 0.4), ("corner kick", 0.2), ("goal kick", 0.2), ("free kick", 0.2))


def head(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


def prf(pred, ref):
    tp, fp, fn = int((pred & ref).sum()), int((pred & ~ref).sum()), int((~pred & ref).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), tp, fp, fn


def best_run(runs, a, b):
    best, bo = None, 0
    for s, e in runs:
        o = min(e, b) - max(s, a)
        if o > bo:
            best, bo = (s, e), o
    return best


def med(v, p=1):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return "n/a" if not len(v) else "%+.*f (%+.*f to %+.*f)" % (p, np.median(v), p, np.percentile(v, 25), p, np.percentile(v, 75))


def ball_speed(B):
    bx, by = B[:, 0] * LENGTH, B[:, 1] * WIDTH
    sp = np.full(len(bx), np.nan)
    sp[5:] = np.hypot(bx[5:] - bx[:-5], by[5:] - by[:-5]) * FPS / 5.0
    return sp


def first_true(mask, a, b, min_len=1):
    """First frame in [a, b) where `mask` is True for at least min_len frames in a row; None if never."""
    seg = mask[a:b].astype(np.int8)
    e = np.diff(np.concatenate(([0], seg, [0])))
    for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if t - s >= min_len:
            return a + int(s)
    return None


def longest_run(mask, a, b, bridge=0):
    """Longest stretch of True in mask[a:b], bridging False gaps of <= bridge frames. -> (start_frame, length_frames) or (None, 0)."""
    a, b = max(int(a), 0), min(int(b), len(mask))
    if b <= a:
        return None, 0
    seg = np.array(mask[a:b], dtype=bool)
    if bridge > 0:
        g = np.diff(np.concatenate(([0], (~seg).astype(np.int8), [0])))
        for s_, t_ in zip(np.flatnonzero(g == 1), np.flatnonzero(g == -1)):
            if s_ > 0 and t_ < len(seg) and t_ - s_ <= bridge:
                seg[s_:t_] = True
    e = np.diff(np.concatenate(([0], seg.astype(np.int8), [0])))
    st, en = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    if not len(st):
        return None, 0
    k = int(np.argmax(en - st))
    return a + int(st[k]), int(en[k] - st[k])


def side_of(x):
    return x if isinstance(x, str) and x in ("H", "A") else "N/A"


def tag_table(sp, runs, G, bsp, n):
    """One dict per set-piece tag row of `sp`: the tag, our nearest D run, first ball-outside, players' recovery, live-play check."""
    live_mask = G["vis"] & ~G["out"] & (bsp >= LIVE_MS)
    rows = []
    for r in sp.itertuples():
        t0 = int(r.start_frame)
        run = best_run(runs, t0 - SEARCH_S * FPS, t0 + SEARCH_S * FPS)
        out = first_true(G["out"], max(0, t0 - 30 * FPS), min(n, t0 + 10 * FPS), 3)
        row = {"code_id": r.code_id, "type": r.type_l, "team": side_of(r.ha), "tag_start": t0, "tag_end": int(r.end_frame),
               "d_start": None, "d_end": None, "out": out, "recover": None, "live_len": 0.0, "live_start": None, "flag": False}
        if run:
            s_, e_ = run
            row["d_start"], row["d_end"] = int(s_), int(e_)
            row["recover"] = first_true(G["ps"] >= RECOVER, s_ + 2 * FPS, min(n, e_ + 15 * FPS), FPS)
            lim = out if out is not None else t0
            ls, ll = longest_run(live_mask, s_, min(e_, lim), int(LIVE_BRIDGE_S * FPS))
            row["live_start"], row["live_len"] = ls, ll / FPS
            row["flag"] = ll / FPS >= LIVE_MIN_S
        rows.append(row)
    return rows


def eligible_tags(sp, include_public=False):
    """Non-kickoff set-piece tags, minus those overlapping the public window (unless include_public)."""
    keep = ~sp.type_l.str.startswith("kick off").fillna(True).astype(bool)
    sp = sp[keep]
    if not include_public:
        sp = sp[~((sp.end_frame > PUBLIC_WINDOW[0]) & (sp.start_frame < PUBLIC_WINDOW[1]))]
    return sp.sort_values("start_frame", kind="mergesort")


def quotas(N, avail):
    """Per-type sample sizes: 40/20/20/20 by largest remainder, capped by availability, shortfalls given to types with spare."""
    raw = {k: N * f for k, f in QUOTA}
    q = {k: int(np.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: -(raw[k] - q[k]))[:N - sum(q.values())]:
        q[k] += 1
    q = {k: min(v, avail.get(k, 0)) for k, v in q.items()}
    while sum(q.values()) < min(N, sum(avail.get(k, 0) for k, _ in QUOTA)):
        for k, _ in QUOTA:
            if sum(q.values()) < N and q[k] < avail.get(k, 0):
                q[k] += 1
    return q


def pick_sample(rows, N, seed):
    avail = {k: sum(1 for x in rows if x["type"] == k) for k, _ in QUOTA}
    q = quotas(N, avail)
    rs = np.random.RandomState(seed)
    chosen = []
    for k, _ in QUOTA:
        idx = [i for i, x in enumerate(rows) if x["type"] == k]
        chosen += [idx[j] for j in rs.permutation(len(idx))[:q[k]]]
    return [rows[i] for i in sorted(chosen, key=lambda i: rows[i]["tag_start"])], q, avail


def rel(frame, ref):
    return "n/a" if frame is None else "%+.1f s" % ((frame - ref) / FPS)


def clk(frame):
    return "   n/a    " if frame is None else fmt_clock(frame)


def print_rows(rows):
    for i, x in enumerate(rows, 1):
        t0 = x["tag_start"]
        print("#%-2d tag %s  f=%d  %-12s Team=%-3s window %.0f s  (id %s)" % (i, clk(t0), t0, x["type"], x["team"], (x["tag_end"] - t0) / FPS, x["code_id"]))
        if x["d_start"] is None:
            print("    our D       none within +-%d s of the tag start" % SEARCH_S)
        else:
            print("    our D       %s (%s) -> %s (%s)  length %.1f s" % (clk(x["d_start"]), rel(x["d_start"], t0), clk(x["d_end"]), rel(x["d_end"], t0), (x["d_end"] - x["d_start"]) / FPS))
        print("    ball out    %s (%s)%s" % (clk(x["out"]), rel(x["out"], t0),
                                          "" if x["out"] is None or x["d_start"] is None else "   our D start minus ball out: %+.1f s" % ((x["d_start"] - x["out"]) / FPS)))
        if x["d_start"] is not None:
            print("    recovered   %s (%s from tag start, %s from our D end)" % (clk(x["recover"]), rel(x["recover"], t0), rel(x["recover"], x["d_end"])))
            if x["live_len"] > 0:
                print("    moving ball inside D before ball-out: %.1f s from %s%s" % (x["live_len"], clk(x["live_start"]), "   <-- LIVE? check in video" if x["flag"] else ""))
            else:
                print("    moving ball inside D before ball-out: none")
        print()


def print_flag_summary(rows):
    print("LIVE-PLAY CHECK over all %d eligible tags (label-free; flag = moving ball >= %.0f s inside our D before the ball goes out)" % (len(rows), LIVE_MIN_S))
    print("%-12s %5s %9s %22s %10s %24s" % ("type", "tags", "has D run", "D start - ball out (s)", "flagged", "live seconds (median, p75)"))
    for kind in TYPES + ("all",):
        g = [x for x in rows if kind == "all" or x["type"] == kind]
        w = [x for x in g if x["d_start"] is not None]
        dso = [(x["d_start"] - x["out"]) / FPS for x in w if x["out"] is not None]
        ll = [x["live_len"] for x in w]
        print("%-12s %5d %8.0f%% %22s %9.0f%% %24s" % (kind, len(g), 100 * len(w) / max(1, len(g)), med(dso),
                                                       100 * np.mean([x["flag"] for x in w]) if w else 0,
                                                       "n/a" if not ll else "%.1f, %.1f" % (np.median(ll), np.percentile(ll, 75))))
    print("negative 'D start - ball out' = D starts before the ball leaves the pitch.")


def write_csv(rows, path):
    cols = ["tag_clock", "tag_frame", "type", "team", "d_start_clock", "d_start_rel_s", "d_end_clock", "d_end_rel_s", "ball_out_clock",
            "ball_out_rel_s", "recover_clock", "live_s", "live_start_clock", "flag", "TRUE_start_clock", "TRUE_restart_clock", "notes"]
    f = lambda fr: "" if fr is None else fmt_clock(fr)
    g = lambda fr, ref: "" if fr is None else "%.1f" % ((fr - ref) / FPS)
    out = [[f(x["tag_start"]), x["tag_start"], x["type"], x["team"], f(x["d_start"]), g(x["d_start"], x["tag_start"]), f(x["d_end"]), g(x["d_end"], x["tag_start"]),
            f(x["out"]), g(x["out"], x["tag_start"]), f(x["recover"]), "%.1f" % x["live_len"], f(x["live_start"]), int(x["flag"]), "", "", ""] for x in rows]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(out, columns=cols).to_csv(path, index=False)
    print("wrote %s (fill TRUE_start_clock / TRUE_restart_clock / notes while watching; local only, do not commit)" % path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=os.path.join("evaluation", "results", "control_touch_v17.csv"))
    ap.add_argument("--list", action="store_true", help="one block per eligible set-piece tag (public window excluded), then exit")
    ap.add_argument("--sample", type=int, default=0, help="seeded stratified sample of N tags (40%% throw-ins, 20%% each other type), then exit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-public", action="store_true", help="do not exclude tags overlapping the public window")
    ap.add_argument("--csv", help="with --list/--sample: also write the rows here, with empty columns for what you read off the video")
    a_ = ap.parse_args()
    np.seterr(invalid="ignore")
    lab = pd.read_csv(a_.sub)["label"].astype(str).to_numpy()
    n = len(lab)
    P, B, team, *_ = load_or_parse(False)
    G = signals(P, B)
    bsp = ball_speed(B)
    blocks = games.half_video_blocks(games.GAME)
    graded = np.zeros(n, dtype=bool)
    for _, f, l in blocks:
        graded[f:l + 1] = True
    d = prepare(load_events())
    sp = d[(d.code_u == "SET PIECES") & ~d.type_l.str.startswith("kick off")]
    goals = d[d.code_u == "GOALS"]
    runs = d_runs(lab, blocks)
    if a_.list or a_.sample:
        rows_all = tag_table(eligible_tags(sp, a_.include_public), runs, G, bsp, n)
        head("set-piece tags: %d eligible (kickoffs%s excluded)" % (len(rows_all), "" if a_.include_public else " and the public window"))
        print_flag_summary(rows_all)
        if a_.sample:
            rows_show, q, avail = pick_sample(rows_all, a_.sample, a_.seed)
            head("seeded sample (seed %d): %d tags; per type %s; available %s" % (a_.seed, len(rows_show), q, avail))
        else:
            rows_show = rows_all
            head("all %d eligible tags in match order" % len(rows_show))
        print_rows(rows_show)
        if a_.csv:
            write_csv(rows_show, a_.csv)
        sys.exit(0)
    gt = load_gt(GT_PATH)
    v = np.flatnonzero(gt != "X")
    s0, e0 = int(v[0]), int(v[-1]) + 1
    truth_d = np.zeros(n, dtype=bool)
    truth_d[s0:e0] = gt[s0:e0] == "D"
    true_runs = [(a, b) for a, b, l in runs_of(gt, s0, e0) if l == "D"]

    # ------------------------------------------------------------------ 1a
    head("1a. PUBLIC window, REAL truth: every set-piece tag window labelled D (n = %d tags in the window)" % int(((sp.end_frame > s0) & (sp.start_frame < e0)).sum()))
    pub = sp[(sp.end_frame > s0) & (sp.start_frame < e0)]
    for r in pub.itertuples():
        print("  tag %s %-12s %s - %s (window %.0f s)" % (r.code_id, r.Type, fmt_clock(r.start_frame)[2:9], fmt_clock(r.end_frame)[2:9], (r.end_frame - r.start_frame) / FPS))
    print("  true D stretches: " + ", ".join("%s-%s" % (fmt_clock(a)[2:9], fmt_clock(b)[2:9]) for a, b in true_runs))
    print("\n  %-34s %9s %8s %6s  %s" % ("window definition (information only)", "precision", "recall", "F1", "TP / FP / FN frames"))
    for name, lo, hi in (("as tagged (start .. end)", 0, None), ("start-10 s .. start+10 s", -10, 10), ("start-15 s .. start+5 s", -15, 5)):
        pred = np.zeros(n, dtype=bool)
        for r in pub.itertuples():
            a = r.start_frame + int(lo * FPS)
            b = r.end_frame if hi is None else r.start_frame + int(hi * FPS)
            pred[max(a, 0):b] = True
        p, r_, f, tp, fp, fn = prf(pred[s0:e0], truth_d[s0:e0])
        print("  %-34s %9.2f %8.2f %6.2f  %d / %d / %d" % (name, p, r_, f, tp, fp, fn))
    print("  (the submission itself: %s)" % " ".join("%s %.2f" % (k, x) for k, x in zip(("precision", "recall", "F1"), prf(lab[s0:e0] == "D", truth_d[s0:e0])[:3])))
    print("  Only two tags fall in the window, and two of the four true stoppages (fouls) have none: treat these as an illustration, not a rate.")

    # ------------------------------------------------------------------ 1b
    head("1b. WHOLE MATCH, reference = the submission's own D (model output, so this is consistency, not accuracy)")
    ref = (lab == "D") & graded
    print("%-30s %9s %8s %6s %10s" % ("windows labelled D", "precision", "recall", "F1", "window s"))
    sets = (("set pieces (as tagged)", [(r.start_frame, r.end_frame) for r in sp.itertuples()]),
            ("set pieces + goal windows", [(r.start_frame, r.end_frame) for r in sp.itertuples()] + [(r.start_frame, r.end_frame) for r in goals.itertuples()]))
    for name, wins in sets:
        pred = np.zeros(n, dtype=bool)
        for a, b in wins:
            pred[a:b] = True
        pred &= graded
        p, r_, f, *_ = prf(pred, ref)
        print("%-30s %9.2f %8.2f %6.2f %10.0f" % (name, p, r_, f, pred.sum() / FPS))
    print("\n%-14s %5s %20s   (share of each type's window frames that our D also marks)" % ("type", "tags", "window frames = D"))
    for kind in TYPES:
        g = sp[sp.type_l == kind]
        fr = tot = 0
        for r in g.itertuples():
            seg = slice(r.start_frame, r.end_frame)
            fr += int(ref[seg].sum())
            tot += int(graded[seg].sum())
        print("%-14s %5d %19.0f%%" % (kind, len(g), 100 * fr / max(1, tot)))

    # ------------------------------------------------------------------ per-tag measurements
    rows = []
    for r in sp.itertuples():
        a, b = r.start_frame - SEARCH_S * FPS, r.start_frame + SEARCH_S * FPS
        run = best_run(runs, a, b)
        out_first = first_true(G["out"], max(0, r.start_frame - 30 * FPS), min(n, r.start_frame + 10 * FPS), 3)
        row = {"kind": r.type_l, "start": r.start_frame, "end": r.end_frame, "run": run,
               "out": None if out_first is None else (out_first - r.start_frame) / FPS}
        if run:
            s, e = run
            row.update(ds=(s - r.start_frame) / FPS, de=(e - r.start_frame) / FPS, dte=(e - r.end_frame) / FPS, length=(e - s) / FPS)
            rec = first_true(G["ps"] >= RECOVER, s + 2 * FPS, min(n, e + 15 * FPS), FPS)
            fast = first_true(G["vis"] & (bsp >= FAST), s + 2 * FPS, min(n, e + 15 * FPS), 1)
            row.update(rec=np.nan if rec is None else (rec - e) / FPS, fast=np.nan if fast is None else (fast - e) / FPS)
        rows.append(row)

    # ------------------------------------------------------------------ 2
    head("2a. PUBLIC window, REAL truth: tag start versus the true dead-ball start and end (n = %d)" % len(pub))
    for r in pub.itertuples():
        t = best_run(true_runs, r.start_frame - 30 * FPS, r.start_frame + 30 * FPS)
        if t:
            print("  tag %s (%s) starts at %s: true D %s-%s -> tag start minus true start %+.1f s, minus true end %+.1f s"
                  % (r.code_id, r.Type, fmt_clock(r.start_frame)[2:9], fmt_clock(t[0])[2:9], fmt_clock(t[1])[2:9], (r.start_frame - t[0]) / FPS, (r.start_frame - t[1]) / FPS))
        else:
            print("  tag %s (%s): no true D stretch within 30 s" % (r.code_id, r.Type))
    head("2b. WHOLE MATCH: where does the dead ball start relative to the tag start? (submission's D = model output; median, quartiles in s)")
    print("%-12s %5s %9s %26s %9s %26s" % ("type", "tags", "has D run", "D run start - tag start", "ball out", "first ball-outside - tag start"))
    for kind in TYPES + ("all",):
        g = [x for x in rows if kind == "all" or x["kind"] == kind]
        w = [x for x in g if x["run"]]
        o = [x for x in g if x["out"] is not None]
        print("%-12s %5d %8.0f%% %26s %8.0f%% %26s" % (kind, len(g), 100 * len(w) / max(1, len(g)), med([x["ds"] for x in w]), 100 * len(o) / max(1, len(g)), med([x["out"] for x in o])))
    print("negative = before the tag start. 'ball out' = the ball outside the pitch lines for >= 3 frames, searched from 30 s before to 10 s after the tag start.")

    # ------------------------------------------------------------------ 3
    head("3a. WHOLE MATCH: where does the dead ball END relative to the tag window? (the tag window is a fixed 35 s)")
    print("%-12s %5s %26s %26s %12s %14s" % ("type", "runs", "D run end - tag start", "D run end - tag END", "length s", "ends > 5 s early"))
    for kind in TYPES + ("all",):
        w = [x for x in rows if x["run"] and (kind == "all" or x["kind"] == kind)]
        early = 100 * np.mean([x["dte"] < -5 for x in w]) if w else float("nan")
        print("%-12s %5d %26s %26s %12s %13.0f%%" % (kind, len(w), med([x["de"] for x in w]), med([x["dte"] for x in w]), "%.0f" % np.median([x["length"] for x in w]) if w else "n/a", early))
    head("3b. three ATD signals for the END of the dead ball, relative to the D run end (the restart touch / control)")
    print("%-12s %5s %26s %9s %26s %9s" % ("type", "runs", "(ii) players recover - D end", "found", "(iii) ball fast - D end", "found"))
    for kind in TYPES + ("all",):
        w = [x for x in rows if x["run"] and (kind == "all" or x["kind"] == kind)]
        r_ = [x["rec"] for x in w]
        f_ = [x["fast"] for x in w]
        print("%-12s %5d %26s %8.0f%% %26s %8.0f%%" % (kind, len(w), med(r_), 100 * np.mean(np.isfinite(r_)) if w else 0, med(f_), 100 * np.mean(np.isfinite(f_)) if w else 0))
    print("\nthe same signals against the TRUE ends in the public window (n = %d; seconds, signal minus true end):" % len(true_runs))
    print("%-22s %9s %14s %16s" % ("true D stretch", "length", "players recover", "ball visible+fast"))
    for a, b in true_runs:
        rec = first_true(G["ps"] >= RECOVER, a + 2 * FPS, min(n, b + 15 * FPS), FPS)
        fast = first_true(G["vis"] & (bsp >= FAST), a + 2 * FPS, min(n, b + 15 * FPS), 1)
        print("%-22s %8.1fs %14s %16s" % (fmt_clock(a)[2:9] + "-" + fmt_clock(b)[2:9], (b - a) / FPS,
                                         "never" if rec is None else "%+.1f s" % ((rec - b) / FPS), "never" if fast is None else "%+.1f s" % ((fast - b) / FPS)))
    print("\nA signal is useful for the end only if its offset from the D end is small and consistent. The D end itself (restart touch) agrees with the"
          "\ntrue public ends to within about 0.5 s where it fired (see the per-stretch table of control_touch_decoder.py).")