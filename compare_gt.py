"""Compare the tag module with the public ground truth over the kickoff + first 5 minutes.

    python compare_gt.py                      # reads gt/BAR-ATH_possession_gt.csv
    python compare_gt.py --gt path/to/gt.csv

Descriptive only: nothing is tuned and nothing is written. The window is taken from the ground
truth itself (the frames whose label is not X). Offsets are signed, in seconds, tag minus truth:
negative = the tag boundary is EARLIER than the real one.

Place in the repo root next to games.py, tag_diagnostics.py and tag_module.py.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tag_diagnostics import FPS, fmt_clock, load_events, prepare  # noqa: E402
from tag_module import build_table, side_of  # noqa: E402

GT_PATH = os.path.join("evaluation/gt", "BAR-ATH_possession_gt_public.csv")
NEAR_S = 10.0     # how far from a GT dead-ball stretch to look for a set-piece tag
CHANGE_S = 15.0   # how far from a GT possession change to look for a tag boundary
CLASSES = ("H", "A", "D")


def head(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


def load_gt(path):
    g = pd.read_csv(path, dtype={"label": "object"})
    g.columns = [c.strip().lower() for c in g.columns]
    lab = g["label"].astype(str).str.strip().str.upper().to_numpy()
    assert (g["frame"].to_numpy() == np.arange(len(g))).all(), "frame column is not 0..n-1"
    return lab


def runs_of(lab, s, e):
    seg = lab[s:e]
    cuts = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    b = np.concatenate(([0], cuts, [len(seg)]))
    return [(s + int(b[i]), s + int(b[i + 1]), str(seg[b[i]])) for i in range(len(b) - 1)]


def txt(x):
    return x if isinstance(x, str) else ""


def sec(frames):
    return round(frames / float(FPS), 1)


def section_gt(runs):
    head("1. ground-truth segments in the window")
    n_poss = sum(1 for _, _, l in runs if l in ("H", "A"))
    print("%d segments: %d possessions (H/A), %d dead-ball (D)" % (len(runs), n_poss, len(runs) - n_poss))
    for i, (a, b, l) in enumerate(runs):
        print("  %2d  %s - %s  %5.1fs  %s" % (i, fmt_clock(a), fmt_clock(b), (b - a) / FPS, l))


def section_rows(t, lab, s, e):
    head("2. tag-module intervals vs ground truth (frames inside the window)")
    rows = t[(t.end_frame > s) & (t.start_frame < e)]
    conf = {st: dict.fromkeys(CLASSES, 0) for st in ("H", "A", "D", "unknown")}
    out = []
    for r in rows.itertuples():
        a, b = max(r.start_frame, s), min(r.end_frame, e)
        seg = lab[a:b]
        cnt = {c: int((seg == c).sum()) for c in CLASSES}
        for c in CLASSES:
            conf[r.state][c] += cnt[c]
        tot = float(b - a)
        maj = max(CLASSES, key=lambda c: cnt[c])
        out.append({"id": r.event_id, "start": fmt_clock(a), "end": fmt_clock(b), "dur_s": sec(b - a),
                    "state": r.state, "kind": r.kind, "strength": r.strength,
                    "%H": round(100 * cnt["H"] / tot), "%A": round(100 * cnt["A"] / tot),
                    "%D": round(100 * cnt["D"] / tot), "gt_major": maj,
                    "agree": "-" if r.state == "unknown" else ("yes" if maj == r.state else "NO")})
    with pd.option_context("display.max_rows", None, "display.width", 250):
        print(pd.DataFrame(out).to_string(index=False))
    print("\nconfusion in frames (rows = tag-module state, cols = ground truth):")
    cm = pd.DataFrame(conf).T[list(CLASSES)]
    cm["total"] = cm.sum(axis=1)
    print(cm.to_string())
    dec = cm.loc[["H", "A", "D"], list(CLASSES)]
    print("frames where the tag module committed to H/A/D: %d of %d, agreeing with the truth on %.1f%%"
          % (dec.values.sum(), cm["total"].sum(), 100.0 * np.trace(dec.values) / max(1, dec.values.sum())))


def section_stoppages(d, runs):
    head("3. ground-truth dead-ball stretches vs set-piece / goal tags (tag minus truth, seconds)")
    sp = d[d.code_u.isin(["SET PIECES", "GOALS"])]
    near = NEAR_S * FPS
    used = set()
    for i, (a, b, l) in enumerate(runs):
        if l != "D":
            continue
        prev = next((x[2] for x in reversed(runs[:i]) if x[2] in ("H", "A")), "-")
        nxt = next((x[2] for x in runs[i + 1:] if x[2] in ("H", "A")), "-")
        print("\nGT D  %s - %s (%.1fs) | team before: %s, team after: %s" % (fmt_clock(a), fmt_clock(b), (b - a) / FPS, prev, nxt))
        hit = sp[(sp.start_frame < b + near) & (sp.end_frame > a - near)]
        if hit.empty:
            print("   no set-piece/goal tag within %.0fs" % NEAR_S)
        for r in hit.itertuples():
            used.add(r.code_id)
            tag_team = side_of(r.ha) or "-"
            print("   tag %-4s %-11s %-22s %s - %s | start %+6.1fs end %+6.1fs | contains the stoppage: %s | "
                  "Team=%s  (= team after? %s, = team before? %s)"
                  % (r.code_id, r.code_u.title(), txt(r.Type), fmt_clock(r.start_frame), fmt_clock(r.end_frame),
                     sec(r.start_frame - a), sec(r.end_frame - b),
                     "yes" if r.start_frame <= a and r.end_frame >= b else "no",
                     tag_team, "yes" if tag_team == nxt else "no", "yes" if tag_team == prev else "no"))
    s, e = runs[0][0], runs[-1][1]
    orphan = sp[(sp.end_frame > s) & (sp.start_frame < e) & ~sp.code_id.isin(used)]
    print("\nset-piece/goal tags in the window with no GT dead-ball stretch near them: %d" % len(orphan))
    for r in orphan.itertuples():
        print("   tag %s %s %s %s - %s" % (r.code_id, r.code_u.title(), txt(r.Type), fmt_clock(r.start_frame), fmt_clock(r.end_frame)))


def section_changes(t, d, runs):
    head("4. ground-truth direct possession changes (H<->A) vs tag boundaries (tag minus truth, seconds)")
    tol = CHANGE_S * FPS
    starts = {}
    for team in ("H", "A"):
        prev = t.state.shift(1)
        same_half = t.half == t.half.shift(1)
        mask = (t.state == team) & ~((prev == team) & same_half)
        starts[team] = t[mask]
    trans = d[d.role == "transition"]
    n = 0
    for (a1, b1, l1), (a2, b2, l2) in zip(runs[:-1], runs[1:]):
        if l1 in ("H", "A") and l2 in ("H", "A") and l1 != l2:
            n += 1
            cand = starts[l2][(starts[l2].start_frame - a2).abs() <= tol]
            best = cand.iloc[(cand.start_frame - a2).abs().argmin()] if len(cand) else None
            tr = trans[(trans.polarity == l2) & ((trans.start_frame - a2).abs() <= tol)]
            tbest = tr.iloc[(tr.start_frame - a2).abs().argmin()] if len(tr) else None
            print("GT %s -> %s at %s | nearest tag-module boundary into %s: %s | nearest transition tag: %s"
                  % (l1, l2, fmt_clock(a2), l2,
                     "none within %.0fs" % CHANGE_S if best is None else "%+.1fs (%s, %s)" % (sec(best.start_frame - a2), best.kind, best.strength),
                     "none" if tbest is None else "%+.1fs" % sec(tbest.start_frame - a2)))
    print("direct H<->A changes in the ground truth: %d" % n)
    print("(changes that pass through D are covered in section 3)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=GT_PATH)
    a = ap.parse_args()
    lab = load_gt(a.gt)
    valid = np.flatnonzero(lab != "X")
    s, e = int(valid[0]), int(valid[-1]) + 1
    assert e - s == len(valid), "non-X labels are not one contiguous block"
    print("ground-truth window: frames %d-%d (%s - %s), %d frames" % (s, e - 1, fmt_clock(s), fmt_clock(e), e - s))
    d = prepare(load_events())
    t = build_table(d)
    runs = runs_of(lab, s, e)
    section_gt(runs)
    section_rows(t, lab, s, e)
    section_stoppages(d, runs)
    section_changes(t, d, runs)
