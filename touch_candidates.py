"""Candidate ball touches from ATD (ball velocity change with a player close by) - a DIAGNOSTIC.

    python touch_candidates.py                      # grid + changes, public window only
    python touch_candidates.py --list               # also list every touch (to check against the video)
    python touch_candidates.py --tau 3 --r 1.5      # the setting used for sections B and C

A candidate touch at frame t needs the ball visible at t-k, t and t+k (k is searched over several values:
the first real-data run showed the ball track is too smooth for k = 2); velocity in = (p_t - p_{t-k})/k and
out = (p_{t+k} - p_t)/k (m/s). It is a touch candidate when |v_out - v_in| >= tau AND the nearest tracked
player to the ball, taken over t-1..t+1, is within r metres. Candidates closer than max(NMS, 2k) frames are merged
(the strongest is kept). The touch's team is the team of that nearest player. Heading change (degrees) is
reported when both speeds exceed 1 m/s: a one-touch redirect shows as a large heading change at a similar speed.

No threshold is chosen here: section A shows a grid. Only the public ground truth (17 possessions) is used,
so nothing here can tune anything; it only tells us whether touches look usable.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atd_explore import FPS, LENGTH, WIDTH, load_or_parse  # noqa: E402
from compare_gt import GT_PATH, load_gt, runs_of  # noqa: E402
from tag_diagnostics import fmt_clock  # noqa: E402

K = 5                         # default frames either side for the velocity estimate (the real ball track is smooth)
KS = (2, 5, 8, 12)            # window half-widths searched in section A
NMS = 5                       # merge candidates closer than this many frames (at least k)
TAUS = (2.0, 4.0, 8.0)        # m/s
RADII = (1.5, 3.0)            # m
SEARCH_BEFORE_S, SEARCH_AFTER_S = 3, 5


def head(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


def detect(P, B, team, a, b, tau, r, k=K, nms=None):
    nms = max(NMS, 2 * k) if nms is None else nms
    """Candidate touches among frames [a, b): dict of arrays (frame, team 0/1, dv, turn, v_in, v_out, dist)."""
    w = max(1, k // 2)                                   # window for the nearest-player search
    pad = k + w
    A, Z = a - pad, b + pad
    bx, by = B[A:Z, 0] * LENGTH, B[A:Z, 1] * WIDTH
    X, Y = P[A:Z, :, 0] * LENGTH, P[A:Z, :, 1] * WIDTH
    d = np.hypot(X - bx[:, None], Y - by[:, None])
    d = np.where(np.isfinite(d), d, np.inf)
    j = d.argmin(1)
    nd = d[np.arange(len(d)), j]
    nt = team[j]
    nd = np.where(np.isfinite(nd), nd, np.inf)
    i = np.arange(pad, len(bx) - pad)
    p0, p1, p2 = (np.column_stack([bx[i + s], by[i + s]]) for s in (-k, 0, k))
    vin, vout = (p1 - p0) * FPS / k, (p2 - p1) * FPS / k
    dv = np.hypot(*(vout - vin).T)
    sin, sout = np.hypot(*vin.T), np.hypot(*vout.T)
    cosang = (vin * vout).sum(1) / np.maximum(sin * sout, 1e-9)
    turn = np.where((sin > 1) & (sout > 1), np.degrees(np.arccos(np.clip(cosang, -1, 1))), np.nan)
    offs = range(-w, w + 1)
    nearw = np.stack([nd[i + o] for o in offs])
    pick = nearw.argmin(0)
    dmin = nearw[pick, np.arange(len(i))]
    tm = np.stack([nt[i + o] for o in offs])[pick, np.arange(len(i))]
    ok = np.isfinite(dv) & (dv >= tau) & (dmin <= r)
    cand = np.flatnonzero(ok)
    keep = []
    for c in cand[np.argsort(-dv[cand])]:
        if all(abs(c - q) > nms for q in keep):
            keep.append(c)
    keep = np.array(sorted(keep), dtype=int)
    return {"frame": A + i[keep], "team": tm[keep], "dv": dv[keep], "turn": turn[keep],
            "v_in": sin[keep], "v_out": sout[keep], "dist": dmin[keep]}


def letter(t):
    return np.where(np.asarray(t) == 0, "H", "A")


def section_character(B, s, e):
    head("0. character of the ball track in the public window")
    bx, by = B[s - 15:e + 15, 0] * LENGTH, B[s - 15:e + 15, 1] * WIDTH
    both = np.isfinite(bx[1:]) & np.isfinite(bx[:-1])
    same = (bx[1:] == bx[:-1]) & (by[1:] == by[:-1]) & both
    print("consecutive visible frames with the ball at EXACTLY the same position: %.1f%% of %d pairs (a low update rate would show many)"
          % (100 * same.sum() / max(1, both.sum()), both.sum()))
    print("|velocity change| over +-k frames, all visible frames (m/s):  k   p50   p90   p99   max")
    for k in (1, 2, 5, 8, 12):
        i = np.arange(k, len(bx) - k)
        p0, p1, p2 = (np.column_stack([bx[i + o], by[i + o]]) for o in (-k, 0, k))
        dv = np.hypot(*(((p2 - p1) - (p1 - p0)) * FPS / k).T)
        dv = dv[np.isfinite(dv)]
        print("%52d %5.1f %5.1f %5.1f %5.1f" % ((k,) + tuple(np.percentile(dv, (50, 90, 99, 100)))))


def section_grid(P, B, team, s, e, lab):
    head("A. candidate touches in the public window, over a grid of thresholds (no setting is chosen)")
    print("window %d frames = %.1f min | touch team = team of the nearest player | on H/A frames, 'agree' = touch team equals the TRUE possession then"
          % (e - s, (e - s) / FPS / 60))
    print("%-4s %-6s %-6s %8s %9s %12s %12s" % ("k", "tau", "r (m)", "touches", "per min", "agree (H/A)", "on D frames"))
    for k in KS:
      for tau in TAUS:
        for r in RADII:
            T = detect(P, B, team, s, e, tau, r, k=k)
            L = lab[T["frame"] - s]
            tl = letter(T["team"])
            ha = np.isin(L, ["H", "A"])
            agree = (tl[ha] == L[ha]).mean() if ha.any() else np.nan
            print("%-4d %-6.1f %-6.1f %8d %9.1f %11.0f%% %11.0f%%" % (k, tau, r, len(T["frame"]), len(T["frame"]) / ((e - s) / FPS / 60.0),
                                                                100 * agree, 100 * (L == "D").mean() if len(L) else np.nan))


def section_changes(P, B, team, s, e, lab_all, tau, r, k=K):
    head("B. at each true H<->A change: touches by the NEW team around it (k=%d, tau=%.1f m/s, r=%.1f m)" % (k, tau, r))
    T = detect(P, B, team, s, e, tau, r, k=k)
    tl = letter(T["team"])
    runs = runs_of(lab_all, s, e)
    ch = [(a2, l1, l2) for (a1, b1, l1), (a2, b2, l2) in zip(runs[:-1], runs[1:]) if l1 in ("H", "A") and l2 in ("H", "A") and l1 != l2]
    print("%d direct changes. offset = seconds from the true change to the first touch by the new team in [-%d s, +%d s]; "
          "'2nd' = another new-team touch within 2 s after it with no old-team touch between" % (len(ch), SEARCH_BEFORE_S, SEARCH_AFTER_S))
    print("%-18s %-8s %9s %5s %-30s" % ("change", "at", "offset s", "2nd", "touches within -3..+5 s (team@offset)"))
    offs, seconds = [], 0
    for b, l1, l2 in ch:
        m = (T["frame"] >= b - SEARCH_BEFORE_S * FPS) & (T["frame"] <= b + SEARCH_AFTER_S * FPS)
        fr, who = T["frame"][m], tl[m]
        new = np.flatnonzero(who == l2)
        off, second = None, False
        if len(new):
            f0 = new[0]
            off = (fr[f0] - b) / float(FPS)
            after = (fr > fr[f0]) & (fr <= fr[f0] + 2 * FPS)
            second = bool(after.any() and (who[after][0] == l2))
            offs.append(off)
            seconds += int(second)
        seq = " ".join("%s@%+.1f" % (w, (f - b) / float(FPS)) for f, w in zip(fr, who))
        print("%-18s %-8s %9s %5s %-30s" % ("%s->%s" % (l1, l2), fmt_clock(b)[2:9], "-" if off is None else "%+.1f" % off,
                                           "-" if off is None else ("yes" if second else "no"), seq))
    if offs:
        o = np.array(offs)
        print("\nchanges with a new-team touch in the window: %d of %d | within +-1 s: %d | median offset %.1f s | with a confirming 2nd touch: %d"
              % (len(o), len(ch), int((np.abs(o) <= 1).sum()), np.median(o), seconds))


def section_list(P, B, team, s, e, lab, tau, r, k=K):
    head("C. every candidate touch in the public window (k=%d, tau=%.1f, r=%.1f)" % (k, tau, r))
    T = detect(P, B, team, s, e, tau, r, k=k)
    print("%-11s %5s %5s %7s %8s %8s %9s %6s" % ("clock", "team", "true", "dv m/s", "speed in", "speed out", "turn deg", "dist"))
    for f, t, dv, tn, vi, vo, ds in zip(T["frame"], letter(T["team"]), T["dv"], T["turn"], T["v_in"], T["v_out"], T["dist"]):
        print("%-11s %5s %5s %7.1f %8.1f %8.1f %9s %6.1f" % (fmt_clock(f)[2:], t, lab[f - s], dv, vi, vo, "-" if not np.isfinite(tn) else "%.0f" % tn, ds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--r", type=float, default=3.0)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    np.seterr(invalid="ignore")
    P, B, team, *_ = load_or_parse(False)
    lab_all = load_gt(GT_PATH)
    v = np.flatnonzero(lab_all != "X")
    s, e = int(v[0]), int(v[-1]) + 1
    lab = lab_all[s:e].astype(str)
    section_character(B, s, e)
    section_grid(P, B, team, s, e, lab)
    section_changes(P, B, team, s, e, lab_all, a.tau, a.r, a.k)
    if a.list:
        section_list(P, B, team, s, e, lab, a.tau, a.r, a.k)