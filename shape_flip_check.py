"""Shape layer for missing-ball stretches: hide the ball in visible stretches and see how well 'hold, flip on sustained shape' works.

    python shape_flip_check.py                         # reads evaluation/results/control_touch_v13_cap.csv

Reference = the decoder's own labels on stretches where the BALL IS VISIBLE and the label is H or A for >= MIN_RUN_S seconds
(about 350 changes over the match). In each stretch the ball is treated as hidden: the state starts as the decoder's label at the
start of the stretch and follows this rule:
    hold the current team; flip to the other team when the shape score has favoured it for T_FLIP seconds in a row
    (and |score| >= MARGIN); the flip is dated at the START of that sustained run.
Shape score (never uses the ball): mean of the standardised  edge, length, spread_y, surge  (home minus away, oriented so + favours home),
smoothed over 3 s. Standardisation uses whole-match mean/std, no labels. 'speed' is left out (no information without the ball).
Reported over a grid of T_FLIP and MARGIN:
  agree     share of frames where the simulated label equals the decoder's
  flips     number of flips made / number of decoder changes in these stretches
  precision share of flips within [-3 s, +10 s] of a decoder change
  recall    share of decoder changes with a flip in that range
  delay     median seconds from the decoder change to the matched flip
This is agreement with the decoder's labels (which are ~80% right in the public window), not with the truth.
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
from local_features import free_features  # noqa: E402
from shape_features import compute_all, get_dirs  # noqa: E402

GRID = 5
SMOOTH_S = 3
SIGN = {"edge": 1, "length": 1, "spread_y": 1, "surge": 1}
MIN_RUN_S = 10
TOL_BEFORE_S, TOL_AFTER_S = 3, 10
T_FLIPS = (3, 5, 8)
MARGINS = (0.0, 0.3, 0.6)


def moving_mean(x, w):
    fin = np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(fin, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(fin)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w // 2), np.minimum(len(x), i + w // 2 + 1)
    c = cn[hi] - cn[lo]
    return np.where(c > 0, (cs[hi] - cs[lo]) / np.maximum(c, 1), np.nan)


def shape_scores(P, team, blocks):
    dirs = get_dirs(P, team)
    F = compute_all(P, team, dirs)
    CH, CA = free_features(P, team)
    D = {"edge": CH["edge"] - CA["edge"], "length": F["length"], "spread_y": F["spread_y"], "surge": F["surge"]}
    grids = [np.arange(f, l + 1, GRID) for _, f, l in blocks]
    stats = {k: (np.nanmean(np.concatenate([D[k][g] for g in grids])), np.nanstd(np.concatenate([D[k][g] for g in grids]))) for k in SIGN}
    out = []
    for g in grids:
        z = np.vstack([SIGN[k] * (D[k][g] - stats[k][0]) / stats[k][1] for k in SIGN])
        out.append(moving_mean(np.nanmean(z, axis=0), max(1, int(SMOOTH_S * FPS / GRID))))
    return grids, out


def simulate(S, init, t_pts, margin):
    """Hold-and-flip over one stretch. S: scores; init: +1 home / -1 away. Returns the state series and flip indices."""
    n = len(S)
    state = np.empty(n, dtype=np.int8)
    cur, i = init, 0
    flips = []
    state[:] = cur
    while i < n:
        want = -cur
        if np.isfinite(S[i]) and np.sign(S[i]) == want and abs(S[i]) >= margin:
            j = i
            while j < n and np.isfinite(S[j]) and np.sign(S[j]) == want and abs(S[j]) >= margin:
                j += 1
            if j - i >= t_pts:
                cur = want
                flips.append(i)
                state[i:] = cur
                i = j
                continue
            i = j
        else:
            i += 1
    return state, flips


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=os.path.join("evaluation", "results", "control_touch_v13_cap.csv"))
    ap.add_argument("--min-run", type=float, default=MIN_RUN_S, help="shortest ball-visible H/A stretch used, in seconds")
    a_ = ap.parse_args()
    MIN_RUN_S = a_.min_run
    np.seterr(invalid="ignore")
    lab = pd.read_csv(a_.sub)["label"].astype(str).to_numpy()
    P, B, team, *_ = load_or_parse(False)
    blocks = games.half_video_blocks(games.GAME)
    vis = np.isfinite(B[:, 0]) & np.isfinite(B[:, 1])
    grids, scores = shape_scores(P, team, blocks)

    stretches = []                                   # (S, decoder state series, decoder change indices)
    for g, S in zip(grids, scores):
        ok = vis[g] & np.isin(lab[g], ["H", "A"])
        e = np.diff(np.concatenate(([0], ok.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            if (b - a) * GRID / FPS >= MIN_RUN_S:
                ref = np.where(lab[g[a:b]] == "H", 1, -1).astype(np.int8)
                chg = np.flatnonzero(ref[1:] != ref[:-1]) + 1
                stretches.append((S[a:b], ref, chg))
    if not stretches:
        print("no ball-visible stretches of at least %g s with an H/A label were found" % MIN_RUN_S)
        sys.exit(0)
    nstr = len(stretches)
    nchg = sum(len(c) for _, _, c in stretches)
    frames = sum(len(r) for _, r, _ in stretches)
    allS = np.concatenate([np.abs(s[np.isfinite(s)]) for s, _, _ in stretches])
    print("%d hidden-ball stretches (>= %d s, ball visible, H/A label): %.0f minutes, %d decoder changes" % (nstr, MIN_RUN_S, frames * GRID / FPS / 60, nchg))
    print("|score| percentiles (10/25/50/75/90): %s  (a margin is only meaningful relative to these)" % " ".join("%.2f" % x for x in np.percentile(allS, (10, 25, 50, 75, 90))))
    print("\n%-7s %-7s %8s %14s %10s %8s %9s" % ("T_FLIP", "margin", "agree", "flips/changes", "precision", "recall", "delay s"))
    tb, ta = TOL_BEFORE_S * FPS // GRID, TOL_AFTER_S * FPS // GRID
    for tf in T_FLIPS:
        for mg in MARGINS:
            agree = nfl = ok_fl = ok_ch = 0
            tot = 0
            delays = []
            for S, ref, chg in stretches:
                st, fl = simulate(S, ref[0], int(tf * FPS / GRID), mg)
                agree += int((st == ref).sum())
                tot += len(ref)
                nfl += len(fl)
                used = set()
                for f in fl:
                    m = [c for c in chg if -tb <= f - c <= ta and c not in used]
                    if m:
                        c = min(m, key=lambda c: abs(f - c))
                        used.add(c)
                        ok_fl += 1
                        delays.append((f - c) * GRID / FPS)
                ok_ch += len(used)
            print("%-7s %-7.1f %7.1f%% %7d/%-6d %9.0f%% %7.0f%% %9s"
                  % ("%d s" % tf, mg, 100 * agree / tot, nfl, nchg, 100 * ok_fl / max(1, nfl), 100 * ok_ch / max(1, nchg),
                     "%.1f" % np.median(delays) if delays else "-"))
    print("\nA flip is 'matched' if it falls within [-%d s, +%d s] of a decoder change. If precision is low, the layer invents changes;"
          "\nif recall is low, it misses them. The sustained run requirement is the hysteresis; the delay shows how late the layer is." % (TOL_BEFORE_S, TOL_AFTER_S))
