"""Team-wide shape and movement features, and how well they separate H from A possession.

    python shape_features.py            # needs cache/atd.npz, the GT file and the Smart Tagging file

Every feature is computed per team from whichever players are tracked at that frame (no minimum, no
slot identities), in attack-normalised coordinates: x' measured toward the team's own attacking goal
(direction from direction.py's kickoff method), y mirrored the same way so left/right are in the
team's own frame. The features compared are DIFFERENCES (home minus away).

Features (metres, m/s): cx centroid (median x'); back / front 10th / 90th percentile of x'; length
front-back; spread_y std of y; iqr_y interquartile range of y (ignores 1-2 players left open on the far
side); lean |median y' - centreline| (how far the team is shifted to one side); surge mean forward
displacement of slots tracked now and 2 s ago (slot identity is stable over 2 s); speed mean of the
ATD speed field; n tracked count (an ARTIFACT DETECTOR: if it predicts possession, tracking coverage
leaks into the other features).

Evaluation is threshold-free (AUC: probability a random H frame scores above a random A frame; 0.5 =
no information, below 0.5 = points the other way) and also counted per possession, because frames inside
a possession are not independent. Pseudo-label results use the tag module's possession/transition
intervals: agreement with noisy labels, NOT accuracy.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import games  # noqa: E402
from atd_explore import FPS, LENGTH, WIDTH, load_or_parse  # noqa: E402
from compare_gt import GT_PATH, load_gt, runs_of  # noqa: E402
from direction import team_median_x  # noqa: E402
from tag_diagnostics import KICKOFF_FRAMES, load_events, prepare  # noqa: E402
from tag_module import build_table  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
BLOCKS = games.half_video_blocks(games.GAME)
SURGE_LAG = 2 * FPS           # 2 s
FEATURES = ["cx", "back", "front", "length", "spread_y", "iqr_y", "lean", "surge", "speed", "n"]
COMPLETE_MIN = 9              # used ONLY to split results into strata, never to drop frames


def head(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


def get_dirs(P, team):
    """{half: +1/-1}: +1 = home attacks toward x = 1. Kickoff method, as in direction.py."""
    dirs = {}
    for half, name in ((1, "1st Half"), (2, "2nd Half")):
        ko = KICKOFF_FRAMES[name]
        (xh, _), (xa, _) = team_median_x(P, team, ko - 25, ko + 1)
        dirs[half] = +1 if xh < xa else -1
    return dirs


def row_quantile(A, q):
    """Per-row quantile ignoring NaN (linear interpolation); fast because it sorts once."""
    S = np.sort(A, axis=1)                                  # NaN sorts to the end
    n = np.isfinite(A).sum(1)
    pos = np.clip((n - 1) * q, 0, None)
    lo, hi = np.floor(pos).astype(int), np.ceil(pos).astype(int)
    r = np.arange(len(A))
    v = S[r, lo] + (S[r, hi] - S[r, lo]) * (pos - lo)
    return np.where(n > 0, v, np.nan)


def team_block(P, team, t, dir_team, s, e):
    """Feature dict for team t over frames [s, e) of one half."""
    idx = np.flatnonzero(team == t)
    def coords(a, b):
        X, Y = P[a:b, idx, 0], P[a:b, idx, 1]
        if dir_team < 0:
            X, Y = 1 - X, 1 - Y
        ok = np.isfinite(X) & np.isfinite(Y)
        return np.where(ok, X * LENGTH, np.nan), np.where(ok, Y * WIDTH, np.nan)
    X, Y = coords(s, e)
    Xp, _ = coords(s - SURGE_LAG, e - SURGE_LAG)
    S = P[s:e, idx, 2]
    f = {"n": np.isfinite(X).sum(1).astype(float)}
    f["cx"] = row_quantile(X, 0.5)
    f["back"], f["front"] = row_quantile(X, 0.1), row_quantile(X, 0.9)
    f["length"] = f["front"] - f["back"]
    f["spread_y"] = np.nanstd(Y, axis=1)
    f["iqr_y"] = row_quantile(Y, 0.75) - row_quantile(Y, 0.25)
    f["lean"] = np.abs(row_quantile(Y, 0.5) - WIDTH / 2.0)
    f["surge"] = np.nanmean(X - Xp, axis=1) / (SURGE_LAG / float(FPS))
    f["speed"] = np.nanmean(S, axis=1)
    return f


def compute_all(P, team, dirs):
    """{feature: (n_frames,) array of home-minus-away, NaN outside the graded blocks} plus nH, nA."""
    n = P.shape[0]
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in FEATURES}
    out["nH"], out["nA"] = np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32)
    for half, first, last in BLOCKS:
        s, e = first, last + 1
        fh = team_block(P, team, 0, dirs[half], s, e)
        fa = team_block(P, team, 1, -dirs[half], s, e)
        for k in FEATURES:
            out[k][s:e] = fh[k] - fa[k]
        out["nH"][s:e], out["nA"][s:e] = fh["n"], fa["n"]
    return out


def auc(score, is_home):
    m = np.isfinite(score)
    s, y = score[m], is_home[m]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fmt(x):
    return "  n/a" if not np.isfinite(x) else "%5.2f" % x


def evaluate_public(F, lab_all):
    head("1. public window: which features separate TRUE H from TRUE A frames?")
    valid = np.flatnonzero(lab_all != "X")
    s, e = int(valid[0]), int(valid[-1]) + 1
    lab = lab_all[s:e].astype(str)
    m = np.isin(lab, ["H", "A"])
    runs = [r for r in runs_of(lab_all, s, e) if r[2] in ("H", "A")]
    long_runs = [r for r in runs if (r[1] - r[0]) >= 5 * FPS]
    print("frames %d (H %d, A %d) | %d possessions (%d lasting >= 5 s)  | AUC > 0.5: higher home-minus-away value goes with home possession"
          % (m.sum(), (lab == "H").sum(), (lab == "A").sum(), len(runs), len(long_runs)))
    print("%-9s %6s %22s %22s" % ("feature", "AUC", "possessions agreeing", "agreeing (>= 5 s)"))
    for k in FEATURES:
        v = F[k][s:e]
        a = auc(v[m], (lab == "H")[m])
        def agree(rs):
            ok = tot = 0
            for a1, b1, l in rs:
                med = np.nanmedian(F[k][a1:b1])
                if np.isfinite(med):
                    tot += 1
                    ok += int((med > 0) == (l == "H"))
            return "%2d of %2d" % (ok, tot)
        print("%-9s %6s %22s %22s" % (k, fmt(a), agree(runs), agree(long_runs)))


def evaluate_pseudo(F, B, table):
    head("2. whole match: agreement with the tag module's possession/transition intervals (NOISY labels, not accuracy)")
    n = len(B)
    lab = np.full(n, "", dtype="<U1")
    rows = table[table.state.isin(["H", "A"]) & table.kind.isin(["possession", "transition"])]
    for r in rows.itertuples():
        lab[r.start_frame:r.end_frame] = r.state
    m = lab != ""
    vis = np.isfinite(B[:, 0]) & np.isfinite(B[:, 1])
    complete = (F["nH"] >= COMPLETE_MIN) & (F["nA"] >= COMPLETE_MIN)
    strata = [("all", m), ("ball visible", m & vis), ("ball missing", m & ~vis),
              ("both teams >= %d tracked" % COMPLETE_MIN, m & complete), ("fewer tracked", m & ~complete)]
    print("labelled frames: %d (H %d, A %d) | in strata: %s"
          % (m.sum(), (lab == "H").sum(), (lab == "A").sum(), ", ".join("%s %d" % (nm, k.sum()) for nm, k in strata[1:])))
    print("%-9s" % "feature" + "".join("%14s" % nm[:13] for nm, _ in strata) + "  %s" % "intervals agreeing")
    for k in FEATURES:
        v = F[k]
        cells = [auc(v[mk], (lab == "H")[mk]) for _, mk in strata]
        ok = tot = 0
        for r in rows.itertuples():
            med = np.nanmedian(v[r.start_frame:r.end_frame])
            if np.isfinite(med):
                tot += 1
                ok += int((med > 0) == (r.state == "H"))
        print("%-9s" % k + "".join("%14s" % fmt(c) for c in cells) + "  %d of %d" % (ok, tot))


if __name__ == "__main__":
    P, B, team, *_ = load_or_parse(False)
    dirs = get_dirs(P, team)
    print("home attack direction by half (+1 = toward x=1): %s" % dirs)
    F = compute_all(P, team, dirs)
    evaluate_public(F, load_gt(GT_PATH))
    evaluate_pseudo(F, B, build_table(prepare(load_events())))
