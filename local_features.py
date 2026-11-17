"""Local structure around the ball, spacing of supporting players, and ball-free cluster features.

    python local_features.py        # needs cache/atd.npz, the GT file and the Smart Tagging file

A. BALL VISIBLE: per team, features of the 4 players nearest the ball (whichever are tracked; NaN only if
   too few exist): d1..d4 distance (m); c5/c10/c15 teammates within 5/10/15 m; sp3 mean pairwise distance
   of the 3 nearest (a triangle) and sp4 of the 4 nearest (a diamond); area3 triangle area (m^2); fwd3 mean
   forward offset of the 3 nearest relative to the ball, in the team's attack direction (negative =
   supporters behind the ball).
B. SPACING: percentiles of sp3 / sp4 / area3 / d1-d3 for the team that has the ball versus the other team.
C. BALL-FREE (every 5th graded frame): tri_R = share of teammate triples whose three pairwise distances are
   all <= R metres (R in 8, 10, 12, 15, 20: several radii on purpose, no threshold is chosen);
   edge = median over players of the mean distance to their two nearest teammates; nn1 = median nearest-
   teammate distance.

Compared quantity: home minus away. AUC is raw: > 0.5 = a higher home-minus-away value goes with home
possession, < 0.5 = the opposite. 'equal n' columns keep only frames where both teams have the same
number of tracked players, to check that tracking coverage is not driving a feature.
Public window = independent but only 17 possessions. Whole match = agreement with the tag module's
possession/transition intervals: NOISY labels, possibly circular, NOT accuracy.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atd_explore import LENGTH, WIDTH, load_or_parse  # noqa: E402
from compare_gt import GT_PATH, load_gt  # noqa: E402
from shape_features import BLOCKS, auc, compute_all, get_dirs, row_quantile  # noqa: E402
from tag_diagnostics import load_events, prepare  # noqa: E402
from tag_module import build_table  # noqa: E402

RADII = (8, 10, 12, 15, 20)
STEP = 5                     # sample every 5th graded frame for the ball-free features
CHUNK = 3000
BALL_FEATS = ["d1", "d2", "d3", "d4", "c5", "c10", "c15", "sp3", "sp4", "area3", "fwd3"]
FREE_FEATS = ["tri_%d" % r for r in RADII] + ["edge", "nn1"]


def head(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


def fmt(x):
    return "  n/a" if not np.isfinite(x) else "%5.2f" % x


# ------------------------------------------------------------------------------------ A: ball-visible
def team_ball_feats(X, Y, bx, by, attack_sign):
    dist = np.hypot(X - bx[:, None], Y - by[:, None])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    order = np.argsort(dist, axis=1)[:, :4]
    r = np.arange(len(X))[:, None]
    dn = dist[r, order]
    xs, ys = X[r, order], Y[r, order]
    f = {"d%d" % (k + 1): np.where(np.isfinite(dn[:, k]), dn[:, k], np.nan) for k in range(4)}
    for rad in (5, 10, 15):
        f["c%d" % rad] = (dist < rad).sum(1).astype(float)

    def pair(i, j):
        return np.hypot(xs[:, i] - xs[:, j], ys[:, i] - ys[:, j])
    d01, d02, d12 = pair(0, 1), pair(0, 2), pair(1, 2)
    f["sp3"] = (d01 + d02 + d12) / 3.0
    f["sp4"] = (d01 + d02 + pair(0, 3) + d12 + pair(1, 3) + pair(2, 3)) / 6.0
    f["area3"] = 0.5 * np.abs((xs[:, 1] - xs[:, 0]) * (ys[:, 2] - ys[:, 0]) - (xs[:, 2] - xs[:, 0]) * (ys[:, 1] - ys[:, 0]))
    xa, bxa = (xs, bx[:, None]) if attack_sign > 0 else (LENGTH - xs, LENGTH - bx[:, None])
    f["fwd3"] = (xa[:, :3] - bxa).mean(1)
    return f


def ball_features(P, B, team, dirs):
    """({name: per-team full-length arrays} for home and away); NaN where the ball is not visible."""
    n = P.shape[0]
    out = [{k: np.full(n, np.nan, dtype=np.float32) for k in BALL_FEATS} for _ in (0, 1)]
    for half, first, last in BLOCKS:
        fr = np.arange(first, last + 1)
        fr = fr[np.isfinite(B[fr, 0]) & np.isfinite(B[fr, 1])]
        bx, by = B[fr, 0] * LENGTH, B[fr, 1] * WIDTH
        for t in (0, 1):
            idx = np.flatnonzero(team == t)
            X, Y = P[fr][:, idx, 0] * LENGTH, P[fr][:, idx, 1] * WIDTH
            ok = np.isfinite(X) & np.isfinite(Y)
            X, Y = np.where(ok, X, np.nan), np.where(ok, Y, np.nan)
            f = team_ball_feats(X, Y, bx, by, dirs[half] if t == 0 else -dirs[half])
            for k in BALL_FEATS:
                out[t][k][fr] = f[k]
    return out


# ------------------------------------------------------------------------------------ C: ball-free
def free_chunk(X, Y):
    """Ball-free cluster features for one chunk of frames of one team (X, Y: (frames, slots) in metres)."""
    D = np.hypot(X[:, :, None] - X[:, None, :], Y[:, :, None] - Y[:, None, :])
    k = D.shape[1]
    D[:, np.arange(k), np.arange(k)] = np.nan
    m = np.isfinite(X).sum(1).astype(float)
    comb = m * (m - 1) * (m - 2) / 6.0
    f = {}
    for rad in RADII:
        A = (np.where(np.isfinite(D), D, np.inf) <= rad).astype(np.float32)
        tri = np.einsum("nij,njk,nki->n", A, A, A) / 6.0
        f["tri_%d" % rad] = np.where(comb > 0, tri / np.maximum(comb, 1), np.nan)
    Dinf = np.sort(np.where(np.isfinite(D), D, np.inf), axis=2)
    two = Dinf[:, :, :2]
    valid = np.isfinite(two).all(axis=2)
    f["edge"] = row_quantile(np.where(valid, two.mean(axis=2), np.nan), 0.5)
    f["nn1"] = row_quantile(np.where(np.isfinite(Dinf[:, :, 0]), Dinf[:, :, 0], np.nan), 0.5)
    return f


def free_features(P, team):
    n = P.shape[0]
    out = [{k: np.full(n, np.nan, dtype=np.float32) for k in FREE_FEATS} for _ in (0, 1)]
    for _, first, last in BLOCKS:
        fr = np.arange(first, last + 1, STEP)
        for a in range(0, len(fr), CHUNK):
            ch = fr[a:a + CHUNK]
            for t in (0, 1):
                idx = np.flatnonzero(team == t)
                X, Y = P[ch][:, idx, 0] * LENGTH, P[ch][:, idx, 1] * WIDTH
                ok = np.isfinite(X) & np.isfinite(Y)
                f = free_chunk(np.where(ok, X, np.nan), np.where(ok, Y, np.nan))
                for k in FREE_FEATS:
                    out[t][k][ch] = f[k]
    return out


# ------------------------------------------------------------------------------------ evaluation
def samples(lab_all, B, table, F):
    s0 = np.flatnonzero(lab_all != "X")
    s, e = int(s0[0]), int(s0[-1]) + 1
    pub = np.full(len(B), "", dtype="<U1")
    pub[s:e] = np.where(np.isin(lab_all[s:e].astype(str), ["H", "A"]), lab_all[s:e].astype(str), "")
    ps = np.full(len(B), "", dtype="<U1")
    for r in table[table.state.isin(["H", "A"]) & table.kind.isin(["possession", "transition"])].itertuples():
        ps[r.start_frame:r.end_frame] = r.state
    eq = F["nH"] == F["nA"]
    vis = np.isfinite(B[:, 0]) & np.isfinite(B[:, 1])
    return pub, ps, eq, vis


def auc_table(feats, D, cols, title):
    head(title)
    print("%-8s" % "feature" + "".join("%15s" % c[0][:14] for c in cols))
    for k in feats:
        print("%-8s" % k + "".join("%15s" % fmt(auc(D[k][m], (lab == "H")[m])) for _, m, lab in cols))


def spacing_table(FH, FA, pub, ps):
    head("B. spacing around the ball: team WITH the ball vs the OTHER team (percentiles 10 / 25 / 50 / 75 / 90)")
    for title, lab in (("public window", pub), ("whole match, tag intervals", ps)):
        m = lab != ""
        print("\n%s (%d frames)" % (title, m.sum()))
        for k in ("d1", "d2", "d3", "sp3", "sp4", "area3", "c10", "fwd3"):
            have = np.where(lab == "H", FH[k], FA[k])[m]
            other = np.where(lab == "H", FA[k], FH[k])[m]
            def p(a):
                a = a[np.isfinite(a)]
                return " ".join("%6.1f" % v for v in np.percentile(a, (10, 25, 50, 75, 90))) if a.size else "n/a"
            print("  %-6s with ball: %s | other: %s" % (k, p(have), p(other)))


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    P, B, team, *_ = load_or_parse(False)
    dirs = get_dirs(P, team)
    F = compute_all(P, team, dirs)
    lab_all = load_gt(GT_PATH)
    table = build_table(prepare(load_events()))
    pub, ps, eq, vis = samples(lab_all, B, table, F)

    BH, BA = ball_features(P, B, team, dirs)
    DB = {k: BH[k] - BA[k] for k in BALL_FEATS}
    cols = [("public", pub != "", pub), ("public equal n", (pub != "") & eq, pub),
            ("match", ps != "", ps), ("match equal n", (ps != "") & eq, ps)]
    auc_table(BALL_FEATS, DB, cols, "A. local features around the ball: AUC of (home - away)  [frames with the ball visible only]")
    spacing_table(BH, BA, pub, ps)

    CH, CA = free_features(P, team)
    DC = {k: CH[k] - CA[k] for k in FREE_FEATS}
    cols = [("public", pub != "", pub), ("public equal n", (pub != "") & eq, pub),
            ("match", ps != "", ps), ("match equal n", (ps != "") & eq, ps),
            ("match no ball", (ps != "") & ~vis, ps)]
    auc_table(FREE_FEATS, DC, cols, "C. ball-free cluster features: AUC of (home - away)  [every %dth graded frame]" % STEP)
