"""Profile the Automatic Tracking Data: what it contains, how it is missing, and whether `s` is speed.

    python atd_explore.py              # parse raw ATD (first run ~1-3 min), cache it, print diagnostics
    python atd_explore.py --rebuild    # ignore the cache and re-parse

Place in the repo root. Reads the raw ATD exactly the way starter/baseline.py does, but keeps
everything instead of only the nearest team. Prints aggregates to the console only.

The cache (cache/atd.npz) is frame-level tracking: it must NOT be committed or shared
(RULES.md: no derived tracking exports). Add `cache/` to .gitignore.

Arrays are indexed by VIDEO frame (frame_id - 1), one row per frame:
    P[frame, slot, (x, y, s)]  players, x/y normalised 0-1, s as given in the file
    B[frame, (x, y, s)]        ball
    team[slot]                 0 = home (Barcelona), 1 = away, from the ATD metadata
"""
import argparse
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "starter"))
import games  # noqa: E402
from baseline import team_names, team_of_each_track  # noqa: E402
from load_atd import atd_paths  # noqa: E402

GAME = games.GAME
FPS = games.GAMES[GAME]["fps"]
LENGTH, WIDTH = games.GAMES[GAME]["pitch"]
N_FRAMES = games.GAMES[GAME]["n_video_frames"]
BLOCKS = games.half_video_blocks(GAME)           # [(half, first, last)] inclusive
# absolute, so pipelines and notebooks find it from any folder; MTAH_CACHE_DIR lets submit.py rebuild it from raw
CACHE = os.path.join(os.environ.get("MTAH_CACHE_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"), "atd.npz")

# plausibility thresholds, used only to COUNT suspicious values, never to drop anything
PLAYER_MAX_MS = 12.0      # sprint speeds top out below this
BALL_MAX_MS = 40.0        # hardest strikes are roughly 30 m/s
JUMP_M = 3.0              # ball moving > 3 m in one frame (75 m/s) is a tracking jump


# ---------------------------------------------------------------------------------------- parse
def parse(raw_path, meta_path):
    team = team_of_each_track(meta_path)
    P = B = None
    n_slots = None
    stats = Counter()
    ball_tok = Counter()
    t0 = time.time()
    with open(raw_path) as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) != 3:
                stats["bad_lines"] += 1
                continue
            f = int(parts[0]) - 1                       # frame_id counts from 1
            if P is None:
                n_slots = parts[1].count(";") + 1
                P = np.full((N_FRAMES, n_slots, 3), np.nan, dtype=np.float32)
                B = np.full((N_FRAMES, 3), np.nan, dtype=np.float32)
            stats["lines"] += 1
            tok = parts[1].replace(";", ",").split(",")
            if len(tok) == 3 * n_slots:                  # fast path: every group has 3 values
                P[f] = np.array(tok, dtype=np.float32).reshape(n_slots, 3)
                stats["fast"] += 1
            else:                                        # a missing group may be a bare 'NaN'
                stats["slow"] += 1
                for k, g in enumerate(parts[1].split(";")[:n_slots]):
                    if g[:1] != "N":
                        v = g.split(",")
                        P[f, k, :len(v)] = np.array(v[:3], dtype=np.float32)
            b = parts[2].strip()
            if b[:1] == "N":
                ball_tok["missing (starts with N)"] += 1
            else:
                v = b.split(",")
                ball_tok["%d values" % len(v)] += 1
                B[f, :min(3, len(v))] = np.array(v[:3], dtype=np.float32)
    print("parsed %d lines in %.0fs" % (stats["lines"], time.time() - t0))
    return P, B, team, stats, ball_tok


def load_or_parse(rebuild):
    meta, raw = atd_paths()
    if os.path.exists(CACHE) and not rebuild:
        z = np.load(CACHE, allow_pickle=True)
        print("loaded cache %s" % CACHE)
        return z["P"], z["B"], z["team"], dict(z["stats"].item()), dict(z["ball_tok"].item()), meta
    P, B, team, stats, ball_tok = parse(raw, meta)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, P=P, B=B, team=team, stats=np.array(dict(stats), dtype=object),
                        ball_tok=np.array(dict(ball_tok), dtype=object))
    print("cached -> %s (do not commit or share)" % CACHE)
    return P, B, team, dict(stats), dict(ball_tok), meta


# --------------------------------------------------------------------------------------- helpers
def head(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


def graded_mask():
    m = np.zeros(N_FRAMES, dtype=bool)
    for _, first, last in BLOCKS:
        m[first:last + 1] = True
    return m


def runs(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def pct(a, qs=(5, 50, 95)):
    a = np.asarray(a, dtype=float)
    return " ".join("p%d=%.1f" % (q, v) for q, v in zip(qs, np.nanpercentile(a, qs))) if a.size else "n/a"


# ----------------------------------------------------------------------------------- diagnostics
def d1_structure(P, B, team, stats, ball_tok):
    head("1. structure")
    n_slots = P.shape[1]
    print("lines parsed %d of %d video frames | bad lines %d | fast path %d, slow path %d"
          % (stats.get("lines", 0), N_FRAMES, stats.get("bad_lines", 0), stats.get("fast", 0), stats.get("slow", 0)))
    print("slots: %d | slots mapped to teams: %d (home %d, away %d)"
          % (n_slots, len(team), int((team == 0).sum()), int((team == 1).sum())))
    if len(team) != n_slots:
        print("  WARNING: metadata slot count != raw slot count; team mapping is unreliable")
    print("ball group formats: %s" % ball_tok)
    print("x/y outside 0-1: players %.2f%% of observations, ball %.2f%%" % (100 * _out_frac(P[..., :2]), 100 * _out_frac(B[:, :2])))


def _out_frac(a):
    ok = np.isfinite(a)
    return float(((a < 0) | (a > 1))[ok].mean()) if ok.any() else 0.0


def d2_availability(P, B, team):
    head("2. availability on graded frames (players are never all tracked at once)")
    g = graded_mask()
    tracked = np.isfinite(P[..., 0]) & np.isfinite(P[..., 1])          # (n, slots)
    nh, na = tracked[:, team == 0].sum(1), tracked[:, team == 1].sum(1)
    print("tracked players per frame  home: %s | away: %s | total: %s"
          % (pct(nh[g], (5, 50, 95)), pct(na[g], (5, 50, 95)), pct((nh + na)[g], (5, 50, 95))))
    print("frames with >11 tracked for one team (duplicates / non-players?): home %.1f%%, away %.1f%%"
          % (100 * (nh[g] > 11).mean(), 100 * (na[g] > 11).mean()))
    print("frames with < 8 tracked for one team: home %.1f%%, away %.1f%%" % (100 * (nh[g] < 8).mean(), 100 * (na[g] < 8).mean()))
    ball = np.isfinite(B[:, 0]) & np.isfinite(B[:, 1])
    print("ball visible on %.1f%% of graded frames (the data docs say ~59%%)" % (100 * ball[g].mean()))
    play = (nh + na) >= 10
    half_gap = np.flatnonzero(~g & (np.arange(N_FRAMES) > BLOCKS[0][2]) & (np.arange(N_FRAMES) < BLOCKS[1][1]))
    if len(half_gap):
        print("halftime gap: median tracked players %d (play: %d)" % (np.median((nh + na)[half_gap]), np.median((nh + na)[g])))
    both = ball[g]
    print("tracked players when the ball is visible: median %d | when missing: median %d"
          % (np.median((nh + na)[g][both]), np.median((nh + na)[g][~both])))


def d3_slots(P, team):
    head("3. slot persistence (a slot is a tracklet, not a person)")
    g = graded_mask()
    tracked = np.isfinite(P[..., 0]) & np.isfinite(P[..., 1])
    print("slot team occupancy%  segments  median_seg_s  max_seg_s")
    rows = []
    for k in range(P.shape[1]):
        m = tracked[:, k] & g
        a, b = runs(m)
        ln = (b - a) / float(FPS)
        rows.append((k, int(team[k]) if k < len(team) else -1, 100 * m[g].mean(), len(a),
                     float(np.median(ln)) if len(ln) else 0.0, float(ln.max()) if len(ln) else 0.0))
    for r in rows:
        print("%4d %4s %8.1f %9d %13.1f %10.1f" % (r[0], "HA?"[r[1]], r[2], r[3], r[4], r[5]))
    occ = np.array([r[2] for r in rows])
    seg = np.array([r[3] for r in rows])
    print("slots occupied < 5%% of graded frames: %d | total segments across slots: %d | median segment length of a slot (s): %.1f"
          % ((occ < 5).sum(), seg.sum(), np.median([r[4] for r in rows if r[3]])))


def _fd_speed(X, Y):
    v = np.hypot(np.diff(X, axis=0), np.diff(Y, axis=0)) * FPS
    pad = np.full((1,) + v.shape[1:], np.nan, dtype=v.dtype)
    return np.concatenate([pad, v], axis=0)                              # speed from t-1 to t


def d4_speed(P, B):
    head("4. is `s` speed in m/s?  (compare with finite-difference speed from positions in metres)")
    g = graded_mask()
    g2 = g & np.roll(g, 1)
    for name, X, Y, S in (("players", P[..., 0] * LENGTH, P[..., 1] * WIDTH, P[..., 2]),
                          ("ball", B[:, 0] * LENGTH, B[:, 1] * WIDTH, B[:, 2])):
        fd = _fd_speed(X, Y)
        S0 = S
        S1 = np.concatenate([np.full((1,) + S.shape[1:], np.nan, dtype=S.dtype), S[:-1]], axis=0)
        mask = g2.reshape((-1,) + (1,) * (fd.ndim - 1))
        ok = np.isfinite(fd) & np.isfinite(S0) & mask
        print("\n%s: pairs with both values %d" % (name, int(ok.sum())))
        if not ok.any():
            continue
        f, s0 = fd[ok], S0[ok]
        s1 = S1[ok]
        for label, s in (("s_t", s0), ("s_(t-1)", s1)):
            k = np.isfinite(s)
            print("  corr(finite-diff, %s) = %.3f" % (label, np.corrcoef(f[k], s[k])[0, 1]))
        big = f > 1.0
        print("  median s_t / finite-diff where finite-diff > 1 m/s: %.2f  (1.0 => same units, ~3.6 => km/h, ~0.04 => per-frame)"
              % np.median(s0[big] / f[big]))
        if np.isfinite(S[g]).any():
            print("  reported s percentiles (p50/p95/p99.9/max): %.2f %.2f %.2f %.2f"
                  % tuple(np.nanpercentile(S[g], [50, 95, 99.9, 100])))
        lim = PLAYER_MAX_MS if name == "players" else BALL_MAX_MS
        print("  share of reported s above %.0f m/s: %.3f%% | finite-diff above it: %.3f%%"
              % (lim, 100 * (S[g][np.isfinite(S[g])] > lim).mean(), 100 * (f > lim).mean()))
        print("  share of non-NaN positions that have NaN s: %.2f%%" % (100 * (np.isfinite(X[g]) & ~np.isfinite(S[g])).sum() / max(1, np.isfinite(X[g]).sum())))


def d5_ball(B):
    head("5. ball: visibility runs, position range, jumps")
    ok = np.isfinite(B[:, 0]) & np.isfinite(B[:, 1])
    miss, vis = [], []
    for _, first, last in BLOCKS:
        seg = ok[first:last + 1]
        a, b = runs(~seg)
        miss += list((b - a) / float(FPS))
        a, b = runs(seg)
        vis += list((b - a) / float(FPS))
    print("missing runs: n=%d  seconds %s  max %.1f" % (len(miss), pct(miss, (50, 90, 99)), max(miss) if miss else 0))
    print("visible runs: n=%d  seconds %s  max %.1f" % (len(vis), pct(vis, (50, 90, 99)), max(vis) if vis else 0))
    g = graded_mask()
    x, y = B[g & ok, 0], B[g & ok, 1]
    print("ball x range %.3f..%.3f, y range %.3f..%.3f | outside 0-1: x %.2f%%, y %.2f%%"
          % (x.min(), x.max(), y.min(), y.max(), 100 * ((x < 0) | (x > 1)).mean(), 100 * ((y < 0) | (y > 1)).mean()))
    X, Y = B[:, 0] * LENGTH, B[:, 1] * WIDTH
    jump = _fd_speed(X, Y) / FPS
    both = g & np.roll(g, 1) & np.isfinite(jump)
    print("consecutive-frame ball moves > %.0f m: %d of %d (%.3f%%)" % (JUMP_M, int((jump[both] > JUMP_M).sum()), int(both.sum()),
                                                                     100 * (jump[both] > JUMP_M).mean()))


def d6_first5(P, B, team):
    head("6. nearest-player distances, first 5 minutes (public window), metres")
    s, e = 11185, 11185 + 7500
    tracked = np.isfinite(P[s:e, :, 0]) & np.isfinite(P[s:e, :, 1])
    bx, by = B[s:e, 0] * LENGTH, B[s:e, 1] * WIDTH
    dx = P[s:e, :, 0] * LENGTH - bx[:, None]
    dy = P[s:e, :, 1] * WIDTH - by[:, None]
    dist = np.hypot(dx, dy)
    dist[~tracked] = np.inf
    dh = dist[:, team == 0].min(1)
    da = dist[:, team == 1].min(1)
    vis = np.isfinite(bx) & np.isfinite(by)
    print("ball visible %.1f%% of the window" % (100 * vis.mean()))
    for nm, d_ in (("home", dh), ("away", da)):
        v = d_[vis]
        print("nearest %s player to ball: finite for %.1f%% of visible-ball frames; %s" % (nm, 100 * np.isfinite(v).mean(), pct(v[np.isfinite(v)], (5, 25, 50, 75, 95))))


def d7_followups(P, B, team):
    head("7. follow-ups: extra tracks, duplicates, speed alignment, ball speed at longer lags")
    g = graded_mask()
    tracked = np.isfinite(P[..., 0]) & np.isfinite(P[..., 1])
    inside = tracked & (P[..., 0] >= 0) & (P[..., 0] <= 1) & (P[..., 1] >= 0) & (P[..., 1] <= 1)
    tot = tracked.sum(1)
    print("total tracked per graded frame: max %d | frames above 22: %.2f%%" % (tot[g].max(), 100 * (tot[g] > 22).mean()))
    fr = np.flatnonzero(g)[::10]                                   # every 10th graded frame
    for nm, t in (("home", 0), ("away", 1)):
        idx = np.flatnonzero(team == t)
        n, ni = tracked[:, idx].sum(1), inside[:, idx].sum(1)
        over = g & (n > 11)
        print("\n%s: frames with >11 tracked: %.1f%%; in those, inside-pitch count is <= 11 in %.0f%% (=> extras are off-pitch)"
              % (nm, 100 * over[g].mean(), 100 * (ni[over] <= 11).mean() if over.any() else 0))
        X, Y = P[fr][:, idx, 0] * LENGTH, P[fr][:, idx, 1] * WIDTH
        d = np.hypot(X[:, :, None] - X[:, None, :], Y[:, :, None] - Y[:, None, :])
        k = d.shape[1]
        d[:, np.arange(k), np.arange(k)] = np.inf
        d = np.where(np.isfinite(d), d, np.inf)
        dmin = d.min(axis=(1, 2))
        print("  closest same-team slot pair: frames with a pair < 1.0 m: %.1f%%, < 0.5 m: %.1f%%"
              % (100 * (dmin < 1.0).mean(), 100 * (dmin < 0.5).mean()))

    print("\nwhich frame does `s` belong to? corr of s_t with displacement over (t-1->t), (t->t+1), centred (t-1->t+1)")
    for name, X, Y, S in (("players", P[..., 0] * LENGTH, P[..., 1] * WIDTH, P[..., 2]),
                          ("ball", B[:, 0] * LENGTH, B[:, 1] * WIDTH, B[:, 2])):
        back = np.full_like(X, np.nan)
        fwd = np.full_like(X, np.nan)
        cen = np.full_like(X, np.nan)
        back[1:] = np.hypot(X[1:] - X[:-1], Y[1:] - Y[:-1]) * FPS
        fwd[:-1] = np.hypot(X[1:] - X[:-1], Y[1:] - Y[:-1]) * FPS
        cen[1:-1] = np.hypot(X[2:] - X[:-2], Y[2:] - Y[:-2]) * FPS / 2.0
        gg = g.reshape((-1,) + (1,) * (X.ndim - 1))
        out = []
        for lab, v in (("back", back), ("fwd", fwd), ("centred", cen)):
            m = gg & np.isfinite(v) & np.isfinite(S)
            out.append("%s %.3f" % (lab, np.corrcoef(v[m], S[m])[0, 1] if m.sum() > 10 else np.nan))
        print("  %-8s %s" % (name, " | ".join(out)))

    print("\nball: reported s against speed measured over longer baselines (lag in frames)")
    X, Y, S = B[:, 0] * LENGTH, B[:, 1] * WIDTH, B[:, 2]
    fin = S[g & np.isfinite(S)]
    if fin.size:
        print("  reported ball s: share exactly 0: %.1f%% | p50/p95/p99/max: %.1f %.1f %.1f %.1f"
              % (100 * (fin == 0).mean(), *np.percentile(fin, [50, 95, 99, 100])))
    for lag in (1, 3, 5, 10):
        v = np.full_like(X, np.nan)
        v[lag:] = np.hypot(X[lag:] - X[:-lag], Y[lag:] - Y[:-lag]) * FPS / lag
        m = g & np.roll(g, lag) & np.isfinite(v) & np.isfinite(S)
        big = m & (v > 1.0)
        if m.sum() > 10:
            print("  lag %2d: pairs %6d | corr %.3f | median s/measured (measured > 1 m/s): %.2f"
                  % (lag, int(m.sum()), np.corrcoef(v[m], S[m])[0, 1], np.median(S[big] / v[big]) if big.any() else np.nan))


def d8_ball_s(P, B, team):
    head("8. what is the ball's third value?  test: does it equal a player's speed at the same frame?")
    g = graded_mask()
    vis = g & np.isfinite(B[:, 0]) & np.isfinite(B[:, 1]) & np.isfinite(B[:, 2])
    fr = np.flatnonzero(vis)[::5]
    X, Y, S = P[fr, :, 0] * LENGTH, P[fr, :, 1] * WIDTH, P[fr, :, 2]
    ok = np.isfinite(X) & np.isfinite(Y) & np.isfinite(S)
    bx, by, bs = B[fr, 0] * LENGTH, B[fr, 1] * WIDTH, B[fr, 2]
    dist = np.hypot(X - bx[:, None], Y - by[:, None])
    dist[~ok] = np.inf
    order = np.argsort(dist, axis=1)
    rank_of = np.argsort(order, axis=1)                         # rank_of[f, slot] = 0 for the nearest slot
    nearest = order[:, 0]
    s_near = S[np.arange(len(fr)), nearest]
    k = np.isfinite(s_near)
    print("frames tested: %d (every 5th graded frame with ball position and s)" % len(fr))
    print("corr(ball s, speed of the NEAREST tracked player) = %.3f" % np.corrcoef(bs[k], s_near[k])[0, 1])
    tol = 0.01
    def match_stats(bvals):
        diff = np.abs(np.where(ok, S, np.inf) - bvals[:, None])
        best = diff.argmin(axis=1)
        hit = diff.min(axis=1) < tol
        return hit, rank_of[np.arange(len(fr)), best]
    hit, rk = match_stats(bs)
    hit_c, _ = match_stats(np.random.default_rng(0).permutation(bs))
    print("ball s equals some player's s (|diff| < %.2f m/s): %.1f%% of frames | shuffled-ball control: %.1f%%"
          % (tol, 100 * hit.mean(), 100 * hit_c.mean()))
    if hit.any():
        print("  when it matches, that player's distance rank to the ball: nearest %.0f%%, top 3 %.0f%%, top 5 %.0f%%"
              % (100 * (rk[hit] == 0).mean(), 100 * (rk[hit] < 3).mean(), 100 * (rk[hit] < 5).mean()))
        t_hit = team[np.argmin(np.abs(np.where(ok, S, np.inf) - bs[:, None]), axis=1)]
        print("  matching player's team: home %.0f%%, away %.0f%%" % (100 * (t_hit[hit] == 0).mean(), 100 * (t_hit[hit] == 1).mean()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    np.seterr(invalid="ignore", divide="ignore")     # NaN-heavy data: comparisons warn needlessly
    P, B, team, stats, ball_tok, meta = load_or_parse(a.rebuild)
    print("teams: H = %s, A = %s | pitch %.1f x %.1f m | %d fps" % (*team_names(meta), LENGTH, WIDTH, FPS))
    d1_structure(P, B, team, stats, ball_tok)
    d2_availability(P, B, team)
    d3_slots(P, team)
    d4_speed(P, B)
    d5_ball(B)
    d6_first5(P, B, team)
    d7_followups(P, B, team)
    d8_ball_s(P, B, team)