"""Which rule started each dead-ball (D) run, and what does the ball look like around that start?  (label-free, aggregate output)

    python dead_start_trace.py                                   # summary + match-wide checks for evaluation/results/control_touch_v17.csv
    python dead_start_trace.py --sample 10 --seed 0              # same 10 tags as tag_timing.py --sample 10 --seed 0, one block each
    python dead_start_trace.py --list --csv cache/trace.csv      # every eligible tag
    python dead_start_trace.py --coverage                        # for every eligible tag: is there D at tag start + 2 s?  For the misses: which physical cues exist, and what would a start-to-restart rule add?
    python dead_start_trace.py --events 0:44:58 0:45:35          # decode a short window with the v17 settings: events, raw D runs, why each ended, what post-processing does
    python dead_start_trace.py --window 0:44:54 0:45:54          # 3 s bins of aggregates for a clock range (repeat --window for several); no tags needed

The decoder (control_touch_decoder.decode) can START a dead ball in only three ways; this script recomputes those events from the
tracking with the decoder's own functions and constants and says which of them coincides (+-2 frames) with the start of each D run:
  OUT          ball outside the lines for >= N_OFF frames                          (off_runs)
  BALL+SLOW    ball outside OR untracked, and smoothed mean player speed <= V_DEAD_PS for >= N_DEAD frames
  SLOW         smoothed mean player speed <= V_DEAD_PS for >= N_LONG frames, ball ignored
  none         none of the three: the run was moved by another step (fill-gap, goal-d, pre-kickoff, a dropped run)
Per tag it also prints, for the nearest D run:
  * if OUT: the outside run (length, furthest distance beyond the line, how it ends: ball back inside / ball untracked, ball speed
    on leaving and on re-entry) and the other outside runs in [tag -30 s, tag +10 s];
  * if the ball is untracked at the start: the last visible ball before it (seconds before, signed distance to the nearest line,
    closing speed toward the line, ball speed);
  * mean player speed 2 s before / after the start;
  * the end: players' speed recovery (as tag_timing.py), whether the ball is visible then, and ball visibility in the last 5 s.
Match-wide (all graded frames, no labels) it shows how often outside runs of each length sit near a set-piece tag, and whether
ball dropouts that begin close to a line and heading toward it are followed by a D start.
Only aggregates are printed: no frame-by-frame ball or player rows.  Tags are incomplete, so 'near a tag' is a hint, not a verdict.
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
import tag_timing as tt  # noqa: E402
from atd_explore import FPS, LENGTH, WIDTH, load_or_parse  # noqa: E402
from control_touch_decoder import N_DEAD, N_LONG, N_OFF, PRE_TAG, V_DEAD_PS, mask_runs, off_runs  # noqa: E402
from d_audit import d_runs  # noqa: E402
from dead_ball_evidence import signals  # noqa: E402
from tag_diagnostics import fmt_clock, load_events, prepare  # noqa: E402

TOL = 2                            # frames: an event 'fires' a D start if it starts this close to it
TAG_BEFORE, TAG_AFTER = 10, 40     # an outside run is 'near a tag' if a tag starts between 10 s before and 40 s after it begins
DROP_MIN_S = 2.0                   # ball dropouts (visible -> untracked) of at least this length
NEAR_LINE_M = 3.0
REST_V, REST_S, LAUNCH_V, WIN_F = 1.0, 0.8, 6.0, 12   # restart-like event: ball rests (<= REST_V m/s for REST_S s), then gets faster than LAUNCH_V within WIN_F frames or is lost
LAST_VIS_BACK_S = 30                                 # how far back to look for the last visible ball before an untracked start
BINS = ((10 / 25.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 1e9))


def head(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


def signed_dist(x, y):
    """Metres from the pitch rectangle: > 0 outside (distance to it), < 0 inside (minus the distance to the nearest line)."""
    dx = np.maximum(np.maximum(-x, x - LENGTH), 0.0)
    dy = np.maximum(np.maximum(-y, y - WIDTH), 0.0)
    inside = -np.minimum(np.minimum(x, LENGTH - x), np.minimum(y, WIDTH - y))
    return np.where((dx > 0) | (dy > 0), np.hypot(dx, dy), inside)


def start_events(G):
    slow = G["ps"] <= V_DEAD_PS                      # NaN -> False, as in the decoder
    return {"OUT": off_runs(G["out"]),
            "BALL+SLOW": mask_runs((G["out"] | G["missing"]) & slow, N_DEAD),
            "SLOW": mask_runs(slow, N_LONG)}


def fired_by(s, EV):
    return [k for k, runs in EV.items() if any(abs(a - s) <= TOL for a, _ in runs)]


def last_visible(vis, s, back_s=10):
    lo = max(0, s - int(back_s * FPS))
    idx = np.flatnonzero(vis[lo:s])
    return None if not len(idx) else lo + int(idx[-1])


def closing_speed(sd, f):
    """m/s the signed distance grows over the 5 frames up to f (positive = heading outward)."""
    return float((sd[f] - sd[f - 5]) * FPS / 5.0) if f >= 5 and np.isfinite(sd[f]) and np.isfinite(sd[f - 5]) else np.nan


def restart_events(vis, bsp, n):
    """[(frame, 'launch' | 'vanish', peak speed)]: the ball rests visible (<= REST_V m/s for >= REST_S s) and then either exceeds LAUNCH_V
    within WIN_F frames ('launch': a kick / throw seen) or is untracked within WIN_F frames ('vanish': a kick that leaves the tracked area).
    Independent of the decoder's touch detector, which needs the ball visible 8 frames on both sides of the contact."""
    rest = vis & (bsp <= REST_V)
    e = np.diff(np.concatenate(([0], rest.astype(np.int8), [0])))
    out = []
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if b - a < REST_S * FPS or b >= n - WIN_F:
            continue
        win = bsp[b:b + WIN_F]
        peak = float(np.nanmax(win)) if np.isfinite(win).any() else np.nan
        if np.isfinite(peak) and peak >= LAUNCH_V:
            out.append((int(b), "launch", peak))
        elif (~vis[b:b + WIN_F]).any():
            out.append((int(b), "vanish", np.nan))
    return out


def out_run_containing(runs, s):
    for a, b in runs:
        if a - TOL <= s < b:
            return a, b
    return None


def out_profile(run, G, sd, bsp, n):
    a, b = run
    ends = "end of data" if b >= n else ("ball back inside" if G["vis"][b] else "ball untracked")
    return {"len": (b - a) / FPS, "far": float(np.nanmax(sd[a:b])), "ends": ends,
            "exit": float(bsp[min(n - 1, a + 5)]), "reentry": float(bsp[min(n - 1, b + 5)]) if ends == "ball back inside" else np.nan}


def mean_ps(ps, a, b):
    seg = ps[max(0, a):max(0, b)]
    return float(np.nanmean(seg)) if np.isfinite(seg).any() else np.nan


def trace_row(x, G, EV, sd, bsp, n, RS=()):
    """Extend a tag_timing.tag_table row with the trigger and the ball profile; None fields when there is no D run."""
    x = dict(x)
    x.update(restarts=[], fired=None, prof=None, last=None, ps_before=np.nan, ps_after=np.nan, vis_run=np.nan, rec_vis=None, vis_end=np.nan, outs=[])
    t0 = x["tag_start"]
    x["outs"] = [(a, (b - a) / FPS) for a, b in EV["OUT"] if t0 - 30 * FPS <= a <= t0 + 10 * FPS]
    if x["d_start"] is None:
        return x
    s, e = x["d_start"], x["d_end"]
    x["fired"] = fired_by(s, EV)
    r = out_run_containing(EV["OUT"], s) if "OUT" in x["fired"] else None
    if r:
        x["prof"] = out_profile(r, G, sd, bsp, n)
        x["prof"]["start"] = r[0]
    if not G["vis"][s]:
        lv = last_visible(G["vis"], s, LAST_VIS_BACK_S)
        if lv is not None:
            x["last"] = {"before": (s - lv) / FPS, "sd": float(sd[lv]), "close": closing_speed(sd, lv), "speed": float(bsp[lv])}
    x["restarts"] = [r for r in RS if s + 2 * FPS < r[0] < e]
    x["ps_before"], x["ps_after"] = mean_ps(G["ps"], s - 2 * FPS, s), mean_ps(G["ps"], s, s + 2 * FPS)
    x["vis_run"] = float(G["vis"][s:e].mean())
    x["vis_end"] = float(G["vis"][max(s, e - 5 * FPS):e].mean())
    if x["recover"] is not None:
        x["rec_vis"] = bool(G["vis"][x["recover"]])
    return x


def fmt_trigger(x):
    if x["d_start"] is None:
        return "no D run"
    f = x["fired"]
    return "+".join(f) if f else "none (moved by fill-gap / goal-d / pre-kickoff / ...)"


def print_blocks(rows):
    for i, x in enumerate(rows, 1):
        t0 = x["tag_start"]
        print("#%-2d tag %s  f=%d  %-12s Team=%s" % (i, tt.clk(t0), t0, x["type"], x["team"]))
        if x["d_start"] is None:
            print("    our D       none within +-%d s of the tag start\n" % tt.SEARCH_S)
            continue
        s, e = x["d_start"], x["d_end"]
        print("    our D       %s (%s) -> %s (%s)  length %.1f s | ball visible in the run %.0f%%" % (tt.clk(s), tt.rel(s, t0), tt.clk(e), tt.rel(e, t0), (e - s) / FPS, 100 * x["vis_run"]))
        print("    START fired by: %s%s" % (fmt_trigger(x), "" if G_VIS_AT(s) else "   [ball untracked at the start]"))
        p = x["prof"]
        if p:
            print("      outside run: %s, %.1f s long, furthest %.1f m beyond the line, ends: %s; ball speed leaving %.1f m/s, re-entering %s"
                  % (tt.clk(p["start"]), p["len"], p["far"], p["ends"], p["exit"], "n/a" if not np.isfinite(p["reentry"]) else "%.1f m/s" % p["reentry"]))
        if x["last"]:
            l = x["last"]
            print("      last visible ball %.1f s before the start: %.1f m %s the line, closing speed %+.1f m/s (+ = heading out), ball speed %.1f m/s"
                  % (l["before"], abs(l["sd"]), "outside" if l["sd"] > 0 else "inside", l["close"], l["speed"]))
        others = [(a, ln) for a, ln in x["outs"] if not (p and a == p["start"])]
        print("      other outside runs in [-30 s, +10 s]: %s" % ("none" if not others else "; ".join("%s (%.1f s)" % (tt.rel(a, t0), ln) for a, ln in others[:5])))
        print("      mean player speed 2 s before / after the start: %.2f / %.2f m/s" % (x["ps_before"], x["ps_after"]))
        if x["restarts"]:
            print("      restart-like events inside the run (ball at rest, then fast or lost), after the 2 s dwell: %s"
                  % "; ".join("%s %s (%s from our D end)" % (k, tt.rel(f, t0), tt.rel(f, e)) for f, k, _ in x["restarts"][:4]))
        else:
            print("      restart-like events inside the run after the 2 s dwell: none")
        rec = "no recovery found" if x["recover"] is None else "%s (%s from our D end), ball %s then" % (tt.clk(x["recover"]), tt.rel(x["recover"], e), "visible" if x["rec_vis"] else "untracked")
        print("    END: players recover %s; ball visible in the last 5 s of the run: %.0f%%\n" % (rec, 100 * x["vis_end"]))


def print_summary(rows):
    print("%-12s %5s %7s | %-6s %-10s %-6s %-6s | %-22s %-12s" % ("type", "tags", "has D", "OUT", "BALL+SLOW", "SLOW", "none", "OUT run len s (med, IQR)", "OUT < 2 s"))
    for kind in tt.TYPES + ("all",):
        g = [x for x in rows if kind == "all" or x["type"] == kind]
        w = [x for x in g if x["d_start"] is not None]
        cnt = lambda k: sum(1 for x in w if k in (x["fired"] or []))
        none = sum(1 for x in w if not x["fired"])
        lens = [x["prof"]["len"] for x in w if x["prof"]]
        short = sum(1 for v in lens if v < 2.0)
        print("%-12s %5d %7d | %-6d %-10d %-6d %-6d | %-22s %-12s" % (kind, len(g), len(w), cnt("OUT"), cnt("BALL+SLOW"), cnt("SLOW"), none,
                                                                   ("%.1f (%.1f to %.1f)" % (np.median(lens), np.percentile(lens, 25), np.percentile(lens, 75))) if lens else "n/a", "%d of %d" % (short, len(lens))))
    print("a start can be fired by several rules at once, so the OUT / BALL+SLOW / SLOW columns can add to more than 'has D'.")
    w = [x for x in rows if x["d_start"] is not None and x["prof"]]
    if w:
        ends = pd.Series([x["prof"]["ends"] for x in w]).value_counts().to_dict()
        print("OUT-fired starts: how the outside run ends: %s" % ends)
    wr = [x for x in rows if x["d_start"] is not None]
    early = [x for x in wr if x["restarts"] and (x["d_end"] - x["restarts"][0][0]) > 2 * FPS]
    first = [(x["restarts"][0][0] - x["tag_start"]) / FPS for x in wr if x["restarts"]]
    print("restart-like events (ball at rest, then fast or lost): %d of %d D runs contain one after the 2 s dwell; in %d of them the first event is more than 2 s before our D end."
          % (sum(1 for x in wr if x["restarts"]), len(wr), len(early)))
    if first:
        print("first such event relative to the tag start (s): %s; within tag +3..+12 s in %d of %d" % (tt.med(first), sum(1 for v in first if 3 <= v <= 12), len(first)))
    rec = [x for x in rows if x["d_start"] is not None and x["recover"] is not None and (x["d_end"] - x["recover"]) > 5 * FPS]
    if rec:
        print("D ends more than 5 s AFTER the players' speed recovered in %d of %d runs; the ball is visible at that recovery in %d of them."
              % (len(rec), sum(1 for x in rows if x["d_start"] is not None), sum(1 for x in rec if x["rec_vis"])))


def match_wide(G, EV, sd, bsp, lab, graded, sp_starts, n):
    stoppage_candidates(G, EV, bsp, lab, graded, sp_starts)
    head("match-wide A: outside runs (>= %d frames) by length: how many start a D, and how many sit near a set-piece tag" % N_OFF)
    st = np.sort(sp_starts)

    def near_tag(f):
        i = np.searchsorted(st, f - TAG_BEFORE * FPS)
        return i < len(st) and st[i] <= f + TAG_AFTER * FPS
    runs = [(a, b) for a, b in EV["OUT"] if graded[a]]
    dstarts = np.array([a for a, _ in d_runs(lab, games.half_video_blocks(games.GAME))])
    print("%-10s %6s %11s | %-28s | %-28s" % ("length", "runs", "D starts at", "ends: ball back inside", "ends: ball untracked"))
    for lo, hi in BINS:
        sel = [(a, b) for a, b in runs if lo <= (b - a) / FPS < hi]
        if not sel:
            continue
        ds = sum(1 for a, _ in sel if len(dstarts) and np.abs(dstarts - a).min() <= TOL)
        cells = []
        for inside in (True, False):
            g = [(a, b) for a, b in sel if (b < n and bool(G["vis"][b])) == inside]
            cells.append("%3d runs, %3.0f%% near a tag" % (len(g), 100 * sum(1 for a, _ in g if near_tag(a)) / len(g)) if g else "  0 runs")
        print("%-10s %6d %10.0f%% | %-28s | %-28s" % ("%g-%s s" % (lo, "end" if hi > 1e8 else "%g" % hi), len(sel), 100 * ds / len(sel), cells[0], cells[1]))
    print("'near a tag' = a non-kickoff set-piece tag starts from %d s before to %d s after the run begins. Tags are incomplete (most throw-ins/goal kicks"
          "\nare not all tagged), so compare the cells with each other. A cell with a clearly lower 'near a tag' share than its neighbours is the likely artefact class." % (TAG_BEFORE, TAG_AFTER))

    head("match-wide B: ball dropouts (visible -> untracked for >= %.0f s) by where the ball was last seen" % DROP_MIN_S)
    miss = ~G["vis"] & graded
    e = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
    drops = [(a, b) for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)) if (b - a) >= DROP_MIN_S * FPS and a > 5]
    cats = {"outside the lines": [], "inside, near a line, heading out": [], "inside, near a line, not heading out": [], "elsewhere": []}
    for a, b in drops:
        lv = a - 1
        cs = closing_speed(sd, lv)
        if not np.isfinite(sd[lv]):
            continue
        if sd[lv] > 0:
            key = "outside the lines"
        elif sd[lv] >= -NEAR_LINE_M:
            key = "inside, near a line, heading out" if (np.isfinite(cs) and cs > 0) else "inside, near a line, not heading out"
        else:
            key = "elsewhere"
        cats[key].append((a, b))
    print("%-38s %9s %14s %22s" % ("last seen", "dropouts", "near a tag", "D at some point -2..+15 s"))
    for k, v in cats.items():
        if not v:
            continue
        nt = sum(1 for a, _ in v if near_tag(a))
        dd = sum(1 for a, _ in v if (lab[max(0, a - 2 * FPS):a + 15 * FPS] == "D").any())
        print("%-38s %9d %13.0f%% %21.0f%%" % (k, len(v), 100 * nt / len(v), 100 * dd / len(v)))
    print("near a line = within %.0f m inside the boundary. If 'inside, near a line, heading out' dropouts are near a tag far more often than 'elsewhere',"
          "\nthe last visible position is a usable physical cue for the true start when the ball is untracked." % NEAR_LINE_M)


def parse_clock(txt):
    sec = 0.0
    for part in txt.split(":"):
        sec = sec * 60 + float(part)
    return int(round(sec * FPS))


def window_profile(G, bsp, EV, lab, a, b, bin_s=3):
    """3 s bins of aggregates over [a, b): why did (or didn't) a D start here?  No frame-by-frame rows are printed."""
    ps, vis, out = G["ps"], G["vis"], G["out"]
    rest = vis & (bsp <= REST_V)
    head("window %s - %s (frames %d-%d), %d s bins" % (fmt_clock(a), fmt_clock(b), a, b, bin_s))
    print("%-10s %5s %5s %5s %9s %6s %10s %-6s %s" % ("from", "vis", "out", "rest", "ball m/s", "ps", "ps/prev10s", "ours", "events starting here"))
    for f in range(a, b, bin_s * FPS):
        e = min(b, f + bin_s * FPS)
        sp = bsp[f:e][vis[f:e]]
        msp = float(np.nanmedian(sp)) if np.isfinite(sp).any() else np.nan
        p = mean_ps(ps, f, e)
        prev = mean_ps(ps, f - 10 * FPS, f)
        seg = lab[f:e]
        top = max("HAD", key=lambda c: (seg == c).sum())
        ev = "".join(sym for name, sym in (("OUT", "O"), ("BALL+SLOW", "B"), ("SLOW", "S")) if any(f <= st < e for st, _ in EV[name]))
        print("%-10s %4.0f%% %4.0f%% %4.0f%% %9s %6.2f %10s %-6s %s" % (
            fmt_clock(f), 100 * vis[f:e].mean(), 100 * out[f:e].mean(), 100 * rest[f:e].mean(), "-" if not np.isfinite(msp) else "%.1f" % msp,
            p, "-" if not (np.isfinite(p) and np.isfinite(prev) and prev > 0) else "%.2f" % (p / prev), "%s%3.0f" % (top, 100 * (seg == top).mean()), ev))
    long_ = lambda m: tt.longest_run(m, a, b, 0)[1] / FPS
    print("longest stretch in the window (s): smoothed player speed <= 1.3 m/s: %.1f | <= 1.6: %.1f | <= 2.0: %.1f | ball untracked: %.1f | ball visible and at rest: %.1f"
          % (long_(ps <= 1.3), long_(ps <= 1.6), long_(ps <= 2.0), long_(~vis), long_(rest)))
    print("lowest 3 s mean of smoothed player speed in the window: %.2f m/s" % np.nanmin([mean_ps(ps, f, min(b, f + 3 * FPS)) for f in range(a, b, FPS)]))
    for name, runs in EV.items():
        hit = [(st, sp_) for st, sp_ in runs if a <= st < b]
        print("%-10s events starting in the window: %s" % (name, "none" if not hit else "; ".join("%s (%.1f s)" % (fmt_clock(st)[:10], (sp_ - st) / FPS) for st, sp_ in hit)))
    d = d_runs(lab, games.half_video_blocks(games.GAME))
    ov = [(x, y) for x, y in d if y > a and x < b]
    print("our D runs overlapping the window: %s" % ("none" if not ov else "; ".join("%s - %s" % (fmt_clock(x)[:10], fmt_clock(y)[:10]) for x, y in ov)))
    print("columns: vis = ball tracked, out = ball outside the lines, rest = ball visible and <= %.0f m/s, ps = mean smoothed player speed, ps/prev10s = that over the previous 10 s;"
          "\n events: O = ball-outside run, B = ball outside/untracked + slow players, S = slow players alone (the decoder's three start rules)." % REST_V)


def segments(arr, off=0):
    out, i = [], 0
    while i < len(arr):
        j = i
        while j < len(arr) and arr[j] == arr[i]:
            j += 1
        out.append((off + i, off + j, str(arr[i])))
        i = j
    return out


def event_trace(P, B, team, G, lab_final, a, b, sp_starts, goal_windows, min_unbacked_s=6):
    """Decode [a, b) with the v17 decoder settings (no shape layer, no goal events, no second touch) and explain every D run:
    which events exist, what started and ended each raw D run, and what --d-min-unbacked would do to it."""
    import make_submission as ms          # imported here: it pulls in the whole decoder stack
    ev = ms.evidence(P, B, team, a, b)
    ctrl = [(a + s_, a + e_, "HA"[t]) for s_, e_, t in ms.control_runs(ev["ctrl"])]
    T = ms.detect(P, B, team, a, b, ms.TOUCH_TAU, ms.TOUCH_R, k=ms.TOUCH_K)
    touches = [(int(f), "HA"[int(t)]) for f, t in zip(T["frame"], T["team"])]
    slow = G["ps"][a:b] <= V_DEAD_PS
    dead = {"O": off_runs(ev["off"]), "B": mask_runs((G["out"][a:b] | G["missing"][a:b]) & slow, N_DEAD), "S": mask_runs(slow, N_LONG)}
    events = [(s_, "control %s starts (%.1f s long)" % (t, (e_ - s_) / FPS)) for s_, e_, t in ctrl]
    events += [(f, "touch by %s" % t) for f, t in touches]
    events += [(a + s_, "D-start rule %s fires (run %.1f s)" % (k, (e_ - s_) / FPS)) for k, runs in dead.items() for s_, e_ in runs]
    events.sort()
    head("events %s - %s (frames %d-%d): v17 decoder settings (no shape layer / goal events / second touch)" % (fmt_clock(a), fmt_clock(b), a, b))
    print("rules: O = ball outside >= %d frames; B = ball outside/untracked and slow players for %.0f s; S = slow players alone for %.0f s. "
          "A D run ENDS at the first control-run start or touch once 2 s have passed." % (N_OFF, N_DEAD / FPS, N_LONG / FPS))
    for f, txt in events:
        print("  %s  %s" % (fmt_clock(f)[:10], txt))
    raw, info = ms.decode_block(P, B, team, G, a, b, None, None, None, second_touch=False)
    raw = raw.copy()
    first = min((f for f, _ in events), default=a) - a
    raw[:first] = "?"                                       # before the first event the window start state is unknown
    head("raw decode of this window (before fill-gap / d-min-unbacked); '?' = before the first event")
    for x, y, l in segments(raw, a):
        print("  %s - %s  %5.1f s  %s" % (fmt_clock(x)[:10], fmt_clock(y)[:10], (y - x) / FPS, l))
    head("why each raw D run ended, and what post-processing does to it")
    st = np.sort(np.asarray(sp_starts))
    at_end = {}
    for f, txt in events:
        at_end.setdefault(f, []).append(txt)
    d_found = False
    for x, y, l in segments(raw, a):
        if l != "D":
            continue
        d_found = True
        ended = "; ".join(t for t in at_end.get(y, []) if t.startswith(("control", "touch"))) or ("the window end" if y >= b else "no control/touch event at that frame")
        i = np.searchsorted(st, x - 10 * FPS)
        backed = bool(i < len(st) and st[i] <= y + 10 * FPS)
        exempt = any(x < ge and y > gs for gs, ge in goal_windows)
        short = (y - x) < min_unbacked_s * FPS
        verdict = "kept" if not (short and not backed and not exempt) else "DROPPED by --d-min-unbacked %g (shorter than %g s, no set-piece tag start within 10 s, not a goal window)" % (min_unbacked_s, min_unbacked_s)
        fin = float((lab_final[x:y] == "D").mean())
        print("  D %s - %s (%.1f s): ended by: %s | tag within 10 s: %s | post-processing: %s | D share in the submission file: %.0f%%"
              % (fmt_clock(x)[:10], fmt_clock(y)[:10], (y - x) / FPS, ended, "yes" if backed else "no", verdict, 100 * fin))
    if not d_found:
        print("  no D run in the raw decode of this window")
    head("the submission file's labels in this window")
    for x, y, l in segments(lab_final[a:b], a):
        print("  %s - %s  %5.1f s  %s" % (fmt_clock(x)[:10], fmt_clock(y)[:10], (y - x) / FPS, l))


def stoppage_candidates(G, EV, bsp, lab, graded, sp_starts, top=20):
    """Slow-player runs (the decoder's rule S: smoothed mean player speed <= V_DEAD_PS for >= N_LONG frames), split by how much of each the submission labels D,
    and by how long the ball is VISIBLY AT REST inside it.  A long slow-player run with the ball at rest and little D is a missed or truncated stoppage."""
    head("match-wide C: slow-player runs (>= %.0f s at <= %.1f m/s) by how much of them our labels call D, and how long the ball rests inside" % (N_LONG / FPS, V_DEAD_PS))
    rest = G["vis"] & (bsp <= REST_V)
    st = np.sort(np.asarray(sp_starts))
    rows = []
    for a, b in EV["SLOW"]:
        if not graded[a]:
            continue
        ln = (b - a) / FPS
        cov = float((lab[a:b] == "D").mean())
        rs = float(rest[a:b].sum()) / FPS
        i = np.searchsorted(st, a - 15 * FPS)
        tag = bool(i < len(st) and st[i] <= b + 15 * FPS)
        rows.append(dict(a=a, b=b, len=ln, cov=cov, rest=rs, unc=(1 - cov) * ln, tag=tag, vis=float(G["vis"][a:b].mean())))
    print("%-22s %6s %9s %11s %12s %13s" % ("share of the run labelled D", "runs", "seconds", "median len", "ball rest>=3s", "near a tag"))
    for lo, hi, nm in ((0, 0.2, "under 20%"), (0.2, 0.8, "20% to 80%"), (0.8, 1.01, "over 80%")):
        g = [r for r in rows if lo <= r["cov"] < hi]
        if g:
            print("%-22s %6d %9.0f %10.1fs %11.0f%% %12.0f%%" % (nm, len(g), sum(r["len"] for r in g), np.median([r["len"] for r in g]),
                                                               100 * np.mean([r["rest"] >= 3 for r in g]), 100 * np.mean([r["tag"] for r in g])))
    cand = [r for r in rows if r["rest"] >= 3 and r["unc"] >= 3]
    ncand = int(graded.sum())
    print("\ncandidates = slow-player run with the ball visibly at rest for >= 3 s inside it and >= 3 s of it NOT labelled D: %d runs, %.0f s uncovered (%.2f%% of graded frames), %d of them near a set-piece tag"
          % (len(cand), sum(r["unc"] for r in cand), 100 * sum(r["unc"] for r in cand) * FPS / max(1, ncand), sum(1 for r in cand if r["tag"])))
    print("%-12s %7s %8s %9s %10s %8s %s" % ("run starts", "length", "ball rest", "labelled D", "uncovered", "ball vis", "set-piece tag within 15 s"))
    for r in sorted(cand, key=lambda r: -r["unc"])[:top]:
        print("%-12s %6.1fs %7.1fs %8.0f%% %9.1fs %7.0f%% %s" % (fmt_clock(r["a"])[:10], r["len"], r["rest"], 100 * r["cov"], r["unc"], 100 * r["vis"], "yes" if r["tag"] else "no"))
    print("these are the stoppages to check in the video: players slow and the ball at rest, yet little or none of it labelled D (the 0:45:03 foul is the model case).")


def spot_class(x, y):
    """Which kind of restart spot the point (metres) is near: corner arc <= 4 m, goal area <= 12 m from a goal-line centre, touchline <= 3 m."""
    corner = min(np.hypot(cx - x, cy - y) for cx in (0, LENGTH) for cy in (0, WIDTH))
    goal = min(np.hypot(gx - x, WIDTH / 2.0 - y) for gx in (0, LENGTH))
    line = min(y, WIDTH - y)
    if corner <= 4:
        return "corner arc", corner
    if goal <= 12:
        return "goal area", goal
    if line <= 3:
        return "touchline", line
    return "elsewhere", min(corner, goal, line)


TYPE_LINES = {"goal kick": "end", "corner kick": "end", "throw-in": "side"}        # which line the ball must be lost / out at, from the laws of the game
TYPE_SPOT = {"goal kick": "goal area", "corner kick": "corner arc", "throw-in": "touchline"}


def near_line_kind(kind, x, y):
    """True if (x, y) (metres) is beyond, or within NEAR_LINE_M inside, an END line (x = 0 / LENGTH) or a SIDE line (y = 0 / WIDTH)."""
    sd = max(-x, x - LENGTH) if kind == "end" else max(-y, y - WIDTH)
    return bool(np.isfinite(sd) and sd >= -NEAR_LINE_M)


OUT_BACK_S = 25      # a ball-outside run counts as the start cue of a tag only if it begins at most this long before the tag start


def coverage_rows(rows, G, EV, sd, bsp, RS, lab, P, B, team, n, out_back_s=OUT_BACK_S, type_gate=True):
    """Per eligible tag: D at tag start + 2 s?  For the misses: the physical cues in [tag - 35 s, tag + 3 s], the restart candidates in [tag + 3 s, tag + 12 s]
    and the seconds a rule 'D from the best start cue to the restart' would add.  No labels from the video are used."""
    import make_submission as ms
    rest = G["vis"] & (bsp <= REST_V)
    e = np.diff(np.concatenate(([0], rest.astype(np.int8), [0])))
    rest_runs = [(int(a), int(b)) for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)) if b - a >= 3 * FPS]
    slow = G["ps"] <= V_DEAD_PS
    miss_ball = ~G["vis"]
    em = np.diff(np.concatenate(([0], miss_ball.astype(np.int8), [0])))
    drops = [(int(a), int(b)) for a, b in zip(np.flatnonzero(em == 1), np.flatnonzero(em == -1)) if b - a >= 2 * FPS]
    res = []
    for x in rows:
        t0 = x["tag_start"]
        r = dict(x)
        r["cov"] = bool(lab[t0 + 2 * FPS] == "D")
        r["share"] = float((lab[t0 + FPS:t0 + 6 * FPS] == "D").mean())
        lo, hi = t0 - 35 * FPS, t0 + 3 * FPS
        kind = TYPE_LINES.get(r["type"])
        outs_all = [(a, b) for a, b in EV["OUT"] if t0 - out_back_s * FPS <= a <= hi]
        lead_all = [a for a, b in EV["OUT"] if t0 - 40 * FPS <= a <= hi]
        r["lead_out"] = (t0 - max(lead_all)) / FPS if lead_all else None
        lead_g = [a for a, b in EV["OUT"] if t0 - 40 * FPS <= a <= hi and kind and near_line_kind(kind, B[a, 0] * LENGTH, B[a, 1] * WIDTH)]
        r["lead_out_typed"] = (t0 - max(lead_g)) / FPS if lead_g else None
        outs = [(a, b) for a, b in outs_all if type_gate and kind and near_line_kind(kind, B[a, 0] * LENGTH, B[a, 1] * WIDTH)] if type_gate else outs_all
        r["out"] = max(outs) if outs else None                              # the latest outside run before the tag, at the line this tag type needs
        r["rejected"] = []
        if outs_all and not outs:
            r["rejected"].append("ball-outside run at the wrong line for a %s" % r["type"])
        dr_all = [(a, b) for a, b in drops if lo <= a <= hi and a > 5]
        dr = [(a, b) for a, b in dr_all if not (type_gate and kind) or near_line_kind(kind, B[a - 1, 0] * LENGTH, B[a - 1, 1] * WIDTH)]
        if dr_all and not dr:
            r["rejected"].append("ball lost at the wrong line for a %s" % r["type"])
        r["drop"] = None
        if dr:
            a, b = max(dr)
            lv = a - 1
            if np.isfinite(sd[lv]):
                r["drop"] = (a, b, float(sd[lv]), closing_speed(sd, lv))
        rr = [(a, b) for a, b in rest_runs if a < t0 + 8 * FPS and b > t0 - 2 * FPS]
        r["rest"] = None
        if rr:
            a, b = max(rr, key=lambda ab: ab[1] - ab[0])
            mx, my = float(np.nanmedian(B[a:b, 0])) * LENGTH, float(np.nanmedian(B[a:b, 1])) * WIDTH
            r["rest"] = (a, b, spot_class(mx, my))
            if type_gate and TYPE_SPOT.get(r["type"]) and r["rest"][2][0] != TYPE_SPOT[r["type"]]:
                r["rejected"].append("ball at rest at the %s, wrong spot for a %s" % (r["rest"][2][0], r["type"]))
                r["rest"] = None
        r["slow"] = tt.longest_run(slow, lo, t0 + 8 * FPS, 0)[1] / FPS
        T = ms.detect(P, B, team, t0, t0 + 15 * FPS, ms.TOUCH_TAU, ms.TOUCH_R, k=ms.TOUCH_K)
        tf = [int(f) for f in T["frame"] if f >= t0 + 3 * FPS]
        rs = [f for f, _, _ in RS if t0 + 3 * FPS <= f <= t0 + 12 * FPS]
        cands = sorted([(f, "touch") for f in tf if f <= t0 + 12 * FPS] + [(f, "launch/vanish") for f in rs])
        r["restart"] = cands[0] if cands else None
        # start cue for the estimate: ball-out run, else a ball lost near/outside a line, else a rest at a restart spot, else the tag start
        s_est, why = t0, "the tag start (no physical start cue)"
        if r["out"]:
            s_est, why = r["out"][0], "the ball-outside run"
        elif r["drop"] and (r["drop"][2] > 0 or (r["drop"][2] >= -NEAR_LINE_M and r["drop"][3] > 0)):
            s_est, why = r["drop"][0], "the ball lost near / outside a line"
        elif r["rest"] and r["rest"][2][0] != "elsewhere":
            s_est, why = r["rest"][0], "the ball at rest at a restart spot"
        e_est, ewhy = (r["restart"][0], r["restart"][1]) if r["restart"] else (t0 + 7 * FPS, "the +7 s prior")
        r["est"] = (max(0, e_est - s_est) / FPS, why, ewhy)
        res.append(r)
    return res


def coverage_report(res, G, n):
    graded_s = sum(l - f + 1 for _, f, l in games.half_video_blocks(games.GAME)) / float(FPS)
    head("tag coverage: is there D at tag start + 2 s?  (%d eligible tags; the stoppage normally lasts until the restart, tag + 6-8 s)" % len(res))
    print("%-12s %5s %9s %8s" % ("type", "tags", "D at +2 s", "missed"))
    for kind in tt.TYPES + ("all",):
        g = [r for r in res if kind == "all" or r["type"] == kind]
        c = sum(r["cov"] for r in g)
        print("%-12s %5d %9d %8d" % (kind, len(g), c, len(g) - c))
    miss = [r for r in res if not r["cov"]]
    head("the %d tags with no D at tag + 2 s: cues in [tag - 35 s, tag + 3 s], restart candidates in [tag + 3 s, tag + 12 s]" % len(miss))
    for i, r in enumerate(miss, 1):
        t0 = r["tag_start"]
        near = "none within +-%d s" % tt.SEARCH_S if r["d_start"] is None else "%s -> %s" % (tt.rel(r["d_start"], t0), tt.rel(r["d_end"], t0))
        o = r["out"]
        d = r["drop"]
        rs_ = r["rest"]
        print("#%-2d tag %s f=%d %-11s | nearest D: %s" % (i, tt.clk(t0), t0, r["type"], near))
        print("      ball-outside run (<= %d s before the tag): %s | ball lost (>= 2 s): %s | ball at rest >= 3 s across the tag start: %s"
              % (OUT_BACK_S, "none" if not o else "%s, %.1f s, ends %s" % (tt.rel(o[0], t0), (o[1] - o[0]) / FPS, "back inside" if G["vis"][o[1]] else "untracked"),
                 "none" if not d else "%s, last seen %.1f m %s the line, closing %+.1f m/s" % (tt.rel(d[0], t0), abs(d[2]), "outside" if d[2] > 0 else "inside", d[3] if np.isfinite(d[3]) else 0.0),
                 "none" if not rs_ else "%.1f s from %s at the %s (%.1f m)" % ((rs_[1] - rs_[0]) / FPS, tt.rel(rs_[0], t0), rs_[2][0], rs_[2][1])))
        if r["rejected"]:
            print("      cues ignored because they are at the wrong line / spot for the tag type: %s" % "; ".join(r["rejected"]))
        print("      slow players: longest stretch <= %.1f m/s is %.1f s | restart candidate: %s | a start-to-restart rule would add about %.0f s (from %s to %s)"
              % (V_DEAD_PS, r["slow"], "none" if not r["restart"] else "%s %s" % (r["restart"][1], tt.rel(r["restart"][0], t0)), r["est"][0], r["est"][1], r["est"][2]))
    if miss:
        cnt = lambda f: sum(1 for r in miss if f(r))
        none_ = cnt(lambda r: not r["out"] and not (r["drop"] and (r["drop"][2] > 0 or (r["drop"][2] >= -NEAR_LINE_M and r["drop"][3] > 0))) and not (r["rest"] and r["rest"][2][0] != "elsewhere"))
        tot = sum(r["est"][0] for r in miss)
        head("summary of the misses")
        print("a ball-outside run exists: %d | ball lost near or outside a line: %d | ball at rest at a restart spot: %d | none of the three: %d (of %d misses)"
              % (cnt(lambda r: r["out"]), cnt(lambda r: r["drop"] and (r["drop"][2] > 0 or (r["drop"][2] >= -NEAR_LINE_M and r["drop"][3] > 0))),
                 cnt(lambda r: r["rest"] and r["rest"][2][0] != "elsewhere"), none_, len(miss)))
        print("slow players >= 4 s in the window: %d | a restart candidate in tag +3..+12 s: %d" % (cnt(lambda r: r["slow"] >= 4), cnt(lambda r: r["restart"])))
        print("estimated seconds a start-to-restart rule would add: %.0f s in total (median %.0f s per miss; %.2f%% of graded frames), counting tags that share a stoppage separately"
              % (tot, np.median([r["est"][0] for r in miss]), 100 * tot / graded_s))
    ld = [r["lead_out"] for r in res if r.get("lead_out") is not None]
    if ld:
        print("\nball-outside lead time over all tags (tag start minus the latest outside run within 40 s before it, any line): %s s (n = %d); %.0f%% of them are within %d s"
              % (tt.med(ld), len(ld), 100 * np.mean([v <= OUT_BACK_S for v in ld]), OUT_BACK_S))
    lg = [r["lead_out_typed"] for r in res if r.get("lead_out_typed") is not None]
    if lg:
        print("same, only outside runs at the line the tag type needs (end line for goal kick / corner, side line for throw-in): %s s (n = %d); %.0f%% within %d s; %d of %d have the run more than %d s before the tag"
              % (tt.med(lg), len(lg), 100 * np.mean([v <= OUT_BACK_S for v in lg]), OUT_BACK_S, sum(1 for v in lg if v > OUT_BACK_S), len(lg), OUT_BACK_S))
    rc = [(r["restart"][0] - r["tag_start"]) / FPS for r in res if r["restart"]]
    print("\nfirst restart candidate relative to the tag start, all tags: %s (n = %d of %d)" % (tt.med(rc) if rc else "n/a", len(rc), len(res)))
    print("video-checked restarts relative to their tag start: +5.7 s (goal kick 1:39:11) and +8.4 s (goal kick at frame 85058); the handoff median for the end of the dead ball is tag + 7.2 s.")


def write_csv(rows, path):
    cols = ["tag_clock", "tag_frame", "type", "team", "d_start_clock", "d_end_clock", "fired_by", "out_len_s", "out_far_m", "out_ends", "last_vis_before_s",
            "last_vis_sd_m", "last_vis_close_ms", "ps_before", "ps_after", "recover_rel_end_s", "recover_ball_visible", "vis_end_5s"]
    f = lambda fr: "" if fr is None else fmt_clock(fr)
    out = []
    for x in rows:
        p, l = x["prof"], x["last"]
        out.append([f(x["tag_start"]), x["tag_start"], x["type"], x["team"], f(x["d_start"]), f(x["d_end"]), fmt_trigger(x),
                    "" if not p else "%.1f" % p["len"], "" if not p else "%.1f" % p["far"], "" if not p else p["ends"],
                    "" if not l else "%.1f" % l["before"], "" if not l else "%.1f" % l["sd"], "" if not l else "%.1f" % l["close"],
                    "" if x["d_start"] is None else "%.2f" % x["ps_before"], "" if x["d_start"] is None else "%.2f" % x["ps_after"],
                    "" if x["recover"] is None or x["d_end"] is None else "%.1f" % ((x["recover"] - x["d_end"]) / FPS),
                    "" if x["rec_vis"] is None else int(x["rec_vis"]), "" if x["d_start"] is None else "%.2f" % x["vis_end"]])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(out, columns=cols).to_csv(path, index=False)
    print("wrote %s (local only, do not commit)" % path)


VIS_ARR = None


def G_VIS_AT(f):
    return bool(VIS_ARR[f])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=os.path.join("evaluation", "results", "control_touch_v17.csv"))
    ap.add_argument("--list", action="store_true", help="a block per eligible tag")
    ap.add_argument("--sample", type=int, default=0, help="the same seeded stratified sample as tag_timing.py --sample N --seed S")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-public", action="store_true")
    ap.add_argument("--no-match-wide", action="store_true")
    ap.add_argument("--coverage", action="store_true", help="tag coverage: D at tag + 2 s, cues for the misses, estimated footprint of a start-to-restart rule")
    ap.add_argument("--events", nargs=2, action="append", metavar=("START", "END"), help="decode this clock range with the v17 settings and explain every D run in it (use 20-40 s ranges)")
    ap.add_argument("--window", nargs=2, action="append", metavar=("START", "END"), help="video clock H:MM:SS or M:SS; prints 3 s bins of aggregates for that range, then exits")
    ap.add_argument("--csv")
    a_ = ap.parse_args()
    np.seterr(invalid="ignore")
    lab = pd.read_csv(a_.sub)["label"].astype(str).to_numpy()
    n = len(lab)
    P, B, team, *_ = load_or_parse(False)
    G = signals(P, B)
    VIS_ARR = G["vis"]
    bsp = tt.ball_speed(B)
    sd = signed_dist(B[:, 0] * LENGTH, B[:, 1] * WIDTH)
    blocks = games.half_video_blocks(games.GAME)
    graded = np.zeros(n, dtype=bool)
    for _, f, l in blocks:
        graded[f:l + 1] = True
    d = prepare(load_events())
    sp_all = d[(d.code_u == "SET PIECES") & ~d.type_l.str.startswith("kick off")]
    runs = d_runs(lab, blocks)
    EV = start_events(G)
    if a_.window or a_.events:
        for w0, w1 in (a_.window or []):
            window_profile(G, bsp, EV, lab, parse_clock(w0), parse_clock(w1))
        if a_.events:
            dd = prepare(load_events())
            sps = dd[(dd.code_u == "SET PIECES") & ~dd.type_l.str.startswith("kick off")].start_frame.to_numpy()
            gws = [(r.start_frame - PRE_TAG, r.end_frame) for r in dd[dd.code_u == "GOALS"].itertuples()]
            for w0, w1 in a_.events:
                event_trace(P, B, team, G, lab, parse_clock(w0), parse_clock(w1), sps, gws)
        sys.exit(0)
    base = tt.tag_table(tt.eligible_tags(sp_all, a_.include_public), runs, G, bsp, n)
    RS = restart_events(G["vis"], bsp, n)
    rows_all = [trace_row(x, G, EV, sd, bsp, n, RS) for x in base]
    if a_.coverage:
        coverage_report(coverage_rows(base, G, EV, sd, bsp, RS, lab, P, B, team, n), G, n)
        sys.exit(0)
    head("D-start triggers over all %d eligible tags (public window %s)" % (len(rows_all), "included" if a_.include_public else "excluded"))
    print_summary(rows_all)
    if a_.sample or a_.list:
        if a_.sample:
            idx = {id(r): i for i, r in enumerate(base)}
            pick, q, avail = tt.pick_sample(base, a_.sample, a_.seed)
            rows_show = [rows_all[idx[id(r)]] for r in pick]
            head("seeded sample (seed %d): %d tags; per type %s" % (a_.seed, len(rows_show), q))
        else:
            rows_show = rows_all
            head("all %d eligible tags" % len(rows_show))
        print_blocks(rows_show)
        if a_.csv:
            write_csv(rows_show, a_.csv)
    if not a_.no_match_wide:
        match_wide(G, EV, sd, bsp, lab, graded, sp_all.start_frame.to_numpy(), n)