"""Fill missed dead balls around set-piece tags.  A post-processor on a finished submission; the decoder and the v17 file are not touched.

    python tag_restart.py --in evaluation/results/control_touch_v17.csv --out evaluation/results/control_touch_v18.csv
    python tag_restart.py --in evaluation/results/control_touch_v17.csv --dry-run        # report only, writes nothing

For every non-kickoff set-piece tag that is OUTSIDE the public window and has no D at tag start + 2 s (the same test as dead_start_trace.py --coverage):
  start  the best physical cue in [tag - 35 s, tag + 3 s], in this order
           1. the latest ball-outside run that begins at most 25 s before the tag start
           2. the ball lost for >= 2 s while near (<= 3 m inside, heading out) or outside a line
           3. the ball visibly at rest for >= 3 s at a restart spot (corner arc, goal area, touchline) across the tag start
           4. otherwise the tag start itself
  end    the first restart candidate in [tag + 3 s, tag + 12 s] (a touch, or the ball at rest then fast / lost), else tag start + 7 s
and the frames from start to end are set to D (graded frames only, at most CAP_S seconds, ending at the restart).  Nothing else is changed: in particular
the possession label after the restart is left as it is.
Cues are gated by the tag type, from the rules of the game: a goal kick or a corner needs the ball lost / out at an END line (or, for the rest cue, in the goal area /
corner arc), a throw-in at a SIDE line / touchline; a cue at the wrong line is ignored and the next one is tried (see dead_start_trace.py --coverage).  The report says, for each filled segment, which team touched the ball at the restart and
whether that differs from the label that follows in the input, so the cost of leaving the owner alone is visible.

No fixed 35 s tag window is labelled D, nothing is tuned on the 17 public possessions, and tags inside the public window are never touched
(--include-public lifts that, for testing only).  Running the script on its own output changes nothing (the filled segments now cover tag + 2 s).
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
import dead_start_trace as dst  # noqa: E402
import tag_timing as tt  # noqa: E402
from atd_explore import FPS, LENGTH, WIDTH, load_or_parse  # noqa: E402
from d_audit import d_runs  # noqa: E402
from dead_ball_evidence import signals  # noqa: E402
from tag_diagnostics import fmt_clock, load_events, prepare  # noqa: E402

CAP_S = 25


def runs_of_label(lab, blocks, label):
    out = []
    for _, f, l in blocks:
        seg = (lab[f:l + 1] == label).astype(np.int8)
        e = np.diff(np.concatenate(([0], seg, [0])))
        out += [(f + int(a), f + int(b)) for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))]
    return out


def nearest_team(P, B, team, f):
    """'H' / 'A' for the tracked player nearest the ball at frame f (None when the ball or all players are missing)."""
    if not (np.isfinite(B[f, 0]) and np.isfinite(B[f, 1])):
        return None
    d = np.hypot(P[f, :, 0] * LENGTH - B[f, 0] * LENGTH, P[f, :, 1] * WIDTH - B[f, 1] * WIDTH)
    if not np.isfinite(d).any():
        return None
    return "HA"[int(team[int(np.nanargmin(d))])]


def plan_fills(res, lab, graded, cap_s=CAP_S):
    """[(tag row, start, end)] for the missed tags; start/end are clipped to graded frames and the cap."""
    plan = []
    for r in res:
        if r["cov"]:
            continue
        t0 = r["tag_start"]
        s_est = t0
        out, drop, rest = r["out"], r["drop"], r["rest"]
        if out:
            s_est = out[0]
        elif drop and (drop[2] > 0 or (drop[2] >= -dst.NEAR_LINE_M and drop[3] > 0)):
            s_est = drop[0]
        elif rest and rest[2][0] != "elsewhere":
            s_est = rest[0]
        e_est = r["restart"][0] if r["restart"] else t0 + 7 * FPS
        if e_est - s_est > cap_s * FPS:
            s_est = e_est - cap_s * FPS
        if e_est <= s_est or not graded[s_est] or not graded[min(e_est, len(graded) - 1)]:
            continue
        plan.append((r, int(s_est), int(e_est)))
    return plan


def spells(lab, blocks):
    """number of H/A possession spells (maximal runs of one team) in the graded blocks."""
    n = 0
    for _, f, l in blocks:
        seg = lab[f:l + 1]
        for t in ("H", "A"):
            m = (seg == t).astype(np.int8)
            n += int((np.diff(np.concatenate(([0], m))) == 1).sum())
    return n


def _plan(lab, include_public, cap_s):
    """Everything main() and restart_fill() share: the tracking, the coverage rows and the fills to make."""
    np.seterr(invalid="ignore")
    n = len(lab)
    P, B, team, *_ = load_or_parse(False)
    G = signals(P, B)
    bsp = tt.ball_speed(B)
    sd = dst.signed_dist(B[:, 0] * LENGTH, B[:, 1] * WIDTH)
    blocks = games.half_video_blocks(games.GAME)
    graded = np.zeros(n, dtype=bool)
    for _, f, l in blocks:
        graded[f:l + 1] = True
    d = prepare(load_events())
    sp = d[(d.code_u == "SET PIECES") & ~d.type_l.str.startswith("kick off")]
    EV = dst.start_events(G)
    RS = dst.restart_events(G["vis"], bsp, n)
    base = tt.tag_table(tt.eligible_tags(sp, include_public), d_runs(lab, blocks), G, bsp, n)
    res = dst.coverage_rows(base, G, EV, sd, bsp, RS, lab, P, B, team, n)
    return plan_fills(res, lab, graded, cap_s), res, P, B, team, blocks


def restart_fill(lab, include_public=True, cap_s=CAP_S):
    """lab with each missed set-piece stoppage set to D; prints nothing (used by experiments/pipelines.py)."""
    new = np.asarray(lab).astype("<U1").copy()
    for _, s_, e_ in _plan(new.copy(), include_public, cap_s)[0]:
        new[s_:e_] = "D"
    return new


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join("evaluation", "results", "control_touch_v17.csv"))
    ap.add_argument("--out", default=os.path.join("evaluation", "results", "control_touch_v18.csv"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap-s", type=float, default=CAP_S)
    ap.add_argument("--include-public", action="store_true")
    a_ = ap.parse_args(argv)
    df = pd.read_csv(a_.inp, dtype={"label": "object"})
    lab = df["label"].astype(str).to_numpy().copy()
    n = len(lab)
    plan, res, P, B, team, blocks = _plan(lab, a_.include_public, a_.cap_s)
    new = lab.copy()
    print("%d eligible tags | %d with D at tag + 2 s | %d without | %d segments to fill (cap %g s)" % (len(res), sum(r["cov"] for r in res), sum(not r["cov"] for r in res), len(plan), a_.cap_s))
    print("%-3s %-11s %-12s %-23s %7s %-34s %-16s %s" % ("#", "tag", "type", "filled D (clock)", "length", "start cue", "restart", "restart touch team vs the label after"))
    tot, over = 0, {"H": 0, "A": 0, "D": 0}
    differ = known = 0
    for i, (r, s_, e_) in enumerate(plan, 1):
        seg_old = new[s_:e_]
        for k in over:
            over[k] += int((seg_old == k).sum())
        new[s_:e_] = "D"
        tot += e_ - s_
        why = r["est"][1]
        who = nearest_team(P, B, team, e_) if r["restart"] and r["restart"][1] == "touch" else None
        after = new[e_] if e_ < n else "X"
        if who and after in ("H", "A"):
            known += 1
            differ += int(who != after)
        print("%-3d %-11s %-12s %-23s %6.1fs %-34s %-16s %s" % (i, fmt_clock(r["tag_start"])[:10], r["type"], "%s - %s" % (fmt_clock(s_)[:10], fmt_clock(e_)[:10]), (e_ - s_) / FPS,
                                                          why[:34], ("%s %s" % (r["restart"][1], tt.rel(r["restart"][0], r["tag_start"]))) if r["restart"] else "the +7 s prior",
                                                          "n/a" if not who else "touch by %s, label after %s%s" % (who, after, "  (DIFFERENT)" if who != after and after in ("H", "A") else "")))
    gs = sum(l - f + 1 for _, f, l in blocks)
    print("\nframes set to D: %d (%.0f s, %.2f%% of graded frames); previously labelled H %d, A %d, D %d" % (tot, tot / FPS, 100.0 * tot / gs, over["H"], over["A"], over["D"]))
    print("possession spells (H/A runs) in the graded blocks: %d before, %d after" % (spells(lab, blocks), spells(new, blocks)))
    print("restart team (nearest player at the touch) differs from the label that follows the filled segment in %d of %d segments where both are known" % (differ, known))
    chg = np.flatnonzero(new != lab)
    pub = (chg >= tt.PUBLIC_WINDOW[0]) & (chg < tt.PUBLIC_WINDOW[1])
    print("frames changed inside the public window (11185-18684): %d%s" % (int(pub.sum()), "" if a_.include_public else "  (must be 0)"))
    if a_.dry_run:
        print("dry run: nothing written")
        return new
    os.makedirs(os.path.dirname(os.path.abspath(a_.out)), exist_ok=True)
    out = df.copy()
    out["label"] = new
    out.to_csv(a_.out, index=False)
    print("wrote %s" % a_.out)
    return new


if __name__ == "__main__":
    main()