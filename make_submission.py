"""Run the best decoder over the WHOLE match and write the submission file.

    python make_submission.py                          # writes evaluation/results/control_touch_v13.csv
    python make_submission.py --out evaluation/results/mine.csv --cluster

Decoder = row 13 of control_touch_decoder.py: ball-control runs + touches, hold the previous team through
loose / missing-ball stretches, dead ball from (ball outside|missing AND slow players) or (slow players alone for 4 s),
a dead ball ends at the first touch or control run after 2 s. --cluster adds the row-14 cluster tag tier (set-piece
windows only; selective but it added nothing on the public window).

Per graded half the decoder runs from the kickoff (first half: the first graded frame; second half: frame 113742).
The 2,934 graded frames before the second-half kickoff are D (published in the FAQ). Everything outside the two
graded blocks is X. The output is exactly one row per video frame (frame,label).

After writing, validate and score it with the repo's tools (run from the repo root):
    python evaluation/validate_submission.py evaluation/results/control_touch_v13.csv
    python evaluation/run_evaluation.py
Also prints label-free diagnostics (possession counts, how much is labelled by 'hold' with no ball, longest holds).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import games  # noqa: E402
from atd_explore import FPS, LENGTH, WIDTH, load_or_parse  # noqa: E402
from cluster_dead_ball import largest_cluster  # noqa: E402
from control_touch_decoder import (CLUSTER_MIN, N_DEAD, N_LONG, PRE_TAG, TOUCH_K, TOUCH_R, TOUCH_TAU,  # noqa: E402
                                   V_DEAD_PS, V_DEAD_PS_TAG, control_runs, decode, evidence, mask_runs, off_runs)
from dead_ball_evidence import signals  # noqa: E402
from shape_flip_check import GRID as SGRID, shape_scores  # noqa: E402
from tag_diagnostics import KICKOFF_FRAMES, fmt_clock  # noqa: E402
from touch_candidates import detect  # noqa: E402

OUT_DEFAULT = os.path.join("evaluation", "results", "control_touch_v13.csv")


def decode_block(P, B, team, G, a, b, tag_windows=None, goal_events=None, shape_events=None, second_touch=True):
    """Labels 'H'/'A'/'D' for frames [a, b) with the row-13 decoder (and optionally the cluster tag tier)."""
    ev = evidence(P, B, team, a, b)
    runs = control_runs(ev["ctrl"])
    offs = off_runs(ev["off"])
    T = detect(P, B, team, a, b, TOUCH_TAU, TOUCH_R, k=TOUCH_K)
    touches = [(int(f) - a, int(t)) for f, t in zip(T["frame"], T["team"])]
    ps, out, miss = G["ps"][a:b], G["out"][a:b], G["missing"][a:b]
    dead = mask_runs((out | miss) & (ps <= V_DEAD_PS), N_DEAD) + mask_runs(ps <= V_DEAD_PS, N_LONG)
    if tag_windows:
        inwin = np.zeros(b - a, dtype=bool)
        for s, e in tag_windows:
            inwin[max(0, s - PRE_TAG - a):max(0, min(e, b) - a)] = True
        L5 = np.empty(b - a)
        for c0 in range(0, b - a, 1500):
            fr = np.arange(a + c0, min(b, a + c0 + 1500))
            L5[c0:c0 + len(fr)] = largest_cluster(P[fr][:, :, 0] * LENGTH, P[fr][:, :, 1] * WIDTH, 5)
        dead += mask_runs(inwin & (L5 >= CLUSTER_MIN) & (ps <= V_DEAD_PS_TAG), N_DEAD)
    lab, changes = decode(b - a, runs, touches, offs + dead + (goal_events or []), use_touches=True, dead_end_touch=True, shape_events=shape_events, second_touch=second_touch)
    return lab, {"control_runs": len(runs), "touches": len(touches), "dead_starts": len(dead), "goal_d": len(goal_events or []), "shape_flips": len(shape_events or []), "changes": len(changes)}


def shape_events_for(grid, S, vis, a, b, t_flip_s):
    """(frame - a, team) at the start of every run, inside a ball-missing stretch, where the shape score keeps one sign for >= t_flip_s."""
    m = (grid >= a) & (grid < b)
    g, s = grid[m], S[m]
    miss = ~vis[g]
    t_pts = max(1, int(t_flip_s * FPS / SGRID))
    out, i, n = [], 0, len(g)
    while i < n:
        if miss[i] and np.isfinite(s[i]) and s[i] != 0:
            sg = np.sign(s[i])
            j = i
            while j < n and miss[j] and np.isfinite(s[j]) and np.sign(s[j]) == sg:
                j += 1
            if j - i >= t_pts:
                out.append((int(g[i]) - a, 0 if sg > 0 else 1))
            i = j
        else:
            i += 1
    return out


def fill_d_gaps(labels, blocks, ps, gap_frames, ps_max, vis=None):
    """Fill the gap between two D runs with D when it is short and the players stay slow in it. Returns [(start, stop, mean ps, ball visible share)]."""
    filled = []
    for _, first, last in blocks:
        seg = (labels[first:last + 1] == "D").astype(np.int8)
        e = np.diff(np.concatenate(([0], seg, [0])))
        starts, stops = (first + np.flatnonzero(e == 1)).tolist(), (first + np.flatnonzero(e == -1)).tolist()
        for (s1, e1), (s2, e2) in zip(zip(starts, stops), zip(starts[1:], stops[1:])):
            if 0 < s2 - e1 <= gap_frames and np.nanmean(ps[e1:s2]) <= ps_max:
                filled.append((e1, s2, float(np.nanmean(ps[e1:s2])), float(vis[e1:s2].mean()) if vis is not None else np.nan))
                labels[e1:s2] = "D"
    return filled


def postprocess_d(labels, blocks, exempt_windows, d_min, d_max, pre_kickoff_ends):
    """Drop dead-ball runs shorter than d_min frames and cap longer ones at d_max frames (both revert to the label before the run).
    Runs overlapping an exempt window (goal windows, +15 s look-back) and the pre-kickoff period are left alone."""
    changed = {"dropped": 0, "capped": 0}
    for _, first, last in blocks:
        seg = (labels[first:last + 1] == "D").astype(np.int8)
        e = np.diff(np.concatenate(([0], seg, [0])))
        for a, b in zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()):
            a, b = first + a, first + b
            if a in pre_kickoff_ends or any(a < ge and b > gs for gs, ge in exempt_windows):
                continue
            prev = labels[a - 1] if a - 1 >= first and labels[a - 1] in ("H", "A") else None
            if prev is None:
                continue
            if d_min and (b - a) < d_min:
                labels[a:b] = prev
                changed["dropped"] += 1
            elif d_max and (b - a) > d_max:
                labels[a + d_max:b] = prev
                changed["capped"] += 1
    return changed


def drop_short_unbacked(labels, blocks, sp_starts, goal_windows, d_min, pre_kickoff_ends):
    """Drop D runs shorter than d_min frames that no tag backs: no set-piece tag start within 10 s, no goal window, not the pre-kickoff run."""
    st = np.sort(np.asarray(sp_starts))
    dropped = 0
    for _, first, last in blocks:
        seg = (labels[first:last + 1] == "D").astype(np.int8)
        e = np.diff(np.concatenate(([0], seg, [0])))
        for a, b in zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()):
            a, b = first + a, first + b
            if (b - a) >= d_min or a in pre_kickoff_ends or any(a < ge and b > gs for gs, ge in goal_windows):
                continue
            i = np.searchsorted(st, a - 10 * FPS)
            if i < len(st) and st[i] <= b + 10 * FPS:
                continue
            prev = labels[a - 1] if a - 1 >= first and labels[a - 1] in ("H", "A") else None
            if prev is not None:
                labels[a:b] = prev
                dropped += 1
    return dropped


def spells(lab):
    out, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            if lab[s] in ("H", "A"):
                out.append((s, i, lab[s]))
            s = i
    return out


def runs_of_label(seg, lab):
    m = (seg == lab).astype(np.int8)
    e = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def diagnostics(labels, B, blocks, tag_windows=None):
    print("\n== label-free diagnostics ==")
    vis = np.isfinite(B[:, 0]) & np.isfinite(B[:, 1])
    all_d = []
    for half, first, last in blocks:
        seg = labels[first:last + 1]
        n = len(seg)
        sp = spells(seg)
        dur = np.array([(e - s) / FPS for s, e, _ in sp]) if sp else np.array([0.0])
        per5 = [sum(1 for s, e, _ in sp if k <= s < k + 5 * 60 * FPS) for k in range(0, n, 5 * 60 * FPS)]
        alive = np.isin(seg, ["H", "A"])
        hold = alive & ~vis[first:last + 1]
        hr = [(j - i) / FPS for i, j in runs_of_label(np.where(hold, "h", "x"), "h")]
        print("half %d: H %.1f%%  A %.1f%%  D %.1f%% | %d possessions (mean %.1f s, median %.1f s; %.0f%% under 3 s, %.0f%% under 5 s) | per 5 min: min %d, median %d, max %d"
              % (half, 100 * (seg == "H").mean(), 100 * (seg == "A").mean(), 100 * (seg == "D").mean(), len(sp), dur.mean(), np.median(dur),
                 100 * (dur < 3).mean(), 100 * (dur < 5).mean(), min(per5), int(np.median(per5)), max(per5)))
        print("        frames labelled H/A with no ball visible (held): %.1f%% of the alive frames; longest such stretch %.0f s, p90 %.0f s"
              % (100 * hold.sum() / max(1, alive.sum()), max(hr) if hr else 0, np.percentile(hr, 90) if hr else 0))
        all_d += [(first + s, first + e) for s, e in runs_of_label(seg, "D")]
    L = np.array([(e - s) / FPS for s, e in all_d])
    dframes = sum(e - s for s, e in all_d)
    novis = sum(int((~vis[s:e]).sum()) for s, e in all_d)
    print("\ndead-ball runs: %d | length median %.1f s, p90 %.1f s, max %.0f s | share of D time in runs > 30 s: %.0f%%, > 60 s: %.0f%%"
          % (len(L), np.median(L), np.percentile(L, 90), L.max(), 100 * L[L > 30].sum() / L.sum(), 100 * L[L > 60].sum() / L.sum()))
    print("share of D frames with the ball NOT visible: %.0f%%  (public window: true D had the ball missing on 34%% of its frames)" % (100 * novis / dframes))
    if tag_windows is not None:
        N = len(labels)
        near = np.zeros(N, dtype=bool)
        for s, e in tag_windows:
            near[max(0, s - PRE_TAG):min(N, e)] = True
        hit = [(s, e) for s, e in all_d if near[s:e].any()]
        inside = sum(int(near[s:e].sum()) for s, e in all_d)
        print("dead-ball runs overlapping a set-piece/goal tag window (extended 15 s back): %d of %d (%.0f%%); %.0f%% of D frames lie inside such a window"
              % (len(hit), len(all_d), 100 * len(hit) / len(all_d), 100 * inside / dframes))
        print("(the tags are incomplete: 2 of the 4 public D stretches had none, so a low share is not proof of false dead ball)")
    print("(the public window has 17 possessions in 5 min, mean ~14 s; real stoppages there lasted 5-20 s, only goals last minutes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--cluster", action="store_true", help="add the row-14 cluster tag tier (needs the Smart Tagging file)")
    ap.add_argument("--goal-d", action="store_true", help="force D inside every goal tag window (from its start to 1 s before its end): VAR check + celebration")
    ap.add_argument("--fill-gap", type=float, default=0.0, help="fill gaps of at most this many seconds between two D runs when the players stay slow (0 = off; try 45)")
    ap.add_argument("--list-fills", type=int, default=0, help="with --fill-gap: print the N longest gaps that were filled, to check in the video")
    ap.add_argument("--no-second-touch", action="store_true", help="two touches by the other team no longer confirm a change (only a control run does)")
    ap.add_argument("--d-min-unbacked", type=float, default=0.0, help="drop tag-unbacked dead-ball runs shorter than this many seconds (0 = off; try 6)")
    ap.add_argument("--shape", action="store_true", help="add the hold-and-flip shape layer for ball-missing stretches")
    ap.add_argument("--t-flip", type=float, default=5.0, help="seconds the shape score must favour the other team before the layer flips")
    ap.add_argument("--d-min", type=float, default=0.0, help="drop non-exempt dead-ball runs shorter than this many seconds (0 = off; try 3)")
    ap.add_argument("--d-max", type=float, default=0.0, help="cap non-exempt dead-ball runs at this many seconds (0 = off; try 45)")
    ap.add_argument("--tags", action="store_true", help="also report how many dead-ball runs overlap a set-piece/goal tag window (needs the Smart Tagging file)")
    a_ = ap.parse_args()
    np.seterr(invalid="ignore")
    P, B, team, *_ = load_or_parse(False)
    G = signals(P, B)
    n = games.GAMES[games.GAME]["n_video_frames"]
    blocks = games.half_video_blocks(games.GAME)
    tag_windows = None
    diag_windows = None
    goal_windows = []
    goal_raw = []
    sp_starts = []
    if a_.cluster or a_.tags or a_.d_min or a_.d_max or a_.goal_d or a_.d_min_unbacked:
        from tag_diagnostics import load_events, prepare
        from tag_module import build_table
        t = build_table(prepare(load_events()))
        d = t[(t.state == "D") & t.kind.str.startswith(("restart", "goal"))]
        diag_windows = list(zip(d.start_frame, d.end_frame))
        g_ = prepare(load_events())
        goal_windows = [(r.start_frame - PRE_TAG, r.end_frame) for r in g_[g_.code_u == "GOALS"].itertuples()]
        goal_raw = [(r.start_frame, r.end_frame) for r in g_[g_.code_u == "GOALS"].itertuples()]
        sp_starts = [r.start_frame for r in g_[(g_.code_u == "SET PIECES") & ~g_.type_l.str.startswith("kick off")].itertuples()]
        tag_windows = diag_windows if a_.cluster else None
    shape_by_block = {}
    if a_.shape:
        vis_all = np.isfinite(B[:, 0]) & np.isfinite(B[:, 1])
        grids, scores = shape_scores(P, team, blocks)
        shape_by_block = {bl[0]: (g, sc) for bl, g, sc in zip(blocks, grids, scores)}
    labels = np.full(n, "X", dtype="<U1")
    for half, first, last in blocks:
        ko = KICKOFF_FRAMES["1st Half" if half == 1 else "2nd Half"]
        a = max(first, ko)
        b = last + 1
        if a > first:
            labels[first:a] = "D"                                   # graded but before kickoff: dead ball (FAQ)
        gev = [(max(gs, a) - a, max(gs, a) - a + 1, ge - a - FPS) for gs, ge in goal_raw if a_.goal_d and ge > a and gs < b]
        sev = shape_events_for(*shape_by_block[half], vis_all, a, b, a_.t_flip) if a_.shape else None
        lab, info = decode_block(P, B, team, G, a, b, tag_windows, gev, sev, second_touch=not a_.no_second_touch)
        labels[a:b] = lab
        print("half %d: decoded frames %d-%d (%s - %s) | %s" % (half, a, b - 1, fmt_clock(a), fmt_clock(b), info))

    if a_.fill_gap:
        fills = fill_d_gaps(labels, blocks, G["ps"], int(a_.fill_gap * FPS), V_DEAD_PS_TAG, G["vis"])
        tot_fill = sum(e - s_ for s_, e, _, _ in fills)
        print("dead-ball gap filling (gaps <= %.0f s with slow players): filled %d gaps, %.0f s in total (%.0f%% of the graded frames)"
              % (a_.fill_gap, len(fills), tot_fill / FPS, 100 * tot_fill / max(1, sum(l - f + 1 for _, f, l in blocks))))
        if a_.list_fills and fills:
            near_f = np.zeros(n, dtype=bool)
            for s_, e_ in (diag_windows or []):
                near_f[max(0, s_ - PRE_TAG):min(n, e_)] = True
            print("the %d longest filled gaps (check a few in the video: is play really stopped?):" % min(a_.list_fills, len(fills)))
            print("  %-10s %7s %11s %9s  %s" % ("from", "length", "player m/s", "ball vis", "inside a tag window (+15 s look-back)"))
            for s_, e_, ps_g, vs_g in sorted(fills, key=lambda x: -(x[1] - x[0]))[:a_.list_fills]:
                print("  %-10s %6.0fs %11.2f %8.0f%%  %s" % (fmt_clock(s_)[:10], (e_ - s_) / FPS, ps_g, 100 * vs_g, "yes" if near_f[s_:e_].any() else "no"))
    if a_.d_min_unbacked:
        pre0 = {first for _, first, _ in blocks}
        nd = drop_short_unbacked(labels, blocks, sp_starts, goal_windows, int(a_.d_min_unbacked * FPS), pre0)
        print("dropped %d tag-unbacked dead-ball runs shorter than %.0f s" % (nd, a_.d_min_unbacked))
    if a_.d_min or a_.d_max:
        pre_ends = {first for _, first, _ in blocks}                 # the pre-kickoff run starts at the first graded frame of a half
        ch = postprocess_d(labels, blocks, goal_windows, int(a_.d_min * FPS), int(a_.d_max * FPS), pre_ends)
        print("dead-ball post-processing (min %.0f s, cap %.0f s): dropped %d short runs, capped %d long runs" % (a_.d_min, a_.d_max, ch["dropped"], ch["capped"]))

    # checks before writing
    assert len(labels) == n
    assert set(np.unique(labels)) <= {"H", "A", "D", "X"}, "unexpected labels"
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    assert not (labels[graded] == "X").any(), "X inside a graded block"
    assert (labels[~graded] == "X").all(), "a label outside the graded blocks"
    os.makedirs(os.path.dirname(os.path.abspath(a_.out)), exist_ok=True)
    with open(a_.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, l) for i, l in enumerate(labels))
    print("\nwrote %s: %d rows | %s" % (a_.out, n, {k: int((labels == k).sum()) for k in "HADX"}))
    diagnostics(labels, B, blocks, diag_windows)