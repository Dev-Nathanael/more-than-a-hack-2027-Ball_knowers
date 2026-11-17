"""First deterministic decoder: possession from ball CONTROL, with TOUCHES as timing, scored on the public window.

    python control_touch_decoder.py            # three variants, scored with evaluation/possession_metrics.py
    python control_touch_decoder.py --spells   # also print the predicted possessions of the last variant

Variants (public window only; 17 possessions, so this is a sanity check, NOT a tuning set):
  1 nearest + smooth50   the repo baseline (nearest player's team, majority vote over +-50 frames)
  2 control only         hold the previous team; change on a confirmed control run
  4 + still -> last toucher   as 3; a ball slower than V_STILL for N_STILL frames confirms the pending change
  5 + still -> dead ball      as 3; those still stretches are labelled D
  6 still: tag -> D, else last toucher   as 4, but a still stretch inside a set-piece/goal tag window is D
  7 touches within 2.0 m  as 3 with the touch radius cut from 3.0 to 2.0 m (touch_refine.py: attribution agrees less beyond ~2 m)
  8 + control needs contact  as 7; a control run changes possession only with a same-team touch near its start, or a ball at rest
  9 + still tag logic        as 8 plus variant 6's still-ball handling
  3 control + touches    as 2, plus touches by the other team start a pending change that a control run or a
                         second touch confirms; a confirmed change is back-dated to the first touch

Control (per frame): ball visible, nearest player within R_CONTROL m, the other team's nearest at least
GAP_CONTROL m farther, and the ball slower than V_CONTROL m/s (over 5 frames). Runs shorter than N_MIN
frames are dropped; same-team runs split by <= MERGE_GAP frames are joined.
Dead ball (first version, weak on purpose): ball off the pitch for >= N_OFF frames -> D until the next control run.
No ball -> hold. All parameters are STARTING VALUES from the earlier diagnostics and football reasoning, not fitted.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "evaluation"))
from atd_explore import FPS, LENGTH, WIDTH, load_or_parse  # noqa: E402
from baseline import smooth as baseline_smooth  # noqa: E402
from compare_gt import GT_PATH, load_gt, runs_of as runs_of_gt  # noqa: E402
from tag_diagnostics import fmt_clock  # noqa: E402
from cluster_dead_ball import largest_cluster  # noqa: E402
from dead_ball_evidence import signals  # noqa: E402
from tag_diagnostics import load_events, prepare  # noqa: E402
from tag_module import build_table  # noqa: E402
from touch_candidates import detect  # noqa: E402
import possession_metrics as pm  # noqa: E402

R_CONTROL, GAP_CONTROL, V_CONTROL = 1.5, 1.0, 8.0
N_MIN, MERGE_GAP, N_CONFIRM = 3, 10, 12          # frames
W_C, W_T = 4 * FPS, 2 * FPS                       # how long a pending change waits (control / second touch)
N_OFF = 10                                        # frames of ball off the pitch that start a dead ball
TOUCH_K, TOUCH_TAU, TOUCH_R = 8, 4.0, 3.0
V_STILL, N_STILL = 1.0, int(2.5 * FPS)
TOUCH_R2 = 2.0                                     # attribution radius supported by touch_refine.py (agreement drops beyond ~2 m)
N_LONG, CLUSTER_MIN = 100, 8                                         # rows 13/14: slow players alone for 4 s; cluster size (5 m link) for the tag tier
V_DEAD_PS, V_DEAD_PS_TAG, N_DEAD, D_MIN, PRE_TAG = 1.3, 1.6, 50, 50, 15 * FPS   # dead-ball module (thresholds from the dead_ball_evidence.py grid)
SOFT_K, SOFT_TAU = 5, 2.0                        # soft contacts for the control gate only (touch_candidates.py grid: ~85-88% agreement)
CONTACT_BEFORE, CONTACT_AFTER, V_REST = 2 * FPS, FPS, 1.5   # a control run needs a same-team touch in [start-2 s, start+1 s] or a ball at rest             # a still ball: slower than 1 m/s for 2.5 s
LETTER = np.array(["H", "A"])


def head(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


# ------------------------------------------------------------------------------ per-frame evidence
def evidence(P, B, team, a, b):
    """Per-frame arrays for frames [a, b): nearest distances, control team (-1 none), off-pitch flag."""
    X, Y = P[a:b, :, 0] * LENGTH, P[a:b, :, 1] * WIDTH
    bx, by = B[a:b, 0] * LENGTH, B[a:b, 1] * WIDTH
    vis = np.isfinite(bx) & np.isfinite(by)
    d = np.hypot(X - bx[:, None], Y - by[:, None])
    d = np.where(np.isfinite(d), d, np.inf)
    dh, da = d[:, team == 0].min(1), d[:, team == 1].min(1)
    bl = B[a - 5:b, :2] * [LENGTH, WIDTH]
    sp = np.hypot(bl[5:, 0] - bl[:-5, 0], bl[5:, 1] - bl[:-5, 1]) * FPS / 5.0       # NaN if either end missing
    near = np.minimum(dh, da)
    gap = np.abs(dh - da)
    ok = vis & (near <= R_CONTROL) & (gap >= GAP_CONTROL) & ~(sp > V_CONTROL)
    ctrl = np.where(ok, np.where(dh < da, 0, 1), -1)
    off = vis & ((B[a:b, 0] < 0) | (B[a:b, 0] > 1) | (B[a:b, 1] < 0) | (B[a:b, 1] > 1))
    return {"dh": dh, "da": da, "vis": vis, "ctrl": ctrl, "off": off, "sp": sp}


def control_runs(ctrl):
    """[(start, stop, team)] of control runs after joining same-team runs split by short gaps and dropping tiny ones."""
    c = ctrl.copy()
    i, n = 0, len(c)
    while i < n:                                      # fill short 'none' gaps between runs of the same team
        if c[i] == -1:
            j = i
            while j < n and c[j] == -1:
                j += 1
            if 0 < i and j < n and c[i - 1] == c[j] and (j - i) <= MERGE_GAP:
                c[i:j] = c[i - 1]
            i = j
        else:
            i += 1
    runs, i = [], 0
    while i < n:
        if c[i] == -1:
            i += 1
            continue
        j = i
        while j < n and c[j] == c[i]:
            j += 1
        if j - i >= N_MIN:
            runs.append((i, j, int(c[i])))
        i = j
    return runs


def mask_runs(m, min_len):
    e = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)) if b - a >= min_len]


def off_runs(off):
    e = np.diff(np.concatenate(([0], off.astype(np.int8), [0])))
    return [(int(s), int(t)) for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)) if t - s >= N_OFF]


def contact_flags(runs, touches, sp):
    """True for each control run that has contact: a touch by the same team near its start, or the ball nearly at rest then."""
    ts = np.array([f for f, _ in touches]) if touches else np.array([], dtype=int)
    tt = np.array([t for _, t in touches]) if touches else np.array([], dtype=int)
    out = []
    for st, _, tm in runs:
        near = bool(((tt == tm) & (ts >= st - CONTACT_BEFORE) & (ts <= st + CONTACT_AFTER)).any())
        slow = bool(st < len(sp) and np.isfinite(sp[st]) and sp[st] <= V_REST)
        out.append(near or slow)
    return out


def still_runs(ev):
    """[(start, stop)] where the ball is visible and slower than V_STILL for at least N_STILL frames."""
    m = ev["vis"] & (ev["sp"] <= V_STILL)
    e = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)) if b - a >= N_STILL]


# ------------------------------------------------------------------------------ the state machine
def decode(n, runs, touches, offs, use_touches, use_dead=True, stills=None, still_mode="off", contact=None, dead_end_touch=False, shape_events=None, second_touch=True):
    """Labels 'H' / 'A' / 'D' for n frames (indices relative to the window start). Returns (labels, changes)."""
    contact = [True] * len(runs) if contact is None else contact
    ev = [(s, 1, "c", t, e - s, c) for (s, e, t), c in zip(runs, contact)]   # (frame, order, kind, team, length, has_contact)
    if use_touches:
        ev += [(f, 0, "t", t, 0, True) for f, t in touches]
    if use_dead:
        ev += [(o[0], 2, "d", -1, (o[2] if len(o) > 2 else 0), True) for o in offs]     # o = (start, stop[, earliest end])
    if shape_events:                                                    # hold-and-flip shape layer (missing-ball stretches only)
        ev += [(f, 1, "sh", tm, 0, True) for f, tm in shape_events]
    if still_mode == "possession" and stills:                         # a still ball confirms the pending change (last toucher)
        ev += [(a + N_STILL, 3, "s", -1, a, True) for a, _ in stills]
    ev.sort()
    lab = np.full(n, "", dtype="<U1")
    cur, cur_start, pend = None, 0, None                              # cur: 0/1, 'D' or None
    d_min_end = 0
    changes = []

    def close(upto):
        if cur is not None and upto > cur_start:
            lab[cur_start:upto] = "D" if cur == "D" else LETTER[cur]

    for f, _, kind, tm, ln, has in ev:
        if pend is not None and f - pend[1] > W_C:
            pend = None
        if kind == "d":
            if cur != "D":
                close(f)
                cur, cur_start, pend = "D", f, None
                d_min_end = ln
            else:
                d_min_end = max(d_min_end, ln)
            continue
        if cur is None or cur == "D":                                 # restart / first possession: no confirmation needed
            dwell = cur == "D" and dead_end_touch and (f - cur_start < D_MIN or f < d_min_end)     # too soon, or before a goal window's end
            if kind == "t" and cur == "D" and dead_end_touch and not dwell and tm in (0, 1):
                close(f)                                              # a kick / throw ends the dead ball; its team restarts
                cur, cur_start, pend = tm, f, None
                changes.append((f, tm))
            elif kind == "c" and not dwell:
                if cur is None:
                    lab[0:f] = LETTER[tm]                             # back-fill the start with the first controller
                else:
                    close(f)
                cur, cur_start, pend = tm, f, None
                changes.append((f, tm))
            continue
        if kind == "c" and not has:                                   # control without contact is not evidence of possession
            continue
        if kind == "sh":                                              # sustained shape evidence for the other team, ball missing
            if tm != cur:
                change = max(f, cur_start)
                close(change)
                cur, cur_start, pend = tm, change, None
                changes.append((change, tm))
            continue
        if kind == "s":
            if pend is not None and pend[0] != cur:
                change = max(pend[1], cur_start)
                close(change)
                cur, cur_start, pend = pend[0], change, None
                changes.append((change, cur))
            continue
        if tm == cur:
            pend = None                                               # the holder is back: any pending change is cancelled
            continue
        change = None
        if kind == "c" and ln >= N_CONFIRM:
            change = pend[1] if (pend and pend[0] == tm) else f      # back-date to the touch that preceded the control
        elif second_touch and kind == "t" and pend and pend[0] == tm and f - pend[1] <= W_T:
            change = pend[1]                                          # second touch confirms
        elif kind == "c" and pend and pend[0] == tm:
            change = pend[1]                                          # short control after a touch confirms
        else:
            pend = (tm, f) if (pend is None or pend[0] != tm) else pend
        if change is not None:
            change = max(change, cur_start)
            close(change)
            cur, cur_start, pend = tm, change, None
            changes.append((change, tm))
    close(n)
    if cur is not None:
        lab[cur_start:n] = "D" if cur == "D" else LETTER[cur]
    lab[lab == ""] = "D"
    return lab, changes


def d_capture(name, lab, lab_all, a, b):
    """For each true D stretch: share predicted D, start/end offsets of the predicted D, and predicted D elsewhere."""
    runs = runs_of_gt(lab_all, a, b)
    pd_ = lab == "D"
    gt_d = np.zeros(len(lab), dtype=bool)
    cells = []
    for s_, e_, l in runs:
        if l != "D":
            continue
        gt_d[s_ - a:e_ - a] = True
        seg = pd_[max(0, s_ - 5 * FPS - a):e_ - a]
        idx = np.flatnonzero(seg) + max(0, s_ - 5 * FPS - a)
        cov = pd_[s_ - a:e_ - a].mean()
        cells.append("%3.0f%% (%s)" % (100 * cov, "start %+.1fs end %+.1fs" % ((idx[0] + a - s_) / FPS, (idx[-1] + 1 + a - e_) / FPS) if len(idx) else "no D"))
    extra = (pd_ & ~gt_d).sum() / FPS
    print("%-26s | %s | D elsewhere %.0f s" % (name, " | ".join(cells), extra))


def still_to_dead(lab, stills):
    out = lab.copy()
    for a, b in stills:
        out[a:b] = "D"
    return out


def paired_bootstrap(gt, pairs, n_draws, seed=0):
    """Paired bootstrap of the spellF1 difference (B minus A) over true possessions, same resamples for both variants."""
    runs = pm.valid_runs(gt)
    gt_sp = pm.possession_spells(gt, runs)
    n_gt = len(gt_sp)
    flags, npred = {}, {}
    for k, codes in PRED.items():
        pr = pm.possession_spells(pm.fill_abstentions(codes, runs), runs)
        ov = pm.spell_overlaps(gt_sp, pr)
        flags[k] = np.vstack([pm.matched_gt_flags(ov, n_gt, t) for t in pm.SPELL_IOUS])
        npred[k] = len(pr)
    rng = np.random.RandomState(seed)
    picks = [rng.randint(0, n_gt, n_gt) for _ in range(n_draws)]

    def f1(k, pk):
        out = []
        for row in flags[k]:
            tp = int(row[pk].sum())
            out.append(2.0 * tp / (2 * tp + max(0, npred[k] - tp) + (n_gt - tp)) if tp else 0.0)
        return float(np.mean(out))
    head("paired bootstrap of the spellF1 difference (variant B minus A), %d draws over the %d true possessions" % (n_draws, n_gt))
    print("%-10s %9s %16s %22s" % ("A -> B", "mean diff", "5-95% band", "share of draws B > A"))
    for a, b in pairs:
        if a in flags and b in flags:
            d = np.array([f1(b, pk) - f1(a, pk) for pk in picks])
            print("%-10s %+9.3f %7.2f to %5.2f %21.0f%%" % ("%d -> %d" % (a, b), d.mean(), np.percentile(d, 5), np.percentile(d, 95), 100 * (d > 0).mean()))


# ------------------------------------------------------------------------------ scoring
def baseline_labels(P, B, team, a, b):
    """The repo baseline reproduced from the cache: nearest player's team, majority vote over +-50 frames."""
    pad = 100
    X, Y = P[a - pad:b + pad, :, 0] * LENGTH, P[a - pad:b + pad, :, 1] * WIDTH
    bx, by = B[a - pad:b + pad, 0] * LENGTH, B[a - pad:b + pad, 1] * WIDTH
    d = np.hypot(X - bx[:, None], Y - by[:, None])
    d = np.where(np.isfinite(d), d, np.inf)
    j = d.argmin(1)
    codes = np.where(np.isfinite(d[np.arange(len(d)), j]) & np.isfinite(bx), team[j], -1).astype(np.int8)
    return baseline_smooth(codes, 50)[pad:pad + (b - a)]


BOOT = 0
PRED = {}


def score(name, gt_codes, window_labels, a, b, n_total):
    full = np.full(n_total, "X", dtype="<U1")
    full[a:b] = window_labels
    PRED[int(name.split()[0])] = pm.label_codes(full)
    r = pm.evaluate(gt_codes, PRED[int(name.split()[0])], bootstrap_n=BOOT)
    t25 = r["transitions"][25]
    pc = r["per_class"]
    print("%-30s %7.3f %9.3f %6d/%-3d %8.2f  %5.2f %5.2f %5.2f  %6.2f %9s"
          % (name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"], r["alive_dead_f1"],
             pc["H"]["f1"], pc["A"]["f1"], pc["D"]["f1"], t25["f1"],
             "-" if t25["mean_abs_latency"] is None else "%.1f" % (t25["mean_abs_latency"] / FPS)) +
          ("   spellF1 5-95%%: %.2f-%.2f" % (r["spell_bootstrap_low"], r["spell_bootstrap_high"]) if BOOT else ""))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spells", action="store_true")
    ap.add_argument("--variant", type=int, default=3, help="which variant --spells prints (3 to 15)")
    ap.add_argument("--bootstrap", type=int, default=0, help="bootstrap draws over true possessions for a spellF1 band (e.g. 1000)")
    a_ = ap.parse_args()
    BOOT = a_.bootstrap
    np.seterr(invalid="ignore")
    P, B, team, *_ = load_or_parse(False)
    lab_all = load_gt(GT_PATH)
    v = np.flatnonzero(lab_all != "X")
    a, b = int(v[0]), int(v[-1]) + 1
    gt = pm.label_codes(pm.read_label_csv(GT_PATH))

    ev = evidence(P, B, team, a, b)
    runs = control_runs(ev["ctrl"])
    offs = off_runs(ev["off"])
    T = detect(P, B, team, a, b, TOUCH_TAU, TOUCH_R, k=TOUCH_K)
    touches = [(int(f) - a, int(t)) for f, t in zip(T["frame"], T["team"])]
    print("window %d frames | ball visible %.1f%% | control frames %.1f%% | control runs %d | touches %d | off-pitch runs %d"
          % (b - a, 100 * ev["vis"].mean(), 100 * (ev["ctrl"] >= 0).mean(), len(runs), len(touches), len(offs)))

    head("public window, repo scorer  (sanity check on 17 possessions - NOT a tuning set)")
    print("%-30s %7s %9s %10s %8s  %5s %5s %5s  %6s %9s" % ("variant", "spellF1", "macroF1", "spells", "alive/D", "F1 H", "F1 A", "F1 D", "trans", "|lat| s"))
    score("1 nearest + smooth50", gt, baseline_labels(P, B, team, a, b), a, b, len(lab_all))
    l2, c2 = decode(b - a, runs, touches, offs, use_touches=False)
    score("2 control only", gt, l2, a, b, len(lab_all))
    l3, c3 = decode(b - a, runs, touches, offs, use_touches=True)
    score("3 control + touches", gt, l3, a, b, len(lab_all))
    stills = still_runs(ev)
    l4, c4 = decode(b - a, runs, touches, offs, use_touches=True, stills=stills, still_mode="possession")
    score("4 + still -> last toucher", gt, l4, a, b, len(lab_all))
    l5 = still_to_dead(l3, stills)
    score("5 + still -> dead ball", gt, l5, a, b, len(lab_all))
    table = build_table(prepare(load_events()))
    dwin = table[(table.state == "D") & table.kind.str.startswith(("restart", "goal", "pre_kickoff"))]
    tagged = [bool(((dwin.search_start < a + e_) & (dwin.search_end > a + s_)).any()) for s_, e_ in stills]
    s_dead = [r for r, t in zip(stills, tagged) if t]
    s_poss = [r for r, t in zip(stills, tagged) if not t]
    l6, c6 = decode(b - a, runs, touches, offs, use_touches=True, stills=s_poss, still_mode="possession")
    l6 = still_to_dead(l6, s_dead)
    score("6 still: tag->D / toucher", gt, l6, a, b, len(lab_all))
    T2 = detect(P, B, team, a, b, TOUCH_TAU, TOUCH_R2, k=TOUCH_K)
    touches2 = [(int(f) - a, int(t)) for f, t in zip(T2["frame"], T2["team"])]
    l7, c7 = decode(b - a, runs, touches2, offs, use_touches=True)
    score("7 touches within 2.0 m", gt, l7, a, b, len(lab_all))
    flags = contact_flags(runs, touches2, ev["sp"])
    l8, c8 = decode(b - a, runs, touches2, offs, use_touches=True, contact=flags)
    score("8 + control needs contact", gt, l8, a, b, len(lab_all))
    l9, c9 = decode(b - a, runs, touches2, offs, use_touches=True, contact=flags, stills=s_poss, still_mode="possession")
    l9 = still_to_dead(l9, s_dead)
    score("9 + still tag logic", gt, l9, a, b, len(lab_all))
    G = signals(P, B)
    ps_, out_, miss_ = G["ps"][a:b], G["out"][a:b], G["missing"][a:b]
    core = mask_runs((out_ | miss_) & (ps_ <= V_DEAD_PS), N_DEAD)
    l11, c11 = decode(b - a, runs, touches, offs + core, use_touches=True, dead_end_touch=True)
    score("11 + dead ball (ATD only)", gt, l11, a, b, len(lab_all))
    inwin = np.zeros(b - a, dtype=bool)
    for r_ in dwin.itertuples():
        inwin[max(0, r_.start_frame - PRE_TAG - a):max(0, r_.end_frame - a)] = True
    tagrun = mask_runs(inwin & ((ps_ <= V_DEAD_PS_TAG) | out_ | miss_), N_DEAD)
    l12, c12 = decode(b - a, runs, touches, offs + core + tagrun, use_touches=True, dead_end_touch=True)
    score("12 + dead ball (+ tag tier)", gt, l12, a, b, len(lab_all))
    slow_only = mask_runs(ps_ <= V_DEAD_PS, N_LONG)
    l13, c13 = decode(b - a, runs, touches, offs + core + slow_only, use_touches=True, dead_end_touch=True)
    score("13 + slow players alone (4 s)", gt, l13, a, b, len(lab_all))
    L5 = np.empty(b - a)
    for c0 in range(0, b - a, 1500):
        fr_ = np.arange(a + c0, min(b, a + c0 + 1500))
        L5[c0:c0 + len(fr_)] = largest_cluster(P[fr_][:, :, 0] * LENGTH, P[fr_][:, :, 1] * WIDTH, 5)
    clus = mask_runs(inwin & (L5 >= CLUSTER_MIN) & (ps_ <= V_DEAD_PS_TAG), N_DEAD)
    l14, c14 = decode(b - a, runs, touches, offs + core + clus, use_touches=True, dead_end_touch=True)
    score("14 + cluster tag tier", gt, l14, a, b, len(lab_all))
    l15, c15 = decode(b - a, runs, touches, offs + core + slow_only, use_touches=True, dead_end_touch=True, second_touch=False)
    score("15 row 13, touches don't confirm", gt, l15, a, b, len(lab_all))
    T3 = detect(P, B, team, a, b, SOFT_TAU, TOUCH_R2, k=SOFT_K)
    soft = [(int(f) - a, int(t)) for f, t in zip(T3["frame"], T3["team"])]
    flags_soft = contact_flags(runs, soft, ev["sp"])
    l10, c10 = decode(b - a, runs, touches2, offs, use_touches=True, contact=flags_soft)
    score("10 control needs soft contact", gt, l10, a, b, len(lab_all))
    head("each true D stretch: share predicted as D, and the offset of the predicted D start/end (seconds; predicted minus true)")
    print("true D stretches: " + ", ".join("%s-%s" % (fmt_clock(x)[2:9], fmt_clock(y)[2:9]) for x, y, l in runs_of_gt(lab_all, a, b) if l == "D"))
    for nm, lv_ in (("3 control + touches", l3), ("11 dead ball (ATD only)", l11), ("12 dead ball (+ tag tier)", l12), ("13 + slow players alone", l13), ("14 + cluster tag tier", l14)):
        d_capture(nm, lv_, lab_all, a, b)
    print("dead-ball runs found: ATD-only %d, tag tier %d, slow-players-only %d, cluster tier %d (tag windows used: %d)" % (len(core), len(tagrun), len(slow_only), len(clus), len(dwin)))
    print("\ntouches within 2.0 m: %d (3.0 m: %d) | control runs with contact: strict %d, soft %d of %d (soft contacts: %d)"
          % (len(touches2), len(touches), sum(flags), sum(flags_soft), len(flags), len(soft)))
    print("\nstill-ball runs (>= %.1f s under %.1f m/s): %d, %d frames in total: %s"
          % (N_STILL / FPS, V_STILL, len(stills), sum(e - s_ for s_, e in stills),
             ", ".join("%s-%s" % (fmt_clock(a + s_)[2:9], fmt_clock(a + e)[2:9]) for s_, e in stills)))
    if BOOT:
        paired_bootstrap(gt, [(1, 3), (3, 7), (7, 8), (7, 10), (1, 10), (2, 3), (3, 11), (3, 12), (11, 12), (11, 13), (11, 14), (13, 15)], BOOT)
    print("variant 6: %d still runs inside a set-piece/goal tag window (-> D), %d elsewhere (-> last toucher)" % (len(s_dead), len(s_poss)))
    print("\ncolumns: spells = predicted/true possessions; alive/D = alive-vs-dead F1; trans = transition F1 within 1 s; |lat| = mean absolute transition error")
    print("changes made: control only %d, control + touches %d (true direct changes 12, true D stretches 4)" % (len(c2), len(c3)))
    if a_.spells:
        lv = {3: l3, 4: l4, 5: l5, 6: l6, 7: l7, 8: l8, 9: l9, 10: l10, 11: l11, 12: l12, 13: l13, 14: l14, 15: l15}[a_.variant]
        head("predicted possessions, variant %d" % a_.variant)
        s0 = 0
        for i in range(1, len(lv) + 1):
            if i == len(lv) or lv[i] != lv[s0]:
                print("  %s - %s  %5.1fs  %s" % (fmt_clock(a + s0)[2:], fmt_clock(a + i)[2:], (i - s0) / FPS, lv[s0]))
                s0 = i