"""Every solution as one function: match data in, a full-length label array out.

This is the single source of truth. The notebooks in solutions/ call these functions, and
experiments/submit.py turns one of them into the file we upload.

    from pipelines import SOLUTIONS
    labels = SOLUTIONS["05"]()          # 199319 labels, H / A / D / X
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tracking import GAME, GAMES, half_video_blocks, load, load_clean, team_features  # noqa: E402
from games import require, source_frames  # noqa: E402
from baseline import HOME, nearest_player_labels, smooth, team_of_each_track  # noqa: E402
from load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402

N_FRAMES = source_frames("tactical")
SOLUTION_06_TAU = 0.75  # self-consistency: decoded mean spell 8.5 s against the priors' 8.7 s
BLOCKS = half_video_blocks(GAME)
PITCH = GAMES[GAME]["pitch"]


def graded_mask():
    mask = np.zeros(N_FRAMES, dtype=bool)
    for _, first, last in BLOCKS:
        mask[first:last + 1] = True
    return mask


def dead_before_kickoff(labels, blocks=BLOCKS, kickoffs=KICKOFF_VIDEO_FRAMES):
    """Graded frames before a half's kickoff are D.

    The second half is graded from 110,808 but kicks off at 113,742; FAQ.md says all 2,934 frames
    in between are D. The kickoff frames are the organisers' published ones, not our choice.
    """
    out = labels.copy()
    for (_, first, last), name in zip(blocks, ("1st Half", "2nd Half")):
        out[first:min(kickoffs[name], last + 1)] = "D"
    return out


def goal_tag_windows():
    """[(start, stop)] of the Smart Tagging goal tags, stop one second before the tag ends.

    After a goal the ball is dead until the kickoff (celebration, VAR check); the tag's last second
    is left to the decoder because the kickoff itself sits there (our earlier --goal-d rule, v17)."""
    import load_smart_tagging
    fps = GAMES[GAME]["fps"]
    return [(t["start_frame"], t["end_frame"] - fps) for t in load_smart_tagging.load()
            if (t["code"] or "").strip().upper() == "GOALS"]


def dead_in_windows(labels, windows, blocks=BLOCKS):
    """Graded frames inside any [start, stop) window are D; ungraded frames are left alone."""
    out = labels.copy()
    for _, first, last in blocks:
        for start, stop in windows:
            a, b = max(start, first), min(stop, last + 1)
            if a < b:
                out[a:b] = "D"
    return out


# Params
SURE_P = 0.90   # pass 1 labels a frame only when its most likely state has at least this probability (9 to 1; was 0.66 until 2026-11-17)
STATES = np.array(["H", "A", "D"])


def sure_labels(post, threshold=SURE_P):
    """Pass 1: the most likely of H / A / D where it reaches threshold, "R" (to be reviewed) where it does
    not, X where there are no probabilities (ungraded frames)."""
    known = np.all(np.isfinite(post), axis=1)
    best = np.argmax(np.where(known[:, None], post, 0.0), axis=1)
    out = np.where(np.max(np.where(known[:, None], post, 0.0), axis=1) >= threshold, STATES[best], "R")
    return np.where(known, out, "X").astype("<U1")


def fill_review(pass1, post, blocks=BLOCKS):
    """Pass 2: fill every review stretch from the sure labels around it, more strictly than frame by frame.

    Same label on both sides: the stretch takes it, so an unsure moment inside a possession never
    becomes a change (no flip without strong evidence). Different labels: exactly one change, placed at
    the split that agrees best with the probabilities (most P(left) before it plus P(right) after it).
    A stretch touching a half's edge takes its one neighbour; a half with no sure frame takes the argmax."""
    out = pass1.copy()
    col = {s: i for i, s in enumerate(STATES)}
    for _, first, last in blocks:
        seg = out[first:last + 1]   # a view
        p = post[first:last + 1]
        rev = (seg == "R").astype(np.int8)
        edges = np.diff(np.concatenate(([0], rev, [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            left = seg[a - 1] if a > 0 else None
            right = seg[b] if b < len(seg) else None
            if left is None and right is None:
                seg[a:b] = STATES[np.argmax(p[a:b], axis=1)]
            elif left is None or right is None or left == right:
                seg[a:b] = left if left is not None else right
            else:
                pl, pr = p[a:b, col[left]], p[a:b, col[right]]
                score = np.concatenate(([0.0], np.cumsum(pl))) + np.concatenate(([pr.sum()], pr.sum() - np.cumsum(pr)))
                k = int(np.argmax(score))
                seg[a:a + k], seg[a + k:b] = left, right
    return out


# Params (pass 2 of solution 11, all from how possession is defined, not tuned)
RESTART_MPS = 3.0    # a restart: the ball moves faster than a ball being placed (tag_timing.LIVE_MS)
SHORT_SPELL = 50     # frames: spells under 2 s inside the other team's possession are checked
CONFIRM = 5          # frames of clear control (0.2 s) that make such a spell a real possession


def strict_fill(pass1, post, cues, blocks=BLOCKS):
    """Pass 2 with football cues; fill_review's probability split where the cue is absent.

    For a review stretch from label L to a different label R, R starts at the first frame of its cue:
    H/A -> A/H: the new team's first control frame (a player within 1.5 m of a slow ball), cues["H"/"A"];
    H/A -> D: the ball past a line, cues["out"]; D -> H/A: the ball moving again, cues["moving"]."""
    out = fill_review(pass1, post, blocks)
    for _, first, last in blocks:
        seg, src = out[first:last + 1], pass1[first:last + 1]
        rev = (src == "R").astype(np.int8)
        edges = np.diff(np.concatenate(([0], rev, [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if a == 0 or b == len(seg) or src[a - 1] == src[b]:
                continue
            left, right = src[a - 1], src[b]
            cue = cues["out"] if right == "D" else cues["moving"] if left == "D" else cues[right]
            hit = np.flatnonzero(cue[first + a:first + b])
            if len(hit):
                seg[a:a + hit[0]], seg[a + hit[0]:b] = left, right
    return out


def vet_short_spells(labels, clear, blocks=BLOCKS, max_len=SHORT_SPELL, need=CONFIRM):
    """A spell of one team shorter than max_len with the OTHER team on both sides is a real possession
    only if that team clearly controls the ball for at least `need` frames in it; otherwise it goes back to
    the surrounding team. Length alone decides nothing: a real possession can last under a second."""
    out = labels.copy()
    for _, first, last in blocks:
        seg = out[first:last + 1]
        cuts = np.concatenate(([0], np.flatnonzero(seg[1:] != seg[:-1]) + 1, [len(seg)]))
        for i in range(1, len(cuts) - 2):
            a, b = cuts[i], cuts[i + 1]
            k, before, after = seg[a], seg[a - 1], seg[b]
            if k in "HA" and before == after and before in "HA" and before != k and b - a < max_len:
                if clear[k][first + a:first + b].sum() < need:
                    seg[a:b] = before
    return out


DUEL_NEAR = 12          # frames: a duel within 0.5 s
DUEL_MULT, CALM_MULT = 1.73, 0.69   # 0.513 / 0.297 and 0.487 / 0.703, solution 12's changes (2026-11-17)
SHAPE_LAG = 40          # frames: team shape agrees with a new owner a median 1.6 s after a seen-ball change


def backdate_blackout_changes(labels, seen, blocks=BLOCKS, lag=SHAPE_LAG):
    """Move each H <-> A change that falls while the ball is missing `lag` frames earlier, but never before
    the ball was last seen (until then the ball itself said who had it) nor into the spell before."""
    out = labels.copy()
    for _, first, last in blocks:
        seg = out[first:last + 1]
        cuts = np.concatenate(([0], np.flatnonzero(seg[1:] != seg[:-1]) + 1))
        for i in range(1, len(cuts)):
            t = cuts[i]
            if seg[t] not in "HA" or seg[t - 1] not in "HA" or seen[first + t]:
                continue
            last_seen = np.flatnonzero(seen[first + cuts[i - 1]:first + t])
            floor = cuts[i - 1] + (last_seen[-1] + 1 if len(last_seen) else 1)
            new = max(t - lag, floor)
            seg[new:t] = seg[t]
    return out


def _self_test():
    lab = np.array(list("HHHHHHHHHHAAAAA")); seen = np.zeros(15, bool); seen[:4] = True
    assert "".join(backdate_blackout_changes(lab, seen, [(1, 0, 14)], lag=3)) == "HHHHHHHAAAAAAAA"
    assert "".join(backdate_blackout_changes(lab, seen, [(1, 0, 14)], lag=9)) == "HHHHAAAAAAAAAAA", "not before the last seen ball"
    p1 = np.array(list("HHRRRRAA")); pp = np.full((8, 3), 1 / 3)
    cues = {"H": np.zeros(8, bool), "A": np.zeros(8, bool), "out": np.zeros(8, bool), "moving": np.zeros(8, bool)}
    cues["A"][4] = True
    assert "".join(strict_fill(p1, pp, cues, [(1, 0, 7)])) == "HHHHAAAA", "A starts at its first control frame"
    lab = np.array(list("HHHAAHHHAAAHHH")); clear = {"H": np.zeros(14, bool), "A": np.zeros(14, bool)}
    clear["A"][8:11] = True
    assert "".join(vet_short_spells(lab, clear, [(1, 0, 13)], max_len=4, need=2)) == "HHHHHHHHAAAHHH", "an unbacked blip goes"
    post = np.array([[.9, .05, .05], [.5, .4, .1], [.4, .5, .1], [.2, .7, .1], [.9, .05, .05], [.5, .45, .05],
                     [np.nan] * 3, [.3, .3, .4], [.8, .1, .1]])
    p1 = sure_labels(post, 0.66)   # toy probabilities written for a 2-to-1 cut
    assert "".join(p1) == "HRRAHRXRH", p1
    out = fill_review(p1, post, [(1, 0, 5), (2, 7, 8)])
    assert "".join(out) == "HHAAHHXHH", out   # H|A split at the crossing; unsure end of a half keeps H; X stays
    p1 = np.array(list("HRRRH")); pp = np.array([[.9, .1, 0], [.2, .8, 0], [.2, .8, 0], [.2, .8, 0], [.9, .1, 0]])
    assert "".join(fill_review(p1, pp, [(1, 0, 4)])) == "HHHHH", "unsure frames between two H stay H"
    out = dead_in_windows(np.array(list("XHHAAXHHAA")), [(0, 3), (8, 20), (5, 5)], [(1, 1, 4), (2, 6, 9)])
    assert "".join(out) == "XDDAAXHHDD", out  # clipped to the graded blocks; empty window does nothing
    labels = np.array(list("XHHAAXHHAA"))
    out = dead_before_kickoff(labels, [(1, 1, 4), (2, 6, 9)], {"1st Half": 1, "2nd Half": 8})
    assert "".join(out) == "XHHAAXDDAA", out  # first half starts at its kickoff: unchanged
    assert "".join(labels) == "XHHAAXHHAA", "input must not be modified"
    out = dead_before_kickoff(labels, [(1, 1, 4), (2, 6, 9)], {"1st Half": 1, "2nd Half": 99})
    assert "".join(out) == "XHHAAXDDDD", "a kickoff past the block end stops at the block"


def _nearest_codes():
    meta, raw = require(["atd_meta", "atd_raw"])
    return nearest_player_labels(raw, team_of_each_track(meta), N_FRAMES, PITCH)


def solution_01(window=50):
    """Baseline: nearest player, D when the ball is unseen, majority vote over +/- window frames."""
    return smooth(_nearest_codes(), window)


def solution_02(k=3, sigma=2.0, window=50):
    """The k nearest players vote, weighted by a Gaussian of distance; otherwise the baseline."""
    from knn_vote import knn_labels
    meta, raw = require(["atd_meta", "atd_raw"])
    return smooth(knn_labels(raw, team_of_each_track(meta), N_FRAMES, PITCH, k, sigma), window)


def solution_03(max_gap=-1, window=50):
    """A missing ball keeps the last owner (every gap when max_gap is -1), then the majority vote."""
    from fill_missing import fill_gaps
    return smooth(fill_gaps(_nearest_codes(), BLOCKS, max_gap), window)


def _shape_and_speed(cleaning="C0"):
    """Shared by 04 and 05: tracking arrays, team-shape evidence for blackouts, speed dead evidence."""
    from blackout_hmm import dead_evidence, fit_shape_model, nearest_team, shape_matrix
    players, ball, team = load_clean(cleaning)
    graded = graded_mask()
    feat = team_features(players, team)
    owner, dist = nearest_team(players, ball, team)
    X = shape_matrix(feat, BLOCKS, PITCH[0])
    model = fit_shape_model(X, owner, dist, graded)[0]
    finite = np.all(np.isfinite(X), axis=1)
    shape_llr = np.zeros(N_FRAMES)
    shape_llr[finite] = np.clip(model.decision_function(X[finite]), -3, 3)
    speed_dead = dead_evidence(feat["speed_all"], graded)[0]
    return players, ball, team, owner, shape_llr, speed_dead


def solution_04(tau=5):
    """H / A / D Viterbi: flat nearest-player evidence, team shape in blackouts, speed for dead balls."""
    from blackout_hmm import P_BALL, decode
    _, _, _, owner, shape_llr, speed_dead = _shape_and_speed()
    ball_llr = np.where(owner == HOME, 1.0, np.where(owner >= 0, -1.0, 0.0)) * np.log(P_BALL / (1 - P_BALL))
    team_llr = np.where(owner < 0, shape_llr, ball_llr)
    return decode(team_llr, speed_dead, BLOCKS, N_FRAMES, True, tau)


def duel_switch_mult(players, ball, team):
    """Per-frame multiplier on the H <-> A switch chance: DUEL_MULT within 0.5 s of a duel (two opponents near
    the ball running at each other), CALM_MULT otherwise. Bayes: P(duel near | change) / P(duel near) and
    P(no duel | change) / P(no duel), measured on solution 12's own labels (self-consistency, no truth)."""
    from features import central_velocity, duels
    pos = players[..., :2]
    vel = np.stack([central_velocity(pos[:, i], 3) for i in range(pos.shape[1])], axis=1)
    near = np.convolve((duels(pos, vel, team, ball) > 0).astype(float), np.ones(2 * DUEL_NEAR + 1), "same") > 0
    return np.where(near, DUEL_MULT, CALM_MULT)


def _control_details(tau, count_near, min_evidence, cleaning="C0", min_hold=0, blackout_scale=1.0, keep_live=False,
                     with_posteriors=False, duel_switch=False):
    """Shared by 05 and 06: labels plus the per-frame evidence behind them, for the viewer.

    min_evidence applies only where the ball is seen; during a blackout control cannot be judged.
    min_hold and blackout_scale are solution 08's two changes to the chain (see solution_08).
    """
    from blackout_hmm import decode
    from control_hmm import (ball_dead_evidence, ball_speed, control_evidence, place_with_hindsight,
                             team_distances)
    players, ball, team, owner, shape_llr, speed_dead = _shape_and_speed(cleaning)
    speed = ball_speed(ball)
    d_home, d_away = team_distances(players, ball, team)
    control = control_evidence(d_home, d_away, speed, count_near)
    team_llr = np.where(owner < 0, shape_llr * blackout_scale, control)
    # keep_live: live-against-dead stays as strong as before the blackout scaling; only H against A is damped
    live_llr = np.abs(np.where(owner < 0, shape_llr, control)) if keep_live else None
    dead_llr = speed_dead + ball_dead_evidence(ball, speed, PITCH)
    switch_mult = duel_switch_mult(players, ball, team) if duel_switch else None
    labels = place_with_hindsight(decode(team_llr, dead_llr, BLOCKS, N_FRAMES, True, tau, switch_mult=switch_mult,
                                         min_hold=min_hold, live_llr=live_llr), team_llr,
                                  BLOCKS, np.where(owner < 0, 0.0, min_evidence))
    labels = dead_before_kickoff(labels)
    out = {"labels": labels, "team_llr": team_llr, "dead_llr": dead_llr, "d_home": d_home,
           "d_away": d_away, "ball_speed": speed, "blackout": owner < 0, "tau": tau}
    if with_posteriors:
        from blackout_hmm import posteriors
        out["post"] = posteriors(team_llr, dead_llr, BLOCKS, N_FRAMES, True, tau, live_llr, switch_mult)
        out["ball_seen"] = np.isfinite(ball[:, 0])
        out["ball_out"] = ball_dead_evidence(ball, speed, PITCH) > 0   # past a line (and not just fast)
    return out


def solution_05_details(tau=1, cleaning="K5"):
    return _control_details(tau, count_near=True, min_evidence=0.0, cleaning=cleaning)


def solution_05(tau=1, cleaning="K5"):
    """Control-weighted evidence, ball out/fast dead evidence, Viterbi, hindsight placement.
    cleaning picks a version from experiments/cleaning.py; K5 (misplaced keepers blanked, wrong-team
    stretches re-linked or moved on the broadcast's say, impossible positions removed; 2026-11-16) is the data the submission uses."""
    return solution_05_details(tau, cleaning)["labels"]


def solution_08_details(tau=1, cleaning="K5", min_hold=0, blackout_scale=None, keep_live=True, with_posteriors=False,
                        duel_switch=False):
    from blackout_hmm import BLACKOUT_SCALE
    return _control_details(tau, count_near=True, min_evidence=0.0, cleaning=cleaning, min_hold=min_hold,
                            blackout_scale=BLACKOUT_SCALE if blackout_scale is None else blackout_scale,
                            keep_live=keep_live, with_posteriors=with_posteriors, duel_switch=duel_switch)


def solution_08(tau=1, cleaning="K5"):
    """05, but team-shape evidence during a blackout counts once a second instead of once a frame
    (shape changes slowly, so 25 frames are one observation). Shape alone then needs about 2 s at full
    strength to change team. Live against dead is left exactly as in 05.

    min_hold (blackout_hmm.MIN_HOLD, a possession lasts at least 1 s) is available but off: the public
    truth contains a possession of 0.76 s, so the labels do not follow such a minimum."""
    return solution_08_details(tau, cleaning)["labels"]


def solution_09(tau=1, cleaning="K5"):
    """08 plus two stoppage rules from our earlier v18 pipeline, both from the laws of the game:
    - goal tags: from the goal to the kickoff the ball is dead (dead_in_windows);
    - set-piece tags: a throw-in, corner, goal kick or free kick needs a stoppage before it, so where
      08 has no D two seconds into the tag, tag_restart.restart_fill adds one from the ball going out
      (at the line the restart type needs) to the restart touch, at most 25 s.
    The possession after the restart is left as 08 has it. Applied to every tag, public window included."""
    sys.path.insert(0, os.path.dirname(HERE))
    from tag_restart import restart_fill
    return restart_fill(dead_in_windows(solution_08(tau, cleaning), goal_tag_windows()))


def solution_10_details(tau=1, cleaning="K5", threshold=SURE_P):
    d = solution_08_details(tau, cleaning, with_posteriors=True)
    pass1 = sure_labels(d["post"], threshold)
    labels = dead_before_kickoff(fill_review(pass1, d["post"]))
    sys.path.insert(0, os.path.dirname(HERE))
    from tag_restart import restart_fill
    labels = restart_fill(dead_in_windows(labels, goal_tag_windows()))
    return {"labels": labels, "pass1": pass1, "post": d["post"]}


def solution_10(tau=1, cleaning="K5"):
    """Two passes over 08's evidence, then 09's stoppage rules.
    Pass 1: each frame gets the probability of H, A and D given all the evidence before and after it
    (forward-backward on 08's chain); it is labelled only if the best state reaches 90% (SURE_P), else "to review".
    Pass 2 (fill_review): a review stretch between two sure labels of the same kind keeps that label; between
    different ones it gets one change, where the probabilities place it best."""
    return solution_10_details(tau, cleaning)["labels"]


def set_piece_tags():
    """[(start frame, type)] of the Smart Tagging set pieces other than kickoffs (throw-in, corner, ...)."""
    import load_smart_tagging
    return [(t["start_frame"], t["tags"].get("Type")) for t in load_smart_tagging.load()
            if (t["code"] or "").strip().upper() == "SET PIECES" and not (t["tags"].get("Type") or "").startswith("Kick Off")]


def solution_12_details(tau=1, cleaning="K5", threshold=SURE_P, duel_switch=False):
    from dead_reasons import classify_dead_runs, drop_unexplained, restart_on_kick
    d = solution_11_details(tau, cleaning, threshold, duel_switch)
    labels = restart_on_kick(d["labels"], d["ball_speed"], BLOCKS)
    args = (BLOCKS, d["ball_out"], set_piece_tags(), goal_tag_windows(), list(KICKOFF_VIDEO_FRAMES.values()))
    labels = drop_unexplained(labels, classify_dead_runs(labels, *args), d["cues"])
    return dict(d, labels=labels, reasons=classify_dead_runs(labels, *args))


def solution_12(tau=1, cleaning="K5"):
    """11, then two rules about stoppages: play restarts on the kick or throw (ball over 5 m/s), not when
    the ball is picked up or placed; and every dead ball needs a reason (goal, set-piece tag, ball out, or
    a stop of 5 s or more): one shorter than 5 s with none goes back to live play (dead_reasons.py)."""
    return solution_12_details(tau, cleaning)["labels"]


def solution_13_details(tau=1, cleaning="K5", threshold=SURE_P):
    d = solution_12_details(tau, cleaning, threshold, duel_switch=True)
    return dict(d, labels=backdate_blackout_changes(d["labels"], d["ball_seen"]))


def solution_13(tau=1, cleaning="K5"):
    """12 with two things the players' behaviour shows (experiments/behaviour_report.py):
    - a change of team is 1.73x likelier within 0.5 s of a duel (two opponents near the ball running at each
      other) and 0.69x without one; the chain's switch chance follows that frame by frame;
    - team shape trails a change by a median 1.6 s, so a change found only from shape while the ball is
      missing is moved 1.6 s earlier, never before the ball was last seen."""
    return solution_13_details(tau, cleaning)["labels"]


def solution_11_details(tau=1, cleaning="K5", threshold=SURE_P, duel_switch=False):
    from control_hmm import L_CONTESTED, L_CONTROL
    d = solution_08_details(tau, cleaning, with_posteriors=True, duel_switch=duel_switch)
    llr = d["team_llr"]
    cues = {"H": llr >= L_CONTESTED - 1e-9, "A": llr <= -L_CONTESTED + 1e-9, "out": d["ball_out"],
            "moving": np.nan_to_num(d["ball_speed"]) > RESTART_MPS}
    clear = {"H": llr >= L_CONTROL - 1e-9, "A": llr <= -L_CONTROL + 1e-9}
    pass1 = sure_labels(d["post"], threshold)
    labels = dead_before_kickoff(vet_short_spells(strict_fill(pass1, d["post"], cues), clear))
    sys.path.insert(0, os.path.dirname(HERE))
    from tag_restart import restart_fill
    labels = restart_fill(dead_in_windows(labels, goal_tag_windows()))
    return {"labels": labels, "pass1": pass1, "post": d["post"], "ball_speed": d["ball_speed"],
            "ball_out": d["ball_out"], "cues": cues, "ball_seen": d["ball_seen"]}


def solution_11(tau=1, cleaning="K5"):
    """10 with a stricter pass 2, from how possession is defined:
    a change between teams lands on the new team's first control frame, a stoppage starts when the ball
    crosses a line, play restarts when the ball moves again (else 10's probability split); then every
    spell under 2 s inside the other team's possession must show 0.2 s of clear control or it is removed."""
    return solution_11_details(tau, cleaning)["labels"]


def solution_14(cleaning="K5"):
    """Every rule as evidence in one probability model; see experiments/unified_hmm.py."""
    from unified_hmm import solution_14 as unified
    return unified(cleaning)


def solution_15(cleaning="K5"):
    """14 + changes the players do not react to are false alarms; see experiments/role_swap.py."""
    from unified_hmm import solution_15 as reacted
    return reacted(cleaning)


def solution_16(cleaning="K5"):
    """15 + changes made while the ball is missing move to where the teams swap roles; see role_swap.py."""
    from unified_hmm import solution_16 as retimed
    return retimed(cleaning)


def solution_17(cleaning="K5"):
    """16 + the ball-free double check: control the ball does not move with is suspect where the movement
    disagrees, and the movement-only opinion is team evidence everywhere; see experiments/double_check.py."""
    from unified_hmm import solution_17 as checked
    return checked(cleaning)


def solution_18(cleaning="K5"):
    """17 + ball events as evidence: nearness by a team that has not played the ball counts a quarter, the
    last team to play it gets evidence once a second, and a ball last seen flying off the pitch is out;
    see experiments/ball_events.py and unified_hmm.event_evidence."""
    from unified_hmm import solution_18 as with_events
    return with_events(cleaning)


def solution_19(cleaning="K5"):
    """18 + a pass over every live frame: it goes to the team that last played the ball (a lone touch in a
    duel does not count), and a spell under 3 s whose team never played the ball is removed; see
    experiments/ball_events.py."""
    from unified_hmm import solution_19 as by_events
    return by_events(cleaning)


def solution_20(cleaning="K5"):
    """19 + one more reason for a stoppage: slow players while 9 or more stand in one penalty area (both teams
    waiting for a corner or a free kick); see unified_hmm.crowded_box_mask."""
    from unified_hmm import solution_20 as with_box
    return with_box(cleaning)


def solution_21(cleaning="K5"):
    """20 + a fuller reaction check: a change of team must show the teams swapping forward speed and width
    together. The weight is fitted on seven labelled outside matches (idsse_reactions.py), not on ours."""
    from unified_hmm import solution_21 as with_reaction
    return with_reaction(cleaning)


def solution_22(cleaning="K5"):
    """21 + one touch is not a possession: a short spell between two spells of the other team stands only if
    that team played the ball twice in a row (a header and a team-mate's touch); a lone header or block goes
    back to the team that had the ball. See ball_events.merge_lone_touch_spells."""
    from unified_hmm import solution_22 as with_two_plays
    return with_two_plays(cleaning)


def solution_23(cleaning="K5"):
    """22 + two more players-only measures in the double check: the team without the ball keeps changing
    velocity (it reacts to every pass) and moves as a block. Found on labelled outside matches
    (idsse_explore.py); the model itself is still fitted on this match's own trusted frames."""
    from unified_hmm import solution_23 as with_reacting
    return with_reacting(cleaning)


def solution_24(cleaning="K5"):
    """22 + no stoppage shorter than 3 s: in seven labelled outside matches only 2.2% of true stoppages are
    that short, while a quarter of ours were. Such a spell is live play and goes to the team that had the ball."""
    from unified_hmm import solution_24 as without_short_stoppages
    return without_short_stoppages(cleaning)


def solution_25():
    """24 on the K6 tracking. K5 deletes a goalkeeper track standing in the half his team attacks, but on about
    1,200 graded frames that track is the real goalkeeper on the ball (the tracker gave him the other keeper's
    slot), so his kick went to the nearest opponent. K6 keeps whichever keeper track is nearer the goal."""
    from unified_hmm import solution_25 as on_k6
    return on_k6()


def solution_26():
    """Which touches by the other team start a possession is learned on seven labelled outside matches
    (idsse_touches.py) instead of ruled: it replaces the duel rule (19) and "two plays" (22), which those
    matches show are wrong more often than counting every touch. Otherwise 25: K6 tracking, the reaction check
    with width, crowded box, no stoppage under 3 s."""
    from unified_hmm import solution_26 as learned_touches
    return learned_touches()


def solution_27():
    """Clean the data, read the match once, clean again with what that reading says, then decide. The second
    cleaning removes ball touches credited to one team while the first pass is at least 90% sure, for half a
    second either side, that the other team has the ball: the ball passed beside an opponent or a track carries
    the wrong team. Otherwise 24."""
    from unified_hmm import solution_27 as second_cleaning
    return second_cleaning()


def solution_28():
    """24 + every goal kick is one stoppage: dead from the ball crossing the goal line until the ball, at rest
    in that goal area, is kicked (Law 16). The goalkeeper fetching and carrying the ball looks like clear control
    by his team, so the chain alone calls it possession."""
    from unified_hmm import solution_28 as goal_kicks
    return goal_kicks()


def solution_29():
    """28 + a last look at every short block: one team for under 4 s inside the other team's possession stands
    only if the teams reacted to it (they swap forward speed and width). A misplaced ball can make such a
    block out of nothing; the players' reaction does not depend on the ball."""
    from unified_hmm import solution_29 as final_check
    return final_check()


def solution_30():
    """29 + a back check from every restart. A throw-in, corner or goal kick goes to the team that did NOT touch
    the ball last. So when a stoppage is followed by the team the labels say had the ball right up to it, and the
    latest touch before the stoppage is the other team's, that last stretch (the touch that put the ball out) is
    the other team's possession. One touch is enough here: the restart confirms it."""
    from unified_hmm import solution_30 as out_touches
    return out_touches()


def solution_31():
    """30 + short "play" between two stoppages that is only the ball being moved to the restart is dead: under
    5 s, and either the players stay slow throughout or the ball is never kicked or thrown (only carried)."""
    from unified_hmm import solution_31 as islands
    return islands()


def solution_32():
    """31 on tracking that joins both cameras: a player both see is the average of the two (3 to 1 for the
    broadcast when bunched), a player only the broadcast sees is added while his team shows under 11."""
    from unified_hmm import solution_32 as fused
    return fused()


def solution_33():
    """32 + a player whose team the two cameras disagree on (over a lasting stretch of his track) does not count
    as either team's control of the ball while he is near it."""
    from unified_hmm import solution_33 as careful
    return careful()


def solution_34():
    """32 + a ball touch counts only when the player who made it is one the tracking is sure of: not a player seen
    by the broadcast camera only, and not one the two cameras put on different teams."""
    from unified_hmm import solution_34 as doubtful
    return doubtful()


def solution_35():
    """34 + the broadcast camera's ball on frames where the tactical camera has none (sightings of 1 s or more
    that join on to the tactical ball within 3 m)."""
    from unified_hmm import solution_35 as both_balls
    return both_balls()


def solution_36():
    """34 + a whistle nobody can see: a sharp collective slowdown (mean speed down 2 m/s to walking pace) is a
    reason for a stoppage even with no ball out, no tag and no challenge (offside)."""
    from unified_hmm import solution_36 as whistles
    return whistles()


def solution_37():
    """36 + the two-plays rule at a whistle: a spell under 3 s with fewer than two plays that ends in a whistle
    stoppage goes back to the team that had the ball before it."""
    from unified_hmm import solution_37 as before_whistle
    return before_whistle()


def solution_42():
    """The steps of solution 40 asked as five questions in order (final_chain.py): trust the data, form a first
    opinion, settle whose ball it is, check that it holds up, and last settle exactly when play is stopped."""
    from final_chain import final
    return final()


def solution_41():
    """40 + nearness to the ball as graded evidence: the three hand-set strengths of "the nearest team has the ball"
    become a table measured on the outside matches (distance of both teams' nearest players, hard pressing)."""
    from unified_hmm import solution_41 as graded
    return graded()


def solution_40():
    """39 + a ball that is clearly out is out. (1) Evidence: where the tactical camera has no ball and the broadcast
    shows it 2 m or more past a line for a second or more, that is a ball out. (2) A rule, applied last: where
    either camera shows that, a live frame becomes dead. This overrides the 3 s minimum stoppage of solution 24:
    the final labels can hold a stoppage as short as 1 s."""
    from unified_hmm import solution_40 as out_seen
    return out_seen()


def solution_39():
    """38 + completed passes: two plays by the same team joined by a ball that is seen, fast and straight are a pass;
    the kick is never a duel touch and an opponent the ball passed did not play it."""
    from unified_hmm import solution_39 as passes
    return passes()


def solution_38():
    """37 + the collaborator's slowdown rule: a stretch where all players slow to walking pace for 3 s or more with
    the ball mostly unseen is a whistle stoppage (experiments/slowdown.py). It finds different stoppages from 36."""
    from unified_hmm import solution_38 as slowdown
    return slowdown()


def solution_06_details(tau=SOLUTION_06_TAU, cleaning="K5"):
    from control_hmm import L_CONTESTED
    return _control_details(tau, count_near=False, min_evidence=L_CONTESTED, cleaning=cleaning)


def solution_06(tau=SOLUTION_06_TAU, cleaning="K5"):
    """05 with the "chasing is not control" rule: possession changes only on the new team's control."""
    return solution_06_details(tau, cleaning)["labels"]


# ---------- mix and match: every evidence piece built once, any combination decoded ----------
#
# A combination picks one option per part:
#   rule      "05": a player 1.5-3 m from a slow ball is weak evidence; "06": it is none, and a
#             change of team waits for the new team's first control frame (the long-kick rule)
#   blackout  "carry": no evidence while the ball is unseen (hold the owner, like solution 03);
#             "shape": the team-shape model (05); "shape+move": shape plus who runs forward and
#             who closes the ball down (the movement findings)
#   pressure  switching up to 3x easier while the ball carrier is under pressure
#   airborne  a nearby player is not control when the two cameras say the ball is probably in the air
#   predicted control tests on the predicted ball for the first second of a blackout, at half weight
#   tags      Smart Tagging votes: phases 0.4, attacking transitions and set pieces 0.85;
#             True counts them on every frame, "live" only where play looks live
COMBO_DEFAULT = {"rule": "05", "blackout": "shape", "pressure": False, "airborne": False,
                 "predicted": False, "tags": False}
AIRBORNE_GAP_M = 2.0
TAG_PHASE, TAG_STRONG = 0.4, 0.85
TAG_TEAM = {"Barcelona": 1.0, "Atletic Club": -1.0}
_PARTS = {}


def _tag_evidence(n):
    """Signed per-frame evidence from Smart Tagging (+ home). Defending tags mirror attacking ones,
    so only the attacking side of each pair votes; throw-ins (team N/A) say nothing."""
    import load_smart_tagging
    out = np.zeros(n)
    for t in load_smart_tagging.load():
        code, sign = (t["code"] or "").upper(), TAG_TEAM.get(t["tags"].get("Team"))
        if sign is None or code.startswith("DEFEND") or code in ("DEFENSIVE TRANSITION", "GOALS"):
            continue
        weight = TAG_STRONG if code in ("ATTACKING TRANSITION", "SET PIECES") else TAG_PHASE
        out[max(t["start_frame"], 0):min(t["end_frame"], n)] += sign * weight
    return out


def combo_parts():
    """Every evidence piece any combination can use, computed once per process."""
    if _PARTS:
        return _PARTS
    from blackout_hmm import dead_evidence, fit_shape_model, nearest_team, shape_matrix
    from control_hmm import ball_dead_evidence, ball_speed, control_evidence, team_distances
    from feature_cache import load_features
    players, ball, team = load()
    graded = graded_mask()
    tfeat = team_features(players, team)
    feat = load_features()
    owner, dist = nearest_team(players, ball, team)
    speed = ball_speed(ball)
    d_home, d_away = team_distances(players, ball, team)
    pred = feat["pred"]
    pd_home, pd_away = team_distances(players, pred, team)

    def shape_llr(X):
        model = fit_shape_model(X, owner, dist, graded)[0]
        finite = np.all(np.isfinite(X), axis=1)
        out = np.zeros(N_FRAMES)
        out[finite] = np.clip(model.decision_function(X[finite]), -3, 3)
        return out

    X = shape_matrix(tfeat, BLOCKS, PITCH[0])
    X_move = np.column_stack([X, feat["move_forward_H"] - feat["move_forward_A"],
                              feat["move_closing_H"].astype(float) - feat["move_closing_A"]])
    press = np.nan_to_num(feat["press_index"])
    _PARTS.update({
        "owner": owner, "blackout": owner < 0, "ball_seen": np.isfinite(ball[:, 0]),
        "ctrl": {r: control_evidence(d_home, d_away, speed, r == "05") for r in ("05", "06")},
        "pred_ctrl": {r: control_evidence(pd_home, pd_away, np.zeros(N_FRAMES), r == "05") for r in ("05", "06")},
        "shape": shape_llr(X), "shape+move": shape_llr(X_move),
        "dead": dead_evidence(tfeat["speed_all"], graded)[0] + ball_dead_evidence(ball, speed, PITCH),
        "airborne": np.nan_to_num(feat["air_gap"]) > AIRBORNE_GAP_M,
        "physics": feat["pred_src"] == 1,
        "press_mult": np.where(feat["press_carrier"] >= 0, 1 + 2 * np.clip(press, 0, 1), 1.0),
        "tags": _tag_evidence(N_FRAMES),
    })
    return _PARTS


def combo_evidence(cfg):
    """(team_llr, dead_llr, switch_mult) for one combination."""
    cfg = {**COMBO_DEFAULT, **cfg}
    parts = combo_parts()
    ctrl = parts["ctrl"][cfg["rule"]]
    if cfg["airborne"]:
        ctrl = np.where(parts["airborne"], 0.0, ctrl)
    blackout = np.zeros(N_FRAMES) if cfg["blackout"] == "carry" else parts[cfg["blackout"]]
    if cfg["predicted"]:
        blackout = blackout + np.where(parts["physics"], 0.5 * parts["pred_ctrl"][cfg["rule"]], 0.0)
    team_llr = np.where(parts["blackout"], blackout, ctrl)
    if cfg["tags"] == "live":
        # a tag names the team with the ball while play is on; during a stoppage its vote would
        # fight the dead-ball state, so it only counts where the dead-ball evidence says live
        team_llr = team_llr + np.where(parts["dead"] <= 0, parts["tags"], 0.0)
    elif cfg["tags"]:
        team_llr = team_llr + parts["tags"]
    switch_mult = parts["press_mult"] if cfg["pressure"] else None
    return team_llr, parts["dead"], switch_mult


def combo_labels(cfg, tau):
    from blackout_hmm import decode
    from control_hmm import L_CONTESTED, place_with_hindsight
    cfg = {**COMBO_DEFAULT, **cfg}
    team_llr, dead_llr, switch_mult = combo_evidence(cfg)
    labels = decode(team_llr, dead_llr, BLOCKS, N_FRAMES, True, tau, switch_mult)
    strict = L_CONTESTED if cfg["rule"] == "06" else 0.0
    min_evidence = np.where(combo_parts()["blackout"], 0.0, strict)
    return place_with_hindsight(labels, team_llr, BLOCKS, min_evidence)


def mean_spell_s(labels):
    spells = live = 0
    for _, first, last in BLOCKS:
        s = labels[first:last + 1]
        spells += int(np.sum((s[1:] != s[:-1]) & (s[1:] != "D"))) + int(s[0] != "D")
        live += int(np.sum(s != "D"))
    return live / max(spells, 1) / 25


def self_consistent_tau(cfg, target_s=8.7, probes=(0.75, 1.5)):
    """The TAU whose decoded mean possession matches the priors (8.7 s), by interpolating two probes
    on a log-log line, then rounding to 2 significant figures so the choice is stable."""
    (t1, t2) = probes
    m1, m2 = mean_spell_s(combo_labels(cfg, t1)), mean_spell_s(combo_labels(cfg, t2))
    slope = np.log(m2 / m1) / np.log(t2 / t1)
    tau = float(np.exp(np.log(t1) + np.log(target_s / m1) / slope)) if slope > 0 else 1.0
    return float(f"{np.clip(tau, 0.2, 5.0):.2g}")


# Chosen without the ground truth: from the combination grid, keep the options that do not add
# sub-second possessions (flicker, the costliest error for spellF1). Airborne guard (+10 per
# match) and Smart Tagging votes (+15 to +30) are left out.
SOLUTION_07 = {"rule": "05", "blackout": "shape+move", "pressure": True, "airborne": False,
               "predicted": True, "tags": False}
SOLUTION_07_TAU = 1.0  # self-consistency: decoded mean possession 8.5 s against the priors' 8.7 s


def solution_07(tau=None):
    """Solution 05 plus movement in blackouts, pressure-dependent switching, and control tests on
    the predicted ball for the first second of a blackout."""
    return combo_labels(SOLUTION_07, tau or SOLUTION_07_TAU or self_consistent_tau(SOLUTION_07))


SOLUTIONS = {
    "01": solution_01,
    "02": solution_02,
    "03": solution_03,
    "04": solution_04,
    "05": solution_05,
    "06": solution_06,
    "07": solution_07,
    "08": solution_08,
    "09": solution_09,
    "10": solution_10,
    "11": solution_11,
    "12": solution_12,
    "13": solution_13,
    "14": solution_14,
    "15": solution_15,
    "16": solution_16,
    "17": solution_17,
    "18": solution_18,
    "19": solution_19,
    "20": solution_20,
    "21": solution_21,
    "22": solution_22,
    "23": solution_23,
    "24": solution_24,
    "25": solution_25,
    "26": solution_26,
    "27": solution_27,
    "28": solution_28,
    "29": solution_29,
    "30": solution_30,
    "31": solution_31,
    "32": solution_32,
    "33": solution_33,
    "34": solution_34,
    "35": solution_35,
    "36": solution_36,
    "37": solution_37,
    "38": solution_38,
    "39": solution_39,
    "40": solution_40,
    "41": solution_41,
    "42": solution_42,
}
