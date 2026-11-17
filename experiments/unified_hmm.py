"""Solution 14: every rule as evidence in ONE probability model, so the label and its confidence agree.

    from unified_hmm import solution_14_details
    d = solution_14_details()      # labels, post (P(H), P(A), P(D) per frame), pass1, reasons

Solutions 09-13 decide most frames with probabilities and then overwrite some with hard rules (goal
windows, set-piece stoppages, restart on the kick, no dead ball without a reason). Here those rules are
log-odds added to the evidence of 08's three-state chain (H, A, D) before forward-backward runs, so a
strong contrary signal can still win and the probability shown is the probability of the label given:

  goal windows        dead +GOAL_L a frame (as strong as the ball being out)
  set pieces          where a tag implies a stoppage the chain missed (tag_restart's plan), dead +TAG_L a frame
  restart on the kick from the ball leaving play until it is next kicked or thrown (over KICK_MPS), the
                      taker or keeper holding the ball is not possession: control x SETUP_CONTROL, dead +SETUP_DEAD
  a reason for D      slow players far from any reason (tag, goal, ball out) count UNEXPLAINED_SCALE as much
                      towards dead ball, so a long stop still becomes D and a short slow spell does not
  duels               the switch chance x1.73 near a duel, x0.69 otherwise (13)
  likely fouls        (foul_detect) from the whistle (FOUL_DELAY after the challenge) until the free kick is
                      taken (the next fast ball), at most FOUL_MAX: dead +FOUL_L a frame, control x SETUP_CONTROL,
                      and slow players there count fully (the foul is their reason)

Then pass 1 (sure at SURE_P) and pass 2 (football cues, short spells need control) as in 11, and the
organisers' pre-kickoff frames as D. Every weight is set from the game, not tuned on the public window.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Params
FPS = 25
GOAL_L = 3.0             # = control_hmm.L_OUT: after a goal the ball is as surely dead as when it is out
TAG_L = 2.0              # a tag's implied stoppage: strong, but clear control (2.94) can still outvote it
KICK_MPS = 5.0           # a placed, carried or rolled ball is slower; a kick or throw is faster (dead_reasons)
SETUP_MAX = 15 * FPS     # a restart follows the ball going out within about 15 s
SETUP_CONTROL = 0.25     # holding the ball for a restart is a quarter as convincing as control in play
DISPUTED_NEUTRAL = False  # solution 33 turns this on for its own run
DOUBTFUL_TOUCHES = False  # solution 34 turns this on for its own run
WHISTLES = False          # solution 36 turns this on for its own run
PASSES = False            # solution 39 turns this on for its own run
BROADCAST_OUT = False     # solution 40 turns this on for its own run
PRESSURE = False          # solution 41 turns this on for its own run
SETUP_DEAD = 0.5
REASON_TAG = 10 * FPS    # a set-piece tag this close explains a stoppage (dead_reasons.TAG_REACH)
REASON_AFTER_OUT = 30 * FPS
UNEXPLAINED_SCALE = 0.5  # slow players alone: half the dead-ball weight
FOUL_DELAY = FPS         # the whistle comes about 1 s after the challenge (foul_detect's slowdown window)
FOUL_MAX = 20 * FPS      # a free kick is taken within about 20 s
FOUL_L = 1.0             # weaker than a tag (2): the foul is inferred from movement, not tagged


def setup_mask(ball_out, fast_ball, blocks, max_frames=SETUP_MAX):
    """Frames from the end of each ball-out run until the next fast ball, at most max_frames: the ball
    is being fetched and placed for the restart."""
    out = np.zeros(len(ball_out), bool)
    for _, first, last in blocks:
        o = ball_out[first:last + 1].astype(np.int8)
        ends = np.flatnonzero(np.diff(np.concatenate((o, [0]))) == -1) + 1   # first frame after an out run
        for e in ends:
            stop = min(e + max_frames, last + 1 - first)
            hit = np.flatnonzero(fast_ball[first + e:first + stop])
            out[first + e:first + (e + hit[0] if len(hit) else stop)] = True
    return out


def foul_mask(fouls, fast_ball, blocks, delay=FOUL_DELAY, max_frames=FOUL_MAX):
    """Frames from each likely foul's whistle until the next fast ball (the free kick), at most max_frames."""
    out = np.zeros(len(fast_ball), bool)
    ends = {first: last for _, first, last in blocks}
    for c in fouls:
        last = next((l for f, l in ends.items() if f <= c <= l), None)
        if last is None:
            continue
        a = c + delay
        stop = min(a + max_frames, last + 1)
        hit = np.flatnonzero(fast_ball[a:stop])
        out[a:a + hit[0] if len(hit) else stop] = True
    return out


def reason_mask(n, tag_frames, goals, ball_out):
    """Frames with a reason for a stoppage nearby: a set-piece tag, a goal window, a recent ball out."""
    m = np.zeros(n, bool)
    for t in tag_frames:
        m[max(t - REASON_TAG, 0):t + REASON_TAG] = True
    for a, b in goals:
        m[max(a, 0):b] = True
    after = np.convolve(ball_out.astype(float), np.ones(REASON_AFTER_OUT), "full")[:n] > 0
    return m | after


MOVE_AGREE = 0.7         # double check: movement this sure of the other team makes unmoving "control" suspect
MOVE_CLIP = 3.0          # movement opinion as log-odds, capped like the shape model
MOVE_UNTRUSTED_ONLY = True   # review of 17 (2026-11-17): movement counts only where the ball is missing or not moving with its player


def movement_evidence(players, ball, team, control, cleaning, reacting=False):
    """(movement log-odds for home, comoving mask, P(home) from movement) — the ball-free double check
    (double_check.py), trained on frames with the ball seen, clearly controlled and moving with its player."""
    from control_hmm import L_CONTROL
    from double_check import comoving, fit_movement_model, movement_matrix, movement_opinion
    from feature_cache import load_features
    from features import central_velocity
    from pipelines import PITCH, graded_mask
    pos = players[..., :2]
    vel = np.stack([central_velocity(pos[:, i], 3) for i in range(pos.shape[1])], axis=1)
    co = comoving(pos, vel, ball, central_velocity(ball, 3))
    X = movement_matrix(pos, vel, team, load_features()["attacks_right"].astype(bool), PITCH[0], reacting)
    trusted = graded_mask() & np.isfinite(ball[:, 0]) & (np.abs(control) >= L_CONTROL - 1e-9) & co
    model = fit_movement_model(X, (control > 0).astype(int), trusted)[0]
    p_home = movement_opinion(model, X)
    with np.errstate(divide="ignore", invalid="ignore"):
        llr = np.clip(np.log(p_home / (1 - p_home)), -MOVE_CLIP, MOVE_CLIP)
    return np.nan_to_num(llr), co, p_home


EVENT_NEAR = 12          # frames (0.5 s): a team "played the ball" if one of its ball events is this close
EVENT_SCALE = 1 / FPS    # = blackout_hmm.BLACKOUT_SCALE
LAST_MAX_AGE = 5 * FPS   # no pass is in flight for longer; an older last touch says nothing


def event_evidence(players, ball, team, control, co):
    """(control with the ball events applied, info) -- "the last team to PLAY the ball has it".

    Possession belongs to the team that last played the ball, not to whoever stands nearest: a marker is
    close to the ball without having it, and a pass in flight is still the passer's. So (ball_events.py):
    - nearness by a team that has not played the ball (no event of its own within 0.5 s, the ball not moving
      with its player) while the other team played it last counts x SETUP_CONTROL;
    - where nobody controls the ball, or the control is unplayed as above, the last team to play it gets
      evidence. Its strength is self-consistent: how often the last-touch team is the controlling team on
      frames we trust (clear control, ball moving with its player)."""
    from ball_events import last_touch_team, path_events
    from control_hmm import L_CONTROL
    from pipelines import graded_mask
    n = len(ball)
    seen = np.isfinite(ball[:, 0])
    frames, who = path_events(ball, players[..., :2], team)
    last = last_touch_team(n, frames, who, seen, LAST_MAX_AGE)
    played = {}
    for side in (0, 1):
        hit = np.zeros(n)
        hit[frames[who == side]] = 1
        played[side] = np.convolve(hit, np.ones(2 * EVENT_NEAR + 1), "same") > 0
    side_ctl = np.where(control > 0, 0, 1)
    own_event = np.where(side_ctl == 0, played[0], played[1])
    unplayed = seen & (control != 0) & ~co & (last >= 0) & (last != side_ctl) & ~own_event
    trusted = graded_mask() & seen & (np.abs(control) >= L_CONTROL - 1e-9) & co & (last >= 0)
    agree = float(np.mean(last[trusted] == side_ctl[trusted]))
    strength = min(np.log(agree / (1 - agree)), L_CONTROL)
    # one touch is one observation, however many frames the ball then flies: counted once a second, like
    # team shape in a blackout (first version counted it every frame: 12 possessions under 1 s, 98 under 3 s)
    strength = strength * EVENT_SCALE
    last_llr = np.where(last == 0, strength, np.where(last == 1, -strength, 0.0))
    control = np.where(unplayed, SETUP_CONTROL * control, control)
    control = np.where(seen & ((control == 0) | unplayed), control + last_llr, control)
    info = {"events": len(frames), "attributed": int((who >= 0).sum()), "agree": agree, "strength": float(strength),
            "unplayed": int((unplayed & graded_mask()).sum()), "frames": frames, "who": who}
    return control, info


BOX_CROWD = 9            # players in one penalty area: both teams have come up for a corner or a free kick
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16


def crowded_box_mask(players, pitch):
    """(T,) True where BOX_CROWD or more players stand in one penalty area. With the players slow, that is
    two teams waiting for a corner or a free kick to be delivered: a reason for a stoppage. Checked against
    the Smart Tagging: 9+ in a box in every one of the 12 windows before a tagged corner, on 5% of clear
    live play (AUC 0.98 corners, 0.95 free kicks)."""
    length, width = pitch
    x, y = players[..., 0], players[..., 1]
    with np.errstate(invalid="ignore"):
        central = np.abs(y - width / 2) <= BOX_HALF_WIDTH
        left = ((x >= 0) & (x <= BOX_DEPTH) & central).sum(axis=1)
        right = ((x >= length - BOX_DEPTH) & (x <= length) & central).sum(axis=1)
    return np.maximum(left, right) >= BOX_CROWD


def evidence(cleaning="K5", double_check=False, events=False, crowded_box=False, options=None):
    """(team_llr, live_llr, dead_llr, switch_mult, extras) for the unified chain. double_check (solution 17):
    control the ball does not move with, while the movement says the other team, counts x SETUP_CONTROL, and
    the movement opinion (once a second) is team evidence everywhere and replaces team shape in blackouts."""
    from blackout_hmm import BLACKOUT_SCALE
    from control_hmm import ball_dead_evidence, ball_speed, control_evidence, team_distances
    from pipelines import (BLOCKS, N_FRAMES, PITCH, _shape_and_speed, duel_switch_mult, goal_tag_windows,
                           set_piece_tags)
    # which optional steps are on: passed in explicitly (final_chain.py), or read from the module switches that the
    # numbered solutions set for their own run
    opt = {"pressure": PRESSURE, "disputed": DISPUTED_NEUTRAL, "broadcast_out": BROADCAST_OUT, "whistles": WHISTLES}
    opt.update(options or {})
    players, ball, team, owner, shape_llr, speed_dead = _shape_and_speed(cleaning)
    speed = ball_speed(ball)
    d_home, d_away = team_distances(players, ball, team)
    control = control_evidence(d_home, d_away, speed, True)
    if opt["pressure"]:        # 41: nearness graded by how near the other team is and whether it presses
        from pressure_evidence import make_control
        control = make_control(players, ball, team)(d_home, d_away, speed, True)
    if opt["disputed"]:        # a player the two cameras put on different teams is near the ball: his nearness
        from piece_teams import disputed_near_ball      # is weak evidence for either team (same weight as a set-up)
        control = np.where(disputed_near_ball(players, ball, team, BLOCKS), SETUP_CONTROL * control, control)
    ball_ev = ball_dead_evidence(ball, speed, PITCH)
    ball_out = ball_ev > 0
    event_info = None
    if events:
        # a ball last seen flying at a line it would cross within 2 s went out, though nobody saw it land
        from ball_events import heading_out
        from control_hmm import L_OUT
        gone = heading_out(ball, BLOCKS, PITCH)
        ball_ev = ball_ev + np.where(gone, L_OUT, 0.0)
        ball_out = ball_out | gone
    if opt["broadcast_out"]:   # 40: the broadcast camera alone shows the ball clearly off the pitch for a while
        from ball_cameras import broadcast_ball_out
        from control_hmm import L_OUT as OUT_L
        seen_out = broadcast_ball_out(players, ball, team, PITCH)
        ball_ev = ball_ev + np.where(seen_out, OUT_L, 0.0)
        ball_out = ball_out | seen_out
    fast = np.nan_to_num(speed) > KICK_MPS
    setup = setup_mask(ball_out, fast, BLOCKS)
    goals = goal_tag_windows()
    tags = set_piece_tags()
    from foul_detect import foul_candidates
    fouls = foul_candidates(players, ball, team, ball_out, BLOCKS)
    if opt["whistles"]:        # 36: a sharp collective slowdown is a whistle even with no challenge and no ball to see
        from foul_detect import OUT_WINDOW, SEPARATION, whistle_candidates
        # not where the ball goes out (the same window detect_fouls uses): players slow down after a ball out too,
        # that stoppage already has its reason, and 37 must not treat it as a whistle (it would undo 30's rule
        # that a last touch which puts the ball out is a possession)
        fouls = sorted(fouls + [w for w in whistle_candidates(players, BLOCKS)
                                if all(abs(w[0] - c) >= SEPARATION for c, _ in fouls)
                                and not ball_out[max(w[0] + OUT_WINDOW[0], 0):w[0] + OUT_WINDOW[1]].any()])
    foul = foul_mask([c for c, _ in fouls], fast, BLOCKS)
    reason = reason_mask(N_FRAMES, [t for t, _ in tags], goals, ball_out) | foul
    if crowded_box:
        reason = reason | crowded_box_mask(players, PITCH)
    move_p = None
    shape_live = shape_llr          # live against dead keeps team shape's strength: movement says who, not whether
    if double_check:
        # double_check="reacting" (solution 23) adds how much each team accelerates and moves as a block
        move_llr, co, move_p = movement_evidence(players, ball, team, control, cleaning, double_check == "reacting")
        suspect = np.isfinite(ball[:, 0]) & ~co & (np.sign(control) * move_llr <= -np.log(MOVE_AGREE / (1 - MOVE_AGREE)))
        control = np.where(suspect, SETUP_CONTROL * control, control)
        shape_llr = move_llr
    if events:
        if not double_check:
            from double_check import comoving
            from features import central_velocity
            pos = players[..., :2]
            vel = np.stack([central_velocity(pos[:, i], 3) for i in range(pos.shape[1])], axis=1)
            co = comoving(pos, vel, ball, central_velocity(ball, 3))
        control, event_info = event_evidence(players, ball, team, control, co)
        event_info["heading_out"] = int(gone.sum())
    control = np.where(setup | foul, SETUP_CONTROL * control, control)
    team_llr = np.where(owner < 0, shape_llr * BLACKOUT_SCALE, control)
    if double_check:
        # the double check is for a ball that may be wrong: where the ball is seen and moves with its player it
        # is trusted, and the movement opinion (slow: teams react over a second or two) would only blur short spells
        untrusted = np.isfinite(ball[:, 0]) & ~co if MOVE_UNTRUSTED_ONLY else np.isfinite(ball[:, 0])
        team_llr = np.where(untrusted, team_llr + shape_llr * BLACKOUT_SCALE, team_llr)
    live_llr = np.abs(np.where(owner < 0, shape_live, control))
    slow = np.where(reason | (speed_dead < 0), speed_dead, UNEXPLAINED_SCALE * speed_dead)
    goal = np.zeros(N_FRAMES)
    for a, b in goals:
        goal[max(a, 0):b] = GOAL_L
    dead = slow + ball_ev + goal + SETUP_DEAD * setup + FOUL_L * foul
    extras = {"ball_speed": speed, "ball_out": ball_out, "ball_seen": np.isfinite(ball[:, 0]), "tags": tags,
              "goals": goals, "fouls": fouls, "move_p": move_p, "event_info": event_info}
    return team_llr, live_llr, dead, duel_switch_mult(players, ball, team), extras


def solution_14_details(cleaning="K5", threshold=None, double_check=False, events=False, crowded_box=False, options=None):
    from blackout_hmm import decode, posteriors
    from control_hmm import L_CONTESTED, L_CONTROL
    from dead_reasons import classify_dead_runs
    from pipelines import (BLOCKS, KICKOFF_VIDEO_FRAMES, N_FRAMES, SURE_P, dead_before_kickoff, strict_fill,
                           sure_labels, vet_short_spells)
    sys.path.insert(0, os.path.dirname(HERE))
    import tag_restart
    team_llr, live_llr, dead, switch, x = evidence(cleaning, double_check, events, crowded_box, options)
    # set pieces: where the chain as it stands has no stoppage at a tag, tag_restart's plan says where one goes
    first = decode(team_llr, dead, BLOCKS, N_FRAMES, True, 1, switch_mult=switch, live_llr=live_llr)
    for _, s_, e_ in tag_restart._plan(first.copy(), True, tag_restart.CAP_S)[0]:
        dead[s_:e_] += TAG_L
    post = posteriors(team_llr, dead, BLOCKS, N_FRAMES, True, 1, live_llr, switch)
    pass1 = sure_labels(post, SURE_P if threshold is None else threshold)
    cues = {"H": team_llr >= L_CONTESTED - 1e-9, "A": team_llr <= -L_CONTESTED + 1e-9, "out": x["ball_out"],
            "moving": np.nan_to_num(x["ball_speed"]) > KICK_MPS}
    clear = {"H": team_llr >= L_CONTROL - 1e-9, "A": team_llr <= -L_CONTROL + 1e-9}
    labels = dead_before_kickoff(vet_short_spells(strict_fill(pass1, post, cues), clear))
    reasons = classify_dead_runs(labels, BLOCKS, x["ball_out"], x["tags"], x["goals"], list(KICKOFF_VIDEO_FRAMES.values()),
                                 fouls=[c for c, _ in x["fouls"]])
    return {"labels": labels, "post": post, "pass1": pass1, "reasons": reasons, "fouls": x["fouls"], "team_llr": team_llr,
            "move_p": x["move_p"], "event_info": x["event_info"], "dead_llr": dead}


def solution_15_details(cleaning="K5", double_check=False, events=False, crowded_box=False, reaction=False):
    """14, then every short spell of the other team inside one team's possession must show the players
    reacting (the teams swap roles) or clear control by the new team; otherwise it was a false alarm,
    e.g. a lofted ball projected next to the wrong player (role_swap.py)."""
    from control_hmm import L_CONTROL
    from feature_cache import load_features
    from features import central_velocity
    from pipelines import BLOCKS
    from role_swap import calm_threshold, forward_speed, reaction_series, vet_by_reaction
    from tracking import load_clean
    d = solution_14_details(cleaning, double_check=double_check, events=events, crowded_box=crowded_box)
    players, _, team = load_clean(cleaning)
    pos = players[..., :2]
    vel = np.stack([central_velocity(pos[:, i], 3) for i in range(pos.shape[1])], axis=1)
    right = load_features()["attacks_right"].astype(bool)
    # reaction (solution 21): forward speed and width swap together at a real change
    fwd = {s: reaction_series(pos, vel, team, right, s) if reaction else forward_speed(vel, team, right, s) for s in (0, 1)}
    clear = {"H": d["team_llr"] >= L_CONTROL - 1e-9, "A": d["team_llr"] <= -L_CONTROL + 1e-9}
    threshold = calm_threshold(d["labels"], fwd, BLOCKS)
    labels, removed = vet_by_reaction(d["labels"], fwd, clear, BLOCKS, threshold)
    return dict(d, labels=labels, removed=removed, threshold=threshold, fwd=fwd)


def solution_16_details(cleaning="K5", double_check=False, events=False, crowded_box=False, reaction=False):
    """15, then each change of team made while the ball is missing moves to where the teams swap roles
    (within 5 s, never before the ball was last seen); see role_swap.retime_blackout_changes."""
    from pipelines import BLOCKS
    from role_swap import retime_blackout_changes
    from tracking import load_clean
    d = solution_15_details(cleaning, double_check, events, crowded_box, reaction)
    _, ball, _ = load_clean(cleaning)
    labels, moved = retime_blackout_changes(d["labels"], d["fwd"], np.isfinite(ball[:, 0]), BLOCKS, d["threshold"])
    return dict(d, labels=labels, moved=moved)


def solution_16(cleaning="K5"):
    return solution_16_details(cleaning)["labels"]


def solution_17_details(cleaning="K5"):
    """16 with the ball-free double check inside the evidence (movement_evidence)."""
    return solution_16_details(cleaning, double_check=True)


def solution_17(cleaning="K5"):
    return solution_17_details(cleaning)["labels"]


def solution_18_details(cleaning="K5", double_check=True, crowded_box=False, reaction=False):
    """17 + ball events (ball_events.py): possession follows who last played the ball, nearness without
    playing it is suspect, and a ball last seen flying off the pitch is out."""
    return solution_16_details(cleaning, double_check=double_check, events=True, crowded_box=crowded_box, reaction=reaction)


def solution_18(cleaning="K5"):
    return solution_18_details(cleaning)["labels"]


def solution_19_details(cleaning="K5", crowded_box=False, reaction=False, double_check=True, learned_touches=False,
                        second_cleaning=False):
    """18, then a pass that asks of every live frame "who played the ball last?" (ball_events.py):
    the frame goes to that team, a lone touch in a duel does not count, and a spell under 1 s with no
    touch of its own goes back. Stoppages and frames with no known last touch stay as 18 has them."""
    from ball_events import event_possession, merge_untouched_spells
    from pipelines import BLOCKS
    from tracking import load_clean
    d = solution_18_details(cleaning, double_check, crowded_box=crowded_box, reaction=reaction)
    players, ball, team = load_clean(cleaning)
    doubtful = None
    if DOUBTFUL_TOUCHES:        # 34: a touch by a player one camera sees, or whose team the cameras dispute, is no evidence
        from cleaning import camera_sources
        from piece_teams import disputed_tracks
        doubtful = (camera_sources(players, load_clean("K5")[0]) == 2) | disputed_tracks(players, team, BLOCKS)
    labels, info = event_possession(d["labels"], ball, players[..., :2], team, learned=learned_touches,
                                    post=d["post"] if second_cleaning else None, doubtful=doubtful, passes=PASSES)
    labels, merged = merge_untouched_spells(labels, info["frames"], info["who"], np.isfinite(ball[:, 0]), BLOCKS)
    return dict(d, labels=labels, events=dict(info, merged=merged), before_events=d["labels"])


def solution_19(cleaning="K5"):
    return solution_19_details(cleaning)["labels"]


def solution_20_details(cleaning="K5"):
    """19 with one more reason for a stoppage: slow players while one penalty area is crowded (crowded_box_mask)."""
    return solution_19_details(cleaning, crowded_box=True)


def solution_20(cleaning="K5"):
    return solution_20_details(cleaning)["labels"]


def solution_21_details(cleaning="K5", double_check=True, **kw):
    """20 with a fuller reaction check: a change of team must show the teams swapping forward speed and width together
    (role_swap.reaction_series, weight from seven labelled outside matches)."""
    return solution_19_details(cleaning, crowded_box=True, reaction=True, double_check=double_check, **kw)


def solution_21(cleaning="K5"):
    return solution_21_details(cleaning)["labels"]


def solution_22_details(cleaning="K5", double_check=True, **kw):
    """21, then one touch is not a possession: a short spell between two spells of the other team needs two
    plays of its own (ball_events.merge_lone_touch_spells)."""
    from ball_events import merge_lone_touch_spells
    from pipelines import BLOCKS
    from tracking import load_clean
    d = solution_21_details(cleaning, double_check, **kw)
    _, ball, _ = load_clean(cleaning)
    e = d["events"]
    labels, lone = merge_lone_touch_spells(d["labels"], e["frames"], e["who"], np.isfinite(ball[:, 0]), BLOCKS)
    return dict(d, labels=labels, lone_touches=lone, before_lone=d["labels"])


def solution_22(cleaning="K5"):
    return solution_22_details(cleaning)["labels"]


MIN_STOPPAGE = 3 * FPS   # of 696 true stoppages in seven labelled outside matches, 2.2% last under 3 s (0.6% under 2 s)


def drop_short_stoppages(labels, blocks, min_len=MIN_STOPPAGE):
    """A stoppage shorter than min_len is not a stoppage: the ball has to go out, be fetched and be put back
    into play, and that takes longer (idsse_stoppages.py / log: the 1st percentile of true stoppages is
    2.5 s). Play was live, so the team that had the ball keeps it; with no team before, the next one has it.
    Stoppages at the start or end of a half are left alone."""
    out = labels.copy()
    dropped = 0
    for _, first, last in blocks:
        seg = out[first:last + 1]
        edges = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg)]
        for a, b in zip(edges[1:-2], edges[2:-1]):
            if seg[a] != "D" or b - a >= min_len:
                continue
            before, after = seg[a - 1], seg[b]
            if before in "HA" or after in "HA":
                seg[a:b] = before if before in "HA" else after
                dropped += 1
    return out, dropped


def solution_24_details(cleaning="K5", **kw):
    """22, then no stoppage shorter than 3 s (drop_short_stoppages)."""
    from pipelines import BLOCKS
    d = solution_22_details(cleaning, **kw)
    labels, dropped = drop_short_stoppages(d["labels"], BLOCKS)
    return dict(d, labels=labels, short_stoppages=dropped, before_stoppages=d["labels"])


def solution_24(cleaning="K5"):
    return solution_24_details(cleaning)["labels"]


GOAL_AREA_M = (5.5, 9.16)    # the goal area: 5.5 m deep, 9.16 m either side of the goal's centre (Law 1)
REST_MPS = 1.0               # a placed ball: slower than this (the tracker's wobble alone gives up to about 0.5) ...
REST_FRAMES = 10             # ... for 0.4 s
REST_KICK_WITHIN = FPS       # the kick shows within 1 s of the rest ending (the ball track is smoothed)
GOAL_KICK_MAX = 45 * FPS     # the ball went out at most this long before the goal kick


def goal_kick_stoppages(labels, ball, speed, blocks, pitch):
    """Dead from the ball crossing a goal line until the goal kick that follows.

    A goal kick is taken with the ball at rest inside the goal area (Law 16). While the goalkeeper fetches
    the ball and carries it there he looks like a player in clear control, and the others jog upfield, so
    neither control nor player speed shows the stoppage. But the two ends of it are visible: the ball
    beyond the goal line, and later the ball at rest in that goal area and then kicked. Everything between
    is dead. Checked against the Smart Tagging: 5 of the 6 such kicks in the match have a Goal Kick tag
    within 10 s (the tag starts about 3 s before the kick). Returns (labels, stoppages set)."""
    from control_hmm import OUT_MARGIN_M
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    depth, half = GOAL_AREA_M
    out = labels.copy()
    done = 0
    with np.errstate(invalid="ignore"):
        slow = np.isfinite(x) & (speed < REST_MPS)
        central = np.abs(y - width / 2) <= half + OUT_MARGIN_M
        ends = ((x <= depth + OUT_MARGIN_M) & (x >= -OUT_MARGIN_M) & central, x < -OUT_MARGIN_M,
                (x >= length - depth - OUT_MARGIN_M) & (x <= length + OUT_MARGIN_M) & central, x > length + OUT_MARGIN_M)
    for area, beyond in (ends[:2], ends[2:]):
        rest = np.convolve((area & slow).astype(int), np.ones(REST_FRAMES, int), "full")[:len(x)] >= REST_FRAMES
        for t in np.flatnonzero(rest[:-1] & ~rest[1:]):
            block = next(((f, l) for _, f, l in blocks if f <= t <= l), None)
            if block is None:
                continue
            fast = np.nan_to_num(speed[t:t + REST_KICK_WITHIN]) > KICK_MPS
            if not fast.any():
                continue
            kick = t + int(np.argmax(fast))
            lo = max(t - GOAL_KICK_MAX, block[0])
            gone = np.flatnonzero(beyond[lo:t])
            if len(gone) == 0:
                continue
            start = lo + gone[-1]
            while start > lo and beyond[start - 1]:        # back to the first frame of that run beyond the line
                start -= 1
            out[start:kick] = "D"
            done += 1
    return out, done


def solution_28_details(cleaning="K5", **kw):
    """24, then every goal kick is one stoppage, from the ball crossing the goal line to the kick (goal_kick_stoppages)."""
    from control_hmm import ball_speed
    from pipelines import BLOCKS, PITCH
    from tracking import load_clean
    d = solution_24_details(cleaning, **kw)
    _, ball, _ = load_clean(cleaning)
    labels, done = goal_kick_stoppages(d["labels"], ball, ball_speed(ball), BLOCKS, PITCH)
    return dict(d, labels=labels, goal_kicks=done, before_goal_kicks=d["labels"])


def solution_28():
    return solution_28_details()["labels"]


FINAL_CHECK_MAX = 4 * FPS    # the last check looks at blocks shorter than this (our rule, 2026-11-17)


def solution_29_details(cleaning="K5", **kw):
    """28, then a last look at every short block: one team for under 4 s inside the other team's possession
    stands only if the teams reacted to it (role_swap.vet_by_reaction on the finished labels). The ball's
    position alone cannot be trusted there: a misplaced ball makes such a block out of nothing. Unlike the check
    in 15, clear control by the new team does not save the block, because that control comes from the same ball.
    Whole-team reaction is used, not the five players nearest the ball: on seven labelled outside matches a real
    short block is told from calm play with AUC 0.80 by the whole team, 0.77 by the nearest five."""
    from pipelines import BLOCKS
    from role_swap import calm_threshold, vet_by_reaction
    d = solution_28_details(cleaning, **kw)
    never = {"H": np.zeros(len(d["labels"]), bool), "A": np.zeros(len(d["labels"]), bool)}
    threshold = calm_threshold(d["labels"], d["fwd"], BLOCKS)
    labels, removed = vet_by_reaction(d["labels"], d["fwd"], never, BLOCKS, threshold, max_len=FINAL_CHECK_MAX)
    return dict(d, labels=labels, final_removed=removed, final_threshold=threshold, before_final=d["labels"])


def solution_29():
    return solution_29_details()["labels"]


ISLAND_MAX = 5 * FPS         # live play between two stoppages is looked at when shorter than this
ISLAND_SLOW = 0.9            # players slow on this share of it: the stoppage never ended


def stoppage_islands(labels, slow, speed, seen, blocks, max_len=ISLAND_MAX, slow_share=ISLAND_SLOW):
    """A short stretch of "play" between two stoppages that is really the ball being moved to where the restart
    is taken: (labels, islands set dead).

    During one long stoppage the ball is fetched, carried and put down. For a moment a player has it at his
    feet and it moves, so the chain calls it possession, with a stoppage either side. Two things tell that
    from a real quick restart, and either is enough:
      - the players stay slow throughout (slow (T,) bool on at least slow_share of the stretch): nobody has
        started to play. On seven labelled outside matches the players are slow on 100% (median) of a window
        inside a stoppage and on 18% of real live play under 5 s between two stoppages; at 90% the rule
        catches 80% of the first and wrongly ends 20% of the second.
      - the ball, seen for at least half of the stretch, never moves faster than a kick or throw (KICK_MPS):
        play restarts with a kick or a throw, and a ball that is only carried or rolled has not been restarted."""
    out = labels.copy()
    done = 0
    fast = np.nan_to_num(speed) > KICK_MPS
    for _, first, last in blocks:
        live = out[first:last + 1] != "D"
        edges = np.flatnonzero(np.diff(np.r_[0, live.astype(np.int8), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if a == 0 or b == len(live) or b - a >= max_len:
                continue                                  # the first and last stretch of a half have no stoppage on one side
            w = slice(first + a, first + b)
            carried = seen[w].mean() >= 0.5 and not fast[w].any()
            if slow[w].mean() >= slow_share or carried:
                out[w] = "D"
                done += 1
    return out, done


def solution_31_details(cleaning="K5"):
    """30, then short "play" between two stoppages that is only the ball being moved to the restart is dead
    (stoppage_islands)."""
    from control_hmm import ball_speed
    from pipelines import BLOCKS, _shape_and_speed
    d = solution_30_details(cleaning)
    _, ball, _, _, _, speed_dead = _shape_and_speed(cleaning)
    labels, done = stoppage_islands(d["labels"], speed_dead > 0, ball_speed(ball), np.isfinite(ball[:, 0]), BLOCKS)
    return dict(d, labels=labels, islands=done, before_islands=d["labels"])


def solution_31():
    return solution_31_details()["labels"]


def solution_32_details():
    """31 read from both cameras: the same chain on the K7 tracking, where the tactical and broadcast players
    are joined frame by frame (cleaning.fuse_cameras, our rule 2026-11-17). Nothing else changes."""
    return solution_31_details("K7")


def solution_32():
    return solution_32_details()["labels"]


def solution_33_details():
    """32, not trusting a player whose team the two cameras disagree on. A track piece is one person, so where
    the broadcast names the other team for a lasting stretch of a piece (on frames where he stands free, so the
    pairing cannot be mistaken), one camera has his team wrong and nothing says which. While such a player is
    within 5 m of the ball, nearness to the ball counts a quarter, and the rest of the chain decides."""
    global DISPUTED_NEUTRAL
    DISPUTED_NEUTRAL = True
    try:
        return solution_31_details("K7")
    finally:
        DISPUTED_NEUTRAL = False


def solution_33():
    return solution_33_details()["labels"]


def solution_34_details(cleaning="K7"):
    """32, with touches counted only from players the tracking is sure of. The "last team to play the ball" step
    credits every ball event to the nearest player; where that player is seen by the broadcast camera only
    (cleaning K7 added him) or is on a stretch of track the two cameras put on different teams, the event is
    dropped: it is evidence for neither team. No position and no team is changed, and the probability model
    reads the same tracking as 32."""
    global DOUBTFUL_TOUCHES
    DOUBTFUL_TOUCHES = True
    try:
        return solution_31_details(cleaning)
    finally:
        DOUBTFUL_TOUCHES = False


def solution_34():
    return solution_34_details()["labels"]


def solution_36_details():
    """34 that also hears a whistle nobody can see. A stoppage needs a reason before slow players count fully as
    one; so far the reasons were the ball out, a set-piece tag, a goal, a crowded box, or a foul (a challenge at
    the ball, then a slowdown). An offside has none of these, and the ball is often out of the picture. Here a
    sharp collective slowdown alone (the players' mean speed falls by 2 m/s to walking pace within about two
    seconds; foul_detect.detect_whistles) counts as a reason, with the same weight and reach as a foul."""
    global WHISTLES
    WHISTLES = True
    try:
        return solution_34_details()
    finally:
        WHISTLES = False


def solution_36():
    return solution_36_details()["labels"]


def solution_37_details():
    """36, then the two-plays rule at a whistle (ball_events.merge_spells_before_whistle): a team that gets the ball
    under 3 s before a whistle stops play, and makes fewer than two plays, has not had a possession. The whistles
    are the likely fouls and unseen whistles the chain already used as reasons; balls going out are not whistles."""
    from ball_events import merge_spells_before_whistle
    from foul_detect import AFTER
    from pipelines import BLOCKS
    d = solution_36_details()
    e = d["events"]
    # a spell the restart back check (30) made on purpose is not undone here
    protect = np.asarray(d["before_back_check"]) != np.asarray(d["before_islands"])
    labels, merged = merge_spells_before_whistle(d["labels"], e["raw_frames"], e["raw_who"],
                                                 [c + AFTER[0] for c, _ in d["fouls"]], BLOCKS, protect=protect)
    return dict(d, labels=labels, before_whistle=merged, before_whistle_rule=d["labels"])


def solution_37():
    return solution_37_details()["labels"]


def solution_38_details(base="37"):
    """The collaborator's rule (branch solution-32-slowdown, experiments/slowdown.py) on the current chain: after
    the base solution's labels are final, a stretch is set dead where all the players slow to walking pace within
    a second or two, stay slow for 3 s or more, and the ball is seen on at most half of it; the stretch is cut at
    kicks. Written on top of 31 by its author; here it runs last, on 37's labels and the K7 tracking (base="34" or
    "36" puts it after those). Solution 38 here; its author called it 32, a number this repository had already used."""
    from control_hmm import ball_speed
    from pipelines import BLOCKS
    from slowdown import slowdown_stoppages
    from tracking import load_clean, team_features
    d = {"34": solution_34_details, "36": solution_36_details, "37": solution_37_details}[base]()
    players, ball, team = load_clean("K7")
    labels, found = slowdown_stoppages(d["labels"], team_features(players, team)["speed_all"],
                                       np.isfinite(ball[:, 0]), np.nan_to_num(ball_speed(ball)) > KICK_MPS, BLOCKS)
    return dict(d, labels=labels, slowdowns=found, before_slowdowns=d["labels"])


def solution_38():
    return solution_38_details()["labels"]


def solution_39_details():
    """38 with completed passes (our rule): a pass that is not intercepted is the passing team's from the kick
    to the reception. Two plays by the same team joined by a ball that is seen, fast and straight are a completed
    pass: neither end can be thrown away as a duel touch, and an opponent the ball merely passed is not a touch
    (ball_events.completed_passes)."""
    global PASSES
    PASSES = True
    try:
        return solution_38_details()
    finally:
        PASSES = False


def solution_39():
    return solution_39_details()["labels"]


def solution_40_details():
    """39 that believes the broadcast camera about one thing only: the ball being off the pitch. Where the
    tactical camera has no ball and the broadcast shows it at least 2 m past a line for a second or more, that is
    a ball out (ball_cameras.broadcast_ball_out): the same evidence, and the same reason for a stoppage, as the
    tactical ball going out. The broadcast ball is still not used to say who has the ball.
    And a ball that is clearly out is out (our rule): where the tactical camera shows the ball 2 m or more past
    a line for a second or more, or the broadcast alone does, the frame is dead whatever else the chain thought."""
    from ball_cameras import broadcast_ball_out, clearly_out
    from pipelines import PITCH
    from tracking import load_clean
    global BROADCAST_OUT
    BROADCAST_OUT = True
    try:
        d = solution_39_details()
    finally:
        BROADCAST_OUT = False
    players, ball, team = load_clean("K7")
    out = clearly_out(ball, PITCH) | broadcast_ball_out(players, ball, team, PITCH)
    labels = d["labels"].copy()
    forced = out & ((labels == "H") | (labels == "A"))
    labels[forced] = "D"
    return dict(d, labels=labels, forced_out=int(forced.sum()), before_forced_out=d["labels"])


def solution_40():
    return solution_40_details()["labels"]


def solution_41_details():
    """40 with nearness to the ball read as graded evidence (pressure_evidence.py): the probability model's three
    hand-set strengths for "the nearest team has the ball" (95 / 70 / 60%) are replaced by a table measured on the
    outside matches, by how near the nearest player is, how near the other team's nearest player is, and whether
    the other team presses hard. Everything after the probability model is unchanged."""
    global PRESSURE
    PRESSURE = True
    try:
        return solution_40_details()
    finally:
        PRESSURE = False


def solution_41():
    return solution_41_details()["labels"]


def solution_35_details():
    """34 with the broadcast camera's ball where the tactical camera has none (cleaning K9): only sightings of
    1 s or more that join on to the tactical ball within 3 m. Neither camera's ball is the more accurate where
    both see it (ball_cameras.py), so the tactical ball is kept wherever it exists."""
    return solution_34_details("K9")


def solution_35():
    return solution_35_details()["labels"]


RESTART_LINE_M = 3.0         # a throw-in or corner is taken within this of a line (taker's reach + projection error)


def solution_30_details(cleaning="K5"):
    """29, then a back check from every restart: the team that restarts did not play the ball out, so where the
    labels say it had the ball right up to the stoppage and the latest touch before it is the other team's,
    that last stretch is the other team's (ball_events.restart_back_check, our rule 2026-11-17)."""
    from ball_events import restart_back_check
    from pipelines import BLOCKS, PITCH, set_piece_tags
    from tracking import load_clean
    d = solution_29_details(cleaning)
    e = d["events"]
    _, ball, _ = load_clean(cleaning)
    length, width = PITCH
    with np.errstate(invalid="ignore"):                    # over a line by any margin; the 1 m margin of the chain guards live play
        beyond = (ball[:, 0] < 0) | (ball[:, 0] > length) | (ball[:, 1] < 0) | (ball[:, 1] > width)
    out_tags = [t for t, kind in set_piece_tags() if kind in ("Throw-in", "Corner Kick", "Goal Kick")]
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):                    # a restart from the boundary: at a line, or in a goal area
        at_line = np.minimum.reduce([x, length - x, y, width - y]) <= RESTART_LINE_M
        in_goal_area = (np.minimum(x, length - x) <= GOAL_AREA_M[0]) & (np.abs(y - width / 2) <= GOAL_AREA_M[1])
    labels, added = restart_back_check(d["labels"], e["raw_frames"], e["raw_who"], BLOCKS, beyond, out_tags,
                                       at_line | in_goal_area)
    return dict(d, labels=labels, out_possessions=added, before_back_check=d["labels"])


def solution_30():
    return solution_30_details()["labels"]


def solution_27_details():
    """Clean, read the match, clean again, decide (our workflow, 2026-11-17). 24 with a second cleaning between
    its two passes: touches that the first pass (the probability model) contradicts are data errors and are
    removed before the "last team to play the ball" pass (ball_events.drop_contradicted_touches)."""
    return solution_24_details(second_cleaning=True)


def solution_27():
    return solution_27_details()["labels"]


def solution_26_details(cleaning="K6"):
    """Which touches count is learned, not ruled: 21's chain on the K6 tracking with ball_events.drop_unlikely_touches
    in place of the duel rule, no "two plays" rule (22), then no stoppage under 3 s (24)."""
    from pipelines import BLOCKS
    d = solution_19_details(cleaning, crowded_box=True, reaction=True, learned_touches=True)
    labels, dropped = drop_short_stoppages(d["labels"], BLOCKS)
    return dict(d, labels=labels, short_stoppages=dropped, before_stoppages=d["labels"])


def solution_26():
    return solution_26_details()["labels"]


def solution_25_details():
    """24 on the K6 tracking: the goalkeeper nearer his goal is kept, where K5 deleted the misplaced keeper track
    even when it was the player on the ball (cleaning.keep_nearer_keeper)."""
    return solution_24_details("K6")


def solution_25():
    return solution_25_details()["labels"]


def solution_23_details(cleaning="K5"):
    """22 with two more measures in the players-only double check: the team without the ball accelerates more
    and moves as a block (double_check.reacting_columns)."""
    return solution_22_details(cleaning, double_check="reacting")


def solution_23(cleaning="K5"):
    return solution_23_details(cleaning)["labels"]


def solution_15(cleaning="K5"):
    return solution_15_details(cleaning)["labels"]


def solution_14(cleaning="K5"):
    return solution_14_details(cleaning)["labels"]


def _short_stoppage_self_test():
    lab = np.array(list("H" * 50 + "D" * 40 + "H" * 50 + "D" * 100 + "A" * 50 + "D" * 20 + "H" * 30 + "D" * 30))
    out, k = drop_short_stoppages(lab, [(1, 0, len(lab) - 1)])
    assert k == 2 and "".join(out[50:90]) == "H" * 40, "a 1.6 s stoppage inside a possession is live play"
    assert (out[140:240] == "D").all(), "a 4 s stoppage stays"
    assert (out[290:310] == "A").all(), "between two teams, the team that had the ball keeps it"
    assert (out[-30:] == "D").all(), "a stoppage at the end of the half stays"


def _goal_kick_self_test():
    n = 600
    ball = np.zeros((n, 2)); ball[:, 1] = 34.0
    ball[:100, 0] = np.linspace(20, -3, 100)          # rolls out over the goal line at x = 0 ...
    ball[100:300, 0] = np.linspace(-3, 4, 200)        # ... is carried back at walking pace ...
    ball[300:340, 0] = 4.0                            # ... placed in the goal area ...
    ball[340:, 0] = 4.0 + np.arange(n - 340) * 0.6    # ... and kicked at 15 m/s
    speed = np.r_[0, np.hypot(*np.diff(ball, axis=0).T) * FPS]
    lab = np.full(n, "H")
    out, k = goal_kick_stoppages(lab, ball, speed, [(1, 0, n - 1)], (105.0, 68.0))
    first_out = int(np.argmax(ball[:, 0] < -1.0))
    assert k == 1 and (out[first_out:339] == "D").all() and (out[:first_out] == "H").all() and (out[345:] == "H").all(), (k, first_out)
    ball[:300, 0] = 20.0                              # the ball never left the pitch: an ordinary pass back
    assert goal_kick_stoppages(lab, ball, np.r_[0, np.hypot(*np.diff(ball, axis=0).T) * FPS], [(1, 0, n - 1)], (105.0, 68.0))[1] == 0


def _island_self_test():
    lab = np.array(list("H" * 100 + "D" * 100 + "H" * 50 + "D" * 100 + "A" * 150 + "D" * 100 + "A" * 60 + "D" * 100 + "H" * 100))
    n = len(lab)
    slow = np.zeros(n, bool); slow[200:250] = True              # nobody moves during the first island (200-250)
    speed = np.full(n, 2.0); speed[610:620] = 12.0              # the last island (600-660) starts with a kick
    speed[400:410] = 12.0
    out, k = stoppage_islands(lab, slow, speed, np.ones(n, bool), [(1, 0, n - 1)])
    assert k == 1 and (out[100:350] == "D").all(), "slow players: the stoppage never ended"
    assert (out[600:660] == "A").all(), "a kick and players moving: a real quick restart"
    assert (out[350:500] == "A").all() and (out[:100] == "H").all(), "longer play is left alone"
    out, k = stoppage_islands(lab, np.zeros(n, bool), np.full(n, 2.0), np.ones(n, bool), [(1, 0, n - 1)])
    assert k == 2, "no kick or throw in either short island: the ball was only carried"


def _self_test():
    _island_self_test()
    _goal_kick_self_test()
    _short_stoppage_self_test()
    out = np.zeros(40, bool); out[5:8] = True
    fast = np.zeros(40, bool); fast[20] = True
    m = setup_mask(out, fast, [(1, 0, 39)], max_frames=30)
    assert m[8:20].all() and not m[20:].any() and not m[:8].any(), np.flatnonzero(m)
    m = setup_mask(out, np.zeros(40, bool), [(1, 0, 39)], max_frames=10)
    assert m[8:18].all() and not m[18:].any(), "capped when no kick is seen"
    fast = np.zeros(200, bool); fast[[60, 150]] = True
    f = foul_mask([30, 100], fast, [(1, 0, 199)], delay=5, max_frames=40)
    assert f[35:60].all() and not f[60:105].any() and f[105:145].all() and not f[145:].any(), np.flatnonzero(f)
    pl = np.full((2, 22, 3), np.nan); pl[0, :10, :2] = [100.0, 34.0]; pl[1, :8, :2] = [100.0, 34.0]
    assert crowded_box_mask(pl, (105.0, 68.0)).tolist() == [True, False]
    r = reason_mask(2000, [1000], [(100, 120)], np.zeros(2000, bool))
    assert r[1000 - REASON_TAG] and not r[1000 + REASON_TAG] and r[110] and not r[50]


if __name__ == "__main__":
    _self_test()
    print("unified_hmm self-test passed")
