"""The whole method in one readable place, as five questions asked in order.

    1. TRUST          What in the data can we believe?              cleaned tracking, both cameras joined (K7)
    2. FIRST OPINION  What does the evidence say, frame by frame?   a probability model over home / away / dead
    3. WHOSE BALL     Which team has it?                            touches, passes, two plays
    4. HOLDS UP       Does that make sense as football?             reactions, restarts
    5. IN PLAY        Exactly when is play stopped?                 the stoppage rules

Why this order. The first opinion already says, roughly, when play is live and who has the ball. Question 3
settles who has the ball on the frames it calls live, from who actually played it. Question 4 checks that
answer against how the teams behave. Question 5 comes last so that nothing after it can undo a stoppage: the
stoppage rules have the final word on when the ball is dead.

Each step below is one small function: labels in, labels out. Nothing is switched on through hidden
module state; the options the evidence needs are written here.

Two orders of the same twelve steps:
  final()    the five questions in order. This is solution 42.
  proven()   the order the steps were invented in. It reproduces solution 40 byte for byte, which proves that
             this file is the same method as the stack of numbered solutions in unified_hmm.py.

    python experiments/final_chain.py        # checks proven() against solution 40, then compares final() with it
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Params
CLEANING = "K7"                                             # question 1: the tracking every step reads
OPTIONS = {"whistles": True, "broadcast_out": True}         # extra reasons for a stoppage in the evidence


# ---------- questions 1 and 2: what we trust, and the first opinion ----------

def context():
    """Everything the steps share: the tracking, the first opinion and what it was built from."""
    import unified_hmm as U
    from cleaning import camera_sources
    from control_hmm import L_CONTROL, ball_speed
    from feature_cache import load_features
    from features import central_velocity
    from piece_teams import disputed_tracks
    from pipelines import BLOCKS, PITCH, _shape_and_speed
    from role_swap import reaction_series
    from tracking import load_clean
    players, ball, team = load_clean(CLEANING)
    first = U.solution_14_details(CLEANING, double_check=True, events=True, crowded_box=True, options=OPTIONS)
    pos = players[..., :2]
    vel = np.stack([central_velocity(pos[:, i], 3) for i in range(pos.shape[1])], axis=1)
    right = load_features()["attacks_right"].astype(bool)   # the same on raw and K7 tracking (checked: 0 frames differ)
    return {
        "players": players, "ball": ball, "team": team, "blocks": BLOCKS, "pitch": PITCH,
        "seen": np.isfinite(ball[:, 0]), "ball_speed": ball_speed(ball),
        "first_opinion": first["labels"], "fouls": first["fouls"],
        # how each team is moving: forward speed and width (the reaction to winning or losing the ball)
        "reaction": {s: reaction_series(pos, vel, team, right, s) for s in (0, 1)},
        "clear_control": {"H": first["team_llr"] >= L_CONTROL - 1e-9, "A": first["team_llr"] <= -L_CONTROL + 1e-9},
        # players we are unsure of: seen by the broadcast camera only, or put on different teams by the two cameras
        "doubtful": (camera_sources(players, load_clean("K5")[0]) == 2) | disputed_tracks(players, team, BLOCKS),
        "players_slow": _shape_and_speed(CLEANING)[5] > 0,
    }


# ---------- question 5: exactly when is play stopped? ----------

def goal_kick_is_one_stoppage(c, labels):
    """Dead from the ball crossing the goal line until the goal kick is taken (solution 28)."""
    from unified_hmm import goal_kick_stoppages
    return goal_kick_stoppages(labels, c["ball"], c["ball_speed"], c["blocks"], c["pitch"])[0]


def placing_the_ball_is_not_play(c, labels):
    """"Play" under 5 s between two stoppages is dead if the players stay slow or the ball is only carried (31)."""
    from unified_hmm import stoppage_islands
    return stoppage_islands(labels, c["players_slow"], c["ball_speed"], c["seen"], c["blocks"])[0]


def whistle_with_no_ball_out(c, labels):
    """Everyone slows to walking pace for 3 s or more with the ball mostly unseen: play was stopped (38)."""
    from slowdown import slowdown_stoppages
    from tracking import team_features
    from unified_hmm import KICK_MPS
    speed_all = team_features(c["players"], c["team"])["speed_all"]
    return slowdown_stoppages(labels, speed_all, c["seen"], np.nan_to_num(c["ball_speed"]) > KICK_MPS, c["blocks"])[0]


def no_stoppage_under_3_seconds(c, labels):
    """A stoppage shorter than 3 s is live play; the team that had the ball keeps it (24)."""
    from unified_hmm import drop_short_stoppages
    return drop_short_stoppages(labels, c["blocks"])[0]


def ball_clearly_out_is_dead(c, labels):
    """Either camera shows the ball 2 m or more past a line for a second or more: dead, however short (40)."""
    from ball_cameras import broadcast_ball_out, clearly_out
    out = clearly_out(c["ball"], c["pitch"]) | broadcast_ball_out(c["players"], c["ball"], c["team"], c["pitch"])
    labels = labels.copy()
    labels[out & ((labels == "H") | (labels == "A"))] = "D"
    return labels


# ---------- question 3: whose ball? ----------

def change_needs_a_reaction(c, labels):
    """A short spell of the other team stands only if the teams swap roles or the new team clearly controls the
    ball (15, 21)."""
    from role_swap import calm_threshold, vet_by_reaction
    c["calm"] = calm_threshold(labels, c["reaction"], c["blocks"])
    return vet_by_reaction(labels, c["reaction"], c["clear_control"], c["blocks"], c["calm"])[0]


def unseen_change_moves_to_the_role_swap(c, labels):
    """A change of team made while the ball is missing moves to where the teams swap roles (16)."""
    from role_swap import retime_blackout_changes
    return retime_blackout_changes(labels, c["reaction"], c["seen"], c["blocks"], c["calm"])[0]


def last_team_to_play_the_ball(c, labels):
    """Each live frame goes to the team that last played the ball. A touch by a doubtful player is no evidence
    (34); a completed pass stays with the passing team (39); a lone touch in a duel does not count; a spell with
    no touch of its own goes back (19)."""
    from ball_events import event_possession, merge_untouched_spells
    labels, c["touches"] = event_possession(labels, c["ball"], c["players"][..., :2], c["team"],
                                            doubtful=c["doubtful"], passes=True)
    return merge_untouched_spells(labels, c["touches"]["frames"], c["touches"]["who"], c["seen"], c["blocks"])[0]


def one_touch_is_not_a_possession(c, labels):
    """A short spell between two spells of the other team needs two plays of its own (22)."""
    from ball_events import merge_lone_touch_spells
    t = c["touches"]
    return merge_lone_touch_spells(labels, t["frames"], t["who"], c["seen"], c["blocks"])[0]


def two_plays_before_a_whistle(c, labels):
    """A team that gets the ball under 3 s before a whistle, with fewer than two plays, has not had it (37).
    Spells the restart check made on purpose are left alone."""
    from ball_events import merge_spells_before_whistle
    from foul_detect import AFTER
    t = c["touches"]
    whistles = [f + AFTER[0] for f, _ in c["fouls"]]
    return merge_spells_before_whistle(labels, t["raw_frames"], t["raw_who"], whistles, c["blocks"],
                                       protect=c.get("made_by_restart_check"))[0]


# ---------- question 4: does it hold up? ----------

def short_block_needs_a_reaction(c, labels):
    """One team for under 4 s inside the other's possession stands only if the teams reacted (29)."""
    from role_swap import calm_threshold, vet_by_reaction
    from unified_hmm import FINAL_CHECK_MAX
    never = {"H": np.zeros(len(labels), bool), "A": np.zeros(len(labels), bool)}
    calm = calm_threshold(labels, c["reaction"], c["blocks"])
    return vet_by_reaction(labels, c["reaction"], never, c["blocks"], calm, max_len=FINAL_CHECK_MAX)[0]


def whoever_restarts_did_not_put_it_out(c, labels):
    """The team that restarts did not play the ball out, so the other team's last touch before the stoppage is a
    possession (30)."""
    from ball_events import restart_back_check
    from pipelines import set_piece_tags
    from unified_hmm import GOAL_AREA_M, RESTART_LINE_M
    length, width = c["pitch"]
    x, y = c["ball"][:, 0], c["ball"][:, 1]
    with np.errstate(invalid="ignore"):
        beyond = (x < 0) | (x > length) | (y < 0) | (y > width)
        at_line = np.minimum.reduce([x, length - x, y, width - y]) <= RESTART_LINE_M
        in_goal_area = (np.minimum(x, length - x) <= GOAL_AREA_M[0]) & (np.abs(y - width / 2) <= GOAL_AREA_M[1])
    out_tags = [t for t, kind in set_piece_tags() if kind in ("Throw-in", "Corner Kick", "Goal Kick")]
    t = c["touches"]
    new = restart_back_check(labels, t["raw_frames"], t["raw_who"], c["blocks"], beyond, out_tags, at_line | in_goal_area)[0]
    c["made_by_restart_check"] = new != labels
    return new


# ---------- the two orders ----------

def run(steps):
    c = context()
    labels = c["first_opinion"]
    for step in steps:
        labels = step(c, labels)
    return labels


PROVEN = [change_needs_a_reaction, unseen_change_moves_to_the_role_swap, last_team_to_play_the_ball,
          one_touch_is_not_a_possession, no_stoppage_under_3_seconds, goal_kick_is_one_stoppage,
          short_block_needs_a_reaction, whoever_restarts_did_not_put_it_out, placing_the_ball_is_not_play,
          two_plays_before_a_whistle, whistle_with_no_ball_out, ball_clearly_out_is_dead]

IN_PLAY = [goal_kick_is_one_stoppage, placing_the_ball_is_not_play, whistle_with_no_ball_out,
           no_stoppage_under_3_seconds, ball_clearly_out_is_dead]
WHOSE_BALL = [change_needs_a_reaction, unseen_change_moves_to_the_role_swap, last_team_to_play_the_ball,
              one_touch_is_not_a_possession, two_plays_before_a_whistle]
HOLDS_UP = [short_block_needs_a_reaction, whoever_restarts_did_not_put_it_out]
FINAL = WHOSE_BALL + HOLDS_UP + IN_PLAY


def proven():
    """Solution 40's labels: the steps in the order they were invented."""
    return run(PROVEN)


def final():
    """Solution 42: the same steps as five questions in order: whose ball, does it hold up, when is play stopped."""
    return run(FINAL)


def _self_test():
    assert sorted(f.__name__ for f in PROVEN) == sorted(f.__name__ for f in FINAL), "both orders use the same steps"
    assert len(set(FINAL)) == len(FINAL) == 12
    c = {"ball": np.array([[50.0, -3.0]] * 40 + [[50.0, 30.0]] * 10), "pitch": (105.0, 68.0)}
    import ball_cameras
    real = ball_cameras.broadcast_ball_out
    ball_cameras.broadcast_ball_out = lambda *a: np.zeros(50, bool)
    try:
        c.update(players=None, team=None)
        got = ball_clearly_out_is_dead(c, np.array(["H"] * 50))
    finally:
        ball_cameras.broadcast_ball_out = real
    assert (got[:40] == "D").all() and (got[40:] == "H").all()
    print("final_chain self-test passed")


if __name__ == "__main__":
    _self_test()
    import hashlib
    from pipelines import SOLUTIONS
    from score import RESULTS, graded_mask, save_and_score, write_submission
    a = proven()
    same = np.array_equal(a, SOLUTIONS["40"]())
    path = write_submission(os.path.join(RESULTS, "final_proven.csv"), a)
    print("proven order equals solution 40: {} (sha256 {})".format(same, hashlib.sha256(open(path, "rb").read()).hexdigest()[:8]))
    b = final()
    g = graded_mask()
    print("final order differs from the proven one on {:.2%} of graded frames".format((a[g] != b[g]).mean()))
    save_and_score(b, "sol42")
