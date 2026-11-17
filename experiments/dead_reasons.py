"""Why the ball is dead: every stoppage gets a reason, and the restart waits for the kick.

    from dead_reasons import classify_dead_runs, drop_unexplained, restart_on_kick

A stoppage in football always has a cause: a goal, the ball over a line (throw-in, goal kick, corner),
a foul or an injury (free kick), the half's start. Smart Tagging names most set pieces; the ball's own
position shows it going out. A dead ball with none of these and shorter than any real restart takes is
a decoder artefact (slow live play), so it goes back to live play. And a restart starts when the ball is
kicked or thrown, not when a keeper or taker picks it up and walks it to the spot.
All thresholds are set from the game, not tuned.
"""

import numpy as np

# Params
FPS = 25
MIN_STOPPAGE = 5 * FPS      # even a quick free kick or throw-in takes about 5 s from whistle to restart
TAG_REACH = 10 * FPS        # a set-piece tag within 10 s of the stoppage names it (tags start after the stop)
OUT_BEFORE = 2 * FPS        # the ball crossed a line up to 2 s before the dead ball began
KICK_MPS = 5.0              # a placed, carried or rolled ball is slower; any kick or throw is faster
KICK_LOOK = 10 * FPS        # the restart comes within 10 s (median about 7 s after the tag)
FOUL_BEFORE = 3 * FPS       # a likely foul this long before a stoppage explains it (the whistle follows the challenge)


def dead_runs(labels, blocks):
    """[(start, stop)] of every D run inside the halves, half-open."""
    out = []
    for _, first, last in blocks:
        seg = labels[first:last + 1]
        cuts = np.concatenate(([0], np.flatnonzero(seg[1:] != seg[:-1]) + 1, [len(seg)]))
        out += [(first + int(cuts[i]), first + int(cuts[i + 1])) for i in range(len(cuts) - 1) if seg[cuts[i]] == "D"]
    return out


def classify_dead_runs(labels, blocks, ball_out, set_pieces, goals, kickoffs, tag_reach=TAG_REACH,
                       out_before=OUT_BEFORE, min_stop=MIN_STOPPAGE, fouls=()):
    """[(start, stop, reason)] for each D run. set_pieces = [(tag start frame, type)], goals = [(start, stop)],
    kickoffs = the halves' kickoff frames, fouls = frames of likely fouls (foul_detect). Reasons, first that
    applies: before kickoff, goal, the set-piece type, ball out (no tag), foul (from movement: a likely foul
    up to FOUL_BEFORE before the stop), long stop (no tag, ball in: foul or injury?), unexplained."""
    out = []
    for a, b in dead_runs(labels, blocks):
        if any(a < k and b >= k for k in kickoffs) or any(b <= k and k - b < 2 for k in kickoffs):
            why = "before kickoff"
        elif any(a < ge and b > gs for gs, ge in goals):
            why = "goal"
        else:
            near = [kind for s, kind in set_pieces if a - tag_reach <= s <= b + tag_reach]
            if near:
                why = near[0] or "set piece"
            elif ball_out[max(a - out_before, 0):b].any():
                why = "ball out (no tag)"
            elif any(a - FOUL_BEFORE <= f <= a + FPS for f in fouls):
                why = "foul (from movement)"
            elif b - a >= min_stop:
                why = "long stop, no tag (foul or injury?)"
            else:
                why = "unexplained"
        out.append((a, b, why))
    return out


def drop_unexplained(labels, reasons, team_cue):
    """Unexplained dead balls go back to live play. Same team on both sides: that team. Different teams:
    the left one until the right one's first control frame (team_cue["H"/"A"]), else the midpoint."""
    out = labels.copy()
    for a, b, why in reasons:
        if why != "unexplained":
            continue
        left = out[a - 1] if a > 0 else "D"
        right = out[b] if b < len(out) else "D"
        if left not in "HA" and right not in "HA":
            continue
        if left not in "HA" or right not in "HA" or left == right:
            out[a:b] = left if left in "HA" else right
            continue
        hit = np.flatnonzero(team_cue[right][a:b])
        k = a + (hit[0] if len(hit) else (b - a) // 2)
        out[a:k], out[k:b] = left, right
    return out


def restart_on_kick(labels, ball_speed, blocks, kick_mps=KICK_MPS, look=KICK_LOOK):
    """Each D -> H/A change moves to the first frame the ball is kicked or thrown (faster than kick_mps),
    looking at most `look` frames ahead and never past the end of that possession."""
    out = labels.copy()
    fast = np.nan_to_num(np.asarray(ball_speed, np.float64)) > kick_mps
    for _, first, last in blocks:
        seg = out[first:last + 1]
        cuts = np.concatenate(([0], np.flatnonzero(seg[1:] != seg[:-1]) + 1, [len(seg)]))
        for i in range(1, len(cuts) - 1):
            t, stop = cuts[i], cuts[i + 1]
            if seg[t - 1] != "D" or seg[t] not in "HA":
                continue
            hit = np.flatnonzero(fast[first + t:first + min(stop, t + look)])
            if len(hit) and hit[0] > 0:
                seg[t:t + hit[0]] = "D"
    return out


def _self_test():
    blocks = [(1, 0, 39)]
    lab = np.array(list("HHHHHDDDHHHHHHHHHHHHAAAAADDDDDDDDAAAAAAA"))
    speed = np.zeros(40); speed[11] = 9.0; speed[36] = 9.0
    k = restart_on_kick(lab, speed, blocks, look=10)
    assert "".join(k[5:12]) == "DDDDDDH", k   # D until the kick at frame 11
    assert "".join(k[33:37]) == "DDDA", k     # the ball sits 3 frames, then the restart kick
    out = np.zeros(40, bool); out[24] = True
    small = dict(tag_reach=5, out_before=2, min_stop=5)
    r = classify_dead_runs(lab, blocks, out, [(30, "Throw-in")], [], [], **small)
    assert [w for _, _, w in r] == ["unexplained", "Throw-in"], r
    r = classify_dead_runs(lab, blocks, out, [], [], [], **small)
    assert [w for _, _, w in r] == ["unexplained", "ball out (no tag)"], r
    assert classify_dead_runs(lab, blocks, out, [], [], [], fouls=[4], **small)[0][2] == "foul (from movement)"
    cue = {"H": np.zeros(40, bool), "A": np.zeros(40, bool)}
    fixed = drop_unexplained(lab, r, cue)
    assert "".join(fixed[:20]) == "H" * 20 and "".join(fixed[25:33]) == "DDDDDDDD", fixed
    assert classify_dead_runs(np.array(list("DDDDHH")), [(1, 0, 5)], np.zeros(6, bool), [], [], [4])[0][2] == "before kickoff"


if __name__ == "__main__":
    _self_test()
    print("dead_reasons self-test passed")
