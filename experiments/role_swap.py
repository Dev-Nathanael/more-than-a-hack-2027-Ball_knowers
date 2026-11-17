"""Reject changes of possession the players do not react to (solution 15).

    from role_swap import vet_by_reaction
    labels = vet_by_reaction(labels, fwd, clear, blocks, threshold)

A real turnover makes the teams swap roles: the side that lost the ball stops going forward and turns
back, the side that won it pushes forward (experiments/reaction_report.py: in the public truth the swap
is +2.4 m/s against +0.4 m/s in calm play). A change that comes from a mis-placed ball (a lofted ball
projected next to the wrong player) leaves everyone as they were.

So a spell of the other team inside one team's possession (L -> W -> L, W shorter than MAX_SPELL) is
kept only if the players swapped roles at its start more than they do in calm play (threshold: the
75th percentile of the swap at calm moments of the same labels) or the new team clearly controlled the
ball for CONFIRM frames. Otherwise it was a false alarm and goes back to L. Self-consistency only: the
threshold comes from the labels' own calm moments, never from the truth.
"""

import numpy as np

# Params
FPS = 25
BEFORE = (-FPS, 0)
AFTER = (FPS // 2, int(2.5 * FPS))
MAX_SPELL = 10 * FPS     # a mis-placed ball misleads for seconds, not for a whole attack
CONFIRM = 5              # frames of clear control (0.2 s) by the new team also confirm the change


def forward_speed(vel, team, attacks_right, side):
    """(T,) mean velocity of `side`'s players towards the goal they attack (m/s)."""
    home_dir = np.where(attacks_right, 1.0, -1.0)
    direction = home_dir if side == 0 else -home_dir
    return np.nanmean(vel[:, team == side, 0], axis=1) * direction


# Fitted on seven labelled outside matches (idsse_reactions.py, IDSSE: 770 real changes against 1,046 calm
# moments), per m/s of forward speed. Leaving each match out in turn the weight stays between 1.23 and 1.40
# and the score separates real changes from calm play with AUC 0.908 (forward speed alone 0.875).
# Sprinting back, fitted on one match at first, drops out on seven (weight -0.02) and is not used.
REACTION_WEIGHTS = {"width": 1.31}


def reaction_series(pos, vel, team, attacks_right, side):
    """(T,) forward speed of `side` plus the other thing a team does when it wins the ball, in forward-speed
    units: it spreads out (width, the spread of its players across the pitch). Used in place of
    forward_speed, so role_flip measures both behaviours swapping at once."""
    home_dir = np.where(attacks_right, 1.0, -1.0)
    direction = home_dir if side == 0 else -home_dir
    cols = team == side
    return (np.nanmean(vel[:, cols, 0] * direction[:, None], axis=1)
            + REACTION_WEIGHTS["width"] * np.nanstd(pos[:, cols, 1], axis=1))


def role_flip(fwd, t, losing):
    """Winners' change in forward speed minus losers', after vs before frame t (m/s)."""
    def change(series):
        return np.nanmean(series[t + AFTER[0]:t + AFTER[1]]) - np.nanmean(series[max(t + BEFORE[0], 0):t + BEFORE[1]])
    return change(fwd[1 - losing]) - change(fwd[losing])


def calm_threshold(labels, fwd, blocks, long_spell=10 * FPS, q=75):
    """The q-th percentile of the role flip at the middle of long possessions (nothing changing there)."""
    vals = []
    for _, first, last in blocks:
        seg = labels[first:last + 1]
        cuts = np.concatenate(([0], np.flatnonzero(seg[1:] != seg[:-1]) + 1, [len(seg)]))
        for i in range(len(cuts) - 1):
            a, b, k = cuts[i], cuts[i + 1], seg[cuts[i]]
            if k in "HA" and b - a >= long_spell:
                vals.append(role_flip(fwd, first + (a + b) // 2, 0 if k == "H" else 1))
    return float(np.nanpercentile(vals, q))


def vet_by_reaction(labels, fwd, clear, blocks, threshold, max_len=MAX_SPELL, need=CONFIRM):
    """(labels, number of spells removed). See the module docstring."""
    out = labels.copy()
    removed = 0
    for _, first, last in blocks:
        seg = out[first:last + 1]
        cuts = np.concatenate(([0], np.flatnonzero(seg[1:] != seg[:-1]) + 1, [len(seg)]))
        for i in range(1, len(cuts) - 2):
            a, b = cuts[i], cuts[i + 1]
            k, before, after = seg[a], seg[a - 1], seg[b]
            if k not in "HA" or before != after or before not in "HA" or before == k or b - a >= max_len:
                continue
            flip = role_flip(fwd, first + a, 0 if before == "H" else 1)
            if not (flip > threshold) and clear[k][first + a:first + b].sum() < need:
                seg[a:b] = before
                removed += 1
    return out, removed


RETIME_WINDOW = 5 * FPS   # look this far either side of a blackout change for the swap


def retime_blackout_changes(labels, fwd, seen, blocks, threshold, window=RETIME_WINDOW):
    """(labels, number moved). A change of team made while the ball is missing was placed from team shape,
    which reacts late; move it to where the role swap peaks (within +-window), if that swap beats calm play
    (threshold), never before the ball was last seen nor across the neighbouring changes. Changes with
    the ball seen keep their place: there the swap peaks on the change itself (median offset 0 s)."""
    out = labels.copy()
    moved = 0
    for _, first, last in blocks:
        seg = out[first:last + 1]
        cuts = np.concatenate(([0], np.flatnonzero(seg[1:] != seg[:-1]) + 1, [len(seg)]))
        for i in range(1, len(cuts) - 1):
            t = cuts[i]
            old, new = seg[t - 1], seg[t]
            if old not in "HA" or new not in "HA" or seen[first + t]:
                continue
            last_seen = np.flatnonzero(seen[first + cuts[i - 1]:first + t])
            lo = cuts[i - 1] + (last_seen[-1] + 1 if len(last_seen) else 1)
            hi = cuts[i + 1] - 1
            cand = np.arange(max(t - window, lo), min(t + window, hi) + 1)
            if not len(cand):
                continue
            losing = 0 if old == "H" else 1
            flips = np.array([role_flip(fwd, first + c, losing) for c in cand])
            if not np.isfinite(flips).any() or np.nanmax(flips) <= threshold:
                continue
            k = int(cand[int(np.nanargmax(flips))])
            if k != t:
                seg[min(k, t):max(k, t)] = new if k < t else old
                moved += 1
    return out, moved


def _reaction_self_test():
    n = 200
    pos = np.zeros((n, 4, 2)); vel = np.zeros((n, 4, 2))
    team = np.array([0, 0, 1, 1])
    pos[:, 0, 1], pos[:, 1, 1] = 20.0, 40.0                  # home 20 m apart ...
    pos[100:, 0, 1], pos[100:, 1, 1] = 10.0, 50.0            # ... then 40 m: it has spread out
    vel[100:, 2:, 0] = -2.0                                  # away start moving towards x = 0, the goal they attack
    right = np.ones(n, bool)
    r = {s: reaction_series(pos, vel, team, right, s) for s in (0, 1)}
    assert r[0][150] - r[0][50] == REACTION_WEIGHTS["width"] * 10      # spread (std) 10 -> 20
    assert r[1][150] - r[1][50] == 2.0


def _self_test():
    _reaction_self_test()
    n = 600
    fwd = {0: np.where(np.arange(n) < 200, 1.0, -1.0), 1: np.where(np.arange(n) < 200, -1.0, 1.0)}  # swap at 200
    lab = np.array(["H"] * 270 + ["A"] * 330)          # the change was placed 70 frames late, ball missing
    seen = np.zeros(n, bool); seen[:150] = True
    out, k = retime_blackout_changes(lab, fwd, seen, [(1, 0, n - 1)], threshold=0.5, window=125)
    assert k == 1 and 188 <= np.flatnonzero(out == "A")[0] <= 200, np.flatnonzero(out == "A")[:1]   # flat peak: 0.5 s gap before the after-window
    seen[:240] = True                                   # ball seen until 240, after the swap: no swap left to find
    out, _ = retime_blackout_changes(lab, fwd, seen, [(1, 0, n - 1)], threshold=0.5, window=125)
    assert np.flatnonzero(out == "A")[0] == 270, "never moved before the last seen ball"
    seen[:] = True                                      # ball seen at the change: leave it
    out, k = retime_blackout_changes(lab, fwd, seen, [(1, 0, n - 1)], threshold=0.5)
    assert k == 0 and (out == lab).all()
    n = 400
    calm = {0: np.full(n, 1.0), 1: np.full(n, -1.0)}
    lab = np.array(["H"] * 100 + ["A"] * 50 + ["H"] * 250)
    clear = {"H": np.zeros(n, bool), "A": np.zeros(n, bool)}
    out, k = vet_by_reaction(lab, calm, clear, [(1, 0, n - 1)], threshold=0.5)
    assert k == 1 and set(out) == {"H"}, "no reaction and no control: a false alarm"
    swap = {0: np.where(np.arange(n) < 100, 1.0, -1.0), 1: np.where(np.arange(n) < 100, -1.0, 1.0)}
    out, k = vet_by_reaction(lab, swap, clear, [(1, 0, n - 1)], threshold=0.5)
    assert k == 0 and (out == lab).all(), "the teams swapped roles: keep it"
    clear["A"][110:120] = True
    out, k = vet_by_reaction(lab, calm, clear, [(1, 0, n - 1)], threshold=0.5)
    assert k == 0, "clear control by the new team: keep it"
    assert abs(role_flip(swap, 100, 0) - 4.0) < 1e-9


if __name__ == "__main__":
    _self_test()
    print("role_swap self-test passed")
