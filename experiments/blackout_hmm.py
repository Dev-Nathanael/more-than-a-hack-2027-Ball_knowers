"""Experiment 3: decide possession and dead balls during ball blackouts from the players.

The ball is missing in about 150 long blackouts, most over 5 s and some over a minute, but the
players are tracked throughout. Three pieces, none tuned on the public ground truth:

  1. Ownership from team shape. A logistic regression learns how the two teams' shape relates to
     who has the ball, trained only on frames where the ball is seen and a player stands within
     1.5 m of it -- the nearest player's team is the pseudo-label, produced by an algorithm, no
     human involved. It is then applied where the ball is missing.
  2. Dead ball from speed. A two-component Gaussian mixture splits the match's average player
     speed into a slow and a fast regime, unsupervised; slow is evidence of a stoppage.
  3. Decoding. A three-state H / A / D Viterbi with a cost for every change of state replaces the
     baseline's majority vote, so the boundary lands where the evidence turns, not half a window
     late.

Every constant below is set from general football knowledge before looking at any score.

    python experiments/blackout_hmm.py          # writes four variants, prints aggregate checks
"""

import argparse
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tracking import GAME, GAMES, ROOT, half_video_blocks, home_attacks_right, load, team_features
from baseline import HOME, write_submission

# Params
TAU = 25            # frames per independent observation: evidence is counted about once a second
SPEED_SMOOTH = 25   # +/- frames of the moving mean applied to player speed before the dead/alive split
NEAR_M = 1.5        # a player this close to a seen ball has it, for the pseudo-labels
P_BALL = 0.8        # how often the nearest player's team is the owner when the ball is seen
P_SWITCH = 1 / 300  # per live frame: possessions last about 12 s
P_STOP = 1 / 780    # per live frame: about 110 stoppages in about 57 live minutes
P_RESTART = 1 / 500  # per dead frame: stoppages last about 20 s
MIN_HOLD = 25       # frames a team must keep the ball before it can lose it to the other team (1 s)
BLACKOUT_SCALE = 1 / 25   # team shape changes slowly: 25 blackout frames are one observation, not 25
BLOCK = 1500        # 60 s blocks, alternated for the train / test split of the shape model
H_, A_, D_ = 0, 1, 2
STATE_LABELS = np.array(["H", "A", "D"])


def nearest_team(players, ball, team):
    """(team of the nearest player, distance in metres); team -1 where there is no ball or no one."""
    d = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    seen = ~np.all(np.isnan(d), axis=1)
    idx = np.zeros(len(d), dtype=int)
    idx[seen] = np.nanargmin(d[seen], axis=1)
    owner = np.where(seen, team[idx], -1)
    return owner, np.where(seen, d[np.arange(len(d)), idx], np.nan)


def shape_matrix(feat, blocks, pitch_length):
    """Direction-invariant shape features per frame: home minus away, or along home's attack."""
    right = home_attacks_right(feat, blocks)
    direction = np.zeros(len(feat["cx_all"]))
    for half, first, last in blocks:
        direction[first:last + 1] = 1.0 if right[half] else -1.0
    return np.column_stack([
        feat["len_H"] - feat["len_A"],
        feat["wid_H"] - feat["wid_A"],
        feat["speed_H"] - feat["speed_A"],
        (feat["cx_all"] - pitch_length / 2) * direction,  # territory: where the play is
        (feat["cx_H"] - feat["cx_A"]) * direction,        # how far home's block sits ahead
    ])


def fit_shape_model(X, owner, dist, graded):
    """Logistic regression of owner on shape, on pseudo-labelled frames; (model, test accuracy, base rate)."""
    usable = graded & (owner >= 0) & (dist <= NEAR_M) & np.all(np.isfinite(X), axis=1)
    test_block = (np.arange(len(X)) // BLOCK) % 2 == 1
    train, test = usable & ~test_block, usable & test_block
    y = (owner == HOME).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X[train], y[train])
    accuracy = float(np.mean(model.predict(X[test]) == y[test]))
    base_rate = float(max(np.mean(y[test]), 1 - np.mean(y[test])))
    return model, accuracy, base_rate, int(train.sum()), int(test.sum())


def dead_evidence(speed_all, graded):
    """Per-frame log-likelihood ratio dead vs alive from a 2-regime split of smoothed mean speed."""
    s = np.where(np.isfinite(speed_all), speed_all, np.nan)
    kernel = np.ones(2 * SPEED_SMOOTH + 1)
    valid = np.isfinite(s).astype(float)
    total = np.convolve(np.nan_to_num(s), kernel, mode="same")
    count = np.convolve(valid, kernel, mode="same")
    smooth = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    use = graded & np.isfinite(smooth) & (smooth > 0)
    gmm = GaussianMixture(2, random_state=0).fit(np.log(smooth[use])[:, None])
    slow = int(np.argmin(gmm.means_.ravel()))
    llr = np.zeros(len(s))
    x = np.log(smooth[use])
    means, sds = gmm.means_.ravel(), np.sqrt(gmm.covariances_.ravel())
    logpdf = -0.5 * ((x[:, None] - means) / sds) ** 2 - np.log(sds)
    llr[use] = logpdf[:, slow] - logpdf[:, 1 - slow]
    regimes = {"slow_mps": float(np.exp(means[slow])), "fast_mps": float(np.exp(means[1 - slow])),
               "slow_weight": float(gmm.weights_[slow])}
    return llr, regimes


def viterbi(emission, trans, init):
    """Most likely state path. emission (T, S), init (S,) and trans are log scores; trans is either
    one (S, S) matrix or one per frame, (T, S, S), entering frame t."""
    n, k = emission.shape
    per_frame = trans.ndim == 3
    back = np.zeros((n, k), dtype=np.int8)
    delta = init + emission[0]
    for t in range(1, n):
        cand = delta[:, None] + (trans[t] if per_frame else trans)
        back[t] = np.argmax(cand, axis=0)
        delta = cand[back[t], np.arange(k)] + emission[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax(delta))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def transitions():
    stay_live = np.log(1 - P_SWITCH - P_STOP)
    return np.array([
        [stay_live, np.log(P_SWITCH), np.log(P_STOP)],
        [np.log(P_SWITCH), stay_live, np.log(P_STOP)],
        [np.log(P_RESTART / 2), np.log(P_RESTART / 2), np.log(1 - P_RESTART)],
    ])


def transitions_min_hold(m):
    """Log transitions of the chain with a minimum possession length of m frames.

    States: H aged 0..m-1 (the last is "settled"), then A aged 0..m-1, then D. A young possession can
    only age or be stopped; only a settled one can pass to the other team. Returns (trans, init,
    label index of each state).
    """
    k, no = 2 * m + 1, -1e18
    trans = np.full((k, k), no)
    for base, other in ((0, m), (m, 0)):
        for i in range(m - 1):
            trans[base + i, base + i + 1] = np.log(1 - P_STOP)
            trans[base + i, k - 1] = np.log(P_STOP)
        s = base + m - 1
        trans[s, s] = np.log(1 - P_SWITCH - P_STOP)
        trans[s, other] = np.log(P_SWITCH)
        trans[s, k - 1] = np.log(P_STOP)
    trans[k - 1, 0] = trans[k - 1, m] = np.log(P_RESTART / 2)
    trans[k - 1, k - 1] = np.log(1 - P_RESTART)
    init = np.full(k, no)
    init[[m - 1, 2 * m - 1, k - 1]] = np.log(1 / 3)     # a half may start mid-possession
    return trans, init, np.array([H_] * m + [A_] * m + [D_])


def transitions_per_frame(switch_mult):
    """(T, 3, 3) log transitions where the H <-> A switch probability is scaled frame by frame."""
    p = np.clip(P_SWITCH * np.asarray(switch_mult, dtype=float), 1e-9, 0.5 - P_STOP)
    out = np.broadcast_to(transitions(), (len(p), 3, 3)).copy()
    out[:, 0, 1] = out[:, 1, 0] = np.log(p)
    out[:, 0, 0] = out[:, 1, 1] = np.log(1 - p - P_STOP)
    return out


def _emission(team_llr, dead_llr, first, last, use_dead, tau, live_llr):
    """Log scores of H, A, D for frames first..last, shared by decode() and posteriors()."""
    t = team_llr[first:last + 1] / tau
    d = dead_llr[first:last + 1] / tau if use_dead else np.full(len(t), -1e9)
    live = np.abs(t) if live_llr is None else live_llr[first:last + 1] / tau
    return np.column_stack([(live + t - np.abs(t)) / 2, (live - t - np.abs(t)) / 2, d])


def _lse(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def forward_backward(emission, trans, init):
    """(T, S) probability of each state at each frame given ALL the evidence, before and after it.

    Same chain as viterbi(): where Viterbi picks the single best path, this says how sure that pick
    is frame by frame, so a frame can be called "sure" (e.g. above 66%) or left for review."""
    n, k = emission.shape
    per_frame = trans.ndim == 3   # (T, S, S): the transition entering frame t, as in viterbi()
    fwd, bwd = np.empty((n, k)), np.zeros((n, k))
    fwd[0] = init + emission[0]
    for t in range(1, n):
        fwd[t] = _lse(fwd[t - 1][:, None] + (trans[t] if per_frame else trans), 0) + emission[t]
    for t in range(n - 2, -1, -1):
        bwd[t] = _lse((trans[t + 1] if per_frame else trans) + (emission[t + 1] + bwd[t + 1])[None, :], 1)
    post = fwd + bwd
    post = np.exp(post - post.max(axis=1, keepdims=True))
    return post / post.sum(axis=1, keepdims=True)


def posteriors(team_llr, dead_llr, blocks, n_frames, use_dead, tau=TAU, live_llr=None, switch_mult=None):
    """(n_frames, 3) P(H), P(A), P(D) per frame from decode()'s chain; NaN outside the halves.
    switch_mult scales the H <-> A switch chance frame by frame, as in decode()."""
    out = np.full((n_frames, 3), np.nan)
    trans, init = transitions(), np.log(np.full(3, 1 / 3))
    for _, first, last in blocks:
        tr = trans if switch_mult is None else transitions_per_frame(switch_mult[first:last + 1])
        out[first:last + 1] = forward_backward(_emission(team_llr, dead_llr, first, last, use_dead, tau, live_llr), tr, init)
    return out


def decode(team_llr, dead_llr, blocks, n_frames, use_dead, tau=TAU, switch_mult=None, min_hold=0, live_llr=None):
    """Labels for the whole video: Viterbi inside each half, X outside.

    switch_mult, if given, scales the chance of a change of team frame by frame (e.g. higher when
    the ball carrier is under pressure). min_hold > 1 makes every possession last at least that many
    frames before the other team can take over (not combined with switch_mult). live_llr, if given, is
    the strength of "play is live" against dead; by default it is |team_llr|, so damping team_llr alone
    would also make dead ball win more often."""
    if min_hold > 1 and switch_mult is not None:
        raise ValueError("min_hold and switch_mult cannot be combined")
    labels = np.full(n_frames, "X", dtype="<U1")
    trans, init = transitions(), np.log(np.full(3, 1 / 3))
    for _, first, last in blocks:
        emission = _emission(team_llr, dead_llr, first, last, use_dead, tau, live_llr)
        if min_hold > 1:
            tr, start, state_label = transitions_min_hold(min_hold)
            labels[first:last + 1] = STATE_LABELS[state_label[viterbi(emission[:, state_label], tr, start)]]
            continue
        tr = trans if switch_mult is None else transitions_per_frame(switch_mult[first:last + 1])
        labels[first:last + 1] = STATE_LABELS[viterbi(emission, tr, init)]
    return labels


def _self_test():
    trans = transitions()
    strong = np.log(0.99) - np.log(0.01)
    e = np.zeros((600, 3))
    e[:300, H_], e[300:, A_] = strong, strong
    e[:, D_] = -1e9
    path = viterbi(e, trans, np.zeros(3))
    assert path[:300].tolist() == [H_] * 300 and path[300:].tolist() == [A_] * 300
    flicker = np.zeros((600, 3))
    flicker[:, H_] = 0.5 / TAU
    flicker[290:300, A_] = strong / TAU  # a 10-frame blip against a long home spell
    flicker[:, D_] = -1e9
    assert set(viterbi(flicker, trans, np.zeros(3)).tolist()) == {H_}, "a short blip must not switch"
    # per-frame transitions equal to the fixed ones give the same path; a much cheaper switch lets the blip through
    assert viterbi(flicker, transitions_per_frame(np.ones(600)), np.zeros(3)).tolist() == viterbi(flicker, trans, np.zeros(3)).tolist()
    assert A_ in viterbi(flicker, transitions_per_frame(np.full(600, 150.0)), np.zeros(3)).tolist()
    # minimum hold: a 10-frame away blip inside clear home control switches the plain chain; when a
    # possession must last 25 frames, taking it means giving up 15 frames of home evidence, so it does not
    llr = np.full(600, 4.0)
    llr[290:300] = -3.0
    plain = decode(llr, np.zeros(600), [(1, 0, 599)], 600, False, tau=1)
    held = decode(llr, np.zeros(600), [(1, 0, 599)], 600, False, tau=1, min_hold=25)
    assert "A" in plain and "A" not in held, "a blip shorter than the minimum hold must not switch"
    llr[290:330] = -3.0      # 40 frames of away control is a possession
    held = decode(llr, np.zeros(600), [(1, 0, 599)], 600, False, tau=1, min_hold=25)
    assert int(np.sum(held == "A")) == 40, int(np.sum(held == "A"))
    runs_a = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], (held == "A").astype(int), [0]]))))[::2]
    assert runs_a.min() >= 25
    # probabilities: rows sum to 1, a clear spell is sure, the frames at a change and inside a blip are not
    p = forward_backward(e, trans, np.zeros(3))
    assert np.allclose(p.sum(axis=1), 1) and p[:250, H_].min() > 0.99 and p[350:, A_].min() > 0.99
    assert np.array_equal(p.argmax(axis=1), viterbi(e, trans, np.zeros(3)))
    assert np.allclose(forward_backward(flicker, transitions_per_frame(np.ones(600)), np.zeros(3)),
                       forward_backward(flicker, trans, np.zeros(3))), "per-frame transitions equal to the fixed ones"
    q = forward_backward(flicker, trans, np.zeros(3))
    assert q[290:300, H_].max() < q[150:250, H_].min(), "the blip must lower how sure home is"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    parser.add_argument("--tau", type=float, nargs="+", default=[TAU],
                        help="frames per independent observation; several values write one set each")
    args = parser.parse_args()
    _self_test()

    players, ball, team = load()
    n_frames, pitch = len(players), GAMES[GAME]["pitch"]
    blocks = half_video_blocks(GAME)
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    feat = team_features(players, team)
    owner, dist = nearest_team(players, ball, team)
    X = shape_matrix(feat, blocks, pitch[0])
    model, accuracy, base_rate, n_train, n_test = fit_shape_model(X, owner, dist, graded)
    print("home attacks right by half: {}".format(home_attacks_right(feat, blocks)))
    print("shape model: {} train / {} test frames; held-out agreement with the nearest player "
          "{:.3f} (always guessing the larger side: {:.3f})".format(n_train, n_test, accuracy, base_rate))

    dead_llr, regimes = dead_evidence(feat["speed_all"], graded)
    print("speed regimes: slow {slow_mps:.2f} m/s, fast {fast_mps:.2f} m/s, slow share {slow_weight:.2f}"
          .format(**regimes))

    ball_llr = np.where(owner == HOME, 1.0, np.where(owner >= 0, -1.0, 0.0)) * np.log(P_BALL / (1 - P_BALL))
    finite = np.all(np.isfinite(X), axis=1)
    shape_llr = np.zeros(n_frames)
    shape_llr[finite] = np.clip(model.decision_function(X[finite]), -3, 3)
    blackout = owner < 0

    variants = {
        "hmm_carry": (ball_llr, False),
        "hmm_shape": (np.where(blackout, shape_llr, ball_llr), False),
        "hmm_dead": (ball_llr, True),
        "hmm_full": (np.where(blackout, shape_llr, ball_llr), True),
    }
    expected = 1 / (P_SWITCH + P_STOP) / 25
    print("the transition priors imply a mean possession spell of {:.1f} s".format(expected))
    for tau in args.tau:
        for name, (team_llr, use_dead) in variants.items():
            labels = decode(team_llr, dead_llr, blocks, n_frames, use_dead, tau)
            tag = "{}_tau{:g}".format(name, tau)
            write_submission(os.path.join(args.out, tag + ".csv"), labels)
            g = labels[graded]
            spells = sum(int(np.sum((s[1:] != s[:-1]) & (s[1:] != "D"))) + int(s[0] != "D")
                         for s in (labels[first:last + 1] for _, first, last in blocks))
            live = int(np.sum(g != "D"))
            print("{:<18} graded share H {:.3f} A {:.3f} D {:.3f}  spells {:>4}  mean {:5.1f} s".format(
                tag, np.mean(g == "H"), np.mean(g == "A"), np.mean(g == "D"), spells,
                live / max(spells, 1) / 25))
