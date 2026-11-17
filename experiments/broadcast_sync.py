"""Find the time offset between the broadcast and the tactical recordings, from the tracking alone.

    python experiments/broadcast_sync.py

The broadcast tracking is on its own timeline. Both files give the ball in pitch coordinates, so
the right shift is the one where the two ball paths coincide. For each 5-minute chunk of the
broadcast, every possible shift is scored by the mean squared distance between the two balls over
the frames where both see it; the best shift of each chunk is its vote. The votes of a half agree to
within a few frames; its offset is the shift near their median with the smallest median distance
between the two balls over the whole half. Nothing is looked at or set by hand, and no labels are used.

feature_cache.sync_offsets() calls compute() the first time the offsets are needed and caches the
result; verify() recomputes them and fails if the cached ones differ. Prints timing offsets and counts only.

How well the offset is pinned down is printed too: the number of 5-minute chunks that voted, how far
their votes spread, and the median distance left between the two balls at the chosen offset.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Params
FPS = 25
CHUNK = 5 * 60 * FPS        # broadcast frames per vote
MIN_SHARED = 1500           # frames where both see the ball (1 min) for a shift to count
MAX_RMS_M = 5.0             # a chunk whose best shift still misses by more has no live ball path (replays)
REFINE = 10                 # frames either side of the median vote searched for the best whole-half fit


def best_shift(tb, chunk, start):
    """(offset, rms m, shared frames): the offset = tactical frame - broadcast frame that best lays
    this broadcast ball chunk (starting at broadcast frame `start`) on the tactical ball, or None."""
    from scipy.signal import correlate
    mt, mc = np.isfinite(tb[:, 0]), np.isfinite(chunk[:, 0])
    t, c = np.where(mt[:, None], tb, 0.0).astype(np.float64), np.where(mc[:, None], chunk, 0.0).astype(np.float64)
    mt, mc = mt.astype(np.float64), mc.astype(np.float64)
    corr = lambda a, b: correlate(a, b, mode="valid", method="fft")  # noqa: E731
    shared = corr(mt, mc)
    ssd = np.zeros_like(shared)
    for k in (0, 1):
        ssd += corr(t[:, k] ** 2, mc) - 2 * corr(t[:, k], c[:, k]) + corr(mt, c[:, k] ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rms = np.sqrt(np.maximum(ssd, 0) / shared)
    rms[np.round(shared) < MIN_SHARED] = np.inf
    lag = int(np.argmin(rms))
    if not np.isfinite(rms[lag]) or rms[lag] > MAX_RMS_M:
        return None
    return lag - start, float(rms[lag]), int(round(shared[lag]))


def offsets(tb, bb, blocks):
    """{half: offset} with tactical frame = broadcast frame + offset, and the votes behind each."""
    votes = {half: [] for half, _, _ in blocks}
    for start in range(0, len(bb) - CHUNK + 1, CHUNK):
        found = best_shift(tb, bb[start:start + CHUNK], start)
        if found is None:
            continue
        mid = start + CHUNK // 2 + found[0]
        for half, first, last in blocks:
            if first <= mid <= last:
                votes[half].append(found[0])
    out = {}
    for half, first, last in blocks:
        if not votes[half]:
            raise ValueError("half {}: no broadcast chunk lines up with the tactical ball".format(half))
        # chunk votes scatter by a few frames; settle on the shift with the smallest median gap
        # between the two balls over the whole half
        centre = int(np.median(votes[half]))
        out[half] = min(range(centre - REFINE, centre + REFINE + 1), key=lambda off: half_gap(tb, bb, off, first, last))
    return out, votes


def half_gap(tb, bb, off, first, last):
    """Median distance between the two balls over one half at this offset."""
    b = np.arange(max(0, first - off), min(len(bb), last - off + 1))
    d = np.hypot(*(bb[b] - tb[b + off]).T)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if len(d) else np.inf


def compute():
    from tracking import GAME, half_video_blocks, load
    _, tb, _ = load("tactical")
    _, bb, _ = load("broadcast")
    return offsets(tb, bb, half_video_blocks(GAME))


def verify():
    """Fail unless the offsets found from the data now are the cached ones feature_cache uses."""
    from feature_cache import OFFSETS
    found, votes = compute()
    if found != OFFSETS:
        raise ValueError("broadcast offsets from the data {} differ from feature_cache.OFFSETS {}".format(found, OFFSETS))
    return found, votes


def _self_test():
    rng = np.random.default_rng(0)
    n = 60000
    path = np.cumsum(rng.normal(0, 0.3, (n, 2)), axis=0)
    tb = path.copy()
    tb[rng.random(n) < 0.4] = np.nan
    bb = path[1234:1234 + 3 * CHUNK] + rng.normal(0, 0.3, (3 * CHUNK, 2))
    bb[rng.random(len(bb)) < 0.4] = np.nan
    found, votes = offsets(tb, bb, [(1, 0, n - 1)])
    assert found == {1: 1234} and votes[1] == [1234] * 3, (found, votes)
    assert best_shift(tb, rng.normal(0, 30, (CHUNK, 2)), 0) is None, "an unrelated path must not line up"


if __name__ == "__main__":
    _self_test()
    found, votes = verify()
    for half, v in votes.items():
        print("half {}: offset {} frames ({:.1f} s); {} chunk votes, spread {} to {}".format(
            half, found[half], found[half] / FPS, len(v), min(v), max(v)))
    from tracking import GAME, half_video_blocks, load
    tb, bb = load("tactical")[1], load("broadcast")[1]
    for half, first, last in half_video_blocks(GAME):
        gaps = {d: half_gap(tb, bb, found[half] + d, first, last) for d in (-2, -1, 0, 1, 2)}
        print("half {}: median distance between the two balls at the offset {:.2f} m; one frame off {:.2f} / {:.2f} m, two off {:.2f} / {:.2f} m".format(
            half, gaps[0], gaps[-1], gaps[1], gaps[-2], gaps[2]))
    print("matches the cached offsets")
