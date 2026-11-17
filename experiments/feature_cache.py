"""Every feature from experiments/features.py for the whole match, built once from raw data and cached.

    from feature_cache import load_features
    feat = load_features()      # dict of per-frame arrays, rebuilt if the cache is missing

Used by the pipelines (solution 07 and the combination grid) and the viewer, so both see the same
numbers. Building takes about two minutes; the cache is match data under data/cache/.
"""

import functools
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tracking import (CACHE_DIR, GAME, GAMES, goalkeepers, half_video_blocks, home_attacks_right,  # noqa: E402
                      load, load_clean, team_features)
import features as F  # noqa: E402

CACHE = os.path.join(CACHE_DIR, "{}_features.npz".format(GAME))


@functools.lru_cache(maxsize=None)
def sync_offsets():
    """{half: offset} with tactical frame = broadcast frame + offset.

    Nothing is typed in: broadcast_sync.compute() finds the offsets from the two ball paths the first
    time they are needed and they are kept in the cache, so another match needs no manual step.
    broadcast_sync.verify() recomputes them and fails if the cached ones have gone stale."""
    path = os.path.join(CACHE_DIR, "{}_broadcast_offsets.json".format(GAME))
    if os.path.exists(path):
        with open(path) as handle:
            return {int(half): int(off) for half, off in json.load(handle).items()}
    from broadcast_sync import compute
    found = {int(half): int(off) for half, off in compute()[0].items()}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(found, handle)
    return found


from load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402


def __getattr__(name):
    """OFFSETS and KICKOFF2_BROADCAST are worked out the first time they are asked for, not when this file is
    imported: importing it reads no match data and writes nothing.
    KICKOFF2_BROADCAST: broadcast frames from here belong to the 2nd half (the organisers' kick-off frame, shifted)."""
    if name == "OFFSETS":
        return sync_offsets()
    if name == "KICKOFF2_BROADCAST":
        return KICKOFF_VIDEO_FRAMES["2nd Half"] - sync_offsets()[2]
    raise AttributeError(name)


LIVE_MATCH_M = 2.0                  # broadcast players within this of tactical ones = a live view


def broadcast_live(bplayers, bteam, tplayers, tteam):
    """Per broadcast frame: True where its visible players sit on tactical players of the same team."""
    nb = len(bplayers)
    live = np.zeros(nb, dtype=bool)
    tframe = np.full(nb, -1)
    b = np.arange(nb)
    offsets = sync_offsets()
    off = np.where(b < KICKOFF_VIDEO_FRAMES["2nd Half"] - offsets[2], offsets[1], offsets[2])
    t = b + off
    ok = (t >= 0) & (t < len(tplayers))
    tframe[ok] = t[ok]
    chunk = 2000
    for s in range(0, nb, chunk):
        idx = np.arange(s, min(s + chunk, nb))
        idx = idx[ok[idx]]
        if not len(idx):
            continue
        misses = []
        for side in (0, 1):
            bp = bplayers[idx][:, bteam == side, :2]            # (n, Sb, 2)
            tp = tplayers[tframe[idx]][:, tteam == side, :2]     # (n, St, 2)
            d = np.hypot(bp[:, :, None, 0] - tp[:, None, :, 0], bp[:, :, None, 1] - tp[:, None, :, 1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                misses.append(np.nanmin(np.where(np.isnan(d), np.inf, d), axis=2))  # (n, Sb)
        m = np.concatenate(misses, axis=1)
        m[~np.isfinite(m)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(m, axis=1)
        enough = np.sum(np.isfinite(m), axis=1) >= 3
        live[idx] = enough & (med < LIVE_MATCH_M)
    return live, tframe


def turn_broadcast(bp, bb, flipped):
    """The broadcast turned the way cleaning.orient turned the tactical pitch, half by half."""
    if not flipped or not any(flipped.values()):
        return bp, bb
    from cleaning import rotate
    pitch = GAMES[GAME]["pitch"]
    kickoff2 = KICKOFF_VIDEO_FRAMES["2nd Half"] - sync_offsets()[2]
    turn = np.isin(np.where(np.arange(len(bb)) < kickoff2, 1, 2), [h for h, f in flipped.items() if f])
    bp = bp.copy()
    bp[..., :2] = rotate(bp[..., :2], turn, pitch)
    return bp, rotate(bb, turn, pitch)


def build_features(tp, tb, tteam, bp, bb, bteam, gk, flipped=None):
    """Every feature from arrays. flipped: {half: True} where cleaning turned the tactical pitch
    (cleaning.orient); the broadcast is turned the same way so the two still line up."""
    blocks = half_video_blocks(GAME)
    length, width = GAMES[GAME]["pitch"]
    bp, bb = turn_broadcast(bp, bb, flipped)
    blive, btframe = broadcast_live(bp, bteam, tp, tteam)
    vel, speed, heading, acc = F.ball_motion(tb)
    pred, pred_src, pred_r = F.predict_ball(tb, vel, tp, (length, width), blocks)
    headings = F.player_headings(tp, pred)
    right = home_attacks_right(team_features(tp, tteam), blocks)
    attacks_right = np.where(np.arange(len(tb)) < (blocks[0][2] + blocks[1][1]) // 2, right[1], right[2])
    bball_on_t = np.full_like(tb, np.nan)
    ok = (btframe >= 0) & blive
    bball_on_t[btframe[ok]] = bb[ok]
    out = {"ball_vel": vel, "ball_speed": speed, "ball_heading": heading.astype(np.float32), "ball_acc": acc,
           "touch": F.touches(tb, tp, tteam), "pred": pred, "pred_src": pred_src, "pred_r": pred_r,
           "heading": headings.astype(np.float16), "attacks_right": attacks_right,
           "air_gap": F.airborne_gap(tb, bball_on_t), "broadcast_live": blive, "broadcast_tframe": btframe}
    out.update({"press_" + k: v for k, v in F.pressure(tp, tteam, tb, speed, headings).items()})
    out.update({"shape_" + k: v for k, v in F.team_shape(tp, tteam, gk, attacks_right, length).items()})
    out.update({"move_" + k: v for k, v in F.team_movement(tp, tteam, gk, attacks_right, pred).items()})
    if F.RADIAL:
        out.update({"radial_" + k: v for k, v in F.radial_features(tp, tteam, gk, pred).items()})
    return out


def features_path(cleaning="C0"):
    if cleaning == "C0":
        return CACHE
    from cleaning import params_hash
    return os.path.join(CACHE_DIR, "{}_features_{}_{}.npz".format(GAME, cleaning, params_hash(cleaning)))


def load_features(cleaning="C0"):
    """Features of one cleaning version (C0 = the data every solution so far was built on)."""
    path = features_path(cleaning)
    if os.path.exists(path):
        return dict(np.load(path))
    from cleaning import V01, VERSIONS, feature_params, flipped_halves
    cfg = VERSIONS.get(cleaning, V01)       # C0 is the raw data: no turning, no feature overrides
    tp, tb, tteam = load_clean(cleaning)
    bp, bb, bteam = load("broadcast")
    flipped = flipped_halves(load("tactical")[0], tteam, half_video_blocks(GAME)) if cfg["orient"] else None
    with feature_params(cfg["feat"]):
        feat = build_features(tp, tb, tteam, bp, bb, bteam, goalkeepers("tactical"), flipped)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **feat)
    return feat
