"""The ATD as numpy arrays, plus the per-frame team features the experiments share.

Parsing 199319 lines of text takes a while, so the arrays are cached under data/cache/, which is
gitignored with the rest of data/ -- the cache is match data and must never be committed.
"""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import FIELD_SEP, GROUP_SEP, HOME, AWAY, team_of_each_track  # noqa: E402

# A run of experiments/iterate.py points this at its own folder, so it rebuilds everything from the
# raw text and leaves its copies there. It must stay inside data/, which git never tracks.
CACHE_DIR = os.path.realpath(os.environ.get("MTAH_CACHE_DIR") or os.path.join(DATA_DIR, "cache"))
if not CACHE_DIR.startswith(os.path.realpath(DATA_DIR) + os.sep):
    raise ValueError("MTAH_CACHE_DIR must be inside the data folder, which git ignores")


def cache_path(source):
    return os.path.join(CACHE_DIR, "{}_{}_atd.npz".format(GAME, source))


def goalkeepers(source="tactical"):
    """Boolean array over track slots: True where the metadata marks the slot as a goalkeeper."""
    import xml.etree.ElementTree as ET
    root = ET.parse(require(["atd_meta"], source=source)[0]).getroot()
    flags = {}
    for player in root.iter("Player"):
        for param in player.iter("ProviderParameter"):
            if param.findtext("Name") == "position_type":
                flags[player.get("id")] = param.findtext("Value") == "Goalkeeper"
    return np.array([flags.get(player.get("id"), False) for player in root.iter("Player")])


def load(source="tactical"):
    """(players (N, S, 3) x/y/speed, ball (N, 2) x/y, team (S,)) in metres and m/s, NaN = unseen.

    Row i is video frame i of that source. Positions are scaled from the 0-1 convention to metres.
    """
    team = team_of_each_track(require(["atd_meta"], source=source)[0])
    cache = cache_path(source)
    if os.path.exists(cache):
        cached = np.load(cache)
        return cached["players"], cached["ball"], team
    players, ball, team = parse(source)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, players=players, ball=ball)
    return players, ball, team


def parse(source="tactical"):
    """load() without the cache: read the raw ATD text every time (replicate.py starts from here)."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], source=source)
    team = team_of_each_track(meta_path)
    n_frames, n_slots = source_frames(source), len(team)
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    bad_lines = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                bad_lines += 1
                continue
            frame = int(parts[0]) - 1
            values = np.array(parts[1].replace(GROUP_SEP, ",").split(","), dtype=np.float32)
            if values.size != n_slots * 3 or not 0 <= frame < n_frames:
                bad_lines += 1
                continue
            players[frame] = values.reshape(n_slots, 3)
            ball[frame] = np.array(parts[2].split(",")[:2], dtype=np.float32)
    if bad_lines:
        print("warning: {} ATD lines did not match the expected layout".format(bad_lines))

    length, width = GAMES[GAME]["pitch"]
    players[:, :, 0] *= length
    players[:, :, 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    return players, ball, team


def load_clean(version="C0"):
    """Tactical tracking cleaned by one version of experiments/cleaning.py, cached per version.

    C0 is load() itself, so every existing solution keeps exactly the data it was built on. The
    cache name carries the version's parameter hash, so changing a parameter never reads stale data.
    """
    players, ball, team = load("tactical")
    if version == "C0":
        return players, ball, team
    from cleaning import clean, params_hash
    cache = os.path.join(CACHE_DIR, "{}_tactical_clean_{}_{}.npz".format(GAME, version, params_hash(version)))
    if os.path.exists(cache):
        cached = np.load(cache)
        return cached["players"], cached["ball"], team
    from cleaning import VERSIONS
    broadcast = load("broadcast") if any(VERSIONS[version].get(k) for k in ("broadcast", "teamcheck", "fuse")) else None
    if broadcast is not None:
        from broadcast_sync import verify
        verify()    # the sync offsets must be the ones the data gives, not just the ones written down
    players, ball, _ = clean(players, ball, team, half_video_blocks(GAME), GAMES[GAME]["pitch"], version, broadcast)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, players=players, ball=ball)
    return players, ball, team


def team_features(players, team):
    """Per-frame dict of team-level numbers. Every value is NaN where that team has no one tracked."""
    x, y, speed = players[:, :, 0], players[:, :, 1], players[:, :, 2]
    out = {}
    # an all-NaN row (nobody tracked) warns in nanmean; NaN is the right answer there
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for name, side in (("H", HOME), ("A", AWAY)):
            cols = team == side
            tx, ty = x[:, cols], y[:, cols]
            cx, cy = np.nanmean(tx, axis=1), np.nanmean(ty, axis=1)
            out["cx_" + name] = cx
            out["len_" + name] = np.nanstd(tx, axis=1)  # spread along the pitch
            out["wid_" + name] = np.nanstd(ty, axis=1)  # spread across it
            out["speed_" + name] = np.nanmean(speed[:, cols], axis=1)
            out["min_x_" + name] = np.nanmin(np.where(np.isnan(tx), np.inf, tx), axis=1)
        out["speed_all"] = np.nanmean(speed, axis=1)
        out["cx_all"] = np.nanmean(x, axis=1)
    return out


def home_attacks_right(features, blocks):
    """{half: True if the home team attacks towards x = pitch length}, from where each team's
    rearmost player stands on average. The side a goalkeeper guards is the side its team defends."""
    out = {}
    for half, first, last in blocks:
        rear_h = np.nanmean(_finite(features["min_x_H"][first:last + 1]))
        rear_a = np.nanmean(_finite(features["min_x_A"][first:last + 1]))
        out[half] = bool(rear_h < rear_a)
    return out


def _finite(values):
    return np.where(np.isfinite(values), values, np.nan)
