"""Step 1: Smart Tagging -> one gap-free interval table over the graded frames.

Each row is a span of video frames with ONE state (H / A / D / unknown), the reason for it (kind),
how much to trust it (strength), whether each edge is believable (firm / soft) and the window in
which ATD should look for the real boundary (search_start / search_end).

Uses no ATD and no ground truth. Tag windows are treated as "something happened near here", not as
exact boundaries: step 2 trims the soft edges using tracking.

    python tag_module.py                         # build, validate, print a summary
    python tag_module.py --out out/tag_intervals.csv   # also write the table (do not commit)

Place in the repo root next to games.py and tag_diagnostics.py.

Precedence when tags overlap (highest first):
    pre_kickoff > goal > restart (set piece) > transition > possession > context-only > untagged
Two tags of the same class naming opposite sides -> state 'unknown', kind 'conflict'.
'Kick Off (Start)' is context only (the ball is live from the first graded frame).
'Defending in X third' tags are exact mirrors of possession tags (verified: 118 of 118 identical
windows, opposite team), so they are not counted as evidence; they only set a 'mirrored' note.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import games  # noqa: E402
from tag_diagnostics import (BLOCKS, FPS, KICKOFF_FRAMES, MIRRORS, fmt_clock,  # noqa: E402
                             load_events, other, prepare)

# ---- parameters ---------------------------------------------------------------------------------
MARGIN_S = 6.0            # default ATD search margin for a SOFT edge (seconds, each side).
# Per-kind (backward at the START, forward at the END), in seconds. GUESSES from ~6 tags read by
# hand in the first 5 minutes: possession tags start 11-29 s AFTER control really began, and end
# within ~4 s of it (one counter-example: 16 s late). Replace with ATD-measured values in step 2.
MARGINS_S = {"possession": (30.0, 3.0), "transition": (6.0, 3.0)}
MERGE_GAP_FRAMES = 2      # untagged gaps this short are rounding artefacts: absorbed, not 'untagged'
GOAL_KICKOFF_TOL_S = 5.0  # a kickoff window starting within this of a goal-tag end belongs to it
MIRROR_UPGRADES = False   # True: a possession tag with a mirrored 'Defending' twin becomes 'strong'.
                          # Off by default: the twin is the same detection, not independent evidence.

PRIORITY = {"pre_kickoff": 6, "goal": 5, "restart": 4, "transition": 3, "possession": 2}
STRENGTH_RANK = {"weak": 0, "medium": 1, "strong": 2, "certain": 3}
POSSESSION_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES")
CONTEXT_CODES = ("LONG BALL", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX")


def side_of(x):
    """'H' / 'A' if x is one, else None (guards against NaN / pd.NA)."""
    return x if isinstance(x, str) and x in ("H", "A") else None


# ---- tags -> evidence items -------------------------------------------------------------------
def _item(start, end, state, cls, kind, team=None, strength="medium", start_firm=False,
          end_firm=False, src=(), note="", label=""):
    return {"start": int(start), "end": int(end), "state": state, "cls": cls, "kind": kind,
            "team": team, "strength": strength, "start_firm": start_firm, "end_firm": end_firm,
            "src": list(src), "note": note, "label": label}


def build_items(df):
    items = []

    # 1. pre-kickoff dead time: a published fact (FAQ/DATA), not a tag.
    for half, first, _ in BLOCKS:
        ko = KICKOFF_FRAMES.get("1st Half" if half == 1 else "2nd Half")
        if ko and ko > first:
            items.append(_item(first, ko, "D", "pre_kickoff", "pre_kickoff", strength="certain",
                               start_firm=True, end_firm=True,
                               note="published: graded D before kickoff"))

    # 2. goals: dead through the VAR check / celebration to the restart. Only the END is trusted.
    goals = df[df.code_u == "GOALS"]
    for r in goals.itertuples():
        scorer = side_of(r.ha)
        items.append(_item(r.start_frame, r.end_frame, "D", "goal", "goal", team=other(scorer),
                           strength="strong", start_firm=False, end_firm=True, src=[r.code_id],
                           note="restart side = conceding team (assumes Team = scorer)"))

    # 3. restarts (set pieces). Fixed-length windows: both edges soft.
    for r in df[df.code_u == "SET PIECES"].itertuples():
        if r.type_l == "kick off (start)":
            # The ground truth already shows a possession at the first graded frame, so this window
            # is not dead time. Kept as context only; the 2nd-half dead time is the pre_kickoff rule.
            items.append(_item(r.start_frame, r.end_frame, None, "context", "context",
                               src=[r.code_id], label="KICKOFF_START|-"))
            continue
        team, note = side_of(r.ha), "fixed-length window"
        if r.type_l == "kick off (after goal)":
            near = goals[(goals.end_frame - r.start_frame).abs() <= GOAL_KICKOFF_TOL_S * FPS]
            if len(near):
                team, note = other(side_of(near.iloc[0].ha)), "restart side from goal rule"
        items.append(_item(r.start_frame, r.end_frame, "D", "restart", "restart:%s" % r.Type,
                           team=team, strength="strong", src=[r.code_id], note=note))

    # 4. transitions: Attacking(X) and Defensive(opponent of X) are the same event -> one item.
    seen = {}
    for r in df[df.code_u.isin(["ATTACKING TRANSITION", "DEFENSIVE TRANSITION"])].itertuples():
        side = side_of(r.ha)
        state = side if r.code_u == "ATTACKING TRANSITION" else other(side)
        if state is None:
            continue
        key = (r.start_frame, r.end_frame, state)
        if key in seen:
            seen[key]["src"].append(r.code_id)
        else:
            seen[key] = _item(r.start_frame, r.end_frame, state, "transition", "transition",
                              strength="medium", src=[r.code_id],
                              note="change to %s near window start" % state)
    items += list(seen.values())

    # 5. possession tags, with the mirror flag.
    mirror = {a: {(s, e): side_of(h) for s, e, h in
                  zip(df[df.code_u == b].start_frame, df[df.code_u == b].end_frame, df[df.code_u == b].ha)}
              for a, b in MIRRORS}
    for r in df[df.code_u.isin(POSSESSION_CODES)].itertuples():
        side = side_of(r.ha)
        if side is None:
            continue
        twin = mirror[r.code_u].get((r.start_frame, r.end_frame))
        mirrored = twin is not None and twin != side
        note = ["start late: control likely began earlier; brief opposite spells possible"]
        if mirrored:
            note.append("mirrored")
        if r.code_u == "CREATING CHANCES":
            note.append("fixed 30s window")
        items.append(_item(r.start_frame, r.end_frame, side, "possession", "possession",
                           strength="strong" if (mirrored and MIRROR_UPGRADES) else "medium",
                           src=[r.code_id], note=";".join(note)))

    # 6. context tags: carried alongside, never decide a state.
    for r in df[df.code_u.isin(CONTEXT_CODES)].itertuples():
        items.append(_item(r.start_frame, r.end_frame, None, "context", "context",
                           src=[r.code_id], label="%s|%s" % (r.code_u, side_of(r.ha) or "-")))
    return items


# ---- items -> gap-free table ------------------------------------------------------------------
def _resolve(act, a, b, first, end_b):
    """One row for the elementary interval [a, b) given its active items."""
    deciders = [i for i in act if i["cls"] != "context"]
    ctx = sorted({i["label"] for i in act if i["cls"] == "context"})
    row = {"start_frame": a, "end_frame": b, "context": ";".join(ctx)}
    if not deciders:
        row.update(state="unknown", kind="context_only" if ctx else "untagged", restart_team="-",
                   strength="weak", start_edge="soft", end_edge="soft", sources="", notes="")
        return row

    top = max(PRIORITY[i["cls"]] for i in deciders)
    win = [i for i in deciders if PRIORITY[i["cls"]] == top]
    states = {i["state"] for i in win}
    if len(states) > 1:
        state, kind, strength = "unknown", "conflict", "weak"
    else:
        state = next(iter(states))
        kinds = {i["kind"] for i in win}
        kind = kinds.pop() if len(kinds) == 1 else win[0]["cls"]
        strength = max((i["strength"] for i in win), key=STRENGTH_RANK.get)
    if win[0]["cls"] in ("pre_kickoff", "goal", "restart"):
        teams = {i["team"] for i in win if i["team"] in ("H", "A")}
        restart_team = next(iter(teams)) if len(teams) == 1 else "unknown"
    else:
        restart_team = "-"
    row.update(
        state=state, kind=kind, restart_team=restart_team, strength=strength,
        start_edge="firm" if a == first or any(i["start"] == a and i["start_firm"] for i in win) else "soft",
        end_edge="firm" if b == end_b or any(i["end"] == b and i["end_firm"] for i in win) else "soft",
        sources=",".join(str(x) for x in sorted({s for i in win for s in i["src"]})),
        notes=";".join(sorted({i["note"] for i in win if i["note"]})))
    return row


def build_table(df, margin_s=MARGIN_S, merge_gap=MERGE_GAP_FRAMES, margins=None):
    margins = MARGINS_S if margins is None else margins
    items = build_items(df)
    rows = []
    for half, first, last in BLOCKS:
        end_b = last + 1
        its = [i for i in items if i["end"] > first and i["start"] < end_b]
        cuts = {first, end_b}
        for i in its:
            cuts.update(c for c in (i["start"], i["end"]) if first < c < end_b)
        cuts = sorted(cuts)
        block = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            act = [i for i in its if i["start"] <= a < i["end"]]
            block.append(dict(_resolve(act, a, b, first, end_b), half=half))
        # absorb rounding-sized untagged gaps into the previous row
        cleaned = []
        for r in block:
            if cleaned and r["kind"] == "untagged" and r["end_frame"] - r["start_frame"] <= merge_gap:
                cleaned[-1]["end_frame"] = r["end_frame"]
                cleaned[-1]["notes"] = ";".join(filter(None, [cleaned[-1]["notes"], "absorbed_gap"]))
            else:
                cleaned.append(r)
        # merge neighbours with identical decision
        merged = []
        for r in cleaned:
            p = merged[-1] if merged else None
            if p and all(p[k] == r[k] for k in ("state", "kind", "restart_team", "strength", "sources")):
                p["end_frame"], p["end_edge"] = r["end_frame"], r["end_edge"]
                p["context"] = ";".join(sorted(set(filter(None, (p["context"] + ";" + r["context"]).split(";")))))
                p["notes"] = ";".join(sorted(set(filter(None, (p["notes"] + ";" + r["notes"]).split(";")))))
            else:
                merged.append(r)
        rows += merged

    t = pd.DataFrame(rows)
    t.insert(0, "event_id", np.arange(len(t)))
    lim = {h: (f, l + 1) for h, f, l in BLOCKS}
    m = [margins.get(k.split(":")[0], (margin_s, margin_s)) for k in t.kind]
    t["start_margin_s"] = [a if e == "soft" else 0.0 for (a, _), e in zip(m, t.start_edge)]
    t["end_margin_s"] = [b if e == "soft" else 0.0 for (_, b), e in zip(m, t.end_edge)]
    t["search_start"] = [max(lim[h][0], s - int(x * FPS)) for h, s, x in zip(t.half, t.start_frame, t.start_margin_s)]
    t["search_end"] = [min(lim[h][1], s + int(x * FPS)) for h, s, x in zip(t.half, t.end_frame, t.end_margin_s)]
    t["start_clock"] = t.start_frame.map(fmt_clock)
    t["end_clock"] = t.end_frame.map(fmt_clock)
    t["dur_s"] = ((t.end_frame - t.start_frame) / float(FPS)).round(1)
    cols = ["event_id", "half", "start_frame", "end_frame", "start_clock", "end_clock", "dur_s", "state",
            "kind", "restart_team", "strength", "start_edge", "end_edge", "start_margin_s", "end_margin_s",
            "search_start", "search_end",
            "sources", "context", "notes"]
    return t[cols]


# ---- checks and helpers -----------------------------------------------------------------------
def validate(t):
    """The table must partition both graded blocks exactly: no gap, no overlap, no empty row."""
    for half, first, last in BLOCKS:
        h = t[t.half == half]
        assert len(h), "no rows for half %s" % half
        assert h.start_frame.iloc[0] == first, "half %s does not start at %d" % (half, first)
        assert h.end_frame.iloc[-1] == last + 1, "half %s does not end at %d" % (half, last + 1)
        assert (h.start_frame.values[1:] == h.end_frame.values[:-1]).all(), "gap/overlap in half %s" % half
        assert (h.end_frame > h.start_frame).all(), "empty row in half %s" % half
    assert set(t.state) <= {"H", "A", "D", "unknown"}
    return True


def frame_states(t, n_frames=None, outside="X"):
    """Per-frame array of 'H'/'A'/'D'/'U' (U = unknown), `outside` elsewhere. For the decoder/baselines."""
    n = n_frames or games.GAMES[games.GAME]["n_video_frames"]
    out = np.full(n, outside, dtype="<U1")
    for s, e, st in zip(t.start_frame, t.end_frame, t.state):
        out[s:e] = "U" if st == "unknown" else st
    return out


def summary(t):
    total = float((t.end_frame - t.start_frame).sum())
    t = t.assign(frames=t.end_frame - t.start_frame)
    print("%d intervals covering %d graded frames" % (len(t), total))
    for col in ("state", "kind", "strength"):
        g = t.groupby(col).agg(rows=("frames", "size"), frames=("frames", "sum"))
        g["share_%"] = (100.0 * g.frames / total).round(1)
        print("\nby %s:\n%s" % (col, g.to_string()))
    print("\nrestart_team on D rows:\n%s" % t[t.state == "D"].groupby("restart_team").size().to_string())
    print("\nsoft start edges: %d / soft end edges: %d of %d" % ((t.start_edge == "soft").sum(),
                                                               (t.end_edge == "soft").sum(), len(t)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=MARGIN_S, help="default margin for kinds without their own entry in MARGINS_S")
    ap.add_argument("--out", help="write the table here (local only; do not commit)")
    args = ap.parse_args()
    table = build_table(prepare(load_events()), margin_s=args.margin)
    validate(table)
    summary(table)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        table.to_csv(args.out, index=False)
        print("\nwrote %s" % args.out)