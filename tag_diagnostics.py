"""Smart Tagging diagnostics: sorting, duplicates, coverage, overlaps, polarity flips.

Place in the repo root (next to games.py). Reads ONLY the Smart Tagging file; prints aggregates
to the console. Nothing here touches tracking data or the ground truth.

    python tag_diagnostics.py                  # print diagnostics
    python tag_diagnostics.py --timeline out/tag_timeline.csv   # also write the elementary timeline

Do not commit the timeline CSV (add out/ to .gitignore) and do not paste it into hosted LLMs.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import games  # noqa: E402

FPS = games.GAMES[games.GAME]["fps"]
BLOCKS = games.half_video_blocks(games.GAME)          # [(half, first_frame, last_frame)] inclusive
KICKOFF_FRAMES = {"1st Half": 11185, "2nd Half": 113742}   # published in DATA.md / FAQ.md

# Club name (the Team column) -> H/A. NOT the file's "Side" column, which is the Left/Central/Right ball-entry lane. Note the file spells it "Atletic Club". Anything else (N/A) -> None.
TEAM_MAP = {"barcelona": "H", "atletic club": "A", "athletic club": "A"}

# Agreed taxonomy (role of each code in the possession logic).
ROLE = {
    "SET PIECES": "stoppage", "GOALS": "stoppage",
    "ATTACKING TRANSITION": "transition", "DEFENSIVE TRANSITION": "transition",
    "BUILD UP": "possession", "PROGRESSION": "possession", "CREATING CHANCES": "possession",
    "DEFENDING IN ATTACKING THIRD": "defending", "DEFENDING IN MIDDLE THIRD": "defending",
    "DEFENDING IN DEFENSIVE THIRD": "defending",
    "LONG BALL": "context", "BALL IN FINAL THIRD": "context", "BALL IN THE BOX": "context",
    "PLAYERS IN THE BOX": "context",
}
POLARITY_ROLES = ("possession", "transition")   # tags that say "side X has the ball"


def other(side):
    return {"H": "A", "A": "H"}.get(side)


# ---------------------------------------------------------------------------------- preparation
def load_events():
    from kloppy import sportscode
    return sportscode.load(games.asset_paths()["smart_tagging"]).to_df()


def prepare(events):
    """Normalise columns, convert to frames, sort by time. Returns a new DataFrame."""
    df = events.copy()
    for col in ("code", "Team", "Type", "Half"):
        df[col] = df[col].astype("string").str.strip()
    df["code_u"] = df["code"].str.upper()
    df["type_l"] = df["Type"].str.lower()
    df["ha"] = df["Team"].str.lower().map(TEAM_MAP)          # 'H' / 'A' / NaN
    df["role"] = df["code_u"].map(ROLE)
    # Timedelta objects or strings such as '0 days 00:07:23.880000' both work here.
    df["start_s"] = pd.to_timedelta(df["timestamp"]).dt.total_seconds()
    df["end_s"] = pd.to_timedelta(df["end_timestamp"]).dt.total_seconds()
    df["start_frame"] = np.rint(df["start_s"] * FPS).astype("int64")
    df["end_frame"] = np.rint(df["end_s"] * FPS).astype("int64")   # half-open, as in the starter
    # polarity: which side the tag says has the ball (None where the tag says nothing).
    pol = df["ha"].where(df["role"].isin(POLARITY_ROLES))
    is_def_tr = df["code_u"] == "DEFENSIVE TRANSITION"
    df["polarity"] = [other(s) if d else s for s, d in zip(pol, is_def_tr)]
    df["polarity"] = df["polarity"].where(df["polarity"].isin(["H", "A"]), None)
    return df.sort_values(["start_frame", "end_frame", "code_id"], kind="mergesort").reset_index(drop=True)


def head(title):
    print("\n" + "=" * 8 + " " + title + " " + "=" * 8)


# ---------------------------------------------------------------------------------- checks
def check_basic(df):
    head("1. basic validity")
    print("instances: %d   unknown codes: %s" % (len(df), sorted(set(df.loc[df.role.isna(), "code_u"]))))
    bad = df[(df.end_frame < df.start_frame) | df[["start_frame", "end_frame"]].isna().any(axis=1)]
    print("end < start or NaN: %d   zero-length: %d" % (len(bad), int((df.end_frame == df.start_frame).sum())))
    print("sorted by start_frame: %s" % df.start_frame.is_monotonic_increasing)
    print("Team values: %s" % df["Team"].value_counts(dropna=False).to_dict())
    unmapped = df[df.ha.isna() & df["Team"].notna() & (df["Team"].str.upper() != "N/A")]
    print("Team values not mapped to a side (excluding N/A): %s" % sorted(set(unmapped["Team"])))
    dur = (df.end_frame - df.start_frame) / FPS
    print("duration (s) by code:\n%s" % dur.groupby(df.code_u).describe()[["count", "min", "50%", "max"]].round(1).to_string())


def check_time_base(df):
    head("2. time base (tagged kickoffs must contain the published kickoff frames)")
    ko = df[(df.code_u == "SET PIECES") & (df.type_l == "kick off (start)")]
    for half, frame in KICKOFF_FRAMES.items():
        hit = ko[(ko.Half == half) & (ko.start_frame <= frame) & (frame <= ko.end_frame)]
        print("%s: real kickoff %d -> contained by a tag: %s" % (half, frame, len(hit) > 0))
    # half label vs frame block
    blk = []
    for _, r in df.iterrows():
        h = next((b[0] for b in BLOCKS if b[1] - 250 <= r.start_frame <= b[2] + 250), None)
        blk.append(h)
    exp = pd.Series(blk).map({1: "1st Half", 2: "2nd Half"})
    mism = int(((exp != df["Half"]) & exp.notna()).sum())
    print("tags whose Half label disagrees with the half block of their start frame: %d" % mism)
    print("tags starting outside both graded blocks (+-10 s): %d" % int(exp.isna().sum()))


def check_duplicates(df):
    head("3. duplicates / identical starts")
    key = ["start_frame", "end_frame", "code_u", "ha"]
    exact = df.duplicated(key, keep=False)
    print("exact duplicates (same start, end, code, side): %d rows" % int(exact.sum()))
    same_start = df[df.duplicated(["start_frame", "code_u"], keep=False)]
    diff_team = same_start.groupby(["start_frame", "code_u"])["ha"].nunique(dropna=False)
    print("same start_frame + same code: %d groups, %d with differing team (different-team ones are legitimate)"
          % (len(diff_team), int((diff_team > 1).sum())))
    multi = df.groupby("start_frame").size()
    print("start frames shared by >1 tag: %d (max %d tags at one frame)" % (int((multi > 1).sum()), int(multi.max())))


def _count(df, first, last):
    n = last - first + 1
    diff = np.zeros(n + 1, dtype=np.int64)
    s = np.clip(df.start_frame.values, first, last + 1) - first
    e = np.clip(df.end_frame.values, first, last + 1) - first
    ok = e > s
    np.add.at(diff, s[ok], 1)
    np.add.at(diff, e[ok], -1)
    return np.cumsum(diff)[:n]


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def check_coverage(df):
    head("4. coverage of graded frames (all tags, then possession-bearing tags only)")
    for label, sub in (("ALL tags", df), ("polarity tags (possession+transition)", df[df.role.isin(POLARITY_ROLES)]),
                       ("stoppage tags", df[df.role == "stoppage"])):
        counts, gaps = [], []
        for _, first, last in BLOCKS:
            c = _count(sub, first, last)
            counts.append(c)
            gaps += [(b - a) / FPS for a, b in _runs(c == 0)]
        c = np.concatenate(counts)
        g = np.array(gaps) if gaps else np.zeros(1)
        print("%-40s covered %.1f%%  by>=2 tags %.1f%% | untagged gaps: n=%d median %.1fs p90 %.1fs max %.1fs"
              % (label, 100.0 * (c > 0).mean(), 100.0 * (c > 1).mean(), len(gaps),
                 np.median(g), np.percentile(g, 90), g.max()))


def check_conflicts(df):
    head("5. overlapping polarity tags that disagree")
    p = df[df.polarity.notna()].reset_index(drop=True)
    n_over = n_conf = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p.start_frame[j] >= p.end_frame[i]:
                break                                   # sorted by start: nothing later overlaps i
            n_over += 1
            if p.polarity[i] != p.polarity[j]:
                n_conf += 1
    print("polarity tags: %d | overlapping pairs: %d | of which opposite polarity: %d" % (len(p), n_over, n_conf))


def check_flips(df):
    head("6. polarity sequence and flips (change windows)")
    p = df[df.polarity.notna()].reset_index(drop=True)
    flips = []
    for i in range(1, len(p)):
        if p.polarity[i] != p.polarity[i - 1]:
            flips.append((p.end_frame[i - 1], p.start_frame[i], p.code_u[i - 1], p.code_u[i]))
    print("polarity tags: %d | consecutive pairs: %d | polarity flips: %d" % (len(p), len(p) - 1, len(flips)))
    if flips:
        w = np.array([(b - a) / FPS for a, b, _, _ in flips])
        print("window width (next.start - prev.end), s: negative = overlap -> "
              "min %.1f  p25 %.1f  median %.1f  p75 %.1f  max %.1f" % (w.min(), *np.percentile(w, [25, 50, 75]), w.max()))
        print("overlapping windows (<0 s): %d | wider than 60 s: %d" % (int((w < 0).sum()), int((w > 60).sum())))
        st = df[df.role == "stoppage"]
        inside = sum(((st.start_frame >= min(a, b)) & (st.start_frame <= max(a, b))).any() for a, b, _, _ in flips)
        print("flip windows containing a stoppage-tag start: %d of %d" % (inside, len(flips)))
    print("first-tag-of-flip codes:\n%s" % pd.Series([f[3] for f in flips]).value_counts().to_string())


def check_transition_pairs(df, tol_s=6.0):
    head("7. do Attacking/Defensive Transition tags describe the same events? (+-%.0f s)" % tol_s)
    att = df[df.code_u == "ATTACKING TRANSITION"]
    dfn = df[df.code_u == "DEFENSIVE TRANSITION"]
    tol = tol_s * FPS
    def matched(a, b, flip):
        hits = 0
        for _, r in a.iterrows():
            want = other(r.ha) if flip else r.ha
            hits += int(((b.ha == want) & ((b.start_frame - r.start_frame).abs() <= tol)).any())
        return hits
    print("Attacking(X) with a Defensive(opponent of X) nearby: %d of %d" % (matched(att, dfn, True), len(att)))
    print("Defensive(X) with an Attacking(opponent of X) nearby: %d of %d" % (matched(dfn, att, True), len(dfn)))
    print("Attacking(X) with a Defensive(X) nearby (same side, would be odd): %d" % matched(att, dfn, False))


def check_set_pieces_and_goals(df):
    head("8. set pieces and goals")
    sp = df[df.role == "stoppage"]
    sp = sp[sp.code_u == "SET PIECES"]
    t = sp.assign(dur=(sp.end_frame - sp.start_frame) / FPS)
    print("set pieces by Type: n, side counts, window duration (s)")
    for typ, g in t.groupby("Type"):
        print("  %-22s n=%2d  H=%d A=%d none=%d  dur median %.1f min %.1f max %.1f"
              % (typ, len(g), (g.ha == "H").sum(), (g.ha == "A").sum(), g.ha.isna().sum(),
                 g.dur.median(), g.dur.min(), g.dur.max()))
    na = df[df["Team"].str.upper().eq("N/A")]
    print("Team == N/A by code/Type:\n%s" % na.groupby(["code_u", na["Type"].fillna("-")]).size().to_string())
    goals = df[df.code_u == "GOALS"]
    kos = sp[sp.type_l == "kick off (after goal)"]
    for _, g in goals.iterrows():
        nxt = kos[kos.start_frame >= g.start_frame].head(1)
        if len(nxt):
            k = nxt.iloc[0]
            print("goal side=%s at frame %d -> next after-goal kickoff side=%s, starts %.1fs after goal end"
                  % (g.ha, g.start_frame, k.ha, (k.start_frame - g.end_frame) / FPS))
        else:
            print("goal side=%s at frame %d -> no later after-goal kickoff" % (g.ha, g.start_frame))


MIRRORS = [("BUILD UP", "DEFENDING IN ATTACKING THIRD"),
           ("PROGRESSION", "DEFENDING IN MIDDLE THIRD"),
           ("CREATING CHANCES", "DEFENDING IN DEFENSIVE THIRD")]


def check_mirrors(df):
    head("9. do 'Defending in X third' tags mirror the possession tags?")
    for a, b in MIRRORS:
        A, B = df[df.code_u == a], df[df.code_u == b]
        m = A.merge(B, on=["start_frame", "end_frame"], suffixes=("_a", "_b"))
        both = m.ha_a.notna() & m.ha_b.notna()
        opp, same = int((both & (m.ha_a != m.ha_b)).sum()), int((both & (m.ha_a == m.ha_b)).sum())
        near = sum(int(((B.start_frame - r).abs() <= FPS).any()) for r in A.start_frame)
        print("%-18s (%d) vs %-30s (%d): identical start+end %d | team opposite %d, same %d, missing %d | "
              "A with a B starting within 1 s: %d"
              % (a, len(A), b, len(B), len(m), opp, same, len(m) - opp - same, near))


def check_goals(df):
    head("10. goal tags: duration, restart timing, which side the Team names")
    pol = df[df.polarity.notna()]
    kos = df[(df.code_u == "SET PIECES") & (df.type_l == "kick off (after goal)")]
    agree = disagree = 0
    for _, g in df[df.code_u == "GOALS"].iterrows():
        k = kos[(kos.start_frame >= g.start_frame)].head(1)
        kt = ("kickoff window starts %.1fs / ends %.1fs relative to goal-tag end"
              % ((k.iloc[0].start_frame - g.end_frame) / FPS, (k.iloc[0].end_frame - g.end_frame) / FPS)) if len(k) else "no kickoff"
        ov = pol[(pol.start_frame < g.end_frame) & (pol.end_frame > g.start_frame)]
        a = int((ov.polarity == g.ha).sum())
        d = int((ov.polarity != g.ha).sum())
        agree, disagree = agree + a, disagree + d
        print("goal Team=%s duration %.0fs | %s | overlapping polarity tags same side %d / other side %d"
              % (g.ha, (g.end_frame - g.start_frame) / FPS, kt, a, d))
    print("total same side %d vs other side %d (mostly 'same' => goal Team is the scorer)" % (agree, disagree))


def fmt_clock(frame):
    """Video clock H:MM:SS.ss from a frame index (the same clock as the tactical video)."""
    sec = frame / float(FPS)
    return "%d:%02d:%05.2f" % (int(sec // 3600), int(sec % 3600 // 60), sec % 60)


UNTAGGED_MODES = {"all": None, "polarity": POLARITY_ROLES, "stoppage": ("stoppage",)}


def untagged_intervals(df, mode="all", min_s=0.0):
    """Maximal stretches of the graded blocks with no active tag of the chosen kind.

    mode: 'all' = no tag of any kind; 'polarity' = no possession/transition tag;
          'stoppage' = no set-piece/goal tag.
    Frames are half-open [start, end). prev/next show the nearest tag of the same kind either side.
    """
    roles = UNTAGGED_MODES[mode]
    sub = df if roles is None else df[df.role.isin(roles)]
    rows = []
    for half, first, last in BLOCKS:
        c = _count(sub, first, last)
        for a, b in _runs(c == 0):
            a, b = int(a) + first, int(b) + first
            if (b - a) / float(FPS) < min_s:
                continue
            prev = sub[sub.end_frame <= a].sort_values("end_frame").tail(1)
            nxt = sub[sub.start_frame >= b].sort_values("start_frame").head(1)
            lab = lambda r: "-" if r.empty else "%s|%s" % (r.iloc[0].code_u, r.iloc[0].ha if isinstance(r.iloc[0].ha, str) else "-")
            rows.append({"half": half, "start_frame": a, "end_frame": b,
                         "start_clock": fmt_clock(a), "end_clock": fmt_clock(b),
                         "dur_s": round((b - a) / float(FPS), 1),
                         "block_edge": a == first or b == last + 1,
                         "prev_tag": lab(prev), "next_tag": lab(nxt)})
    return pd.DataFrame(rows)


def print_untagged(df, mode="all", min_s=0.0):
    head("untagged intervals (mode=%s, min %.1fs)" % (mode, min_s))
    u = untagged_intervals(df, mode, min_s)
    if u.empty:
        print("none")
        return u
    print("%d intervals, %.0fs total (block_edge=True: truncated by the graded-block boundary)"
          % (len(u), u.dur_s.sum()))
    with pd.option_context("display.max_rows", None, "display.width", 250, "display.max_colwidth", 60):
        print(u.drop(columns=["start_frame", "end_frame"]).assign(
            frames=u.start_frame.astype(str) + "-" + u.end_frame.astype(str)).to_string(index=False))
    return u


# ---------------------------------------------------------------------------------- timeline
def build_timeline(df):
    """Gap-free elementary intervals over the graded blocks.

    Each row is a maximal span [start_frame, end_frame) over which the SET of active tags does not
    change; `active` lists 'CODE|side' items, empty when untagged. Tags are not modified.
    """
    rows = []
    for half, first, last in BLOCKS:
        cuts = {first, last + 1}
        for s, e in zip(df.start_frame, df.end_frame):
            cuts.update(c for c in (s, e) if first < c <= last)
        cuts = sorted(cuts)
        for a, b in zip(cuts[:-1], cuts[1:]):
            act = df[(df.start_frame <= a) & (df.end_frame > a)]
            rows.append({"half": half, "start_frame": a, "end_frame": b,
                         "active": ";".join("%s|%s" % (c, s if isinstance(s, str) else "-")
                                            for c, s in zip(act.code_u, act.ha))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--untagged", choices=sorted(UNTAGGED_MODES), help="print only the untagged intervals and exit")
    ap.add_argument("--min-gap", type=float, default=0.0, help="with --untagged: hide gaps shorter than this many seconds")
    ap.add_argument("--timeline", help="optional path for the elementary-interval CSV (do not commit)")
    args = ap.parse_args()
    data = prepare(load_events())
    if args.untagged:
        print_untagged(data, args.untagged, args.min_gap)
        sys.exit(0)
    check_basic(data)
    check_time_base(data)
    check_duplicates(data)
    check_coverage(data)
    check_conflicts(data)
    check_flips(data)
    check_transition_pairs(data)
    check_set_pieces_and_goals(data)
    check_mirrors(data)
    check_goals(data)
    if args.timeline:
        os.makedirs(os.path.dirname(os.path.abspath(args.timeline)), exist_ok=True)
        tl = build_timeline(data)
        tl.to_csv(args.timeline, index=False)
        print("\nwrote %d elementary intervals -> %s" % (len(tl), args.timeline))