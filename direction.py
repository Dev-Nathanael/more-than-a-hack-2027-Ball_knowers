"""Attack direction per half, from team-wide positions only (no player identities).

    python direction.py          # needs cache/atd.npz (atd_explore.py) and the Smart Tagging file

Method 1 (primary): at kickoff each team stands in its own half, so the median x of a team's tracked
players says which end it defends. Several windows around the kickoff are printed so you can see
whether they agree. The two halves must come out opposite.
Method 2 (independent check): while a team is 'CREATING CHANCES' the ball should be mostly in the half
it attacks; during its 'BUILD UP' mostly in the half it defends. If Method 1 were wrong, these flip.

Convention: x in the ATD is 0-1 along the pitch length. dir = +1 means the team attacks toward x = 1.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atd_explore import load_or_parse  # noqa: E402
from tag_diagnostics import KICKOFF_FRAMES, load_events, prepare  # noqa: E402

# windows (start, end) in frames relative to the kickoff frame
WINDOWS = {"ko-5s..ko-4s": (-125, -100), "ko-2s..ko": (-50, 0), "ko-1s..ko+0.04s": (-25, 1), "ko..ko+1s": (0, 25)}
PRIMARY = "ko-1s..ko+0.04s"
MIN_MARGIN = 0.10          # centroids closer than this (about 10 m) = not a trustworthy kickoff reading


def head(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


def team_median_x(P, team, a, b):
    out = []
    for t in (0, 1):
        x = P[a:b, :, 0][:, team == t]
        x = x[np.isfinite(x)]
        out.append((float(np.median(x)) if x.size else np.nan, x.size / max(1, b - a)))
    return out


def kickoff_directions(P, team):
    head("1. kickoff positions: median x of each team's tracked players")
    dirs = {}
    for half, name in ((1, "1st Half"), (2, "2nd Half")):
        ko = KICKOFF_FRAMES[name]
        print("\n%s kickoff, frame %d" % (name, ko))
        reads = {}
        for w, (a, b) in WINDOWS.items():
            (xh, nh), (xa, na) = team_median_x(P, team, ko + a, ko + b)
            reads[w] = (xh, xa)
            print("  %-18s home x=%.3f (%.1f tracked/frame) | away x=%.3f (%.1f tracked/frame) | home attacks %s"
                  % (w, xh, nh, xa, na, "+x" if xh < xa else "-x"))
        xh, xa = reads[PRIMARY]
        ok = np.isfinite(xh) and np.isfinite(xa) and abs(xh - xa) >= MIN_MARGIN
        agree = len({xh < xa for xh, xa in reads.values() if np.isfinite(xh) and np.isfinite(xa)}) == 1
        dirs[half] = (+1 if xh < xa else -1) if ok else None
        print("  -> home direction %s | margin %.3f %s | windows agree: %s"
              % ({1: "+x", -1: "-x", None: "UNDETERMINED"}[dirs[half]], abs(xh - xa), "ok" if ok else "TOO SMALL", agree))
    if dirs[1] is not None and dirs[2] is not None:
        print("\nhalves opposite (expected after the change of ends): %s" % (dirs[1] == -dirs[2]))
    return dirs


def tag_check(B, dirs):
    head("2. independent check with tags: ball progress while a team is CREATING CHANCES / BUILD UP")
    print("progress = ball x measured toward the tagged team's attacking goal (0 = own goal line, 1 = opponent's)")
    d = prepare(load_events())
    for code, expect in (("CREATING CHANCES", "> 0.5"), ("BUILD UP", "< 0.5")):
        print("\n%s (expect progress %s)" % (code, expect))
        sub = d[(d.code_u == code) & d.ha.isin(["H", "A"])]
        tot = good = 0
        for half, name in ((1, "1st Half"), (2, "2nd Half")):
            if dirs.get(half) is None:
                continue
            for side in ("H", "A"):
                tdir = dirs[half] if side == "H" else -dirs[half]
                prog = []
                for r in sub[(sub.Half == name) & (sub.ha == side)].itertuples():
                    x = B[r.start_frame:r.end_frame, 0]
                    x = x[np.isfinite(x)]
                    if x.size >= 25:
                        prog.append(float(np.median(x if tdir > 0 else 1 - x)))
                if prog:
                    p = np.array(prog)
                    ok = (p > 0.5) if code == "CREATING CHANCES" else (p < 0.5)
                    tot += len(p)
                    good += int(ok.sum())
                    print("  %s %s: %2d tags with ball visible, median progress %.2f, consistent with the direction: %.0f%%"
                          % (name, side, len(p), np.median(p), 100 * ok.mean()))
        if tot:
            print("  overall consistent: %.0f%% of %d tags (about 50%% would mean the direction carries no information)" % (100 * good / tot, tot))


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    P, B, team, *_ = load_or_parse(False)
    dirs = kickoff_directions(P, team)
    tag_check(B, dirs)
