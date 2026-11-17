"""How well do ball and player signals separate TRUE dead-ball frames from live play? (public window only)

    python dead_ball_evidence.py        # needs cache/atd.npz and the public ground truth

Signals per frame (the player speed is the mean of the ATD speed field over all tracked players, smoothed
over SMOOTH frames, centred):
  out      ball visible and outside the pitch lines
  missing  ball not tracked
  slow     ball visible and slower than 1 m/s (5-frame)
  pspeed   mean player speed; 'low' = at or below a threshold
For D (dead ball, 1469 frames in 4 stretches) versus live (H or A) frames the tables give, per signal:
  recall      share of true D frames where the signal is on
  false alarm share of live frames where it is on
  precision   share of frames where it is on that are really D (depends on the D share, ~20% here)
Section 2 combines 'ball out / missing / slow' with 'players slow' for a GRID of thresholds (none is chosen).
Section 3 shows the mean player speed and ball visibility in the 10 s before, during and after each true D stretch.
Four stretches cannot tune anything; this is only a check of whether the idea can work at all.
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atd_explore import FPS, LENGTH, WIDTH, load_or_parse  # noqa: E402
from compare_gt import GT_PATH, load_gt, runs_of  # noqa: E402
from shape_features import auc  # noqa: E402
from shape_smoothing import smooth  # noqa: E402
from tag_diagnostics import fmt_clock  # noqa: E402

SMOOTH = 25                      # frames, centred moving mean for the player speed
THRESHOLDS = (1.0, 1.3, 1.6)     # m/s, grid for 'players slow'


def head(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


def signals(P, B):
    n = P.shape[0]
    vis = np.isfinite(B[:, 0]) & np.isfinite(B[:, 1])
    bx, by = B[:, 0] * LENGTH, B[:, 1] * WIDTH
    sp = np.full(n, np.nan)
    sp[5:] = np.hypot(bx[5:] - bx[:-5], by[5:] - by[:-5]) * FPS / 5.0
    out = vis & ((B[:, 0] < 0) | (B[:, 0] > 1) | (B[:, 1] < 0) | (B[:, 1] > 1))
    S = P[:, :, 2]
    fin = np.isfinite(S)
    ps = np.where(fin.sum(1) > 0, np.where(fin, S, 0).sum(1) / np.maximum(fin.sum(1), 1), np.nan)
    return {"vis": vis, "out": out, "missing": ~vis, "slow": vis & (sp <= 1.0), "ps": smooth(ps, SMOOTH), "ps_raw": ps}


def rates(sig, dead, live):
    on = sig & (dead | live)
    return sig[dead].mean(), sig[live].mean(), (on & dead).sum() / max(1, on.sum())


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    P, B, team, *_ = load_or_parse(False)
    lab_all = load_gt(GT_PATH)
    v = np.flatnonzero(lab_all != "X")
    s, e = int(v[0]), int(v[-1]) + 1
    G = signals(P, B)
    lab = lab_all[s:e].astype(str)
    dead, live = lab == "D", np.isin(lab, ["H", "A"])
    w = {k: x[s:e] for k, x in G.items()}
    print("public window %d frames: D %d (%.0f%%), live %d" % (e - s, dead.sum(), 100 * dead.mean(), live.sum()))

    head("1. single signals: true D frames versus live frames")
    print("%-30s %8s %12s %10s" % ("signal", "recall", "false alarm", "precision"))
    for name, sig in (("ball outside the pitch", w["out"]), ("ball not tracked", w["missing"]), ("ball visible and slow", w["slow"]),
                      ("ball out | missing | slow", w["out"] | w["missing"] | w["slow"])):
        r, f, p = rates(sig, dead, live)
        print("%-30s %7.0f%% %11.0f%% %9.0f%%" % (name, 100 * r, 100 * f, 100 * p))
    for t in THRESHOLDS:
        r, f, p = rates(w["ps"] <= t, dead, live)
        print("%-30s %7.0f%% %11.0f%% %9.0f%%" % ("player speed <= %.1f m/s" % t, 100 * r, 100 * f, 100 * p))
    m = dead | live
    print("\nAUC of mean player speed for D versus live (0.5 = no information; 1 - AUC if it points the other way): %.2f"
          % (1 - auc(w["ps"][m], dead[m])))
    print("median player speed: D %.2f m/s, live %.2f m/s" % (np.nanmedian(w["ps"][dead]), np.nanmedian(w["ps"][live])))

    head("2. combinations: (ball out | missing | slow) AND players slow")
    print("%-34s %8s %12s %10s" % ("rule", "recall", "false alarm", "precision"))
    ball = w["out"] | w["missing"] | w["slow"]
    for t in THRESHOLDS:
        r, f, p = rates(ball & (w["ps"] <= t), dead, live)
        print("%-34s %7.0f%% %11.0f%% %9.0f%%" % ("ball cue AND players <= %.1f m/s" % t, 100 * r, 100 * f, 100 * p))
    for t in THRESHOLDS:
        r, f, p = rates((w["out"] | w["missing"]) & (w["ps"] <= t), dead, live)
        print("%-34s %7.0f%% %11.0f%% %9.0f%%" % ("out|missing AND players <= %.1f m/s" % t, 100 * r, 100 * f, 100 * p))

    head("3. each true D stretch: mean player speed and ball visibility before / during / after (10 s each)")
    print("%-22s %8s | %-22s | %-22s | %-22s" % ("stretch", "length", "10 s before (speed, vis)", "during", "10 s after"))
    for a1, b1, l in runs_of(lab_all, s, e):
        if l != "D":
            continue
        cells = []
        for lo, hi in ((a1 - 10 * FPS, a1), (a1, b1), (b1, b1 + 10 * FPS)):
            lo, hi = max(lo, s), min(hi, e)
            cells.append("%.2f m/s, %3.0f%%" % (np.nanmean(G["ps"][lo:hi]), 100 * G["vis"][lo:hi].mean()))
        print("%-22s %7.1fs | %-22s | %-22s | %-22s" % (fmt_clock(a1)[2:9] + "-" + fmt_clock(b1)[2:9], (b1 - a1) / FPS, *cells))
