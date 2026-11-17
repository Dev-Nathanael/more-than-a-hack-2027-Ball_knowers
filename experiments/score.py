"""One call to save a solution's labels and score them the way the leaderboard does.

    from score import save_and_score
    save_and_score(labels, "my_solution")

Writes evaluation/results/<name>.csv, checks it with the official validator, and prints the
public-window scores plus the full-match possession count. Prints aggregates only.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)
import possession_metrics as pm  # noqa: E402
from baseline import write_submission  # noqa: E402
from games import GAME, half_video_blocks, source_frames  # noqa: E402

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
RESULTS = os.path.join(ROOT, "evaluation", "results")


def save_and_score(labels, name, bootstrap=1000):
    """Validate, write and score one full-length label array; returns the scorer's result dict."""
    path = write_submission(os.path.join(RESULTS, name + ".csv"), labels)
    frame = pm.read_label_csv(path)
    problem = pm.validate_submission(frame, source_frames("tactical"))
    if problem:
        raise ValueError("{} fails validation: {}".format(name, problem))

    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    pred = pm.label_codes(frame)
    r = pm.evaluate(gt, pred, bootstrap_n=bootstrap)
    hits = [v["n_matched"] for v in r["spell_f1"].values()]

    halves = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
    full = pm.possession_stats(pm.fill_abstentions(pred, halves), halves)

    print("{}  ->  {}".format(name, os.path.relpath(path, ROOT)))
    band = ("(90% range {:.2f}-{:.2f})".format(r["spell_bootstrap_low"], r["spell_bootstrap_high"])
            if bootstrap else "")
    print("  spellF1 {:.4f}  {}   macroF1 {:.4f}".format(r["spell_f1_mean"], band, r["macro_f1"]))
    print("  possessions caught at 30/40/50/60/70% overlap: {} of {}".format(
        " / ".join(map(str, hits)), r["n_gt_spells"]))
    print("  public window: {} possessions predicted (truth {})   alive/dead F1 {:.3f}".format(
        r["n_pred_spells"], r["n_gt_spells"], r["alive_dead_f1"]))
    print("  full match: {} possessions, mean {:.1f} s".format(full["n_spells"], full["mean_duration_s"]))
    return r


def graded_mask():
    mask = np.zeros(source_frames("tactical"), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first:last + 1] = True
    return mask
