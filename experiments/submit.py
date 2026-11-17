"""Build the file we upload: run one solution, validate it, fingerprint it, optionally prove it repeats.

    python experiments/submit.py --solution 05 --team <team_name>
    python experiments/submit.py --solution 05 --team <team_name> --check-reproducible

Writes evaluation/results/<team_name>.csv (gitignored, so it never reaches the public repo), runs
the official validation, prints its SHA-256, and scores it on the public window. With
--check-reproducible it runs the solution again in a fresh process with an empty cache folder (so
the tracking is re-read from the raw files) and fails unless both files are byte-identical -- the organisers may rerun our repository and expect exactly this file.

experiments/expected_fingerprints.txt records the SHA-256 of the file we uploaded (a hash only, no match
data). When the solution built is listed there, MATCH or MISMATCH is printed, so any edit to the code can be
checked against the uploaded file in one command.
"""

import argparse
import hashlib
import os

from pipelines import SOLUTIONS
from score import RESULTS, save_and_score


EXPECTED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected_fingerprints.txt")


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def expected(path=EXPECTED):
    """{solution: sha256} from the tracked file: one "<solution> <sha256>" per line, # starts a comment."""
    out = {}
    if os.path.exists(path):
        with open(path) as handle:
            for line in handle:
                parts = line.split("#")[0].split()
                if len(parts) == 2:
                    out[parts[0]] = parts[1].lower()
    return out


def verdict(solution, digest, known):
    """"MATCH", "MISMATCH" or None (no fingerprint recorded for this solution)."""
    if solution not in known:
        return None
    return "MATCH" if known[solution] == digest.lower() else "MISMATCH"


def _self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data, listing = os.path.join(tmp, "a.csv"), os.path.join(tmp, "expected.txt")
        with open(data, "wb") as handle:
            handle.write(b"frame,label\n0,H\n")
        digest = sha256(data)
        assert digest == hashlib.sha256(b"frame,label\n0,H\n").hexdigest()
        with open(listing, "w") as handle:
            handle.write("# comment\n40 {}  # the upload\n41 {}\n".format(digest.upper(), "0" * 64))
        known = expected(listing)
        assert verdict("40", digest, known) == "MATCH" and verdict("41", digest, known) == "MISMATCH"
        assert verdict("05", digest, known) is None and expected(os.path.join(tmp, "missing.txt")) == {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--solution", required=True, choices=sorted(SOLUTIONS))
    parser.add_argument("--team", required=True, help="the file is named after it, as SUBMISSION.md asks")
    parser.add_argument("--check-reproducible", action="store_true")
    args = parser.parse_args()

    _self_test()
    save_and_score(SOLUTIONS[args.solution](), args.team)
    path = os.path.join(RESULTS, args.team + ".csv")
    first = sha256(path)
    print("  sha256 {}".format(first))
    result = verdict(args.solution, first, expected())
    if result:
        print("  {}: the fingerprint recorded for solution {} in expected_fingerprints.txt".format(result, args.solution))

    if args.check_reproducible:
        # A fresh process with an empty cache folder: everything is rebuilt from the provided files, so a
        # stale cache cannot make two runs agree.
        import shutil
        import subprocess
        import sys
        from tracking import DATA_DIR
        fresh = os.path.join(DATA_DIR, "cache", "_fresh_rerun")
        shutil.rmtree(fresh, ignore_errors=True)
        os.makedirs(fresh)
        rerun = os.path.join(RESULTS, args.team + "_rerun.csv")
        try:
            subprocess.run([sys.executable, os.path.abspath(__file__), "--solution", args.solution,
                            "--team", args.team + "_rerun"], check=True, stdout=subprocess.DEVNULL,
                           env=dict(os.environ, MTAH_CACHE_DIR=fresh))
            second = sha256(rerun)
        finally:
            shutil.rmtree(fresh, ignore_errors=True)
            if os.path.exists(rerun):
                os.remove(rerun)
        if second != first:
            raise SystemExit("NOT reproducible: a rebuild from an empty cache gave sha256 {}".format(second))
        print("  reproducible: a rebuild from the raw files with an empty cache gave the identical file")
    if result == "MISMATCH":
        raise SystemExit("MISMATCH: this is not the file whose fingerprint is recorded for solution {}".format(args.solution))
