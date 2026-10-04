#!/usr/bin/env python3
"""Calibration matrix: which null sees which kind of planted spread, on this log's own rows.

Python standard library only, deterministic, counts only. See ``patient_zero/calibrate.py`` for the planted
kinds, the nulls and the reversed-clock z. Write the output outside the repo (or in its git-ignored out/)::

    python tools/calibrate.py --adapter wiki --input PATH/TO/collusion --out ../out/calibration.json \\
        --prereg docs/prereg/PAGE-INSTRUCTION.txt --expect PATH/TO/e7-results.json

``--prereg`` names the pre-registration written before the run; its sha256 goes into the output and onto
the page-instruction row. ``--expect`` names an earlier run's JSON (E7). Then the run is gated: the order
null on the real tokens (H0) and the E6/exp2 plantings (H1) must reproduce, and every cell that earlier run
holds must come out exactly the same. If any check fails, the page-instruction cells are withheld (removed
from the output, never printed) and only the mismatches are reported. Without ``--expect`` nothing is gated.

Exit codes: 0 done (gates passed or none asked), 1 a gate failed (page-instruction cells withheld),
2 setup error.
"""

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from patient_zero import calibrate as CAL  # noqa: E402
from patient_zero import nullmodel as N  # noqa: E402
from patient_zero.load import ADAPTERS, load  # noqa: E402
from patient_zero.tokens import MAX_PHRASE_SHARE, MIN_AGENTS, find_tokens  # noqa: E402


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def gates(result, expected):
    """H0, H1 and every-cell checks against an earlier run's JSON. Returns a dict with ``pass``."""
    e0 = expected.get("H0", {}).get("expected") or {k: expected.get("H0", {}).get(k)
                                                     for k in ("observed", "mean", "sd")}
    h0 = result["H0"]
    h0_pass = (e0.get("observed") is not None and h0["observed"] == e0["observed"]
               and h0["mean"] is not None and abs(h0["mean"] - e0["mean"]) <= 0.01
               and h0["sd"] is not None and abs(h0["sd"] - e0["sd"]) <= 0.001)
    e1 = expected.get("H1", {}).get("expected") or {}
    h1_pass = bool(e1) and all(result["H1"].get(k) == v for k, v in e1.items())
    cells = CAL.compare_expected(json.loads(json.dumps(result)), expected)
    return {"H0": {"got": h0, "expected": e0, "pass": h0_pass},
            "H1": {"got": result["H1"], "expected": e1, "pass": h1_pass},
            "H2_every_earlier_cell": {"mismatches": len(cells), "first_mismatches": cells[:20],
                                      "pass": not cells},
            "pass": h0_pass and h1_pass and not cells}


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="Calibration matrix: planted kinds against every null (counts only).")
    ap.add_argument("--adapter", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--input", required=True, help="export folder (wiki) or JSON Lines file (generic)")
    ap.add_argument("--out", required=True, help="JSON output, outside the repo (or in its out/ folder)")
    ap.add_argument("--table", help="also write the matrix as a Markdown table here")
    ap.add_argument("--shuffles", type=int, default=100)
    ap.add_argument("--seed", type=int, default=N.DEFAULT_SEED)
    ap.add_argument("--prereg", help="pre-registration text written before the run (its sha256 is recorded)")
    ap.add_argument("--expect", help="an earlier run's JSON (E7): gate on H0, H1 and every cell it holds")
    ap.add_argument("--min-agents", type=int, default=MIN_AGENTS)
    ap.add_argument("--max-phrase-share", type=float, default=MAX_PHRASE_SHARE)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()

    def log(*a):
        print("[%7.1fs]" % (time.perf_counter() - t0), *a, file=out, flush=True)

    try:
        from patient_zero.pipeline import check_out_dir  # late import: pipeline imports report
        for path in [args.out] + ([args.table] if args.table else []):
            check_out_dir(os.path.dirname(os.path.abspath(path)))
        if args.shuffles < 2:
            raise ValueError("--shuffles must be 2 or more")
        prereg_sha = sha256_file(args.prereg) if args.prereg else None
        expected = None
        if args.expect:
            with open(args.expect, encoding="utf-8") as fh:
                expected = json.load(fh)
        rows, _stats = load(args.adapter, args.input)
    except (OSError, ValueError) as exc:
        print("calibrate: ERROR: %s" % exc, file=out)
        return 2
    if prereg_sha:
        log("prereg sha256", prereg_sha)
    real, _ = find_tokens(rows, min_agents=args.min_agents, max_phrase_share=args.max_phrase_share)
    log("rows", len(rows), "real tokens", len(real))
    result = CAL.run(rows, real, shuffles=args.shuffles, seed=args.seed, log=log)
    result = {"tool": "patient-zero calibrate", "prereg_sha256": prereg_sha, **result}
    for r in result["matrix"]:
        if r["kind"] == "page_instruction":
            r["prereg_sha256"] = prereg_sha
            r["added"] = "after E7 (new pre-registration)"
    code = 0
    if expected is not None:
        g = gates(result, expected)
        result["gates"] = g
        log("gates H0", g["H0"]["pass"], "H1", g["H1"]["pass"], "every earlier cell",
            g["H2_every_earlier_cell"]["pass"], "(mismatches %d)" % g["H2_every_earlier_cell"]["mismatches"])
        if not g["pass"]:
            result = CAL.withhold(result, reason="the harness did not reproduce the earlier run")
            code = 1
    text = json.dumps(result, indent=1, ensure_ascii=True)
    for path, body in ((args.out, text + "\n"), (args.table, None)):
        if not path:
            continue
        if body is None:
            body = CAL.markdown_table(result) + "\n"
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(body)
    log("done", "(page-instruction cells withheld)" if code else "", "->", args.out)
    return code


if __name__ == "__main__":
    sys.exit(main())
