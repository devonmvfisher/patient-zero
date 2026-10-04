#!/usr/bin/env python3
"""Build calibration.json (schema ``pz-calibration/0``) for docs/can-it-see.svg from two real runs.

Python standard library only. Deterministic: the same inputs give the same bytes. Counts only.

Reads, never writes:

``--calibrate``  the output of ``tools/calibrate.py`` (planted kinds on the real rows, every null);
                 only its order-null (N3) cells are used, the same null as the real result.
``--crossgrid``  the output of ``python -m patient_zero.crossgrid`` (the real links split into
                 cross-page and same-page under that null, plus its random-row controls).
``--checks``     optional: checks.json, to stop here if its order null is not the same run.

Writes ``--out`` (outside the repo, or in its git-ignored out/)::

    python tools/calibrate.py --adapter wiki --input DATA --out ../out/calibrate.json \\
        --prereg docs/prereg/PAGE-INSTRUCTION.txt --expect PATH/TO/e7-results.json
    python -m patient_zero.crossgrid --adapter wiki --input DATA --out ../out/crossgrid.json
    python tools/make_calibration.py --calibrate ../out/calibrate.json --crossgrid ../out/crossgrid.json \\
        --checks ../out/checks.json --out ../out/calibration.json
    python -m patient_zero.canisee --calibration ../out/calibration.json --checks ../out/checks.json

It refuses to write (exit 2, one plain sentence) when the two runs did not use the same null and the same
shuffles: the seed, the shuffle count, the real token and row counts must match, and the cross-page and
same-page split must add up to the calibrate run's real all-links count and mean. It also refuses a calibrate
run whose gates failed, and a planted kind whose contact flag it does not know (fail closed).

The random-row control at 400 tokens (crossgrid ``random_c2``, pre-registered in PZ-5) is copied under
``controls``. The figure does not draw it yet; it is the cell that shows whether the cross-page bar reads
structureless tokens at 1.0.
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from patient_zero.safe import guard_out  # noqa: E402

SCHEMA = "pz-calibration/0"
FIELDS = ("observed", "mean", "sd", "p05", "p95", "ratio", "shuffles_ge_observed", "shuffles_le_observed")
SPLIT_TOL = 0.011  # two means rounded to 0.01 can miss their rounded sum by 0.01
HORIZON = "none"  # the shipped rule: no exposure horizon (crossgrid's column that equals the tracer)

# Planted kind -> (what it is, whether agents had contact). From patient_zero/calibrate.py's generators.
KINDS = {
    "carry": ("seen on page P, used later on page Q", True),
    "contagion": ("E6: adopters on the seed's own page", True),
    "same": ("exp2: adopters on the seed's own page", True),
    "mixed": ("exp2: half carried, half same-page", True),
    "pool_contagion": ("power pool: same-page", True),
    "pool_carry": ("power pool: carry", True),
    "page_instruction": ("page-wide shared cause, no contact (prereg PAGE-INSTRUCTION)", False),
    "decoy_fam": ("E6 decoy: family-day first rows, no contact", False),
    "decoy_fam2": ("exp2 decoy: family-day, no contact", False),
    "decoy_bcast": ("exp2 decoy: broadcast to one day, no contact", False),
    "random": ("random rows, no contact", False),
    "repeat": ("self-repeat, no contact", False),
}


class CalibrationError(ValueError):
    """The two runs cannot be joined honestly."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read(path, what):
    try:
        with open(path, encoding="utf-8-sig") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise CalibrationError("%s %s is not valid JSON (%s)" % (what, path, exc.msg)) from None


def cell(src, where):
    if not isinstance(src, dict):
        raise CalibrationError("%s is missing" % where)
    missing = [k for k in ("observed", "mean", "p05", "p95") if not isinstance(src.get(k), (int, float))]
    if missing:
        raise CalibrationError("%s has no %s" % (where, ", ".join(missing)))
    return {k: src[k] for k in FIELDS if k in src}


def build(cal, grid, checks=None, sources=None):
    """The pz-calibration/0 document. Raises CalibrationError."""
    if cal.get("tool") != "patient-zero calibrate":
        raise CalibrationError("--calibrate is not a tools/calibrate.py output")
    g = grid.get("grid", grid)
    if g.get("tool") != "patient-zero cross-page grid":
        raise CalibrationError("--crossgrid is not a patient_zero.crossgrid output")
    gates = cal.get("gates")
    if gates is not None and not gates.get("pass"):
        raise CalibrationError("the calibrate run's gates did not pass; its page-instruction cells were withheld")
    gnull = g.get("null") or {}
    if (gnull.get("kind"), gnull.get("scope")) != ("order", "agent_day"):
        raise CalibrationError("the cross-page grid did not use the agent-day order null")
    if (cal.get("seed"), cal.get("shuffles")) != (gnull.get("seed"), gnull.get("shuffles")):
        raise CalibrationError("the two runs used different shuffles: calibrate seed %s x %s, crossgrid seed %s x %s"
                               % (cal.get("seed"), cal.get("shuffles"), gnull.get("seed"), gnull.get("shuffles")))
    if (cal.get("rows"), cal.get("real_tokens")) != (g.get("rows"), g.get("tokens")):
        raise CalibrationError("the two runs read different data: %s rows / %s tokens vs %s rows / %s tokens"
                               % (cal.get("rows"), cal.get("real_tokens"), g.get("rows"), g.get("tokens")))
    n3 = (cal.get("nulls") or {}).get("N3_order_agent_day") or {}
    h0 = cal.get("H0") or {}
    real_all = cell(n3.get("real"), "calibrate nulls.N3_order_agent_day.real")
    if any(real_all.get(k) != h0.get(k) for k in ("observed", "mean", "sd")):
        raise CalibrationError("calibrate H0 does not equal its own N3 real cell")
    total = g.get("total") or {}
    cross = cell((total.get("cross_page") or {}).get(HORIZON), "crossgrid total.cross_page.none")
    same = cell((total.get("same_page") or {}).get(HORIZON), "crossgrid total.same_page.none")
    if (cross["observed"] + same["observed"] != h0.get("observed")
            or abs(cross["mean"] + same["mean"] - h0.get("mean", 0)) > SPLIT_TOL):
        raise CalibrationError("crossgrid cross-page + same-page do not add up to the calibrate run's real links: "
                               "not the same shuffles")
    if checks is not None:
        order = (checks.get("order_null") or {})
        named = order.get("links_named") or {}
        if any(named.get(k) != h0.get(k) for k in ("observed", "mean", "sd")):
            raise CalibrationError("checks.json order_null.links_named does not equal the calibrate run's H0")
        if (order.get("seed"), order.get("shuffles"), order.get("scope")) != (cal["seed"], cal["shuffles"], "agent_day"):
            raise CalibrationError("checks.json used another order null")
    planted = {}
    kinds = n3.get("kinds") or {}
    for kind in sorted(kinds):
        if kind not in KINDS:
            raise CalibrationError("planted kind %r has no known contact flag; add it to KINDS first" % kind)
        what, contact = KINDS[kind]
        entry = {"what": what, "contact": contact, "tokens": (cal.get("planted_tokens") or {}).get(kind),
                 "all_links": cell(kinds[kind], "calibrate N3 cell %s" % kind)}
        if kind == "page_instruction":
            entry["prereg_sha256"] = cal.get("prereg_sha256")
        planted[kind] = entry
    for need in ("carry", "contagion"):
        if need not in planted:
            raise CalibrationError("the calibrate run has no %s cell" % need)
    doc = {
        "schema": SCHEMA,
        "tool": "patient-zero make_calibration",
        "sources": sources or [],
        "gated": gates is not None,
        "null": {"kind": "order", "scope": "agent_day", "shuffles": cal["shuffles"], "seed": cal["seed"],
                 "min_lag_seconds": 0, "strict_exposure": False,
                 "h0": {k: h0[k] for k in ("observed", "mean", "sd")}},
        "planted": planted,
        "real": {"all_links": real_all, "cross_page": cross, "same_page": same},
        "controls": {},
    }
    c2 = ((g.get("controls") or {}).get("random_c2") or {})
    if c2:
        doc["controls"]["random_400"] = {
            "what": "random rows, no contact (crossgrid random_c2, prereg PZ-5)", "contact": False,
            "tokens": c2.get("tokens"),
            "cross_page": cell((c2.get("cross_page") or {}).get(HORIZON), "crossgrid random_c2 cross_page.none"),
        }
    return doc


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="calibration.json for docs/can-it-see.svg from a calibrate run "
                                             "and a cross-page grid run (counts only).")
    ap.add_argument("--calibrate", required=True, help="tools/calibrate.py output")
    ap.add_argument("--crossgrid", required=True, help="python -m patient_zero.crossgrid output")
    ap.add_argument("--checks", help="checks.json (optional cross-check of the order null)")
    ap.add_argument("--out", required=True, help="calibration.json, outside the repo or in its out/ folder")
    args = ap.parse_args(argv)
    inputs = [args.calibrate, args.crossgrid, args.checks]
    try:
        from patient_zero.pipeline import check_out_dir  # late import: pipeline imports report
        check_out_dir(os.path.dirname(os.path.abspath(args.out)))
        guard_out(args.out, inputs, "--out")
        cal = _read(args.calibrate, "--calibrate")
        grid = _read(args.crossgrid, "--crossgrid")
        checks = _read(args.checks, "--checks") if args.checks else None
        sources = [{"name": os.path.basename(p), "sha256": sha256_file(p)} for p in inputs if p]
        doc = build(cal, grid, checks, sources)
    except (OSError, ValueError) as exc:
        print("make_calibration: ERROR: %s" % exc, file=out)
        return 2
    path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)
    print("make_calibration: wrote %s (%d planted kinds, controls: %s; sha256 %s)"
          % (path, len(doc["planted"]), ", ".join(sorted(doc["controls"])) or "none", sha256_file(path)), file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
