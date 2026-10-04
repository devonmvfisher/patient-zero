"""Tests for tools/make_calibration.py: the bridge from a calibrate run and a cross-page grid run
to calibration.json (pz-calibration/0), which docs/can-it-see.svg is drawn from.

Standard library unittest only. Made-up counts; no data is read.
"""

import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from patient_zero import canisee as C  # noqa: E402

_spec = importlib.util.spec_from_file_location("make_calibration", os.path.join(ROOT, "tools", "make_calibration.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def cell(observed, mean, p05, p95, sd=1.5):
    return {"observed": observed, "mean": mean, "sd": sd, "p05": p05, "p95": p95, "shuffles_ge_observed": 50,
            "ratio": round(observed / mean, 3), "z": 0.0}


REAL = cell(515, 520.5, 505.0, 536.0, sd=9.25)


def calibrate_doc(gates_pass=True, page_instruction=True):
    kinds = {"carry": cell(300, 150.5, 140.0, 161.0), "contagion": cell(200, 199.9, 200.0, 200.0),
             "decoy_fam": cell(20, 22.0, 19.0, 25.0), "random": cell(9, 8.5, 6.0, 11.0)}
    if page_instruction:
        kinds["page_instruction"] = cell(60, 59.98, 60.0, 60.0)
    doc = {"tool": "patient-zero calibrate", "prereg_sha256": "ab" * 32, "shuffles": 100, "seed": 7,
           "rows": 1000, "real_tokens": 70, "planted_tokens": {k: 40 for k in kinds},
           "H0": {k: REAL[k] for k in ("observed", "mean", "sd")},
           "nulls": {"N1_label_day": {"kinds": {}},
                     "N3_order_agent_day": {"kinds": kinds, "real": dict(REAL)}}}
    if gates_pass is not None:
        doc["gates"] = {"pass": gates_pass}
    return doc


def crossgrid_doc():
    def both(cross, same):
        return {"cross_page": {"none": cross, "24h": cross}, "same_page": {"none": same, "24h": same}}
    return {"prereg_sha256": "cd" * 32,
            "grid": {"tool": "patient-zero cross-page grid",
                     "null": {"kind": "order", "scope": "agent_day", "shuffles": 100, "seed": 7},
                     "rows": 1000, "tokens": 70,
                     "total": both(cell(80, 100.25, 90.0, 110.0), cell(435, 420.25, 410.0, 431.0)),
                     "controls": {"random_c2": dict(both(cell(30, 35.0, 31.0, 39.0), cell(5, 5.0, 3.0, 7.0)),
                                                    tokens=400, role="calibration")}}}


def checks_doc():
    return {"tool": "patient-zero checks",
            "order_null": {"kind": "order", "scope": "agent_day", "shuffles": 100, "seed": 7, "min_lag_seconds": 0,
                           "strict_exposure": False, "adopters": 1000, "links_named": dict(REAL)}}


def _bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def run(args):
    buf = io.StringIO()
    return M.main(args, out=buf), buf.getvalue()


class Build(unittest.TestCase):
    def test_schema_rows_and_contact_flags(self):
        doc = M.build(calibrate_doc(), crossgrid_doc(), checks_doc())
        self.assertEqual(doc["schema"], C.SCHEMA)
        self.assertTrue(doc["gated"])
        self.assertEqual(doc["planted"]["carry"]["contact"], True)
        self.assertEqual(doc["planted"]["page_instruction"]["contact"], False)
        self.assertEqual(doc["planted"]["page_instruction"]["prereg_sha256"], "ab" * 32)
        self.assertEqual(doc["real"]["cross_page"]["observed"] + doc["real"]["same_page"]["observed"],
                         doc["null"]["h0"]["observed"])
        self.assertEqual(doc["controls"]["random_400"]["tokens"], 400)
        self.assertEqual(doc["controls"]["random_400"]["cross_page"]["observed"], 30)
        for kind in doc["planted"].values():  # only the cell fields travel, never z or ratio
            self.assertNotIn("z", kind["all_links"])

    def test_canisee_draws_every_row_from_it(self):
        doc = M.build(calibrate_doc(), crossgrid_doc(), checks_doc())
        rows = C.build_rows(doc, checks_doc())
        self.assertEqual([r["status"] for r in rows], ["drawn"] * 5)
        self.assertEqual([r["row"].key for r in rows],
                         ["carry", "same_page", "page_instruction", "real_all", "real_cross"])
        svg = C.draw(doc, checks_doc(), {"calibration.json": "0" * 64, "checks.json": "1" * 64})
        self.assertIn("<svg", svg)

    def test_withheld_page_instruction_draws_pending(self):
        doc = M.build(calibrate_doc(page_instruction=False), crossgrid_doc(), checks_doc())
        self.assertNotIn("page_instruction", doc["planted"])
        rows = C.build_rows(doc, checks_doc())
        self.assertEqual(rows[2]["status"], "pending")

    def test_ungated_run_is_marked(self):
        doc = M.build(calibrate_doc(gates_pass=None), crossgrid_doc())
        self.assertFalse(doc["gated"])

    def test_deterministic(self):
        a = json.dumps(M.build(calibrate_doc(), crossgrid_doc()), sort_keys=True)
        b = json.dumps(M.build(calibrate_doc(), crossgrid_doc()), sort_keys=True)
        self.assertEqual(a, b)


class Refusals(unittest.TestCase):
    def fails(self, cal, grid, needle, checks=None):
        with self.assertRaises(M.CalibrationError) as ctx:
            M.build(cal, grid, checks)
        self.assertIn(needle, str(ctx.exception))

    def test_failed_gates(self):
        self.fails(calibrate_doc(gates_pass=False), crossgrid_doc(), "gates did not pass")

    def test_other_shuffles(self):
        grid = crossgrid_doc()
        grid["grid"]["null"]["seed"] = 8
        self.fails(calibrate_doc(), grid, "different shuffles")

    def test_other_null_kind(self):
        grid = crossgrid_doc()
        grid["grid"]["null"]["scope"] = "day"
        self.fails(calibrate_doc(), grid, "agent-day order null")

    def test_other_data(self):
        grid = crossgrid_doc()
        grid["grid"]["tokens"] = 71
        self.fails(calibrate_doc(), grid, "different data")

    def test_split_does_not_add_up(self):
        grid = crossgrid_doc()
        grid["grid"]["total"]["cross_page"]["none"]["observed"] = 81
        self.fails(calibrate_doc(), grid, "do not add up")
        grid = crossgrid_doc()
        grid["grid"]["total"]["same_page"]["none"]["mean"] = 421.25
        self.fails(calibrate_doc(), grid, "do not add up")

    def test_unknown_kind_fails_closed(self):
        cal = calibrate_doc()
        cal["nulls"]["N3_order_agent_day"]["kinds"]["new_kind"] = cell(1, 1.0, 0.0, 2.0)
        self.fails(cal, crossgrid_doc(), "no known contact flag")

    def test_checks_from_another_run(self):
        chk = checks_doc()
        chk["order_null"]["links_named"]["mean"] = 520.6
        self.fails(calibrate_doc(), crossgrid_doc(), "does not equal", checks=chk)
        chk = checks_doc()
        chk["order_null"]["seed"] = 8
        self.fails(calibrate_doc(), crossgrid_doc(), "another order null", checks=chk)

    def test_wrong_files(self):
        self.fails(crossgrid_doc(), crossgrid_doc(), "not a tools/calibrate.py output")
        self.fails(calibrate_doc(), calibrate_doc(), "not a patient_zero.crossgrid output")

    def test_missing_cell_field(self):
        cal = calibrate_doc()
        del cal["nulls"]["N3_order_agent_day"]["kinds"]["carry"]["p95"]
        self.fails(cal, crossgrid_doc(), "has no p95")


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pz-makecal-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.paths = {}
        for name, doc in (("calibrate.json", calibrate_doc()), ("crossgrid.json", crossgrid_doc()),
                          ("checks.json", checks_doc())):
            self.paths[name] = os.path.join(self.tmp, name)
            with open(self.paths[name], "w", encoding="utf-8") as fh:
                json.dump(doc, fh)

    def args(self, out):
        return ["--calibrate", self.paths["calibrate.json"], "--crossgrid", self.paths["crossgrid.json"],
                "--checks", self.paths["checks.json"], "--out", out]

    def test_writes_same_bytes_twice(self):
        out = os.path.join(self.tmp, "calibration.json")
        code, text = run(self.args(out))
        self.assertEqual(code, 0, text)
        first = _bytes(out)
        code, _ = run(self.args(out))
        self.assertEqual(_bytes(out), first)
        doc = json.loads(first)
        self.assertEqual([s["name"] for s in doc["sources"]], ["calibrate.json", "crossgrid.json", "checks.json"])
        self.assertNotIn(b"\r", first)

    def test_out_over_an_input_is_refused_and_input_kept(self):
        target = self.paths["crossgrid.json"]
        before = hashlib.sha256(_bytes(target)).hexdigest()
        code, text = run(self.args(target))
        self.assertEqual(code, 2)
        self.assertTrue(text.startswith("make_calibration: ERROR: "), text)
        self.assertEqual(text.count("\n"), 1)
        self.assertEqual(hashlib.sha256(_bytes(target)).hexdigest(), before)

    def test_out_inside_repo_is_refused(self):
        code, text = run(self.args(os.path.join(ROOT, "docs", "calibration.json")))
        self.assertEqual(code, 2)
        self.assertIn("outside the repo", text)
        self.assertFalse(os.path.exists(os.path.join(ROOT, "docs", "calibration.json")))

    def test_bad_json_is_one_sentence(self):
        with open(self.paths["crossgrid.json"], "w", encoding="utf-8") as fh:
            fh.write("{not json")
        code, text = run(self.args(os.path.join(self.tmp, "calibration.json")))
        self.assertEqual(code, 2)
        self.assertIn("is not valid JSON", text)
        self.assertNotIn("Traceback", text)


if __name__ == "__main__":
    unittest.main()
