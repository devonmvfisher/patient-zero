"""Tests for the can-it-see figure (patient_zero/canisee.py) and figcheck's data mode.

Standard library unittest only. Everything runs on made-up calibration and
checks files. The committed figure is redrawn and checked only when PZ_OUT
names a folder holding the real calibration.json and checks.json; the label
and token scan on it runs only when PZ_DATA and PZ_TREES are set too.
"""

import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from patient_zero import canisee as C  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = _load("figcheck")
NS = "{http://www.w3.org/2000/svg}"
NULL = {"scope": "agent_day", "shuffles": 100, "seed": 7, "min_lag_seconds": 0, "strict_exposure": False}
REAL = {"observed": 515, "mean": 520.5, "sd": 9.25, "p05": 505.0, "p95": 536.0}


def cell(observed, mean, p05, p95):
    return {"observed": observed, "mean": mean, "sd": 1.5, "p05": p05, "p95": p95, "shuffles_ge_observed": 50}


def calibration(page_instruction=None):
    doc = {"schema": C.SCHEMA,
           "null": dict(NULL, kind="order", h0={k: REAL[k] for k in ("observed", "mean", "sd")}),
           "planted": {"carry": {"contact": True, "tokens": 40, "all_links": cell(300, 150.5, 140.0, 161.0)},
                       "contagion": {"contact": True, "tokens": 40, "all_links": cell(200, 199.9, 200.0, 200.0)}},
           "real": {"cross_page": cell(80, 100.25, 90.0, 110.0), "same_page": cell(435, 420.25, 410.0, 431.0)}}
    if page_instruction is not None:
        doc["planted"]["page_instruction"] = {"contact": False, "tokens": 40, "all_links": page_instruction}
    return doc


def checks():
    return {"tool": "patient-zero checks", "order_null": dict(NULL, adopters=1000, links_named=dict(REAL))}


def run(main, args):
    buf = io.StringIO()
    return main(args, out=buf), buf.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pz-canisee-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write(self, cal=None, chk=None):
        self.cal_path = os.path.join(self.tmp, "calibration.json")
        self.chk_path = os.path.join(self.tmp, "checks.json")
        for path, doc in ((self.cal_path, cal or calibration()), (self.chk_path, chk or checks())):
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=1)
        self.svg_path = os.path.join(self.tmp, "can-it-see.svg")
        code, text = run(C.main, ["--calibration", self.cal_path, "--checks", self.chk_path, "--svg", self.svg_path])
        self.assertEqual(code, 0, text)
        with open(self.svg_path, encoding="utf-8") as fh:
            self.svg = fh.read()
        return text

    def figcheck(self, svg_text=None):
        path = self.svg_path
        if svg_text is not None:
            path = os.path.join(self.tmp, "edited.svg")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(svg_text)
        code, text = run(F.main, ["--svg", path, "--sources", self.cal_path, self.chk_path, "--out-dir", self.tmp,
                                  "--json", "checks-report.json"])
        with open(os.path.join(self.tmp, "checks-report.json"), encoding="utf-8") as fh:
            return code, text, json.load(fh)

    def root(self, svg=None):
        return ET.fromstring((svg or self.svg).split("\n", 1)[1])

    def rows(self, svg=None):
        return {g.get("data-row"): g for g in self.root(svg).iter() if g.get("class") == "cis-row"}


class Rows(Base):
    def test_positions_and_words(self):
        rows = {r["row"].key: r for r in C.build_rows(calibration(), checks())}
        self.assertEqual([r.key for r in C.ROWS],
                         ["carry", "same_page", "page_instruction", "real_all", "random_all", "real_cross",
                          "random_cross"])
        self.assertEqual(rows["random_cross"]["status"], "pending")
        self.assertEqual(rows["random_all"]["status"], "pending")
        self.assertEqual(rows["carry"]["verdict"], "above the band: seen")
        self.assertEqual(rows["same_page"]["verdict"], "inside the band: blind spot")
        self.assertEqual(rows["page_instruction"]["status"], "pending")
        self.assertEqual(rows["real_all"]["verdict"], "inside the band")
        self.assertEqual(rows["real_cross"]["verdict"], "below the band")
        self.assertAlmostEqual(rows["carry"]["ratio"], 300 / 150.5)
        self.assertAlmostEqual(rows["real_cross"]["lo"], 90.0 / 100.25)

    def test_decoy_words(self):
        inside = {r["row"].key: r for r in C.build_rows(calibration(cell(50, 50.5, 45.0, 56.0)), checks())}
        self.assertEqual(inside["page_instruction"]["verdict"], "inside the band: not flagged")
        above = {r["row"].key: r for r in C.build_rows(calibration(cell(90, 50.5, 45.0, 56.0)), checks())}
        self.assertEqual(above["page_instruction"]["verdict"], "above the band: wrongly flagged")

    def test_every_row_uses_the_same_null(self):
        ids = set(r["null"] for r in C.build_rows(calibration(), checks()))
        self.assertEqual(ids, {"order:agent_day:lag0:default:seed7:n100"})

    def test_refuses_a_different_null(self):
        for field, value in (("seed", 8), ("shuffles", 30), ("strict_exposure", True), ("min_lag_seconds", 120),
                             ("scope", "agent")):
            chk = checks()
            chk["order_null"][field] = value
            with self.assertRaises(C.CanISeeError, msg=field):
                C.build_rows(calibration(), chk)

    def test_refuses_the_same_settings_with_other_shuffles(self):
        chk = checks()
        chk["order_null"]["links_named"]["mean"] = 521.0
        with self.assertRaisesRegex(C.CanISeeError, "different shuffles"):
            C.build_rows(calibration(), chk)

    def test_refuses_a_split_from_other_shuffles(self):
        for side, field, delta in (("cross_page", "mean", 1.0), ("same_page", "observed", 1)):
            cal = calibration()
            cal["real"][side][field] += delta
            with self.assertRaisesRegex(C.CanISeeError, "add up", msg=side):
                C.build_rows(cal, checks())
        cal = calibration()
        del cal["real"]["same_page"]  # without both halves there is nothing to add up
        self.assertEqual(len(C.build_rows(cal, checks())), 7)

    def test_refuses_missing_rows_and_wrong_schema(self):
        cal = calibration()
        del cal["real"]["cross_page"]
        with self.assertRaises(C.CanISeeError):
            C.build_rows(cal, checks())
        cal = calibration()
        cal["schema"] = "pz-calibration/9"
        with self.assertRaises(C.CanISeeError):
            C.build_rows(cal, checks())
        cal = calibration()
        del cal["planted"]["carry"]["contact"]
        with self.assertRaises(C.CanISeeError):
            C.build_rows(cal, checks())
        with self.assertRaises(C.CanISeeError):
            C.build_rows(calibration(), {"tool": "no order null"})

    def test_axis_covers_everything_and_labels_one(self):
        rows = C.build_rows(calibration(), checks())
        lo_t, hi_t, label_t, _grid = C.axis(rows)
        for r in rows:
            if r["status"] == "drawn":
                for v in (r["ratio"], r["lo"], r["hi"]):
                    self.assertTrue(lo_t / 10.0 <= v <= hi_t / 10.0)
        self.assertEqual((10 - lo_t) % label_t, 0)


class Figure(Base):
    def test_figcheck_passes(self):
        self.write()
        code, text, report = self.figcheck()
        self.assertEqual(code, 0, text)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(set(report["checks"]), set(F.DATA_PLANNED))
        self.assertEqual(report["checks"]["numbers_match_sources"]["rows_drawn"], 4)
        self.assertEqual(report["checks"]["numbers_match_sources"]["rows_pending"], 3)
        self.assertEqual(report["checks"]["same_null"]["null_id"], "order:agent_day:lag0:default:seed7:n100")

    def test_figcheck_passes_with_the_decoy_drawn(self):
        self.write(cal=calibration(cell(50, 50.5, 45.0, 56.0)))
        self.assertEqual(self.rows()["page_instruction"].get("data-status"), "drawn")
        code, text, _report = self.figcheck()
        self.assertEqual(code, 0, text)

    def test_skip_pending(self):
        self.write()
        code, text = run(C.main, ["--calibration", self.cal_path, "--checks", self.chk_path, "--svg", self.svg_path,
                                  "--skip-pending"])
        self.assertEqual(code, 0, text)
        self.assertNotIn("not run yet", text)
        with open(self.svg_path, encoding="utf-8") as fh:
            self.svg = fh.read()
        self.assertEqual(sorted(self.rows()), ["carry", "real_all", "real_cross", "same_page"])
        self.assertNotIn("not run yet", self.svg)
        code, text, _report = self.figcheck()
        self.assertEqual(code, 0, text)

    def test_one_null_id_on_every_row_and_in_the_footer(self):
        self.write()
        rows = self.rows()
        self.assertEqual(len(rows), 7)
        self.assertEqual(set(g.get("data-null") for g in rows.values()), {"order:agent_day:lag0:default:seed7:n100"})
        tag = hashlib.sha256(b"order:agent_day:lag0:default:seed7:n100").hexdigest()[:12]
        self.assertIn("null id %s: seed 7, 100 shuffles" % tag, self.svg)

    def test_footer_hashes(self):
        self.write()
        for path, name in ((self.cal_path, "calibration.json"), (self.chk_path, "checks.json")):
            self.assertIn("%s sha256 %s" % (name, C.sha256_file(path)[:16]), self.svg)

    def test_no_z_one_axis(self):
        self.write()
        texts = [t.text or "" for t in self.root().iter(NS + "text")]
        for s in texts:
            self.assertIsNone(F.Z_RE.search(s), s)
        self.assertEqual(sum(1 for g in self.root().iter() if g.get("class") == "cis-axis"), 1)
        self.assertNotIn("data-z", self.svg)

    def test_pending_row_draws_no_number(self):
        self.write()
        g = self.rows()["page_instruction"]
        self.assertEqual(g.get("data-status"), "pending")
        self.assertFalse([el for el in g.iter() if el.tag in (NS + "circle", NS + "rect")])
        self.assertFalse(re.search(r"\d", "".join(g.itertext())))

    def test_one_accent_dot_and_a_grey_band_per_drawn_row(self):
        self.write()
        for key, g in self.rows().items():
            dots = [el for el in g.iter(NS + "circle")]
            bands = [el for el in g.iter(NS + "rect")]
            if g.get("data-status") == "drawn":
                self.assertEqual((len(dots), len(bands)), (1, 1), key)
                self.assertEqual(dots[0].get("fill"), C.ACCENT)
                self.assertEqual(bands[0].get("fill"), C.BAND_FILL)

    def test_phone_check(self):
        self.write()
        phone = F.phone_check(self.root())
        self.assertTrue(phone["pass"])
        self.assertGreaterEqual(phone["phone_px"], 10)
        self.assertAlmostEqual(phone["phone_px"], C.phone_px(C.W, C.SIZE_SMALL), places=3)
        wide = ET.fromstring('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="100">'
                             '<g font-size="11"><text x="1" y="20">a</text></g>'
                             '<text x="1" y="40" font-size="12">b</text></svg>')
        phone = F.phone_check(wide)
        self.assertEqual(phone["smallest_font"], 11.0)  # inherited from the group
        self.assertAlmostEqual(phone["phone_px"], 11 * 358 / 960.0, places=3)
        self.assertFalse(phone["pass"])

    def test_text_fits_and_does_not_collide(self):
        self.write()
        boxes = []
        for t in self.root().iter(NS + "text"):
            size, x = float(t.get("font-size")), float(t.get("x"))
            width = len(t.text or "") * size * 0.58
            x0 = {"middle": x - width / 2, "end": x - width}.get(t.get("text-anchor"), x)
            self.assertGreaterEqual(x0, 0, t.text)
            self.assertLessEqual(x0 + width, C.W, t.text)
            boxes.append((float(t.get("y")), x0, x0 + width, size))
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                if abs(a[0] - b[0]) < max(a[3], b[3]):
                    self.assertTrue(a[2] <= b[1] or b[2] <= a[1], (a, b))

    def test_deterministic(self):
        self.write()
        hashes = {"calibration.json": C.sha256_file(self.cal_path), "checks.json": C.sha256_file(self.chk_path)}
        self.assertEqual(C.draw(calibration(), checks(), hashes), C.draw(calibration(), checks(), hashes))
        with open(self.cal_path, encoding="utf-8") as fh:
            cal = json.load(fh)
        with open(self.chk_path, encoding="utf-8") as fh:
            chk = json.load(fh)
        self.assertEqual(C.draw(cal, chk, hashes), self.svg)

    def test_cli_errors(self):
        self.write()
        code, text = run(C.main, ["--calibration", self.cal_path, "--checks", self.chk_path, "--svg",
                                  os.path.join(self.tmp, "x.png")])
        self.assertEqual(code, 2)
        self.assertIn(".svg", text)
        code, _text = run(C.main, ["--calibration", os.path.join(self.tmp, "missing.json"), "--checks",
                                   self.chk_path, "--svg", self.svg_path])
        self.assertEqual(code, 2)
        bad = checks()
        bad["order_null"]["seed"] = 9
        with open(self.chk_path, "w", encoding="utf-8") as fh:
            json.dump(bad, fh)
        code, text = run(C.main, ["--calibration", self.cal_path, "--checks", self.chk_path, "--svg", self.svg_path])
        self.assertEqual(code, 2)
        self.assertIn("different nulls", text)


class FigcheckCatches(Base):
    """Each edit breaks exactly the check that should catch it."""

    def setUp(self):
        super().setUp()
        self.write()

    def failing(self, svg_text=None):
        code, text, report = self.figcheck(svg_text)
        self.assertEqual(code, 1, text)
        return sorted(name for name, c in report["checks"].items() if not c["pass"])

    def test_a_changed_printed_number(self):
        self.assertIn(">300 vs 150.5<", self.svg)
        self.assertIn("numbers_match_sources", self.failing(self.svg.replace(">300 vs 150.5<", ">301 vs 150.5<")))

    def test_a_changed_ratio(self):
        edited = re.sub(r">1\.993 · ", ">1.990 · ", self.svg)
        self.assertNotEqual(edited, self.svg)
        self.assertIn("numbers_match_sources", self.failing(edited))

    def test_a_moved_dot(self):
        g = self.rows()["real_cross"]
        cx = next(g.iter(NS + "circle")).get("cx")
        edited = self.svg.replace('class="cis-dot" cx="%s"' % cx, 'class="cis-dot" cx="%.1f"' % (float(cx) + 3))
        self.assertEqual(self.failing(edited), ["geometry_matches"])

    def test_a_widened_band(self):
        g = self.rows()["real_all"]
        w = next(g.iter(NS + "rect")).get("width")
        edited = self.svg.replace('width="%s" height="%d"' % (w, C.BAND_H), 'width="%.1f" height="%d"'
                                  % (float(w) + 5, C.BAND_H), 1)
        self.assertNotEqual(edited, self.svg)
        self.assertIn("geometry_matches", self.failing(edited))

    def test_a_row_on_another_null(self):
        edited = self.svg.replace('data-row="real_cross" data-source="calibration.json" data-path="real.cross_page" '
                                  'data-null="order:agent_day:lag0:default:seed7:n100"',
                                  'data-row="real_cross" data-source="calibration.json" data-path="real.cross_page" '
                                  'data-null="order:agent_day:lag0:default:seed7:n30"')
        self.assertNotEqual(edited, self.svg)
        self.assertIn("same_null", self.failing(edited))

    def test_sources_that_disagree_on_the_null(self):
        chk = checks()
        chk["order_null"]["links_named"]["sd"] = 9.5
        with open(self.chk_path, "w", encoding="utf-8") as fh:
            json.dump(chk, fh, indent=1)
        failing = self.failing()
        self.assertIn("same_null", failing)
        self.assertIn("hashes_match", failing)

    def test_a_split_that_does_not_add_up(self):
        cal = calibration()
        cal["real"]["same_page"]["mean"] = 421.25  # not drawn, so only the null check can see it
        with open(self.cal_path, "w", encoding="utf-8") as fh:
            json.dump(cal, fh, indent=1)
        self.assertEqual(self.failing(), ["hashes_match", "same_null"])

    def test_a_source_changed_after_drawing(self):
        cal = calibration()
        cal["planted"]["carry"]["all_links"]["mean"] = 151.5
        with open(self.cal_path, "w", encoding="utf-8") as fh:
            json.dump(cal, fh, indent=1)
        failing = self.failing()
        for name in ("numbers_match_sources", "geometry_matches", "hashes_match"):
            self.assertIn(name, failing)

    def test_a_dropped_row(self):
        edited = re.sub(r'<g class="cis-row" id="cis-row-real_cross".*?</g>', "", self.svg)
        self.assertNotEqual(edited, self.svg)
        self.assertEqual(self.failing(edited), ["rows_complete"])

    def test_rows_out_of_order(self):
        carry = re.search(r'<g class="cis-row" id="cis-row-carry".*?</g>', self.svg).group(0)
        same = re.search(r'<g class="cis-row" id="cis-row-same_page".*?</g>', self.svg).group(0)
        edited = self.svg.replace(carry, "@@SWAP@@").replace(same, carry).replace("@@SWAP@@", same)
        self.assertIn("rows_complete", self.failing(edited))

    def test_a_decoy_left_out_while_its_numbers_exist(self):
        with open(self.cal_path, "w", encoding="utf-8") as fh:
            json.dump(calibration(cell(50, 50.5, 45.0, 56.0)), fh, indent=1)
        edited = re.sub(r'<g class="cis-row" id="cis-row-page_instruction".*?</g>', "", self.svg)
        self.assertIn("rows_complete", self.failing(edited))

    def test_a_wrong_word(self):
        edited = self.svg.replace("inside the band: blind spot", "above the band: seen")
        self.assertEqual(self.failing(edited), ["words_match"])

    def test_a_z_added(self):
        edited = self.svg.replace(">0.798 · below the band<", ">0.798 · below the band, z -4.1<")
        self.assertNotEqual(edited, self.svg)
        failing = self.failing(edited)
        self.assertIn("no_z", failing)
        self.assertIn("numbers_match_sources", failing)

    def test_an_unexplained_number(self):
        edited = self.svg.replace(">Did it see? The real wiki<", ">Did it see? The real wiki (7,671 tokens)<")
        self.assertNotEqual(edited, self.svg)
        self.assertEqual(self.failing(edited), ["numbers_match_sources"])

    def test_a_pending_row_that_hides_a_number(self):
        with open(self.cal_path, "w", encoding="utf-8") as fh:
            json.dump(calibration(cell(50, 50.5, 45.0, 56.0)), fh, indent=1)
        self.assertIn("numbers_match_sources", self.failing())

    def test_a_wrong_hash(self):
        real = C.sha256_file(self.chk_path)[:16]
        edited = self.svg.replace("checks.json sha256 " + real, "checks.json sha256 " + "0" * 16)
        self.assertIn("hashes_match", self.failing(edited))

    def test_small_text(self):
        edited = self.svg.replace('font-size="17" fill="#898781" class="cis-hash"',
                                  'font-size="12" fill="#898781" class="cis-hash"')
        self.assertEqual(self.failing(edited), ["phone_check"])

    def test_bad_input(self):
        path = os.path.join(self.tmp, "broken.svg")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("<svg")
        code, _text = run(F.main, ["--svg", path, "--sources", self.cal_path, "--out-dir", self.tmp])
        self.assertEqual(code, 2)


@unittest.skipUnless(os.environ.get("PZ_OUT"), "set PZ_OUT to the folder with the real calibration.json, checks.json")
class CommittedFigure(unittest.TestCase):
    def setUp(self):
        self.out = os.environ["PZ_OUT"]
        self.cal = os.path.join(self.out, "calibration.json")
        self.chk = os.path.join(self.out, "checks.json")
        self.svg = os.path.join(ROOT, "docs", "can-it-see.svg")

    def test_redraw_matches_the_committed_figure(self):
        with open(self.cal, encoding="utf-8") as fh:
            cal = json.load(fh)
        with open(self.chk, encoding="utf-8") as fh:
            chk = json.load(fh)
        hashes = {"calibration.json": C.sha256_file(self.cal), "checks.json": C.sha256_file(self.chk)}
        with open(self.svg, encoding="utf-8") as fh:
            self.assertEqual(C.draw(cal, chk, hashes), fh.read())

    def test_figcheck_passes_on_the_committed_figure(self):
        tmp = tempfile.mkdtemp(prefix="pz-canisee-real-")
        try:
            code, text = run(F.main, ["--svg", self.svg, "--sources", self.cal, self.chk, "--out-dir", tmp])
            self.assertEqual(code, 0, text)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipUnless(os.environ.get("PZ_DATA") and os.environ.get("PZ_TREES"),
                         "set PZ_DATA and PZ_TREES for the label and token scan")
    def test_no_label_and_no_token_text(self):
        scan = _load("labelscan")
        code, text = run(scan.main, ["--data", os.environ["PZ_DATA"], "--trees", os.environ["PZ_TREES"], self.svg])
        self.assertEqual(code, 0, text.splitlines()[-1])


def with_control(control_cell, contact=False, all_cell=None):
    doc = calibration(cell(50, 50.5, 45.0, 56.0))
    doc["controls"] = {"random_400": {"contact": contact, "tokens": 400, "cross_page": control_cell,
                                      "all_links": all_cell or cell(60, 64.5, 61.0, 68.0)}}
    return doc


class ControlRow(Base):
    """The random-row control beside the real cross-page row (no-spread tokens, scored the same way)."""

    def test_words(self):
        below = {r["row"].key: r for r in C.build_rows(with_control(cell(30, 35.0, 31.0, 39.0)), checks())}
        self.assertEqual(below["random_cross"]["role"], "control")
        self.assertEqual(below["random_cross"]["verdict"], "below the band: biased low")
        inside = {r["row"].key: r for r in C.build_rows(with_control(cell(35, 35.0, 31.0, 39.0)), checks())}
        self.assertEqual(inside["random_cross"]["verdict"], "inside the band: unbiased")
        above = {r["row"].key: r for r in C.build_rows(with_control(cell(45, 35.0, 31.0, 39.0)), checks())}
        self.assertEqual(above["random_cross"]["verdict"], "above the band: biased high")

    def test_all_links_control_words(self):
        rows = {r["row"].key: r for r in C.build_rows(
            with_control(cell(30, 35.0, 31.0, 39.0), all_cell=cell(64, 64.5, 61.0, 68.0)), checks())}
        self.assertEqual(rows["random_all"]["verdict"], "inside the band: unbiased")
        rows = {r["row"].key: r for r in C.build_rows(with_control(cell(30, 35.0, 31.0, 39.0)), checks())}
        self.assertEqual(rows["random_all"]["verdict"], "below the band: biased low")

    def test_a_control_with_contact_is_refused(self):
        with self.assertRaises(C.CanISeeError):
            C.build_rows(with_control(cell(30, 35.0, 31.0, 39.0), contact=True), checks())

    def test_figcheck_passes_with_every_row_drawn(self):
        self.write(cal=with_control(cell(30, 35.0, 31.0, 39.0)))
        rows = self.rows()
        self.assertEqual([k for k in rows], ["carry", "same_page", "page_instruction", "real_all", "random_all",
                                             "real_cross", "random_cross"])
        self.assertEqual(rows["random_all"].get("data-status"), "drawn")
        self.assertEqual(rows["random_cross"].get("data-status"), "drawn")
        code, text, report = self.figcheck()
        self.assertEqual(code, 0, text)
        self.assertEqual(report["checks"]["numbers_match_sources"]["rows_drawn"], 7)
        self.assertEqual(report["checks"]["numbers_match_sources"]["rows_pending"], 0)
        self.assertIn("biased low", self.svg)

    def test_figcheck_catches_a_wrong_control_word(self):
        self.write(cal=with_control(cell(30, 35.0, 31.0, 39.0)))
        edited = self.svg.replace("biased low", "unbiased", 1)
        self.assertNotEqual(edited, self.svg)
        with open(self.svg_path, "w", encoding="utf-8") as fh:
            fh.write(edited)
        code, text, report = self.figcheck()
        self.assertNotEqual(code, 0, text)
        self.assertFalse(report["checks"]["words_match"]["pass"])

    def test_figcheck_catches_a_dropped_control_row_that_has_numbers(self):
        self.write(cal=with_control(cell(30, 35.0, 31.0, 39.0)))
        edited = re.sub(r'<g class="cis-row" id="cis-row-random_cross".*?</g>', "", self.svg, flags=re.S)
        self.assertNotEqual(edited, self.svg)
        with open(self.svg_path, "w", encoding="utf-8") as fh:
            fh.write(edited)
        code, text, report = self.figcheck()
        self.assertNotEqual(code, 0, text)
        self.assertFalse(report["checks"]["rows_complete"]["pass"])


if __name__ == "__main__":
    unittest.main()
