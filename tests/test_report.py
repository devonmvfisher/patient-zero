"""Tests for the report (patient_zero/report.py) and the figure check (tools/figcheck.py).

Standard library unittest only. Everything runs on the synthetic example log
or on made-up trees. The browser render runs only when PZ_FIGCHECK=1 (it
starts a headless Chrome or Edge and takes about a minute). Failure messages
carry counts only, never names or text from the data.
"""

import importlib.util
import io
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from patient_zero import pipeline as P  # noqa: E402
from patient_zero import report as R  # noqa: E402

EXAMPLE = os.path.join(ROOT, "examples", "fake_log.ndjson")


def _load_figcheck():
    spec = importlib.util.spec_from_file_location("figcheck", os.path.join(ROOT, "tools", "figcheck.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = _load_figcheck()


def run(main, args):
    buf = io.StringIO()
    return main(args, out=buf), buf.getvalue()


def inline_svgs(page):
    """Each inline <svg> of a page, parsed as XML (namespace added)."""
    out = []
    for chunk in re.findall(r"<svg\b.*?</svg>", page, flags=re.S):
        out.append(ET.fromstring(chunk.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)))
    return out


def rec(tid, branch, links, mean=1.0, ge=0, agents=None):
    return {"token_id": tid, "largest_branch": branch, "links_named": links, "shuffle_links_mean": mean,
            "shuffle_ge_observed": ge, "agents": agents or branch}


def node(n, t, family, status, parent=None, agent=None):
    return {"n": n, "agent": agent or "agent-%d" % n, "t_min": float(t), "family": family, "group": "g",
            "status": status, "parent": parent, "lag_min": None}


def detail(nodes, branch_root=0, seed=0, token="host:secret.example", tid="abcdef012345"):
    status = {s: sum(1 for nd in nodes if nd["status"] == s) for s in R.STATUS_NAMES}
    links = sum(1 for nd in nodes if nd["parent"] is not None)
    return {"token_id": tid, "token": token, "cls": "host", "kind": "host", "agents": len(nodes), "status": status,
            "links_named": links, "links_unnamed": 0, "depth": 2, "largest_branch": 3, "uses": len(nodes),
            "places": 2, "first_use": "2026-01-01T00:00:00Z", "span_hours": 1.0, "risky": [],
            "seed_branch": 3, "seed_min_to_3rd": None, "seed_min_to_10th": None, "branch_min_to_3rd": None,
            "branch_min_to_10th": None, "shuffle_links_mean": 1.5, "shuffle_ge_observed": 3,
            "first": 0, "seed": seed, "branch_root": branch_root, "nodes": nodes}


class ExampleReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-report-")
        cls.out = os.path.join(cls.tmp, "run")
        cls.code, cls.text = run(P.main, ["--adapter", "generic", "--input", EXAMPLE, "--out", cls.out])
        with open(os.path.join(cls.out, "report.html"), encoding="utf-8") as fh:
            cls.page = fh.read()
        cls.facts, cls.records, cls.detail = R.load_outputs(cls.out)
        cls.svg_path = os.path.join(cls.tmp, "docs", "lead.svg")
        cls.code_svg, cls.text_svg = run(R.main, ["--out", cls.out, "--svg", cls.svg_path])
        with open(cls.svg_path, encoding="utf-8") as fh:
            cls.svg = fh.read()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_files_and_facts(self):
        self.assertEqual(self.code, 0, self.text)
        self.assertEqual(self.code_svg, 0, self.text_svg)
        for name in ("facts.json", "trees.jsonl", "top_trees.json", "report.html"):
            self.assertTrue(os.path.isfile(os.path.join(self.out, name)))
        self.assertEqual(self.facts["facts_version"], P.FACTS_VERSION)
        top = self.facts["outputs"]["top_trees.json"]
        self.assertEqual(top["trees"], len(self.detail["trees"]))
        self.assertEqual(top["lead"], self.detail["lead"])
        self.assertIn("lead tree: %s" % top["lead"], self.text)
        self.assertIsNone(self.facts["params"]["lead"])

    def test_page_is_offline(self):
        low = self.page.lower()
        for bad in ("<script", "http://", "https://", " src=", "<link", "url(", "@import"):
            self.assertNotIn(bad, low)
        self.assertTrue(self.page.startswith("<!doctype html>"))
        self.assertGreaterEqual(len(inline_svgs(self.page)), 4)  # lead, speed, shuffle, status
        for heading in ("Lead spread tree", "Spread speed", "Agent influence", "Observed against the shuffle",
                        "Parallel starts", "Method and limits"):
            self.assertIn(heading, self.page)

    def test_no_human_label_and_no_row_text(self):
        both = self.page + json.dumps(self.detail)
        self.assertNotIn("Host", both)  # the example's human speaker
        self.assertNotIn("Morning plan", both)
        self.assertNotIn("reading list link is broken", both)

    def test_detail_matches_records(self):
        by_id = {r["token_id"]: r for r in self.records}
        for tree in self.detail["trees"]:
            r = by_id[tree["token_id"]]
            self.assertEqual(len(tree["nodes"]), r["agents"])
            self.assertEqual(sum(1 for nd in tree["nodes"] if nd["parent"] is not None), r["links_named"])
            for nd in tree["nodes"]:
                if nd["parent"] is not None:
                    self.assertLess(nd["parent"], nd["n"])
                    self.assertGreaterEqual(nd["t_min"], tree["nodes"][nd["parent"]]["t_min"])
            self.assertEqual(tree["nodes"][0]["t_min"], 0.0)

    def test_deterministic_and_rerender_matches(self):
        again = R.render_report(self.facts, self.records, self.detail)
        self.assertEqual(again, self.page)
        self.assertEqual(R.public_svg(self.detail, facts=self.facts), self.svg)

    def test_verdict_on_top_and_in_the_cli(self):
        word, sentence = R.chance_verdict(self.facts["slots"]["F5"])
        self.assertIn(word, ("above chance", "below chance", "within chance"))
        self.assertIn('<p class="verdict">', self.page)
        self.assertLess(self.page.index('class="verdict"'), self.page.index("Lead spread tree</h2>"))
        self.assertIn("against the shuffle: %s" % word, self.text)
        self.assertNotIn("None min", self.text)

    def test_public_svg_says_how_the_tree_was_picked(self):
        for line in R.selection_lines(self.detail["lead_rule"], self.facts):
            self.assertIn(R._e(line), self.svg)
        self.assertTrue(R.selection_lines(self.detail["lead_rule"], self.facts))

    def test_public_svg_has_no_labels_or_token_text(self):
        root = ET.fromstring(self.svg.split("\n", 1)[1])
        self.assertTrue(root.tag.endswith("svg"))
        agents = set(nd["agent"] for t in self.detail["trees"] for nd in t["nodes"])
        self.assertTrue(agents)
        for agent in agents:
            self.assertNotIn(agent, self.svg)
        lead = next(t for t in self.detail["trees"] if t["token_id"] == self.detail["lead"])
        self.assertNotIn(lead["token"].split(":")[-1], self.svg)
        self.assertIn(lead["token_id"], self.svg)
        self.assertNotIn("<title>", self.svg.split("</title>", 1)[1])  # no per-node tooltips

    def test_public_svg_passes_source_checks(self):
        checks = F.source_checks(F.parse_svg(self.svg_path))
        for name in F.PLANNED:
            self.assertTrue(checks[name]["pass"], name)

    def test_report_lead_override(self):
        other = [t["token_id"] for t in self.detail["trees"] if t["token_id"] != self.detail["lead"]]
        self.assertTrue(other)
        code, text = run(R.main, ["--out", self.out, "--lead", other[0]])
        self.assertEqual(code, 0, text)
        with open(os.path.join(self.out, "report.html"), encoding="utf-8") as fh:
            page = fh.read()
        self.assertIn(R.RULE_GIVEN, page)
        self.assertIn(other[0], page)
        code, text = run(R.main, ["--out", self.out, "--lead", "000000000000"])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", text)
        run(R.main, ["--out", self.out])  # restore the default page for other tests
        with open(os.path.join(self.out, "report.html"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), self.page)

    def test_cli_lead(self):
        out = os.path.join(self.tmp, "lead-run")
        last = R._linked(self.records)[-1]["token_id"]
        code, text = run(P.main, ["--adapter", "generic", "--input", EXAMPLE, "--out", out, "--lead", last])
        self.assertEqual(code, 0, text)
        facts, _records, det = R.load_outputs(out)
        self.assertEqual(det["lead"], last)
        self.assertEqual(det["lead_rule"], R.RULE_GIVEN)
        self.assertEqual(facts["params"]["lead"], last)
        self.assertNotEqual(facts["core_sha256"], self.facts["core_sha256"])
        code, text = run(P.main, ["--adapter", "generic", "--input", EXAMPLE, "--out", out, "--lead", "nope"])
        self.assertEqual(code, 2)

    def test_stale_outputs_refused(self):
        copy = os.path.join(self.tmp, "stale")
        shutil.copytree(self.out, copy)
        with open(os.path.join(copy, "trees.jsonl"), "a", encoding="utf-8") as fh:
            fh.write("\n")
        with self.assertRaises(ValueError):
            R.load_outputs(copy)
        code, text = run(R.main, ["--out", copy])
        self.assertEqual(code, 2)

    def test_svg_flag_must_be_svg(self):
        code, text = run(R.main, ["--out", self.out, "--svg", os.path.join(self.tmp, "lead.png")])
        self.assertEqual(code, 2)


class Picking(unittest.TestCase):
    def test_lead_prefers_trees_that_beat_every_shuffle(self):
        records = [rec("aaa", 10, 20, ge=40), rec("bbb", 5, 6, ge=0), rec("ccc", 5, 8, ge=0), rec("ddd", 0, 0)]
        self.assertEqual(R.pick_lead(records), ("ccc", R.RULE_BEAT))  # ties on branch go to more links
        self.assertEqual(R.pick_scale(records), "aaa")
        self.assertEqual(R.pick_lead(records, "aaa"), ("aaa", R.RULE_GIVEN))
        with self.assertRaises(ValueError):
            R.pick_lead(records, "zzz")

    def test_lead_falls_back_without_a_shuffle(self):
        records = [rec("aaa", 3, 2, mean=None, ge=None), rec("bbb", 4, 3, mean=None, ge=None)]
        self.assertEqual(R.pick_lead(records), ("bbb", R.RULE_LARGEST))
        self.assertEqual(R.pick_lead([rec("x", 1, 0)]), (None, R.RULE_NONE))
        self.assertIsNone(R.pick_scale([]))

    def test_detail_ids(self):
        records = [rec("t%02d" % k, 50 - k, 60 - k, ge=(0 if k % 2 else 5)) for k in range(40)]
        ids = R.detail_ids(records, "t39")
        self.assertEqual(ids[:2], ["t39", "t00"])
        self.assertEqual(len(ids), len(set(ids)))
        beaters = [r["token_id"] for r in R._linked(records) if R.beats_every_shuffle(r)]
        for tid in beaters[:R.DETAIL_TOP]:
            self.assertIn(tid, ids)


class Drawing(unittest.TestCase):
    def test_tree_svg_all_statuses_and_other_families(self):
        nodes = [node(0, 0, "fa", "first"), node(1, 5, "fb", "linked", 0), node(2, 9, "fc", "linked", 1),
                 node(3, 12, "fd", "parallel"), node(4, 20, "fe", "unnamed"), node(5, 30, "fa", "clock_dropped"),
                 node(6, 31, "fa", "linked", 3)]
        d = detail(nodes, branch_root=0)
        svg = R.svg_tree(d, 100, public=True)
        root = ET.fromstring(svg.split("\n", 1)[1])
        marks = [el for el in root.iter() if (el.get("class") or "") == "node"]
        self.assertEqual(len(marks), len(nodes))
        self.assertIn("2 other page families", svg)
        self.assertNotIn("secret.example", svg)
        self.assertNotIn("agent-1", svg)
        self.assertIn("parallel start", svg)
        self.assertIn("dropped by the clock rule", svg)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.svg")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(svg)
            checks = F.source_checks(F.parse_svg(path))
        self.assertTrue(all(checks[name]["pass"] for name in F.PLANNED))
        private = R.svg_tree(d, 100, public=False, standalone=False)
        self.assertIn("secret.example", private)
        self.assertIn("agent-1", private)

    def test_big_tree_draws_the_largest_branch(self):
        nodes = [node(0, 0, "fa", "first")]
        nodes += [node(k, k, "fa", "parallel") for k in range(1, 100)]  # lone parallel starts
        nodes += [node(100, 100, "fb", "parallel")]
        nodes += [node(100 + k, 100 + k, "fb", "linked", 100 + k - 1) for k in range(1, 120)]
        d = detail(nodes, branch_root=100)
        picked, mode = R.drawn_nodes(d)
        self.assertEqual(mode, "branch")
        self.assertEqual(picked, list(range(100, 220)))
        svg = R.svg_tree(d, 0, public=True)
        self.assertIn("the largest branch, 120 of 220 adopters", svg)
        self.assertIn("no label shuffle", svg)

    def test_time_axis(self):
        scale, ticks, label = R.time_axis([0, 10, 20, 45])
        self.assertIn("minutes", label)
        self.assertAlmostEqual(scale(ticks[-1][0]), 1.0)
        self.assertGreaterEqual(ticks[-1][0], 45)
        scale, ticks, label = R.time_axis([0, 1, 2, 3, 4, 5000])
        self.assertIn("log scale", label)
        self.assertEqual(scale(0), 0.0)
        self.assertLessEqual(scale(5000), 1.0)
        scale, ticks, label = R.time_axis([0, 300, 600, 900])
        self.assertIn("hours", label)
        self.assertEqual(R.time_axis([])[1][0], (0, "0"))

    def test_colours_follow_use_count(self):
        d = detail([node(0, 0, "b", "first"), node(1, 1, "a", "linked", 0), node(2, 2, "a", "linked", 1)])
        colour, legend = R.tree_colours(d, [0, 1, 2])
        self.assertEqual(colour["a"], R.SERIES[0])
        self.assertEqual(colour["b"], R.SERIES[1])
        self.assertEqual(len(legend), 2)


class FigcheckParts(unittest.TestCase):
    @staticmethod
    def png(width, rows, colour_type=2, filters=None):
        bpp = 3 if colour_type == 2 else 4
        raw = b""
        prev = bytes(width * bpp)
        for y, row in enumerate(rows):
            f = (filters or [0] * len(rows))[y]
            line = bytes(row)
            if f == 2:  # up
                line = bytes((line[x] - prev[x]) & 255 for x in range(len(line)))
            elif f == 1:  # sub
                line = bytes((line[x] - (line[x - bpp] if x >= bpp else 0)) & 255 for x in range(len(line)))
            raw += bytes([f]) + line
            prev = bytes(row)

        def chunk(kind, body):
            return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

        head = struct.pack(">IIBBBBB", width, len(rows), 8, colour_type, 0, 0, 0)
        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")

    def test_read_png_filters(self):
        rows = [[10, 20, 30, 40, 50, 60], [70, 80, 90, 1, 2, 3], [255, 0, 128, 7, 8, 9]]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.png")
            with open(path, "wb") as fh:
                fh.write(self.png(2, rows, filters=[0, 2, 1]))
            img = F.read_png(path)
        self.assertEqual(img[:3], (2, 3, 3))
        self.assertEqual(F.pixel(img, 1, 1), (1, 2, 3))
        self.assertEqual(F.pixel(img, 0, 2), (255, 0, 128))

    def test_text_box_and_hide_text(self):
        el = ET.fromstring('<text x="30" y="200" font-size="10" text-anchor="middle" '
                           'transform="rotate(-90 30 200)">abcdefghij</text>')
        x0, y0, x1, y1 = F.text_box(el)
        self.assertLess(x1 - x0, y1 - y0)  # rotated: taller than wide
        self.assertTrue(y0 < 200 < y1)
        hidden = F.hide_text('<svg width="1"><text>a</text></svg>')
        self.assertIn("<style>text{visibility:hidden}</style>", hidden)

    def test_source_checks_catch_a_missing_swatch(self):
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><title>t</title>'
               '<text id="a-title" x="1" y="10">T</text><text id="a-x-label" x="1" y="90">x</text>'
               '<text id="a-y-label" x="5" y="50">y</text>'
               '<rect class="legend-swatch" x="1" y="20" width="12" height="12" fill="#2a78d6"/>'
               '<circle class="node" cx="50" cy="50" r="4" fill="#2a78d6"/>'
               '<circle class="node" cx="60" cy="50" r="4" fill="#fcfcfb" stroke="#eb6834"/></svg>')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.svg")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(svg)
            checks = F.source_checks(F.parse_svg(path))
        self.assertTrue(checks["title_present"]["pass"])
        self.assertTrue(checks["axis_labels_present"]["pass"])
        self.assertFalse(checks["legend_colours_match"]["pass"])
        self.assertEqual(checks["legend_colours_match"]["mark_colours"], ["#2a78d6", "#eb6834"])


def f5(observed, mean, p05, p95, ge, shuffles=100, share=(0.4, 0.5)):
    return {"links_observed": observed, "links_expected_mean": mean, "links_expected_p05": p05,
            "links_expected_p95": p95, "shuffles_ge_observed": ge, "shuffles": shuffles, "scope": "day",
            "linked_share_observed": share[0], "linked_share_expected_mean": share[1]}


class Verdict(unittest.TestCase):
    """The plain-words comparison with the shuffle, on made-up numbers."""

    def test_below_above_within(self):
        word, text = R.chance_verdict(f5(28250, 45188.7, 44684.2, 45632.2, 100))
        self.assertEqual(word, "below chance")
        self.assertIn("100 of 100 shuffles", text)
        self.assertIn("not evidence of spread beyond chance", text)
        self.assertIn("within each UTC day", text)
        self.assertIn("does not show that agents avoid each other's tricks", text)
        self.assertIn("patient_zero.checks", text)
        word, text = R.chance_verdict(f5(80, 40.0, 35.0, 46.0, 0))
        self.assertEqual(word, "above chance")
        self.assertIn("hand-check", text)
        self.assertNotIn("avoid each other's tricks", text)
        word, text = R.chance_verdict(f5(7, 5.5, 4.0, 7.0, 14))
        self.assertEqual(word, "within chance")  # on the 95th percentile is still inside
        self.assertIn("not evidence of spread beyond chance", text)

    def test_no_shuffle(self):
        self.assertEqual(R.chance_verdict({}), (None, None))
        self.assertEqual(R.chance_verdict(f5(1, 1.0, 1.0, 1.0, 0, shuffles=0)), (None, None))

    def test_share_line_is_optional(self):
        word, text = R.chance_verdict(f5(7, 5.5, 4.0, 7.0, 14, share=(None, None)))
        self.assertNotIn("Share of later adopters", text)

    def test_selection_lines(self):
        facts = {"nullmodel": {"shuffles": 100, "trees_beating_every_shuffle": 42,
                               "chance_beats_upper_bound": 75.95, "tokens": 7671}}
        lines = R.selection_lines(R.RULE_BEAT, facts)
        self.assertIn("largest of 42 trees that beat every shuffle", lines[0])
        self.assertIn("up to 76 such trees among 7,671", lines[0])
        self.assertIn("not proof", lines[1])
        small = {"nullmodel": {"shuffles": 100, "trees_beating_every_shuffle": 2,
                               "chance_beats_upper_bound": 0.03, "tokens": 3}}
        lines = R.selection_lines(R.RULE_BEAT, small)
        self.assertIn("up to 0.03 such trees among 3", lines[0])  # 2 decimals under 1, never "0.0"
        self.assertNotIn("not proof", " ".join(lines))  # 2 trees beat every shuffle, more than 0.03
        self.assertIn("2 or more with a chance of at most 0.015", lines[1])  # Markov: 0.0297 / 2
        self.assertIn("not proof", R.selection_lines(R.RULE_BEAT, {"nullmodel": dict(
            small["nullmodel"], trees_beating_every_shuffle=0)})[1])
        self.assertIn("no tree beat every shuffle", R.selection_lines(R.RULE_LARGEST, None)[0])
        self.assertEqual(R.selection_lines(R.RULE_GIVEN, None), ["Picked by hand (--lead)."])
        self.assertEqual(R.selection_lines(R.RULE_NONE, None), [])

    def test_bound_text_never_reads_zero(self):
        self.assertEqual(R.bound_text(75.95), "76")
        self.assertEqual(R.bound_text(7.6), "7.6")
        self.assertEqual(R.bound_text(0.0297), "0.03")
        self.assertEqual(R.bound_text(0.5), "0.50")
        self.assertEqual(R.bound_text(3 / 1001), "0.003")  # would round to 0.00
        self.assertEqual(R.bound_text(None), "n/a")
        # recomputed from the counts: facts.json may have rounded a tiny bound to 0.0
        self.assertAlmostEqual(R.beat_bound({"shuffles": 1000, "tokens": 3,
                                             "chance_beats_upper_bound": 0.0}), 3 / 1001)

    def test_lead_text_drops_not_proof_only_beyond_the_bound(self):
        real = {"nullmodel": {"shuffles": 100, "trees_beating_every_shuffle": 42,
                              "chance_beats_upper_bound": 75.95, "tokens": 7671}}
        text = R._lead_text(R.RULE_BEAT, real)
        self.assertIn("chance alone would let up to 76 trees do that", text)
        self.assertIn("not proof that spread beat chance", text)
        small = {"nullmodel": {"shuffles": 100, "trees_beating_every_shuffle": 2,
                               "chance_beats_upper_bound": 0.03, "tokens": 3}}
        text = R._lead_text(R.RULE_BEAT, small)
        self.assertIn("up to 0.03 trees do that (tokens / (shuffles + 1)) on average", text)
        self.assertIn("2 or more with a chance of at most 0.015", text)
        self.assertNotIn("not proof", text)
        narrow = {"nullmodel": {"shuffles": 100, "trees_beating_every_shuffle": 77,
                                "chance_beats_upper_bound": 75.95, "tokens": 7671}}
        self.assertIn("at most 0.99", R._lead_text(R.RULE_BEAT, narrow))  # a narrow win reads as narrow

    def test_note_lines_sit_in_the_public_figure(self):
        d = detail([node(0, 0, "a", "first"), node(1, 5, "a", "linked", 0), node(2, 9, "a", "linked", 1)])
        svg = R.svg_tree(d, 100, public=True, note_lines=["Picked by hand (--lead)."])
        self.assertIn("Picked by hand (--lead).", svg)
        ET.fromstring(svg.split("\n", 1)[1] if svg.startswith("<?xml") else svg)


@unittest.skipUnless(os.environ.get("PZ_FIGCHECK") == "1", "set PZ_FIGCHECK=1 to render with a headless browser")
class FigcheckRender(unittest.TestCase):
    def test_render_passes_on_the_example_figure(self):
        tmp = tempfile.mkdtemp(prefix="pz-figcheck-test-")
        try:
            out = os.path.join(tmp, "run")
            self.assertEqual(run(P.main, ["--adapter", "generic", "--input", EXAMPLE, "--out", out])[0], 0)
            svg = os.path.join(tmp, "lead.svg")
            self.assertEqual(run(R.main, ["--out", out, "--svg", svg])[0], 0)
            code, text = run(F.main, ["--svg", svg, "--out-dir", tmp])
            self.assertEqual(code, 0, text)
            with open(os.path.join(tmp, "lead-checks.json"), encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["status"], "PASS")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
