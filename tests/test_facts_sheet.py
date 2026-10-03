"""Tests for tools/facts_sheet.py: slots F1-F14, findings A-C, stale-file guards, HTML escaping.

unittest, standard library only. Every number, name and token here is made up.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, os.path.join(ROOT, "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

import facts_sheet as FS  # noqa: E402

PLANTED = {"cascades_planted": 3, "cascades_found": 3, "inherited_token_live": True, "inherited_false_links": 0,
           "wiki_fixture_links": 0, "parallel_flagged": 2, "parallel_planted": 2}
NOW = {"commit": "abc1234", "dirty": False, "code_lines": 100, "test_functions": 10}
BAD_LABEL = "<script>alert(1)</script>"


def stat(observed, mean, sd, ge, shuffles=100):
    return {"observed": observed, "mean": mean, "sd": sd, "p05": mean - 2 * sd, "p95": mean + 2 * sd,
            "ratio": round(observed / mean, 3), "z": round((observed - mean) / sd, 2),
            "shuffles_ge_observed": ge}


def make_facts():
    return {
        "stage": "test", "facts_version": 3, "core_sha256": "c0ffee",
        "code": {"commit": "abc1234", "dirty": False}, "runtime": {"target_seconds": 300},
        "load": {"rows": 500}, "tokens": {"kept_total": 40},
        "cascade": {"adoptions": 300, "status": {"first": 40, "linked": 100, "parallel": 150, "unnamed": 5,
                                                 "clock_dropped": 5}},
        "trees": {"largest_branch_max": 12},
        "nullmodel": {"tokens": 40, "trees_beating_every_shuffle": 2, "chance_beats_upper_bound": 0.4},
        "risky": {"narrow_types": ["bypass"], "trees_with_type": {"bypass": 3, "answer": 20},
                  "trees_with_narrow_type": 3, "trees_with_narrow_type_and_named_links": 2,
                  "named_links_vs_shuffle": {"narrow": {"observed": 10, "mean": 12.0, "ratio": 0.833}}},
        "slots": {
            "F3": {"candidate_tokens": 900, "tokens_3plus_agents": 40,
                   "tokens_3plus_agents_by_class": {"host": 10, "phrase": 30}},
            "F4": {"links_named": 100, "links_from_exposure_only_rows": 5},
            "F5": {"links_observed": 100, "links_expected_mean": 160.0, "links_expected_sd": 5.0,
                   "links_expected_p05": 152.0, "links_expected_p95": 168.0, "ratio_observed_to_expected": 0.625,
                   "shuffles_ge_observed": 100, "shuffles": 100, "seed": 7, "scope": "day",
                   "adopters_observed": 300, "adopters_expected_mean": 380.0, "linked_share_observed": 0.385,
                   "linked_share_expected_mean": 0.47, "linked_share_ratio": 0.819},
            "F6": {"tokens_5plus_agents": 12, "tokens_10plus_agents": 3, "branches_5plus_agents": 4,
                   "branches_10plus_agents": 1},
            "F7": {"median_minutes_to_10th": 50.0, "trees_reaching_10_in_seed_branch": 2,
                   "median_minutes_to_3rd": 9.0, "trees_reaching_3_in_seed_branch": 9,
                   "fastest_minutes_to_10th": 0.5},
            "F8": {"parallel_starts": 150, "trees_with_effective_seed": 30, "seed_is_not_first_user": 10},
            "F9": {"links_dropped_by_clock": 5, "links_shifted_by_clock": 7, "tokens_dropped_by_clock_rule": 0},
            "F14": {"runtime_seconds": 12.3, "code_lines": 100, "test_functions": 10},
        },
    }


def make_top():
    nodes = [
        {"agent": BAD_LABEL, "family": "fam-a", "status": "first", "t_min": 0.0},
        {"agent": "Bee", "family": "fam-a", "status": "linked", "t_min": 5.0},
        {"agent": "Cat", "family": "fam-b", "status": "parallel", "t_min": 120.0},
        {"agent": "Dot", "family": "fam-a", "status": "linked", "t_min": 130.0},
    ]
    tree = {"token_id": "0123456789ab", "token": "host:example.test", "cls": "host", "agents": 4, "uses": 5,
            "places": 3, "span_hours": 2.2, "links_named": 6, "shuffle_links_mean": 2.0, "shuffle_ge_observed": 0,
            "status": {"first": 1, "linked": 2, "parallel": 1}, "seed": 0, "first": 0, "branch_root": 2,
            "seed_branch": 2, "seed_min_to_3rd": None, "seed_min_to_10th": None, "largest_branch": 2,
            "branch_min_to_3rd": None, "branch_min_to_10th": None, "depth": 1, "risky": ["bypass"], "nodes": nodes}
    return {"lead": "0123456789ab", "lead_rule": "test rule", "shuffles": 100, "trees": [tree], "version": 1}


def make_checks():
    return {
        "tool": "patient-zero checks", "rows": 500, "tokens": 40,
        "observed": {"default_min_lag_0s": {"links_named": 100}, "default_min_lag_120s": {"links_named": 90},
                     "strict_min_lag_0s": {"links_named": 30}},
        "order_null": {"shuffles": 100, "seed": 7, "links_named": stat(100, 101.0, 2.0, 70),
                       "by_class": {"host": {"ratio": 0.98}, "phrase": {"ratio": 1.0}},
                       "trees_above_every_shuffle": 1, "trees_below_every_shuffle": 1, "chance_bound_each_way": 0.4,
                       "lead": {"token_id": "0123456789ab", "observed": 6, "mean": 3.0, "shuffles_ge_observed": 0}},
        "order_null_strict": {"shuffles": 30, "links_named": stat(30, 40.0, 2.0, 30, 30),
                              "lead": {"token_id": "0123456789ab", "observed": 4, "mean": 2.0,
                                       "shuffles_ge_observed": 1}},
        "order_null_min_lag_120s": {"shuffles": 30, "links_named": stat(90, 95.0, 1.5, 30, 30),
                                    "lead": {"token_id": "0123456789ab", "observed": 5, "mean": 3.0,
                                             "shuffles_ge_observed": 2}},
    }


def write_json(folder, name, obj):
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh)
    return path


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class FactsSheet(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pz-facts-")
        self.facts = make_facts()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_all(self, checks=True):
        top_path = write_json(self.tmp, "top_trees.json", make_top())
        self.facts["outputs"] = {"top_trees.json": {"sha256": sha(top_path)}}
        write_json(self.tmp, "facts.json", self.facts)
        if checks:
            write_json(self.tmp, "checks.json", make_checks())

    def build(self, **kw):
        return FS.build(self.tmp, label="v9", stamp="2026-01-01 00:00", planted=PLANTED, now=NOW, **kw)

    def test_all_slots_and_three_findings(self):
        self.write_all()
        md, page, summary = self.build()
        self.assertIsNone(page)
        for slot in FS.SLOT_IDS:
            self.assertIn("| %s |" % slot, md)
        for key in ("A", "B", "C"):
            self.assertIn("### %s. " % key, md)
        self.assertEqual(md.count("- Limit: "), 3)
        self.assertEqual(summary["warnings"], [])

    def test_silence_pick_is_highest_primary_ratio(self):
        self.write_all()
        _md, _page, summary = self.build()
        self.assertEqual(summary["pick"], "B")  # lead 6 against an order-null mean of 3.0
        self.assertAlmostEqual(summary["ratios"]["B"], 2.0)
        self.assertAlmostEqual(summary["ratios"]["A"], 0.99)
        self.assertAlmostEqual(summary["ratios"]["C"], 0.75)
        self.assertIn("within chance", summary["titles"]["A"])
        self.assertIn("below chance", summary["titles"]["C"])

    def test_rules_for_the_numbers(self):
        self.write_all()
        md, _page, _summary = self.build()
        f4 = next(line for line in md.splitlines() if line.startswith("| F4 |"))
        self.assertIn("30 when the adopting save does not count as a visit", f4)
        self.assertIn("90 with a 120 s", f4)
        f5 = next(line for line in md.splitlines() if line.startswith("| F5 |"))
        self.assertIn("380 adopters against 300 observed", f5)
        self.assertIn("added after the planned null's result", f5)
        f7 = next(line for line in md.splitlines() if line.startswith("| F7 |"))
        self.assertIn("withheld", f7)
        self.assertNotIn("0.5", f7)  # the fastest value is not printed

    def test_without_checks_falls_back_to_planned_shuffle(self):
        self.write_all(checks=False)
        md, _page, summary = self.build()
        self.assertTrue(any("no checks.json" in w for w in summary["warnings"]))
        self.assertAlmostEqual(summary["ratios"]["A"], 0.625)
        self.assertIn("below chance", summary["titles"]["A"])
        self.assertIn("Parallel starts outnumber links", summary["titles"]["C"])

    def test_mismatched_checks_are_not_used(self):
        self.write_all()
        bad = make_checks()
        bad["rows"] = 499
        write_json(self.tmp, "checks.json", bad)
        _md, _page, summary = self.build()
        self.assertTrue(any("checks.json not used" in w for w in summary["warnings"]))

    def test_stale_top_trees_refused(self):
        self.write_all()
        top = make_top()
        top["trees"][0]["agents"] = 5
        write_json(self.tmp, "top_trees.json", top)
        with self.assertRaises(FS.SheetError):
            self.build()

    def test_missing_facts_refused(self):
        with self.assertRaises(FS.SheetError):
            self.build()

    def test_stale_hand_check_dropped(self):
        self.facts["hand_check"] = {"plausible": 2, "sample": 10, "facts_core_sha256_at_check": "other"}
        self.write_all()
        md, _page, summary = self.build()
        self.assertTrue(any("hand_check not used" in w for w in summary["warnings"]))
        self.assertIn("| F10 | Hand check result, X of 10 plausible | not done |", md)

    def test_html_escapes_data_and_has_no_sentences_promise(self):
        self.write_all()
        svg = os.path.join(self.tmp, "lead.svg")
        with open(svg, "w", encoding="utf-8") as fh:
            fh.write('<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="10" height="5" '
                     'viewBox="0 0 10 5"><title>t</title></svg>')
        plan = os.path.join(self.tmp, "plan.md")
        with open(plan, "w", encoding="utf-8") as fh:
            fh.write("# Plan\n\n## 8. Outline for the write-up\n\n**Rule:** yours.\n\n| # | Section |\n|---|---|\n"
                     "| 1 | Title |\n\n- one\n- two\n\n## 9. Next\n\n| x | y |\n")
        _md, page, _summary = self.build(want_html=True, svg_path=svg, outline_path=plan,
                                         ask="Lead finding: A, B or C?", notes=["a note"])
        self.assertNotIn(BAD_LABEL, page)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)
        self.assertIn("The write-up is yours to write.", page)
        self.assertIn("no sample sentences", page)
        self.assertIn('viewBox="0 0 10 5"', page)
        self.assertNotIn('width="10"', page)
        self.assertNotIn("<?xml", page)
        self.assertIn("<th>Section</th>", page)
        self.assertIn("<strong>Rule:</strong>", page)
        self.assertNotIn("<td>x</td>", page)  # stops at the next heading
        self.assertNotIn("http://", page.replace('xmlns="http://www.w3.org/2000/svg"', ""))
        self.assertIn("prefers-color-scheme: dark", page)

    def test_slot_notes(self):
        self.assertEqual(FS.parse_slot_notes(["f8=chance 30%"]), {"F8": "chance 30%"})
        for bad in ("F15=x", "F8", "F8= "):
            with self.assertRaises(FS.SheetError):
                FS.parse_slot_notes([bad])
        self.write_all()
        md, _page, _summary = self.build(slot_notes={"F8": "chance 30%"})
        self.assertIn("Note: chance 30%", md)

    def test_cli_writes_files_and_counts_only(self):
        self.write_all()
        import io
        out = io.StringIO()
        md_path = os.path.join(self.tmp, "FACTS-v9.md")
        code = FS.main(["--out", self.tmp, "--label", "v9"], out=out)
        self.assertEqual(code, 0)
        self.assertTrue(os.path.isfile(md_path))
        self.assertNotIn("Bee", out.getvalue())
        bad = io.StringIO()
        self.assertEqual(FS.main(["--out", os.path.join(self.tmp, "missing")], out=bad), 2)
        self.assertIn("facts_sheet: ERROR", bad.getvalue())


class Helpers(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(FS.num(28250), "28,250")
        self.assertEqual(FS.num(45188.72), "45,188.7")
        self.assertEqual(FS.num(28153.0, 0), "28,153")
        self.assertEqual(FS.num(None), "n/a")
        self.assertEqual(FS.tiny(0.05 / 7671), "0.0000065")
        self.assertEqual(FS.pct(1, 4), "25.0%")

    def test_verdict(self):
        self.assertEqual(FS.verdict(stat(100, 101.0, 2.0, 70)), "within chance")
        self.assertEqual(FS.verdict(stat(90, 101.0, 2.0, 100)), "below chance")
        self.assertEqual(FS.verdict(stat(110, 101.0, 2.0, 0)), "above chance")
        self.assertIsNone(FS.verdict(None))

    def test_token_text(self):
        self.assertEqual(FS.token_text("phrase:to the map"), "to the map")
        self.assertEqual(FS.token_text("convention:control:HOLD"), "HOLD")


class Planted(unittest.TestCase):
    def test_planted_fixtures_come_back(self):
        result = FS.planted_check()
        self.assertNotIn("error", result)
        self.assertEqual(result["cascades_found"], result["cascades_planted"])
        self.assertEqual(result["cascades_planted"], 3)
        self.assertEqual(result["inherited_false_links"], 0)
        self.assertEqual(result["parallel_flagged"], 2)


if __name__ == "__main__":
    unittest.main()
