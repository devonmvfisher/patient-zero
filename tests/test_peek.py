"""Tests for tools/peek.py (unittest, standard library only).

All fixture text is made up here or comes from examples/fake_log.ndjson.
Outputs go to a temporary folder outside the repo.
"""

import importlib.util
import json
import os
import shutil
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOL = os.path.join(REPO, "tools", "peek.py")
EXAMPLE = os.path.join(REPO, "examples", "fake_log.ndjson")
_spec = importlib.util.spec_from_file_location("peek", TOOL)
pk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pk)

from patient_zero.cascade import Link  # noqa: E402
from patient_zero.load import ROLE_AGENT, Row  # noqa: E402
from patient_zero.tokens import Token  # noqa: E402


def _token(key):
    cls, rest = key.split(":", 1)
    kind, value = rest.split(":", 1) if cls == "convention" else (cls, rest)
    return Token(key=key, cls=cls, kind=kind, value=value, rows=(), agents=3, first_users=(), risky=())


def _row(row_id, place, text, t):
    return Row(row_id=row_id, agent="a" + row_id, role=ROLE_AGENT, time=t, uncertainty=1.0,
               time_grade=None, place=place, family="fam", group="g", text=text)


class SnippetTest(unittest.TestCase):
    def test_short_line_is_kept_whole(self):
        self.assertEqual(pk.snippet("short line", 0, 5), "short line")

    def test_long_line_is_cut_around_the_token(self):
        line = "x" * 150 + " lanternfish.example.org " + "y" * 150
        start = line.index("lanternfish")
        out = pk.snippet(line, start, start + len("lanternfish.example.org"))
        self.assertLessEqual(len(out), 80)
        self.assertIn("lanternfish.example.org", out)
        self.assertTrue(out.startswith("...") and out.endswith("..."))

    def test_token_at_line_start_has_no_leading_marker(self):
        line = "copperheron_signal " + "z" * 200
        out = pk.snippet(line, 0, len("copperheron_signal"))
        self.assertTrue(out.startswith("copperheron_signal"))
        self.assertTrue(out.endswith("..."))
        self.assertLessEqual(len(out), 80)

    def test_escapes_and_never_splits_one(self):
        line = "é\\" * 60 + " MARKER " + "中" * 60
        start = line.index("MARKER")
        out = pk.snippet(line, start, start + 6)
        self.assertLessEqual(len(out), 80)
        self.assertIn("MARKER", out)
        self.assertTrue(all(32 <= ord(ch) < 127 for ch in out))
        body = out.strip(".")
        self.assertNotRegex(body, r"\\u[0-9a-f]{0,3}$")  # no half escape at the cut


class SpanTest(unittest.TestCase):
    def test_phrase_matches_across_case_and_punctuation(self):
        line = "Note: please RELAY, the-Map tonight."
        span = pk.find_span(line, _token("phrase:relay the map"))
        self.assertEqual(line[span[0]:span[1]], "RELAY, the-Map")

    def test_phrase_needs_whole_words(self):
        self.assertIsNone(pk.find_span("unrelay the map", _token("phrase:relay the map")))

    def test_host_is_found_in_any_case(self):
        line = "fetch via https://Ferry.Example.org/x"
        span = pk.find_span(line, _token("host:ferry.example.org"))
        self.assertEqual(line[span[0]:span[1]].lower(), "ferry.example.org")

    def test_control_word_prefers_exact_case(self):
        line = "hold on. HOLD until noon"
        span = pk.find_span(line, _token("convention:control:HOLD"))
        self.assertEqual(line[span[0]:span[1]], "HOLD")

    def test_human_handles_are_masked_as_whole_words(self):
        out, n = pk.mask_humans("thanks Quillon, not Quillons", ["Quillon"])
        self.assertEqual(out, "thanks [human], not Quillons")
        self.assertEqual(n, 1)


class PageCheckTest(unittest.TestCase):
    A_LINE = "use ferry.example.org for fetches"

    def _setup(self, b_text, base_lines, after_lines, same_place=True):
        rows = [_row("r1", "p1", self.A_LINE, 100.0),
                _row("r2", "p1" if same_place else "p2", b_text, 200.0)]
        link = Link(token="host:ferry.example.org", source="ar1", source_row=0, target="ar2",
                    target_row=1, visit_row=1, lag=100.0)
        saves = {"r1": (None, frozenset([self.A_LINE])),
                 "r0": (None, frozenset(base_lines)),
                 "r2": ("r0", frozenset(after_lines))}
        return link, rows, saves

    def test_b_edits_a_line(self):
        b_line = self.A_LINE + " (edited)"
        link, rows, saves = self._setup(b_line, [self.A_LINE], [b_line])
        self.assertEqual(pk.page_check(link, rows, self.A_LINE, saves), (True, True))

    def test_b_adds_own_line_beside_a(self):
        b_line = "switching to ferry.example.org too"
        link, rows, saves = self._setup(b_line, [self.A_LINE], [self.A_LINE, b_line])
        self.assertEqual(pk.page_check(link, rows, self.A_LINE, saves), (True, False))

    def test_a_line_gone_before_b(self):
        b_line = "switching to ferry.example.org too"
        link, rows, saves = self._setup(b_line, ["other text"], ["other text", b_line])
        self.assertEqual(pk.page_check(link, rows, self.A_LINE, saves), (False, False))

    def test_unknown_without_bodies(self):
        link, rows, _saves = self._setup("x ferry.example.org", [], [])
        self.assertEqual(pk.page_check(link, rows, self.A_LINE, {}), (None, None))

    def test_population_counts(self):
        b_line = self.A_LINE + " (edited)"
        link, rows, saves = self._setup(b_line, [self.A_LINE], [b_line])
        tree = type("T", (), {"links": (link,), "cls": "host"})()
        key = "host:ferry.example.org"
        counts = pk.population_check(rows, {key: _token(key)}, {key: tree}, saves)
        self.assertEqual(counts["total"], {
            "links": 1, "a_line_found": 1, "a_line_on_page_at_visit": 1, "b_save_changed_a_line": 1,
            "same_line": 0, "on_page_not_changed_not_identical": 0})
        self.assertEqual(counts["by_class"]["host"], counts["total"])


class ExampleRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pz-peek-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _peek(self, name):
        out = os.path.join(self.tmp, name + ".txt")
        result = pk.peek("generic", EXAMPLE, out)
        with open(out, "r", encoding="utf-8") as fh:
            return result, fh.read()

    def test_deterministic_and_short(self):
        first, text1 = self._peek("one")
        second, text2 = self._peek("two")
        self.assertEqual(text1, text2)
        self.assertEqual(first["items"], second["items"])
        self.assertEqual(first["n"], min(pk.PEEK_N, first["population"]))
        self.assertGreater(first["n"], 0)
        for it in first["items"]:
            self.assertLessEqual(len(it["source_snippet"]), 80)
            self.assertLessEqual(len(it["target_snippet"]), 80)
            self.assertTrue(it["source_placed"] and it["target_placed"])
            self.assertIsNone(it["b_save_changed_a_line"])  # no page bodies in a generic log
        self.assertNotIn("Host", text1)  # the example's human speaker
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "one.json")))

    def test_record_writes_hand_check_and_keeps_core_hash(self):
        result, _text = self._peek("peek")
        peek_json = os.path.join(self.tmp, "peek.json")
        result["facts_core_sha256"] = "c0ffee"
        with open(peek_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        facts_path = os.path.join(self.tmp, "facts.json")
        with open(facts_path, "w", encoding="utf-8") as fh:
            json.dump({"core_sha256": "c0ffee", "slots": {"pending": {"F10": "hand check (J9)"}}}, fh)
        marks = [{"n": it["n"], "token_id": it["token_id"],
                  "mark": "plausible" if i == 0 else "coincidence", "reason": "made-up reason %d" % i}
                 for i, it in enumerate(result["items"])]
        marks_path = os.path.join(self.tmp, "marks.json")
        with open(marks_path, "w", encoding="utf-8") as fh:
            json.dump({"reader": "test reader", "marks": marks}, fh)
        md_path = os.path.join(self.tmp, "hand-check.md")
        section = pk.record(marks_path, peek_json, facts_path, md_path)
        self.assertEqual(section["precision"], "1 of %d" % len(marks))
        with open(facts_path, "r", encoding="utf-8") as fh:
            facts = json.load(fh)
        self.assertEqual(facts["core_sha256"], "c0ffee")
        self.assertEqual(facts["hand_check"]["plausible"], 1)
        self.assertEqual(facts["slots"]["F10"]["precision"], section["precision"])
        self.assertNotIn("F10", facts["slots"]["pending"])
        with open(md_path, "r", encoding="utf-8") as fh:
            md = fh.read()
        for it in result["items"]:
            self.assertNotIn(it["target_snippet"], md)  # the table carries no snippet text

    def test_record_refuses_stale_facts_and_bad_marks(self):
        result, _text = self._peek("peek")
        result["facts_core_sha256"] = "c0ffee"
        peek_json = os.path.join(self.tmp, "peek.json")
        with open(peek_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        facts_path = os.path.join(self.tmp, "facts.json")
        with open(facts_path, "w", encoding="utf-8") as fh:
            json.dump({"core_sha256": "different"}, fh)
        marks_path = os.path.join(self.tmp, "marks.json")
        good = [{"n": it["n"], "token_id": it["token_id"], "mark": "unclear", "reason": "r"}
                for it in result["items"]]
        with open(marks_path, "w", encoding="utf-8") as fh:
            json.dump({"marks": good}, fh)
        md_path = os.path.join(self.tmp, "hand-check.md")
        with self.assertRaises(ValueError):
            pk.record(marks_path, peek_json, facts_path, md_path)
        with open(facts_path, "w", encoding="utf-8") as fh:
            json.dump({"core_sha256": "c0ffee"}, fh)
        for bad in (good[:-1], [dict(good[0], mark="maybe")] + good[1:],
                    [dict(good[0], reason="")] + good[1:], [dict(good[0], reason="café")] + good[1:]):
            with open(marks_path, "w", encoding="utf-8") as fh:
                json.dump({"marks": bad}, fh)
            with self.assertRaises(ValueError):
                pk.record(marks_path, peek_json, facts_path, md_path)
        self.assertFalse(os.path.exists(md_path))


if __name__ == "__main__":
    unittest.main()
