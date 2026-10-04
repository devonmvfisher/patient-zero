"""Tests for the one-command self-check (tools/can_it_see.py) and the figure's subject label.

Standard library unittest only; no data needed. Everything runs on made-up logs.

What is pinned here:

1. the made-up demo log, byte for byte (its SHA-256), and that it holds only invented names and words;
2. the three planted verdicts the demo prints: seen, blind spot, not flagged, with their counts;
3. that "real wiki" is never printed for a log that is not the wiki export, in the card or in the figure;
4. that the default subject leaves every label of the published figure as it is.
"""

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

_spec = importlib.util.spec_from_file_location("can_it_see", os.path.join(ROOT, "tools", "can_it_see.py"))
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)

# The made-up demo log as tools/can_it_see.py writes it (seed 20261004, 600 rows, LF line endings).
DEMO_SHA256 = "cb3206756bbd32199c36ba7f34f093e72bf3fc5113a8f971b4b702f651af512d"
DEMO_BYTES = 124623

# The card the demo prints (order null, seed 20261003, 100 shuffles). The first three rows are planted on the
# made-up log's own rows, so their answers are known; the other four are the made-up log itself and its controls.
DEMO_CARD = [
    "planted: carried to another page: 520 vs 354.02, 1.469, above the band: seen",
    "planted: spread on one page: 435 vs 435.0, 1.000, inside the band: blind spot",
    "decoy: the page tells every writer: 73 vs 73.0, 1.000, inside the band: not flagged",
    "this log: all pairs: 73 vs 76.6, 0.953, inside the band",
    "control: random rows, all pairs: 1,809 vs 1,794.92, 1.008, inside the band: unbiased",
    "this log: cross-page pairs: 54 vs 57.06, 0.946, inside the band",
    "control: random rows, cross-page: 1,363 vs 1,340.51, 1.017, inside the band: unbiased",
]

NULL = {"scope": "agent_day", "shuffles": 100, "seed": 7, "min_lag_seconds": 0, "strict_exposure": False}
REAL = {"observed": 515, "mean": 520.5, "sd": 9.25, "p05": 505.0, "p95": 536.0}


def cell(observed, mean, p05, p95):
    return {"observed": observed, "mean": mean, "sd": 1.5, "p05": p05, "p95": p95, "shuffles_ge_observed": 50}


def calibration():
    return {"schema": C.SCHEMA,
            "null": dict(NULL, kind="order", h0={k: REAL[k] for k in ("observed", "mean", "sd")}),
            "planted": {"carry": {"contact": True, "tokens": 40, "all_links": cell(300, 150.5, 140.0, 161.0)},
                        "contagion": {"contact": True, "tokens": 40, "all_links": cell(200, 199.9, 200.0, 200.0)}},
            "real": {"cross_page": cell(80, 100.25, 90.0, 110.0), "same_page": cell(435, 420.25, 410.0, 431.0)}}


def checks():
    return {"tool": "patient-zero checks", "order_null": dict(NULL, adopters=1000, links_named=dict(REAL))}


def run(args):
    buf = io.StringIO()
    return W.main(args, out=buf), buf.getvalue()


class MadeUpLog(unittest.TestCase):
    def test_bytes_are_pinned(self):
        data = W.made_up_log()
        self.assertEqual(data, W.made_up_log(), "the made-up log is not deterministic")
        self.assertEqual(data.count(b"\r"), 0)
        self.assertEqual(len(data), DEMO_BYTES)
        self.assertEqual(hashlib.sha256(data).hexdigest(), DEMO_SHA256,
                         "the demo log changed: the card pinned below was measured on the old one")

    def test_only_invented_names_and_words(self):
        rows = [json.loads(line) for line in W.made_up_log().decode("ascii").splitlines()]
        self.assertEqual(len(rows), W.DEMO_ROWS)
        allowed = set(W.WORDS) | set(W.NOTES)
        for row in rows:
            self.assertEqual(sorted(row), ["agent", "group", "id", "place", "text", "time", "uncertainty_seconds"])
            self.assertEqual(row["group"], "made-up")
            self.assertRegex(row["agent"], r"^Agent\d\d$")
            self.assertRegex(row["place"], r"^board/p\d\d$")
            self.assertLessEqual(set(row["text"].split()), allowed)
        self.assertEqual(len({row["agent"] for row in rows}), W.DEMO_AGENTS)
        self.assertEqual(len({row["place"] for row in rows}), W.DEMO_PLACES)
        times = [row["time"] for row in rows]
        self.assertEqual(times, sorted(times))
        self.assertLess(times[-1] - W.DEMO_START, W.DEMO_DAYS * 86400)

    def test_another_seed_is_another_log(self):
        self.assertNotEqual(W.made_up_log(seed=W.DEMO_SEED + 1, rows=50), W.made_up_log(rows=50))


class Demo(unittest.TestCase):
    """One demo run (about 15 seconds), read by every test below."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-can-it-see-")
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.code, cls.text = run(["--demo", "--out", cls.tmp])
        cls.lines = [line.strip() for line in cls.text.splitlines()]

    def test_exit_and_files(self):
        self.assertEqual(self.code, 0, self.text)
        for name in W.OUTPUTS + (W.DEMO_NAME,):
            self.assertTrue(os.path.isfile(os.path.join(self.tmp, name)), name)
        with open(os.path.join(self.tmp, W.DEMO_NAME), "rb") as fh:
            self.assertEqual(hashlib.sha256(fh.read()).hexdigest(), DEMO_SHA256)

    def test_the_three_planted_verdicts(self):
        card = [line for line in self.lines if line.startswith(("planted: ", "decoy: "))]
        self.assertEqual(len(card), 3, self.text)
        self.assertTrue(card[0].endswith("above the band: seen"), card[0])
        self.assertTrue(card[1].endswith("inside the band: blind spot"), card[1])
        self.assertTrue(card[2].endswith("inside the band: not flagged"), card[2])

    def test_the_card_is_pinned(self):
        start = self.lines.index(DEMO_CARD[0]) if DEMO_CARD[0] in self.lines else -1
        self.assertGreaterEqual(start, 0, self.text)
        self.assertEqual(self.lines[start:start + len(DEMO_CARD)], DEMO_CARD)

    def test_says_made_up_and_never_real_wiki(self):
        self.assertIn("made-up log: 600 rows", self.text)
        self.assertIn("no spread is planted in it", self.text)
        self.assertNotIn("real wiki", self.text.lower())
        with open(os.path.join(self.tmp, "can-it-see.svg"), encoding="utf-8") as fh:
            svg = fh.read()
        self.assertNotIn("real wiki", svg.lower())
        self.assertIn(">Can it see? Planted on this log<", svg)
        self.assertIn(">Did it see? This log<", svg)
        self.assertIn(">this log: all pairs<", svg)

    def test_same_null_as_the_published_figure(self):
        with open(os.path.join(self.tmp, "calibration.json"), encoding="utf-8") as fh:
            null = json.load(fh)["null"]
        self.assertEqual((null["kind"], null["scope"], null["seed"], null["shuffles"]),
                         ("order", "agent_day", 20261003, 100))
        self.assertIn("order null, seed 20261003, 100 shuffles", self.text)

    def test_counts_only(self):
        for name in W.OUTPUTS:
            with open(os.path.join(self.tmp, name), encoding="utf-8") as fh:
                text = fh.read()
            self.assertNotRegex(text, r"Agent\d\d", name)
            for piece in ("board/p", "notes-", "kestrel", "thistle", "juniper", "osprey"):
                self.assertNotIn(piece, text, name)


class OwnLog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pz-can-it-see-own-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_small_log_prints_a_card_that_says_this_log(self):
        log = os.path.join(ROOT, "examples", "planted_chat.ndjson")
        code, text = run(["--input", log, "--out", self.tmp, "--shuffles", "20"])
        self.assertEqual(code, 0, text)
        self.assertIn("this log: all pairs: ", text)
        self.assertNotIn("real wiki", text.lower())
        self.assertNotIn("made-up log:", text)  # only --demo writes and names the made-up log
        self.assertIn("decoy: the page tells every writer: not run yet", text)
        self.assertIn("could not be planted on this log", text)
        with open(os.path.join(self.tmp, "can-it-see.svg"), encoding="utf-8") as fh:
            self.assertNotIn("real wiki", fh.read().lower())

    def test_a_log_too_small_to_plant_on_stops_with_one_sentence(self):
        log = os.path.join(ROOT, "examples", "fake_log.ndjson")
        code, text = run(["--input", log, "--out", self.tmp, "--shuffles", "20"])
        self.assertEqual(code, 2, text)
        self.assertNotIn("Traceback", text)
        self.assertIn("no card", text)
        self.assertIn("cannot hold the planted kinds", text)
        self.assertNotIn("planted: carried", text)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "can-it-see.svg")))

    def test_other_column_names_stop_at_step_one(self):
        code, text = run(["--input", os.path.join(ROOT, "examples", "fake_log.csv"), "--out", self.tmp])
        self.assertEqual(code, 2, text)
        self.assertIn("stopped at step 1 of 5", text)
        self.assertIn("it has no --map", text)
        self.assertNotIn("Traceback", text)

    def test_setup_errors_are_one_sentence_and_write_nothing(self):
        log = os.path.join(ROOT, "examples", "planted_chat.ndjson")
        inside = os.path.join(ROOT, "docs", "never-written")
        for args in ([], ["--demo", "--input", log, "--out", self.tmp], ["--input", log],
                     ["--input", os.path.join(self.tmp, "missing.ndjson"), "--out", self.tmp],
                     ["--demo", "--out", inside], ["--demo", "--out", self.tmp, "--shuffles", "1"]):
            code, text = run(args)
            self.assertEqual(code, 2, args)
            self.assertTrue(text.startswith("can_it_see: ERROR: "), text)
            self.assertEqual(text.count("\n"), 1, text)
            self.assertEqual(os.listdir(self.tmp), [], args)
        self.assertFalse(os.path.exists(inside))

    def test_odd_logs_stop_with_plain_sentences_and_stay_unchanged(self):
        from tests.test_hostile import FIXTURES, NOT_PLAIN  # the odd made-up logs every reader is tested on
        for name, (data, _want, phrase) in sorted(FIXTURES.items()):
            log = os.path.join(self.tmp, name + ".ndjson")
            with open(log, "wb") as fh:
                fh.write(data)
            with self.subTest(fixture=name):
                code, text = run(["--input", log, "--out", os.path.join(self.tmp, "out-" + name),
                                  "--shuffles", "2"])
                self.assertEqual(code, 2, text)  # 3 rows at most: too small for a card even when it loads
                self.assertIn("no card", text)
                for needle in NOT_PLAIN:
                    self.assertNotIn(needle, text)
                if _want == 2:
                    self.assertIn(phrase, text)
                    self.assertIn("stopped at step 1 of 5", text)
                with open(log, "rb") as fh:
                    self.assertEqual(fh.read(), data, name)

    def test_no_step_may_write_over_the_log(self):
        for name in W.OUTPUTS:
            log = os.path.join(self.tmp, name)
            shutil.copyfile(os.path.join(ROOT, "examples", "planted_chat.ndjson"), log)
            with open(log, "rb") as fh:
                before = fh.read()
            code, text = run(["--input", log, "--out", self.tmp])
            self.assertEqual(code, 2, name)
            self.assertIn("which is also an input", text)
            with open(log, "rb") as fh:
                self.assertEqual(fh.read(), before, name)
            os.remove(log)


class Subject(unittest.TestCase):
    HASHES = {"calibration.json": "a" * 64, "checks.json": "b" * 64}

    def test_default_subject_changes_nothing(self):
        self.assertEqual(C.SUBJECT, "real wiki")
        self.assertEqual([C.label_for(row) for row in C.ROWS], [row.label for row in C.ROWS])
        self.assertIs(C.sections_for(), C.SECTIONS)
        plain = C.draw(calibration(), checks(), self.HASHES)
        self.assertEqual(plain, C.draw(calibration(), checks(), self.HASHES, subject=C.SUBJECT))
        self.assertEqual([r["label"] for r in C.build_rows(calibration(), checks())], [row.label for row in C.ROWS])

    def test_published_figure_keeps_its_labels(self):
        with open(os.path.join(ROOT, "docs", "can-it-see.svg"), encoding="utf-8") as fh:
            svg = fh.read()
        for text in [row.label for row in C.ROWS] + [heading for _key, heading in C.SECTIONS]:
            self.assertIn(">%s<" % text, svg)

    def test_another_subject_changes_those_four_texts_and_nothing_else(self):
        plain = C.draw(calibration(), checks(), self.HASHES)
        named = C.draw(calibration(), checks(), self.HASHES, subject="this log")
        self.assertNotIn("real wiki", named)
        swapped = (plain.replace(">real wiki: all pairs<", ">this log: all pairs<")
                   .replace(">real wiki: cross-page pairs<", ">this log: cross-page pairs<")
                   .replace(">Can it see? Planted on the real wiki<", ">Can it see? Planted on this log<")
                   .replace(">Did it see? The real wiki<", ">Did it see? This log<"))
        self.assertEqual(named, swapped)
        self.assertEqual(plain.count("real wiki"), 4)

    def test_planted_and_control_labels_never_carry_the_subject(self):
        labels = [r["label"] for r in C.build_rows(calibration(), checks(), "team chat")]
        self.assertEqual([lab for lab in labels if lab.startswith("team chat: ")],
                         ["team chat: all pairs", "team chat: cross-page pairs"])
        self.assertEqual([lab for lab in labels if not lab.startswith("team chat: ")],
                         [row.label for row in C.ROWS if not row.label.startswith("real wiki: ")])

    def test_subject_must_be_a_short_plain_name(self):
        for bad in ("", " this log", "this log ", "<b>", "a&b", "x" * 41, "two\nlines", 'say "hi"', None, 7):
            with self.assertRaises(C.CanISeeError, msg=repr(bad)):
                C.check_subject(bad)
        for good in ("this log", "real wiki", "team-chat 2", "x" * 40):
            self.assertEqual(C.check_subject(good), good)

    def test_command_line_takes_the_subject_and_refuses_a_bad_one(self):
        tmp = tempfile.mkdtemp(prefix="pz-can-it-see-subject-")
        self.addCleanup(shutil.rmtree, tmp, True)
        paths = {}
        for name, doc in (("calibration.json", calibration()), ("checks.json", checks())):
            paths[name] = os.path.join(tmp, name)
            with open(paths[name], "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
        base = ["--calibration", paths["calibration.json"], "--checks", paths["checks.json"]]
        svg = os.path.join(tmp, "named.svg")
        buf = io.StringIO()
        self.assertEqual(C.main(base + ["--svg", svg, "--subject", "this log"], out=buf), 0, buf.getvalue())
        self.assertIn("this log: all pairs: ", buf.getvalue())
        self.assertNotIn("real wiki", buf.getvalue())
        bad = os.path.join(tmp, "bad.svg")
        buf = io.StringIO()
        self.assertEqual(C.main(base + ["--svg", bad, "--subject", "<script>"], out=buf), 2)
        self.assertIn("--subject must be", buf.getvalue())
        self.assertFalse(os.path.exists(bad))
        buf = io.StringIO()
        plain = os.path.join(tmp, "plain.svg")
        self.assertEqual(C.main(base + ["--svg", plain], out=buf), 0)
        self.assertIn("real wiki: all pairs: ", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
