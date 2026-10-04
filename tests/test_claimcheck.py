"""Tests for tools/claimcheck.py: every number in README, docs/*.svg and WRITEUP* traces to a run's files.

unittest, standard library only. The example run is built once from the
committed made-up log (examples/fake_log.ndjson). Documents are checked in a
temporary copy of the repo, so the real README, figure and any write-up are
never touched. The real-run checks run only when PZ_RUN names a real run's
output folder (facts.json, top_trees.json, and checks.json when there is
one), or when PZ_DATA names the collusion.wiki export (the pipeline then runs
first, which takes 1 to 5 minutes).
"""

import glob
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, os.path.join(ROOT, "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

import claimcheck as CC  # noqa: E402

MINUS = chr(0x2212)
RESTORE = ("README.md", "claims-allow.json", "docs/lead.svg")


def run(args):
    buf = io.StringIO()
    code = CC.main(args, out=buf)
    return code, buf.getvalue()


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def copy_repo(dst):
    """A working copy of the repo (no .git, no out/) inside ``dst``; returns its path."""
    repo = os.path.join(dst, "repo")
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git", "__pycache__", "out"))
    return repo


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def allow_with(repo, runs=None, entries=None, replace=False):
    """Rewrite the copy's claims-allow.json with extra runs and entries."""
    path = os.path.join(repo, CC.ALLOW_NAME)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if runs:
        data["runs"].update(runs)
    if replace:
        data["entries"] = []
    data["entries"].extend(entries or [])
    write(path, json.dumps(data, indent=1))
    return path


def facts(folder):
    with open(os.path.join(folder, "facts.json"), encoding="utf-8") as fh:
        return json.load(fh)


class ExampleBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-claims-")
        cls.example = os.path.join(cls.tmp, "example-run")
        os.makedirs(cls.example)
        CC.build_example(ROOT, cls.example)
        cls.ex_facts = facts(cls.example)
        cls.repo = copy_repo(cls.tmp)  # one copy per class; setUp puts the edited files back

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="pz-claims-case-", dir=self.tmp)
        for rel in RESTORE:
            shutil.copyfile(os.path.join(ROOT, *rel.split("/")), os.path.join(self.repo, *rel.split("/")))
        for path in glob.glob(os.path.join(self.repo, "WRITEUP*")):
            os.remove(path)

    def check(self, *extra):
        return run(["--repo", self.repo, "--run", "example=" + self.example] + list(extra))


class TextScanning(unittest.TestCase):
    def scan(self, text, md=False):
        return [(t.kind, t.text) for t in CC.scan_text(text, markdown=md)[0]]

    def test_numbers_as_shown(self):
        self.assertEqual(self.scan("28,250 links; ratio 0.625. z -0.95, 35.1% and 1.6x (33)"),
                         [("number", "28,250"), ("number", "0.625"), ("number", "-0.95"), ("number", "35.1%"),
                          ("number", "1.6x"), ("number", "33")])

    def test_names_with_digits_are_skipped(self):
        toks, skipped = CC.scan_text("F8 J11 SHA-256 ISO-8601 10th 3rd ec8b9e2777d5 PZ-11 Apache-2.0 out/2026")
        self.assertEqual(toks, [])
        self.assertEqual(skipped, 10)

    def test_compounds_ranges_and_lists(self):
        self.assertEqual(self.scan("a 12-word run, 5-9 agents, 1,200/800/500, z=8.0"),
                         [("number", "12"), ("number", "5"), ("number", "9"), ("number", "1,200"), ("number", "800"),
                          ("number", "500"), ("number", "8.0")])

    def test_dates_times_versions(self):
        self.assertEqual(self.scan("generated 2026-09-03 at 16:43 -0500 with v1.2.3, 2026-01-10T09:00:00Z"),
                         [("date", "2026-09-03"), ("time", "16:43"), ("time", "-0500"), ("version", "v1.2.3"),
                          ("date", "2026-01-10T09:00:00Z")])

    def test_markdown_list_marker_skipped(self):
        self.assertEqual(self.scan("1. Download 3 files", md=True), [("number", "3")])
        self.assertEqual(self.scan("1. Download 3 files", md=False), [("number", "1"), ("number", "3")])

    def test_glued_units_and_signs_are_numbers(self):
        self.assertEqual(self.scan("lag 583s, 8.4h, 9.9σ, p≈0.99, n≥7, ×1.6, US$80, 3rd, 10th"),
                         [("number", "583"), ("number", "8.4"), ("number", "9.9"), ("number", "0.99"),
                          ("number", "7"), ("number", "1.6"), ("number", "80")])
        self.assertEqual(self.scan("28.3k links, 2M rows"), [("abbreviated", "28.3k"), ("abbreviated", "2M")])

    def test_unicode_minus(self):
        tok = CC.scan_text("z " + MINUS + "8.2")[0][0]
        self.assertEqual((tok.value, tok.signed), (-8.2, True))

    def test_svg_reads_text_not_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f.svg")
            write(path, '<svg width="960"><title>Lead 12</title><rect x="24" y="130" width="12"/>\n'
                        '<text x="5" y="9">count 7,671 &#x27;quoted&#x27; and <tspan>19.7</tspan></text></svg>')
            hits, segs, _ = CC.read_hits(path, "f.svg")
        self.assertEqual([h.token.text for h in hits], ["12", "7,671", "19.7"])
        self.assertEqual([h.line for h in hits], [1, 2, 2])
        self.assertIn("'quoted'", segs[1])

    def test_html_writeup_drops_tags_and_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "WRITEUP.html")
            write(path, '<p style="width:40px">We saw <b>931</b> of 931.</p>\n<script>var a = 77;</script>\n')
            hits, _segs, _ = CC.read_hits(path, "WRITEUP.html")
        self.assertEqual([h.token.text for h in hits], ["931", "931"])


class Matching(unittest.TestCase):
    def tok(self, text):
        return CC.scan_text(text)[0][0]

    def test_shown_decimals_commas_percent_sign(self):
        cases = [("28,250", 28250, True), ("28,251", 28250, False), ("0.625", 0.625, True), ("19.7", 19.73, True),
                 ("19.8", 19.73, False), ("76", 75.95, True), ("35.1%", 0.3508, True), ("35.2%", 0.3508, False),
                 ("-0.95", -0.95, True), ("0.95", -0.95, False), ("-0.95", 0.95, False), ("583", 583.0, True),
                 ("1.6x", 1.592, True), ("79.7%", 79.7, True)]
        for text, value, want in cases:
            with self.subTest(text=text, value=value):
                self.assertEqual(CC.matches_value(self.tok(text), value), want)

    def test_pool_finds_best_source(self):
        pool = CC.Pool([("wiki.checks:a", 28250), ("wiki.facts:cascade.links_named", 28250), ("wiki.facts:b", 7)])
        self.assertEqual(pool.find(self.tok("28,250")), ("wiki.facts:cascade.links_named", 28250))
        self.assertIsNone(pool.find(self.tok("28,251")))
        self.assertEqual(pool.nearest(self.tok("28,251"), 1)[0][1], "wiki.facts:cascade.links_named")

    def test_dropped_minus_sign_fails(self):
        pool = CC.Pool([("wiki.checks:strict.z", -8.2)])
        self.assertIsNone(pool.find(self.tok("8.2")))
        self.assertEqual(pool.find_magnitude(self.tok("8.2")), ("wiki.checks:strict.z", -8.2))
        self.assertEqual(pool.find(self.tok(MINUS + "8.2")), ("wiki.checks:strict.z", -8.2))
        row = {}
        self.assertEqual(CC._judge(self.tok("8.2"), None, pool, {}, ROOT, [], row), "fail")
        self.assertIn("minus sign is not shown", row["why"])
        self.assertTrue(CC._caught(self.tok("-8.2"), None, pool, {}, ROOT, []))  # 8.2 would now fail

    def test_expression_language(self):
        runs = {"r": {"sources": {"facts": {"a": 10, "b.c": 4}}}}
        self.assertEqual(CC.evaluate("{r.facts:a} / ({r.facts:a} + {r.facts:b.c}) * 100", runs, ROOT),
                         1000 / 14.0)
        self.assertEqual(CC.evaluate("{code:patient_zero/safe.py:SHORT_MAX}", runs, ROOT), 80)
        with self.assertRaises(CC.Unverified):
            CC.evaluate("{wiki.facts:a}", runs, ROOT)
        with self.assertRaises(CC.Unverified):
            CC.evaluate("{r.checks:a}", runs, ROOT)
        for bad in ("__import__('os').getcwd()", "{r.facts:a} ** 2", "{r.facts:missing}", "{r.lead}", "1 +"):
            with self.subTest(bad=bad):
                with self.assertRaises(CC.CheckError):
                    CC.evaluate(bad, runs, ROOT)


class ExampleRun(ExampleBase):
    """The committed example run: the README traces, and edits to numbers fail with file:line."""

    def test_readme_traces_on_example_run(self):
        code, out = self.check("--only", "README.md")
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"README\.md: \d+ numbers: .* 0 FAIL")
        self.assertNotIn("STALE", out)
        self.assertNotIn("FAIL README", out)

    def test_default_builds_example_and_skips_figure_without_wiki(self):
        code, out = run(["--repo", self.repo])
        self.assertEqual(code, 0, out)
        self.assertIn("run example: core ", out)
        self.assertIn("docs/lead.svg: skipped (needs run wiki)", out)
        seconds = float(re.search(r"time: ([\d.]+) s", out).group(1))
        self.assertLess(seconds, 10.0)  # measured about 1 s with the example build; see RESULT

    def test_check_itself_is_fast(self):
        runs = [CC.load_run("example", self.example)]
        allow = CC.load_allow(os.path.join(self.repo, CC.ALLOW_NAME))
        started = time.perf_counter()
        result = CC.check(self.repo, CC.default_docs(self.repo), runs, allow)
        self.assertLess(time.perf_counter() - started, 2.0)
        self.assertEqual(result["fail"], 0)

    def _writeup(self, seed_text):
        mean = self.ex_facts["nullmodel"]["links_named"]["mean"]
        rows = self.ex_facts["load"]["rows"]
        text = ("# Notes\n\nThe example log has %d rows.\nSeed %s; links expected under the shuffle %.1f.\n"
                % (rows, seed_text, mean))
        path = os.path.join(self.repo, "WRITEUP.md")
        write(path, text)
        return path

    def test_changed_number_fails_with_file_line(self):
        allow_with(self.repo, runs={"WRITEUP*": ["example"]})
        path = self._writeup("20261003")
        code, out = self.check()
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"WRITEUP\.md: 3 numbers: 3 matched")
        pool = CC.make_pool([CC.load_run("example", self.example)])
        self.assertIsNone(pool.find(CC.scan_text("20261004")[0][0]))  # the edit is not some other count
        self._writeup("20261004")
        code, out = self.check()
        self.assertEqual(code, 1, out)
        self.assertRegex(out, r"FAIL WRITEUP\.md:4:6  20261004  not in facts/checks/calibration/lead; nearest "
                              r"example\.facts:")
        self.assertIn("claimcheck: FAIL", out)
        self.assertTrue(os.path.isfile(path))

    def test_writeup_is_never_written(self):
        allow_with(self.repo, runs={"WRITEUP*": ["example"]})
        path = self._writeup("99999999")
        before, mtime = sha(path), os.path.getmtime(path)
        code, out = self.check("--list")
        self.assertEqual(code, 1, out)
        self.assertEqual((sha(path), os.path.getmtime(path)), (before, mtime))

    def test_writeup_skipped_without_its_run(self):
        self._writeup("20261003")
        code, out = self.check()
        self.assertEqual(code, 0, out)
        self.assertIn("WRITEUP.md: skipped (needs run wiki)", out)

    def test_pinned_text_and_derived_value_catch_drift(self):
        allow_with(self.repo, runs={"WRITEUP*": ["example"]}, entries=[
            {"file": "WRITEUP.md", "at": "Seed {};", "text": "20261003", "reason": "the seed we used"},
            {"file": "WRITEUP.md", "at": "has {} rows", "from": "{example.facts:load.rows}", "reason": "rows in the example log"}])
        self._writeup("20261003")
        code, out = self.check()
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"WRITEUP\.md: 3 numbers: 1 matched, 1 pinned to a source, 1 allowed by reason")
        path = os.path.join(self.repo, "WRITEUP.md")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        write(path, text.replace("Seed 20261003", "Seed 20261004").replace("has 20 rows", "has 21 rows"))
        code, out = self.check()
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL WRITEUP.md:4:6  20261004  allow entry", out)
        self.assertRegex(out, r"FAIL WRITEUP\.md:3:\d+  21  allow entry \d+: the sources give 20")

    def test_stale_entry_listed_and_strict_fails(self):
        allow_with(self.repo, entries=[{"file": "README.md", "at": "no such words {} here", "text": "5",
                                        "reason": "an entry whose words are gone"}])
        code, out = self.check("--only", "README.md")
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"STALE allow entry \d+ \(README\.md, no such words \{\} here\): matches nothing now")
        code, out = self.check("--only", "README.md", "--strict")
        self.assertEqual(code, 1, out)

    def test_unverified_when_a_run_is_missing(self):
        allow_with(self.repo, runs={"WRITEUP*": ["example"]}, entries=[
            {"file": "WRITEUP.md", "at": "has {} rows", "from": "{wiki.facts:load.rows}", "reason": "rows in the example log"}])
        self._writeup("20261003")
        code, out = self.check()
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"UNVERIFIED WRITEUP\.md:3:\d+  20  allow entry \d+: run wiki not given")

    def test_bad_allow_entries_stop_the_check(self):
        bad = [{"file": "README.md", "text": "5"},
               {"file": "README.md", "text": "5", "reason": "fine reason", "form": "{x}"},
               {"file": "README.md", "at": "no placeholder", "reason": "fine reason"},
               {"file": "README.md", "at": "{} and {}", "reason": "fine reason"},
               {"file": "README.md", "reason": "fine reason"},
               {"file": "README.md", "at": "{} lines", "from": "__import__('os')", "reason": "fine reason"}]
        for entry in bad:
            with self.subTest(entry=entry):
                allow_with(self.repo, entries=[entry], replace=True)
                code, out = self.check("--only", "README.md")
                self.assertEqual(code, 2, out)
                self.assertIn("claimcheck: error:", out)

    def test_stale_run_files_stop_the_check(self):
        run_dir = os.path.join(self.work, "run")
        shutil.copytree(self.example, run_dir)
        with open(os.path.join(run_dir, "top_trees.json"), "a", encoding="utf-8") as fh:
            fh.write(" ")
        code, out = run(["--repo", self.repo, "--run", "example=" + run_dir, "--only", "README.md"])
        self.assertEqual(code, 2, out)
        self.assertIn("top_trees.json does not belong to its facts.json", out)
        code, out = run(["--repo", self.repo, "--run", "example=" + run_dir, "--only", "README.md", "--skip-stale"])
        self.assertEqual(code, 0, out)
        self.assertIn("left out top_trees.json", out)
        write(os.path.join(run_dir, "checks.json"), json.dumps({"rows": 1, "tokens": 1}))
        code, out = run(["--repo", self.repo, "--run", "example=" + run_dir, "--only", "README.md", "--skip-stale"])
        self.assertEqual(code, 0, out)
        self.assertIn("left out checks.json (stale: rows 1 vs 20", out)

    def test_drift_power_is_measured(self):
        allow_with(self.repo, runs={"WRITEUP*": ["example"]}, entries=[
            {"file": "WRITEUP.md", "at": "has {} rows", "from": "{example.facts:load.rows}", "reason": "rows in the example log"},
            {"file": "WRITEUP.md", "at": "^Tick {}$", "reason": "an axis tick, any value"}])
        path = self._writeup("20261003")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("Tick 4\n")
        runs = [CC.load_run("example", self.example)]
        allow = CC.load_allow(os.path.join(self.repo, CC.ALLOW_NAME))
        result = CC.check(self.repo, [path], runs, allow)
        caught = {r["text"]: r["caught"] for r in result["numbers"]}
        self.assertTrue(caught["20"])  # pinned to load.rows: 19 and 21 fail
        self.assertTrue(caught["20261003"])  # no other count is 20261002 or 20261004
        self.assertFalse(caught["4"])  # a pattern entry lets any tick through
        self.assertEqual(result["power"]["checked"], 4)

    def test_json_receipt_outside_repo_only(self):
        code, out = self.check("--only", "README.md", "--json", os.path.join(self.repo, "claims.json"))
        self.assertEqual(code, 2, out)
        receipt = os.path.join(self.work, "claims.json")
        code, out = self.check("--only", "README.md", "--json", receipt)
        self.assertEqual(code, 0, out)
        with open(receipt, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertTrue(data["ok"])
        self.assertTrue(all(r["file"] == "README.md" for r in data["numbers"]))
        self.assertTrue(all(len(r["context"]) <= CC.CONTEXT for r in data["numbers"]))


def _real_run_dir(tmp):
    if os.environ.get("PZ_RUN"):
        return os.environ["PZ_RUN"]
    from patient_zero import pipeline
    out = os.path.join(tmp, "wiki-run")
    code = pipeline.main(["--adapter", "wiki", "--input", os.environ["PZ_DATA"], "--out", out], out=io.StringIO())
    if code:
        raise RuntimeError("real pipeline run failed")
    return out


@unittest.skipUnless(os.environ.get("PZ_RUN") or os.environ.get("PZ_DATA"),
                     "set PZ_RUN to a real run's output folder (or PZ_DATA to the collusion.wiki export)")
class RealRun(ExampleBase):
    """The real run: README, the public figure and any write-up trace; edits fail with file:line."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.real = _real_run_dir(cls.tmp)
        cls.real_facts = facts(cls.real)

    def check(self, *extra):
        return super().check("--run", "wiki=" + self.real, *extra)

    def test_repo_documents_trace(self):
        code, out = run(["--repo", ROOT, "--run", "example=" + self.example, "--run", "wiki=" + self.real])
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"docs/lead\.svg: \d+ numbers: .* 0 FAIL")
        self.assertRegex(out, r"README\.md: \d+ numbers: .* 0 FAIL")

    def test_links_count_edit_fails_with_file_line(self):
        links = self.real_facts["cascade"]["links_named"]
        path = os.path.join(self.repo, "WRITEUP.md")
        write(path, "# Notes\n{:,} named links.\n".format(links))
        code, out = self.check()
        self.assertEqual(code, 0, out)
        pool = CC.make_pool([CC.load_run("wiki", self.real), CC.load_run("example", self.example)])
        edited = "{:,}".format(links + 1)  # 28,250 becomes 28,251 on the current export
        self.assertIsNone(pool.find(CC.scan_text(edited)[0][0]))
        write(path, "# Notes\n{} named links.\n".format(edited))
        code, out = self.check()
        self.assertEqual(code, 1, out)
        line = re.search(r"FAIL WRITEUP\.md:2:1  (\S+)  not in facts/checks/calibration/lead; nearest (.*)", out)
        self.assertIsNotNone(line, out)
        self.assertEqual(line.group(1), edited)
        self.assertIn("{:,}".format(links), line.group(2))  # names the value it drifted from

    def test_figure_edit_fails_with_file_line(self):
        svg = os.path.join(self.repo, "docs", "lead.svg")
        total = "{:,}".format(self.real_facts["trees"]["total"])
        with open(svg, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("among %s." % total, text)
        write(svg, text.replace("among %s." % total, "among %s." % "{:,}".format(self.real_facts["trees"]["total"] + 1)))
        code, out = self.check()
        self.assertEqual(code, 1, out)
        self.assertRegex(out, r"FAIL docs/lead\.svg:2:\d+  [\d,]+  allow entry \d+: the sources give %s"
                         % re.escape(total))

    def test_real_check_is_fast(self):
        runs = [CC.load_run("example", self.example), CC.load_run("wiki", self.real)]
        allow = CC.load_allow(os.path.join(ROOT, CC.ALLOW_NAME))
        started = time.perf_counter()
        result = CC.check(ROOT, CC.default_docs(ROOT), runs, allow)
        self.assertLess(time.perf_counter() - started, 2.0)
        self.assertEqual(result["fail"], 0)


if __name__ == "__main__":
    unittest.main()
