"""Hostile input in, safe output out (unittest, standard library only).

Every log here is made up: three events by agents A1, A2 and A3 who all write
"alpha beta gamma delta", bent in one way per fixture. Outputs go to a
temporary folder outside the repo.

* Hostile: the 9 odd generic logs (a UTF-8 byte-order mark, a blank line,
  fractional seconds, a bad time, a null agent, a number as text, a duplicate
  id, a cut-off last line, a time with a space), plus a UTF-16 log and a
  cp1252 log, through every command that reads a generic log (cli.py, load,
  checks, peek): no traceback, exit 0 or exit 2 with one plain line that
  names the line, and the input file is never changed.
* ByteOrderMark: a log saved with a UTF-8 or UTF-16 byte-order mark gives the
  same core sections as the plain UTF-8 file; only ``input`` (file size and
  hash) differs.
* OutputGuard: checks ``--out``, peek ``--out`` (and its .json twin), peek
  ``--record --md`` and facts_sheet ``--md``/``--html`` refuse a path that is
  one of their inputs, exit 2 with a plain sentence, and leave the input's
  sha256 unchanged.
* MoreWriters: the same for cli.py ``--out`` (a plain sentence instead of an
  OS error), tokens and cascade ``--summary`` and offwiki ``--json``/``--facts``.
* HandEditedFiles: a marks file or a plan outline saved with a byte-order mark.
* GuardOut: the shared helper ``patient_zero.safe.guard_out``.
"""

import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from patient_zero import checks as K  # noqa: E402
from patient_zero import load as L  # noqa: E402
from patient_zero import pipeline as P  # noqa: E402
from patient_zero.safe import guard_out, same_file  # noqa: E402

EXAMPLE = os.path.join(ROOT, "examples", "fake_log.ndjson")


def _tool(name):
    spec = importlib.util.spec_from_file_location("hostile_" + name, os.path.join(ROOT, "tools", name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PK = _tool("peek")
FS = _tool("facts_sheet")

L1 = (b'{"id":"e1","agent":"A1","time":"2026-01-10T09:00:00Z","place":"p","group":"g","family":"f",'
      b'"text":"alpha beta gamma delta"}\n')
L2 = L1.replace(b'"e1"', b'"e2"').replace(b'"A1"', b'"A2"').replace(b"09:00:00Z", b"09:05:00Z")
L3 = L1.replace(b'"e1"', b'"e3"').replace(b'"A1"', b'"A3"').replace(b"09:00:00Z", b"09:09:00Z")
PLAIN = L1 + L2 + L3
UTF8_BOM = b"\xef\xbb\xbf"

# name -> (bytes, expected exit, phrase in the cli.py line)
FIXTURES = {
    "bom": (UTF8_BOM + PLAIN, 0, "3 rows, 3 agents"),
    "blank": (L1 + b"\n" + L2 + L3, 0, "3 rows, 3 agents"),
    "frac": (L1 + L2.replace(b"09:05:00Z", b"09:05:00.12345Z") + L3, 0, "3 rows, 3 agents"),
    "badtime": (L1 + L2.replace(b'"2026-01-10T09:05:00Z"', b'"yesterday"') + L3, 2, "line 2: bad time"),
    "nullagent": (PLAIN.replace(b'"agent":"A3"', b'"agent":null'), 0, "3 rows, 2 agents"),
    "numtext": (L1 + L2 + L3.replace(b'"alpha beta gamma delta"', b"12"), 2, "line 3: text must be a string"),
    "dupid": (L1 + b"\n" + L1, 2, "line 3: duplicate id"),
    "truncated": (L1 + L2 + b'{"id":"e3","agent":"A3"', 2, "line 3: not valid JSON"),
    "space": (L1 + L2.replace(b'"2026-01-10T09:05:00Z"', b'"2026-01-10T2026-01-10 09:05"') + L3, 2,
              "line 2: bad time"),
    # added with the byte-order-mark fix: what Windows PowerShell 5 writes, and an ANSI (cp1252) save
    "utf16": (b"\xff\xfe" + PLAIN.decode("ascii").encode("utf-16-le"), 0, "3 rows, 3 agents"),
    "cp1252": (L1 + L2.replace(b"alpha beta", b"caf\xe9 beta") + L3, 2, "line 2: not UTF-8 text"),
    # added by the PZ-8 skeptic: numbers JSON allows that no clock can print, and deep nesting
    "inftime": (L1 + L2.replace(b'"2026-01-10T09:05:00Z"', b"1e400") + L3, 2, "line 2: bad time"),
    "nantime": (L1 + L2.replace(b'"2026-01-10T09:05:00Z"', b"NaN") + L3, 2, "line 2: bad time"),
    "bigtime": (L1 + L2.replace(b'"2026-01-10T09:05:00Z"', b"1e20") + L3, 2, "line 2: bad time"),
    "bigint": (L1 + L2.replace(b'"2026-01-10T09:05:00Z"', b"1" + b"0" * 400) + L3, 2, "line 2: bad time"),
    "nanunc": (L1 + L2.replace(b'"place"', b'"uncertainty_seconds":NaN,"place"') + L3, 2,
               "line 2: uncertainty_seconds"),
    "bigunc": (L1 + L2.replace(b'"place"', b'"uncertainty_seconds":1' + b"0" * 400 + b',"place"') + L3, 2,
               "line 2: uncertainty_seconds"),
    "deepnest": (L1 + L2.replace(b'"place"', b'"x":' + b"[" * 100000 + b"]" * 100000 + b',"place"') + L3, 2,
                 "line 2: not valid JSON"),
}
NOT_PLAIN = ("Traceback", "Errno", "WinError", "codec can't", "decode using")


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _write(path, data):
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _call(fn, argv):
    """(exit code, printed text); an exception escaping fn fails the test that called it."""
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err):
        code = fn(argv, out=buf)
    return code, buf.getvalue() + err.getvalue()


class Hostile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-hostile-")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _entry_points(self, name, path):
        out = os.path.join(self.tmp, "out-" + name)
        return (
            ("patient-zero", P.main, ["--adapter", "generic", "--input", path, "--out", out, "--shuffles", "5"]),
            ("load", L.main, ["--adapter", "generic", "--input", path]),
            ("checks", K.main, ["--adapter", "generic", "--input", path, "--shuffles", "3"]),
            ("peek", PK.main, ["--adapter", "generic", "--input", path, "--out",
                               os.path.join(self.tmp, "peek-%s.txt" % name)]),
        )

    def test_every_reader_exits_cleanly(self):
        for name, (data, want, phrase) in sorted(FIXTURES.items()):
            path = _write(os.path.join(self.tmp, name + ".ndjson"), data)
            before = _sha(path)
            for prefix, fn, argv in self._entry_points(name, path):
                with self.subTest(fixture=name, command=prefix):
                    try:
                        code, text = _call(fn, argv)
                    except Exception as exc:  # noqa: BLE001 - a traceback is the failure
                        self.fail("%s on %s raised %s: %s" % (prefix, name, type(exc).__name__, exc))
                    self.assertEqual(code, want, text)
                    for needle in NOT_PLAIN:
                        self.assertNotIn(needle, text)
                    if want == 2:
                        self.assertEqual(len(text.strip().splitlines()), 1, text)
                        self.assertTrue(text.startswith(prefix + ": ERROR:"), text)
                        self.assertIn(phrase, text)
                    elif prefix == "patient-zero":
                        self.assertIn(phrase, text.splitlines()[0])
            self.assertEqual(_sha(path), before, name)

    def test_cli_script_prints_no_traceback(self):
        # the real script, one fixture per outcome (each start costs about 0.7 s); the in-process
        # test above runs every fixture through every reader
        for name in ("bom", "truncated", "cp1252"):
            data, want, phrase = FIXTURES[name]
            path = _write(os.path.join(self.tmp, "s-" + name + ".ndjson"), data)
            proc = subprocess.run(
                [sys.executable, os.path.join(ROOT, "cli.py"), "--adapter", "generic", "--input", path,
                 "--out", os.path.join(self.tmp, "s-out-" + name), "--shuffles", "5"],
                capture_output=True, text=True, timeout=120)
            with self.subTest(fixture=name):
                self.assertEqual(proc.returncode, want, proc.stdout + proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)
                self.assertIn(phrase, proc.stdout.splitlines()[0])

    def test_cp1252_names_the_line_even_late_in_a_big_file(self):
        no_ids = PLAIN.replace(b'"id":"e1",', b"").replace(b'"id":"e2",', b"").replace(b'"id":"e3",', b"")
        body = no_ids * 300  # 900 lines, far past the text reader's first chunk; rows get ids line-N
        lines = body.count(b"\n")
        path = _write(os.path.join(self.tmp, "late.ndjson"),
                      body + L2.replace(b'"id":"e2",', b"").replace(b"alpha", b"caf\xe9"))
        with self.assertRaises(ValueError) as caught:
            L.load("generic", path)
        self.assertIn("line %d: not UTF-8 text" % (lines + 1), str(caught.exception))


class ByteOrderMark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-bom-")
        cls.facts, cls.trees = {}, {}
        for label, data in (("plain", PLAIN), ("bom", UTF8_BOM + PLAIN),
                            ("utf16", FIXTURES["utf16"][0])):
            path = _write(os.path.join(cls.tmp, label + ".ndjson"), data)
            out = os.path.join(cls.tmp, "out-" + label)
            code, text = _call(P.main, ["--adapter", "generic", "--input", path, "--out", out])
            assert code == 0, text
            with open(os.path.join(out, "facts.json"), encoding="utf-8") as fh:
                cls.facts[label] = json.load(fh)
            with open(os.path.join(out, "trees.jsonl"), "rb") as fh:
                cls.trees[label] = fh.read()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _same_core_except_input(self, label):
        plain, other = self.facts["plain"], self.facts[label]
        for key in P.CORE_KEYS:
            if key != "input":
                self.assertEqual(other[key], plain[key], key)
        self.assertNotEqual(other["input"], plain["input"])
        self.assertEqual(other["input"]["files"][0]["name"].split(".")[1], "ndjson")
        self.assertEqual(self.trees[label], self.trees["plain"])
        self.assertEqual(other["load"]["rows"], 3)
        self.assertGreater(other["trees"]["total"], 0)

    def test_utf8_bom_gives_the_same_core(self):
        self._same_core_except_input("bom")
        self.assertEqual(self.facts["bom"]["input"]["files"][0]["bytes"],
                         self.facts["plain"]["input"]["files"][0]["bytes"] + 3)

    def test_utf16_gives_the_same_core(self):
        self._same_core_except_input("utf16")

    def test_gzipped_log_with_bom_loads(self):
        path = os.path.join(self.tmp, "bom.ndjson.gz")
        with gzip.open(path, "wb") as fh:
            fh.write(UTF8_BOM + PLAIN)
        rows, _stats = L.load("generic", path)
        plain_rows, _ = L.load("generic", os.path.join(self.tmp, "plain.ndjson"))
        self.assertEqual(rows, plain_rows)

    def test_sniff_encoding(self):
        cases = ((PLAIN, "utf-8-sig"), (UTF8_BOM + PLAIN, "utf-8-sig"), (b"\xff\xfe{\x00", "utf-16"),
                 (b"\xfe\xff\x00{", "utf-16"), (b"\xff\xfe\x00\x00", "utf-32"), (b"", "utf-8-sig"))
        for i, (data, want) in enumerate(cases):
            path = _write(os.path.join(self.tmp, "sniff-%d.ndjson" % i), data)
            self.assertEqual(L.sniff_encoding(path), want)

    def test_a_bom_inside_a_line_is_still_refused(self):
        path = _write(os.path.join(self.tmp, "mid.ndjson"), L1 + UTF8_BOM + L2)
        with self.assertRaises(ValueError) as caught:
            L.load("generic", path)
        self.assertIn("line 2: not valid JSON", str(caught.exception))


class RunFolder(unittest.TestCase):
    """A made-up run (cli.py on examples/fake_log.ndjson), a peek with a marks file, a plan and a figure."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-guard-")
        cls.log = os.path.join(cls.tmp, "log.ndjson")
        shutil.copyfile(EXAMPLE, cls.log)
        cls.run_dir = os.path.join(cls.tmp, "run")
        code, text = _call(P.main, ["--adapter", "generic", "--input", cls.log, "--out", cls.run_dir])
        assert code == 0, text
        cls.facts = os.path.join(cls.run_dir, "facts.json")
        cls.top = os.path.join(cls.run_dir, "top_trees.json")
        cls.peek_txt = os.path.join(cls.tmp, "peek.txt")
        code, text = _call(PK.main, ["--adapter", "generic", "--input", cls.log, "--out", cls.peek_txt,
                                     "--facts", cls.facts])
        assert code == 0, text
        cls.peek_json = os.path.join(cls.tmp, "peek.json")
        with open(cls.peek_json, encoding="utf-8") as fh:
            cls.items = json.load(fh)["items"]
        cls.marks = os.path.join(cls.tmp, "marks.json")
        with open(cls.marks, "w", encoding="utf-8") as fh:
            json.dump(cls.marks_object(), fh)
        cls.outline = os.path.join(cls.tmp, "plan.md")
        with open(cls.outline, "w", encoding="utf-8") as fh:
            fh.write("# Plan\n\n## Outline\n\n- one\n")
        cls.svg = os.path.join(cls.tmp, "lead.svg")
        shutil.copyfile(os.path.join(ROOT, "docs", "lead.svg"), cls.svg)

    @classmethod
    def marks_object(cls):
        return {"reader": "test", "marks": [{"n": i["n"], "token_id": i["token_id"], "mark": "unclear",
                                             "reason": "made up"} for i in cls.items]}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _refused(self, fn, argv, watched, phrase):
        before = _sha(watched)
        code, text = _call(fn, argv)
        self.assertEqual(code, 2, text)
        self.assertIn(phrase, text)
        self.assertEqual(len(text.strip().splitlines()), 1, text)
        for needle in NOT_PLAIN:
            self.assertNotIn(needle, text)
        self.assertEqual(_sha(watched), before)
        return text


class OutputGuard(RunFolder):
    """Each command pointed at its own input exits 2 with a plain sentence; the input is unchanged."""

    # checks --out
    def test_checks_out_is_its_input(self):
        self._refused(K.main, ["--adapter", "generic", "--input", self.log, "--out", self.log, "--shuffles", "3"],
                      self.log, "--out names %s, which is also an input" % self.log)

    def test_checks_out_inside_the_input_folder(self):
        export = os.path.join(self.tmp, "export")
        os.makedirs(export, exist_ok=True)
        rev = _write(os.path.join(export, "revisions.jsonl.gz"), b"made up")
        self._refused(K.main, ["--adapter", "wiki", "--input", export, "--out", rev], rev,
                      "inside the input folder")
        self._refused(K.main, ["--adapter", "wiki", "--input", export, "--out",
                               os.path.join(export, "checks.json")], rev, "inside the input folder")

    def test_checks_out_is_a_folder(self):
        self._refused(K.main, ["--adapter", "generic", "--input", self.log, "--out", self.run_dir,
                               "--shuffles", "3"], self.log, "which is a folder")

    # peek --out, its .json twin, and --record --md
    def test_peek_out_is_its_input(self):
        self._refused(PK.main, ["--adapter", "generic", "--input", self.log, "--out", self.log],
                      self.log, "--out names %s, which is also an input" % self.log)

    def test_peek_twin_is_facts(self):
        self._refused(PK.main, ["--adapter", "generic", "--input", self.log, "--out",
                                os.path.join(self.run_dir, "facts.txt"), "--facts", self.facts],
                      self.facts, "the .json twin of --out names")

    def test_peek_out_ending_in_json(self):
        target = os.path.join(self.tmp, "sample.json")
        self._refused(PK.main, ["--adapter", "generic", "--input", self.log, "--out", target],
                      self.log, "must not end in .json")
        self.assertFalse(os.path.exists(target))

    def test_peek_record_md_is_an_input(self):
        for target in (self.facts, self.marks, self.peek_json):
            self._refused(PK.main, ["--record", self.marks, "--peek", self.peek_json, "--facts", self.facts,
                                    "--md", target], target, "--md names")

    # facts_sheet --md / --html
    def test_facts_sheet_md_is_facts(self):
        self._refused(FS.main, ["--out", self.run_dir, "--md", self.facts], self.facts,
                      "--md names %s, which is also an input" % self.facts)

    def test_facts_sheet_html_is_facts_or_outline_or_figure(self):
        self._refused(FS.main, ["--out", self.run_dir, "--html", self.facts], self.facts, "--html names")
        self._refused(FS.main, ["--out", self.run_dir, "--html", self.outline, "--outline", self.outline],
                      self.outline, "--html names")
        self._refused(FS.main, ["--out", self.run_dir, "--html", self.svg, "--svg", self.svg],
                      self.svg, "--html names")

    def test_facts_sheet_md_is_top_trees(self):
        self._refused(FS.main, ["--out", self.run_dir, "--md", self.top], self.top, "--md names")

    def test_facts_sheet_md_and_html_same_file(self):
        target = os.path.join(self.tmp, "sheet-both.txt")
        self._refused(FS.main, ["--out", self.run_dir, "--md", target, "--html", target], self.facts,
                      "both name")
        self.assertFalse(os.path.exists(target))

    def test_good_paths_still_write(self):
        md, page = os.path.join(self.tmp, "sheet.md"), os.path.join(self.tmp, "sheet.html")
        self.assertEqual(_call(FS.main, ["--out", self.run_dir, "--md", md, "--html", page, "--svg", self.svg,
                                         "--outline", self.outline])[0], 0)
        self.assertTrue(os.path.isfile(md) and os.path.isfile(page))
        target = os.path.join(self.tmp, "checks.json")
        self.assertEqual(_call(K.main, ["--adapter", "generic", "--input", self.log, "--out", target,
                                        "--shuffles", "3"])[0], 0)
        self.assertTrue(os.path.isfile(target))
        self.assertEqual(_call(PK.main, ["--adapter", "generic", "--input", self.log, "--out",
                                         os.path.join(self.tmp, "peek2.txt"), "--facts", self.facts])[0], 0)


class MoreWriters(RunFolder):
    """cli.py --out, tokens and cascade --summary, offwiki --json/--facts: same rule, same plain sentence."""

    def test_cli_out_is_the_input_file(self):
        text = self._refused(P.main, ["--adapter", "generic", "--input", self.log, "--out", self.log],
                             self.log, "--out names %s, which is also an input" % self.log)
        self.assertNotIn("183", text)

    def test_cli_out_is_some_other_file(self):
        self._refused(P.main, ["--adapter", "generic", "--input", self.log, "--out", self.peek_txt],
                      self.peek_txt, "which is a file; give a folder")

    def test_cli_out_folder_holds_the_input_under_an_output_name(self):
        folder = os.path.join(self.tmp, "mixed")
        os.makedirs(folder, exist_ok=True)
        log = os.path.join(folder, "trees.jsonl")
        shutil.copyfile(EXAMPLE, log)
        self._refused(P.main, ["--adapter", "generic", "--input", log, "--out", folder], log,
                      "trees.jsonl, which is also an input")

    def test_cli_out_inside_the_wiki_folder(self):
        export = os.path.join(self.tmp, "export-cli")
        os.makedirs(export, exist_ok=True)
        rev = _write(os.path.join(export, "revisions.jsonl.gz"), b"made up")
        self._refused(P.main, ["--adapter", "wiki", "--input", export, "--out", os.path.join(export, "out")],
                      rev, "inside the input folder")

    def test_tokens_and_cascade_summary_is_the_input(self):
        from patient_zero import cascade as C
        from patient_zero import tokens as T
        for fn in (T.main, C.main):
            self._refused(fn, ["--adapter", "generic", "--input", self.log, "--summary", self.log],
                          self.log, "--summary names")

    def test_offwiki_json_is_facts_or_inside_the_export(self):
        from patient_zero import offwiki as O
        export = os.path.join(self.tmp, "export-s1")
        os.makedirs(export, exist_ok=True)
        rev = _write(os.path.join(export, "revisions.jsonl.gz"), b"made up")
        self._refused(O.main, ["--input", export, "--facts", self.facts, "--json", self.facts], self.facts,
                      "--json names")
        self._refused(O.main, ["--input", export, "--json", os.path.join(export, "s1.json")], rev,
                      "inside the input folder")
        self._refused(O.main, ["--input", export, "--facts", rev], rev, "--facts names")


class HandEditedFiles(RunFolder):
    """Files a person edits by hand (the marks file, the plan outline) may carry a UTF-8 byte-order mark."""

    def test_marks_file_with_bom_records(self):
        work = os.path.join(self.tmp, "bom-record")
        os.makedirs(work, exist_ok=True)
        facts = shutil.copyfile(self.facts, os.path.join(work, "facts.json"))
        marks = _write(os.path.join(work, "marks.json"),
                       UTF8_BOM + json.dumps(self.marks_object()).encode("utf-8"))
        md = os.path.join(work, "hand-check.md")
        code, text = _call(PK.main, ["--record", marks, "--peek", self.peek_json, "--facts", facts, "--md", md])
        self.assertEqual(code, 0, text)
        with open(facts, encoding="utf-8") as fh:
            self.assertIn("hand_check", json.load(fh))

    def test_outline_with_bom_on_its_first_heading(self):
        plan = _write(os.path.join(self.tmp, "plan-bom.md"), UTF8_BOM + b"## Outline\n\n- first point\n")
        page = os.path.join(self.tmp, "sheet-bom.html")
        code, text = _call(FS.main, ["--out", self.run_dir, "--md", os.path.join(self.tmp, "sheet-bom.md"),
                                     "--html", page, "--svg", self.svg, "--outline", plan])
        self.assertEqual(code, 0, text)
        with open(page, encoding="utf-8") as fh:
            self.assertIn("first point", fh.read())


class GuardOut(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pz-guardout-")
        self.log = _write(os.path.join(self.tmp, "log.ndjson"), PLAIN)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_same_file_spelled_another_way(self):
        os.makedirs(os.path.join(self.tmp, "sub"))
        other = os.path.join(self.tmp, "sub", "..", "log.ndjson")
        self.assertTrue(same_file(other, self.log))
        with self.assertRaises(ValueError):
            guard_out(other, [self.log])
        if os.name == "nt":
            with self.assertRaises(ValueError):
                guard_out(self.log.upper(), [self.log])

    def test_hard_link_is_the_same_file(self):
        link = os.path.join(self.tmp, "hard.ndjson")
        try:
            os.link(self.log, link)
        except (OSError, AttributeError, NotImplementedError):
            self.skipTest("no hard links here")
        with self.assertRaises(ValueError):
            guard_out(link, [self.log])

    def test_inside_an_input_folder(self):
        with self.assertRaises(ValueError) as caught:
            guard_out(os.path.join(self.tmp, "deep", "x.json"), [self.tmp])
        self.assertIn("inside the input folder", str(caught.exception))

    def test_temporary_twin_is_an_input(self):
        tmp_input = _write(os.path.join(self.tmp, "x.json.tmp"), b"{}")
        with self.assertRaises(ValueError) as caught:
            guard_out(os.path.join(self.tmp, "x.json"), [tmp_input])
        self.assertIn("temporary file", str(caught.exception))

    def test_new_path_and_empty_inputs_pass(self):
        guard_out(os.path.join(self.tmp, "new.json"), [self.log, None, "", os.path.join(self.tmp, "gone.json")])
        guard_out(None, [self.log])
        self.assertFalse(same_file(os.path.join(self.tmp, "a"), os.path.join(self.tmp, "b")))


if __name__ == "__main__":
    unittest.main()
