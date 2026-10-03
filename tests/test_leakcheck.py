"""Tests for tools/leakcheck.py (unittest, standard library only).

All fixture text is made up here. IP-shaped strings and the half-IP field
name are assembled at run time, so this file passes the leak check itself.
"""

import gzip
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(os.path.dirname(HERE), "tools", "leakcheck.py")
_spec = importlib.util.spec_from_file_location("leakcheck", TOOL)
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)

# Made-up sentences (14+ words each).
A = "lanterns hum softly while seven copper herons debate the price of imaginary river stones"
B = "a patient gardener counts every snail twice before granting it passage across the moss bridge"
C = "nobody expected the brass orchestra of tiny clocks to rehearse a symphony made entirely of sneezes"
D = "the lighthouse keeper archived forty years of fog in jars labelled with the names of ships"
E = "violet tram tickets drift upward whenever the conductor whistles an unfinished lullaby backwards in winter"
LOW = " ".join(["zero", "one"] * 7)

FIELD = "ip" + "16"
V4 = ".".join(["198", "51", "100", "7"])
V4_WILD = ".".join(["203", "0", "x", "x"])
V4_STAR = ".".join(["203", "0", "*", "*"])
V6_FULL = ":".join(["2001", "db8", "0", "0", "0", "0", "0", "1"])
V6_SHORT = "fe80" + "::" + "1"


def _write_jsonl_gz(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _fixture_revisions():
    r1 = A + "\nshort line"
    r2 = r1 + "\n" + B
    r3 = r2 + "\n" + C
    return [
        # page created: both lines added
        {"rev_id": "fx:1", "diff_base": None, "body": r1,
         "hunks": [{"op": "insert", "a0": 0, "a1": 0, "b0": 0, "b1": 2}]},
        # one line added by an insert hunk
        {"rev_id": "fx:2", "diff_base": "fx:1", "body": r2,
         "hunks": [{"op": "insert", "a0": 2, "a1": 2, "b0": 2, "b1": 3}]},
        # no hunks: difflib against the base finds the added line
        {"rev_id": "fx:3", "diff_base": "fx:2", "body": r3, "hunks": []},
        # E sits in the body but no insert/replace hunk covers it: never indexed
        {"rev_id": "fx:4", "diff_base": "fx:3", "body": E + "\n" + A,
         "hunks": [{"op": "delete", "a0": 1, "a1": 4, "b0": 1, "b1": 1}]},
        # a repetitive line (2 distinct words) is ignored
        {"rev_id": "fx:5", "diff_base": None, "body": LOW,
         "hunks": [{"op": "insert", "a0": 0, "a1": 0, "b0": 0, "b1": 1}]},
    ]


class LeakcheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-leakcheck-")
        cls.data = os.path.join(cls.tmp, "data")
        os.makedirs(cls.data)
        _write_jsonl_gz(os.path.join(cls.data, "revisions.jsonl.gz"), _fixture_revisions())
        _write_jsonl_gz(os.path.join(cls.data, "records.jsonl.gz"), [{"id": "x1", "text": D}])
        cls.index_path = os.path.join(cls.tmp, "out", "grams.bin")
        cls.meta = lc.build_index(cls.data, cls.index_path)
        cls.index = lc.load_index(cls.index_path)
        cls.repo_count = 0

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def make_repo(self, files):
        LeakcheckTest.repo_count += 1
        repo = os.path.join(self.tmp, "repo%d" % self.repo_count)
        for rel, content in files.items():
            path = os.path.join(repo, *rel.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            mode = "wb" if isinstance(content, bytes) else "w"
            with open(path, mode, **({} if mode == "wb" else {"encoding": "utf-8"})) as fh:
                fh.write(content)
        os.makedirs(repo, exist_ok=True)
        return repo

    def run_main(self, args):
        buf = io.StringIO()
        code = lc.main(args, out=buf)
        return code, buf.getvalue()

    def kinds(self, text):
        return [f.kind for f in lc.scan_text(text, self.index)]

    # ------------------------------------------------ index build

    def test_build_reads_added_text_only(self):
        stats = self.meta["sources"]["revisions.jsonl.gz"]
        self.assertEqual(stats["saves"], 5)
        self.assertEqual(stats["difflib_fallback"], 1)
        self.assertEqual(self.meta["sources"]["records.jsonl.gz"]["with_text"], 1)
        self.assertEqual(self.meta["ngram"], 12)

    def test_index_holds_hashes_not_text(self):
        with open(self.index_path, "rb") as fh:
            raw = fh.read()
        self.assertNotIn(b"lanterns", raw)
        self.assertEqual(len(raw), 8 * self.meta["count"])

    # ------------------------------------------------ copied runs

    def test_copied_run_found_when_rewrapped_and_recased(self):
        text = "Lanterns hum softly, while SEVEN copper\nherons debate the price of imaginary (river) stones!"
        self.assertEqual(self.kinds(text), ["copied-12-words"])

    def test_insert_hunk_difflib_fallback_and_records_are_indexed(self):
        for sentence in (B, C, D):
            self.assertEqual(self.kinds(sentence), ["copied-12-words"], sentence[:30])

    def test_text_outside_added_hunks_is_not_indexed(self):
        self.assertEqual(self.kinds(E), [])

    def test_eleven_words_pass(self):
        self.assertEqual(self.kinds(" ".join(A.split()[:11])), [])

    def test_repetitive_runs_ignored(self):
        self.assertEqual(self.kinds(LOW), [])

    def test_one_finding_per_run_with_line_number(self):
        found = lc.scan_text("intro line\nsecond line\n" + A + "\n" + B, self.index)
        self.assertEqual([(f.kind, f.line) for f in found],
                         [("copied-12-words", 3), ("copied-12-words", 4)])

    # ------------------------------------------------ IP shapes and field name

    def test_ip_shapes_flagged(self):
        for text in ("host " + V4 + ".", V4 + ":8080", "net " + V4_WILD, "net " + V4_STAR,
                     V6_FULL, "link " + V6_SHORT + " up"):
            self.assertEqual(self.kinds(text), ["ip-address"], text)

    def test_numbers_that_are_not_ips_pass(self):
        for text in ("Python 3.12", "version 1.2.3", "build 10.0.26200.1", "at 13:05:00",
                     "2026-10-03T13:05:00-05:00", "x[::-1]", "a[1::2]", "999.1.1.1",
                     "1.2.3.4.5", "ratio 0.5"):
            self.assertEqual(self.kinds(text), [], text)

    def test_field_name_flagged(self):
        for text in ('{"%s": "x"}' % FIELD, FIELD.upper(), "row_" + FIELD):
            self.assertEqual(self.kinds(text), ["field-name"], text)
        for text in ("zip" + "16", FIELD + "0"):
            self.assertEqual(self.kinds(text), [], text)

    def test_snippet_is_escaped_single_line_and_short(self):
        text = "tab\there é " + V4 + " " + "y" * 200 + "\nnext"
        (finding,) = lc.scan_text(text, self.index)
        self.assertLessEqual(len(finding.snippet), 80)
        self.assertTrue(finding.snippet.isascii())
        self.assertNotIn("\n", finding.snippet)

    # ------------------------------------------------ repo scan and CLI

    def test_clean_repo_passes(self):
        repo = self.make_repo({"README.md": "Run it.\nPython 3.12\n", "pkg/a.py": "x = y[::-1]\n"})
        code, out = self.run_main(["--repo", repo, "--index", self.index_path])
        self.assertEqual(code, 0, out)
        self.assertIn("leakcheck: OK", out)

    def test_forbidden_files_flagged_without_reading(self):
        repo = self.make_repo({
            "a.GZ": b"\x1f\x8b not really gzip",
            "sub/rows.jsonl": '{"host": "%s"}\n' % V4,
            "pack.zip": b"PK",
        })
        findings, count = lc.scan(repo, (), self.index)
        self.assertEqual(count, 3)
        self.assertEqual(sorted(f.kind for f in findings), ["forbidden-file"] * 3)

    def test_git_folder_skipped(self):
        repo = self.make_repo({".git/config": "url = %s\n" % V4, "ok.txt": "fine\n"})
        code, out = self.run_main(["--repo", repo, "--index", self.index_path])
        self.assertEqual(code, 0, out)

    def test_leaks_in_repo_and_extra_file_fail(self):
        repo = self.make_repo({"notes.md": "about " + V4 + "\n"})
        extra = os.path.join(self.tmp, "writeup-draft.md")
        with open(extra, "w", encoding="utf-8") as fh:
            fh.write("Finding one.\n" + B + "\n")
        code, out = self.run_main(["--repo", repo, "--index", self.index_path, extra])
        self.assertEqual(code, 1, out)
        self.assertIn("notes.md:1 [ip-address]", out)
        self.assertIn("[copied-12-words]", out)

    def test_missing_index_is_setup_error(self):
        repo = self.make_repo({"ok.txt": "fine\n"})
        code, out = self.run_main(["--repo", repo, "--index", os.path.join(self.tmp, "none.bin")])
        self.assertEqual(code, 2, out)

    def test_missing_extra_file_is_setup_error(self):
        repo = self.make_repo({"ok.txt": "fine\n"})
        code, out = self.run_main(["--repo", repo, "--index", self.index_path,
                                   os.path.join(self.tmp, "missing.md")])
        self.assertEqual(code, 2, out)

    def test_index_inside_repo_refused(self):
        repo = self.make_repo({"ok.txt": "fine\n"})
        with self.assertRaises(lc.SetupError):
            lc.build_index(self.data, os.path.join(repo, "out", "grams.bin"), repo_root=repo)
        shutil.copy(self.index_path, os.path.join(repo, "grams.bin"))
        shutil.copy(self.index_path + ".json", os.path.join(repo, "grams.bin.json"))
        code, out = self.run_main(["--repo", repo, "--index", os.path.join(repo, "grams.bin")])
        self.assertEqual(code, 2, out)

    def test_settings_mismatch_is_setup_error(self):
        bad = os.path.join(self.tmp, "bad.bin")
        shutil.copy(self.index_path, bad)
        with open(self.index_path + ".json", encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["ngram"] = 8
        with open(bad + ".json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        with self.assertRaises(lc.SetupError):
            lc.load_index(bad)


if __name__ == "__main__":
    unittest.main()
