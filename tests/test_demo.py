"""The README demo: a made-up log with a known answer (examples/planted_chat.ndjson).

Standard library unittest only. The shipped log is the planted chat of
tests/test_planted.py, written by its write_chat() and labelled made-up in
every line's group field; its answer key is PLANTED in that test. These tests
run the README's demo command and check that:

* the shipped file equals write_chat() byte for byte;
* the run prints "above chance" against the label shuffle;
* the report's adopter table, and top_trees.json, give the PLANTED parents
  parent for parent;
* the captions under the lead tree print the chance bound to 2 decimals (no
  "0.0") and drop "not proof", since more trees beat every shuffle than the
  bound.
"""

import html
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from patient_zero import pipeline as P  # noqa: E402
from patient_zero import report as R  # noqa: E402
from tests import test_planted as TP  # noqa: E402

DEMO = os.path.join(ROOT, "examples", "planted_chat.ndjson")
README = os.path.join(ROOT, "README.md")
DEMO_COMMAND = "python cli.py --adapter generic --input examples/planted_chat.ndjson --out out/example"
ZERO = re.compile(r"(?<![\d.])0\.0(?![\d])")  # "0.0" as a whole number, not inside 0.03 or 10.0


def page_lines(page):
    """report.html as plain text, one line per paragraph, table row or SVG text."""
    page = re.sub(r"<style.*?</style>", "", page, flags=re.S)
    page = re.sub(r"<(?:br|p|div|h\d|tr|li|summary|figcaption|text)\b[^>]*>", "\n", page)
    text = html.unescape(re.sub(r"<[^>]+>", " ", page))
    return [re.sub(r"\s+", " ", line).strip() for line in text.split("\n") if line.strip()]


def table_after(page, summary):
    """Rows (lists of cell texts) of the first table after a <summary> with this text."""
    start = page.index("<summary>%s</summary>" % html.escape(summary))
    body = page[start:page.index("</table>", start)]
    rows = []
    for tr in re.findall(r"<tr>(.*?)</tr>", body, flags=re.S):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S)
        rows.append([html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in cells])
    return rows


class ShippedFile(unittest.TestCase):
    def test_equals_write_chat_byte_for_byte(self):
        tmp = tempfile.mkdtemp(prefix="pz-demo-file-")
        try:
            with open(TP.write_chat(tmp), "rb") as fh:
                want = fh.read()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        with open(DEMO, "rb") as fh:
            got = fh.read()
        self.assertEqual(got, want, "examples/planted_chat.ndjson differs from tests/test_planted.write_chat(); "
                                    "regenerate it (see write_chat's docstring)")
        self.assertNotIn(b"\r", got)

    def test_every_line_is_labelled_made_up(self):
        with open(DEMO, encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh]
        self.assertEqual(len(lines), len(TP.CHAT))
        self.assertEqual(set(line["group"] for line in lines), {"made-up"})

    def test_readme_runs_the_planted_demo(self):
        with open(README, encoding="utf-8") as fh:
            readme = fh.read()
        self.assertIn(DEMO_COMMAND, readme)
        self.assertIn("PLANTED", readme)
        self.assertIn("examples/fake_log.ndjson", readme)  # kept for the format section
        first_block = readme.split("```")[1]
        self.assertIn(DEMO_COMMAND, first_block, "the README's first command block must run the planted demo")


class DemoRun(unittest.TestCase):
    """The README command: cli.py on the shipped planted log."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-demo-run-")
        out_dir = os.path.join(cls.tmp, "example")
        buf = io.StringIO()
        cls.code = P.main(["--adapter", "generic", "--input", DEMO, "--out", out_dir], out=buf)
        cls.printed = buf.getvalue()
        with open(os.path.join(out_dir, "report.html"), encoding="utf-8") as fh:
            cls.page = fh.read()
        with open(os.path.join(out_dir, "top_trees.json"), encoding="utf-8") as fh:
            cls.detail = json.load(fh)
        with open(os.path.join(out_dir, "facts.json"), encoding="utf-8") as fh:
            cls.facts = json.load(fh)
        cls.lines = page_lines(cls.page)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_prints_above_chance(self):
        self.assertEqual(self.code, 0)
        self.assertIn("against the shuffle: above chance", self.printed)
        self.assertIn("Against the label shuffle: above chance.", " ".join(self.lines))

    def test_every_planted_tree_comes_back_parent_for_parent(self):
        trees = {t["token"]: t for t in self.detail["trees"]}
        for key, want in TP.PLANTED.items():
            with self.subTest(token=key):
                self.assertIn(key, trees)
                nodes = trees[key]["nodes"]
                got = {n["agent"]: (nodes[n["parent"]]["agent"] if n["parent"] is not None else None)
                       for n in nodes}
                self.assertEqual(got, want["parents"])
                self.assertEqual(nodes[trees[key]["first"]]["agent"], want["first_user"])
                self.assertEqual(nodes[trees[key]["seed"]]["agent"], want["effective_seed"])

    def test_report_adopter_table_matches_the_key(self):
        lead = next(t for t in self.detail["trees"] if t["token_id"] == self.detail["lead"])
        self.assertIn(lead["token"], TP.PLANTED)
        rows = table_after(self.page, "Table view: every adopter of the lead tree")
        head, body = rows[0], rows[1:]
        a, s = head.index("agent"), head.index("source #")
        agents = [r[a] for r in body]
        got = {r[a]: (agents[int(r[s])] if r[s] else None) for r in body}
        self.assertEqual(got, TP.PLANTED[lead["token"]]["parents"])

    def test_captions_print_the_bound_and_drop_not_proof(self):
        nm = self.facts["nullmodel"]
        self.assertGreater(nm["trees_beating_every_shuffle"], R.beat_bound(nm))
        captions = [line for line in self.lines
                    if "beat every shuffle" in line or "beat every one of" in line or "chance bound" in line]
        self.assertGreaterEqual(len(captions), 4)  # lead text, 2 figure lines, the chance-bound line
        for line in captions:
            with self.subTest(caption=line[:60]):
                self.assertIsNone(ZERO.search(line), line)
                self.assertNotIn("not proof", line)
        joined = " ".join(captions)
        self.assertIn("up to 0.03 trees", joined)
        self.assertIn("up to 0.03 such trees", joined)
        self.assertIn("chance bound 0.03", joined)
        self.assertNotIn("not proof", self.page)


if __name__ == "__main__":
    unittest.main()
