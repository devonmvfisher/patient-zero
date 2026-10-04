"""Red-team regressions for what reaches an output (unittest, standard library only).

Two findings from the lane x3 red team of ea867f1 (its D01 and D02), re-cut for
the merged tree:

* GroupFamilyGrade: an IP-shaped string in a ``group``, ``family`` or
  ``time_grade`` label used to be written to facts.json, trees.jsonl,
  top_trees.json and report.html. The loader now masks it, in both adapters
  and for JSON Lines and CSV alike, the way it masks text and place names.
  Rows, links and totals stay the same; labels that differ only in the masked
  string are counted as one label.
* InputName: the input file's name in facts.json is shown like any data
  string: IP-shaped text masked, escaped, at most 80 characters. Its size and
  SHA-256 are still those of the file, and the file is not touched.

Every row is made up. The addresses are assembled at run time from the
documentation ranges, so this file passes the leak check itself.
"""

import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from patient_zero import load as L  # noqa: E402
from patient_zero import pipeline as P  # noqa: E402
from patient_zero.safe import SHORT_MAX  # noqa: E402

V4_A = ".".join(["203", "0", "113", "19"])
V4_B = ".".join(["203", "0", "113", "23"])
V6 = ":".join(["2001", "db8", "0", "0", "0", "0", "0", "1"])
ADDRESSES = (V4_A, V4_B, V6)
OUTPUTS = ("facts.json", "trees.jsonl", "top_trees.json", "report.html")


def _rows():
    """Four agents pass one code word along on one page; every optional label holds an address."""
    return [{"id": "r%d" % i, "agent": "Cutting" + "abcd"[i], "time": 100 + 100 * i, "place": "garden/bench",
             "text": "LANTERNBUD seen %d" % i, "group": V4_A if i % 2 == 0 else V4_B,
             "family": "bed-" + V4_A, "time_grade": V6} for i in range(4)]


def _write_ndjson(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))


def _write_gz(path, objs):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for obj in objs:
            fh.write(json.dumps(obj) + "\n")


def _run(args):
    out = io.StringIO()
    code = P.main(args + ["--shuffles", "0"], out=out)
    return code, out.getvalue()


class GroupFamilyGrade(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-redteam-privacy-")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _check_run(self, log, out_dir):
        code, printed = _run(["--adapter", "generic", "--input", log, "--out", out_dir])
        self.assertEqual(code, 0, printed)
        self.assertEqual(sorted(os.listdir(out_dir)), sorted(OUTPUTS))
        for name in OUTPUTS:
            with open(os.path.join(out_dir, name), encoding="utf-8") as fh:
                text = fh.read()
            for address in ADDRESSES:
                self.assertNotIn(address, text, name)
        for address in ADDRESSES:
            self.assertNotIn(address, printed)
        with open(os.path.join(out_dir, "facts.json"), encoding="utf-8") as fh:
            facts = json.load(fh)
        self.assertEqual(facts["load"]["rows"], 4)
        self.assertEqual(facts["load"]["agents"], 4)
        self.assertEqual(facts["cascade"]["links_named"], 3)  # the chain is still found
        self.assertEqual(facts["load"]["groups"], {L.IP_MASK: 4})  # two masked groups are one label; no row lost
        return facts

    def test_json_lines_log(self):
        log = os.path.join(self.tmp, "labels.ndjson")
        _write_ndjson(log, _rows())
        self._check_run(log, os.path.join(self.tmp, "out-ndjson"))

    def test_csv_log(self):
        rows = _rows()
        log = os.path.join(self.tmp, "labels.csv")
        fields = ["id", "agent", "time", "place", "text", "group", "family", "time_grade"]
        with open(log, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(",".join(fields) + "\n")
            for row in rows:
                fh.write(",".join(str(row[f]) for f in fields) + "\n")
        self._check_run(log, os.path.join(self.tmp, "out-csv"))

    def test_rows_keep_everything_else(self):
        log = os.path.join(self.tmp, "rows.ndjson")
        _write_ndjson(log, _rows())
        rows, stats = L.load_generic(log)
        self.assertEqual([r.group for r in rows], [L.IP_MASK] * 4)
        self.assertEqual([r.family for r in rows], ["bed-" + L.IP_MASK] * 4)
        self.assertEqual([r.time_grade for r in rows], [L.IP_MASK] * 4)
        self.assertEqual([r.agent for r in rows], ["Cuttinga", "Cuttingb", "Cuttingc", "Cuttingd"])
        self.assertEqual([r.place for r in rows], ["garden/bench"] * 4)
        self.assertEqual(stats["rows"], 4)

    def test_plain_labels_are_untouched(self):
        log = os.path.join(self.tmp, "plain.ndjson")
        rows = _rows()
        for i, row in enumerate(rows):
            row.update(group="plot %d.5" % (i % 2), family="bed 1.2.3", time_grade="exact")
        _write_ndjson(log, rows)
        loaded, stats = L.load_generic(log)
        self.assertEqual(sorted(set(r.group for r in loaded)), ["plot 0.5", "plot 1.5"])
        self.assertEqual(set(r.family for r in loaded), {"bed 1.2.3"})
        self.assertEqual(stats["groups"], {"plot 0.5": 2, "plot 1.5": 2})

    def test_wiki_adapter(self):
        folder = os.path.join(self.tmp, "wiki")
        os.makedirs(folder)
        _write_gz(os.path.join(folder, "revisions.jsonl.gz"), [
            {"rev_id": "s%d" % i, "page_id": "garden", "wiki": V4_A if i else V4_B, "label": "Graft%d" % i,
             "time": 100 + i, "time_grade": V6, "uncertainty_seconds": 0, "body": "made-up leaf %d" % i,
             "hunks": [], "diff_base": None} for i in range(2)])
        _write_gz(os.path.join(folder, "pages.jsonl.gz"), [{"page_id": "garden", "page_family": "bed-" + V4_A}])
        _write_gz(os.path.join(folder, "labels.jsonl.gz"), [])
        rows, stats = L.load_wiki(folder)
        self.assertEqual([(r.group, r.family, r.time_grade) for r in rows],
                         [(L.IP_MASK, "bed-" + L.IP_MASK, L.IP_MASK)] * 2)
        self.assertEqual(stats["groups"], {L.IP_MASK: 2})
        self.assertEqual(stats["time_grades"], {L.IP_MASK: 2})
        self.assertEqual(stats["rows"], 2)

    def test_mask_label_leaves_other_values_alone(self):
        self.assertIsNone(L.mask_label(None))
        self.assertEqual(L.mask_label(7), 7)
        self.assertEqual(L.mask_label("dse"), "dse")
        self.assertEqual(L.mask_label(""), "")
        self.assertEqual(L.mask_label("wiki " + V4_A), "wiki " + L.IP_MASK)


class InputName(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pz-redteam-name-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _facts(self, name):
        log = os.path.join(self.tmp, name)
        data = (json.dumps({"agent": "Graft", "time": 100, "place": "garden", "text": "made-up heading"})
                + "\n").encode("utf-8")
        with open(log, "wb") as fh:
            fh.write(data)
        out_dir = os.path.join(self.tmp, "out")
        code, printed = _run(["--adapter", "generic", "--input", log, "--out", out_dir])
        self.assertEqual(code, 0, printed)
        with open(os.path.join(out_dir, "facts.json"), encoding="utf-8") as fh:
            facts = json.load(fh)
        with open(log, "rb") as fh:
            self.assertEqual(fh.read(), data)  # the input is not touched
        record = facts["input"]["files"][0]
        self.assertEqual(record["bytes"], len(data))
        self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())
        return record["name"], out_dir

    def test_long_non_ascii_name_is_escaped_and_cut(self):
        name, _out = self._facts("Märchen-" + "bud" * 29 + ".ndjson")
        self.assertLessEqual(len(name), SHORT_MAX)
        self.assertTrue(all(32 <= ord(ch) < 127 for ch in name), name)
        self.assertTrue(name.startswith("M\\u00e4rchen-bud"), name)

    def test_address_in_name_is_masked_everywhere(self):
        name, out_dir = self._facts("log-%s.ndjson" % V4_A)
        self.assertEqual(name, "log-%s.ndjson" % L.IP_MASK)
        for output in OUTPUTS:
            with open(os.path.join(out_dir, output), encoding="utf-8") as fh:
                self.assertNotIn(V4_A, fh.read(), output)

    def test_plain_name_is_unchanged(self):
        name, _out = self._facts("chat.ndjson")
        self.assertEqual(name, "chat.ndjson")


if __name__ == "__main__":
    unittest.main()
