"""Tests for the label and token scan (tools/labelscan.py). Made-up labels and tokens only."""

import gzip
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

_spec = importlib.util.spec_from_file_location("labelscan", os.path.join(ROOT, "tools", "labelscan.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

LABELS = [{"label": "Wren", "is_human_handle": False}, {"label": "Host Person", "is_human_handle": True},
          {"label": "  ", "is_human_handle": False}, {"label": "Café Bot", "is_human_handle": False}]
TOKENS = ["host:secret.example", "convention:control:HALT", "phrase:green tea leaf", "convention:prefix:"]
SECRET_WORDS = ("Wren", "Host Person", "secret.example", "HALT", "green tea leaf", "Caf")


def svg(*texts, attr=""):
    body = "".join('<text x="1" y="%d"%s>%s</text>' % (20 * (i + 1), attr, t) for i, t in enumerate(texts))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="100" '
            'height="100"><title>figure</title>%s</svg>\n' % body)


class LabelScan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pz-labelscan-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.data = os.path.join(self.tmp, "data")
        os.makedirs(self.data)
        with gzip.open(os.path.join(self.data, "labels.jsonl.gz"), "wt", encoding="utf-8") as fh:
            for row in LABELS:
                fh.write(json.dumps(dict(row, secret_field="never read")) + "\n")
        self.trees = os.path.join(self.tmp, "trees.jsonl")
        with open(self.trees, "w", encoding="utf-8") as fh:
            for token in TOKENS:
                fh.write(json.dumps({"token": token, "first_user": "Wren"}) + "\n")

    def scan(self, text, name="fig.svg"):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        buf = io.StringIO()
        code = S.main(["--data", self.data, "--trees", self.trees, path], out=buf)
        out = buf.getvalue()
        for word in SECRET_WORDS:
            self.assertNotIn(word, out)  # the scan never prints what it found
        return code, out

    def test_clean_figure(self):
        code, out = self.scan(svg("Can it see?", "pairs ÷ order-null mean", "0.995 · inside the band"))
        self.assertEqual(code, 0)
        self.assertIn("OK (0 agent labels, 0 token text)", out)
        self.assertIn("3 labels (1 flagged human, 1 blank skipped), 4 tokens", out)

    def test_label_found_as_a_whole_word_any_case(self):
        code, out = self.scan(svg("seed: wren, 3 pairs"))
        self.assertEqual(code, 1)
        self.assertIn("1 label hits, 0 token hits", out)
        self.assertIn("line:length 2:4", out)  # line 1 is the <title>

    def test_inside_a_longer_word_is_not_a_hit(self):
        code, _out = self.scan(svg("Wrenfield and wrens", "HALTING"))
        self.assertEqual(code, 0)

    def test_attributes_are_not_visible_text(self):
        code, _out = self.scan(svg("plain", attr=' id="Wren" class="secret.example"'))
        self.assertEqual(code, 0)

    def test_human_and_escaped_labels(self):
        self.assertEqual(self.scan(svg("by Host Person"))[0], 1)
        self.assertEqual(self.scan(svg("by Caf\\u00e9 Bot"))[0], 1)  # the escaped form, as safe.short writes it

    def test_tokens_after_their_prefix_and_last_parts(self):
        for text in ("see secret.example now", "HALT", "a green tea leaf"):
            code, out = self.scan(svg(text))
            self.assertEqual(code, 1, text)
            self.assertIn("0 label hits, 1 token hits", out)

    def test_other_files_scan_whole_text(self):
        code, _out = self.scan("notes: the halt word\n", name="notes.md")
        self.assertEqual(code, 1)

    def test_setup_error(self):
        buf = io.StringIO()
        code = S.main(["--data", os.path.join(self.tmp, "nowhere"), "--trees", self.trees, self.trees], out=buf)
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
