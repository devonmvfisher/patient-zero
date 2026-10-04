"""Pin the shipped example so the README's core_sha256 holds in any checkout.

Standard library unittest only. The example log is made up; no export data.

``core_sha256`` hashes the input file's bytes, so a checkout that rewrites
line endings (git ``core.autocrlf=true`` without ``.gitattributes``) changes
the example's hash and the run's core_sha256. These tests fail on such a
checkout, and they fail when a code change moves the example's core_sha256
without the README being updated.
"""

import hashlib
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

EXAMPLE = os.path.join(ROOT, "examples", "planted_chat.ndjson")
README = os.path.join(ROOT, "README.md")
GITATTRIBUTES = os.path.join(ROOT, ".gitattributes")

# The shipped example, LF line endings, as committed.
EXAMPLE_BYTES = 3428
EXAMPLE_SHA256 = "205cf6a7f8c41dfbb8bb01432f5aa14027805c7ca70b005bf17bcb26bfeccbb9"

CORE_RE = re.compile(r"core_sha256: ([0-9a-f]{64})")


def readme_core():
    with open(README, "r", encoding="utf-8") as fh:
        values = sorted(set(CORE_RE.findall(fh.read())))
    return values


class ExamplePinned(unittest.TestCase):
    def test_example_bytes_pinned(self):
        with open(EXAMPLE, "rb") as fh:
            data = fh.read()
        self.assertEqual(data.count(b"\r"), 0,
                         "example has CR bytes: the checkout rewrote line endings (see .gitattributes)")
        self.assertEqual(len(data), EXAMPLE_BYTES)
        self.assertEqual(hashlib.sha256(data).hexdigest(), EXAMPLE_SHA256)

    def test_gitattributes_keeps_lf(self):
        with open(GITATTRIBUTES, "r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
        self.assertIn("* text=auto eol=lf", lines)
        self.assertIn("*.gz binary", lines)
        self.assertIn("*.png binary", lines)

    def test_readme_quotes_one_core(self):
        self.assertEqual(len(readme_core()), 1, "README must quote exactly one example core_sha256")


class ExampleRunMatchesReadme(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-pinned-")
        buf = io.StringIO()
        cls.code = P.main(["--adapter", "generic", "--input", EXAMPLE, "--out", cls.tmp], out=buf)
        cls.text = buf.getvalue()
        with open(os.path.join(cls.tmp, "facts.json"), encoding="utf-8") as fh:
            cls.facts = json.load(fh)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_input_section_is_the_pinned_file(self):
        self.assertEqual(self.code, 0)
        files = self.facts["input"]["files"]
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["name"], "planted_chat.ndjson")
        self.assertEqual(files[0]["bytes"], EXAMPLE_BYTES)
        self.assertEqual(files[0]["sha256"], EXAMPLE_SHA256)

    def test_printed_core_matches_readme(self):
        printed = CORE_RE.findall(self.text)
        self.assertEqual(len(printed), 1)
        self.assertEqual(printed[0], self.facts["core_sha256"])
        self.assertEqual([printed[0]], readme_core(),
                         "the example's core_sha256 changed: update the README value, or find why")


if __name__ == "__main__":
    unittest.main()
