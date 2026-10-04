"""Pin the pre-registration texts in docs/prereg, byte for byte.

Standard library unittest only; no data needed. A pre-registration is evidence only while its bytes are the
bytes that were hashed before the run, so:

1. every file in docs/prereg has the SHA-256 recorded here, with LF line endings in any checkout;
2. there is no file in docs/prereg that is not listed here, and none listed that is missing;
3. the README lists every file with the first 8 hex of its hash;
4. the plant seeds the README's inject command uses are the ones the PZ-4 text fixed.
"""

import hashlib
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = os.path.join(ROOT, "docs", "prereg")

# file -> SHA-256 of its bytes as written (and as hashed at the time, before the run it describes)
PINNED = {
    "PAGE-INSTRUCTION.txt": "467c3e9280085ef819dfebf3b220c95f6c3e7434bf6b1ad27d7e6bddd972ddf6",
    "PREREG-PZ4.txt": "7d8bc48c94d4f4710b90bdf43a96eb1fe0621497613ac2a957024e9cd381eec0",
    "PREREG-PZ5.txt": "c3666fe449c642e07dc933712a2027422efb76e7936f379bc974f948b0301c06",
    "PREREG-PZ5x.txt": "abb0ba2f038a1dba391d4c98cc49cbe981bc539d047ccfb5cc73d988ce649569",
}


def read_bytes(name):
    with open(os.path.join(PREREG, name), "rb") as fh:
        return fh.read()


class Prereg(unittest.TestCase):
    def test_the_folder_holds_exactly_the_pinned_files(self):
        self.assertEqual(sorted(os.listdir(PREREG)), sorted(PINNED))

    def test_bytes_are_pinned(self):
        for name, sha in sorted(PINNED.items()):
            data = read_bytes(name)
            self.assertEqual(data.count(b"\r"), 0, "%s has CR bytes: the checkout rewrote line endings" % name)
            self.assertEqual(hashlib.sha256(data).hexdigest(), sha, "%s is not the text that was hashed" % name)

    def test_readme_lists_every_file_with_its_hash(self):
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
            readme = fh.read()
        for name, sha in sorted(PINNED.items()):
            row = next((line for line in readme.splitlines() if line.startswith("| `docs/prereg/%s` |" % name)), None)
            self.assertIsNotNone(row, "README has no row for docs/prereg/%s" % name)
            self.assertIn("`%s`" % sha[:8], row)

    def test_inject_seeds_in_the_readme_are_the_pre_registered_ones(self):
        text = read_bytes("PREREG-PZ4.txt").decode("ascii")
        fixed = re.search(r"REPLICATES \(the 5 named seeds[^:]*\): ([0-9, ]+)\.", text)
        self.assertIsNotNone(fixed)
        seeds = fixed.group(1).replace(" ", "")
        self.assertEqual(seeds, "20261041,20261042,20261043,20261044,20261045")
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
            self.assertIn("--plant-seeds %s " % seeds, fh.read())

    def test_each_text_says_it_was_written_before_its_numbers(self):
        for name in sorted(PINNED):
            head = read_bytes(name).decode("ascii")[:900]
            self.assertRegex(head, r"BEFORE any|BEFORE the|before any", name)


if __name__ == "__main__":
    unittest.main()
