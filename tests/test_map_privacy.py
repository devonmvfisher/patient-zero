"""Privacy regressions for --map on JSON logs (skeptic fix on top of PZ-10).

unittest, standard library only. Every name here is made up.

* A JSON human flag written as 1 or "yes" hides the speaker, as it does in
  CSV; an unclear flag stops the run instead of naming the speaker.
* An optional field mapped to a JSON object stops the run: str(object)
  would carry the object's whitelisted keys, a hidden handle among them,
  into the outputs.
"""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from patient_zero import load as L  # noqa: E402

HANDLE = "QuillHuman"
NAMES = (HANDLE, "Otter", "Moth", "Wren")


class JsonPrivacy(unittest.TestCase):
    def _load(self, objs, spec=None):
        fd, path = tempfile.mkstemp(suffix=".ndjson", prefix="pz-priv-")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("\n".join(json.dumps(o) for o in objs) + "\n")
            return L.load("generic", path, L.parse_map(spec))
        finally:
            os.remove(path)

    def _lines(self, flag):
        return [{"who": n, "is_h": (flag if n == HANDLE else 0), "at": i + 1, "pg": "p", "body": "t %d" % i}
                for i, n in enumerate(NAMES)]

    def test_mapped_human_flag_one_or_yes_hides_the_speaker(self):
        for flag in (1, "yes", "true", "1", True):
            rows, stats = self._load(self._lines(flag), "agent=who,human=is_h,time=at,place=pg,text=body")
            self.assertEqual(stats["human_rows"], 1, flag)
            self.assertNotIn(HANDLE, repr(rows), flag)

    def test_unclear_human_flag_stops_the_run(self):
        for flag in ("maybe", 2, {"x": 1}):
            with self.assertRaises(ValueError) as ctx:
                self._load(self._lines(flag), "agent=who,human=is_h,time=at,place=pg,text=body")
            self.assertIn("human must be", str(ctx.exception))
            self.assertNotIn(HANDLE, str(ctx.exception))

    def test_human_labels_for_peek_read_the_same_flags(self):
        fd, path = tempfile.mkstemp(suffix=".ndjson", prefix="pz-priv-")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("\n".join(json.dumps(o) for o in self._lines("yes")) + "\n")
            got = L.generic_human_labels(path, L.parse_map("agent=who,human=is_h,time=at,place=pg,text=body"))
        finally:
            os.remove(path)
        self.assertEqual(got, {HANDLE})

    def test_optional_field_on_an_object_stops_the_run(self):
        objs = [{"author": {"name": n, "human": n == HANDLE}, "at": i + 1, "pg": "p", "body": "t %d" % i}
                for i, n in enumerate(NAMES)]
        base = "agent=author.name,human=author.human,time=at,place=pg,text=body"
        for field in ("id", "group", "family", "time_grade"):
            with self.assertRaises(ValueError) as ctx:
                self._load(objs, base + ",%s=author" % field)
            self.assertIn("%s must be a string or a number" % field, str(ctx.exception))
            self.assertNotIn(HANDLE, str(ctx.exception))
        rows, stats = self._load(objs, base)  # the same log, mapped to the leaves, still loads
        self.assertEqual(stats["human_rows"], 1)
        self.assertNotIn(HANDLE, repr(rows))


if __name__ == "__main__":
    unittest.main()
