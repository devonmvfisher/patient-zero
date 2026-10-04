"""Red-team regression for the phrase filter's floor (unittest, standard library only).

Finding D05 of the lane x3 red team of ea867f1, re-cut for the merged tree: a
phrase is dropped as a template only when it is found in more than
``--max-phrase-share`` of the agent rows AND in more than 25 of them
(``PHRASE_CAP_FLOOR``). The README and the ``--help`` text gave the share
alone, so on a small log a reader expected phrases to be dropped that the
finder keeps. The rule itself is unchanged; these tests pin it and check that
both texts now give the floor. Every row is made up.
"""

import contextlib
import io
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from patient_zero import load as L  # noqa: E402
from patient_zero import pipeline as P  # noqa: E402
from patient_zero import tokens as T  # noqa: E402

PHRASE = "velvet clover lantern"
KEY = "phrase:" + PHRASE


def _rows(n):
    """n agents, one row each, all writing the same made-up phrase on one page."""
    return [L.Row(row_id="r%d" % i, agent="Graft%d" % i, role=L.ROLE_AGENT, time=100.0 * i, uncertainty=0.0,
                  time_grade=None, place="garden", family=None, group=None, text=PHRASE) for i in range(n)]


class PhraseFloor(unittest.TestCase):
    def test_small_log_keeps_a_phrase_every_agent_wrote(self):
        found, stats = T.find_tokens(_rows(3))  # 3 of 3 rows is far above the share, but below the floor
        self.assertIn(KEY, found)
        self.assertEqual(stats["params"]["phrase_cap_floor"], T.PHRASE_CAP_FLOOR)

    def test_floor_is_the_last_count_kept(self):
        kept, _ = T.find_tokens(_rows(T.PHRASE_CAP_FLOOR))
        dropped, stats = T.find_tokens(_rows(T.PHRASE_CAP_FLOOR + 1))
        self.assertIn(KEY, kept)
        self.assertNotIn(KEY, dropped)
        self.assertEqual(stats["classes"]["phrase"]["common_dropped"], 1)

    def test_above_the_floor_the_share_decides(self):
        found, _ = T.find_tokens(_rows(T.PHRASE_CAP_FLOOR + 1), max_phrase_share=1.0)
        self.assertIn(KEY, found)

    def test_help_gives_the_floor(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), self.assertRaises(SystemExit) as stopped:
            P.main(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        text = " ".join(stream.getvalue().split())
        self.assertIn("in more than %d of them" % T.PHRASE_CAP_FLOOR, text)

    def test_readme_gives_the_floor(self):
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
            readme = fh.read()
        line = [ln for ln in readme.splitlines() if ln.startswith("- **Tokens**")]
        self.assertEqual(len(line), 1)
        self.assertIn("`--max-phrase-share`", line[0])
        match = re.search(r"in more than (\d+) of them", line[0])
        self.assertIsNotNone(match, "the Tokens line must give the floor of the phrase cap")
        self.assertEqual(int(match.group(1)), T.PHRASE_CAP_FLOOR)


if __name__ == "__main__":
    unittest.main()
