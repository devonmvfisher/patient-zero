"""Tests for patient_zero/checks.py (order null, strict exposure, minimum lag, activity).

unittest, standard library only. Every name, host and line here is made up,
except in RealData, which runs only when PZ_DATA names the collusion.wiki
export folder and reports counts only.

* Reorder: each agent keeps its rows (page and text) and its own times per
  day; blank and human rows never move; the same seed gives the same order.
* Strict: the adoption save alone is not exposure under strict=True; an
  earlier save on the source's page is.
* MinLag: a link closer than the minimum lag moves to an older use or drops.
* OrderNullPlanted: the planted chain and star beat the order null; a log
  whose rows are random already sits inside its spread.
* Activity: the most active agent becomes exposure-only.
* Output: counts only, deterministic, and the command line guards --out.
"""

import io
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from patient_zero import checks as K  # noqa: E402
from patient_zero import load as L  # noqa: E402
from patient_zero import nullmodel as N  # noqa: E402
from patient_zero import tokens as T  # noqa: E402
from patient_zero.safe import token_id  # noqa: E402
from tests import test_nullmodel as TN  # noqa: E402
from tests import test_planted as TP  # noqa: E402

M = 60


def chat_rows():
    tmp = tempfile.mkdtemp(prefix="pz-checks-")
    try:
        rows, _ = L.load("generic", TP.write_chat(tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tokens, _ = T.find_tokens(rows)
    return rows, tokens


def statuses(rows, tokens, key, **kw):
    ix = N.build_index(rows, tokens)
    ix = K.with_min_lag(ix, kw.pop("min_lag", 0))
    count = N.count_links(ix, ix.agent, detail=True, **kw)
    t = ix.keys.index(key)
    return {ix.agent_names[a]: status for a, status, _src in count.detail[t]}


def drop_seconds(obj):
    if isinstance(obj, dict):
        return {k: drop_seconds(v) for k, v in obj.items() if k != "seconds"}
    return obj


class Reorder(unittest.TestCase):
    def test_agents_keep_rows_and_own_times(self):
        rows, tokens = chat_rows()
        new_rows, new_tokens = K.reorder(rows, tokens, random.Random(7))
        self.assertEqual(len(new_rows), len(rows))
        for agent in set(r.agent for r in rows if not r.exposure_only):
            with self.subTest(agent=agent):
                old = [r for r in rows if r.agent == agent]
                new = [r for r in new_rows if r.agent == agent]
                self.assertEqual(Counter((r.place, r.text, r.row_id) for r in old),
                                 Counter((r.place, r.text, r.row_id) for r in new))
                self.assertEqual(sorted(r.time for r in old), sorted(r.time for r in new))
        self.assertEqual([r.time for r in new_rows], sorted(r.time for r in new_rows))

    def test_token_rows_follow_the_moved_rows(self):
        rows, tokens = chat_rows()
        new_rows, new_tokens = K.reorder(rows, tokens, random.Random(3))
        for key, tok in tokens.items():
            with self.subTest(token=key):
                self.assertEqual(sorted(rows[i].row_id for i in tok.rows),
                                 sorted(new_rows[i].row_id for i in new_tokens[key].rows))

    def test_times_stay_inside_the_agents_day(self):
        rows = TN.three_day_rows()
        tokens, _ = T.find_tokens(rows)
        for seed in range(5):
            new_rows, _ = K.reorder(rows, tokens, random.Random(seed))
            by_id = {r.row_id: r for r in rows}
            for r in new_rows:
                self.assertEqual(int(r.time // K.DAY_SECONDS), int(by_id[r.row_id].time // K.DAY_SECONDS))

    def test_blank_and_human_rows_never_move(self):
        rows = TN.three_day_rows()
        tokens, _ = T.find_tokens(rows)
        fixed = {r.row_id: r.time for r in rows if r.exposure_only}
        self.assertTrue(fixed)
        for seed in range(5):
            new_rows, _ = K.reorder(rows, tokens, random.Random(seed), scope=K.SCOPE_AGENT)
            self.assertEqual({r.row_id: r.time for r in new_rows if r.exposure_only}, fixed)

    def test_same_seed_same_order(self):
        rows, tokens = chat_rows()
        a = K.reorder(rows, tokens, random.Random(11))[0]
        b = K.reorder(rows, tokens, random.Random(11))[0]
        self.assertEqual([r.row_id for r in a], [r.row_id for r in b])

    def test_bad_scope(self):
        rows, tokens = chat_rows()
        with self.assertRaises(ValueError):
            K.reorder(rows, tokens, random.Random(1), scope="week")


class Strict(unittest.TestCase):
    ROWS = [
        ("Ana", 0, "r/1", "notes at tidewell.org", 1),
        ("Ben", 10 * M, "r/1", "tidewell.org confirmed", 1),  # only evidence: this save itself
        ("Cal", 15 * M, "r/1", "reading the page", 1),  # an earlier save on the page
        ("Cal", 20 * M, "r/2", "tidewell.org mirrored here", 1),
    ]

    def test_adoption_save_alone_is_not_exposure_when_strict(self):
        rows = TP.rows_of(self.ROWS)
        tokens, _ = T.find_tokens(rows)
        key = "host:tidewell.org"
        self.assertEqual(statuses(rows, tokens, key), {"Ana": "first", "Ben": "linked", "Cal": "linked"})
        self.assertEqual(statuses(rows, tokens, key, strict=True),
                         {"Ana": "first", "Ben": "parallel", "Cal": "linked"})

    def test_default_count_is_unchanged(self):
        for label, rows in TN.fixture_logs():
            tokens, _ = T.find_tokens(rows)
            ix = N.build_index(rows, tokens)
            with self.subTest(log=label):
                self.assertEqual(N.count_links(ix, ix.agent).status,
                                 N.count_links(ix, ix.agent, strict=False).status)
                self.assertLessEqual(N.count_links(ix, ix.agent, strict=True).status["linked"],
                                     N.count_links(ix, ix.agent).status["linked"])


class MinLag(unittest.TestCase):
    ROWS = [
        ("Ana", 0, "r/1", "notes at tidewell.org", 1),
        ("Ben", 30, "r/1", "tidewell.org confirmed", 1),  # 30 s after Ana
        ("Cal", 10 * M, "r/1", "tidewell.org for me too", 1),
    ]

    def test_short_link_drops_and_credit_moves(self):
        rows = TP.rows_of(self.ROWS)
        tokens, _ = T.find_tokens(rows)
        key = "host:tidewell.org"
        self.assertEqual(statuses(rows, tokens, key), {"Ana": "first", "Ben": "linked", "Cal": "linked"})
        self.assertEqual(statuses(rows, tokens, key, min_lag=60),
                         {"Ana": "first", "Ben": "clock_dropped", "Cal": "linked"})
        ix = N.build_index(rows, tokens)
        got = K.observed_variants(ix, lags=(0, 60))
        self.assertEqual(got["default_min_lag_0s"]["links_named"], 2)
        self.assertEqual(got["default_min_lag_60s"]["links_named"], 1)

    def test_min_lag_zero_is_the_index_itself(self):
        rows = TP.rows_of(self.ROWS)
        tokens, _ = T.find_tokens(rows)
        ix = N.build_index(rows, tokens)
        self.assertIs(K.with_min_lag(ix, 0), ix)
        self.assertEqual(K.with_min_lag(ix, 60).unc, tuple(u + 30.0 for u in ix.unc))


class OrderNullPlanted(unittest.TestCase):
    def test_planted_cascades_beat_the_order_null(self):
        rows, tokens = chat_rows()
        lead = token_id("host:lantern-depot.io")
        got = K.order_null(rows, tokens, shuffles=40, seed=20261003, lead=lead)
        self.assertEqual(got["adopters"], N.adopters(N.count_links(N.build_index(rows, tokens),
                                                                   N.build_index(rows, tokens).agent).status))
        self.assertGreater(got["links_named"]["observed"], got["links_named"]["mean"])
        self.assertEqual(got["lead"]["observed"], 3)
        self.assertLess(got["lead"]["mean"], 3)
        star = statuses(rows, tokens, "convention:control:BLUEHERON")
        self.assertEqual(star, {"Eve": "first", "Finn": "linked", "Gus": "linked", "Hal": "linked"})

    def test_random_log_sits_inside_the_spread(self):
        rows = TN.random_log(9, n=300)
        tokens, _ = T.find_tokens(rows)
        got = K.order_null(rows, tokens, shuffles=40, seed=5)
        z = got["links_named"]["z"]
        self.assertIsNotNone(z)
        self.assertLess(abs(z), 3.5)

    def test_rejects_zero_shuffles(self):
        rows, tokens = chat_rows()
        with self.assertRaises(ValueError):
            K.order_null(rows, tokens, shuffles=0)


class Activity(unittest.TestCase):
    def test_most_active_agent_becomes_exposure_only(self):
        rows = TP.rows_of([
            ("Ana", 0, "r/1", "notes at tidewell.org", 1),
            ("Ana", 1 * M, "r/9", "hello", 1),
            ("Ana", 2 * M, "r/9", "hello again", 1),
            ("Ben", 10 * M, "r/1", "tidewell.org confirmed", 1),
            ("Cal", 20 * M, "r/1", "tidewell.org for me too", 1),
        ])
        tokens, _ = T.find_tokens(rows)
        ix = N.build_index(rows, tokens)
        got = K.without_top(rows, ix, shares=(0.01,))["top_1_pct"]
        self.assertEqual(got["agents_made_exposure_only"], 1)
        self.assertEqual(got["their_share_of_agent_rows"], 0.6)
        self.assertEqual(got["links_named"], 1)  # Cal from Ben
        self.assertEqual(got["links_from_exposure_only_rows"], 1)  # Ben from Ana, now unnamed
        self.assertEqual(got["adopters"], 2)


class Output(unittest.TestCase):
    def test_counts_only_and_deterministic(self):
        rows, tokens = chat_rows()
        a = K.run_checks(rows, tokens, shuffles=6, seed=1, lead=token_id("host:lantern-depot.io"))
        b = K.run_checks(rows, tokens, shuffles=6, seed=1, lead=token_id("host:lantern-depot.io"))
        self.assertEqual(drop_seconds(a), drop_seconds(b))
        text = json.dumps(a)
        for name in TP.CHAT_NAMES:
            self.assertNotIn('"%s"' % name, text)
        for key in tokens:
            self.assertNotIn(tokens[key].value, text)
        self.assertEqual(a["order_null"]["shuffles"], 6)
        self.assertEqual(a["order_null_strict"]["shuffles"], 2)

    def test_cli_runs_and_guards_out(self):
        tmp = tempfile.mkdtemp(prefix="pz-checks-cli-")
        try:
            path = TP.write_chat(tmp)
            buf = io.StringIO()
            target = os.path.join(tmp, "checks.json")
            self.assertEqual(K.main(["--adapter", "generic", "--input", path, "--shuffles", "3",
                                     "--out", target], out=buf), 0)
            with open(target, "r", encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["order_null"]["shuffles"], 3)
            bad = os.path.join(ROOT, "patient_zero", "checks-out.json")
            buf = io.StringIO()
            self.assertEqual(K.main(["--adapter", "generic", "--input", path, "--shuffles", "3",
                                     "--out", bad], out=buf), 2)
            self.assertIn("ERROR", buf.getvalue())
            self.assertFalse(os.path.exists(bad))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(os.environ.get("PZ_DATA"), "set PZ_DATA to the collusion.wiki export folder")
class RealData(unittest.TestCase):
    def test_order_null_keeps_adopters_and_reports_counts(self):
        rows, _ = L.load("wiki", os.environ["PZ_DATA"])
        tokens, _ = T.find_tokens(rows)
        got = K.order_null(rows, tokens, shuffles=2, seed=20261003)
        self.assertEqual(got["adopters"], 73176)
        self.assertEqual(got["links_named"]["observed"], 28250)
        self.assertEqual(got["shuffles"], 2)
        self.assertIn("z", got["links_named"])


if __name__ == "__main__":
    unittest.main()
