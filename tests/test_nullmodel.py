"""Tests for patient_zero/nullmodel.py (the label-shuffle baseline).

unittest, standard library only. Every name, host and line here is made up,
except in RealData, which runs only when PZ_DATA names the collusion.wiki
export folder and reports counts only.

* CounterMatchesCascade: with the true labels, the fast counter gives every
  adopter the same status and source as patient_zero.cascade.
* Shuffle: labels move only inside a day (or a place and day), blank and
  human rows never move, the same seed gives the same numbers.
* PlantedAgainstShuffle: the planted chain and star are never reached by a
  shuffle; the one-room cascade is (a stated limit of any label shuffle).
* RandomLog: in a log whose labels are random already, the observed count
  sits inside the shuffle spread.
* Summaries: mean, spread, percentiles, ratio and the empirical p-value.
"""

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

from patient_zero import cascade as C  # noqa: E402
from patient_zero import load as L  # noqa: E402
from patient_zero import nullmodel as N  # noqa: E402
from patient_zero import tokens as T  # noqa: E402
from tests import test_planted as TP  # noqa: E402

M = 60
DAY = 86400
HOSTS = ("alder-mill.net", "brine-works.org", "copper-gate.io", "dune-relay.net", "ember-yard.org",
         "fjord-lamp.io", "grove-kiln.net", "heron-dock.org")


def random_log(seed, n=600, agents=40, places=12, days=3):
    """A log whose agent labels are random already: who saw what has no bearing on who wrote what."""
    rng = random.Random(seed)
    specs = []
    for _ in range(n):
        t = rng.uniform(0, days * DAY)
        agent = "ag%02d" % rng.randrange(agents)
        place = "room/%02d" % rng.randrange(places)
        text = ("notes on %s today" % rng.choice(HOSTS)) if rng.random() < 0.3 else "plain status update"
        specs.append((t, agent, place, text))
    specs.sort()
    return TP.rows_of([(a, t, p, text, 1) for t, a, p, text in specs])


def three_day_rows():
    return TP.rows_of([
        ("Ana", 0, "r/1", "notes at tidewell.org"),
        ("Ben", 5 * M, "r/1", "hello"),
        (None, 6 * M, "r/2", "tidewell.org listed", 0, L.ROLE_BLANK),
        ("Cal", 7 * M, "r/2", "tidewell.org seen"),
        ("Ben", DAY + 5 * M, "r/3", "tidewell.org confirmed"),
        ("Dee", DAY + 9 * M, "r/3", "tidewell.org too"),
        ("Ana", DAY + 20 * M, "r/1", "back again"),
        (None, DAY + 30 * M, "r/1", "a human note", 0, L.ROLE_HUMAN),
        ("Eli", 2 * DAY + 1 * M, "r/3", "reading r3"),
        ("Cal", 2 * DAY + 2 * M, "r/4", "tidewell.org mirror"),
        ("Eli", 2 * DAY + 4 * M, "r/4", "tidewell.org here"),
        ("Dee", 2 * DAY + 6 * M, "r/4", "tidewell.org again"),
    ])


def fixture_logs():
    """(name, rows) for every made-up fixture."""
    out = []
    tmp = tempfile.mkdtemp(prefix="pz-null-")
    try:
        out.append(("chat", L.load("generic", TP.write_chat(tmp))[0]))
        wiki_dir = os.path.join(tmp, "wiki")
        os.makedirs(wiki_dir)
        TP._write_gz(os.path.join(wiki_dir, "revisions.jsonl.gz"), [
            {"rev_id": rid, "page_id": page, "wiki": "alpha", "label": label, "time": TP.iso(t),
             "time_grade": "reqlog", "uncertainty_seconds": 1, "body": body, "hunks": hunks,
             "diff_base": base}
            for rid, page, label, t, body, hunks, base in TP.WIKI
        ])
        TP._write_gz(os.path.join(wiki_dir, "pages.jsonl.gz"), [
            {"page_id": "wiki/Harbour", "page_family": "relay-coordination"},
            {"page_id": "wiki/Garden", "page_family": "probe-test"},
            {"page_id": "wiki/Library", "page_family": "source-cache-url-list"},
        ])
        out.append(("wiki", L.load("wiki", wiki_dir)[0]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out.append(("example", L.load("generic", os.path.join(ROOT, "examples", "fake_log.ndjson"))[0]))
    out.append(("three_days", three_day_rows()))
    out.append(("clock_shift", TP.rows_of([
        ("Ana", 0, "r/1", "notes at tidewell.org", 1),
        ("Ben", 100, "r/1", "tidewell.org confirmed", 1),
        ("Cal", 100.5, "r/1", "tidewell.org for me too", 1),
    ])))
    out.append(("visit_in_clock", TP.rows_of([
        ("Ana", 0, "r/1", "notes at tidewell.org", 5),
        ("Ben", 3, "r/1", "hi", 5),
        ("Ben", 10 * M, "r/2", "tidewell.org works", 5),
        ("Cal", 20 * M, "r/3", "tidewell.org here", 5),
    ])))
    out.append(("random", random_log(5, n=300)))
    return out


def assert_matches_cascade(test, rows, label):
    tokens, _ = T.find_tokens(rows)
    trees, stats = C.build_trees(rows, tokens)
    ix = N.build_index(rows, tokens)
    count = N.count_links(ix, ix.agent, detail=True)
    with test.subTest(log=label):
        test.assertEqual(count.status, stats["status"])
        test.assertEqual(count.trees_with_named, stats["trees_with_named_links"])
        for t, key in enumerate(ix.keys):
            tree = trees[key]
            mine = [(ix.agent_names[a], status, src) for a, status, src in count.detail[t]]
            want = [(a.agent, a.status, a.link.source_row if a.link is not None else -1)
                    for a in tree.adoptions]
            test.assertEqual(mine, want)
            test.assertEqual(count.named[t], sum(1 for a in tree.adoptions if a.status == C.STATUS_LINKED))


class CounterMatchesCascade(unittest.TestCase):
    def test_every_fixture(self):
        for label, rows in fixture_logs():
            assert_matches_cascade(self, rows, label)

    def test_parallel_fixture(self):
        rows = TP.rows_of([
            ("Wes", -1 * M, "room/a", "morning all"),
            ("Uma", 0, "room/a", "found ferrylight.net, it hosts the list"),
            ("Vic", 10 * M, "room/b", "ferrylight.net seems to have the list"),
            ("Wes", 20 * M, "room/c", "list copy at ferrylight.net"),
            ("Vic", 25 * M, "room/a", "hello room a"),
            ("Xan", 30 * M, "room/b", "catching up on this room"),
            ("Xan", 40 * M, "room/d", "using ferrylight.net now"),
        ])
        assert_matches_cascade(self, rows, "parallel")

    def test_pairs_follow_named_links(self):
        rows = [r._replace(family="fam-" + r.place.split("/")[-1]) for r in three_day_rows()]
        tokens, _ = T.find_tokens(rows)
        trees, stats = C.build_trees(rows, tokens)
        ix = N.build_index(rows, tokens)
        pairs = Counter()
        N.count_links(ix, ix.agent, pairs=pairs)
        want = Counter()
        for tree in trees.values():
            for link in tree.links:
                if link.source is not None:
                    want[(rows[link.source_row].family, rows[link.target_row].family)] += 1
        got = Counter({(ix.families[s], ix.families[d]): c for (s, d), c in pairs.items()})
        self.assertEqual(got, want)
        self.assertEqual(sum(got.values()), stats["links_named"])


class Shuffle(unittest.TestCase):
    def setUp(self):
        self.rows = three_day_rows()
        self.tokens, _ = T.find_tokens(self.rows)
        self.ix = N.build_index(self.rows, self.tokens)

    def _by_group(self, labels, key):
        out = {}
        for i in self.ix.agent_rows:
            out.setdefault(key(i), Counter())[labels[i]] += 1
        return out

    def test_labels_move_only_within_a_day(self):
        day = lambda i: int(self.ix.times[i] // N.DAY_SECONDS)  # noqa: E731
        rng = random.Random(3)
        want = self._by_group(self.ix.agent, day)
        moved = 0
        for _ in range(20):
            labels = N.shuffle_labels(self.ix, rng)
            self.assertEqual(self._by_group(labels, day), want)
            moved += sum(1 for i in self.ix.agent_rows if labels[i] != self.ix.agent[i])
        self.assertGreater(moved, 0)

    def test_blank_and_human_rows_never_move(self):
        rng = random.Random(4)
        fixed = [i for i, a in enumerate(self.ix.agent) if a < 0]
        self.assertEqual(len(fixed), 2)
        for _ in range(20):
            labels = N.shuffle_labels(self.ix, rng)
            self.assertEqual([labels[i] for i in fixed], [-1, -1])
            self.assertTrue(all(labels[i] >= 0 for i in self.ix.agent_rows))

    def test_place_day_scope_keeps_every_agents_pages(self):
        key = lambda i: (self.ix.place[i], int(self.ix.times[i] // N.DAY_SECONDS))  # noqa: E731
        groups = N.shuffle_groups(self.ix, N.SCOPE_PLACE_DAY)
        rng = random.Random(5)
        want = self._by_group(self.ix.agent, key)
        for _ in range(20):
            labels = N.shuffle_labels(self.ix, rng, groups)
            self.assertEqual(self._by_group(labels, key), want)
        null = N.run(self.rows, self.tokens, shuffles=20, scope=N.SCOPE_PLACE_DAY)
        self.assertEqual(set(null.agent_place_pairs), {null.observed.agent_place_pairs})

    def test_bad_scope_and_negative_shuffles(self):
        with self.assertRaises(ValueError):
            N.shuffle_groups(self.ix, "week")
        with self.assertRaises(ValueError):
            N.run(self.rows, self.tokens, shuffles=-1)

    def test_same_seed_same_numbers(self):
        a = N.run(self.rows, self.tokens, shuffles=30, seed=7)
        b = N.run(self.rows, self.tokens, shuffles=30, seed=7)
        self.assertEqual(a.named, b.named)
        self.assertEqual(a.status, b.status)
        self.assertEqual(a.pairs_sum, b.pairs_sum)
        self.assertEqual(N.facts_section(a), N.facts_section(b))

    def test_other_seed_other_labels(self):
        rows = random_log(9, n=200)
        tokens, _ = T.find_tokens(rows)
        ix = N.build_index(rows, tokens)
        self.assertNotEqual(N.shuffle_labels(ix, random.Random(1)), N.shuffle_labels(ix, random.Random(2)))

    def test_zero_shuffles(self):
        null = N.run(self.rows, self.tokens, shuffles=0)
        section = N.facts_section(null)
        self.assertEqual(section["shuffles"], 0)
        self.assertIsNone(section["links_named"]["mean"])
        self.assertIsNone(section["trees_beating_every_shuffle"])
        self.assertEqual(N.per_token(null)[null.index.keys[0]], (None, None))


class PlantedAgainstShuffle(unittest.TestCase):
    """SWARM.md: the planted cascades must be found, and the shuffle must not find them."""

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.mkdtemp(prefix="pz-null-")
        try:
            cls.rows, _ = L.load("generic", TP.write_chat(tmp))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        cls.tokens, _ = T.find_tokens(cls.rows)
        cls.null = N.run(cls.rows, cls.tokens, shuffles=100)
        cls.per_token = N.per_token(cls.null)
        cls.section = N.facts_section(cls.null)

    def test_chain_and_star_never_reached_by_a_shuffle(self):
        for key in ("host:lantern-depot.io", "convention:control:BLUEHERON"):
            with self.subTest(token=key):
                t = self.null.index.keys.index(key)
                self.assertEqual(self.null.observed.named[t], 3)
                mean, ge = self.per_token[key]
                self.assertEqual(ge, 0)
                self.assertLess(mean, 1.5)

    def test_planted_total_beats_every_shuffle(self):
        links = self.section["links_named"]
        self.assertEqual(links["observed"], 8)
        self.assertEqual(links["shuffles_ge_observed"], 0)
        self.assertGreater(links["ratio"], 2)

    def test_one_room_cascade_is_reachable_by_chance(self):
        """All four users post in one room, so any labelling links them: a stated limit."""
        mean, ge = self.per_token["file:quill_map.csv"]
        self.assertGreater(ge, 10)
        self.assertGreater(mean, 1)


class RandomLog(unittest.TestCase):
    def test_observed_inside_shuffle_spread(self):
        for seed in (11, 12, 13):
            with self.subTest(seed=seed):
                rows = random_log(seed)
                tokens, _ = T.find_tokens(rows)
                trees, stats = C.build_trees(rows, tokens)
                null = N.run(rows, tokens, shuffles=100)
                links = N.facts_section(null, stats["links_named"])["links_named"]
                self.assertTrue(N.facts_section(null, stats["links_named"])["observed_matches_cascade"])
                self.assertGreater(links["observed"], 0)
                self.assertLess(abs(links["z"]), 3)
                self.assertTrue(links["min"] <= links["observed"] <= links["max"])


class Summaries(unittest.TestCase):
    def test_summarize(self):
        s = N.summarize([4, 1, 3, 2], 5)
        self.assertEqual(s["mean"], 2.5)
        self.assertEqual(s["sd"], 1.291)
        self.assertEqual((s["min"], s["max"]), (1, 4))
        self.assertEqual(s["p05"], 1.15)
        self.assertEqual(s["median"], 2.5)
        self.assertEqual(s["p95"], 3.85)
        self.assertEqual(s["ratio"], 2.0)
        self.assertEqual(s["shuffles_ge_observed"], 0)
        self.assertEqual(s["p_value"], 0.2)
        self.assertEqual(s["z"], 1.94)

    def test_summarize_edges(self):
        s = N.summarize([0, 0], 0)
        self.assertIsNone(s["ratio"])
        self.assertIsNone(s["z"])
        self.assertEqual(s["shuffles_ge_observed"], 2)
        one = N.summarize([3], 3)
        self.assertEqual((one["sd"], one["p05"], one["p95"]), (0.0, 3, 3))
        empty = N.summarize([], 3)
        self.assertIsNone(empty["mean"])
        self.assertEqual(empty["shuffles"], 0)

    def test_percentile(self):
        self.assertIsNone(N.percentile([], 0.5))
        self.assertEqual(N.percentile([10], 0.95), 10)
        self.assertEqual(N.percentile([0, 10], 0.25), 2.5)

    def test_groups_and_shares(self):
        rows = three_day_rows()
        tokens, _ = T.find_tokens(rows)
        null = N.run(rows, tokens, shuffles=10)
        every = range(len(null.index.keys))
        self.assertEqual(N.group_values(null, every), [s["linked"] for s in null.status])
        self.assertEqual(N.group_summary(null, [])["observed"], 0)
        status = {"first": 2, "linked": 3, "unnamed": 1, "clock_dropped": 0, "parallel": 2}
        self.assertEqual(N.adopters(status), 8)
        self.assertEqual(N.linked_share(status), 0.5)
        self.assertEqual(N.linked_share({k: 0 for k in N.STATUS_KEYS}), 0.0)


@unittest.skipUnless(os.environ.get("PZ_DATA"), "set PZ_DATA to the collusion.wiki export folder")
class RealData(unittest.TestCase):
    """The fast counter against the cascade on all 4 wikis, every adopter (counts only on failure)."""

    def test_counter_matches_cascade_on_real_data(self):
        rows, _ = L.load("wiki", os.environ["PZ_DATA"])
        tokens, _ = T.find_tokens(rows)
        trees, stats = C.build_trees(rows, tokens)
        ix = N.build_index(rows, tokens)
        count = N.count_links(ix, ix.agent, detail=True)
        self.assertEqual(count.status, stats["status"])
        mismatched = 0
        for t, key in enumerate(ix.keys):
            mine = [(ix.agent_names[a], s, src) for a, s, src in count.detail[t]]
            want = [(a.agent, a.status, a.link.source_row if a.link is not None else -1)
                    for a in trees[key].adoptions]
            mismatched += mine != want
        self.assertEqual(mismatched, 0)
        null = N.run(rows, tokens, shuffles=2, index=ix)
        self.assertEqual(len(null.named), 2)
        self.assertTrue(N.facts_section(null, stats["links_named"])["observed_matches_cascade"])


if __name__ == "__main__":
    unittest.main()
