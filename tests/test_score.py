"""Tests for patient_zero/score.py: speed, influence, risky flags, page-family flow.

unittest, standard library only. Every name, host and line here is made up.
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from patient_zero import cascade as C  # noqa: E402
from patient_zero import nullmodel as N  # noqa: E402
from patient_zero import pipeline as P  # noqa: E402
from patient_zero import score as S  # noqa: E402
from patient_zero import tokens as T  # noqa: E402
from tests import test_planted as TP  # noqa: E402

M = 60


def ad(agent, t, status, parent=None):
    return C.Adoption(agent=agent, row=int(t), time=float(t), status=status, parent=parent, link=None)


def tree_of(adoptions, key="host:t.example-x.org", seed="auto"):
    adoptions = tuple(adoptions)
    if seed == "auto":
        parents = {a.parent for a in adoptions if a.parent is not None}
        seed = next((a.agent for a in adoptions if a.agent in parents), None)
    return C.Tree(token=key, cls="host", kind="host", adoptions=adoptions, links=(),
                  first_user=adoptions[0].agent, effective_seed=seed,
                  parallel_starts=tuple(a.agent for a in adoptions if a.status == C.STATUS_PARALLEL),
                  dropped=0, shifted=0, depth=0, places=1)


# A (seed) -> B -> C, A -> E; D started on its own -> F
TREE_1 = tree_of([
    ad("A", 0, C.STATUS_FIRST), ad("B", 60, C.STATUS_LINKED, "A"), ad("C", 180, C.STATUS_LINKED, "B"),
    ad("D", 200, C.STATUS_PARALLEL), ad("E", 600, C.STATUS_LINKED, "A"), ad("F", 900, C.STATUS_LINKED, "D"),
], key="host:one.example-x.org")
# A -> G -> B
TREE_2 = tree_of([
    ad("A", 0, C.STATUS_FIRST), ad("G", 10, C.STATUS_LINKED, "A"), ad("B", 20, C.STATUS_LINKED, "G"),
], key="host:two.example-x.org")
# a chain of 11: X00 -> X01 -> ... -> X10, 30 s apart
CHAIN = tree_of([ad("X00", 0, C.STATUS_FIRST)] +
                [ad("X%02d" % i, 30 * i, C.STATUS_LINKED, "X%02d" % (i - 1)) for i in range(1, 11)],
                key="host:chain.example-x.org")


class Speed(unittest.TestCase):
    def test_branch_minutes_counts_only_the_roots_branch(self):
        size, minutes = S.branch_minutes(TREE_1, "A")
        self.assertEqual(size, 4)  # A, B, C, E; D and F are another branch
        self.assertEqual(minutes, {3: 10.0, 10: None})
        size, minutes = S.branch_minutes(TREE_1, "D")
        self.assertEqual((size, minutes), (2, {3: None, 10: None}))

    def test_chain_reaches_tenth(self):
        size, minutes = S.branch_minutes(CHAIN, "X00")
        self.assertEqual(size, 11)
        self.assertEqual(minutes, {3: 1.5, 10: 5.0})

    def test_unknown_root(self):
        with self.assertRaises(ValueError):
            S.branch_minutes(TREE_1, "nobody")

    def test_tree_speed(self):
        self.assertEqual(S.tree_speed(CHAIN, "X00"), {
            "seed_branch": 11, "seed_min_to_3rd": 1.5, "seed_min_to_10th": 5.0,
            "branch_min_to_3rd": 1.5, "branch_min_to_10th": 5.0})
        lone = tree_of([ad("A", 0, C.STATUS_FIRST), ad("B", 9, C.STATUS_PARALLEL)])
        self.assertEqual(S.tree_speed(lone), {
            "seed_branch": 0, "seed_min_to_3rd": None, "seed_min_to_10th": None,
            "branch_min_to_3rd": None, "branch_min_to_10th": None})

    def test_speed_table(self):
        def rec(tid, to3, to10):
            return {"token_id": tid, "cls": "host", "kind": "host", "agents": 12, "seed_branch": 11,
                    "largest_branch": 11, "seed_min_to_3rd": to3, "seed_min_to_10th": to10,
                    "branch_min_to_3rd": to3, "branch_min_to_10th": to10}

        table = S.speed_table([rec("aaa", 2.0, 30.0), rec("bbb", 0.5, None), rec("ccc", 2000.0, 5000.0)])
        to10 = table["seed_to_10th"]
        self.assertEqual(to10["trees"], 2)
        self.assertEqual(to10["median_minutes"], 2515.0)
        self.assertEqual(to10["fastest"]["token_id"], "aaa")
        self.assertEqual(to10["buckets"], {"le_10_min": 0, "le_60_min": 1, "le_6_h": 0, "le_24_h": 0,
                                           "over_24_h": 1})
        to3 = table["seed_to_3rd"]
        self.assertEqual((to3["trees"], to3["median_minutes"], to3["fastest"]["minutes"]), (3, 2.0, 0.5))
        self.assertEqual(to3["buckets"]["le_10_min"], 2)
        empty = S.speed_table([rec("ddd", None, None)])["seed_to_10th"]
        self.assertEqual((empty["trees"], empty["median_minutes"], empty["fastest"]), (0, None, None))


class Influence(unittest.TestCase):
    def setUp(self):
        self.table = S.influence_table({TREE_1.token: TREE_1, TREE_2.token: TREE_2})

    def test_counts(self):
        top = {row["agent"]: row for row in self.table["top"]}
        self.assertEqual(top["A"], {"rank": 1, "agent": "A", "adopters": 3, "downstream": 5,
                                    "trees_seeded": 2, "trees_first": 2, "trees_joined": 2,
                                    "parallel_starts": 0, "agents_reached": 3})
        self.assertEqual((top["B"]["adopters"], top["B"]["trees_joined"], top["B"]["agents_reached"]), (1, 2, 1))
        self.assertEqual((top["D"]["parallel_starts"], top["D"]["trees_seeded"]), (1, 0))

    def test_ranking_and_ties(self):
        self.assertEqual([row["agent"] for row in self.table["top"]], ["A", "B", "D", "G", "C", "E", "F"])

    def test_summary(self):
        t = self.table
        self.assertEqual((t["agents"], t["agents_with_adopters"], t["named_links"]), (7, 4, 6))
        self.assertEqual((t["top_1_share"], t["top_10_share"]), (0.5, 1.0))
        self.assertEqual((t["top_1pct_agents"], t["top_1pct_share"]), (1, 0.5))
        self.assertEqual(t["median_adopters_of_agents_with_adopters"], 1.0)

    def test_labels_escaped_and_top_cut(self):
        tree = tree_of([ad("Zoë", 0, C.STATUS_FIRST), ad("Ivo", 5, C.STATUS_LINKED, "Zoë")])
        table = S.influence_table({tree.token: tree}, top=1)
        self.assertEqual(len(table["top"]), 1)
        self.assertEqual(table["top"][0]["agent"], "Zo\\u00eb")
        self.assertEqual(S.influence_table({})["top"], [])


def flow_rows():
    fam = {"r/1": "alpha", "r/2": "beta", "r/3": None}
    rows = TP.rows_of([
        ("Ana", 0, "r/1", "notes at tidewell.org"),
        ("Ben", 5 * M, "r/1", "hello"),
        ("Ben", 10 * M, "r/2", "tidewell.org seen"),
        ("Cal", 12 * M, "r/2", "hi"),
        ("Cal", 15 * M, "r/2", "tidewell.org too"),
        ("Dee", 20 * M, "r/3", "quiet room"),
    ])
    return [r._replace(family=fam[r.place]) for r in rows]


class FamilyFlow(unittest.TestCase):
    def setUp(self):
        self.rows = flow_rows()
        self.tokens, _ = T.find_tokens(self.rows)
        self.trees, _ = C.build_trees(self.rows, self.tokens)

    def test_pairs_without_shuffle(self):
        flow = S.family_flow(self.rows, self.trees)
        self.assertEqual(flow["named_links"], 2)
        pairs = {(p["from"], p["to"]): p for p in flow["top_pairs"]}
        self.assertEqual(set(pairs), {("alpha", "beta"), ("beta", "beta")})
        self.assertIsNone(pairs[("alpha", "beta")]["shuffle_mean"])
        self.assertEqual(flow["within_family"], {"observed": 1, "shuffle_mean": None, "ratio": None})
        self.assertEqual(flow["across_families"]["observed"], 1)
        self.assertEqual(flow["within_share"], 0.5)
        self.assertIsNone(flow["observed_matches_nullmodel"])
        self.assertEqual([f["family"] for f in flow["links_out_of_family"]], ["alpha", "beta"])

    def test_pairs_with_shuffle(self):
        null = N.run(self.rows, self.tokens, shuffles=20)
        flow = S.family_flow(self.rows, self.trees, null)
        self.assertTrue(flow["observed_matches_nullmodel"])
        self.assertIsNotNone(flow["within_family"]["shuffle_mean"])
        self.assertIn(S.NO_FAMILY, null.index.families)


class Risky(unittest.TestCase):
    def test_counts_by_type(self):
        records = [{"risky": ["proxy", "answer"], "links_named": 2},
                   {"risky": ["answer"], "links_named": 0},
                   {"risky": [], "links_named": 5}]
        table = S.risky_table(records)
        self.assertEqual(table["trees_with_type"]["proxy"], 1)
        self.assertEqual(table["trees_with_type"]["answer"], 2)
        self.assertEqual(table["trees_with_type_and_named_links"]["answer"], 1)
        self.assertEqual(table["named_links_in_trees_with_type"]["proxy"], 2)
        self.assertEqual((table["trees_with_any_type"], table["trees_with_narrow_type"],
                          table["trees_with_narrow_type_and_named_links"]), (2, 1, 1))
        self.assertNotIn("named_links_vs_shuffle", table)
        self.assertEqual(table["types"], list(T.RISKY_TYPES))

    def test_against_shuffle(self):
        rows = TP.rows_of([
            ("Ana", 0, "r/1", "proxy list at tidewell.org"),
            ("Ben", 5 * M, "r/1", "tidewell.org confirmed"),
            ("Cal", 9 * M, "r/1", "tidewell.org for me too"),
            ("Ana", 20 * M, "r/2", "garden notes at moss-lane.net"),
            ("Ben", 25 * M, "r/2", "moss-lane.net ok"),
            ("Cal", 29 * M, "r/2", "moss-lane.net works"),
        ])
        tokens, _ = T.find_tokens(rows)
        self.assertEqual(tokens["host:tidewell.org"].risky, ("proxy",))
        trees, stats = C.build_trees(rows, tokens)
        null = N.run(rows, tokens, shuffles=10)
        records = [P.tree_record(trees[k], tokens[k], rows) for k in sorted(trees)]
        vs = S.risky_table(records, tokens, null)["named_links_vs_shuffle"]
        self.assertEqual(vs["narrow"]["observed"], 2)
        self.assertEqual(vs["no_type"]["observed"], stats["links_named"] - 2)
        self.assertEqual(vs["broad_only"]["observed"], 0)
        self.assertEqual(vs["by_type"]["proxy"]["observed"], 2)
        self.assertEqual(sorted(vs["by_type"]), sorted(T.RISKY_TYPES))


class BuildScores(unittest.TestCase):
    def test_sections(self):
        result = P.run("generic", os.path.join(ROOT, "examples", "fake_log.ndjson"), shuffles=10)
        records = P.tree_records(result)
        scores = S.build_scores(result, records)
        self.assertEqual(sorted(scores), ["family_flow", "influence", "speed"])
        self.assertEqual(scores["influence"]["named_links"], result.tree_stats["links_named"])
        self.assertEqual(scores["family_flow"]["named_links"], result.tree_stats["links_named"])
        self.assertTrue(scores["family_flow"]["observed_matches_nullmodel"])

    def test_branch_sizes_reexported(self):
        self.assertIs(P.branch_sizes, S.branch_sizes)
        self.assertEqual(S.branch_sizes(TREE_1), {"A": 4, "D": 2})
        self.assertEqual(S.subtree_sizes(TREE_1), {"A": 4, "B": 2, "C": 1, "D": 2, "E": 1, "F": 1})


if __name__ == "__main__":
    unittest.main()
