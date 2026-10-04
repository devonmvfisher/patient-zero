"""Tests for patient_zero/inject.py (detection limit by replicated injection).

unittest, standard library only. Every name and line here is made up, except
in RealData, which runs only when PZ_DATA names the collusion.wiki export
folder and reports counts only.

* Plant: a carry cascade puts every use on its own page, one use per agent;
  the same seed gives the same pool, another seed another pool.
* Split: same-page plus cross-page links equal the named links; a hand-made
  log has one of each; planted carry links are all cross-page.
* Null: the real aggregate under injection equals checks.order_null with the
  same seed, and planted tokens never change the real tokens' shuffles.
* Rates: the detection rule, the minimum detectable carry, the extension
  grid and every non-monotone step, on made-up tables.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from patient_zero import checks as K  # noqa: E402
from patient_zero import inject as I  # noqa: E402
from patient_zero import load as L  # noqa: E402
from patient_zero import nullmodel as N  # noqa: E402
from tests import test_planted as TP  # noqa: E402

DAY = 86400
M = 60
HOSTS = ("alder-mill.net", "brine-works.org", "copper-gate.io", "dune-relay.net")


def busy_specs(seed=5, n=1500, agents=30, places=8, days=2):
    """(time, agent, place, text) for a made-up busy log, time-sorted."""
    rng = random.Random(seed)
    specs = []
    for _ in range(n):
        t = rng.uniform(0, days * DAY)
        text = ("notes on %s" % rng.choice(HOSTS)) if rng.random() < 0.2 else "plain status update"
        specs.append((t, "ag%02d" % rng.randrange(agents), "room/%02d" % rng.randrange(places), text))
    specs.sort()
    return specs


def busy_rows(seed=5, **kw):
    return TP.rows_of([(a, t, p, text, 1) for t, a, p, text in busy_specs(seed, **kw)])


def real_tokens(rows):
    from patient_zero.tokens import find_tokens
    tokens, _ = find_tokens(rows)
    return tokens


def cell(k, detected, links=10, z=None):
    return {"k": k, "detected": detected, "planted_links": links, "z": z if z is not None else (4.0 if detected else 1.0)}


def table_of(pattern, ks):
    """pattern: {name: string of 1/0 per k} -> a cells() style table (both statistics the same)."""
    return {name: {stat: [cell(k, ch == "1", links=10 * k) for k, ch in zip(ks, bits)] for stat in I.STATS}
            for name, bits in pattern.items()}


class Plant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = busy_rows()
        cls.lay = I.layout(cls.rows)

    def test_cascades_are_kept_with_enough_adopters(self):
        pool = I.draw_pool(self.rows, self.lay, 7, 10)
        self.assertEqual(len(pool), 10)
        for use_rows in pool:
            self.assertGreaterEqual(len(use_rows), I.MIN_ADOPTERS + 1)
            self.assertLessEqual(len(use_rows), I.CARRY_CAP)

    def test_every_use_on_its_own_page_one_use_per_agent(self):
        for use_rows in I.draw_pool(self.rows, self.lay, 7, 10):
            self.assertEqual(len({self.rows[i].place for i in use_rows}), len(use_rows))
            self.assertEqual(len({self.rows[i].agent for i in use_rows}), len(use_rows))
            self.assertTrue(all(not self.rows[i].exposure_only for i in use_rows))

    def test_adopters_come_in_clock_order_after_their_parent(self):
        rng = random.Random(3)
        for _ in range(20):
            start = rng.choice(self.lay.agent_rows)
            use_rows, parent = I.plant_carry(self.rows, self.lay, rng, start)
            when = {self.rows[i].agent: self.rows[i].time for i in use_rows}
            for child, par in parent.items():
                self.assertGreater(when[child], when[par])

    def test_same_seed_same_pool_other_seed_other_pool(self):
        a = I.draw_pool(self.rows, self.lay, 11, 8)
        b = I.draw_pool(self.rows, self.lay, 11, 8)
        c = I.draw_pool(self.rows, self.lay, 12, 8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_segments_continue_in_order(self):
        p, toks = I.plant_set(self.rows, "x", ((7, 3), (8, 2)), self.lay)
        self.assertEqual(len(p.keys), 5)
        self.assertEqual([toks[k].rows for k in p.keys[:3]], I.draw_pool(self.rows, self.lay, 7, 3))
        self.assertEqual([toks[k].rows for k in p.keys[3:]], I.draw_pool(self.rows, self.lay, 8, 2))
        self.assertTrue(all(t.cls == "planted_carry" for t in toks.values()))

    def test_first_k_of_a_longer_pool_is_the_short_pool(self):
        self.assertEqual(I.draw_pool(self.rows, self.lay, 9, 12)[:5], I.draw_pool(self.rows, self.lay, 9, 5))

    def test_empty_log_gives_no_cascades(self):
        self.assertEqual(I.draw_pool([], I.layout([]), 7, 5), [])


class Split(unittest.TestCase):
    def test_same_plus_cross_equals_named_links(self):
        rows = busy_rows(seed=8)
        tokens = real_tokens(rows)
        self.assertTrue(tokens)
        ix = N.build_index(rows, tokens)
        same, cross = I.split_counts(ix)
        named = N.count_links(ix, ix.agent).named
        self.assertEqual([s + x for s, x in zip(same, cross)], list(named))

    def test_hand_made_log_has_one_same_and_one_cross_link(self):
        rows = TP.rows_of([
            ("Ana", 0, "r/1", "tok"),
            ("Ben", 5 * M, "r/1", "reading"),
            ("Ben", 10 * M, "r/2", "tok"),
            ("Cal", 20 * M, "r/1", "tok"),
        ])
        toks = {"t": I.PlantedToken("t", "planted_carry", (0, 2, 3))}
        ix = N.build_index(rows, toks)
        same, cross = I.split_counts(ix)
        self.assertEqual((same[0], cross[0]), (1, 1))

    def test_planted_carry_links_are_all_cross_page(self):
        rows = busy_rows()
        p, toks = I.plant_set(rows, "x", ((7, 10),))
        ix = N.build_index(rows, toks)
        same, cross = I.split_counts(ix)
        self.assertEqual(sum(same), 0)
        self.assertGreater(sum(cross), 0)


class Null(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = busy_rows(seed=8)
        cls.tokens = real_tokens(cls.rows)
        _p, cls.planted = I.plant_set(cls.rows, "x", ((7, 6),))

    def test_real_aggregate_equals_checks_order_null(self):
        null = I.injection_null(self.rows, self.tokens, self.planted, shuffles=6, seed=41)
        ref = K.order_null(self.rows, self.tokens, shuffles=6, seed=41)["links_named"]
        sm = N.summarize(null.real_sims["aggregate"], null.real_observed["aggregate"])
        self.assertEqual(null.real_observed["aggregate"], ref["observed"])
        self.assertEqual((sm["mean"], sm["sd"], sm["shuffles_ge_observed"]),
                         (ref["mean"], ref["sd"], ref["shuffles_ge_observed"]))

    def test_planted_tokens_never_change_real_shuffles(self):
        with_p = I.injection_null(self.rows, self.tokens, self.planted, shuffles=4, seed=41)
        without = I.injection_null(self.rows, self.tokens, {}, shuffles=4, seed=41)
        self.assertEqual(with_p.real_sims, without.real_sims)
        self.assertEqual(with_p.real_observed, without.real_observed)

    def test_clashing_keys_and_zero_shuffles_refused(self):
        key = next(iter(self.tokens))
        clash = {key: I.PlantedToken(key, "planted_carry", (0, 1))}
        with self.assertRaises(ValueError):
            I.injection_null(self.rows, self.tokens, clash, shuffles=2)
        with self.assertRaises(ValueError):
            I.injection_null(self.rows, self.tokens, self.planted, shuffles=0)

    def test_cells_add_planted_counts_to_the_real_ones(self):
        null = I.injection_null(self.rows, self.tokens, self.planted, shuffles=4, seed=41)
        plant = I.Plant("x", ((7, 6),), tuple(sorted(self.planted)))
        got = I.cells(null, plant, (0, 3, 6))
        for stat, p in (("aggregate", 0), ("cross_page", 1)):
            self.assertEqual(got[stat][0]["observed"], null.real_observed[stat])
            three = sum(null.token_observed[k][p] for k in plant.keys[:3])
            self.assertEqual(got[stat][1]["observed"], null.real_observed[stat] + three)
            self.assertEqual(got[stat][1]["cascades"], 3)
        self.assertTrue(got["aggregate"][2]["planted_links"] >= got["aggregate"][1]["planted_links"])

    def test_short_pool_is_flagged(self):
        null = I.injection_null(self.rows, self.tokens, self.planted, shuffles=2, seed=41)
        plant = I.Plant("x", ((7, 6),), tuple(sorted(self.planted)))
        got = I.cells(null, plant, (6, 9))
        self.assertNotIn("short", got["aggregate"][0])
        self.assertTrue(got["aggregate"][1]["short"])


class Rates(unittest.TestCase):
    KS = (40, 60, 80, 100)

    def test_detection_rule(self):
        self.assertTrue(I.is_detected({"z": 3.0, "shuffles_ge_observed": 1}))
        self.assertFalse(I.is_detected({"z": 2.99, "shuffles_ge_observed": 0}))
        self.assertFalse(I.is_detected({"z": 9.0, "shuffles_ge_observed": 2}))
        self.assertFalse(I.is_detected({"z": None, "shuffles_ge_observed": 0}))

    def test_need_is_four_of_five(self):
        self.assertEqual(I.default_need(5), 4)
        self.assertEqual(I.default_need(10), 8)
        self.assertEqual(I.default_need(1), 1)

    def test_minimum_detectable_carry_and_share(self):
        t = table_of({"a": "0011", "b": "0111", "c": "0011", "d": "0001", "e": "0111"}, self.KS)
        r = I.rate_summary(t, list(t), self.KS, reference_links=1000)
        agg = r["aggregate"]
        self.assertEqual([x["detected"] for x in agg["rates"]], [0, 2, 4, 5])
        self.assertEqual(agg["minimum_detectable_carry"]["k"], 80)
        self.assertEqual(agg["minimum_detectable_carry"]["planted_links_median"], 800)
        self.assertEqual(agg["minimum_detectable_carry"]["pct_of_reference_median"], 80.0)
        self.assertTrue(agg["monotone"])
        self.assertEqual(agg["first_detected_k_per_draw"]["b"], 60)

    def test_not_reached_uses_the_extension_and_says_so(self):
        ks, ext = self.KS, (120, 160)
        t = table_of({n: "000011" for n in "abcde"}, ks + ext)
        r = I.rate_summary(t, list(t), ks, ext, reference_links=1000)
        mdc = r["cross_page"]["minimum_detectable_carry"]
        self.assertIsNone(mdc["k"])
        self.assertFalse(mdc["reached_in_grid"])
        self.assertEqual(mdc["beyond_k"], 100)
        self.assertEqual(mdc["extension_k"], 120)

    def test_non_monotone_rate_is_flagged_not_fixed(self):
        t = table_of({"a": "1101", "b": "1101", "c": "1111", "d": "1001", "e": "0001"}, self.KS)
        r = I.rate_summary(t, list(t), self.KS)
        agg = r["aggregate"]
        self.assertEqual([x["detected"] for x in agg["rates"]], [4, 3, 1, 5])
        self.assertFalse(agg["monotone"])
        self.assertEqual([(s["from_k"], s["to_k"]) for s in agg["non_monotone_steps"]], [(40, 60), (60, 80)])
        self.assertIn({"draw": "a", "detected_at_k": 40, "missed_at_k": 80}, agg["draw_drops"])
        self.assertEqual(agg["minimum_detectable_carry"]["k"], 40)  # as measured, never smoothed

    def test_anchors_never_count(self):
        t = table_of({"a": "1111", "b": "0000", "anchor": "1111"}, self.KS)
        r = I.rate_summary(t, ["a", "b"], self.KS)
        self.assertEqual(r["replicates"], 2)
        self.assertEqual([x["detected"] for x in r["aggregate"]["rates"]], [1, 1, 1, 1])


class Output(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-inject-")
        cls.log = os.path.join(cls.tmp, "busy.ndjson")
        with open(cls.log, "w", encoding="utf-8", newline="\n") as fh:
            for j, (t, a, p, text) in enumerate(busy_specs(seed=8, n=900)):
                fh.write(json.dumps({"id": "e%04d" % j, "agent": a, "time": 1767225600 + t, "place": p,
                                     "text": text}) + "\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cli(self, *extra):
        buf = io.StringIO()
        code = I.main(["--adapter", "generic", "--input", self.log, "--plant-seeds", "11,12,13", "--ks", "2,4",
                       "--extension-ks", "8", "--shuffles", "5", "--anchor", "orig=7:4,8:4"] + list(extra), out=buf)
        return code, buf.getvalue()

    def test_counts_only_and_every_draw_listed(self):
        code, text = self.cli()
        self.assertEqual(code, 0)
        got = json.loads(text)
        for name in ("ag0", "room/", "plain status", "alder-mill", "planted:"):
            self.assertNotIn(name, text)
        names = [p["name"] for p in got["plant_sets"]]
        self.assertEqual(names, ["orig", "seed_11", "seed_12", "seed_13"])
        self.assertEqual([p["role"] for p in got["plant_sets"]], ["anchor"] + ["replicate"] * 3)
        for p in got["plant_sets"]:
            self.assertEqual([c["k"] for c in p["cells"]["aggregate"]], [0, 2, 4, 8])
        self.assertEqual(got["rates"]["replicates"], 3)
        self.assertEqual(got["rates"]["need"], 3)
        self.assertIn("total", got["seconds"])

    def test_deterministic(self):
        def strip(obj):
            if isinstance(obj, dict):
                return {k: strip(v) for k, v in obj.items() if k != "seconds"}
            if isinstance(obj, list):
                return [strip(v) for v in obj]
            return obj
        a = strip(json.loads(self.cli()[1]))
        b = strip(json.loads(self.cli()[1]))
        self.assertEqual(a, b)

    def test_out_inside_repo_refused_and_bad_input_reported(self):
        code, text = self.cli("--out", os.path.join(ROOT, "inject.json"))
        self.assertEqual(code, 2)
        self.assertIn("ERROR", text)
        buf = io.StringIO()
        code = I.main(["--adapter", "generic", "--input", os.path.join(self.tmp, "missing.ndjson"),
                       "--plant-seeds", "1"], out=buf)
        self.assertEqual(code, 2)
        self.assertIn("ERROR", buf.getvalue())

    def test_out_file_written(self):
        path = os.path.join(self.tmp, "o", "inject.json")
        code, text = self.cli("--out", path)
        self.assertEqual(code, 0)
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh), json.loads(text))

    def test_repeated_seeds_and_bad_ks_refused(self):
        rows = busy_rows()
        with self.assertRaises(ValueError):
            I.run(rows, {}, [3, 3], ks=(2,), shuffles=2)
        with self.assertRaises(ValueError):
            I.run(rows, {}, [3], ks=(), shuffles=2)


@unittest.skipUnless(os.environ.get("PZ_DATA"), "set PZ_DATA to the collusion.wiki export folder")
class RealData(unittest.TestCase):
    """E7's carry draw reproduces: planted links at every E7 amount (counts only; no shuffles)."""

    def test_e7_draw_planted_links(self):
        rows, _ = L.load("wiki", os.environ["PZ_DATA"])
        plant, toks = I.plant_set(rows, "e7_original", ((20261023, 40), (20261034, 160)))
        self.assertEqual(len(plant.keys), 200)
        ix = N.build_index(rows, toks)
        same, cross = I.split_counts(ix)
        pos = {k: t for t, k in enumerate(ix.keys)}
        self.assertEqual(sum(same), 0)
        want = {10: 274, 20: 490, 40: 1041, 80: 2170, 120: 3272, 160: 4306, 200: 5561}
        got = {k: sum(cross[pos[key]] for key in plant.keys[:k]) for k in want}
        self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
