"""Tests for patient_zero/calibrate.py and tools/calibrate.py (the calibration matrix).

unittest, standard library only. Every name, page and line here is made up, except in RealData, which runs
only when PZ_DATA names the collusion.wiki export folder and checks counts only.

* PageInstruction: the page-wide shared cause. Permuting which agent used the token first leaves the
  follower set unchanged (the same permutations do change a contagion cascade, so the test has teeth);
  one agent's coin ignores every other agent; only writers on the page inside the window can follow.
* Planting: every kind lands on agent rows; the page instruction takes 40 distinct pages with at least 5
  followers each; adding it leaves every E7 planting unchanged; the tracer links a shared cause exactly
  like spread.
* Matrix: deterministic, counts only, the page-instruction row sits directly under same-page contagion,
  the pre-registered lines are scored PASS or FAIL, and a failed gate withholds the new row.
* RealData (PZ_DATA): the E6 and exp2 plantings reproduce E7's H1 counts exactly.
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
if os.path.join(ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "tools"))

from patient_zero import calibrate as CAL  # noqa: E402
from patient_zero import load as L  # noqa: E402
from patient_zero import nullmodel as N  # noqa: E402
from patient_zero import tokens as T  # noqa: E402

import calibrate as TOOL  # noqa: E402  (tools/calibrate.py)

BASE = 1768039200.0  # 2026-01-10T10:00:00Z
H = 3600
DAY = 86400
HOSTS = ("alder-mill.net", "brine-works.org", "copper-gate.io", "dune-relay.net", "ember-yard.org")


def row(i, agent, t, place, family=None, role=None, unc=1.0, text="plain status update"):
    role = role or (L.ROLE_AGENT if agent else L.ROLE_BLANK)
    return L.Row(row_id="r%05d" % i, agent=agent if role == L.ROLE_AGENT else None, role=role,
                 time=BASE + float(t), uncertainty=float(unc), time_grade=None, place=place,
                 family=family, group=None, text=text)


def sort_rows(rows):
    return sorted(rows, key=lambda r: (r.time, r.row_id))


def one_page(order, place="wiki/Harbour", gap=H, extra=()):
    """Agents in ``order`` each save once on ``place``, ``gap`` apart, from BASE + 1 h. ``extra`` rows added."""
    rows = [row(i, a, (i + 1) * gap, place) for i, a in enumerate(order)]
    rows += [row(len(rows) + j, *spec) for j, spec in enumerate(extra)]
    return sort_rows(rows)


def background(seed=11, agents=120, pages=60, days=6, per_day=100):
    """A made-up wiki-like log: families of pages, busy days, a few blank and human rows."""
    rng = random.Random(seed)
    rows = []
    for d in range(days):
        for a in rng.sample(range(agents), per_day):
            home = rng.sample(range(pages), 3)
            for _ in range(rng.randint(2, 8)):
                p = rng.choice(home) if rng.random() < 0.7 else rng.randrange(pages)
                text = ("notes on %s" % rng.choice(HOSTS)) if rng.random() < 0.15 else "plain status update"
                rows.append(row(len(rows), "ag%02d" % a, d * DAY + rng.uniform(0, DAY), "wiki/P%02d" % p,
                                family="fam%d" % (p % 5), text=text))
        for _ in range(6):
            p = rng.randrange(pages)
            rows.append(row(len(rows), None, d * DAY + rng.uniform(0, DAY), "wiki/P%02d" % p,
                            family="fam%d" % (p % 5), role=rng.choice((L.ROLE_BLANK, L.ROLE_HUMAN))))
    return sort_rows(rows)


def assert_no_new_cells(test, result):
    """No measured cell of the page instruction is left (its planting counts may stay)."""
    test.assertNotIn("page_instruction", result["accuracy"])
    test.assertNotIn("page_instruction", result["time_reversal"])
    for name, res in result["nulls"].items():
        test.assertNotIn("page_instruction", res["kinds"], name)
    test.assertNotIn("page_instruction", [r["kind"] for r in result["matrix"]])
    test.assertNotIn("Page instruction", CAL.markdown_table(result))


AGENTS12 = ["Ada", "Bram", "Cleo", "Dax", "Eve", "Finn", "Gus", "Hana", "Ivo", "Jun", "Kit", "Lux"]


class PageInstruction(unittest.TestCase):
    def plant(self, rows, place="wiki/Harbour", t0=BASE, q=CAL.Q, key=CAL.PAGE_INSTRUCTION_SEED):
        bg = CAL.Background(rows)
        use_rows, followers = CAL.plant_page_instruction(bg, place, t0, key, q)
        return bg, use_rows, followers

    def test_permuting_first_user_keeps_follower_set(self):
        _bg, ur0, base = self.plant(one_page(AGENTS12))
        self.assertGreaterEqual(len(base), 2)  # the made-up page has followers to keep
        self.assertLess(len(base), len(AGENTS12))  # and writers who do not follow
        firsts = set()
        for s in range(25):
            order = list(AGENTS12)
            random.Random(s).shuffle(order)
            rows = one_page(order)
            bg, ur, followers = self.plant(rows)
            firsts.add(rows[ur[0]].agent)
            with self.subTest(seed=s):
                self.assertEqual(followers, base)
                self.assertEqual(sorted(rows[i].agent for i in ur), base)
        self.assertGreater(len(firsts), 1, "the permutations must change who used the token first")

    def test_same_permutations_change_a_contagion_cascade(self):
        sets = set()
        for s in range(25):
            order = list(AGENTS12)
            random.Random(s).shuffle(order)
            rows = one_page(order)
            bg = CAL.Background(rows)
            ur, _par = CAL.plant_contagion(bg, random.Random(5), 0)
            sets.add(tuple(sorted(rows[i].agent for i in ur)))
        self.assertGreater(len(sets), 1, "contagion depends on who came first; the test must be able to fail")

    def test_coin_ignores_other_agents(self):
        _bg, _ur, base = self.plant(one_page(AGENTS12))
        for gone in AGENTS12:
            with self.subTest(removed=gone):
                _bg2, _ur2, f2 = self.plant(one_page([a for a in AGENTS12 if a != gone]))
                self.assertEqual(f2, [a for a in base if a != gone])

    def test_window_place_and_roles(self):
        extra = [("Mo", 80 * H, "wiki/Harbour"),          # after the 72 h window
                 ("Ned", 2 * H, "wiki/Garden"),           # another page
                 (None, 3 * H, "wiki/Harbour"),           # blank row
                 ("Oli", -2 * H, "wiki/Harbour")]         # before the instruction
        rows = one_page(AGENTS12, extra=extra)
        _bg, ur, followers = self.plant(rows, q=1.0)
        self.assertEqual(followers, sorted(AGENTS12))
        self.assertTrue(all(rows[i].place == "wiki/Harbour" and not rows[i].exposure_only for i in ur))
        self.assertEqual(self.plant(rows, q=0.0)[2], [])

    def test_first_save_in_window_is_the_use_row(self):
        rows = sort_rows([row(0, "Ada", 1 * H, "wiki/Harbour"), row(1, "Ada", 5 * H, "wiki/Harbour"),
                          row(2, "Bram", 2 * H, "wiki/Harbour")])
        _bg, ur, followers = self.plant(rows, q=1.0)
        self.assertEqual(followers, ["Ada", "Bram"])
        self.assertEqual([rows[i].time - BASE for i in ur], [1 * H, 2 * H])

    def test_deterministic_and_keyed(self):
        rows = one_page(AGENTS12)
        self.assertEqual(self.plant(rows)[1:], self.plant(rows)[1:])
        self.assertTrue(CAL.follows(1, "p", "a", q=1.0))
        self.assertFalse(CAL.follows(1, "p", "a", q=0.0))
        draws = [CAL.follows(CAL.PAGE_INSTRUCTION_SEED, "p%03d" % i, "a") for i in range(2000)]
        self.assertAlmostEqual(sum(draws) / len(draws), CAL.Q, delta=0.04)

    def test_tracer_links_a_shared_cause_like_spread(self):
        rows = one_page(AGENTS12)
        _bg, ur, followers = self.plant(rows, q=1.0)
        tok = {"planted:page_instruction:000": CAL.PTok("planted:page_instruction:000", "planted_page_instruction",
                                                        tuple(ur))}
        ix = N.build_index(rows, tok)
        count = N.count_links(ix, ix.agent)
        self.assertEqual(count.status["first"], 1)
        self.assertEqual(count.status["linked"], len(followers) - 1)  # every later follower gets a "parent"


class Planting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = background(agents=160, pages=90, per_day=150)  # dense enough for 40 pages
        cls.P = CAL.plant_all(cls.rows)

    def test_every_kind_on_agent_rows(self):
        for key, tok in self.P.tokens.items():
            with self.subTest(key=key):
                self.assertTrue(tok.rows)
                self.assertTrue(all(not self.rows[i].exposure_only for i in tok.rows))
                self.assertEqual(list(tok.rows), sorted(tok.rows))

    def test_page_instruction_pages_and_floor(self):
        keys = [k for k, kd in self.P.kind_of.items() if kd == "page_instruction"]
        self.assertEqual(len(keys), 40)
        pages = [{self.rows[i].place for i in self.P.tokens[k].rows} for k in keys]
        self.assertTrue(all(len(p) == 1 for p in pages), "a page instruction stays on its page")
        self.assertEqual(len({next(iter(p)) for p in pages}), 40, "40 distinct pages")
        for k in keys:
            agents = [self.rows[i].agent for i in self.P.tokens[k].rows]
            self.assertEqual(len(agents), len(set(agents)), "one use row per follower")
            self.assertGreaterEqual(len(agents), CAL.PAGE_INSTRUCTION_MIN_FOLLOWERS)
            span = self.rows[self.P.tokens[k].rows[-1]].time - self.rows[self.P.tokens[k].rows[0]].time
            self.assertLessEqual(span, CAL.HORIZON)
        self.assertNotIn("page_instruction", {kd for k, kd in self.P.kind_of.items() if k in self.P.truth})

    def test_new_kind_leaves_e7_plantings_unchanged(self):
        without = CAL.plant_all(self.rows, page_instruction=False)
        mine = {k: t for k, t in self.P.tokens.items() if self.P.kind_of[k] != "page_instruction"}
        self.assertEqual(mine, without.tokens)
        self.assertEqual(without.truth, self.P.truth)

    def test_planting_is_deterministic(self):
        again = CAL.plant_all(self.rows)
        self.assertEqual(again.tokens, self.P.tokens)
        self.assertEqual(again.info, self.P.info)

    def test_planting_info_counts(self):
        info = self.P.info["page_instruction"]
        self.assertEqual(info["tokens"], 40)
        self.assertGreaterEqual(info["pages_tried"], 40)
        self.assertEqual(info["followers"], sum(len(self.P.tokens[k].rows) for k, kd in self.P.kind_of.items()
                                                if kd == "page_instruction"))


class Matrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = background()
        cls.real, _ = T.find_tokens(cls.rows)
        cls.res = CAL.run(cls.rows, cls.real, shuffles=6)

    def test_deterministic(self):
        again = CAL.run(self.rows, self.real, shuffles=6)
        drop = lambda d: {k: v for k, v in d.items() if k != "seconds"}  # noqa: E731
        self.assertEqual(json.dumps(drop(again), sort_keys=True), json.dumps(drop(self.res), sort_keys=True))

    def test_row_order_puts_page_instruction_under_contagion(self):
        kinds = [r["kind"] for r in self.res["matrix"]]
        self.assertEqual(kinds[:2], ["contagion", "page_instruction"])
        self.assertTrue(all(set(r["nulls"]) == set(CAL.NULL_NAMES) for r in self.res["matrix"]))
        table = CAL.markdown_table(self.res).splitlines()
        self.assertTrue(table[2].startswith("| Same-page contagion (E6) |"))
        self.assertTrue(table[3].startswith("| Page instruction"))

    def test_counts_only(self):
        text = json.dumps(self.res)
        agents = {r.agent for r in self.rows if r.agent}
        self.assertEqual([a for a in sorted(agents) if '"%s"' % a in text], [])
        for host in HOSTS:
            self.assertNotIn(host, text)

    def test_lines_scored_pass_or_fail(self):
        lines = self.res["page_instruction_lines"]
        self.assertIn(lines["L1_label_nulls_flag_it"]["result"], ("PASS", "FAIL"))
        self.assertIn(lines["L2_order_null_and_reversed_clock_read_nothing"]["result"], ("PASS", "FAIL"))

    def test_carry_beats_the_reversed_clock_and_same_page_does_not(self):
        tr = self.res["time_reversal"]
        self.assertGreater(tr["carry"]["forward"], tr["carry"]["reversed"])
        self.assertGreater(tr["carry"]["z_paired"], 2)

    def test_withhold_removes_every_new_cell(self):
        w = CAL.withhold(self.res)
        assert_no_new_cells(self, w)
        self.assertTrue(w["page_instruction_lines"].startswith("WITHHELD"))
        self.assertIn("contagion", w["accuracy"])

    def test_compare_expected(self):
        exp = json.loads(json.dumps({k: self.res[k] for k in ("nulls", "accuracy")}))
        self.assertEqual(CAL.compare_expected(json.loads(json.dumps(self.res)), exp), [])
        exp["nulls"]["N1_label_day"]["kinds"]["carry"]["z"] = 99.0
        self.assertEqual(CAL.compare_expected(json.loads(json.dumps(self.res)), exp),
                         ["nulls/N1_label_day/kinds/carry/z"])


class Paired(unittest.TestCase):
    def test_paired_z(self):
        self.assertEqual(CAL.paired_z([0, 0, 0]), 0.0)
        self.assertIsNone(CAL.paired_z([1, 1, 1]))
        self.assertIsNone(CAL.paired_z([]))
        self.assertEqual(CAL.paired_z([1, -1, 1, -1]), 0.0)
        self.assertGreater(CAL.paired_z([3, 2, 4, 3]), 2)

    def test_no_reading(self):
        self.assertTrue(CAL._no_reading({"z": 1.9}))
        self.assertFalse(CAL._no_reading({"z": -2.0}))
        self.assertTrue(CAL._no_reading({"z": None, "sd": 0, "mean": 5, "observed": 5}))
        self.assertFalse(CAL._no_reading({"z": None, "sd": 0, "mean": 5, "observed": 6}))


class Tool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pz-cal-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_log(self, rows):
        path = os.path.join(self.tmp, "log.ndjson")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps({"id": r.row_id, "agent": r.agent or "", "time": r.time, "place": r.place,
                                     "text": r.text, "family": r.family, "uncertainty_seconds": r.uncertainty,
                                     "human": r.role == L.ROLE_HUMAN}) + "\n")
        return path

    def test_gate_fail_withholds_and_exits_1(self):
        log = self.write_log(background(seed=3, per_day=50))
        first = os.path.join(self.tmp, "first.json")
        self.assertEqual(TOOL.main(["--adapter", "generic", "--input", log, "--out", first, "--shuffles", "3"],
                                   out=io.StringIO()), 0)
        with open(first, encoding="utf-8") as fh:
            exp = json.load(fh)
        exp["H0"] = {"expected": dict(exp["H0"])}
        exp["H1"] = {"expected": dict(exp["H1"])}
        good = os.path.join(self.tmp, "good.json")
        with open(good, "w", encoding="utf-8") as fh:
            json.dump(exp, fh)
        second = os.path.join(self.tmp, "second.json")
        self.assertEqual(TOOL.main(["--adapter", "generic", "--input", log, "--out", second, "--shuffles", "3",
                                    "--expect", good], out=io.StringIO()), 0)
        with open(second, encoding="utf-8") as fh:
            ok = json.load(fh)
        self.assertTrue(ok["gates"]["pass"])
        self.assertIn("page_instruction", ok["accuracy"])
        exp["H1"]["expected"]["carry_links"] += 1
        bad = os.path.join(self.tmp, "bad.json")
        with open(bad, "w", encoding="utf-8") as fh:
            json.dump(exp, fh)
        third = os.path.join(self.tmp, "third.json")
        buf = io.StringIO()
        self.assertEqual(TOOL.main(["--adapter", "generic", "--input", log, "--out", third, "--shuffles", "3",
                                    "--expect", bad], out=buf), 1)
        with open(third, encoding="utf-8") as fh:
            withheld = json.load(fh)
        assert_no_new_cells(self, withheld)
        self.assertTrue(withheld["page_instruction_lines"].startswith("WITHHELD"))
        self.assertFalse(withheld["gates"]["pass"])

    def test_refuses_out_inside_repo(self):
        buf = io.StringIO()
        code = TOOL.main(["--adapter", "generic", "--input", os.path.join(ROOT, "examples", "fake_log.ndjson"),
                          "--out", os.path.join(ROOT, "calibration.json")], out=buf)
        self.assertEqual(code, 2)
        self.assertIn("outside the repo", buf.getvalue())


@unittest.skipUnless(os.environ.get("PZ_DATA"), "set PZ_DATA to the collusion.wiki export folder")
class RealData(unittest.TestCase):
    def test_e7_plantings_reproduce(self):
        rows, _ = L.load("wiki", os.environ["PZ_DATA"])
        P = CAL.plant_all(rows)
        planted = {k: t for k, t in P.tokens.items() if P.kind_of[k] != "page_instruction"}
        kind_of = {k: v for k, v in P.kind_of.items() if k in planted}
        ixp = N.build_index(rows, planted)
        cp = N.count_links(ixp, ixp.agent, detail=True)
        h1 = CAL.h1_counts(CAL.accuracy(ixp, cp, kind_of, P.truth))
        self.assertEqual(h1, {"contagion_true_linked": 931, "contagion_true_adopters": 931,
                              "contagion_parent_exact": 68, "decoy_fam_links": 140, "carry_links": 1041,
                              "same_links": 1073, "decoy_bcast_links": 28, "decoy_fam2_links": 139})
        self.assertEqual(P.info["page_instruction"]["tokens"], 40)


if __name__ == "__main__":
    unittest.main()
