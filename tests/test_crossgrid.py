"""Tests for patient_zero/crossgrid.py (cross-page links by class and exposure horizon).

unittest, standard library only. Every name, host and line here is made up,
except in RealData, which runs only when PZ_DATA names the collusion.wiki
export folder and reports counts only.

* Horizon: a hand-made log whose cross-page links are 48 h, 145 h and 480 h
  old lands in exactly the right columns; a lag of exactly 24 h counts at 24 h.
* Cascade: the observed cells equal the cross-page and same-page links of
  ``cascade.build_trees`` (an independent implementation) on every fixture.
* Sums and nesting: class cells sum to the pooled total, cross plus same at
  "none" equals the tracer's own link count, and horizons nest, in the
  observed count and in every shuffle.
* E7b logic: the "none" column of the grid equals an E7b-style split computed
  in this file, shuffle for shuffle.
* Controls: the random-row sampler is E7c's; planted carry cascades make no
  same-page links; calibration and power flags follow their rules.
* Readings: the V rule and the breadth table, branch by branch.
* Output: counts only, deterministic; the command line guards --out.
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

from patient_zero import cascade as C  # noqa: E402
from patient_zero import checks as K  # noqa: E402
from patient_zero import crossgrid as G  # noqa: E402
from patient_zero import load as L  # noqa: E402
from patient_zero import nullmodel as N  # noqa: E402
from patient_zero import tokens as T  # noqa: E402
from tests import test_nullmodel as TN  # noqa: E402
from tests import test_planted as TP  # noqa: E402

M = 60
H = 3600
DAY = 86400
NAMES = [h[0] for h in G.HORIZONS]


def chat_rows():
    tmp = tempfile.mkdtemp(prefix="pz-grid-")
    try:
        rows, _ = L.load("generic", TP.write_chat(tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tokens, _ = T.find_tokens(rows)
    return rows, tokens


def horizon_rows():
    """tidewell.org: cross-page links 48 h (Ben), 145 h (Dee) and ~480 h (Eve) old; one same-page link (Cal)."""
    return TP.rows_of([
        ("Ana", 0, "r/1", "notes at tidewell.org"),
        ("Ben", 10 * M, "r/1", "reading the page"),
        ("Cal", 20 * M, "r/1", "tidewell.org confirmed"),
        ("Ben", 2 * DAY, "r/2", "tidewell.org mirrored here"),
        ("Dee", 8 * DAY, "r/2", "checking the second room"),
        ("Dee", 8 * DAY + H, "r/3", "tidewell.org for the third room"),
        ("Eve", 20 * DAY, "r/1", "back on the first room"),
        ("Eve", 20 * DAY + H, "r/4", "tidewell.org in the fourth room"),
    ])


def drop_seconds(obj):
    if isinstance(obj, dict):
        return {k: drop_seconds(v) for k, v in obj.items() if k != "seconds"}
    return obj


def cascade_cells(rows, tokens, horizons=G.HORIZONS):
    """Independent count: cross/same named links per class and horizon from cascade.build_trees."""
    trees, _ = C.build_trees(rows, tokens)
    out = {}
    for key, tree in trees.items():
        cell = out.setdefault(tree.cls, {"cross": [0] * len(horizons), "same": [0] * len(horizons)})
        for a in tree.adoptions:
            if a.status != C.STATUS_LINKED:
                continue
            link = a.link
            side = "cross" if rows[link.source_row].place != rows[link.target_row].place else "same"
            for h, (_n, sec) in enumerate(horizons):
                if sec is None or link.lag <= sec:
                    cell[side][h] += 1
    return out


def e7b_split(index):
    """E7b's split_counts (kirk/cards/km-run/e7b_crosspage.py), pooled over all tokens."""
    c = N.count_links(index, index.agent, detail=True)
    same = cross = 0
    for t, det in enumerate(c.detail):
        firsts = {}
        for i in index.uses[t]:
            a = index.agent[i]
            if a >= 0 and a not in firsts:
                firsts[a] = i
        for (a, st, b), (a2, f) in zip(det, firsts.items()):
            assert a == a2
            if st == "linked":
                if index.place[b] == index.place[f]:
                    same += 1
                else:
                    cross += 1
    return same, cross


class Horizon(unittest.TestCase):
    def test_links_land_in_the_right_columns(self):
        rows = horizon_rows()
        tokens, _ = T.find_tokens(rows)
        got = G.grid(rows, tokens, shuffles=2, seed=1)
        host = got["classes"]["host"]
        self.assertEqual(host["tokens"], 1)
        self.assertEqual([host["cross_page"][n]["observed"] for n in NAMES], [0, 1, 2, 3])
        self.assertEqual([host["same_page"][n]["observed"] for n in NAMES], [1, 1, 1, 1])

    def test_lags_are_source_to_first_use(self):
        rows = horizon_rows()
        tokens, _ = T.find_tokens(rows)
        ix = N.build_index(rows, tokens)
        lags, _count = G.link_lags(ix, ix.agent)
        t = ix.keys.index("host:tidewell.org")
        cross, same, adopters = lags[t]
        self.assertEqual(adopters, 5)
        self.assertEqual(sorted(cross), [2 * DAY, 6 * DAY + H, 20 * DAY + H - 20 * M])
        self.assertEqual(same, [20 * M])

    def test_exact_horizon_counts(self):
        rows = TP.rows_of([
            ("Ana", 0, "r/1", "notes at tidewell.org"),
            ("Ben", 60, "r/1", "reading"),
            ("Cal", 120, "r/1", "tidewell.org confirmed"),
            ("Ben", DAY, "r/2", "tidewell.org mirrored"),  # exactly 24 h after Ana's use
        ])
        tokens, _ = T.find_tokens(rows)
        got = G.grid(rows, tokens, shuffles=1, seed=1)
        self.assertEqual(got["classes"]["host"]["cross_page"]["24h"]["observed"], 1)

    def test_bad_horizons_and_shuffles(self):
        rows, tokens = chat_rows()
        with self.assertRaises(ValueError):
            G.grid(rows, tokens, shuffles=0)
        with self.assertRaises(ValueError):
            G.grid(rows, tokens, shuffles=1, horizons=(("a", 1.0), ("a", 2.0)))


class MatchesCascade(unittest.TestCase):
    def test_observed_cells_equal_cascade_links(self):
        logs = TN.fixture_logs() + [("horizon", horizon_rows())]
        for label, rows in logs:
            tokens, _ = T.find_tokens(rows)
            got = G.grid(rows, tokens, shuffles=1, seed=3)
            want = cascade_cells(rows, tokens)
            with self.subTest(log=label):
                for cls in T.CLASSES:
                    w = want.get(cls, {"cross": [0] * 4, "same": [0] * 4})
                    self.assertEqual([got["classes"][cls]["cross_page"][n]["observed"] for n in NAMES], w["cross"])
                    self.assertEqual([got["classes"][cls]["same_page"][n]["observed"] for n in NAMES], w["same"])


class SumsAndNesting(unittest.TestCase):
    def test_checks_hold_on_every_fixture(self):
        logs = TN.fixture_logs() + [("horizon", horizon_rows()), ("random_big", TN.random_log(9, n=400))]
        for label, rows in logs:
            tokens, _ = T.find_tokens(rows)
            controls = G.default_controls(rows) if len([r for r in rows if not r.exposure_only]) >= 10 else ()
            got = G.grid(rows, tokens, controls, shuffles=5, seed=11)
            with self.subTest(log=label):
                self.assertTrue(all(got["checks"]["class_cells_sum_to_total"].values()))
                self.assertTrue(got["checks"]["horizons_nest"])
                self.assertTrue(got["checks"]["cross_plus_same_equals_tracer_links"])
                ix = N.build_index(rows, tokens)
                self.assertEqual(got["total"]["tracer_named_links"], N.count_links(ix, ix.agent).status["linked"])
                for n in NAMES:
                    self.assertEqual(sum(got["classes"][c]["cross_page"][n]["observed"] for c in T.CLASSES),
                                     got["total"]["cross_page"][n]["observed"])
                    self.assertAlmostEqual(sum(got["classes"][c]["cross_page"][n]["mean"] or 0 for c in T.CLASSES),
                                           got["total"]["cross_page"][n]["mean"] or 0, places=2)

    def test_controls_never_enter_the_real_total(self):
        rows = TN.random_log(9, n=400)
        tokens, _ = T.find_tokens(rows)
        bare = G.grid(rows, tokens, shuffles=4, seed=2)
        with_controls = G.grid(rows, tokens, G.default_controls(rows), shuffles=4, seed=2)
        self.assertEqual(drop_seconds(bare["total"]), drop_seconds(with_controls["total"]))
        self.assertEqual(drop_seconds(bare["classes"]), drop_seconds(with_controls["classes"]))


class E7bLogic(unittest.TestCase):
    def test_none_column_is_the_e7b_split_shuffle_for_shuffle(self):
        for label, rows in (("chat", chat_rows()[0]), ("random", TN.random_log(4, n=300))):
            tokens, _ = T.find_tokens(rows)
            got = G.grid(rows, tokens, shuffles=6, seed=20261003)
            ix = N.build_index(rows, tokens)
            same0, cross0 = e7b_split(ix)
            rng = random.Random(20261003)
            sims = []
            for _ in range(6):
                nr, nt = K.reorder(rows, tokens, rng, K.SCOPE_AGENT_DAY)
                sims.append(e7b_split(N.build_index(nr, nt))[1])
            want = N.summarize(sims, cross0)
            cell = got["total"]["cross_page"]["none"]
            with self.subTest(log=label):
                self.assertEqual(cell["observed"], cross0)
                self.assertEqual(got["total"]["same_page"]["none"]["observed"], same0)
                self.assertEqual((cell["mean"], cell["sd"], cell["z"]), (want["mean"], want["sd"], want["z"]))


class Controls(unittest.TestCase):
    def test_random_rows_use_the_e7c_sampler(self):
        rows = TN.random_log(9, n=400)
        got = G.random_row_tokens(rows, "rc", 7, 10, 20261031)
        agent_rows = [i for i, r in enumerate(rows) if not r.exposure_only]
        rng = random.Random(20261031)
        want = [tuple(sorted(set(rng.sample(agent_rows, 10)))) for _ in range(7)]
        self.assertEqual([got[k].rows for k in sorted(got)], want)
        for tok in got.values():
            self.assertEqual(tok.cls, "control:rc")
            self.assertTrue(all(not rows[i].exposure_only for i in tok.rows))
        self.assertEqual(got, G.random_row_tokens(rows, "rc", 7, 10, 20261031))

    def test_random_rows_need_enough_agent_rows(self):
        rows = horizon_rows()
        with self.assertRaises(ValueError):
            G.random_row_tokens(rows, "rc", 1, 50, 1)

    def test_planted_carry_makes_only_cross_page_links(self):
        rows = TN.random_log(9, n=600)
        planted = G.planted_carry_tokens(rows, "ck", 6, 20261023)
        self.assertTrue(planted)
        for tok in planted.values():
            self.assertGreaterEqual(len(tok.rows), 5)
            places = [rows[i].place for i in tok.rows]
            self.assertEqual(len(places), len(set(places)))  # every use on its own page
        self.assertEqual(planted, G.planted_carry_tokens(rows, "ck", 6, 20261023))
        got = G.grid(rows, {}, [("ck", G.ROLE_POWER, planted)], shuffles=3, seed=1)
        entry = got["controls"]["ck"]
        self.assertEqual([entry["same_page"][n]["observed"] for n in NAMES], [0, 0, 0, 0])
        self.assertGreater(entry["cross_page"]["none"]["observed"], 0)
        self.assertIn("powered", entry["cross_page"]["none"])
        self.assertEqual(got["horizon_calibrated"], {n: None for n in NAMES})

    def test_bad_controls_are_refused(self):
        rows, tokens = chat_rows()
        wrong = {"x": G.ControlToken("x", "phrase", (0, 1))}
        with self.assertRaises(ValueError):
            G.grid(rows, tokens, [("x", G.ROLE_CALIBRATION, wrong)], shuffles=1)
        ok = {"control:x:0": G.ControlToken("control:x:0", "control:x", (0, 1))}
        with self.assertRaises(ValueError):
            G.grid(rows, tokens, [("x", "judge", ok)], shuffles=1)

    def test_calibration_marks_a_failing_horizon(self):
        rows = TN.random_log(9, n=400)
        tokens, _ = T.find_tokens(rows)
        got = G.grid(rows, tokens, G.default_controls(rows), shuffles=8, seed=5)
        for n in NAMES:
            passes = [e["cross_page"][n]["pass"] for e in got["controls"].values() if e["role"] == "calibration"]
            self.assertEqual(got["horizon_calibrated"][n], all(passes))


class Readings(unittest.TestCase):
    def test_v_rule(self):
        base = {"observed": 10, "mean": 5.0, "sd": 1.0}
        self.assertEqual(G.reading(dict(base, z=5.0, shuffles_ge_observed=0, shuffles_le_observed=100)), "above")
        self.assertEqual(G.reading(dict(base, z=5.0, shuffles_ge_observed=2, shuffles_le_observed=98)), "at chance")
        self.assertEqual(G.reading(dict(base, z=-4.0, shuffles_ge_observed=100, shuffles_le_observed=0)), "below")
        self.assertEqual(G.reading(dict(base, z=-4.0, shuffles_ge_observed=100, shuffles_le_observed=2)),
                         "at chance")
        self.assertEqual(G.reading(dict(base, z=-2.9, shuffles_ge_observed=100, shuffles_le_observed=0)),
                         "at chance")
        self.assertEqual(G.reading(dict(base, z=None, shuffles_ge_observed=100, shuffles_le_observed=100)),
                         "at chance (no spread)")

    def test_calibration_pass(self):
        self.assertTrue(G.calibration_pass({"observed": 5, "mean": 4.0, "z": 1.99}))
        self.assertFalse(G.calibration_pass({"observed": 5, "mean": 4.0, "z": -2.0}))
        self.assertTrue(G.calibration_pass({"observed": 3, "mean": 3.0, "z": None}))
        self.assertFalse(G.calibration_pass({"observed": 3, "mean": 2.5, "z": None}))

    def test_breadth_table(self):
        sims_total = [100, 102, 98, 101, 99]  # mean 100
        sd_rest = [10, 12, 8, 11, 9]  # mean 10, sd ~1.58

        def read(obs_total, obs_rest):
            return G.breadth(obs_total, sims_total, obs_rest, sd_rest)["reading"]

        self.assertEqual(read(90, 14), "REST above chance")
        self.assertEqual(read(98, 4), "REST carries more")  # R 0.98, proportional 9.8
        self.assertEqual(read(60, 6), "broad")  # R 0.6, proportional 6, below chance
        self.assertEqual(read(20, 6), "partly broad")  # below chance, far above proportional 2
        self.assertEqual(read(60, 10), "driven by L")  # at chance, far above proportional 6
        self.assertEqual(read(95, 10), "undetermined (low power)")
        self.assertEqual(G.breadth(90, [100], 5, [10])["reading"], "undetermined (too few shuffles)")
        self.assertEqual(G.breadth(90, [100, 100], 5, [10, 10])["reading"], "undetermined (no spread)")


class Output(unittest.TestCase):
    def test_counts_only_and_deterministic(self):
        rows, tokens = chat_rows()
        a = G.grid(rows, tokens, G.default_controls(rows), shuffles=6, seed=1)
        b = G.grid(rows, tokens, G.default_controls(rows), shuffles=6, seed=1)
        self.assertEqual(drop_seconds(a), drop_seconds(b))
        text = json.dumps(a)
        for name in TP.CHAT_NAMES:
            self.assertNotIn('"%s"' % name, text)
        for key, tok in tokens.items():
            self.assertNotIn(tok.value, text)
            self.assertNotIn(key, text)
        self.assertNotIn("control:random_c1:", text)
        self.assertIn("random_c1", a["controls"])

    def test_text_table_names_every_class_and_horizon(self):
        rows, tokens = chat_rows()
        got = G.grid(rows, tokens, shuffles=3, seed=1)
        table = G.text_table(got)
        for word in list(T.CLASSES) + NAMES + ["TOTAL", "breadth", "checks"]:
            self.assertIn(word, table)

    def test_cli_runs_and_guards_out(self):
        tmp = tempfile.mkdtemp(prefix="pz-grid-cli-")
        try:
            path = TP.write_chat(tmp)
            buf = io.StringIO()
            target = os.path.join(tmp, "grid.json")
            self.assertEqual(G.main(["--adapter", "generic", "--input", path, "--shuffles", "3",
                                     "--out", target], out=buf), 0)
            with open(target, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertEqual(saved["null"]["shuffles"], 3)
            self.assertEqual(sorted(saved["controls"]), ["carry_k", "random_c1", "random_c2"])
            self.assertIn("TOTAL", buf.getvalue())
            buf = io.StringIO()
            self.assertEqual(G.main(["--adapter", "generic", "--input", path, "--shuffles", "2",
                                     "--no-controls"], out=buf), 0)
            bad = os.path.join(ROOT, "patient_zero", "grid-out.json")
            buf = io.StringIO()
            self.assertEqual(G.main(["--adapter", "generic", "--input", path, "--shuffles", "3",
                                     "--out", bad], out=buf), 2)
            self.assertIn("ERROR", buf.getvalue())
            self.assertFalse(os.path.exists(bad))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(os.environ.get("PZ_DATA"), "set PZ_DATA to the collusion.wiki export folder")
class RealData(unittest.TestCase):
    def test_observed_cells_reproduce_e7b_and_e7c(self):
        rows, _ = L.load("wiki", os.environ["PZ_DATA"])
        tokens, _ = T.find_tokens(rows)
        got = G.grid(rows, tokens, G.default_controls(rows), shuffles=1, seed=20261003)
        self.assertEqual(got["total"]["tracer_named_links"], 28250)
        self.assertEqual(got["total"]["cross_page"]["none"]["observed"], 5737)
        self.assertEqual(got["total"]["same_page"]["none"]["observed"], 22513)
        self.assertTrue(all(got["checks"]["class_cells_sum_to_total"].values()))
        self.assertTrue(got["checks"]["horizons_nest"])
        self.assertEqual(got["controls"]["random_c1"]["cross_page"]["none"]["observed"], 29)
        self.assertEqual(got["controls"]["carry_k"]["cross_page"]["none"]["observed"], 1041)


if __name__ == "__main__":
    unittest.main()
