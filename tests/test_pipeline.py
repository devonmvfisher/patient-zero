"""Tests for the end-to-end run (patient_zero/pipeline.py and cli.py).

Standard library unittest only. The synthetic example log runs every time;
the real-data run (gate G1) runs only when PZ_DATA names the collusion.wiki
export folder. Failure messages carry counts only, never names or text.
"""

import contextlib
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

from patient_zero import cascade as C  # noqa: E402
from patient_zero import load as L  # noqa: E402
from patient_zero import pipeline as P  # noqa: E402

EXAMPLE = os.path.join(ROOT, "examples", "fake_log.ndjson")
RECORD_FIELDS = (
    "token_id", "token", "cls", "kind", "agents", "status", "links_named", "links_unnamed",
    "shifted_by_clock", "depth", "largest_branch", "largest_branch_root", "max_children", "uses",
    "places", "families", "groups", "first_user", "first_use", "first_family", "first_group",
    "effective_seed", "seed_use", "seed_family", "seed_children", "seed_is_first_user",
    "span_hours", "named_lag_median_s", "risky", "seed_branch", "seed_min_to_3rd",
    "seed_min_to_10th", "branch_min_to_3rd", "branch_min_to_10th", "shuffle_links_mean",
    "shuffle_ge_observed",
)
FACT_KEYS = ("tool", "facts_version", "input", "params", "load", "tokens", "cascade", "trees",
             "families", "groups", "risky", "outputs", "nullmodel", "scores", "slots", "core_keys",
             "core_sha256", "code", "runtime")


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def run_cli(args):
    buf = io.StringIO()
    code = P.main(args, out=buf)
    return code, buf.getvalue()


class ExampleRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-pipeline-")
        cls.out_a = os.path.join(cls.tmp, "a")
        cls.out_b = os.path.join(cls.tmp, "b")
        cls.code_a, cls.text_a = run_cli(["--adapter", "generic", "--input", EXAMPLE, "--out", cls.out_a])
        cls.code_b, _ = run_cli(["--adapter", "generic", "--input", EXAMPLE, "--out", cls.out_b])
        with open(os.path.join(cls.out_a, "facts.json"), encoding="utf-8") as fh:
            cls.facts = json.load(fh)
        with open(os.path.join(cls.out_a, "trees.jsonl"), encoding="utf-8") as fh:
            cls.trees_text = fh.read()
        cls.records = [json.loads(line) for line in cls.trees_text.splitlines()]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_exit_and_files(self):
        self.assertEqual(self.code_a, 0)
        self.assertEqual(self.code_b, 0)
        self.assertIn("core_sha256", self.text_a)
        for name in ("facts.json", "trees.jsonl"):
            self.assertTrue(os.path.isfile(os.path.join(self.out_a, name)))
        self.assertFalse([n for n in os.listdir(self.out_a) if n.endswith(".tmp")])

    def test_facts_shape(self):
        for key in FACT_KEYS:
            self.assertIn(key, self.facts)
        self.assertEqual(self.facts["core_keys"], list(P.CORE_KEYS))
        self.assertEqual(self.facts["input"]["files"][0]["name"], "fake_log.ndjson")
        self.assertNotIn(json.dumps(ROOT)[1:-1], json.dumps(self.facts))  # base names only
        for slot in ("F3", "F4", "F5", "F6", "F7", "F8", "F9", "F12", "F14", "pending"):
            self.assertIn(slot, self.facts["slots"])
        for slot in ("F5", "F7"):
            self.assertNotIn(slot, self.facts["slots"]["pending"])
        self.assertIn("nullmodel", P.CORE_KEYS)
        self.assertIn("scores", P.CORE_KEYS)

    def test_numbers_agree(self):
        f = self.facts
        self.assertEqual(f["trees"]["total"], 3)
        self.assertEqual(len(self.records), f["trees"]["total"])
        self.assertEqual(f["outputs"]["trees.jsonl"]["lines"], len(self.records))
        self.assertEqual(f["slots"]["F3"]["tokens_3plus_agents"], f["tokens"]["kept_total"])
        self.assertEqual(f["slots"]["F4"]["links_named"], sum(r["links_named"] for r in self.records))
        self.assertEqual(f["slots"]["F4"]["links_named"], f["cascade"]["links_named"])
        self.assertEqual(f["slots"]["F8"]["parallel_starts"],
                         sum(r["status"]["parallel"] for r in self.records))
        self.assertEqual(sum(fam["named_links_from_here"] for fam in f["families"]),
                         f["cascade"]["links_named"])
        self.assertEqual(sum(g["rows"] for g in f["groups"]), f["load"]["rows"])
        self.assertTrue(f["nullmodel"]["observed_matches_cascade"])
        self.assertEqual(f["nullmodel"]["links_named"]["observed"], f["cascade"]["links_named"])
        self.assertEqual(f["slots"]["F5"]["links_observed"], f["cascade"]["links_named"])
        self.assertEqual(f["slots"]["F5"]["shuffles"], 100)
        self.assertEqual(f["params"]["shuffles"], 100)
        self.assertEqual(f["scores"]["influence"]["named_links"], f["cascade"]["links_named"])
        self.assertTrue(f["scores"]["family_flow"]["observed_matches_nullmodel"])
        speed = f["scores"]["speed"]["seed_to_3rd"]
        self.assertEqual(speed["trees"], sum(1 for r in self.records if r["seed_min_to_3rd"] is not None))

    def test_records(self):
        for rec in self.records:
            self.assertEqual(tuple(rec), RECORD_FIELDS)
            self.assertRegex(rec["token_id"], r"^[0-9a-f]{12}$")
            self.assertEqual(sum(rec["status"].values()), rec["agents"])
            self.assertLessEqual(rec["largest_branch"], rec["agents"])
            self.assertIsNotNone(rec["first_user"])
        self.assertEqual([r["agents"] for r in self.records],
                         sorted((r["agents"] for r in self.records), reverse=True))
        lead = self.records[0]
        self.assertEqual(lead["token"], "convention:control:BLUEHERON")
        self.assertEqual(lead["largest_branch"], 4)
        self.assertEqual(lead["depth"], 3)
        self.assertEqual(lead["seed_branch"], 4)
        self.assertIsNotNone(lead["seed_min_to_3rd"])
        self.assertIsNone(lead["seed_min_to_10th"])
        for rec in self.records:
            self.assertGreaterEqual(rec["shuffle_links_mean"], 0)
            self.assertTrue(0 <= rec["shuffle_ge_observed"] <= 100)

    def test_strings_short_and_ascii(self):
        for s in _strings(self.records):
            self.assertLessEqual(len(s), P.SHORT_MAX)
            self.assertTrue(all(32 <= ord(ch) < 127 for ch in s))

    def test_no_human_label_and_no_row_text(self):
        both = self.trees_text + json.dumps(self.facts)
        self.assertNotIn("Host", both)  # the example's human speaker
        self.assertNotIn("Morning plan", both)  # row text never leaves the run
        self.assertNotIn("reading list link is broken", both)

    def test_deterministic(self):
        with open(os.path.join(self.out_b, "facts.json"), encoding="utf-8") as fh:
            facts_b = json.load(fh)
        with open(os.path.join(self.out_b, "trees.jsonl"), encoding="utf-8") as fh:
            trees_b = fh.read()
        self.assertEqual(trees_b, self.trees_text)
        self.assertEqual(facts_b["core_sha256"], self.facts["core_sha256"])


class ShuffleOptions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pz-pipeline-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def facts(self, *extra):
        out = os.path.join(self.tmp, "o%d" % len(os.listdir(self.tmp)))
        code, text = run_cli(["--adapter", "generic", "--input", EXAMPLE, "--out", out] + list(extra))
        self.assertEqual(code, 0, text)
        with open(os.path.join(out, "facts.json"), encoding="utf-8") as fh:
            return json.load(fh), text

    def test_zero_shuffles(self):
        facts, text = self.facts("--shuffles", "0")
        self.assertEqual(facts["params"]["shuffles"], 0)
        self.assertEqual(facts["slots"]["F5"]["shuffles"], 0)
        self.assertIsNone(facts["slots"]["F5"]["links_expected_mean"])
        self.assertNotIn("shuffle (", text)

    def test_seed_and_scope_change_only_the_shuffle(self):
        base, text = self.facts()
        self.assertIn("shuffle (day, 100 x, seed %d)" % P.N.DEFAULT_SEED, text)
        other, _ = self.facts("--seed", "7", "--shuffles", "40")
        scoped, _ = self.facts("--shuffle-scope", "place_day")
        self.assertEqual(other["params"]["seed"], 7)
        self.assertEqual(scoped["nullmodel"]["scope"], "place_day")
        for facts in (other, scoped):
            self.assertEqual(facts["cascade"], base["cascade"])
            self.assertEqual(facts["nullmodel"]["links_named"]["observed"],
                             base["nullmodel"]["links_named"]["observed"])
            self.assertNotEqual(facts["core_sha256"], base["core_sha256"])

    def test_bad_scope_rejected(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            run_cli(["--adapter", "generic", "--input", EXAMPLE, "--out", self.tmp, "--shuffle-scope", "week"])


class Guards(unittest.TestCase):
    def test_out_dir_inside_repo_refused(self):
        with self.assertRaises(ValueError):
            P.check_out_dir(os.path.join(ROOT, "patient_zero", "x"))
        with self.assertRaises(ValueError):
            P.check_out_dir(ROOT)
        P.check_out_dir(os.path.join(ROOT, "out", "example"))  # git-ignored: allowed
        P.check_out_dir(tempfile.gettempdir())

    def test_cli_refuses_repo_out_and_writes_nothing(self):
        target = os.path.join(ROOT, "docs", "never-here")
        code, text = run_cli(["--adapter", "generic", "--input", EXAMPLE, "--out", target])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", text)
        self.assertFalse(os.path.exists(target))

    def test_no_tokens_and_mixed_groups(self):
        tmp = tempfile.mkdtemp(prefix="pz-pipeline-")
        try:
            log = os.path.join(tmp, "tiny.ndjson")
            with open(log, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"agent": "A", "time": 0, "place": "p", "text": "hello", "group": "gä"}) + "\n")
                fh.write(json.dumps({"agent": "B", "time": 60, "place": "p", "text": "hi"}) + "\n")
            code, text = run_cli(["--adapter", "generic", "--input", log, "--out", os.path.join(tmp, "out")])
            self.assertEqual(code, 0, text)
            with open(os.path.join(tmp, "out", "facts.json"), encoding="utf-8") as fh:
                facts = json.load(fh)
            self.assertEqual(facts["trees"]["total"], 0)
            self.assertIsNone(facts["slots"]["F8"]["share_seed_is_not_first_user"])
            self.assertEqual(sorted(facts["load"]["groups"]), [P.NO_FAMILY, "g\\u00e4"])
            with open(os.path.join(tmp, "out", "trees.jsonl"), encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_input(self):
        tmp = tempfile.mkdtemp(prefix="pz-pipeline-")
        try:
            code, text = run_cli(["--adapter", "wiki", "--input", os.path.join(tmp, "nope"),
                                  "--out", os.path.join(tmp, "out")])
            self.assertEqual(code, 2)
            self.assertIn("ERROR", text)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class Helpers(unittest.TestCase):
    def test_short_escapes(self):
        self.assertIsNone(P.short(None))
        self.assertEqual(P.short("plain"), "plain")
        self.assertEqual(P.short("a\\b"), "a\\\\b")
        self.assertEqual(P.short("tab\there"), "tab\\u0009here")
        self.assertEqual(P.short("für"), "f\\u00fcr")
        self.assertEqual(P.short("‮evil"), "\\u202eevil")
        self.assertEqual(P.short("\U0001F600"), "\\U0001f600")

    def test_short_cuts_at_escape_boundary(self):
        out = P.short("ä" * 100)
        self.assertLessEqual(len(out), P.SHORT_MAX)
        self.assertTrue(out.endswith("..."))
        self.assertRegex(out[:-3], r"^(\\u00e4)+$")
        self.assertEqual(len(P.short("x" * 200)), P.SHORT_MAX)
        self.assertEqual(P.short("y" * P.SHORT_MAX), "y" * P.SHORT_MAX)

    def test_token_id(self):
        self.assertEqual(P.token_id("host:a.org"), P.token_id("host:a.org"))
        self.assertNotEqual(P.token_id("host:a.org"), P.token_id("host:b.org"))
        self.assertRegex(P.token_id("x"), r"^[0-9a-f]{12}$")

    def test_branch_sizes(self):
        def a(agent, n, status, parent=None):
            return C.Adoption(agent=agent, row=n, time=float(n), status=status, parent=parent, link=None)

        adoptions = (
            a("A", 0, C.STATUS_FIRST), a("B", 1, C.STATUS_LINKED, "A"), a("C", 2, C.STATUS_LINKED, "B"),
            a("D", 3, C.STATUS_PARALLEL), a("E", 4, C.STATUS_LINKED, "D"), a("F", 5, C.STATUS_LINKED, "A"),
            a("G", 6, C.STATUS_UNNAMED),
        )
        tree = C.Tree(token="t", cls="host", kind="host", adoptions=adoptions, links=(), first_user="A",
                      effective_seed="A", parallel_starts=("D",), dropped=0, shifted=0, depth=2, places=1)
        self.assertEqual(P.branch_sizes(tree), {"A": 4, "D": 2, "G": 1})


@unittest.skipUnless(os.environ.get("PZ_DATA"), "set PZ_DATA to the collusion.wiki export folder")
class RealDataRun(unittest.TestCase):
    """Gate G1: the whole pipeline on all 4 wikis, under the 5-minute target."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-pipeline-real-")
        cls.code, _ = run_cli(["--adapter", "wiki", "--input", os.environ["PZ_DATA"], "--out", cls.tmp])
        with open(os.path.join(cls.tmp, "facts.json"), encoding="utf-8") as fh:
            cls.facts = json.load(fh)
        with open(os.path.join(cls.tmp, "trees.jsonl"), encoding="utf-8") as fh:
            cls.records = [json.loads(line) for line in fh]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_end_to_end(self):
        f = self.facts
        self.assertEqual(self.code, 0)
        self.assertEqual(f["load"]["rows"], 14591)
        self.assertEqual(len(f["groups"]), 4)
        self.assertEqual(f["trees"]["total"], f["tokens"]["kept_total"])
        self.assertEqual(len(self.records), f["trees"]["total"])
        self.assertTrue(f["runtime"]["under_target"])
        self.assertEqual(sorted(x["name"] for x in f["input"]["files"]), sorted(P.WIKI_FILES))
        self.assertTrue(f["nullmodel"]["observed_matches_cascade"])
        self.assertEqual(f["nullmodel"]["shuffles"], 100)
        self.assertEqual(f["nullmodel"]["links_named"]["shuffles"], 100)
        self.assertTrue(f["scores"]["family_flow"]["observed_matches_nullmodel"])
        self.assertEqual(f["scores"]["influence"]["named_links"], f["cascade"]["links_named"])

    def test_human_handles_never_named(self):
        humans = set()
        path = os.path.join(os.environ["PZ_DATA"], "labels.jsonl.gz")
        for _n, lab in L._read_json_lines(path, L._LABEL_KEYS):
            if lab.get("is_human_handle") is True and isinstance(lab.get("label"), str):
                humans.add(P.short(lab["label"].strip()))
        named = set()
        for rec in self.records:
            for key in ("first_user", "effective_seed", "largest_branch_root"):
                if rec[key] is not None:
                    named.add(rec[key])
        influence = set(row["agent"] for row in self.facts["scores"]["influence"]["top"])
        self.assertEqual(len(humans & named), 0)
        self.assertEqual(len(humans & influence), 0)

    def test_strings_short(self):
        too_long = sum(1 for s in _strings(self.records) if len(s) > P.SHORT_MAX)
        self.assertEqual(too_long, 0)
        non_ascii = sum(1 for s in _strings(self.records) if re.search(r"[^\x20-\x7e]", s))
        self.assertEqual(non_ascii, 0)
        scores = self.facts["scores"]
        data_strings = [row["agent"] for row in scores["influence"]["top"]]
        data_strings += [p[k] for p in scores["family_flow"]["top_pairs"] for k in ("from", "to")]
        self.assertEqual(sum(1 for s in data_strings if len(s) > P.SHORT_MAX), 0)
        self.assertEqual(sum(1 for s in data_strings if re.search(r"[^\x20-\x7e]", s)), 0)


if __name__ == "__main__":
    unittest.main()
