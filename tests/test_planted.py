"""Planted-cascade tests for patient_zero/cascade.py (checkpoint C2).

unittest, standard library only. Every name, host and line here is made up.

* ChatPlanted: a small chat log (generic adapter) with three planted
  cascades: a chain, a star, and a chain whose second link falls inside the
  clock uncertainty (so the effective seed is not the first user). All three
  must come back exactly: every parent, the first user and the effective seed.
* WikiInheritedText: a wiki export where later editors keep an earlier
  editor's line in the page body without adding it. Counting only added text,
  that inherited text must make 0 links (a naive full-body reading would make 3).
* ParallelStart: later users who began with no exposure to any earlier user
  are flagged as parallel starts; one of them becomes the effective seed.
* Rules: smaller cases for the exposure and clock rules.
"""

import gzip
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from patient_zero import cascade as C  # noqa: E402
from patient_zero import load as L  # noqa: E402
from patient_zero import tokens as T  # noqa: E402

BASE = 1768039200.0  # 2026-01-10T10:00:00Z


def iso(seconds_after_base):
    return L.iso_utc(BASE + seconds_after_base)


def make_row(i, agent, t, place, text, unc=0.0, role=None):
    if role is None:
        role = L.ROLE_AGENT if agent else L.ROLE_BLANK
    return L.Row(
        row_id="r%03d" % i,
        agent=agent if role == L.ROLE_AGENT else None,
        role=role,
        time=BASE + float(t),
        uncertainty=float(unc),
        time_grade=None,
        place=place,
        family=None,
        group=None,
        text=text,
    )


def rows_of(specs):
    """specs: (agent or None, seconds after BASE, place, text[, unc[, role]]), in time order."""
    rows = [make_row(i, *spec) for i, spec in enumerate(specs)]
    rows.sort(key=lambda r: (r.time, r.row_id))
    return rows


def trees_for(rows, min_agents=3):
    tokens, _ = T.find_tokens(rows, min_agents=min_agents)
    trees, stats = C.build_trees(rows, tokens)
    return trees, stats


# ---------------------------------------------------------------- chat fixture

M = 60  # one minute

# (id, agent, seconds after BASE, room, text). Uncertainty is 1 s on every line.
CHAT = [
    # cascade 1, a chain through three rooms: Ada -> Bram -> Cleo -> Dax
    ("c01", "Ada", 0 * M, "chat/north", "mirror at lantern-depot.io is quick today"),
    ("c02", "Moss", 2 * M, "chat/north", "lunch moved to one o'clock"),
    ("c03", "Bram", 5 * M, "chat/north", "thanks, reading it now"),
    ("c04", "Bram", 10 * M, "chat/east", "east team: lantern-depot.io has the files"),
    ("c05", "Cleo", 15 * M, "chat/east", "noted, will pass it on"),
    ("c06", "Cleo", 20 * M, "chat/south", "south room, try lantern-depot.io first"),
    ("c07", "Dax", 25 * M, "chat/south", "switching to lantern-depot.io as well"),
    # cascade 2, a star: Eve's word reaches three desks that never see each other
    ("c08", "Eve", 30 * M, "chat/plaza", "BLUEHERON: put this word on urgent notes"),
    ("c09", "Finn", 32 * M, "chat/plaza", "ok"),
    ("c10", "Finn", 34 * M, "chat/finn-desk", "BLUEHERON checking the queue"),
    ("c11", "Gus", 36 * M, "chat/plaza", "seen"),
    ("c12", "Moss", 37 * M, "chat/plaza", "who has the projector key"),
    ("c13", "Gus", 38 * M, "chat/gus-desk", "BLUEHERON queue looks clear"),
    ("c14", "Hal", 40 * M, "chat/plaza", "fine by me"),
    ("c15", "Hal", 42 * M, "chat/hal-desk", "BLUEHERON taking the late shift"),
    # cascade 3, Jun's line lands 1 s after Ivy's: inside both clocks (1 s + 1 s)
    ("c16", "Ivy", 50 * M, "chat/harbor", "uploaded quill_map.csv for everyone"),
    ("c17", "Jun", 50 * M + 1, "chat/harbor", "same here, quill_map.csv attached"),
    ("c18", "Kai", 55 * M, "chat/harbor", "using quill_map.csv from now on"),
    ("c19", "Moss", 57 * M, "chat/harbor", "tea is ready downstairs"),
    ("c20", "Lux", 60 * M, "chat/harbor", "quill_map.csv works for me too"),
]

PLANTED = {
    "host:lantern-depot.io": {
        "parents": {"Ada": None, "Bram": "Ada", "Cleo": "Bram", "Dax": "Cleo"},
        "first_user": "Ada", "effective_seed": "Ada", "depth": 3, "dropped": 0,
    },
    "convention:control:BLUEHERON": {
        "parents": {"Eve": None, "Finn": "Eve", "Gus": "Eve", "Hal": "Eve"},
        "first_user": "Eve", "effective_seed": "Eve", "depth": 1, "dropped": 0,
    },
    "file:quill_map.csv": {
        "parents": {"Ivy": None, "Jun": None, "Kai": "Jun", "Lux": "Kai"},
        "first_user": "Ivy", "effective_seed": "Jun", "depth": 2, "dropped": 1,
    },
}

CHAT_NAMES = sorted(set(line[1] for line in CHAT))


def write_chat(folder, name="chat.ndjson"):
    """Write the planted chat log; every line is labelled made-up in its group field.

    The shipped demo examples/planted_chat.ndjson is this file, byte for byte
    (tests/test_demo.py checks it). To regenerate it from the repo root:
    python -c "from tests.test_planted import write_chat; write_chat('examples', 'planted_chat.ndjson')"
    """
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rid, agent, t, room, text in CHAT:
            fh.write(json.dumps({
                "id": rid, "agent": agent, "time": iso(t), "place": room,
                "uncertainty_seconds": 1, "group": "made-up", "text": text,
            }) + "\n")
    return path


class ChatPlanted(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-planted-")
        cls.path = write_chat(cls.tmp)
        cls.rows, _ = L.load("generic", cls.path)
        cls.tokens, _ = T.find_tokens(cls.rows)
        cls.trees, cls.stats = C.build_trees(cls.rows, cls.tokens)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_all_three_planted_cascades_found(self):
        for key, want in PLANTED.items():
            with self.subTest(token=key):
                self.assertIn(key, self.trees, "token finder lost the planted token")
                tree = self.trees[key]
                self.assertEqual(C.parents_of(tree), want["parents"])
                self.assertEqual(tree.first_user, want["first_user"])
                self.assertEqual(tree.effective_seed, want["effective_seed"])
                self.assertEqual(tree.depth, want["depth"])
                self.assertEqual(tree.dropped, want["dropped"])
                self.assertEqual(tree.parallel_starts, ())

    def test_chain_statuses_and_links(self):
        tree = self.trees["host:lantern-depot.io"]
        self.assertEqual([a.status for a in tree.adoptions], ["first", "linked", "linked", "linked"])
        self.assertEqual(C.children_of(tree), {"Ada": ["Bram"], "Bram": ["Cleo"], "Cleo": ["Dax"]})
        bram = tree.adoptions[1].link
        # Bram saw Ada's line through his own earlier message in the same room
        self.assertEqual(self.rows[bram.visit_row].row_id, "c03")
        self.assertEqual(self.rows[bram.source_row].row_id, "c01")
        self.assertEqual(self.rows[bram.target_row].row_id, "c04")
        self.assertEqual(bram.lag, 10 * M)

    def test_star_has_one_seed_and_three_children(self):
        tree = self.trees["convention:control:BLUEHERON"]
        self.assertEqual(C.children_of(tree), {"Eve": ["Finn", "Gus", "Hal"]})
        self.assertEqual(tree.places, 4)

    def test_clock_drop_moves_the_effective_seed(self):
        tree = self.trees["file:quill_map.csv"]
        self.assertEqual([a.status for a in tree.adoptions], ["first", "clock_dropped", "linked", "linked"])
        self.assertIsNone(tree.adoptions[1].link)
        self.assertNotEqual(tree.first_user, tree.effective_seed)

    def test_every_link_is_in_clock_order(self):
        for tree in self.trees.values():
            for link in tree.links:
                u, b = self.rows[link.source_row], self.rows[link.target_row]
                self.assertTrue(T.in_clock_order(u.time, u.uncertainty, b.time, b.uncertainty))
                self.assertGreater(link.lag, 0)

    def test_stats_are_counts(self):
        s = self.stats
        self.assertGreaterEqual(s["links_named"], 8)  # 3 + 3 + 2 planted named links
        self.assertGreaterEqual(s["links_dropped_by_clock"], 1)
        self.assertEqual(s["status"]["linked"], s["links_named"])
        self.assertEqual(sum(s["status"].values()), s["adoptions"])

    def test_deterministic(self):
        rows, _ = L.load("generic", self.path)
        trees, stats = trees_for(rows)
        stats.pop("seconds")
        mine = dict(self.stats)
        mine.pop("seconds")
        self.assertEqual(stats, mine)
        self.assertEqual(trees, self.trees)

    def test_cli_prints_counts_only(self):
        out = io.StringIO()
        summary = os.path.join(self.tmp, "out", "cascade-summary.json")
        code = C.main(["--adapter", "generic", "--input", self.path, "--summary", summary], out=out)
        self.assertEqual(code, 0)
        text = out.getvalue()
        with open(summary, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), text)
        data = json.loads(text)
        self.assertEqual(data["job"], "J4 cascade")
        self.assertEqual(data["trees"], len(self.trees))
        for name in CHAT_NAMES:
            self.assertNotIn(name, text)
        for needle in ("lantern", "BLUEHERON", "quill", "chat/"):
            self.assertNotIn(needle, text)

    def test_cli_error_is_reported(self):
        out = io.StringIO()
        code = C.main(["--adapter", "generic", "--input", os.path.join(self.tmp, "missing.ndjson")], out=out)
        self.assertEqual(code, 2)
        self.assertIn("cascade: ERROR", out.getvalue())


# ---------------------------------------------------------------- wiki fixture


def _write_gz(path, items):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")


def _h(op, a0, a1, b0, b1):
    return {"op": op, "a0": a0, "a1": a1, "b0": b0, "b1": b1}


HARBOUR_1 = "Harbour notes\nmaps live at saltmarsh-atlas.org\nend"
HARBOUR_2 = HARBOUR_1 + "\nBram checked the tide table"
HARBOUR_3 = HARBOUR_2 + "\nCleo tidied the header line"
HARBOUR_4 = HARBOUR_3.replace("Harbour notes", "Harbour notes, autumn", 1)

# (rev_id, page_id, label, seconds after BASE, body, hunks, diff_base)
WIKI = [
    ("w1", "wiki/Harbour", "Ada", 0, HARBOUR_1, [], None),
    ("w2", "wiki/Harbour", "Bram", 10 * M, HARBOUR_2, [_h("insert", 3, 3, 3, 4)], "w1"),
    ("w3", "wiki/Harbour", "Cleo", 20 * M, HARBOUR_3, [_h("insert", 4, 4, 4, 5)], "w2"),
    ("w4", "wiki/Harbour", "Dax", 30 * M, HARBOUR_4, [_h("replace", 0, 1, 0, 1)], "w3"),
    ("w5", "wiki/Garden", "Eve", 60 * M, "seed list\nsee saltmarsh-atlas.org for the map", [], None),
    ("w6", "wiki/Library", "Finn", 90 * M, "catalogue\nsaltmarsh-atlas.org has a mirror", [], None),
]
INHERITED = "host:saltmarsh-atlas.org"


class WikiInheritedText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-wiki-")
        _write_gz(os.path.join(cls.tmp, "revisions.jsonl.gz"), [
            {"rev_id": rid, "page_id": page, "wiki": "alpha", "label": label, "time": iso(t),
             "time_grade": "reqlog", "uncertainty_seconds": 1, "body": body, "hunks": hunks,
             "diff_base": base}
            for rid, page, label, t, body, hunks, base in WIKI
        ])
        _write_gz(os.path.join(cls.tmp, "pages.jsonl.gz"), [
            {"page_id": "wiki/Harbour", "page_family": "relay-coordination"},
            {"page_id": "wiki/Garden", "page_family": "probe-test"},
            {"page_id": "wiki/Library", "page_family": "source-cache-url-list"},
        ])
        cls.rows, _ = L.load("wiki", cls.tmp)
        cls.trees, cls.stats = trees_for(cls.rows)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_later_editors_did_not_add_the_line(self):
        by_id = {r.row_id: r for r in self.rows}
        for rid in ("w2", "w3", "w4"):
            self.assertNotIn("saltmarsh", by_id[rid].text)
        self.assertIn("saltmarsh", by_id["w1"].text)

    def test_inherited_text_makes_zero_links(self):
        self.assertIn(INHERITED, self.trees)  # the token is live: 3 agents added it
        tree = self.trees[INHERITED]
        self.assertEqual([a.agent for a in tree.adoptions], ["Ada", "Eve", "Finn"])
        self.assertEqual(tree.links, ())
        self.assertIsNone(tree.effective_seed)
        self.assertEqual(tree.parallel_starts, ("Eve", "Finn"))
        self.assertEqual(self.stats["links_named"], 0)
        self.assertEqual(self.stats["links_from_exposure_only_rows"], 0)

    def test_naive_full_body_reading_would_make_links(self):
        """Sanity check of the fixture: reading whole bodies credits the editors."""
        bodies = {rid: body for rid, _p, _l, _t, body, _h, _b in WIKI}
        naive = [r._replace(text=bodies[r.row_id]) for r in self.rows]
        trees, _ = trees_for(naive)
        self.assertEqual(C.parents_of(trees[INHERITED]),
                         {"Ada": None, "Bram": "Ada", "Cleo": "Bram", "Dax": "Cleo", "Eve": None,
                          "Finn": None})


# ---------------------------------------------------------------- parallel starts


class ParallelStart(unittest.TestCase):
    def setUp(self):
        self.rows = rows_of([
            ("Wes", -1 * M, "room/a", "morning all"),  # visits room/a before Uma's use: no exposure
            ("Uma", 0, "room/a", "found ferrylight.net, it hosts the list"),
            ("Vic", 10 * M, "room/b", "ferrylight.net seems to have the list"),
            ("Wes", 20 * M, "room/c", "list copy at ferrylight.net"),
            ("Vic", 25 * M, "room/a", "hello room a"),  # after Vic's first use: does not count
            ("Xan", 30 * M, "room/b", "catching up on this room"),
            ("Xan", 40 * M, "room/d", "using ferrylight.net now"),
        ])
        self.trees, self.stats = trees_for(self.rows)
        self.tree = self.trees["host:ferrylight.net"]

    def test_parallel_starts_flagged(self):
        self.assertEqual(self.tree.parallel_starts, ("Vic", "Wes"))
        self.assertEqual([a.status for a in self.tree.adoptions], ["first", "parallel", "parallel", "linked"])
        self.assertEqual(self.stats["parallel_starts"], 2)

    def test_parallel_starter_becomes_effective_seed(self):
        self.assertEqual(self.tree.first_user, "Uma")
        self.assertEqual(self.tree.effective_seed, "Vic")
        self.assertEqual(C.parents_of(self.tree), {"Uma": None, "Vic": None, "Wes": None, "Xan": "Vic"})
        self.assertEqual(len(self.tree.links), 1)


# ---------------------------------------------------------------- rules


class Rules(unittest.TestCase):
    def test_most_recent_visible_use_gets_credit(self):
        rows = rows_of([
            ("Ana", 0, "r/1", "notes at tidewell.org"),
            ("Ben", 5 * M, "r/1", "tidewell.org confirmed"),
            ("Cal", 9 * M, "r/1", "tidewell.org for me too"),
        ])
        tree = trees_for(rows)[0]["host:tidewell.org"]
        self.assertEqual(C.parents_of(tree), {"Ana": None, "Ben": "Ana", "Cal": "Ben"})

    def test_clock_shift_credits_an_older_use(self):
        rows = rows_of([
            ("Ana", 0, "r/1", "notes at tidewell.org", 1),
            ("Ben", 100, "r/1", "tidewell.org confirmed", 1),
            ("Cal", 100.5, "r/1", "tidewell.org for me too", 1),  # Ben's line is 0.5 s earlier
        ])
        trees, stats = trees_for(rows)
        tree = trees["host:tidewell.org"]
        self.assertEqual(C.parents_of(tree), {"Ana": None, "Ben": "Ana", "Cal": "Ana"})
        self.assertEqual(tree.shifted, 1)
        self.assertEqual(stats["links_shifted_by_clock"], 1)
        self.assertEqual(tree.dropped, 0)

    def test_visit_inside_clock_uncertainty_is_not_exposure(self):
        rows = rows_of([
            ("Ana", 0, "r/1", "notes at tidewell.org", 5),
            ("Ben", 3, "r/1", "hi", 5),  # visit 3 s later, clocks +/- 5 s: order unknown
            ("Ben", 10 * M, "r/2", "tidewell.org works", 5),
            ("Cal", 20 * M, "r/3", "tidewell.org here", 5),
        ])
        tree = trees_for(rows)[0]["host:tidewell.org"]
        self.assertEqual([a.status for a in tree.adoptions], ["first", "clock_dropped", "parallel"])
        self.assertEqual(tree.links, ())

    def test_exposure_only_source_is_unnamed(self):
        rows = rows_of([
            ("Ana", 0, "r/1", "notes at tidewell.org"),
            ("Ben", 5 * M, "r/2", "tidewell.org confirmed"),
            (None, 6 * M, "r/9", "tidewell.org listed here", 0, L.ROLE_BLANK),
            ("Cal", 7 * M, "r/9", "reading the blank page"),
            ("Cal", 8 * M, "r/3", "tidewell.org seen"),
        ])
        tree = trees_for(rows)[0]["host:tidewell.org"]
        cal = tree.adoptions[2]
        self.assertEqual(cal.status, "unnamed")
        self.assertIsNone(cal.parent)
        self.assertIsNone(cal.link.source)
        self.assertEqual(rows[cal.link.source_row].role, L.ROLE_BLANK)
        self.assertNotIn(None, [a.agent for a in tree.adoptions])

    def test_first_use_order_and_visits(self):
        rows = rows_of([
            ("Ana", 0, "r/1", "x"),
            ("Ben", 1, "r/1", "y"),
            ("Ana", 2, "r/2", "z"),
            (None, 3, "r/1", "w", 0, L.ROLE_HUMAN),
        ])
        visits = C.build_visits(rows)
        self.assertEqual(visits, {"Ana": {"r/1": [0], "r/2": [2]}, "Ben": {"r/1": [1]}})
        self.assertEqual(C.first_uses(rows, [0, 1, 2, 3]), [("Ana", 0), ("Ben", 1)])

    def test_example_file_runs(self):
        rows, _ = L.load("generic", os.path.join(ROOT, "examples", "fake_log.ndjson"))
        trees, stats = trees_for(rows)
        self.assertEqual(stats["trees"], len(trees))
        for tree in trees.values():
            self.assertEqual(tree.first_user, tree.adoptions[0].agent)


if __name__ == "__main__":
    unittest.main()
