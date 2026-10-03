"""Tests for patient_zero/load.py (unittest, standard library only).

All fixture text is made up here. IP-shaped strings and the half-IP field
name are assembled at run time, so this file passes the leak check itself.

The real-data check (checkpoint C1) runs only when PZ_DATA names the
collusion.wiki export folder; otherwise it is skipped.
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

from patient_zero import load as L  # noqa: E402

FIELD = "ip" + "16"
HALF = ".".join(["203", "113"])
V4 = ".".join(["198", "51", "100", "7"])
V4_WILD = ".".join(["203", "0", "x", "x"])
V6 = ":".join(["2001", "db8", "0", "0", "0", "0", "0", "1"])
HUMAN = "PersonHandleQ"
EXAMPLE = os.path.join(ROOT, "examples", "fake_log.ndjson")


def _write_gz(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _save(rev_id, page_id, label, time, body, hunks, base=None, wiki="alpha", **extra):
    row = {
        "rev_id": rev_id, "page_id": page_id, "page_key": "k-" + rev_id, "wiki": wiki,
        "name": page_id, "label": label, "time": time, "time_grade": "reqlog",
        "uncertainty_seconds": 1, "body": body, "lines": len(body.split("\n")),
        "hunks": hunks, "diff_base": base, FIELD: HALF, "change_summary": "summary text",
    }
    row.update(extra)
    return row


def _h(op, a0, a1, b0, b1):
    return {"op": op, "a0": a0, "a1": a1, "b0": b0, "b1": b1}


class WikiFixture(unittest.TestCase):
    """A tiny made-up export with every case the adapter handles."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="pz-load-")
        saves = [
            # Page created: hunk covers the whole body.
            _save("a~P1@1", "alpha/P1", "AgentOne", "2026-01-01T10:00:00Z",
                  "first line\nsecond line", [_h("insert", 0, 0, 0, 2)]),
            # Insert one line, replace one, delete one: only new-side insert/replace lines count.
            _save("a~P1@2", "alpha/P1", "AgentTwo", "2026-01-01T10:05:00Z",
                  "first line\nadded by two\nsecond line changed",
                  [_h("insert", 1, 1, 1, 2), _h("replace", 1, 2, 2, 3), _h("delete", 2, 3, 3, 3)],
                  base="a~P1@1"),
            # No hunks but a base: difflib finds the added line.
            _save("a~P1@3", "alpha/P1", "AgentOne", "2026-01-01T10:10:00Z",
                  "first line\nadded by two\nsecond line changed\nlate tail from one", [],
                  base="a~P1@2"),
            # No hunks, identical to base: nothing added.
            _save("a~P1@4", "alpha/P1", "AgentTwo", "2026-01-01T10:12:00Z",
                  "first line\nadded by two\nsecond line changed\nlate tail from one", [],
                  base="a~P1@3"),
            # Blank label on another wiki: exposure-only.
            _save("b~P2@1", "beta/P2", "", "2026-01-01T10:01:00Z",
                  "blank note", [_h("insert", 0, 0, 0, 1)], wiki="beta"),
            # Human handle: exposure-only, never named.
            _save("b~P2@2", "beta/P2", HUMAN, "2026-01-01T10:02:00Z",
                  "blank note\nhuman note", [_h("insert", 1, 1, 1, 2)], base="b~P2@1", wiki="beta"),
            # IP-shaped values in text are masked.
            _save("a~P3@1", "alpha/P3", "AgentThree", "2026-01-01T09:59:00Z",
                  "host %s and %s and %s here" % (V4, V4_WILD, V6), [_h("insert", 0, 0, 0, 1)]),
        ]
        pages = [
            {"page_id": "alpha/P1", "page_family": "relay", "n_" + FIELD: 2},
            {"page_id": "beta/P2", "page_family": "probe", "n_" + FIELD: 1},
            {"page_id": "alpha/P3", "page_family": "", "n_" + FIELD: 1},
        ]
        labels = [
            {"label": "AgentOne", "is_human_handle": False, "stored_revision_" + FIELD: [HALF]},
            {"label": "AgentTwo", "is_human_handle": False},
            {"label": "AgentThree", "is_human_handle": False},
            {"label": HUMAN, "is_human_handle": True},
            {"label": "", "is_human_handle": False},
        ]
        _write_gz(os.path.join(cls.dir, "revisions.jsonl.gz"), saves)
        _write_gz(os.path.join(cls.dir, "pages.jsonl.gz"), pages)
        _write_gz(os.path.join(cls.dir, "labels.jsonl.gz"), labels)
        cls.rows, cls.stats = L.load("wiki", cls.dir)
        cls.by_id = {r.row_id: r for r in cls.rows}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_counts(self):
        s = self.stats
        self.assertEqual(s["rows"], 7)
        self.assertEqual(s["labels_with_text"], 4)  # three agents plus the human handle
        self.assertEqual(s["agents"], 3)
        self.assertEqual(s["blank_saves"], 1)
        self.assertEqual(s["human_saves"], 1)
        self.assertEqual(s["human_labels"], 1)
        self.assertEqual(s["groups"], {"alpha": 5, "beta": 2})
        self.assertEqual(s["group_count"], 2)
        self.assertEqual(s["text_from_hunks"], 5)
        self.assertEqual(s["text_from_difflib"], 2)
        self.assertEqual(s["saves_without_added_text"], 1)
        self.assertEqual(s["ip_masked_in_text"], 3)
        self.assertEqual(s["families"], 2)

    def test_page_created_adds_whole_body(self):
        self.assertEqual(self.by_id["a~P1@1"].text, "first line\nsecond line")

    def test_hunks_count_only_added_lines(self):
        self.assertEqual(self.by_id["a~P1@2"].text, "added by two\nsecond line changed")

    def test_difflib_fallback(self):
        self.assertEqual(self.by_id["a~P1@3"].text, "late tail from one")
        self.assertEqual(self.by_id["a~P1@4"].text, "")

    def test_exposure_only_rows(self):
        blank, human = self.by_id["b~P2@1"], self.by_id["b~P2@2"]
        self.assertIsNone(blank.agent)
        self.assertEqual(blank.role, L.ROLE_BLANK)
        self.assertTrue(blank.exposure_only)
        self.assertIsNone(human.agent)
        self.assertEqual(human.role, L.ROLE_HUMAN)
        self.assertEqual(human.text, "human note")  # text kept for exposure
        self.assertFalse(self.by_id["a~P1@1"].exposure_only)

    def test_human_handle_never_leaves_loader(self):
        blob = repr(self.rows) + json.dumps(self.stats)
        self.assertNotIn(HUMAN, blob)

    def test_half_ip_field_dropped(self):
        blob = repr(self.rows) + json.dumps(self.stats)
        self.assertNotIn(FIELD, blob)
        self.assertNotIn(HALF, blob)
        self.assertNotIn("summary text", blob)  # non-whitelisted fields never kept
        self.assertEqual(set(L.Row._fields),
                         {"row_id", "agent", "role", "time", "uncertainty", "time_grade",
                          "place", "family", "group", "text"})

    def test_ip_values_masked(self):
        text = self.by_id["a~P3@1"].text
        for value in (V4, V4_WILD, V6):
            self.assertNotIn(value, text)
        self.assertEqual(text, "host [ip] and [ip] and [ip] here")

    def test_place_family_time(self):
        r = self.by_id["a~P1@2"]
        self.assertEqual(r.place, "alpha/P1")
        self.assertEqual(r.family, "relay")
        self.assertEqual(r.group, "alpha")
        self.assertEqual(r.time_grade, "reqlog")
        self.assertEqual(r.uncertainty, 1.0)
        self.assertEqual(L.iso_utc(r.time), "2026-01-01T10:05:00Z")
        self.assertIsNone(self.by_id["a~P3@1"].family)  # empty family -> None

    def test_rows_sorted_by_time(self):
        times = [r.time for r in self.rows]
        self.assertEqual(times, sorted(times))
        self.assertEqual(self.rows[0].row_id, "a~P3@1")

    def test_deterministic(self):
        rows2, stats2 = L.load("wiki", self.dir)
        self.assertEqual(rows2, self.rows)
        self.assertEqual(stats2, self.stats)

    def test_ip_shaped_label_is_blank(self):
        tmp = tempfile.mkdtemp(prefix="pz-load-ip-")
        try:
            _write_gz(os.path.join(tmp, "revisions.jsonl.gz"),
                      [_save("x~P@1", "x/P", V4, "2026-01-01T00:00:00Z", "t", [_h("insert", 0, 0, 0, 1)])])
            _write_gz(os.path.join(tmp, "pages.jsonl.gz"), [{"page_id": "x/P", "page_family": "f"}])
            rows, stats = L.load_wiki(tmp)
            self.assertIsNone(rows[0].agent)
            self.assertEqual(rows[0].role, L.ROLE_BLANK)
            self.assertEqual(stats["ip_shaped_labels_blanked"], 1)
            self.assertNotIn(V4, repr(rows))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_files(self):
        tmp = tempfile.mkdtemp(prefix="pz-load-empty-")
        try:
            with self.assertRaises(FileNotFoundError):
                L.load_wiki(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class GenericAdapter(unittest.TestCase):
    def _load_lines(self, lines, suffix=".ndjson"):
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="pz-gen-")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                for line in lines:
                    fh.write((line if isinstance(line, str) else json.dumps(line)) + "\n")
            return L.load("generic", path)
        finally:
            os.remove(path)

    def test_example_file(self):
        rows, stats = L.load("generic", EXAMPLE)
        self.assertEqual(stats["rows"], 14)
        self.assertEqual(stats["agents"], 5)
        self.assertEqual(stats["blank_rows"], 1)
        self.assertEqual(stats["human_rows"], 1)
        self.assertEqual(stats["places"], 3)
        self.assertNotIn("Host", [r.agent for r in rows])
        self.assertEqual([r.row_id for r in rows], ["e%02d" % i for i in range(1, 15)])
        self.assertEqual(rows[11].uncertainty, 30.0)

    def test_fields_and_defaults(self):
        rows, stats = self._load_lines([
            {"agent": "A", "time": 100, "place": "room", "text": "hi", "extra": "dropped", FIELD: HALF},
            {"agent": "B", "time": "1970-01-01T00:03:20", "place": 7, "text": "yo"},
            {"agent": "C", "time": "1970-01-01T01:00:00+01:00", "place": "room", "text": "%s ok" % V4},
        ])
        self.assertEqual([r.agent for r in rows], ["C", "A", "B"])  # C is t=0 after the offset
        self.assertEqual(rows[0].time, 0.0)
        self.assertEqual(rows[1].time, 100.0)
        self.assertEqual(rows[2].time, 200.0)  # no zone means UTC
        self.assertEqual(rows[2].place, "7")
        self.assertEqual(rows[1].row_id, "line-1")
        self.assertEqual(rows[1].uncertainty, 0.0)
        self.assertEqual(rows[0].text, "[ip] ok")
        self.assertEqual(stats["ip_masked_in_text"], 1)
        blob = repr(rows)
        self.assertNotIn("dropped", blob)
        self.assertNotIn(HALF, blob)

    def test_human_and_blank(self):
        rows, stats = self._load_lines([
            {"agent": "Person", "human": True, "time": 1, "place": "p", "text": "a"},
            {"agent": None, "time": 2, "place": "p", "text": "b"},
            {"agent": "  ", "time": 3, "place": "p", "text": "c"},
        ])
        self.assertEqual([r.role for r in rows], [L.ROLE_HUMAN, L.ROLE_BLANK, L.ROLE_BLANK])
        self.assertTrue(all(r.agent is None for r in rows))
        self.assertNotIn("Person", repr(rows))
        self.assertEqual(stats["agents"], 0)

    def test_errors_name_the_line(self):
        cases = [
            ({"agent": "A", "time": 1, "place": "p"}, "missing field"),
            ({"agent": "A", "time": "not a time", "place": "p", "text": "t"}, "bad time"),
            ({"agent": "A", "time": 1, "place": "p", "text": 5}, "text must be"),
            ({"agent": "A", "time": 1, "place": "p", "text": "t", "uncertainty_seconds": -1}, "uncertainty"),
            ("[1, 2]", "JSON object"),
            ("{not json", "not valid JSON"),
        ]
        for line, msg in cases:
            with self.subTest(msg=msg):
                with self.assertRaises(ValueError) as ctx:
                    self._load_lines([{"agent": "Z", "time": 0, "place": "p", "text": ""}, line])
                self.assertIn("line 2", str(ctx.exception))
                self.assertIn(msg, str(ctx.exception))

    def test_duplicate_id(self):
        with self.assertRaises(ValueError):
            self._load_lines([
                {"id": "x", "agent": "A", "time": 1, "place": "p", "text": ""},
                {"id": "x", "agent": "B", "time": 2, "place": "p", "text": ""},
            ])

    def test_unknown_adapter(self):
        with self.assertRaises(ValueError):
            L.load("nope", EXAMPLE)

    def test_cli_prints_counts_only(self):
        buf = io.StringIO()
        self.assertEqual(L.main(["--adapter", "generic", "--input", EXAMPLE], out=buf), 0)
        stats = json.loads(buf.getvalue())
        self.assertEqual(stats["rows"], 14)
        self.assertNotIn("BLUEHERON", buf.getvalue())
        self.assertNotIn("Wren", buf.getvalue())


class ScrubIps(unittest.TestCase):
    def test_masks(self):
        for value in (V4, V4_WILD, ".".join(["10", "0", "*", "*"]), V6, "fe80" + "::" + "1ab"):
            with self.subTest(value=value):
                text, n = L.scrub_ips("at %s now" % value)
                self.assertEqual((text, n), ("at [ip] now", 1))

    def test_leaves_lookalikes(self):
        for text in ("version 1.2.3", "pi is 3.14159", "at 12:30:45 today", HALF,
                     "octets " + ".".join(["300", "1", "1", "1"]), "std::vector", "a.b.c.d"):
            with self.subTest(text=text):
                self.assertEqual(L.scrub_ips(text), (text, 0))

    def test_is_ip_shaped(self):
        self.assertTrue(L.is_ip_shaped(" %s " % V4))
        self.assertFalse(L.is_ip_shaped("host %s" % V4))
        self.assertFalse(L.is_ip_shaped(""))


class AddedText(unittest.TestCase):
    def test_no_hunks_no_base_is_whole_body(self):
        self.assertEqual(L.added_text("a\nb", [], None), "a\nb")
        self.assertEqual(L.added_text("", [], None), "")

    def test_delete_only(self):
        self.assertEqual(L.added_text("a", [_h("delete", 1, 2, 1, 1)], "a\nb"), "")


@unittest.skipUnless(os.environ.get("PZ_DATA"), "set PZ_DATA to the collusion.wiki export folder")
class RealDataCheckpointC1(unittest.TestCase):
    """Loader counts must match the published export profile."""

    def test_counts_match_profile(self):
        _rows, s = L.load("wiki", os.environ["PZ_DATA"])
        self.assertEqual(s["rows"], 14591)
        self.assertEqual(s["labels_with_text"], 3102)
        self.assertEqual(s["blank_saves"], 899)
        self.assertEqual(s["group_count"], 4)
        self.assertEqual(s["groups"], {"dse": 13403, "probier": 1013, "fractal": 169, "dorfwiki": 6})
        self.assertEqual(s["human_labels"], 3)
        self.assertEqual(s["pages"], 4579)
        self.assertEqual(s["time_span"], ["2026-05-24T06:02:19Z", "2026-07-02T17:51:22Z"])


if __name__ == "__main__":
    unittest.main()
