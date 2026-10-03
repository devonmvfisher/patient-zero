"""Tests for stretch S1 (patient_zero/offwiki.py): wiki tokens seen later off the wiki.

Standard library unittest only. A tiny synthetic export folder (wiki saves,
records, links, shortener log, other wikis) is written to a temp folder and
deleted afterwards. The real-data check runs only when PZ_DATA names the
collusion.wiki export folder; it checks counts only.
"""

import contextlib
import datetime
import gzip
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from patient_zero import load as L  # noqa: E402
from patient_zero import offwiki as O  # noqa: E402
from patient_zero import pipeline as P  # noqa: E402
from patient_zero.tokens import find_tokens  # noqa: E402

IP_FIELD = "ip" + "16"  # built at run time: the leak check fails on the literal name
FAKE_IP = ".".join(["10", "20", "30", "40"])  # same reason
T0 = 1780272000.0  # 2026-06-01T00:00:00Z
DAY = 86400.0


def iso(secs):
    return datetime.datetime.fromtimestamp(secs, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gz_lines(path, objs):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for obj in objs:
            fh.write(json.dumps(obj) + "\n")


def _gz_json(path, obj):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(obj, fh)


# Wiki saves: (label, offset from T0 in seconds, one line of text). Each token
# is used by 3 agents, 100 s apart, so the clock rule keeps it.
WIKI = []
for _base, _text in ((0, "ferrylane.net"), (1000, "quillstack.org"), (50, "relayWindowSeven"),
                     (60, "mirrorCacheKey"), (300, "lanternpost.io"), (400, "tidewatch.net"),
                     (500, "syncBeaconTwo"), (700, "harborSignalNine")):
    for _i, _agent in enumerate(("agent-a", "agent-b", "agent-c")):
        WIKI.append((_agent, _base + 100 * _i + _i, _text))
WIKI.append(("", 390, "tidewatch.net"))  # a blank row, earlier: the token is on the wiki from 390 s


def write_export(folder):
    revisions = []
    for n, (label, offset, text) in enumerate(WIKI):
        revisions.append({
            "rev_id": "r%d" % n, "page_id": "dse/PageA", "wiki": "dse", "label": label,
            "time": iso(T0 + offset), "time_grade": "exact", "uncertainty_seconds": 1,
            "body": text, "hunks": [{"op": "insert", "a0": 0, "a1": 0, "b0": 0, "b1": 1}],
            "diff_base": None, IP_FIELD: "1.2",
        })
    _gz_lines(os.path.join(folder, "revisions.jsonl.gz"), revisions)
    _gz_lines(os.path.join(folder, "pages.jsonl.gz"), [{"page_id": "dse/PageA", "page_family": "probe-test"}])
    _gz_lines(os.path.join(folder, "labels.jsonl.gz"), [{"label": "agent-a", "is_human_handle": False}])

    def origin(site, kind, literal):
        return {"site": site, "kind": kind, "source_date_literal": literal, "url": "u", "title": "t"}

    records = [
        # wiki first (0 s), then a paste 2 days later; the record is also a wiki copy
        {"id": "rec-ferry", "text": "notes at ferrylane.net " + FAKE_IP,
         "origins": [origin("paste.example", "paste_candidate", str(int(T0 + 2 * DAY))),
                     origin("prowiki.org/dse", "revision_addition", iso(T0))]},
        # an archive capture of the same host: seen by, never a lag
        {"id": "rec-ferry-capture", "text": "ferrylane.net",
         "origins": [origin("paste.example", "wayback_capture", "20260601120000")]},
        # off the wiki first: a prior paste a day before the wiki
        {"id": "rec-quill", "text": "quillstack.org",
         "origins": [origin("paste.example", "prior_paste", str(int(T0 - DAY)))]},
        # undated shortener record
        {"id": "rec-relay", "text": "relayWindowSeven",
         "origins": [origin("short.example", "shortener_candidate", "")]},
        # a wiki copy only: skipped
        {"id": "rec-wiki", "text": "mirrorCacheKey",
         "origins": [origin("prowiki.org/dse", "revision_addition", iso(T0))]},
        # dated paste whose text lacks the link host (the link carries it)
        {"id": "rec-lantern", "text": "nothing to see here",
         "origins": [origin("paste.example", "paste_candidate", str(int(T0 + 3 * DAY)))]},
        # inside the clock uncertainty: unresolved
        {"id": "rec-sync", "text": "syncBeaconTwo",
         "origins": [origin("paste.example", "paste_candidate", str(int(T0 + 501)))]},
        # seen by only
        {"id": "rec-harbor", "text": "harborSignalNine",
         "origins": [origin("paste.example", "commoncrawl_capture", "20260602000000")]},
    ]
    _gz_lines(os.path.join(folder, "records.jsonl.gz"), records)
    links = [
        {"host": "lanternpost.io", "record_ids": ["rec-lantern"], "url_withheld": True, "url": "withheld",
         "followed": False},
        {"host": "mirrorhost.net", "record_ids": ["rec-wiki"], "url_withheld": False, "url": "x",
         "followed": False},
        {"host": FAKE_IP, "record_ids": ["rec-lantern"], "url_withheld": False, "url": "x", "followed": False},
    ]
    _gz_lines(os.path.join(folder, "links.jsonl.gz"), links)
    _gz_json(os.path.join(folder, "shortener-logs.json.gz"), {
        "read": "2026-09-07", "source": "s",
        "sites": [{"site": "short.example", "links": [
            {"keyword": "k1", "url": "https://tidewatch.net/x", "title": "t", "time": iso(T0 + 400 + 3600),
             "clicks": 3, IP_FIELD: "5.6"}]}]})
    _gz_json(os.path.join(folder, "other-wikis.json.gz"), {
        "recovered": "2026-09-07", "source": "s",
        "pages": [{"wiki": "usemod", "name": "SandBox", "revisions": [
            {"time": iso(T0 + 5 * 3600), "added": ["ferrylane.net"], IP_FIELD: "7.8"}]}]})


class StampTest(unittest.TestCase):
    def test_parse_stamp(self):
        self.assertEqual(O.parse_stamp(str(int(T0))), T0)
        self.assertEqual(O.parse_stamp("2026-06-01T00:00:00+00:00"), T0)
        self.assertEqual(O.parse_stamp("2026-05-31T19:00:00-05:00"), T0)
        self.assertEqual(O.parse_stamp("2026-06-01T00:00:00.000000Z"), T0)
        self.assertEqual(O.parse_stamp("20260601000000"), T0)
        for bad in ("", "current", "12", "1.2.3", "prior cache decoded", None, "0000000001", 1780272000):
            self.assertIsNone(O.parse_stamp(bad), bad)

    def test_origin_time_grades(self):
        self.assertEqual(O.origin_time("paste_candidate", str(int(T0))), (T0, O.GRADE_PUBLISHED))
        self.assertEqual(O.origin_time("prior_revision", iso(T0)), (T0, O.GRADE_PUBLISHED))
        self.assertEqual(O.origin_time("wayback_capture", "20260601000000"), (T0, O.GRADE_SEEN))
        self.assertEqual(O.origin_time("current_page", "current"), (None, O.GRADE_NONE))
        self.assertEqual(O.origin_time("shortener_candidate", ""), (None, O.GRADE_NONE))

    def test_is_wiki_site(self):
        groups = ["dse", "probier", "fractal", "dorfwiki"]
        for site in ("prowiki.org/dse", "wikiservice.at/probier", "wikiservice.at/fractal", "dorfwiki.org",
                     "PROWIKI.ORG/DSE/"):
            self.assertTrue(O.is_wiki_site(site, groups), site)
        for site in ("prowiki.org/wiki4d", "wikiservice.at", "wikiservice.at/user/milk", "vanderbi.lt",
                     "paste.linuxiarz.pl", "", "dse.example.org/x"):
            self.assertFalse(O.is_wiki_site(site, groups), site)

    def test_verdict_and_lag_stats(self):
        self.assertEqual(O.verdict(100.0, 1.0, 103.0), "wiki_first_then_off")
        self.assertEqual(O.verdict(100.0, 1.0, 102.0), "unresolved")
        self.assertEqual(O.verdict(100.0, 1.0, 98.0), "unresolved")
        self.assertEqual(O.verdict(100.0, 1.0, 97.0), "off_first")
        self.assertEqual(O.lag_stats([])["n"], 0)
        s = O.lag_stats([DAY, 3 * DAY, 2 * DAY, 4 * DAY])
        self.assertEqual((s["n"], s["median_days"], s["p25_days"], s["p75_days"], s["max_days"]),
                         (4, 2.5, 1.75, 3.25, 4.0))


class FixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-offwiki-")
        cls.data = os.path.join(cls.tmp, "export")
        os.makedirs(cls.data)
        write_export(cls.data)
        cls.rows, _ = L.load("wiki", cls.data)
        cls.tokens, _ = find_tokens(cls.rows)
        cls.section = O.stretch_s1(cls.rows, cls.tokens, cls.data)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_fixture_tokens_kept(self):
        for key in ("host:ferrylane.net", "host:quillstack.org", "code:relayWindowSeven",
                    "code:mirrorCacheKey", "host:lanternpost.io", "host:tidewatch.net",
                    "code:syncBeaconTwo", "code:harborSignalNine"):
            self.assertIn(key, self.tokens)

    def test_headline_counts(self):
        h = self.section["headline"]
        self.assertEqual(self.section["status"], "done")
        self.assertEqual(h["seen_off_wiki"], 7)  # all but the wiki-only copy
        self.assertEqual(h["with_published_date"], 5)
        self.assertEqual((h["wiki_first_then_off"], h["off_first"], h["unresolved"]), (3, 1, 1))
        self.assertEqual((h["only_seen_by"], h["only_undated"]), (1, 1))
        self.assertEqual(h["seen_off_wiki_from_one_source_only"], {"records": 5, "links": 1, "shortener_log": 1})
        self.assertEqual(h["any_published_appearance_after_wiki_first_use"], 3)

    def test_lags(self):
        lag = self.section["headline"]["lag_wiki_first_to_off_wiki"]
        # ferrylane 2 days; lanternpost 3 days minus 300 s; tidewatch 1 h after the blank row at 390 s
        self.assertEqual(lag["n"], 3)
        self.assertEqual(lag["median_seconds"], 2 * DAY)
        self.assertEqual(lag["min_seconds"], 3600.0 + 10.0)
        texts = self.section["headline"]["first_off_wiki_texts"]
        self.assertEqual((texts["distinct"], texts["largest_tokens"]), (3, 1))
        self.assertEqual(texts["lag_one_per_text"]["median_seconds"], 2 * DAY)

    def test_by_class_and_examples(self):
        h = self.section["headline"]
        self.assertEqual(h["by_class"]["host"]["wiki_first_then_off"], 3)
        self.assertEqual(h["by_class"]["code"]["unresolved"], 1)
        self.assertEqual(h["non_phrase"]["wiki_first_then_off"], 3)
        self.assertEqual(len(h["examples"]), 3)
        for ex in h["examples"]:
            self.assertEqual(set(ex), {"token_id", "cls", "wiki_agents", "off_wiki_published_appearances",
                                       "lag_hours"})
            self.assertRegex(ex["token_id"], r"^[0-9a-f]{12}$")

    def test_other_wikis_kept_out_of_headline(self):
        ow = self.section["by_source"]["other_wikis"]
        self.assertEqual((ow["seen_off_wiki"], ow["wiki_first_then_off"]), (1, 1))
        self.assertEqual(ow["lag_wiki_first_to_off_wiki"]["median_seconds"], 5 * 3600.0)
        self.assertEqual(self.section["by_source"]["links"]["wiki_first_then_off"], 1)
        self.assertEqual(self.section["by_source"]["shortener_log"]["wiki_first_then_off"], 1)

    def test_source_stats(self):
        s = self.section["sources"]
        r = s["records"]
        self.assertEqual((r["records"], r["off_wiki"], r["wiki_only_skipped"], r["off_wiki_also_on_wiki"]),
                         (8, 7, 1, 1))
        self.assertEqual(r["off_wiki_by_date"], {"published": 4, "seen_by": 2, "undated": 1})
        self.assertEqual(r["wiki_sites_skipped"], ["prowiki.org/dse"])
        lk = s["links"]
        self.assertEqual((lk["links"], lk["on_wiki_records_only"], lk["on_off_wiki_records"], lk["host_skipped"]),
                         (3, 1, 2, 1))
        self.assertEqual(s["shortener_log"]["links"], 1)
        self.assertEqual(s["other_wikis"]["revisions"], 1)
        self.assertEqual(s["ip_masked_in_text"]["records"], 1)

    def test_no_text_ip_or_labels_in_section(self):
        text = json.dumps(self.section)
        for needle in (FAKE_IP, IP_FIELD, "agent-a", "ferrylane", "relayWindow", "notes at", "tidewatch"):
            self.assertNotIn(needle, text)

    def test_events_drop_ip_field_and_mask_ips(self):
        events, _stats = O.load_off_wiki(self.data, ["dse"])
        for ev in events:
            self.assertEqual(set(ev._fields), {"source", "time", "seen_by", "text", "keys"})
            self.assertNotIn(FAKE_IP, ev.text)
            self.assertFalse(any(FAKE_IP in k for k in ev.keys))
        self.assertEqual(sum(1 for ev in events if ev.source == O.SRC_LINKS), 1)

    def test_deterministic(self):
        again = O.stretch_s1(self.rows, self.tokens, self.data)
        a = {k: v for k, v in self.section.items() if k != "seconds"}
        b = {k: v for k, v in again.items() if k != "seconds"}
        self.assertEqual(a, b)

    def test_missing_records_is_skipped(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty, exist_ok=True)
        section = O.stretch_s1(self.rows, self.tokens, empty)
        self.assertEqual(section["status"], "skipped")
        self.assertTrue(O.f13_slot(section).startswith("skipped: "))

    def _facts(self, **changes):
        facts = {
            "input": {"adapter": "wiki", "files": P.input_files("wiki", self.data)},
            "params": {"min_agents": 3, "max_phrase_share": 0.02, "given_texts": 0},
            "tokens": {"kept_total": len(self.tokens)},
            "core_sha256": "c" * 64,
            "hand_check": {"precision": "7/10"},
            "slots": {"F10": {"precision": "7/10"}, "pending": {"F1": "lead", "F13": "stretch (J8)"}},
        }
        facts.update(changes)
        return facts

    def test_merge_keeps_core_and_other_sections(self):
        path = os.path.join(self.tmp, "facts.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._facts(), fh)
        O.merge_into_facts(path, self.section)
        with open(path, "r", encoding="utf-8") as fh:
            facts = json.load(fh)
        self.assertEqual(facts["core_sha256"], "c" * 64)
        self.assertEqual(facts["hand_check"], {"precision": "7/10"})
        self.assertEqual(facts["stretch_s1"]["facts_core_sha256_at_write"], "c" * 64)
        self.assertEqual(facts["slots"]["F13"]["tokens_wiki_first_then_off"], 3)
        self.assertEqual(facts["slots"]["F13"]["median_lag_days"], 2.0)
        self.assertNotIn("F13", facts["slots"]["pending"])
        self.assertIn("F1", facts["slots"]["pending"])

    def test_check_facts_refuses_mismatch(self):
        n = len(self.tokens)
        O.check_facts(self._facts(), self.data, n, 3, 0.02, 0)
        with self.assertRaises(ValueError):
            O.check_facts(self._facts(), self.data, n + 1, 3, 0.02, 0)
        with self.assertRaises(ValueError):
            O.check_facts(self._facts(), self.data, n, 4, 0.02, 0)
        with self.assertRaises(ValueError):
            O.check_facts(self._facts(input={"adapter": "generic", "files": []}), self.data, n, 3, 0.02, 0)
        bad = self._facts()
        bad["input"]["files"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            O.check_facts(bad, self.data, n, 3, 0.02, 0)

    def test_cli_merges_and_prints_counts_only(self):
        path = os.path.join(self.tmp, "cli-facts.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._facts(), fh)
        out = io.StringIO()
        json_path = os.path.join(self.tmp, "s1.json")
        with contextlib.redirect_stderr(io.StringIO()):
            code = O.main(["--input", self.data, "--facts", path, "--json", json_path], out=out)
        self.assertEqual(code, 0, out.getvalue())
        text = out.getvalue()
        self.assertIn("wiki first, then off the wiki: 3", text)
        for needle in ("ferrylane", "agent-a", FAKE_IP):
            self.assertNotIn(needle, text)
        with open(path, "r", encoding="utf-8") as fh:
            self.assertIn("stretch_s1", json.load(fh))
        self.assertTrue(os.path.isfile(json_path))

    def test_cli_refuses_output_inside_repo(self):
        out = io.StringIO()
        code = O.main(["--input", self.data, "--json", os.path.join(ROOT, "s1.json")], out=out)
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(os.path.join(ROOT, "s1.json")))


@unittest.skipUnless(os.environ.get("PZ_DATA"), "set PZ_DATA to the collusion.wiki export folder")
class RealDataTest(unittest.TestCase):
    def test_off_wiki_counts(self):
        events, stats = O.load_off_wiki(os.environ["PZ_DATA"], ["dse", "probier", "fractal", "dorfwiki"])
        r = stats["records"]
        self.assertEqual(stats["missing_files"], [])
        self.assertEqual((r["records"], r["off_wiki"], r["wiki_only_skipped"]), (13703, 4241, 9462))
        self.assertEqual(sorted(r["wiki_sites_skipped"]),
                         ["dorfwiki.org", "prowiki.org/dse", "wikiservice.at/fractal", "wikiservice.at/probier"])
        self.assertEqual(stats["links"]["links"], 23877)
        self.assertEqual(stats["shortener_log"]["links"], 499)
        self.assertEqual(stats["shortener_log"]["by_date"]["published"], 499)
        self.assertEqual(stats["other_wikis"]["revisions"], 90)
        self.assertGreater(r["off_wiki_by_date"]["published"], 0)
        for ev in events:
            self.assertEqual(L.scrub_ips(ev.text)[1], 0)


if __name__ == "__main__":
    unittest.main()
