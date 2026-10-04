"""Tests for pointing patient-zero at any log: ``--map`` and CSV/TSV input.

unittest, standard library only. Every name, host and line here is made up.

* PlantedFormats: the planted chat from tests/test_planted.py is written
  three more ways: a CSV with other column names, a gzipped TSV with POSIX
  seconds, and nested Slack-shaped JSON Lines (Events API envelopes). Each
  must load to the same rows as the JSON Lines original, keep all three
  planted cascades, and give the same facts.json in every core section
  except ``input``.
* BadMap: a map key the log does not have exits 2 with one plain sentence
  that names the missing column, and writes nothing.
* CsvDetails, ParseMap: the smaller rules.
* Examples: the two README examples run, and the CSV one matches the JSON
  Lines example in every core section except ``input``.
"""

import csv
import gzip
import importlib.util
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
from patient_zero import checks as K  # noqa: E402
from patient_zero import load as L  # noqa: E402
from patient_zero import pipeline as P  # noqa: E402
from patient_zero import tokens as T  # noqa: E402
from tests.test_planted import BASE, CHAT, PLANTED, write_chat  # noqa: E402

FIELD = "ip" + "16"  # the half-IP field name, assembled at run time
HALF = ".".join(["203", "113"])
EXAMPLES = os.path.join(ROOT, "examples")

CSV_MAP = ("id=msg_id,agent=author,time=posted_at,place=room,text=message,group=team,"
           "uncertainty_seconds=clock_error_s")
SLACK_MAP = ("id=event.client_msg_id,agent=event.user_profile.name,time=event.ts,place=event.channel,"
             "text=event.text,uncertainty_seconds=event.metadata.event_payload.clock_error_s,"
             "group=event.metadata.event_payload.team")
CSV_EXAMPLE_MAP = "agent=user,time=ts,place=channel,text=body"
SLACK_EXAMPLE_MAP = "id=event.client_msg_id,agent=event.user,time=event.ts,place=event.channel,text=event.text"


def run_cli(args):
    buf = io.StringIO()
    code = P.main(args, out=buf)
    return code, buf.getvalue()


def ct_iso(seconds_after_base):
    """The same instant as test_planted.iso, written in US Central time with a space."""
    import datetime
    zone = datetime.timezone(datetime.timedelta(hours=-6))
    stamp = datetime.datetime.fromtimestamp(BASE + seconds_after_base, tz=zone)
    return stamp.strftime("%Y-%m-%d %H:%M:%S%z")[:-2] + ":00"


def write_csv(folder):
    """The planted chat as CSV: other column names, extra columns, Central-time stamps."""
    path = os.path.join(folder, "chat.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["msg_id", "posted_at", "room", "author", "message", "team", "clock_error_s",
                    "secret_note", FIELD])
        for n, (rid, agent, t, room, text) in enumerate(CHAT):
            w.writerow([rid, ct_iso(t), room, agent, text, "made-up", "1", "DROPME-%d" % n, HALF])
    return path


def write_tsv_gz(folder):
    """The planted chat as gzipped TSV with POSIX seconds and the columns in another order."""
    path = os.path.join(folder, "chat.tsv.gz")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["message", "author", "room", "posted_at", "msg_id", "team", "clock_error_s"])
        for rid, agent, t, room, text in CHAT:
            w.writerow([text, agent, room, "%d" % (BASE + t), rid, "made-up", "1"])
    return path


def write_slack(folder, name="chat-slack.ndjson"):
    """The planted chat as Slack Events API envelopes, with the agent two objects deep."""
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rid, agent, t, room, text in CHAT:
            fh.write(json.dumps({
                "type": "event_callback", "team_id": "T0DEMO", FIELD: HALF,
                "event": {
                    "type": "message", "client_msg_id": rid, "channel": room, "text": text,
                    "ts": "%d.000000" % (BASE + t),
                    "user_profile": {"name": agent, "real_name": "DROPME"},
                    "metadata": {"event_type": "agent_log",
                                 "event_payload": {"clock_error_s": 1, "team": "made-up"}},
                },
            }) + "\n")
    return path


# ---------------------------------------------------------------- the planted chat, four ways


class PlantedFormats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-map-")
        cls.inputs = {
            "ndjson": (write_chat(cls.tmp), None),
            "csv": (write_csv(cls.tmp), CSV_MAP),
            "tsv.gz": (write_tsv_gz(cls.tmp), CSV_MAP),
            "slack": (write_slack(cls.tmp), SLACK_MAP),
        }
        cls.rows, cls.facts, cls.files, cls.codes = {}, {}, {}, {}
        for name, (path, spec) in cls.inputs.items():
            cls.rows[name] = L.load("generic", path, L.parse_map(spec))[0]
            out = os.path.join(cls.tmp, "out-" + name)
            args = ["--adapter", "generic", "--input", path, "--out", out]
            if spec:
                args += ["--map", spec]
            cls.codes[name] = run_cli(args)[0]
            with open(os.path.join(out, "facts.json"), encoding="utf-8") as fh:
                cls.facts[name] = json.load(fh)
            cls.files[name] = {}
            for fname in ("trees.jsonl", "top_trees.json", "report.html"):
                with open(os.path.join(out, fname), "rb") as fh:
                    cls.files[name][fname] = fh.read()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_runs_exit_0(self):
        self.assertEqual(self.codes, {name: 0 for name in self.inputs})

    def test_rows_identical(self):
        self.assertEqual(len(self.rows["ndjson"]), len(CHAT))
        for name in ("csv", "tsv.gz", "slack"):
            with self.subTest(fmt=name):
                self.assertEqual(self.rows[name], self.rows["ndjson"])

    def test_planted_cascades_found_in_every_format(self):
        for name, rows in self.rows.items():
            tokens, _ = T.find_tokens(rows)
            trees, _ = C.build_trees(rows, tokens)
            for key, want in PLANTED.items():
                with self.subTest(fmt=name, token=key):
                    self.assertEqual(C.parents_of(trees[key]), want["parents"])
                    self.assertEqual(trees[key].first_user, want["first_user"])
                    self.assertEqual(trees[key].effective_seed, want["effective_seed"])

    def test_facts_identical_except_input(self):
        base = self.facts["ndjson"]
        core = [k for k in P.CORE_KEYS if k != "input"]
        for name in ("csv", "tsv.gz", "slack"):
            with self.subTest(fmt=name):
                diff = [k for k in core if self.facts[name][k] != base[k]]
                self.assertEqual(diff, [])
                self.assertNotEqual(self.facts[name]["core_sha256"], base["core_sha256"])  # input differs

    def test_input_records_format_and_map(self):
        self.assertEqual(sorted(self.facts["ndjson"]["input"]), ["adapter", "files"])  # as before --map
        csv_in = self.facts["csv"]["input"]
        self.assertEqual(csv_in["format"], "csv")
        self.assertEqual(csv_in["map"]["agent"], "author")
        self.assertEqual(self.facts["tsv.gz"]["input"]["format"], "tsv")
        self.assertNotIn("format", self.facts["slack"]["input"])
        self.assertEqual(self.facts["slack"]["input"]["map"]["agent"], "event.user_profile.name")

    def test_output_files_identical(self):
        for name in ("csv", "tsv.gz", "slack"):
            for fname in ("trees.jsonl", "top_trees.json"):
                with self.subTest(fmt=name, file=fname):
                    self.assertEqual(self.files[name][fname], self.files["ndjson"][fname])

    def test_report_names_the_map(self):
        self.assertIn(b"--map agent=author", self.files["csv"]["report.html"])
        self.assertIn(b"read as csv", self.files["csv"]["report.html"])
        self.assertNotIn(b"--map", self.files["ndjson"]["report.html"])

    def test_unmapped_columns_never_load(self):
        for name in ("csv", "slack"):
            blob = repr(self.rows[name])
            self.assertNotIn("DROPME", blob)
            self.assertNotIn(HALF, blob)
            self.assertNotIn("T0DEMO", blob)


# ---------------------------------------------------------------- bad maps


class BadMap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pz-badmap-")
        self.csv = write_csv(self.tmp)
        self.slack = write_slack(self.tmp)
        self.out = os.path.join(self.tmp, "out")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fails(self, path, spec, *needles, adapter="generic"):
        args = ["--adapter", adapter, "--input", path, "--out", self.out]
        if spec is not None:
            args += ["--map", spec]
        code, text = run_cli(args)
        self.assertEqual(code, 2, text)
        self.assertTrue(text.startswith("patient-zero: ERROR: "), text)
        self.assertEqual(text.count("\n"), 1, "one line, no traceback")
        self.assertNotIn("Traceback", text)
        for needle in needles:
            self.assertIn(needle, text)
        self.assertFalse(os.path.exists(self.out), "nothing written")
        return text

    def test_csv_column_missing_exits_2_naming_it(self):
        text = self._fails(self.csv, CSV_MAP.replace("agent=author", "agent=usr"),
                           "chat.csv has no column 'usr' (--map agent=usr).", "Its columns are: msg_id, ")
        self.assertIn("author", text)  # the columns it does have, to pick from

    def test_json_path_missing_exits_2_naming_it(self):
        self._fails(self.slack, SLACK_MAP.replace("user_profile.name", "user_profile.nme"),
                    "chat-slack.ndjson line 1 has no column 'event.user_profile.nme' "
                    "(--map agent=event.user_profile.nme).")

    def test_optional_column_missing_exits_2(self):
        self._fails(self.csv, CSV_MAP.replace("group=team", "group=teem"), "has no column 'teem'")
        self._fails(self.slack, SLACK_MAP.replace("payload.team", "payload.teem"),
                    "no line has column 'event.metadata.event_payload.teem'")

    def test_unmapped_required_column_says_how_to_map(self):
        self._fails(self.csv, None, "chat.csv has no column 'agent'", "--map agent=COLUMN")

    def test_bad_map_text(self):
        self._fails(self.csv, "who=author", "'who' is not a field patient-zero reads", "agent, time")
        self._fails(self.csv, "agent", "should look like field=column")
        self._fails(self.csv, "agent=author,agent=room", "names agent twice")

    def test_map_refused_for_wiki(self):
        self._fails(self.tmp, "agent=label", "works only with --adapter generic", adapter="wiki")

    def test_agent_pointing_at_an_object(self):
        self._fails(self.slack, SLACK_MAP.replace("user_profile.name", "user_profile"),
                    "agent must be a string or a number", "author.name")

    def test_load_and_checks_take_the_map(self):
        buf = io.StringIO()
        self.assertEqual(L.main(["--adapter", "generic", "--input", self.csv, "--map", CSV_MAP], out=buf), 0)
        self.assertEqual(json.loads(buf.getvalue())["agents"], 13)
        buf = io.StringIO()
        self.assertEqual(L.main(["--adapter", "generic", "--input", self.csv, "--map", "agent=usr"], out=buf), 2)
        self.assertIn("no column 'usr'", buf.getvalue())
        buf = io.StringIO()
        code = K.main(["--adapter", "generic", "--input", self.slack, "--map", SLACK_MAP, "--shuffles", "3"],
                      out=buf)
        self.assertEqual(code, 0)
        self.assertGreaterEqual(json.loads(buf.getvalue())["observed"]["default_min_lag_0s"]["links_named"], 8)

    def test_peek_takes_the_map(self):
        spec = importlib.util.spec_from_file_location("peek_for_map", os.path.join(ROOT, "tools", "peek.py"))
        peek = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(peek)
        buf = io.StringIO()
        code = peek.main(["--adapter", "generic", "--input", self.csv, "--map", CSV_MAP, "--n", "3",
                          "--out", os.path.join(self.tmp, "peek", "peek.txt")], out=buf)
        self.assertEqual(code, 0, buf.getvalue())
        buf = io.StringIO()
        code = peek.main(["--adapter", "generic", "--input", self.csv, "--map", "agent=usr",
                          "--out", os.path.join(self.tmp, "peek", "peek2.txt")], out=buf)
        self.assertEqual(code, 2)
        self.assertIn("no column 'usr'", buf.getvalue())
        # snippets mask human handles: peek must find them through the map and in CSV too
        want = peek.human_handles("generic", os.path.join(EXAMPLES, "fake_log.ndjson"))
        self.assertEqual(len(want), 1)
        self.assertEqual(peek.human_handles("generic", os.path.join(EXAMPLES, "fake_log.csv"),
                                            L.parse_map(CSV_EXAMPLE_MAP)), want)


# ---------------------------------------------------------------- CSV details


class CsvDetails(unittest.TestCase):
    def _load(self, text, spec=None, suffix=".csv"):
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="pz-csv-")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
            return L.load("generic", path, L.parse_map(spec))
        finally:
            os.remove(path)

    def test_bom_quotes_newlines_and_blank_lines(self):
        rows, stats = self._load(
            '\ufeffagent,time,place,text\n'
            'A,100,room,"two\nlines, one cell"\n'
            '\n'
            ',,,\n'
            'B,1970-01-01T00:03:20Z,room,"say ""hi"""\n')
        self.assertEqual([r.agent for r in rows], ["A", "B"])
        self.assertEqual(rows[0].text, "two\nlines, one cell")
        self.assertEqual(rows[1].text, 'say "hi"')
        self.assertEqual(rows[1].time, 200.0)
        self.assertEqual(rows[0].row_id, "line-3")  # the line where the record ends
        self.assertEqual(stats["rows"], 2)

    def test_empty_optional_cells_are_absent(self):
        rows, _ = self._load("agent,time,place,text,group,family,uncertainty_seconds,id,time_grade\n"
                             "A,1,p,t,,,,,\n"
                             "B,2,p,t,g,f,2.5,x9,reqlog\n")
        self.assertEqual((rows[0].group, rows[0].family, rows[0].uncertainty, rows[0].row_id,
                          rows[0].time_grade), (None, None, 0.0, "line-2", None))
        self.assertEqual((rows[1].group, rows[1].family, rows[1].uncertainty, rows[1].row_id,
                          rows[1].time_grade), ("g", "f", 2.5, "x9", "reqlog"))

    def test_human_words(self):
        rows, stats = self._load("agent,time,place,text,human\n"
                                 "A,1,p,t,TRUE\nB,2,p,t,yes\nC,3,p,t,1\nD,4,p,t,no\nE,5,p,t,\nF,6,p,t,0\n")
        self.assertEqual([r.role for r in rows], [L.ROLE_HUMAN] * 3 + [L.ROLE_AGENT] * 3)
        self.assertEqual(stats["human_rows"], 3)
        self.assertNotIn("'A'", repr(rows))
        with self.assertRaises(ValueError) as ctx:
            self._load("agent,time,place,text,human\nA,1,p,t,maybe\n")
        self.assertIn("line 2: human must be", str(ctx.exception))

    def test_bad_cells_name_the_line(self):
        cases = [
            ("agent,time,place,text,uncertainty_seconds\nA,1,p,t,abc\n", "line 2: uncertainty_seconds"),
            ("agent,time,place,text,uncertainty_seconds\nA,1,p,t,nan\n", "line 2: uncertainty_seconds"),
            ("agent,time,place,text\nA,1,p,t\nB,soon,p,t\n", "line 3: bad time"),
            ("agent,time,place,text\nA,1,p\n", "line 2: missing field(s) text"),
            ("id,agent,time,place,text\nx,A,1,p,t\nx,B,2,p,t\n", "line 3: duplicate id"),
        ]
        for text, needle in cases:
            with self.subTest(needle=needle):
                with self.assertRaises(ValueError) as ctx:
                    self._load(text)
                self.assertIn(needle, str(ctx.exception))

    def test_duplicate_header_only_matters_when_read(self):
        rows, _ = self._load("agent,time,place,text,note,note\nA,1,p,t,x,y\n")
        self.assertEqual(len(rows), 1)
        with self.assertRaises(ValueError) as ctx:
            self._load("who,who,time,place,text\nA,B,1,p,t\n", "agent=who")
        self.assertIn("column 'who' appears 2 times", str(ctx.exception))

    def test_column_names_are_escaped_in_errors(self):
        """A hostile header cannot put terminal control codes into the error line."""
        with self.assertRaises(ValueError) as ctx:
            self._load("\x1b[2Jboom,time,place,text\nA,1,p,t\n", "agent=user")
        self.assertNotIn("\x1b", str(ctx.exception))
        self.assertIn("\\u001b[2Jboom", str(ctx.exception))
        wide = ",".join("c%d" % i for i in range(30))
        with self.assertRaises(ValueError) as ctx:
            self._load(wide + "\n" + wide + "\n", "agent=user")
        self.assertIn("c19 and 10 more.", str(ctx.exception))

    def test_empty_file_and_header_only(self):
        self.assertEqual(self._load("")[0], [])
        self.assertEqual(self._load("agent,time,place,text\n")[0], [])

    def test_tsv_and_field_limit(self):
        long_text = "word " * 40000  # 200,000 characters: over the csv module's default cell limit
        rows, _ = self._load("agent\ttime\tplace\ttext\nA\t1\tp\t%s\n" % long_text, suffix=".tsv")
        self.assertEqual(len(rows[0].text), len(long_text))
        self.assertEqual(csv.field_size_limit(), 131072)  # restored afterwards

    def test_ip_scrub_still_applies(self):
        v4 = ".".join(["198", "51", "100", "7"])
        rows, stats = self._load("user,ts,channel,body\n%s,1,room-%s,mirror at %s\n" % (v4, v4, v4),
                                 "agent=user,time=ts,place=channel,text=body")
        self.assertEqual(rows[0].role, L.ROLE_BLANK)
        self.assertEqual(rows[0].text, "mirror at [ip]")
        self.assertEqual(rows[0].place, "room-[ip]")
        self.assertEqual(stats["ip_shaped_labels_blanked"], 1)


# ---------------------------------------------------------------- small rules


class ParseMap(unittest.TestCase):
    def test_parses_and_trims(self):
        self.assertEqual(L.parse_map(" agent = user , time=ts,place=channel,text=author.name,"),
                         {"agent": "user", "time": "ts", "place": "channel", "text": "author.name"})
        self.assertEqual(L.parse_map(None), {})
        self.assertEqual(L.parse_map("  "), {})

    def test_bad_specs(self):
        for spec, needle in (("agent", "field=column"), ("agent=", "field=column"), ("=user", "field=column"),
                             ("who=user", "not a field"), ("agent=a,agent=b", "twice"),
                             ("agent=a..b", "empty part")):
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError) as ctx:
                    L.parse_map(spec)
                self.assertIn(needle, str(ctx.exception))

    def test_resolve(self):
        self.assertEqual(L._resolve({"a.b": 1, "a": {"b": 2}}, "a.b"), 1)  # a dotted key as-is first
        self.assertEqual(L._resolve({"a": {"b": {"c": 3}}}, "a.b.c"), 3)
        self.assertIs(L._resolve({"a": {"b": 2}}, "a.c"), L._MISSING)
        self.assertIs(L._resolve({"a": "text"}, "a.b"), L._MISSING)
        self.assertIsNone(L._resolve({"a": {"b": None}}, "a.b"))

    def test_generic_format(self):
        for name, want in (("x.csv", "csv"), ("X.CSV.GZ", "csv"), ("x.tsv", "tsv"), ("x.ndjson", "jsonl"),
                           ("x.json", "jsonl"), ("x.jsonl.gz", "jsonl"), ("log", "jsonl")):
            self.assertEqual(L.generic_format(name), want, name)

    def test_posix_seconds_as_text(self):
        self.assertEqual(L.parse_time("1768039200"), 1768039200.0)
        self.assertEqual(L.parse_time(" 1768039200.000100 "), 1768039200.0001)
        self.assertEqual(L.parse_time("2026-01-10T10:00:00Z"), 1768039200.0)
        self.assertEqual(L.parse_time("2026-01-10 04:00:00-06:00"), 1768039200.0)

    def test_api_refuses_unknown_field(self):
        with self.assertRaises(ValueError):
            L.load_generic(os.path.join(EXAMPLES, "fake_log.ndjson"), {"who": "agent"})


# ---------------------------------------------------------------- the README examples


class Examples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pz-examples-")
        cls.facts, cls.codes = {}, {}
        for name, path, spec in (("ndjson", "fake_log.ndjson", None), ("csv", "fake_log.csv", CSV_EXAMPLE_MAP),
                                 ("slack", "fake_slack.ndjson", SLACK_EXAMPLE_MAP)):
            out = os.path.join(cls.tmp, name)
            args = ["--adapter", "generic", "--input", os.path.join(EXAMPLES, path), "--out", out]
            cls.codes[name] = run_cli(args + (["--map", spec] if spec else []))[0]
            with open(os.path.join(out, "facts.json"), encoding="utf-8") as fh:
                cls.facts[name] = json.load(fh)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_examples_run(self):
        self.assertEqual(self.codes, {"ndjson": 0, "csv": 0, "slack": 0})

    def test_csv_example_matches_ndjson_example(self):
        base, other = self.facts["ndjson"], self.facts["csv"]
        self.assertEqual([k for k in P.CORE_KEYS if k != "input" and other[k] != base[k]], [])

    def test_slack_example_has_the_same_spread(self):
        """Same talk minus the human speaker's line: same trees and links, Slack ids for names."""
        rows, _ = L.load("generic", os.path.join(EXAMPLES, "fake_log.ndjson"))
        slack, _ = L.load("generic", os.path.join(EXAMPLES, "fake_slack.ndjson"), L.parse_map(SLACK_EXAMPLE_MAP))
        self.assertEqual(len(slack), len(rows) - 1)
        names = {}
        by_id = {r.row_id: r for r in rows}
        for r in slack:
            names[r.agent] = by_id[r.row_id].agent
        trees_a = C.build_trees(rows, T.find_tokens(rows)[0])[0]
        trees_b = C.build_trees(slack, T.find_tokens(slack)[0])[0]
        self.assertEqual(sorted(trees_a), sorted(trees_b))
        for key in trees_a:
            renamed = {names[a]: (names[p] if p else None) for a, p in C.parents_of(trees_b[key]).items()}
            self.assertEqual(renamed, C.parents_of(trees_a[key]), key)
        for k in ("cascade", "trees", "nullmodel"):
            self.assertEqual(self.facts["slack"][k], self.facts["ndjson"][k], k)


if __name__ == "__main__":
    unittest.main()
