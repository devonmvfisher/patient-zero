"""Tests for patient_zero/tokens.py (unittest, standard library only).

All fixture text is made up here. IP-shaped strings are assembled at run
time, so this file passes the leak check itself.

The real-data check runs only when PZ_DATA names the collusion.wiki export
folder; otherwise it is skipped.
"""

import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from patient_zero import load as L  # noqa: E402
from patient_zero import tokens as T  # noqa: E402

EXAMPLE = os.path.join(ROOT, "examples", "fake_log.ndjson")
V4 = ".".join(["198", "51", "100", "7"])


def keys(line):
    return {k for k, _c, _k, _v in T.line_tokens(line)[0]}


def make_row(i, agent, t, text, unc=0.0, role=None, place="p/one"):
    if role is None:
        role = L.ROLE_AGENT if agent else L.ROLE_BLANK
    return L.Row(
        row_id="r%03d" % i,
        agent=agent if role == L.ROLE_AGENT else None,
        role=role,
        time=float(t),
        uncertainty=float(unc),
        time_grade=None,
        place=place,
        family=None,
        group=None,
        text=text,
    )


def rows_of(specs):
    """specs: (agent or None, time, text[, unc[, role]]) in time order."""
    return [make_row(i, *spec) for i, spec in enumerate(specs)]


class Hosts(unittest.TestCase):
    def test_url_host_is_normalised(self):
        k = keys("see https://user@WWW.Ferry-Mirror.example.net:8080/a/b.txt now")
        self.assertIn("host:ferry-mirror.example.net", k)
        self.assertIn("file:b.txt", k)
        self.assertFalse(any(x.startswith("host:www.") for x in k))

    def test_bare_domain_found_email_skipped(self):
        self.assertIn("host:quill-cache.re", keys("mirror at quill-cache.re works"))
        self.assertFalse(any(x.startswith("host:") for x in keys("write to someone@inkpot.io")))

    def test_common_host_is_stoplisted(self):
        found, stopped = T.line_tokens("ref https://de.wikipedia.org/wiki/Foo here")
        self.assertFalse(any(k.startswith("host:") for k, *_ in found))
        self.assertEqual(stopped["host"], 1)

    def test_example_placeholder_only_exact(self):
        self.assertFalse(any(x.startswith("host:") for x in keys("visit example.org today")))
        self.assertIn("host:ferry.example.org", keys("visit ferry.example.org today"))

    def test_percent_escapes_decoded(self):
        found, _ = T.row_tokens("q=https%3A%2F%2Fwww.lantern-depot.io%2Fdata%252Fcounty-list.json")
        self.assertIn("host:lantern-depot.io", found)
        self.assertIn("file:county-list.json", found)
        self.assertFalse(any(k.startswith(("host:2f", "file:2f", "file:252f")) for k in found))

    def test_decoding_remasks_ips(self):
        encoded = "".join("%%%02X" % ord(ch) for ch in V4)
        self.assertEqual(T.decode_line("x " + encoded + " y"), "x [ip] y")


class FilesCommandsCode(unittest.TestCase):
    def test_files(self):
        k = keys("saved notes as Relay_Notes.TXT and ran build.sh, see robots.txt")
        self.assertIn("file:relay_notes.txt", k)
        self.assertIn("file:build.sh", k)
        self.assertNotIn("file:robots.txt", k)
        self.assertNotIn("code:Relay_Notes", k)

    def test_commands(self):
        self.assertIn("command:git clone", keys("$ git clone some/repo"))
        self.assertIn("command:pip install", keys("sudo pip install thing"))
        self.assertIn("command:curl", keys("curl -s thing"))
        self.assertIn("command:npm run", keys("then type `npm run build` to finish"))
        self.assertIn("command:git", keys("git is nice"))  # a stop word is no sub-command
        self.assertFalse(any(x.startswith("command:") for x in keys("Git is nice")))
        self.assertFalse(any(x.startswith("command:") for x in keys("git")))

    def test_code_words(self):
        k = keys("call fetch_page_list and loadMirrorTable then mirror.refresh(x) and `cachePath`")
        for want in ("code:fetch_page_list", "code:loadMirrorTable", "code:mirror.refresh", "code:cachePath"):
            self.assertIn(want, k)
        self.assertNotIn("code:iPhone", keys("my iPhone broke"))


class Phrases(unittest.TestCase):
    def test_content_words_needed(self):
        k = keys("the lantern quorum gathers and it is the end")
        self.assertIn("phrase:the lantern quorum", k)
        self.assertIn("phrase:lantern quorum gathers", k)
        self.assertNotIn("phrase:and it is", k)

    def test_never_across_urls(self):
        k = keys("copper kettle https://a-host.io/x brass bell")
        self.assertFalse(any(x.startswith("phrase:") for x in k))


class Conventions(unittest.TestCase):
    def test_prefixes(self):
        k = keys("nodes relay-node-7 and relay-node-12, team_alpha")
        self.assertIn("convention:prefix:relay-", k)
        self.assertIn("convention:prefix:team_", k)
        none = keys("send an e-mail about state-of-the-art work, see user:Someone")
        self.assertFalse(any(x.startswith("convention:prefix:") for x in none))

    def test_mailbox_names(self):
        k = keys("drop it in agentInbox or dead-drop-3")
        self.assertIn("convention:mailbox:agentinbox", k)
        self.assertIn("convention:mailbox:dead-drop-3", k)
        self.assertNotIn("code:agentInbox", k)

    def test_control_words(self):
        self.assertIn("convention:control:HOLD", keys("HOLD: waiting on review"))
        self.assertIn("convention:control:VETO", keys("my vote is [VETO] for now"))
        self.assertIn("convention:control:STOP", keys("we must STOP now"))
        self.assertIn("convention:control:BLUEHERON", keys("tag BLUEHERON here"))
        self.assertIn("convention:control:XYZ", keys("XYZ: go"))
        self.assertNotIn("convention:control:HTML", keys("the HTML: page"))
        self.assertNotIn("convention:control:ABC", keys("the ABC song"))
        self.assertNotIn("convention:control:UND", keys("UND dann"))


class Risky(unittest.TestCase):
    def test_types_only_in_fixed_order(self):
        self.assertEqual(T.risky_types("a note about a proxy and a captcha"), ("captcha", "proxy"))
        self.assertEqual(T.risky_types("plain words"), ())

    def test_attached_per_line(self):
        found, _ = T.row_tokens("lantern-depot.io is a relay\nquill-cache.re is fine")
        self.assertEqual(found["host:lantern-depot.io"][3], frozenset(["relay"]))
        self.assertEqual(found["host:quill-cache.re"][3], frozenset())


class ClockRule(unittest.TestCase):
    def test_pairs(self):
        self.assertFalse(T.in_clock_order(0, 1, 2, 1))
        self.assertTrue(T.in_clock_order(0, 1, 2.5, 1))
        self.assertFalse(T.clock_resolved([(0, 1), (1, 1), (2, 1)]))
        self.assertTrue(T.clock_resolved([(0, 1), (1, 1), (3, 1)]))


class FindTokens(unittest.TestCase):
    def test_three_agents_kept_two_dropped(self):
        rows = rows_of([
            ("Ash", 0, "mirror lantern-depot.io and quill-cache.re"),
            ("Birch", 100, "lantern-depot.io again, quill-cache.re too"),
            ("Cedar", 200, "lantern-depot.io works"),
            ("Ash", 300, "still lantern-depot.io"),
        ])
        toks, stats = T.find_tokens(rows)
        tok = toks["host:lantern-depot.io"]
        self.assertEqual(tok.agents, 3)
        self.assertEqual(tok.rows, (0, 1, 2, 3))
        self.assertEqual(tok.first_users, (("Ash", 0), ("Birch", 1), ("Cedar", 2)))
        self.assertNotIn("host:quill-cache.re", toks)
        self.assertEqual(stats["classes"]["host"]["kept"], 1)
        self.assertEqual(stats["classes"]["host"]["under_min_agents"], 1)

    def test_exposure_only_rows_never_count_as_users(self):
        rows = rows_of([
            ("Ash", 0, "see quill-cache.re"),
            (None, 50, "see quill-cache.re", 0.0, L.ROLE_BLANK),
            (None, 60, "see quill-cache.re", 0.0, L.ROLE_HUMAN),
            ("Birch", 100, "see quill-cache.re"),
        ])
        toks, _ = T.find_tokens(rows)
        self.assertNotIn("host:quill-cache.re", toks)
        toks, _ = T.find_tokens(rows, min_agents=2)
        tok = toks["host:quill-cache.re"]
        self.assertEqual(tok.agents, 2)
        self.assertEqual(tok.rows, (0, 1, 2, 3))
        self.assertEqual([a for a, _i in tok.first_users], ["Ash", "Birch"])

    def test_clock_rule_drops_unordered_first_uses(self):
        tight = rows_of([("Ash", 0, "use plinth-hub.io", 1.0), ("Birch", 1, "use plinth-hub.io", 1.0),
                         ("Cedar", 2, "use plinth-hub.io", 1.0)])
        toks, stats = T.find_tokens(tight)
        self.assertNotIn("host:plinth-hub.io", toks)
        self.assertEqual(stats["classes"]["host"]["clock_dropped"], 1)
        spaced = rows_of([("Ash", 0, "use plinth-hub.io", 1.0), ("Birch", 1, "use plinth-hub.io", 1.0),
                          ("Cedar", 3, "use plinth-hub.io", 1.0)])
        toks, _ = T.find_tokens(spaced)
        self.assertIn("host:plinth-hub.io", toks)

    def test_given_text_tokens_dropped(self):
        rows = rows_of([("Ash", 0, "see plinth-hub.io"), ("Birch", 10, "see plinth-hub.io"),
                        ("Cedar", 20, "see plinth-hub.io")])
        toks, stats = T.find_tokens(rows, given_texts=["Your goal: keep plinth-hub.io tidy."])
        self.assertNotIn("host:plinth-hub.io", toks)
        self.assertEqual(stats["classes"]["host"]["given_dropped"], 1)

    def test_common_phrase_dropped(self):
        rows = rows_of([("agent%02d" % i, i * 10, "copper kettle song") for i in range(30)])
        toks, stats = T.find_tokens(rows)
        self.assertNotIn("phrase:copper kettle song", toks)
        self.assertEqual(stats["classes"]["phrase"]["common_dropped"], 1)
        toks, _ = T.find_tokens(rows, max_phrase_share=1.0)
        self.assertIn("phrase:copper kettle song", toks)

    def test_deterministic_and_counts_only(self):
        rows, _ = L.load("generic", EXAMPLE)
        a, sa = T.find_tokens(rows)
        b, sb = T.find_tokens(rows)
        self.assertEqual(a, b)
        sa.pop("seconds"), sb.pop("seconds")
        self.assertEqual(sa, sb)
        text = json.dumps(sa)
        for secret in ("ferry", "BLUEHERON", "Wren", "Otter", "reading"):
            self.assertNotIn(secret, text)

    def test_example_log(self):
        rows, _ = L.load("generic", EXAMPLE)
        toks, stats = T.find_tokens(rows)
        self.assertIn("host:ferry.example.org", toks)
        self.assertIn("convention:control:BLUEHERON", toks)
        self.assertEqual(toks["convention:control:BLUEHERON"].agents, 4)
        self.assertEqual(stats["convention_kinds_kept"]["control"], 1)
        self.assertNotIn("phrase:ferry example org", toks)


class CommandLine(unittest.TestCase):
    def test_summary_file_counts_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "tokens-summary.json")
            out = io.StringIO()
            code = T.main(["--adapter", "generic", "--input", EXAMPLE, "--summary", path], out=out)
            self.assertEqual(code, 0)
            with open(path, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
            self.assertIn("classes", summary)
            self.assertEqual(summary["kept_total"], sum(c["kept"] for c in summary["classes"].values()))
            for secret in ("ferry", "BLUEHERON", "Wren"):
                self.assertNotIn(secret, json.dumps(summary))
                self.assertNotIn(secret, out.getvalue())

    def test_bad_input_exit_2(self):
        out = io.StringIO()
        code = T.main(["--adapter", "generic", "--input", os.path.join(HERE, "no-such-file.ndjson")], out=out)
        self.assertEqual(code, 2)


@unittest.skipUnless(os.environ.get("PZ_DATA"), "PZ_DATA not set (real-data check)")
class RealData(unittest.TestCase):
    def test_every_main_class_has_kept_tokens(self):
        rows, _ = L.load("wiki", os.environ["PZ_DATA"])
        toks, stats = T.find_tokens(rows)
        self.assertEqual(stats["rows"], 14591)
        for cls in ("host", "file", "code", "phrase", "convention"):
            self.assertGreater(stats["classes"][cls]["kept"], 0, cls)
        self.assertEqual(stats["kept_total"], len(toks))
        for tok in toks.values():
            self.assertGreaterEqual(tok.agents, T.MIN_AGENTS)
            self.assertIn(tok.cls, T.CLASSES)
            self.assertEqual(len(tok.first_users), T.FIRST_USERS)


if __name__ == "__main__":
    unittest.main()
