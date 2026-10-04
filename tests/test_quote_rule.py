"""The quote rule: a log line that starts with ``> `` (or ``&gt; ``) is not its writer's own words.

Standard library unittest only; made-up chat lines, no data needed.

What is pinned:

1. which lines are quotes (the mark must be the first characters of the line);
2. a row that only quotes a token is not a use: its writer is no adopter, gets
   no link, and does not count towards ``min_agents``;
3. a row with a quoted line and an own line that holds the token is a use;
4. a quoted line is not an exposure either: a reader who met the token only in
   a quote is a parallel start (the limit the README states);
5. a line that starts with ``> `` is a quote even when a command follows; the
   ``>`` prompt still counts inside a backtick span;
6. a given text and an off-wiki text are read whole (``row_tokens(text,
   whole=True)``), so nothing they found before can go missing;
7. ``tools/peek.py`` never shows a quoted line as the line a token came from;
8. no shipped example log holds a quoted line, so the pinned example core
   cannot move through this rule.
"""

import csv
import importlib.util
import json
import os
import tempfile
import unittest

from patient_zero import cascade as C, load as L, pipeline as P, tokens as T

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("peek_for_quote_rule", os.path.join(REPO, "tools", "peek.py"))
pk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pk)

KEY = "code:lanternmoss"
USE = "the code is `lanternmoss` today"
QUOTE = "> the code is `lanternmoss` today"


def chat(*lines):
    """Rows from (agent, place, text) triples, 100 s apart, exact clocks."""
    return [L.Row("m%02d" % i, agent, L.ROLE_AGENT, i * 100.0, 0.0, None, place, None, None, text)
            for i, (agent, place, text) in enumerate(lines)]


def trace(rows):
    tokens, stats = T.find_tokens(rows)
    trees, tree_stats = C.build_trees(rows, tokens)
    return tokens, stats, trees, tree_stats


class QuoteMarks(unittest.TestCase):
    def test_a_quote_starts_with_the_mark_and_a_space(self):
        for line in ("> the code is here", "&gt; the code is here", "> > nested quote", "> "):
            with self.subTest(line=line):
                self.assertTrue(T.is_quote(line))

    def test_everything_else_is_the_writer_s_own_words(self):
        for line in (">the code is here", "&gt;the code is here", " > indented", ">> two marks", ">>> prompt",
                     "see > here", "\"the code is here\"", "she said: > the code", "-> arrow", ""):
            with self.subTest(line=line):
                self.assertFalse(T.is_quote(line))


class QuotedLinesAreNotUses(unittest.TestCase):
    def test_a_row_that_only_quotes_is_no_adopter_and_gets_no_link(self):
        rows = chat(("Ada", "room/a", USE), ("Bram", "room/a", "`lanternmoss` works for me"),
                    ("Cleo", "room/a", "`lanternmoss` here too"),
                    ("Dax", "room/a", QUOTE + "\nwhat does that mean?"))
        tokens, _stats, trees, tree_stats = trace(rows)
        self.assertEqual(tokens[KEY].rows, (0, 1, 2))  # Dax's row is not one of the token's rows
        self.assertEqual(tokens[KEY].agents, 3)
        tree = trees[KEY]
        self.assertEqual([a.agent for a in tree.adoptions], ["Ada", "Bram", "Cleo"])
        self.assertEqual(C.parents_of(tree), {"Ada": None, "Bram": "Ada", "Cleo": "Bram"})
        self.assertEqual(tree_stats["links_named"], 2)  # it was 3 when the quote counted as Dax's use

    def test_the_slack_form_of_the_mark_is_a_quote_too(self):
        rows = chat(("Ada", "room/a", USE), ("Bram", "room/a", "`lanternmoss` works for me"),
                    ("Cleo", "room/a", "`lanternmoss` here too"),
                    ("Dax", "room/a", "&gt; the code is `lanternmoss` today"))
        tokens, _stats, trees, _tree_stats = trace(rows)
        self.assertEqual(tokens[KEY].agents, 3)
        self.assertNotIn("Dax", C.parents_of(trees[KEY]))

    def test_a_quoter_does_not_count_towards_min_agents(self):
        rows = chat(("Ada", "room/a", USE), ("Bram", "room/a", "`lanternmoss` works for me"),
                    ("Cleo", "room/a", QUOTE))
        tokens, stats = T.find_tokens(rows)
        self.assertNotIn(KEY, tokens)
        self.assertEqual(stats["classes"]["code"]["under_min_agents"], 1)
        self.assertEqual(stats["classes"]["code"]["kept"], 0)

    def test_an_own_line_beside_the_quote_is_a_use(self):
        rows = chat(("Ada", "room/a", USE), ("Bram", "room/a", "`lanternmoss` works for me"),
                    ("Cleo", "room/a", QUOTE + "\nthen I will say `lanternmoss` as well"))
        tokens, _stats, trees, _tree_stats = trace(rows)
        self.assertEqual(tokens[KEY].agents, 3)
        self.assertEqual(C.parents_of(trees[KEY])["Cleo"], "Bram")

    def test_a_quote_inside_a_line_is_still_the_writer_s_own_words(self):
        rows = chat(("Ada", "room/a", USE), ("Bram", "room/a", "`lanternmoss` works for me"),
                    ("Cleo", "room/a", "Ada wrote: > the code is `lanternmoss` today"))
        tokens, _stats = T.find_tokens(rows)
        self.assertEqual(tokens[KEY].agents, 3)  # the limit the README states: one mark, at the line start

    def test_a_quoted_line_is_not_an_exposure(self):
        # Bram quotes Ada's line in room/b. Cleo only ever reads room/b, then uses the token in room/c.
        rows = chat(("Ada", "room/a", USE), ("Bram", "room/a", "morning, reading along"),
                    ("Bram", "room/b", QUOTE), ("Cleo", "room/b", "noted, thank you"),
                    ("Cleo", "room/c", "`lanternmoss` is the word"), ("Dax", "room/a", "`lanternmoss` it is"))
        tokens, _stats, trees, _tree_stats = trace(rows)
        tree = trees[KEY]
        self.assertEqual(tokens[KEY].rows, (0, 4, 5))
        self.assertEqual(tree.parallel_starts, ("Cleo",))  # met it only in a quote: no link, stated as a limit
        self.assertEqual(C.parents_of(tree), {"Ada": None, "Cleo": None, "Dax": "Ada"})

    def test_quoted_risky_word_does_not_mark_the_token(self):
        rows = chat(("Ada", "room/a", USE), ("Bram", "room/a", "`lanternmoss` works for me"),
                    ("Cleo", "room/a", "> use `lanternmoss` to bypass the gate\n`lanternmoss` here too"))
        tokens, _stats = T.find_tokens(rows)
        self.assertEqual(tokens[KEY].risky, ())


class QuoteAgainstThePrompt(unittest.TestCase):
    def test_a_quote_mark_before_a_command_is_a_quote_not_a_prompt(self):
        rows = chat(*[(name, "room/a", "> git clone ferry-docks") for name in ("Ada", "Bram", "Cleo")])
        tokens, stats = T.find_tokens(rows)
        self.assertEqual(tokens, {})
        self.assertEqual(stats["classes"]["command"]["distinct_seen"], 0)

    def test_other_prompts_and_backtick_spans_still_give_the_command(self):
        for text in ("$ git clone ferry-docks", "% git clone ferry-docks", "git clone ferry-docks",
                     "run `> git clone ferry-docks` next"):
            with self.subTest(text=text):
                rows = chat(*[(name, "room/a", text) for name in ("Ada", "Bram", "Cleo")])
                self.assertIn("command:git clone", T.find_tokens(rows)[0])


class WhatIsStillReadWhole(unittest.TestCase):
    def test_row_tokens_reads_quoted_lines_only_when_asked(self):
        self.assertEqual(T.row_tokens(QUOTE)[0], {})  # a log row: the rule is on unless a caller opts out
        self.assertEqual(sorted(T.row_tokens(QUOTE + "\n`orchardbell`")[0]), ["code:orchardbell"])
        self.assertIn(KEY, T.row_tokens(QUOTE, whole=True)[0])
        both = T.row_tokens(QUOTE + "\n`orchardbell`", whole=True)[0]
        self.assertIn(KEY, both)
        self.assertIn("code:orchardbell", both)
        self.assertEqual(T.row_tokens(USE, whole=True), T.row_tokens(USE))  # no quoted line: the same either way

    def test_an_off_wiki_text_is_read_whole(self):
        from patient_zero import offwiki as O
        token = T.Token(KEY, "code", "code", "lanternmoss", (0,), 3, (), ())
        events = [O.OffEvent(O.SRC_OTHER, 100.0, None, QUOTE, frozenset()),
                  O.OffEvent(O.SRC_OTHER, 200.0, None, "no token on this line", frozenset())]
        hits = O.find_hits(events, {KEY: token})
        self.assertEqual([hit[3] for hit in hits[KEY]], [0])  # found in the quoted line, as before the rule

    def test_a_given_text_drops_its_tokens_quoted_or_not(self):
        self.assertIn(KEY, T.given_keys([QUOTE]))
        rows = chat(("Ada", "room/a", USE), ("Bram", "room/a", USE), ("Cleo", "room/a", USE))
        tokens, stats = T.find_tokens(rows, given_texts=[QUOTE])
        self.assertNotIn(KEY, tokens)
        self.assertEqual(stats["classes"]["code"]["given_dropped"], 1)


class Peek(unittest.TestCase):
    def test_the_line_shown_for_a_token_is_never_a_quoted_line(self):
        token = T.Token(KEY, "code", "code", "lanternmoss", (0,), 3, (), ())
        self.assertEqual(pk.token_line(QUOTE + "\nI will say `lanternmoss` as well", token)[0],
                         "I will say `lanternmoss` as well")
        self.assertEqual(pk.token_line(QUOTE, token), (None, None))


class EndToEnd(unittest.TestCase):
    def test_a_generic_log_with_a_quoting_reply_makes_no_link_to_the_quoter(self):
        lines = [("Ada", "room/a", USE), ("Bram", "room/a", "`lanternmoss` works for me"),
                 ("Cleo", "room/a", "`lanternmoss` here too"), ("Dax", "room/a", QUOTE)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "quoting_chat.ndjson")
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                for i, (agent, place, text) in enumerate(lines):
                    fh.write(json.dumps({"id": "m%02d" % i, "agent": agent, "place": place, "text": text,
                                         "time": "2026-01-10T09:%02d:00Z" % i}) + "\n")
            result = P.run("generic", path, shuffles=3)
        self.assertEqual(result.tree_stats["links_named"], 2)
        self.assertEqual(result.tree_stats["adoptions"], 3)
        self.assertNotIn("Dax", C.parents_of(result.trees[KEY]))


class ShippedExamples(unittest.TestCase):
    def test_no_example_log_holds_a_quoted_line(self):
        folder = os.path.join(REPO, "examples")
        texts = []
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if name.endswith(".ndjson"):
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            event = json.loads(line)
                            texts.append((event.get("event") or event).get("text", ""))
            elif name.endswith(".csv"):
                with open(path, encoding="utf-8", newline="") as fh:
                    texts += [row.get("body") or row.get("text") or "" for row in csv.DictReader(fh)]
        self.assertGreater(len(texts), 40)
        quoted = [t for t in texts if any(T.is_quote(line) for line in t.split("\n"))]
        self.assertEqual(quoted, [])


if __name__ == "__main__":
    unittest.main()
