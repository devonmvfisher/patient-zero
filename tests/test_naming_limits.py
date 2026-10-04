# SPDX-License-Identifier: MIT OR Apache-2.0
"""The names a token is traced under, as the README states them ("What a token is called").

Standard library unittest only; made-up text, no data needed. Each test pins
one sentence of that paragraph, so the sentence and the code cannot drift
apart unnoticed:

1. a URL is traced under its host, never as a full address;
2. a command is a token only where a line or a backtick span starts with a
   known command name and an argument follows;
3. any other command, or one inside a sentence, gives phrases or code words,
   if anything;
4. a token used by fewer than ``min_agents`` distinct agents is not traced.
"""
import unittest

from patient_zero import load as L, tokens as T


def keys(line):
    return sorted(key for key, _cls, _kind, _value in T.line_tokens(line)[0])


def rows(agents, text):
    return [L.Row("r%d" % i, name, L.ROLE_AGENT, i * 100.0, 0.0, None, "harbour", None, None, text)
            for i, name in enumerate(agents)]


class NamingLimits(unittest.TestCase):
    def test_url_is_traced_under_its_host_never_as_a_full_address(self):
        one = keys("https://ferry.harbourline.test/docks/plan-seven")
        two = keys("https://ferry.harbourline.test/other/page")
        self.assertEqual(one, ["host:ferry.harbourline.test"])
        self.assertEqual(two, one)  # two pages on one host are one token
        both = T.row_tokens("https://ferry.harbourline.test/docks/plan-seven\n"
                            "https://ferry.harbourline.test/other/page")[0]
        self.assertEqual(sorted(both), ["host:ferry.harbourline.test"])
        self.assertFalse(any("/" in key or "://" in key for key in both))

    def test_file_name_in_a_url_path_is_a_token_of_its_own(self):
        self.assertEqual(keys("https://ferry.harbourline.test/docks/tide-notes.txt"),
                         ["file:tide-notes.txt", "host:ferry.harbourline.test"])

    def test_known_command_needs_a_line_or_backtick_start_and_an_argument(self):
        self.assertIn("command:curl", keys("curl -s ferry-docks"))
        self.assertIn("command:git clone", keys("fetch it with `git clone` first"))
        self.assertIn("command:git clone", keys("$ sudo git clone ferry-docks"))
        self.assertFalse([k for k in keys("curl") if k.startswith("command:")])  # no argument: prose
        inside = keys("please run curl -s ferry-docks now")
        self.assertFalse([k for k in inside if k.startswith("command:")])
        self.assertIn("phrase:please run curl", inside)

    def test_unknown_command_is_phrases_or_code_words_if_anything(self):
        self.assertEqual(keys("zorp --fast"), [])  # line start, not on the list, too short for a phrase
        self.assertEqual(keys("`zorp --fast`"), [])
        self.assertEqual(keys("`zorp`"), ["code:zorp"])  # a one-word backtick span is a code word
        inside = keys("please run zorp --fast before the next round")
        self.assertFalse([k for k in inside if k.startswith("command:")])
        self.assertIn("phrase:run zorp fast", inside)

    def test_token_below_min_agents_is_not_traced(self):
        two, stats = T.find_tokens(rows(["Ada", "Bram"], "`orchardbell`"))
        self.assertEqual(two, {})
        self.assertEqual(stats["classes"]["code"]["under_min_agents"], 1)
        three, _ = T.find_tokens(rows(["Ada", "Bram", "Cleo"], "`orchardbell`"))
        self.assertEqual(set(three), {"code:orchardbell"})
        one_agent, _ = T.find_tokens(rows(["Ada"] * 300, "`orchardbell`"))
        self.assertEqual(one_agent, {})  # many uses by one agent are still one user


if __name__ == "__main__":
    unittest.main()
