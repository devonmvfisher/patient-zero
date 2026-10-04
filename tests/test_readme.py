"""Guards for the README, the front door a judge or a stranger reads first.

Standard library unittest only; no data needed. Five checks:

1. every repo path the README quotes exists, so no pointer goes nowhere;
2. every "README.md, section ..." pointer in the code names a real heading;
3. the German wiki numbers in the README's "At a glance" block agree with the
   committed figure, docs/lead.svg, so a redrawn figure cannot leave stale
   numbers in the README (or the other way round);
4. no command the README gives to copy draws a hand-picked tree over that
   figure, so following the README cannot break check 3;
5. the paragraph that says who drafted the code names the one file another
   maker wrote, and that file is still the bytes that maker returned.
"""

import glob
import hashlib
import html
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def prose(md):
    """The README without fenced code blocks (commands there use placeholder paths)."""
    return re.sub(r"```.*?```", "", md, flags=re.S)


def numbers(text):
    """Every number in a string, as floats (thousands commas removed)."""
    return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]


class Readme(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md = read("README.md")
        cls.svg = read("docs", "lead.svg")
        cls.svg_lines = [html.unescape(t) for t in re.findall(r"<text[^>]*>([^<]*)</text>", cls.svg)]

    def test_quoted_repo_paths_exist(self):
        paths = set()
        for quoted in re.findall(r"`([^`\s]+)`", prose(self.md)):
            if "/" in quoted or re.search(r"\.(py|svg|md|ndjson|json|html|jsonl)$", quoted):
                paths.add(quoted)
        for target in re.findall(r"\]\(([^)\s]+)\)", self.md):
            if not re.match(r"[a-z]+://", target):
                paths.add(target)
        self.assertTrue(paths)
        names = set()
        for folder, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in (".git", "out", "__pycache__")]
            names.update(files)
        missing = []
        for p in sorted(paths):
            if p.startswith((".", "out/", "PATH/")) or p.endswith(".jsonl") or "TOKEN_ID" in p:
                continue  # file extensions, run outputs, placeholders and the input format, not repo files
            if p in ("facts.json", "trees.jsonl", "top_trees.json", "report.html", "checks.json", "calibration.json"):
                continue  # names of files a run (or checks / calibrate) writes
            if "*" in p:
                continue  # a glob such as docs/*.svg, not one file
            if re.match(r"^pz-[a-z]+/\d+$", p):
                continue  # a schema id, not a path
            if "/" not in p:
                if p not in names:  # a bare module name such as cascade.py, inside patient_zero/
                    missing.append(p)
            elif not os.path.exists(os.path.join(ROOT, *p.split("/"))):
                missing.append(p)
        self.assertEqual(missing, [], "README quotes paths that are not in the repo")

    def test_section_pointers_in_code_name_real_headings(self):
        headings = set(h.strip() for h in re.findall(r"^#{1,6}\s+(.*)$", self.md, flags=re.M))
        pointers = []
        for path in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
            if os.sep + "tests" + os.sep in path:
                continue
            with open(path, encoding="utf-8") as fh:
                pointers += re.findall(r'README\.md, section \\?"([^"\\]+)\\?"', fh.read())
                fh.seek(0)
                bare = re.findall(r"README\.md, section ([A-Z][a-z]+)[,.]", fh.read())
            self.assertEqual(bare, [], "unquoted README section pointer in %s" % os.path.basename(path))
        self.assertTrue(pointers)
        for name in pointers:
            self.assertIn(name, headings)

    def test_at_a_glance_wiki_numbers_match_the_figure(self):
        bullet = next(line for line in self.md.splitlines() if line.startswith("- **German wiki run:**"))
        have = numbers(bullet)
        whole = next(line for line in self.svg_lines if line.startswith("Whole run: "))
        for n in numbers(whole):
            self.assertTrue(any(round(x) == n for x in have), "figure number %s missing from README" % n)
        picked = next(line for line in self.svg_lines if line.startswith("Picked as the largest of "))
        trees, bound = numbers(picked)[:2]
        self.assertIn(trees, have)
        self.assertIn(bound, have)
        # the order null is not called "within chance": its random-row controls read below its band
        self.assertNotIn("within chance", whole)
        self.assertNotIn("within chance", bullet)
        self.assertIn("can-it-see.svg", whole)
        self.assertIn("below chance", whole)
        self.assertIn("below chance", bullet)

    def test_no_fenced_command_draws_a_hand_picked_tree_over_a_published_figure(self):
        """A copied command must leave docs/lead.svg as the check above needs it.

        ``report --lead`` writes "Picked by hand (--lead)." into the figure. The README once gave
        ``--lead TOKEN_ID --svg docs/lead.svg`` as a command to copy: run as written, it replaced the
        published figure, and the check above then failed for the reader. A fenced command that names
        a tree by hand must draw under the git-ignored out/ folder.
        """
        blocks = re.findall(r"```\n(.*?)```", self.md, flags=re.S)
        self.assertTrue(blocks)
        for line in "\n".join(blocks).splitlines():
            words = line.split()
            if "--lead" in words and "--svg" in words[:-1]:
                target = words[words.index("--svg") + 1]
                self.assertTrue(target.startswith("out/"), "%s would be drawn over by: %s" % (target, line))

    def test_who_drafted_the_code_names_the_file_another_maker_wrote(self):
        """Claude did not write every file here, and the README must not read as if it did.

        tests/test_redteam_summary_contracts.py was written by a Codex agent working as a red team
        and was taken byte for byte: the SHA-256 below is the file that came back. The paragraph
        that says who drafted the code must name that file and its maker. If the file is ever
        edited, this fails, and "byte for byte" has to leave the README with it.
        """
        named = "tests/test_redteam_summary_contracts.py"
        who = [p for p in self.md.split("\n\n") if "drafted by Claude" in p and "Co-Authored-By" in p]
        self.assertEqual(len(who), 1, "one paragraph says who drafted the code")
        self.assertIn("`%s`" % named, who[0])
        self.assertIn("Codex", who[0])
        self.assertIn("byte for byte", who[0])
        with open(os.path.join(ROOT, *named.split("/")), "rb") as fh:
            self.assertEqual(hashlib.sha256(fh.read()).hexdigest(),
                             "5e7d115b76a7afcf58b62352d71bb20767da863a0e815523252494891e4a7f99",
                             "%s is not the file the red team returned" % named)


if __name__ == "__main__":
    unittest.main()
