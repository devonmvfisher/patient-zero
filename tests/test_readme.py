"""Guards for the README, the front door a judge or a stranger reads first.

Standard library unittest only; no data needed. Three checks:

1. every repo path the README quotes exists, so no pointer goes nowhere;
2. every "README.md, section ..." pointer in the code names a real heading;
3. the German wiki numbers in the README's "At a glance" block agree with the
   committed figure, docs/lead.svg, so a redrawn figure cannot leave stale
   numbers in the README (or the other way round).
"""

import glob
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


if __name__ == "__main__":
    unittest.main()
