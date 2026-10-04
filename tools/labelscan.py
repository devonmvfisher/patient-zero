#!/usr/bin/env python3
"""Label and token scan: do agent labels or token text show up in public files?

Python standard library only. From the data folder it reads only the ``label``
and ``is_human_handle`` fields of ``labels.jsonl.gz`` (the loader's own
whitelist reader), and from a run's ``trees.jsonl`` only the ``token`` field.
It prints counts, line numbers and match lengths only: never a label, a token
or a matched string.

Matching is case-folded and runs on what a viewer reads: for an SVG, every
<text> and <title>; for any other file, the whole text. A needle matches as a
whole word: where its first or last character is a letter or digit, the
character next to it in the file must not be one. Labels are scanned raw and
in their escaped form (``safe.short``). Token text is the part after the class
prefix (``host:``, ``phrase:``...) and, for nested prefixes, also the last part
when it has 3 or more characters.

    python tools/labelscan.py --data PATH/TO/collusion --trees ../out/trees.jsonl docs/can-it-see.svg

Exit codes: 0 no hit, 1 a hit, 2 setup error.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from patient_zero import load as L  # noqa: E402
from patient_zero.safe import short  # noqa: E402

MIN_LAST_PART = 3


def label_needles(data_dir):
    """Case-folded label strings (raw and escaped), plus counts."""
    path = os.path.join(data_dir, "labels.jsonl.gz")
    if not os.path.isfile(path):
        raise FileNotFoundError("no labels.jsonl.gz in the data folder")
    needles, stats = set(), {"labels": 0, "human": 0, "blank": 0}
    for _n, lab in L._read_json_lines(path, L._LABEL_KEYS):
        value = lab.get("label")
        if not isinstance(value, str) or not value.strip():
            stats["blank"] += 1
            continue
        stats["labels"] += 1
        stats["human"] += lab.get("is_human_handle") is True
        for form in (value.strip(), short(value.strip())):
            needles.add(form.casefold())
    return needles, stats


def token_needles(trees_path):
    needles, count = set(), 0
    with open(trees_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            token = json.loads(line).get("token")
            if not isinstance(token, str) or ":" not in token:
                continue
            count += 1
            rest = token.split(":", 1)[1]
            if rest.strip():
                needles.add(rest.casefold())
            last = token.split(":")[-1]
            if len(last) >= MIN_LAST_PART:
                needles.add(last.casefold())
    return needles, {"tokens": count}


def visible_text(path):
    with open(path, "rb") as fh:
        raw = fh.read().decode("utf-8", errors="replace")
    if path.lower().endswith(".svg"):
        try:
            root = ET.fromstring(raw.split("\n", 1)[1] if raw.startswith("<?xml") else raw)
        except ET.ParseError:
            return raw
        parts = []
        for el in root.iter():
            tag = el.tag.split("}", 1)[-1]
            if tag in ("text", "title"):
                parts.append("".join(el.itertext()))
        return "\n".join(parts)
    return raw


def find_hits(text, needles):
    """[(line, length)] of whole-word, case-folded needle matches in text."""
    hay = text.casefold()
    hits = []
    for needle in needles:
        start = hay.find(needle)
        while start >= 0:
            end = start + len(needle)
            left_ok = not needle[0].isalnum() or start == 0 or not hay[start - 1].isalnum()
            right_ok = not needle[-1].isalnum() or end == len(hay) or not hay[end].isalnum()
            if left_ok and right_ok:
                hits.append((hay.count("\n", 0, start) + 1, len(needle)))
            start = hay.find(needle, start + 1)
    return sorted(hits)


def scan(paths, labels, tokens):
    report = {}
    for path in paths:
        text = visible_text(path)
        report[path] = {"label_hits": find_hits(text, labels), "token_hits": find_hits(text, tokens),
                        "chars": len(text)}
    return report


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="Fail if agent labels or token text appear in the given files.")
    ap.add_argument("paths", nargs="+", help="files to scan (SVG: visible text only)")
    ap.add_argument("--data", required=True, help="the export folder (labels.jsonl.gz)")
    ap.add_argument("--trees", required=True, help="a run's trees.jsonl (token field)")
    args = ap.parse_args(argv)
    try:
        labels, lstats = label_needles(args.data)
        tokens, tstats = token_needles(args.trees)
        report = scan(args.paths, labels, tokens)
    except (OSError, ValueError) as exc:
        print("labelscan: SETUP ERROR: %s" % exc, file=out)
        return 2
    print("labelscan: %d labels (%d flagged human, %d blank skipped), %d tokens, %d files"
          % (lstats["labels"], lstats["human"], lstats["blank"], tstats["tokens"], len(args.paths)), file=out)
    total = 0
    for path, r in report.items():
        n = len(r["label_hits"]) + len(r["token_hits"])
        total += n
        print("%s %s: %d label hits, %d token hits (%d visible chars)%s"
              % ("FAIL" if n else "ok", os.path.basename(path), len(r["label_hits"]), len(r["token_hits"]),
                 r["chars"], "; at line:length " + ", ".join("%d:%d" % h for h in r["label_hits"] + r["token_hits"])
                 if n else ""), file=out)
    print("labelscan: %s" % ("FAIL (%d hits)" % total if total else "OK (0 agent labels, 0 token text)"), file=out)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
