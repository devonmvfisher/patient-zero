#!/usr/bin/env python3
"""Claim check: every number in the public text must come from a run's own files.

Python standard library only, deterministic, read-only on every text it checks.

It pulls every number from ``README.md``, from the text of ``docs/*.svg``
(``<title>``, ``<desc>`` and ``<text>`` elements, never coordinates) and from
any ``WRITEUP*`` file in the repo folder, plus files named with ``--doc``.
Each number must do one of two things:

1. equal a value in a run's ``facts.json``, ``checks.json`` or
   ``calibration.json``, or in the lead tree's summary fields of
   ``top_trees.json`` (the file ``docs/lead.svg`` is drawn from); or
2. be covered by an entry in ``claims-allow.json`` that gives a reason.

Equal means equal as shown: ``28,250`` is 28250, ``19.7`` matches 19.73 (one
shown decimal), ``76`` matches 75.95, ``35.1%`` matches a share of 0.3508,
and a number written without a sign also matches its negative (``z 8.2``).
Dates (``2026-09-03``) match date strings in the run files or an allow entry.
Numbers glued to letters are names, not claims, and are skipped: ``F8``,
``J11``, ``SHA-256``, ``10th``, token ids such as ``ec8b9e2777d5``. A Markdown
list marker (``1.``) is skipped too.

An allow entry (``claims-allow.json``)::

    {"file": "README.md", "at": "cut to {} characters",
     "from": "{code:patient_zero/safe.py:SHORT_MAX}", "reason": "..."}

* ``file``: which files it covers (a pattern, for example ``docs/*.svg``).
* ``at``: finds one number by the words around it; ``{}`` marks the number,
  ``^`` and ``$`` pin the start and end of a line or SVG text element.
* ``text``: the number as it must be shown. With ``at``, a different number in
  that place is a failure, so a pinned number cannot drift either.
* ``from``: the value it must equal, recomputed on every run: references to a
  run file (``{wiki.facts:cascade.links_named}``, ``{wiki.lead:agents}``,
  ``{wiki.checks:order_null.shuffles}``) or to a constant in the code
  (``{code:tools/leakcheck.py:NGRAM}``), joined with ``+ - * / //`` and
  brackets.
* ``reason``: why the number is right. Required.

``runs`` in the same file says which runs each document talks about (for
example ``"docs/*.svg": ["wiki"]``). A document whose runs are all given is
checked strictly; when one of its runs is missing, numbers that cannot be
traced are listed as unverified instead of failing; with none of its runs,
the document is skipped and says so.

Runs: ``--run NAME=DIR`` (repeatable). By default the tool also builds the
example run from ``examples/fake_log.ndjson`` in a temporary folder (named
``example``; under a second) and uses ``../out`` as the run named ``wiki``
when it holds a facts.json. ``checks.json`` and ``top_trees.json`` are used
only when they belong to that facts.json (same row, token and link counts;
the top_trees sha256 that facts.json records). A stale file stops the check.

The check also measures itself. For every number it passes, it asks whether
the same number with its last shown digit moved by one (28,250 to 28,249 and
28,251) would still pass. Small numbers often would, because some other
count in the run files has that value. The summary says for how many numbers
a change would be caught, and ``--list`` names the weak ones.

It only flags. It never edits, rewrites or suggests wording for any text it
reads; the write-up is the author's own text. Output is counts, file:line
places, key names from the run files and short escaped contexts (at most 80
characters) from the checked documents.

Usage (from the repo folder)::

    python tools/claimcheck.py                      # README, docs/*.svg, WRITEUP*
    python tools/claimcheck.py --run wiki=../out --list
    python tools/claimcheck.py --doc ../out/FACTS-v1.md --json ../out/claims.json

Exit codes: 0 every number traced, 1 a number failed (or, with ``--strict``,
an allow entry no longer matches anything), 2 bad input or a stale run file.
"""

import argparse
import ast
import bisect
import contextlib
import fnmatch
import glob
import hashlib
import html
import io
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import namedtuple

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from patient_zero.safe import short  # noqa: E402

ALLOW_NAME = "claims-allow.json"
EXAMPLE_LOG = os.path.join("examples", "fake_log.ndjson")
RUN_FILES = ("facts", "checks", "calibration", "lead")
SOURCE_RANK = {name: i for i, name in enumerate(RUN_FILES)}
ENTRY_KEYS = {"file", "at", "text", "from", "reason"}
CONTEXT = 80

Token = namedtuple("Token", "kind text value dec pct mult signed start end")
Hit = namedtuple("Hit", "file line col token segment")

# ---------------------------------------------------------------- errors


class CheckError(Exception):
    """Bad input or a stale run file; the message says what to fix."""


# ---------------------------------------------------------------- numbers in text

_CHUNK_RE = re.compile(r"[^\s()\[\]{}<>\"'`*|;!?≈≤≥]+")  # "p≈0.99" -> "p", "0.99"
_NUMBER_RE = re.compile(
    r"(?P<sign>[-+\u2212])?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<dec>\d+))?(?P<suf>%|x|\u00d7)?\Z")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?Z?)?\Z")
_TIME_RE = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?\Z")
_TZ_RE = re.compile(r"[-+\u2212]\d{4}\Z")
_VERSION_RE = re.compile(r"v?\d+(?:\.\d+){2,}\Z")
_SPLIT_RE = re.compile(r"([/=\u2013\u2014]|(?<=[\w%.])-)")
_LIST_MARK_RE = re.compile(r"\s*(?:#{1,6}\s+)?(\d+)[.)]\s")
_LEAD = "~\u2248$\u20ac\u00a3\u00b1\u00d7"
_TRAIL = ".,:;+"
_CURRENCY_RE = re.compile(r"[A-Z]{1,3}\$(?=\d)")  # "US$80", "CA$10"
# A number with a glued unit is still a claim: "583s", "8.4h", "9.9\u03c3" are checked as numbers;
# "28.3k" or "2M" cannot be checked as shown, so they fail unless an allow entry pins them.
_UNIT_RE = re.compile(r"(?P<num>[-+\u2212]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
                      r"(?:(?P<unit>ms|s|min|h|d|\u03c3)|(?P<abbr>k|K|M|bn|B))\Z")


def parse_number(text, start=0):
    """Token for one number as shown (``28,250``, ``-0.95``, ``35.1%``, ``1.6x``), or None."""
    m = _NUMBER_RE.match(text)
    if not m:
        return None
    digits = m.group("int").replace(",", "")
    dec = m.group("dec") or ""
    value = float(digits + ("." + dec if dec else ""))
    sign = m.group("sign")
    if sign in ("-", "\u2212"):
        value = -value
    suf = m.group("suf")
    return Token("number", text, value, len(dec), suf == "%", suf in ("x", "\u00d7"), bool(sign),
                 start, start + len(text))


def _strip(chunk, offset):
    m = _CURRENCY_RE.match(chunk)
    if m:
        chunk, offset = chunk[m.end():], offset + m.end()
    while chunk and chunk[0] in _LEAD:
        chunk, offset = chunk[1:], offset + 1
    while chunk and chunk[-1] in _TRAIL:
        chunk = chunk[:-1]
    return chunk, offset


def scan_text(text, markdown=False):
    """Return (tokens, skipped) for one line or one SVG text element.

    ``skipped`` counts chunks that hold digits but are names (``F8``, ``SHA-256``,
    ``10th``, ``ec8b9e2777d5``), paths or list markers.
    """
    tokens, skipped = [], 0
    marker = None
    if markdown:
        lm = _LIST_MARK_RE.match(text)
        if lm:
            marker = lm.start(1)
    for cm in _CHUNK_RE.finditer(text):
        chunk, off = _strip(cm.group(0), cm.start())
        if not chunk or not any(ch.isdigit() for ch in chunk):
            continue
        if marker is not None and off == marker:
            skipped += 1
            continue
        if tokens and tokens[-1].kind == "time" and _TZ_RE.match(chunk):  # "16:43 -0500"
            tokens.append(Token("time", chunk, None, 0, False, False, False, off, off + len(chunk)))
            continue
        for kind, rx in (("date", _DATE_RE), ("time", _TIME_RE), ("version", _VERSION_RE)):
            if rx.match(chunk):
                tokens.append(Token(kind, chunk, None, 0, False, False, False, off, off + len(chunk)))
                break
        else:
            found, prev_numeric, prev_sep, pos = [], None, None, off
            names = 0
            for k, piece in enumerate(_SPLIT_RE.split(chunk)):
                if k % 2:  # re.split puts the separators at odd places
                    prev_sep, pos = piece, pos + len(piece)
                    continue
                if piece == "":
                    continue
                tok = parse_number(piece, pos)
                um = _UNIT_RE.match(piece) if tok is None else None
                if um and um.group("unit"):
                    tok = parse_number(um.group("num"), pos)
                elif um:
                    tok = Token("abbreviated", piece, None, 0, False, False, False, pos, pos + len(piece))
                if tok is None and "," in piece:  # "1,2" written without spaces
                    sub, subpos = [], pos
                    for part in piece.split(","):
                        t = parse_number(part, subpos) if part else None
                        if t is None:
                            sub = None
                            break
                        sub.append(t)
                        subpos += len(part) + 1
                    if sub:
                        tok = sub
                ok = tok is not None and (prev_numeric is None or prev_numeric or prev_sep == "=")
                if ok:
                    found.extend(tok if isinstance(tok, list) else [tok])
                elif any(ch.isdigit() for ch in piece):
                    names += 1
                prev_numeric = tok is not None and ok
                pos += len(piece)
            tokens.extend(found)
            if names and not found:
                skipped += 1
    return tokens, skipped


# ---------------------------------------------------------------- documents


def _display(path, repo):
    ap = os.path.abspath(path)
    try:
        rel = os.path.relpath(ap, repo)
    except ValueError:
        return ap.replace(os.sep, "/")
    if rel.startswith(".."):
        return ap.replace(os.sep, "/")
    return rel.replace(os.sep, "/")


def default_docs(repo):
    docs = []
    readme = os.path.join(repo, "README.md")
    if os.path.isfile(readme):
        docs.append(readme)
    docs.extend(sorted(glob.glob(os.path.join(repo, "docs", "*.svg"))))
    docs.extend(sorted(p for p in glob.glob(os.path.join(repo, "WRITEUP*")) if os.path.isfile(p)))
    return docs


def _line_col(text, index):
    line = text.count("\n", 0, index) + 1
    col = index - (text.rfind("\n", 0, index) + 1) + 1
    return line, col


_SVG_TEXT_RE = re.compile(r"<(title|desc|text)\b[^>]*>(.*?)</\1\s*>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]*>")
_HTML_DROP_RE = re.compile(r"<(script|style)\b.*?</\1\s*>|<!--.*?-->", re.DOTALL | re.IGNORECASE)


def segments(path):
    """Yield (line, col, text, markdown) for the human-readable text of one document.

    Opened read-only. SVG: the text of title, desc and text elements (entities
    decoded, inner tags dropped). HTML: each line with tags dropped. Anything
    else: each line as written.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".svg":
        for m in _SVG_TEXT_RE.finditer(raw):
            line, col = _line_col(raw, m.start(2))
            text = html.unescape(_TAG_RE.sub("", m.group(2)))
            yield line, col, " ".join(text.split()), False
    elif ext in (".html", ".htm"):
        cleaned = _HTML_DROP_RE.sub(lambda m: "\n" * m.group(0).count("\n"), raw)
        for n, line in enumerate(cleaned.split("\n"), 1):
            yield n, 1, html.unescape(_TAG_RE.sub(" ", line)), False
    else:
        for n, line in enumerate(raw.split("\n"), 1):
            yield n, 1, line.rstrip("\r"), ext in (".md", ".markdown", "")


def read_hits(path, shown):
    hits, skipped = [], 0
    segs = []
    for line, col, text, md in segments(path):
        segs.append(text)
        tokens, sk = scan_text(text, markdown=md)
        skipped += sk
        for tok in tokens:
            hits.append(Hit(shown, line, col + tok.start, tok, len(segs) - 1))
    return hits, segs, skipped


def context(text, tok):
    lo = max(0, tok.start - 34)
    hi = min(len(text), tok.end + 34)
    piece = ("..." if lo else "") + text[lo:hi].strip() + ("..." if hi < len(text) else "")
    return short(piece, CONTEXT)


# ---------------------------------------------------------------- run files


def _leaves(obj, prefix, out, dates):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _leaves(v, "%s.%s" % (prefix, k) if prefix else str(k), out, dates)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _leaves(v, "%s[%d]" % (prefix, i), out, dates)
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out[prefix] = obj
    elif isinstance(obj, str) and len(obj) <= 40 and re.match(r"\d{4}-\d{2}-\d{2}", obj):
        dates[prefix] = obj


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise CheckError("cannot read %s: %s" % (path, exc))


def _get(obj, *keys):
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_run(name, folder, skip_stale=False):
    """Numbers from one run folder: {source: {path: value}}, dates, notes."""
    facts_path = os.path.join(folder, "facts.json")
    if not os.path.isfile(facts_path):
        raise CheckError("run %s: no facts.json in %s" % (name, folder))
    facts = _load_json(facts_path)
    run = {"name": name, "folder": folder, "core": facts.get("core_sha256") or "", "sources": {},
           "dates": {}, "notes": [], "files": ["facts"]}
    values = {}
    _leaves(facts, "", values, run["dates"])
    run["sources"]["facts"] = values

    def stale(what, why):
        msg = "run %s: %s does not belong to its facts.json (%s)" % (name, what, why)
        if not skip_stale:
            raise CheckError(msg + "; rerun it, or pass --skip-stale to leave it out")
        run["notes"].append("left out " + what + " (stale: " + why + ")")

    checks_path = os.path.join(folder, "checks.json")
    if os.path.isfile(checks_path):
        checks = _load_json(checks_path)
        pairs = (("rows", checks.get("rows"), _get(facts, "load", "rows")),
                 ("tokens", checks.get("tokens"), _get(facts, "tokens", "kept_total")),
                 ("links", _get(checks, "observed", "default_min_lag_0s", "links_named"),
                  _get(facts, "cascade", "links_named")))
        bad = ["%s %s vs %s" % (k, a, b) for k, a, b in pairs if a is not None and b is not None and a != b]
        if bad:
            stale("checks.json", "; ".join(bad))
        else:
            vals = {}
            _leaves(checks, "", vals, run["dates"])
            run["sources"]["checks"] = vals
            run["files"].append("checks")

    cal_path = os.path.join(folder, "calibration.json")
    if os.path.isfile(cal_path):
        cal = _load_json(cal_path)
        tied = cal.get("facts_core_sha256") if isinstance(cal, dict) else None
        if tied and run["core"] and tied != run["core"]:
            stale("calibration.json", "facts_core_sha256 %s vs %s" % (str(tied)[:8], run["core"][:8]))
        else:
            vals = {}
            _leaves(cal, "", vals, run["dates"])
            run["sources"]["calibration"] = vals
            run["files"].append("calibration")
            if not tied:
                run["notes"].append("calibration.json carries no facts_core_sha256, so it is not tied to facts.json")

    top_path = os.path.join(folder, "top_trees.json")
    if os.path.isfile(top_path):
        want = _get(facts, "outputs", "top_trees.json", "sha256")
        have = _sha256(top_path)
        if want and want != have:
            stale("top_trees.json", "sha256 %s vs %s recorded" % (have[:8], str(want)[:8]))
        else:
            top = _load_json(top_path)
            lead_id = top.get("lead") if isinstance(top, dict) else None
            lead = None
            for tree in (top.get("trees") or []) if isinstance(top, dict) else []:
                if isinstance(tree, dict) and tree.get("token_id") == lead_id:
                    lead = tree
                    break
            if lead is not None:
                vals = {}
                _leaves({k: v for k, v in lead.items() if k != "nodes"}, "", vals, {})
                if isinstance(top.get("shuffles"), int):
                    vals["shuffles"] = top["shuffles"]
                run["sources"]["lead"] = vals
                run["files"].append("lead")
    return run


def build_example(repo, folder):
    """Run the pipeline on the committed example log into ``folder`` (made-up data)."""
    from patient_zero import pipeline  # noqa: E402 (imported only when needed)
    log = os.path.join(repo, EXAMPLE_LOG)
    if not os.path.isfile(log):
        raise CheckError("no example log at " + log)
    buf = io.StringIO()
    with contextlib.redirect_stderr(io.StringIO()):
        code = pipeline.main(["--adapter", "generic", "--input", log, "--out", folder], out=buf)
    if code:
        raise CheckError("the example run failed (exit %s)" % code)
    return folder


_CODE_CACHE = {}
_CONST_RE = re.compile(r"([A-Z][A-Z0-9_]*)[ ]*=[ ]*(-?[0-9][0-9_]*(?:[.][0-9]+)?)[ ]*(?:#.*)?$")


def code_constant(repo, ref):
    """A module-level number constant, ``file.py:NAME`` (``NAME = 80`` at the start of a line).

    Only the named file is read, as text; nothing is imported or run.
    """
    rel, sep, name = ref.rpartition(":")
    if not sep or not rel.endswith(".py") or rel.startswith(("/", ".")) or ".." in rel.split("/"):
        raise CheckError("bad code reference %s: use {code:path/file.py:NAME}" % short(ref, 60))
    key = (repo, rel)
    if key not in _CODE_CACHE:
        consts = {}
        try:
            with open(os.path.join(repo, *rel.split("/")), "r", encoding="utf-8") as fh:
                for line in fh:
                    m = _CONST_RE.match(line.rstrip("\r\n"))
                    if m:
                        consts.setdefault(m.group(1), float(m.group(2).replace("_", "")))
        except OSError:
            raise CheckError("no code file %s" % rel)
        _CODE_CACHE[key] = consts
    if name not in _CODE_CACHE[key]:
        raise CheckError("no number constant %s in %s" % (name, rel))
    return _CODE_CACHE[key][name]


# ---------------------------------------------------------------- matching


def half_unit(tok):
    return 0.5 * 10 ** (-tok.dec) + 1e-9 * max(1.0, abs(tok.value))


def matches_value(tok, x):
    """Does the shown number equal x, as shown (decimals, %, sign)?

    A number shown without a sign never equals a negative value: dropping the
    minus sign of z -8.2 turns "below chance" into "above chance".
    """
    h = half_unit(tok)
    cands = [x * 100.0, x] if tok.pct else [x]
    for c in cands:
        if abs(c - tok.value) <= h:
            return True
    return False


class Pool:
    """All numbers of some runs, searchable by shown value."""

    def __init__(self, items):
        self.size = len(items)
        self.values = dict(items)
        self.lists = {}
        for name, fn in (("raw", lambda v: v), ("mag", abs), ("pct", lambda v: v * 100.0),
                         ("pctmag", lambda v: abs(v) * 100.0)):
            pairs = sorted((fn(v), lab) for lab, v in items)
            self.lists[name] = ([p[0] for p in pairs], pairs)
        self.dates = {}

    def find(self, tok):
        """Best (label, value) for a token, or None (signs must agree)."""
        return self._search(tok, ["raw"] + (["pct"] if tok.pct else []))

    def find_magnitude(self, tok):
        """(label, true value) of a negative value an unsigned token would equal without its sign."""
        if tok.signed:
            return None
        found = self._search(tok, ["mag"] + (["pctmag"] if tok.pct else []))
        return (found[0], self.values[found[0]]) if found else None

    def _search(self, tok, names):
        h = half_unit(tok)
        best = None
        for name in names:
            vals, pairs = self.lists[name]
            lo = bisect.bisect_left(vals, tok.value - h)
            hi = bisect.bisect_right(vals, tok.value + h)
            for v, lab in pairs[lo:hi]:
                key = (abs(v - tok.value), _label_rank(lab), lab)
                if best is None or key < best[0]:
                    best = (key, lab, v)
        return (best[1], best[2]) if best else None

    def nearest(self, tok, k=2):
        vals, pairs = self.lists["pct" if tok.pct else "raw"]
        if not vals:
            return []
        i = bisect.bisect_left(vals, tok.value)
        cand = pairs[max(0, i - k):i + k]
        cand.sort(key=lambda p: (abs(p[0] - tok.value), _label_rank(p[1])))
        return cand[:k]

    def find_date(self, text):
        for lab, s in sorted(self.dates.items()):
            if s == text or s.startswith(text):
                return lab
        return None


def _label_rank(label):
    src = label.split(":", 1)[0].split(".", 1)[-1]
    return SOURCE_RANK.get(src, 9)


def make_pool(runs):
    items, dates = [], {}
    for run in runs:
        for src, vals in run["sources"].items():
            for path, v in vals.items():
                items.append(("%s.%s:%s" % (run["name"], src, path), v))
        for path, s in run["dates"].items():
            dates["%s:%s" % (run["name"], path)] = s
    pool = Pool(items)
    pool.dates = dates
    return pool


# ---------------------------------------------------------------- allow entries

_REF_RE = re.compile(r"\{([^{}]+)\}")
_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult,
                  ast.Div, ast.FloorDiv, ast.USub, ast.UAdd)


class Unverified(Exception):
    """A reference names a run or run file that was not given."""


def evaluate(expr, runs_by_name, repo):
    """Value of a ``from`` expression, or raise Unverified / CheckError."""
    missing = []

    def resolve(m):
        ref = m.group(1).strip()
        head, sep, rest = ref.partition(":")
        if not sep:
            raise CheckError("bad reference {%s}: use {run.file:key} or {code:file.py:NAME}" % ref)
        if head == "code":
            return repr(float(code_constant(repo, rest)))
        run_name, dot, src = head.partition(".")
        if not dot or src not in RUN_FILES:
            raise CheckError("bad reference {%s}: the file part must be one of %s" % (ref, ", ".join(RUN_FILES)))
        run = runs_by_name.get(run_name)
        if run is None:
            missing.append("run %s not given" % run_name)
            return "0.0"
        vals = run["sources"].get(src)
        if vals is None:
            missing.append("%s has no %s" % (run_name, src))
            return "0.0"
        if rest not in vals:
            raise CheckError("run %s: %s has no number at %s" % (run_name, src, rest))
        return repr(float(vals[rest]))

    text = _REF_RE.sub(resolve, expr)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise CheckError("bad from expression: " + short(expr))
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES) or (isinstance(node, ast.Constant)
                                                     and not isinstance(node.value, (int, float))):
            raise CheckError("from expressions allow numbers, references and + - * / // only: " + short(expr))
    if missing:
        raise Unverified("; ".join(sorted(set(missing))))
    try:
        return float(eval(compile(tree, "<from>", "eval"), {"__builtins__": {}}, {}))  # nodes checked above
    except ZeroDivisionError:
        raise CheckError("from expression divides by zero: " + short(expr))


def _match_file(pattern, shown):
    if fnmatch.fnmatchcase(shown, pattern):
        return True
    return "/" not in pattern and fnmatch.fnmatchcase(shown.rsplit("/", 1)[-1], pattern)


def load_allow(path):
    """Read and validate claims-allow.json; a missing file means no entries."""
    if not path or not os.path.isfile(path):
        return {"runs": [], "entries": [], "path": None}
    data = _load_json(path)
    if not isinstance(data, dict) or data.get("version") != 1:
        raise CheckError("%s: expected {\"version\": 1, \"runs\": {...}, \"entries\": [...]}" % path)
    runs = data.get("runs") or {}
    if not isinstance(runs, dict) or not all(isinstance(v, list) for v in runs.values()):
        raise CheckError("%s: runs must map a file pattern to a list of run names" % path)
    entries = []
    for i, e in enumerate(data.get("entries") or [], 1):
        where = "%s entry %d" % (os.path.basename(path), i)
        if not isinstance(e, dict):
            raise CheckError(where + ": not an object")
        extra = set(e) - ENTRY_KEYS
        if extra:
            raise CheckError(where + ": unknown key(s) " + ", ".join(sorted(extra)))
        if not isinstance(e.get("file"), str) or not e["file"]:
            raise CheckError(where + ": needs a file pattern")
        if not isinstance(e.get("reason"), str) or len(e["reason"].strip()) < 8:
            raise CheckError(where + ": needs a reason (a few words at least)")
        if "at" not in e and "text" not in e:
            raise CheckError(where + ": needs at or text")
        if "at" in e:
            at = e["at"]
            if not isinstance(at, str) or at.count("{}") != 1:
                raise CheckError(where + ": at must hold {} exactly once")
            body = at[1:] if at.startswith("^") else at
            body = body[:-1] if body.endswith("$") else body
            left, right = body.split("{}")
            rx = ("^" if at.startswith("^") else "") + re.escape(left) + \
                r"(?P<num>[-+\u2212]?\d[\w,.:\-]*[%\u00d7]?)" + re.escape(right) + ("$" if at.endswith("$") else "")
            e = dict(e, _rx=re.compile(rx))
        if "text" in e and not isinstance(e["text"], str):
            raise CheckError(where + ": text must be a string")
        if "from" in e and not isinstance(e["from"], str):
            raise CheckError(where + ": from must be a string")
        e = dict(e, _n=i, _used=0)
        entries.append(e)
    return {"runs": list(runs.items()), "entries": entries, "path": path}


def runs_for(shown, allow, given):
    for pattern, names in allow["runs"]:
        if _match_file(pattern, shown):
            return list(names)
    return list(given)


# ---------------------------------------------------------------- the check


def _variant(tok, step):
    """The token with its last shown digit moved by ``step`` (+1 or -1), or None."""
    delta = step * 10 ** (-tok.dec)
    value = tok.value + delta
    if not tok.signed and value < 0 <= tok.value:
        return None
    body = tok.text.lstrip("+-\u2212")
    commas = "," in body.split(".")[0]
    shown = ("{:,.%df}" if commas else "{:.%df}") % tok.dec
    shown = shown.format(abs(value))
    if tok.signed:
        shown = ("-" if value < 0 else "+") + shown
    elif value < 0:
        shown = "-" + shown
    if tok.pct:
        shown += "%"
    elif tok.mult:
        shown += "x"
    return tok._replace(text=shown, value=value)


def check(repo, docs, runs, allow, strict=False):
    """Check every number in ``docs``; return the result dict (nothing is written)."""
    by_name = {r["name"]: r for r in runs}
    given = [r["name"] for r in runs]
    pools = {}
    result = {"files": [], "numbers": [], "stale": [], "skipped_names": 0}
    file_hits = []
    for path in docs:
        shown = _display(path, repo)
        want = runs_for(shown, allow, given)
        have = [n for n in want if n in by_name]
        missing = [n for n in want if n not in by_name]
        info = {"file": shown, "runs": have, "missing_runs": missing, "status": "checked"}
        result["files"].append(info)
        if want and not have:
            info["status"] = "skipped (needs run %s)" % ", ".join(missing)
            continue
        hits, segs, skipped = read_hits(path, shown)
        result["skipped_names"] += skipped
        key = tuple(have)
        if key not in pools:
            pools[key] = make_pool([by_name[n] for n in have])
        file_hits.append((info, hits, segs, pools[key]))

    entries = allow["entries"]
    for info, hits, segs, pool in file_hits:
        shown = info["file"]
        mine = [e for e in entries if _match_file(e["file"], shown)]
        cover = {}
        for e in mine:  # entries with "at" first: they pin one place
            if "_rx" not in e:
                continue
            for si, text in enumerate(segs):
                for m in e["_rx"].finditer(text):
                    for idx, h in enumerate(hits):
                        if h.segment == si and h.token.start == m.start("num") and idx not in cover:
                            cover[idx] = e
                            e["_used"] += 1
        for idx, h in enumerate(hits):
            if idx in cover:
                continue
            for e in mine:
                if "_rx" not in e and e.get("text") == h.token.text:
                    cover[idx] = e
                    e["_used"] += 1
                    break
        counts = {"numbers": 0, "matched": 0, "allowed": 0, "pinned": 0, "unverified": 0, "fail": 0}
        for idx, h in enumerate(hits):
            tok = h.token
            row = {"file": shown, "line": h.line, "col": h.col, "text": short(tok.text, 40), "kind": tok.kind,
                   "context": context(segs[h.segment], tok)}
            e = cover.get(idx)
            status = _judge(tok, e, pool, by_name, repo, info["missing_runs"], row)
            row["status"] = status
            if e is not None:
                row["allow_entry"] = e["_n"]
            if status in ("matched", "allowed", "pinned") and tok.kind == "number":
                row["caught"] = _caught(tok, e, pool, by_name, repo, mine)
            counts["numbers"] += 1
            counts[status] += 1
            result["numbers"].append(row)
        info.update(counts)

    checked_files = {i["file"] for i, _h, _s, _p in file_hits}
    for e in entries:
        if e["_used"]:
            continue
        if any(_match_file(e["file"], f) for f in checked_files):
            result["stale"].append({"entry": e["_n"], "file": e["file"],
                                    "where": short(e.get("at") or e.get("text"), 60)})
    nums = [r for r in result["numbers"] if "caught" in r]
    result["power"] = {"checked": len(nums), "caught": sum(1 for r in nums if r["caught"])}
    result["fail"] = sum(1 for r in result["numbers"] if r["status"] == "fail")
    result["unverified"] = sum(1 for r in result["numbers"] if r["status"] == "unverified")
    result["ok"] = result["fail"] == 0 and not (strict and result["stale"])
    return result


def _judge(tok, e, pool, by_name, repo, missing_runs, row):
    """Status of one number: matched, allowed, pinned, unverified or fail (row gets the receipt)."""
    if e is not None:
        if "text" in e and e["text"] != tok.text:
            row["why"] = "allow entry %d pins %s here" % (e["_n"], short(e["text"], 30))
            return "fail"
        if "from" in e:
            try:
                x = evaluate(e["from"], by_name, repo)
            except Unverified as exc:
                row["why"] = "allow entry %d: %s" % (e["_n"], exc)
                return "unverified"
            row["source"] = "from: " + short(e["from"], 70)
            row["value"] = x
            if tok.kind != "number" or not matches_value(tok, x):
                row["why"] = "allow entry %d: the sources give %s%s" % (
                    e["_n"], _fmt(x), " (the minus sign is not shown)" if tok.kind == "number"
                    and x < 0 and matches_value(tok, -x) else "")
                return "fail"
            return "pinned"
        row["source"] = "reason: " + short(e["reason"], 70)
        return "allowed"
    if tok.kind == "number":
        found = pool.find(tok)
        if found:
            row["source"], row["value"] = found
            return "matched"
    elif tok.kind == "date":
        lab = pool.find_date(tok.text)
        if lab:
            row["source"] = lab
            return "matched"
    if missing_runs:
        row["why"] = "not in the runs given; run %s missing" % ", ".join(missing_runs)
        return "unverified"
    neg = pool.find_magnitude(tok) if tok.kind == "number" else None
    if neg:
        row["why"] = "the minus sign is not shown; %s = %s" % (neg[0], _fmt(neg[1]))
    elif tok.kind == "number":
        near = pool.nearest(tok)
        row["why"] = "not in facts/checks/calibration/lead" + (
            "; nearest " + ", ".join("%s = %s" % (lab, _fmt(v)) for v, lab in near) if near else "")
    elif tok.kind == "abbreviated":
        row["why"] = "an abbreviated number cannot be checked as shown; write it in full or pin it with text"
    else:
        row["why"] = "a %s with no allow entry" % tok.kind
    return "fail"


def _sign_dropped(tok):
    """The token with its minus sign removed (z -8.2 shown as z 8.2), or None."""
    if not tok.signed or tok.value >= 0:
        return None
    return tok._replace(text=tok.text.lstrip("-−"), value=-tok.value, signed=False)


def _caught(tok, e, pool, by_name, repo, mine):
    """Would moving the last shown digit by one (either way), or dropping a minus sign, make it fail?"""
    for v in (_variant(tok, 1), _variant(tok, -1), _sign_dropped(tok)):
        if v is None:
            continue
        if e is not None and "_rx" in e:
            if "text" in e:
                continue  # the pinned text no longer matches: caught
            if "from" in e:
                try:
                    x = evaluate(e["from"], by_name, repo)
                except Unverified:
                    return False
                if matches_value(v, x):
                    return False
                continue
            return False  # a pattern entry (axis ticks) allows any value
        other = [x for x in mine if "_rx" not in x and x.get("text") == v.text]
        if other or pool.find(v):
            return False
    return True


def _fmt(x):
    if isinstance(x, float) and x.is_integer() and abs(x) < 1e15:
        x = int(x)
    if isinstance(x, int):
        return "{:,}".format(x)
    return ("%.6g" % x)


# ---------------------------------------------------------------- report


def report(result, runs, out, listing=False, max_report=50):
    for r in runs:
        out.write("run %s: core %s, %s (%s)%s\n" % (
            r["name"], (r["core"] or "?")[:8], ", ".join(r["files"]), short(r.get("shown") or r["folder"], 70),
            "; " + "; ".join(r["notes"]) if r["notes"] else ""))
    for f in result["files"]:
        if f["status"] != "checked":
            out.write("%s: %s\n" % (f["file"], f["status"]))
            continue
        out.write("%s: %d numbers: %d matched, %d pinned to a source, %d allowed by reason, %d unverified, "
                  "%d FAIL%s\n" % (f["file"], f["numbers"], f["matched"], f["pinned"], f["allowed"],
                                   f["unverified"], f["fail"],
                                   " (runs missing: %s)" % ", ".join(f["missing_runs"]) if f["missing_runs"] else ""))
    shown = {}
    for row in result["numbers"]:
        if row["status"] == "fail":
            shown[row["file"]] = shown.get(row["file"], 0) + 1
            if shown[row["file"]] <= max_report:
                out.write("FAIL %s:%d:%d  %s  %s  | %s\n" % (row["file"], row["line"], row["col"], row["text"],
                                                           row["why"], row["context"]))
    for row in result["numbers"]:
        if row["status"] == "unverified":
            out.write("UNVERIFIED %s:%d:%d  %s  %s\n" % (row["file"], row["line"], row["col"], row["text"],
                                                       row["why"]))
    for s in result["stale"]:
        out.write("STALE allow entry %d (%s, %s): matches nothing now\n" % (s["entry"], s["file"], s["where"]))
    if listing:
        for row in result["numbers"]:
            if row["status"] in ("matched", "pinned", "allowed"):
                weak = "" if row.get("caught", True) else "  [weak: a change of 1 would still pass]"
                out.write("  %s:%d  %s  %s  <- %s%s\n" % (row["file"], row["line"], row["status"], row["text"],
                                                       short(row.get("source", ""), 90), weak))
    p = result["power"]
    out.write("drift check: moving the last shown digit by one, or dropping a minus sign, is caught for "
              "%d of %d passing numbers%s\n" % (
        p["caught"], p["checked"], " (%.0f%%)" % (100.0 * p["caught"] / p["checked"]) if p["checked"] else ""))
    total = len(result["numbers"])
    out.write("claimcheck: %s (%d numbers in %d files, %d FAIL, %d unverified, %d stale allow entries; "
              "%d names with digits skipped)\n" % ("PASS" if result["ok"] else "FAIL", total,
                                                   sum(1 for f in result["files"] if f["status"] == "checked"),
                                                   result["fail"], result["unverified"], len(result["stale"]),
                                                   result["skipped_names"]))


def _inside(path, folder):
    path, folder = os.path.abspath(path), os.path.abspath(folder)
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:
        return False


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(
        prog="claimcheck.py",
        description="Check that every number in README.md, docs/*.svg and WRITEUP* comes from a run's "
                    "facts.json, checks.json, calibration.json or lead tree, or from claims-allow.json. "
                    "Read-only on every document.")
    ap.add_argument("--repo", default=REPO_DIR, help="repo folder (default: this script's repo)")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=DIR",
                    help="a run's output folder (repeatable); default: the example run, plus wiki=../out "
                         "when it holds facts.json")
    ap.add_argument("--no-example", action="store_true", help="do not build the example run")
    ap.add_argument("--doc", action="append", default=[], help="another document to check (read-only; repeatable)")
    ap.add_argument("--only", action="append", default=[], metavar="PATTERN",
                    help="check only documents matching this pattern (repeatable), e.g. README.md")
    ap.add_argument("--allow", default=None, help="allow file (default: claims-allow.json in the repo)")
    ap.add_argument("--strict", action="store_true", help="also fail when an allow entry matches nothing")
    ap.add_argument("--skip-stale", action="store_true",
                    help="leave out a checks.json, calibration.json or top_trees.json that does not belong "
                         "to its facts.json, instead of stopping")
    ap.add_argument("--list", action="store_true", help="print every passing number with its source")
    ap.add_argument("--json", default=None, help="write the full result (every number and its source) here")
    ap.add_argument("--max-report", type=int, default=50, help="failures printed per file")
    args = ap.parse_args(argv)
    started = time.perf_counter()
    repo = os.path.abspath(args.repo)
    try:
        if args.json and _inside(args.json, repo) and not _inside(args.json, os.path.join(repo, "out")):
            raise CheckError("--json must be outside the repo (or inside its git-ignored out/ folder)")
        named = []
        for spec in args.run:
            name, sep, folder = spec.partition("=")
            if not sep or not name or not folder:
                raise CheckError("--run takes NAME=DIR, for example wiki=../out")
            named.append((name, folder))
        names = [n for n, _ in named]
        if len(set(names)) != len(names):
            raise CheckError("each --run needs its own name")
        default_wiki = os.path.join(os.path.dirname(repo), "out")
        if "wiki" not in names and not args.run and os.path.isfile(os.path.join(default_wiki, "facts.json")):
            named.append(("wiki", default_wiki))
        allow = load_allow(args.allow or os.path.join(repo, ALLOW_NAME))
        docs = default_docs(repo) + [os.path.abspath(d) for d in args.doc]
        for d in docs:
            if not os.path.isfile(d):
                raise CheckError("no such document: " + d)
        if args.only:
            docs = [d for d in docs if any(_match_file(p, _display(d, repo)) for p in args.only)]
        with tempfile.TemporaryDirectory(prefix="claimcheck-") as tmp:
            runs = []
            if not args.no_example and "example" not in names:
                runs.append(load_run("example", build_example(repo, tmp)))
                runs[-1]["shown"] = "built from " + EXAMPLE_LOG.replace(os.sep, "/")
            for name, folder in named:
                runs.append(load_run(name, folder, skip_stale=args.skip_stale))
                runs[-1]["shown"] = _display(folder, repo)
            if not runs:
                raise CheckError("no run to check against: pass --run NAME=DIR or drop --no-example")
            result = check(repo, docs, runs, allow, strict=args.strict)
        result["seconds"] = round(time.perf_counter() - started, 3)
        report(result, runs, out, listing=args.list, max_report=args.max_report)
        out.write("time: %.2f s\n" % result["seconds"])
        if args.json:
            payload = {"tool": "patient-zero claimcheck", "ok": result["ok"],
                       "runs": [{"name": r["name"], "core_sha256": r["core"], "files": r["files"],
                                 "notes": r["notes"]} for r in runs],
                       "files": result["files"], "numbers": result["numbers"], "stale": result["stale"],
                       "power": result["power"], "seconds": result["seconds"]}
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1, sort_keys=True)
                fh.write("\n")
        return 0 if result["ok"] else 1
    except CheckError as exc:
        out.write("claimcheck: error: %s\n" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
