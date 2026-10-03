#!/usr/bin/env python3
"""Leak check for the patient-zero repo. Python standard library only.

It fails when the repo (or any extra file named on the command line) holds:

1. a ``.gz``, ``.jsonl`` or ``.zip`` file (their contents are never read);
2. an IP-shaped string: IPv4, IPv4 with ``x``/``*`` wildcards, or IPv6;
3. the half-IP field name used by the source export;
4. a run of 12 words copied from the data (records text, and the text each
   wiki save added).

Check 4 needs a hashed 12-gram index built once from the data. The index
holds 64-bit hashes only, never text, and it must live outside the repo
(default: ``../out/leakcheck-12grams.bin`` next to the repo folder)::

    python tools/leakcheck.py --build-index --data PATH/TO/collusion

Before every commit, and on any text written for the entry::

    python tools/leakcheck.py               # scan the repo
    python tools/leakcheck.py WRITEUP.md    # scan the repo plus extra files

Exit codes: 0 clean, 1 leak found, 2 setup error (for example a missing index).

Words are runs of letters and digits after case folding, so a copied run is
found even when re-wrapped, re-cased or re-punctuated. A 12-word window with
fewer than 4 distinct words (``0 0 0 ...``) is ignored on both sides.
"""

import argparse
import bisect
import datetime
import difflib
import gzip
import hashlib
import json
import os
import re
import sys
from array import array
from collections import namedtuple

NGRAM = 12
MIN_DISTINCT = 4
HASH_NAME = "blake2b-64"
INDEX_VERSION = 1
FORBIDDEN_EXTENSIONS = (".gz", ".jsonl", ".zip")
SKIP_DIRS = {".git"}
SNIPPET_MAX = 80
ADDED_OPS = {"insert", "replace"}

# The half-IP field name is assembled at run time so this file never holds it.
FIELD_NAME = "ip" + "16"
_FIELD_RE = re.compile(r"(?<![a-z0-9])" + FIELD_NAME + r"(?![0-9])", re.IGNORECASE)
_WORD_RE = re.compile(r"[^\W_]+")
_IPV4_RE = re.compile(r"(?<![0-9.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![0-9]|\.[0-9])")
_IPV4_WILD_RE = re.compile(
    r"(?<![0-9.])(\d{1,3})\.(\d{1,3})\.(?:x{1,3}|\*)\.(?:x{1,3}|\*)(?![0-9a-z])", re.IGNORECASE
)
_IPV6_RE = re.compile(r"(?<![0-9a-z:])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![0-9a-z:])", re.IGNORECASE)

Finding = namedtuple("Finding", "path line kind snippet")


class SetupError(Exception):
    """The check cannot run (for example the index is missing or misplaced)."""


# ---------------------------------------------------------------- words


def words_with_offsets(text):
    """Return [(word, offset)] with case-folded words of letters and digits."""
    return [(m.group(0).casefold(), m.start()) for m in _WORD_RE.finditer(text)]


def gram_hash(words):
    digest = hashlib.blake2b(" ".join(words).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def iter_gram_hashes(words):
    """Yield (start, hash) for every countable 12-word window."""
    for i in range(len(words) - NGRAM + 1):
        window = words[i:i + NGRAM]
        if len(set(window)) >= MIN_DISTINCT:
            yield i, gram_hash(window)


def snippet(text, offset=0):
    """Escaped, single-line text from offset, at most SNIPPET_MAX characters."""
    raw = text[offset:offset + SNIPPET_MAX * 2]
    escaped = raw.encode("unicode_escape").decode("ascii")
    return escaped[:SNIPPET_MAX]


def _is_inside(path, root):
    path, root = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives
        return False


# ---------------------------------------------------------------- data side


def _read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def added_segments(body, hunks, base_body):
    """Text blocks a save added: hunk lines for insert/replace ops.

    Lines are ``body.split("\\n")``, matching the export's line numbers. With no
    hunks, fall back to difflib against the base body, or the whole body when
    there is no base (page created).
    """
    lines = body.split("\n")
    if hunks:
        ops = [(h.get("op"), h.get("b0"), h.get("b1")) for h in hunks]
    elif base_body is None:
        ops = [("insert", 0, len(lines))]
    else:
        matcher = difflib.SequenceMatcher(None, base_body.split("\n"), lines, autojunk=False)
        ops = [(op, b0, b1) for op, _a0, _a1, b0, b1 in matcher.get_opcodes()]
    out = []
    for op, b0, b1 in ops:
        if op in ADDED_OPS and isinstance(b0, int) and isinstance(b1, int) and b1 > b0:
            out.append("\n".join(lines[b0:b1]))
    return out


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_index(data_dir, index_path, repo_root=None):
    """Hash every 12-word window of the data's added text into index_path."""
    if repo_root and _is_inside(index_path, repo_root):
        raise SetupError("refusing to write the index inside the repo: " + index_path)
    if repo_root and _is_inside(data_dir, repo_root):
        raise SetupError("refusing to read data from inside the repo: " + data_dir)
    hashes = set()
    sources = {}

    rev_path = os.path.join(data_dir, "revisions.jsonl.gz")
    if not os.path.isfile(rev_path):
        raise SetupError("no revisions.jsonl.gz in " + data_dir)
    bodies = {}
    stats = {"saves": 0, "segments": 0, "difflib_fallback": 0, "page_created_whole": 0}
    rows = []
    for r in _read_jsonl_gz(rev_path):
        # Only these four fields are read; everything else is never touched.
        rows.append((r.get("rev_id"), r.get("body") or "", r.get("hunks") or [], r.get("diff_base")))
        bodies[r.get("rev_id")] = r.get("body") or ""
    for rev_id, body, hunks, base_id in rows:
        stats["saves"] += 1
        base_body = None
        if not hunks:
            if base_id is not None and base_id in bodies:
                base_body = bodies[base_id]
                stats["difflib_fallback"] += 1
            else:
                stats["page_created_whole"] += 1
        for seg in added_segments(body, hunks, base_body):
            stats["segments"] += 1
            words = [w for w, _ in words_with_offsets(seg)]
            hashes.update(h for _, h in iter_gram_hashes(words))
    del bodies, rows
    stats["sha256"] = _sha256_file(rev_path)
    sources["revisions.jsonl.gz"] = stats

    rec_path = os.path.join(data_dir, "records.jsonl.gz")
    if os.path.isfile(rec_path):
        rstats = {"records": 0, "with_text": 0}
        for r in _read_jsonl_gz(rec_path):
            rstats["records"] += 1
            text = r.get("text")
            if isinstance(text, str) and text:
                rstats["with_text"] += 1
                words = [w for w, _ in words_with_offsets(text)]
                hashes.update(h for _, h in iter_gram_hashes(words))
        rstats["sha256"] = _sha256_file(rec_path)
        sources["records.jsonl.gz"] = rstats

    arr = array("Q", sorted(hashes))
    if sys.byteorder != "little":
        arr.byteswap()
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    with open(index_path, "wb") as fh:
        arr.tofile(fh)
    meta = {
        "version": INDEX_VERSION,
        "ngram": NGRAM,
        "min_distinct": MIN_DISTINCT,
        "hash": HASH_NAME,
        "byteorder": "little",
        "count": len(arr),
        "sources": sources,
        "built_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    with open(index_path + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return meta


class GramIndex:
    """Sorted 64-bit hashes of the data's 12-word windows."""

    def __init__(self, hashes):
        self._arr = hashes

    def __len__(self):
        return len(self._arr)

    def __contains__(self, value):
        i = bisect.bisect_left(self._arr, value)
        return i < len(self._arr) and self._arr[i] == value


def load_index(index_path, repo_root=None):
    if repo_root and _is_inside(index_path, repo_root):
        raise SetupError("the index must live outside the repo: " + index_path)
    meta_path = index_path + ".json"
    if not (os.path.isfile(index_path) and os.path.isfile(meta_path)):
        raise SetupError(
            "no 12-gram index at %s; build it with --build-index --data DIR" % index_path
        )
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    want = {"version": INDEX_VERSION, "ngram": NGRAM, "min_distinct": MIN_DISTINCT, "hash": HASH_NAME}
    if any(meta.get(k) != v for k, v in want.items()):
        raise SetupError("index settings differ from this script; rebuild with --build-index")
    arr = array("Q")
    with open(index_path, "rb") as fh:
        arr.frombytes(fh.read())
    if sys.byteorder != "little":
        arr.byteswap()
    if len(arr) != meta.get("count"):
        raise SetupError("index file is damaged; rebuild with --build-index")
    return GramIndex(arr)


# ---------------------------------------------------------------- repo side


def _line_of(starts, offset):
    return bisect.bisect_right(starts, offset)


def _octets_ok(groups):
    return all(0 <= int(g) <= 255 for g in groups)


def _ipv6_ok(candidate):
    import ipaddress

    try:
        ipaddress.IPv6Address(candidate)
    except ValueError:
        return False
    groups = [g for g in candidate.split(":") if g]
    if "::" not in candidate:
        return len(groups) == 8
    # Compressed form: skip Python slices (two short groups at most), keep
    # addresses with three groups or one group of three or more hex digits.
    return len(groups) >= 2 and (len(groups) >= 3 or any(len(g) >= 3 for g in groups))


def scan_text(text, index=None, path="<text>"):
    """Return findings for one text. index=None skips the 12-gram check."""
    starts = [m.end() for m in re.finditer("\n", text)]
    found = []

    def add(offset, kind):
        found.append(Finding(path, _line_of(starts, offset) + 1, kind, snippet(text, offset)))

    for m in _IPV4_RE.finditer(text):
        if _octets_ok(m.groups()):
            add(m.start(), "ip-address")
    for m in _IPV4_WILD_RE.finditer(text):
        if _octets_ok(m.groups()):
            add(m.start(), "ip-address")
    for m in _IPV6_RE.finditer(text):
        if _ipv6_ok(m.group(0)):
            add(m.start(), "ip-address")
    for m in _FIELD_RE.finditer(text):
        add(m.start(), "field-name")
    if index is not None:
        words = words_with_offsets(text)
        run_end = -1
        for i, h in iter_gram_hashes([w for w, _ in words]):
            if h in index:
                if i > run_end:  # a new copied run; report its first line only
                    add(words[i][1], "copied-12-words")
                run_end = i + NGRAM - 1
    found.sort(key=lambda f: (f.line, f.kind))
    return found


def iter_repo_files(repo_root):
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            yield os.path.join(dirpath, name)


def scan_file(path, index=None, shown=None):
    shown = shown or path
    if path.lower().endswith(FORBIDDEN_EXTENSIONS):
        return [Finding(shown, 0, "forbidden-file", "data-type file; contents not read")]
    with open(path, "rb") as fh:
        text = fh.read().decode("utf-8", errors="replace")
    return scan_text(text, index, shown)


def scan(repo_root, extra_paths=(), index=None):
    """Scan every file under repo_root (except .git) plus extra_paths."""
    findings, count = [], 0
    for path in iter_repo_files(repo_root):
        count += 1
        findings.extend(scan_file(path, index, os.path.relpath(path, repo_root).replace(os.sep, "/")))
    for path in extra_paths:
        if not os.path.isfile(path):
            raise SetupError("no such file: " + path)
        count += 1
        findings.extend(scan_file(path, index, path))
    return findings, count


# ---------------------------------------------------------------- CLI


def default_repo():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_index(repo_root):
    return os.path.join(os.path.dirname(os.path.abspath(repo_root)), "out", "leakcheck-12grams.bin")


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="Fail if the repo or given files leak data.")
    ap.add_argument("paths", nargs="*", help="extra files to scan (for example the write-up)")
    ap.add_argument("--repo", default=None, help="repo folder (default: this script's repo)")
    ap.add_argument("--index", default=None, help="12-gram index (default: ../out/leakcheck-12grams.bin)")
    ap.add_argument("--build-index", action="store_true", help="build the 12-gram index and exit")
    ap.add_argument("--data", default=None, help="data folder for --build-index")
    ap.add_argument("--max-report", type=int, default=20, help="findings printed per file")
    args = ap.parse_args(argv)
    repo = os.path.abspath(args.repo or default_repo())
    index_path = os.path.abspath(args.index or default_index(repo))
    try:
        if args.build_index:
            if not args.data:
                raise SetupError("--build-index needs --data DIR")
            meta = build_index(os.path.abspath(args.data), index_path, repo)
            print("leakcheck: index built, %d hashes -> %s" % (meta["count"], index_path), file=out)
            for name, stats in meta["sources"].items():
                counts = ", ".join("%s=%s" % (k, v) for k, v in stats.items() if k != "sha256")
                print("  %s: %s" % (name, counts), file=out)
            return 0
        index = load_index(index_path, repo)
        findings, count = scan(repo, args.paths, index)
    except SetupError as exc:
        print("leakcheck: SETUP ERROR: %s" % exc, file=out)
        return 2
    print("leakcheck: scanned %d files against %d data 12-grams" % (count, len(index)), file=out)
    per_file = {}
    for f in findings:
        per_file.setdefault(f.path, []).append(f)
    for path, items in per_file.items():
        for f in items[:args.max_report]:
            print("FAIL %s:%d [%s] %s" % (f.path, f.line, f.kind, f.snippet), file=out)
        if len(items) > args.max_report:
            print("FAIL %s: ... and %d more" % (path, len(items) - args.max_report), file=out)
    if findings:
        print("leakcheck: FAIL (%d findings)" % len(findings), file=out)
        return 1
    print("leakcheck: OK", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
