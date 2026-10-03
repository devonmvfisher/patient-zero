"""Run patient-zero end to end: load, find tokens, build spread trees, write outputs.

Python standard library only, deterministic (no randomness in this stage).
One command does the whole run::

    python cli.py --adapter wiki --input PATH/TO/collusion --out PATH/TO/out
    python cli.py --adapter generic --input examples/fake_log.ndjson --out out/example

``--out`` must be outside the repo, or inside the repo's git-ignored ``out/``
folder. Two files are written there:

``facts.json``
    Counts only: input file names, sizes and SHA-256 hashes; parameters;
    loader counts; token counts per class; spread-tree counts; page-family and
    group (wiki) tables; risky-word types; the fact-sheet slots this stage can
    fill; run times. ``core_sha256`` hashes every section that should not
    change between two runs on the same input (not times, not the code state).

``trees.jsonl``
    One line per spread tree, biggest first, summary fields only:

    * ``token_id`` (12 hex characters of the token key's SHA-256, a stable
      handle) and ``token`` (the key, escaped, at most 80 characters);
    * class, adopter counts by status, link counts, depth, largest branch;
    * page families and groups by use count;
    * first user and effective seed (agent labels, escaped, at most 80
      characters) with the time, family and group of their first use;
    * risky-word types (type names only).

    No row text, no page names. Blank and human rows have no label after
    loading, so their labels can never appear.

Strings taken from the data pass through ``short``: anything outside
printable ASCII and every backslash is escaped, and the result is cut to 80
characters. Nothing here opens a URL or runs a command from the data.
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from statistics import median
from typing import NamedTuple

from patient_zero.cascade import STATUSES, build_trees
from patient_zero.load import ADAPTERS, iso_utc, load
from patient_zero.tokens import CLASSES, MAX_PHRASE_SHARE, MIN_AGENTS, RISKY_TYPES, find_tokens

FACTS_VERSION = 1
SHORT_MAX = 80
TARGET_SECONDS = 300
NO_FAMILY = "(none)"
WIKI_FILES = ("revisions.jsonl.gz", "pages.jsonl.gz", "labels.jsonl.gz")
# Risky types that name a mechanism; "answer", "share" and "relay" are broad words.
NARROW_RISKY = ("bypass", "workaround", "blocked", "captcha", "rate_limit", "proxy", "cors")
CORE_KEYS = ("input", "params", "load", "tokens", "cascade", "trees", "families", "groups",
             "risky", "outputs")
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Result(NamedTuple):
    """Everything one run computed, kept in memory for later stages."""

    rows: list
    load_stats: dict
    tokens: dict
    token_stats: dict
    trees: dict
    tree_stats: dict
    seconds: dict  # stage -> seconds


# ---------------------------------------------------------------- small helpers


def short(value, limit=SHORT_MAX):
    """Escape a data string (non-printable-ASCII and backslashes) and cut it to ``limit``."""
    if value is None:
        return None
    pieces = []
    for ch in str(value):
        code = ord(ch)
        if ch == "\\":
            pieces.append("\\\\")
        elif 32 <= code < 127:
            pieces.append(ch)
        elif code < 0x10000:
            pieces.append("\\u%04x" % code)
        else:
            pieces.append("\\U%08x" % code)
    if sum(len(p) for p in pieces) <= limit:
        return "".join(pieces)
    kept, size = [], 0
    for piece in pieces:  # never cut inside an escape
        if size + len(piece) > limit - 3:
            break
        kept.append(piece)
        size += len(piece)
    return "".join(kept) + "..."


def token_id(key):
    """Stable 12-hex handle for a token key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path, folder):
    path = os.path.normcase(os.path.realpath(path))
    folder = os.path.normcase(os.path.realpath(folder))
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:  # different drives
        return False


def check_out_dir(out_dir, repo_dir=REPO_DIR):
    """Refuse an output folder inside the repo, except the repo's git-ignored out/."""
    if _inside(out_dir, repo_dir) and not _inside(out_dir, os.path.join(repo_dir, "out")):
        raise ValueError("--out must be outside the repo (or inside its git-ignored out/ folder)")


def input_files(adapter, path):
    """[{name, bytes, sha256}] for the files the adapter reads (base names only)."""
    if adapter == "wiki":
        paths = [os.path.join(path, n) for n in WIKI_FILES if os.path.isfile(os.path.join(path, n))]
    else:
        paths = [path]
    return [{"name": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": _sha256_file(p)}
            for p in paths]


def code_info(repo_dir=REPO_DIR):
    """Commit, dirty flag, code lines and test count of this checkout (None when unknown)."""
    info = {"commit": None, "dirty": None, "code_lines": None, "test_functions": None}
    kwargs = {"capture_output": True, "text": True, "timeout": 20}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW: no console pops up
    try:
        head = subprocess.run(["git", "-C", repo_dir, "rev-parse", "--short=7", "HEAD"], **kwargs)
        if head.returncode == 0:
            info["commit"] = head.stdout.strip()
            status = subprocess.run(["git", "-C", repo_dir, "status", "--porcelain"], **kwargs)
            if status.returncode == 0:
                info["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    code_files = sorted(glob.glob(os.path.join(repo_dir, "patient_zero", "*.py")))
    code_files += [p for p in (os.path.join(repo_dir, "cli.py"),) if os.path.isfile(p)]
    if code_files:
        lines = 0
        for p in code_files:
            with open(p, "r", encoding="utf-8") as fh:
                lines += sum(1 for line in fh if line.strip())
        info["code_lines"] = lines
    test_files = sorted(glob.glob(os.path.join(repo_dir, "tests", "test_*.py")))
    if test_files:
        count = 0
        for p in test_files:
            with open(p, "r", encoding="utf-8") as fh:
                count += len(re.findall(r"^\s*def test_\w+\(", fh.read(), re.MULTILINE))
        info["test_functions"] = count
    return info


def safe_keys(obj):
    """Copy with every dict key a short escaped string (None becomes "(none)").

    Loader stats key some counts by group or time grade, which come from the
    data and may be None; JSON with sorted keys needs strings.
    """
    if isinstance(obj, dict):
        return {(NO_FAMILY if k is None else short(k)): safe_keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_keys(v) for v in obj]
    return obj


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------- the run


def run(adapter, input_path, min_agents=MIN_AGENTS, max_phrase_share=MAX_PHRASE_SHARE,
        given_texts=()):
    """Load, find tokens and build spread trees. Returns a Result (in memory)."""
    seconds = {}
    t0 = time.perf_counter()
    rows, load_stats = load(adapter, input_path)
    seconds["load"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    tokens, token_stats = find_tokens(rows, min_agents=min_agents, given_texts=given_texts,
                                      max_phrase_share=max_phrase_share)
    seconds["tokens"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    trees, tree_stats = build_trees(rows, tokens)
    seconds["cascade"] = time.perf_counter() - t0
    token_stats = {k: v for k, v in token_stats.items() if k != "seconds"}
    tree_stats = {k: v for k, v in tree_stats.items() if k != "seconds"}
    return Result(rows, load_stats, tokens, token_stats, trees, tree_stats, seconds)


def branch_sizes(tree):
    """Root agent -> agents in its branch (the root plus everyone linked below it)."""
    size = {a.agent: 1 for a in tree.adoptions}
    for a in reversed(tree.adoptions):  # parents come earlier in first-use order
        if a.parent is not None:
            size[a.parent] += size[a.agent]
    return {a.agent: size[a.agent] for a in tree.adoptions if a.parent is None}


def _top(counter, n):
    return sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[:n]


def tree_record(tree, token, rows):
    """Summary fields of one spread tree (see the module docstring)."""
    use_rows = token.rows
    families = Counter(rows[i].family or NO_FAMILY for i in use_rows)
    groups = Counter(rows[i].group or NO_FAMILY for i in use_rows)
    status = Counter(a.status for a in tree.adoptions)
    named = [link for link in tree.links if link.source is not None]
    children = Counter(a.parent for a in tree.adoptions if a.parent is not None)
    big_root, big_size = None, 0
    for root, size in branch_sizes(tree).items():  # first-use order: ties go to the earlier root
        if size > big_size:
            big_root, big_size = root, size
    first = tree.adoptions[0]
    seed = next((a for a in tree.adoptions if a.agent == tree.effective_seed), None)
    return {
        "token_id": token_id(tree.token),
        "token": short(tree.token),
        "cls": tree.cls,
        "kind": tree.kind,
        "agents": len(tree.adoptions),
        "status": {s: status[s] for s in STATUSES},
        "links_named": len(named),
        "links_unnamed": len(tree.links) - len(named),
        "shifted_by_clock": tree.shifted,
        "depth": tree.depth,
        "largest_branch": big_size,
        "largest_branch_root": short(big_root),
        "max_children": max(children.values()) if children else 0,
        "uses": len(use_rows),
        "places": tree.places,
        "families": [[short(f), n] for f, n in _top(families, 3)],
        "groups": [[short(g), n] for g, n in _top(groups, 10)],
        "first_user": short(tree.first_user),
        "first_use": iso_utc(first.time),
        "first_family": short(rows[first.row].family),
        "first_group": short(rows[first.row].group),
        "effective_seed": short(tree.effective_seed),
        "seed_use": iso_utc(seed.time) if seed else None,
        "seed_family": short(rows[seed.row].family) if seed else None,
        "seed_children": children[seed.agent] if seed else 0,
        "seed_is_first_user": (tree.effective_seed == tree.first_user) if seed else None,
        "span_hours": round((tree.adoptions[-1].time - first.time) / 3600.0, 2),
        "named_lag_median_s": round(median(link.lag for link in named), 1) if named else None,
        "risky": list(token.risky),
    }


def tree_records(result):
    """Records for every tree, biggest first (agents, then named links, then token key)."""
    out = []
    for key, tree in result.trees.items():
        if tree.adoptions:
            out.append((tree, tree_record(tree, result.tokens[key], result.rows)))
    out.sort(key=lambda tr: (-tr[1]["agents"], -tr[1]["links_named"], tr[0].token))
    return [rec for _tree, rec in out]


# ---------------------------------------------------------------- facts


def _family_table(result):
    rows = result.rows
    table = defaultdict(Counter)
    for row in rows:
        table[row.family or NO_FAMILY]["rows"] += 1
    for tree in result.trees.values():
        if tree.adoptions:
            table[rows[tree.adoptions[0].row].family or NO_FAMILY]["trees_first_used_here"] += 1
        for link in tree.links:
            if link.source is None:
                continue
            src = rows[link.source_row].family or NO_FAMILY
            dst = rows[link.target_row].family or NO_FAMILY
            table[src]["named_links_from_here"] += 1
            table[dst]["named_links_into_here"] += 1
            if src == dst:
                table[src]["named_links_within"] += 1
    fields = ("rows", "trees_first_used_here", "named_links_from_here", "named_links_into_here",
              "named_links_within")
    ordered = sorted(table.items(), key=lambda kv: (-kv[1]["named_links_from_here"], -kv[1]["rows"], kv[0]))
    return [dict([("family", short(name))] + [(f, c[f]) for f in fields]) for name, c in ordered]


def _group_table(result):
    rows = result.rows
    table = defaultdict(Counter)
    for row in rows:
        table[row.group or NO_FAMILY]["rows"] += 1
    for tree in result.trees.values():
        if tree.adoptions:
            table[rows[tree.adoptions[0].row].group or NO_FAMILY]["trees_first_used_here"] += 1
        for link in tree.links:
            if link.source is not None:
                src = rows[link.source_row].group or NO_FAMILY
                table[src]["named_links_from_here"] += 1
                if (rows[link.target_row].group or NO_FAMILY) != src:
                    table[src]["named_links_to_other_group"] += 1
    fields = ("rows", "trees_first_used_here", "named_links_from_here", "named_links_to_other_group")
    ordered = sorted(table.items(), key=lambda kv: (-kv[1]["rows"], kv[0]))
    return [dict([("group", short(name))] + [(f, c[f]) for f in fields]) for name, c in ordered]


def _risky_table(records):
    by_type = Counter()
    links_by_type = Counter()
    linked_by_type = Counter()
    for rec in records:
        for t in rec["risky"]:
            by_type[t] += 1
            links_by_type[t] += rec["links_named"]
            if rec["links_named"]:
                linked_by_type[t] += 1
    return {
        "types": list(RISKY_TYPES),
        "narrow_types": list(NARROW_RISKY),
        "trees_with_type": {t: by_type[t] for t in RISKY_TYPES},
        "trees_with_type_and_named_links": {t: linked_by_type[t] for t in RISKY_TYPES},
        "named_links_in_trees_with_type": {t: links_by_type[t] for t in RISKY_TYPES},
        "trees_with_any_type": sum(1 for r in records if r["risky"]),
        "trees_with_narrow_type": sum(1 for r in records if set(r["risky"]) & set(NARROW_RISKY)),
        "trees_with_narrow_type_and_named_links": sum(
            1 for r in records if r["links_named"] and set(r["risky"]) & set(NARROW_RISKY)),
    }


def _trees_table(records):
    def count(pred):
        return sum(1 for r in records if pred(r))

    return {
        "total": len(records),
        "with_named_links": count(lambda r: r["links_named"] > 0),
        "agents_5plus": count(lambda r: r["agents"] >= 5),
        "agents_10plus": count(lambda r: r["agents"] >= 10),
        "largest_branch_3plus": count(lambda r: r["largest_branch"] >= 3),
        "largest_branch_5plus": count(lambda r: r["largest_branch"] >= 5),
        "largest_branch_10plus": count(lambda r: r["largest_branch"] >= 10),
        "largest_branch_max": max((r["largest_branch"] for r in records), default=0),
        "with_effective_seed": count(lambda r: r["effective_seed"] is not None),
        "seed_is_not_first_user": count(lambda r: r["seed_is_first_user"] is False),
        "with_parallel_starts": count(lambda r: r["status"]["parallel"] > 0),
        "by_class": {c: count(lambda r, c=c: r["cls"] == c) for c in CLASSES},
        "by_class_with_named_links": {c: count(lambda r, c=c: r["cls"] == c and r["links_named"] > 0)
                                      for c in CLASSES},
    }


def build_facts(result, records, files, params, trees_meta, code, started, total_seconds):
    """The facts.json object (counts only)."""
    ts = result.token_stats
    cs = result.tree_stats
    trees = _trees_table(records)
    risky = _risky_table(records)
    with_seed = trees["with_effective_seed"]
    facts = {
        "tool": "patient-zero",
        "facts_version": FACTS_VERSION,
        "stage": "J5 first full run: load, tokens, spread trees (shuffle baseline, scores and "
                 "report are later stages)",
        "input": {"adapter": params["adapter"], "files": files},
        "params": {
            "min_agents": params["min_agents"],
            "max_phrase_share": params["max_phrase_share"],
            "given_texts": params["given_texts"],
            "randomness": "none in this stage",
        },
        "load": safe_keys(result.load_stats),
        "tokens": safe_keys(ts),
        "cascade": safe_keys(cs),
        "trees": trees,
        "families": _family_table(result),
        "groups": _group_table(result),
        "risky": risky,
        "outputs": {"trees.jsonl": trees_meta},
    }
    facts["slots"] = {
        "F3": {
            "candidate_tokens": sum(ts["classes"][c]["distinct_seen"] for c in CLASSES),
            "tokens_3plus_agents": ts["kept_total"],
            "tokens_3plus_agents_by_class": {c: ts["classes"][c]["kept"] for c in CLASSES},
        },
        "F4": {"links_named": cs["links_named"],
               "links_from_exposure_only_rows": cs["links_from_exposure_only_rows"]},
        "F6": {"tokens_5plus_agents": trees["agents_5plus"],
               "tokens_10plus_agents": trees["agents_10plus"],
               "branches_5plus_agents": trees["largest_branch_5plus"],
               "branches_10plus_agents": trees["largest_branch_10plus"]},
        "F8": {"parallel_starts": cs["parallel_starts"],
               "trees_with_effective_seed": with_seed,
               "seed_is_not_first_user": trees["seed_is_not_first_user"],
               "share_seed_is_not_first_user": round(trees["seed_is_not_first_user"] / with_seed, 4)
               if with_seed else None},
        "F9": {"links_dropped_by_clock": cs["links_dropped_by_clock"],
               "links_shifted_by_clock": cs["links_shifted_by_clock"],
               "tokens_dropped_by_clock_rule": sum(ts["classes"][c]["clock_dropped"] for c in CLASSES)},
        "F12": {"trees_with_type": risky["trees_with_type"],
                "trees_with_narrow_type": risky["trees_with_narrow_type"]},
        "F14": {"runtime_seconds": round(total_seconds, 1), "code_lines": code["code_lines"],
                "test_functions": code["test_functions"]},
        "pending": {
            "F1": "lead finding: picked after the shuffle baseline and scores (J6, J10)",
            "F2": "lead tree page family: with F1",
            "F5": "shuffle baseline (J6 nullmodel)",
            "F7": "speed to 10th adopter (J6 score)",
            "F10": "hand check (J9)",
            "F11": "planted tests (C2, tests/test_planted.py)",
            "F13": "stretch (J8)",
        },
    }
    facts["core_keys"] = list(CORE_KEYS)
    facts["core_sha256"] = hashlib.sha256(
        _canonical({k: facts[k] for k in CORE_KEYS}).encode("ascii")).hexdigest()
    facts["code"] = code
    facts["runtime"] = {
        "seconds": {k: round(v, 2) for k, v in result.seconds.items()},
        "total_seconds": round(total_seconds, 2),
        "target_seconds": TARGET_SECONDS,
        "under_target": total_seconds < TARGET_SECONDS,
        "python": platform.python_version(),
        "started_utc": started,
    }
    return facts


def write_outputs(out_dir, result, params, started_clock, started_utc):
    """Write trees.jsonl and facts.json to out_dir. Returns the facts object."""
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    records = tree_records(result)
    lines = "".join(json.dumps(r, ensure_ascii=True, separators=(",", ":")) + "\n" for r in records)
    trees_path = os.path.join(out_dir, "trees.jsonl")
    _write_atomic(trees_path, lines)
    trees_meta = {"lines": len(records), "sha256": hashlib.sha256(lines.encode("ascii")).hexdigest()}
    result.seconds["records"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    files = input_files(params["adapter"], params["input"])
    code = code_info()
    result.seconds["hash_and_code_info"] = time.perf_counter() - t0
    total = time.perf_counter() - started_clock
    facts = build_facts(result, records, files, params, trees_meta, code, started_utc, total)
    _write_atomic(os.path.join(out_dir, "facts.json"), json.dumps(facts, indent=1, ensure_ascii=True) + "\n")
    return facts


# ---------------------------------------------------------------- command line


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(
        prog="cli.py",
        description="Run patient-zero end to end and write facts.json and trees.jsonl "
                    "(counts and summary fields only; no row text).")
    ap.add_argument("--adapter", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--input", required=True, help="export folder (wiki) or JSON Lines file (generic)")
    ap.add_argument("--out", required=True, help="output folder, outside the repo")
    ap.add_argument("--min-agents", type=int, default=MIN_AGENTS)
    ap.add_argument("--max-phrase-share", type=float, default=MAX_PHRASE_SHARE)
    ap.add_argument("--given", action="append", default=[],
                    help="a shared goal or prompt text file; its tokens are dropped (repeatable)")
    args = ap.parse_args(argv)
    started_clock = time.perf_counter()
    started_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        check_out_dir(args.out)
        given = []
        for path in args.given:
            with open(path, "r", encoding="utf-8") as fh:
                given.append(fh.read())
        result = run(args.adapter, args.input, min_agents=args.min_agents,
                     max_phrase_share=args.max_phrase_share, given_texts=given)
        params = {"adapter": args.adapter, "input": args.input, "min_agents": args.min_agents,
                  "max_phrase_share": args.max_phrase_share, "given_texts": len(given)}
        facts = write_outputs(args.out, result, params, started_clock, started_utc)
    except (OSError, ValueError) as exc:
        print("patient-zero: ERROR: %s" % exc, file=out)
        return 2
    s = facts["slots"]
    print("patient-zero: %d rows, %d agents, %d groups" % (
        facts["load"]["rows"], facts["load"]["agents"], len(facts["groups"])), file=out)
    print("tokens (3+ agents): %d of %d candidates" % (
        s["F3"]["tokens_3plus_agents"], s["F3"]["candidate_tokens"]), file=out)
    print("trees: %d, with named links %d, branches of 5+ agents %d, of 10+ %d" % (
        facts["trees"]["total"], facts["trees"]["with_named_links"],
        s["F6"]["branches_5plus_agents"], s["F6"]["branches_10plus_agents"]), file=out)
    print("links named: %d, dropped by clock %d, parallel starts %d" % (
        s["F4"]["links_named"], s["F9"]["links_dropped_by_clock"], s["F8"]["parallel_starts"]), file=out)
    print("runtime: %.1f s (target under %d s)" % (facts["runtime"]["total_seconds"], TARGET_SECONDS),
          file=out)
    print("wrote: %s and %s" % (os.path.join(args.out, "facts.json"),
                                os.path.join(args.out, "trees.jsonl")), file=out)
    print("core_sha256: %s" % facts["core_sha256"], file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
