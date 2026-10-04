"""Run patient-zero end to end: load, find tokens, build spread trees, check them
against a label shuffle, score them, write outputs.

Python standard library only, deterministic: the only randomness is the label
shuffle (``patient_zero.nullmodel``), driven by one fixed seed.
One command does the whole run::

    python cli.py --adapter wiki --input PATH/TO/collusion --out PATH/TO/out
    python cli.py --adapter generic --input examples/fake_log.ndjson --out out/example
    python cli.py --adapter generic --input examples/fake_log.csv --out out/csv         --map agent=user,time=ts,place=channel,text=body

``--map`` (generic adapter only) names the log's own column for each field;
see ``patient_zero.load.parse_map``. A CSV or TSV input, or a map, is
recorded in the ``input`` section of facts.json.

``--out`` must be outside the repo, or inside the repo's git-ignored ``out/``
folder, and must be a folder that is not an input and not inside the input
folder (``check_out_paths``). Four files are written there:

``facts.json``
    Counts: input file names, sizes and SHA-256 hashes; parameters; loader
    counts; token counts per class; spread-tree counts; page-family and group
    (wiki) tables; risky-word types, against the shuffle; the shuffle baseline
    (``nullmodel``); scores (influence, speed, page-family flow); the
    fact-sheet slots this stage can fill; run times. The influence table names
    the top agents by label (escaped, at most 80 characters), like
    trees.jsonl. ``core_sha256`` hashes every section that should not change
    between two runs on the same input and seed (not times, not the code
    state).

``trees.jsonl``
    One line per spread tree, biggest first, summary fields only:

    * ``token_id`` (12 hex characters of the token key's SHA-256, a stable
      handle) and ``token`` (the key, escaped, at most 80 characters);
    * class, adopter counts by status, link counts, depth, largest branch;
    * page families and groups by use count;
    * first user and effective seed (agent labels, escaped, at most 80
      characters) with the time, family and group of their first use;
    * risky-word types (type names only);
    * speed: minutes from the effective seed (and from the largest branch's
      root) to the 3rd and 10th agent linked below it;
    * the shuffle: mean named links under the shuffle and how many shuffles
      reached the observed count.

    No row text, no page names. Blank and human rows have no label after
    loading, so their labels can never appear.

``top_trees.json``
    Node-level detail (one node per adopter: escaped label, minutes after the
    first use, page family, wiki, status, source node) for the lead tree, the
    largest tree, and the top 20 trees by largest branch overall and among
    trees that beat every shuffle (``patient_zero.report.detail_set``).
    ``--lead TOKEN_ID`` picks the lead tree and adds it if needed.

``report.html``
    The offline report built from the three files above
    (``patient_zero.report``; no scripts, no external files).

Strings taken from the data pass through ``short``: anything outside
printable ASCII and every backslash is escaped, and the result is cut to 80
characters. The input file names in facts.json are shown the same way, with
IP-shaped text masked. Nothing here opens a URL or runs a command from the
data.
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
from typing import NamedTuple, Optional

from patient_zero import nullmodel as N
from patient_zero import report as R
from patient_zero import score as S
from patient_zero.cascade import STATUSES, build_trees
from patient_zero.load import ADAPTERS, MAP_HELP, input_note, iso_utc, load, parse_map, scrub_ips
from patient_zero.safe import NO_FAMILY, SHORT_MAX, guard_out, short, token_id  # noqa: F401 (re-exported)
from patient_zero.score import NARROW_RISKY, branch_sizes  # noqa: F401 (re-exported)
from patient_zero.tokens import CLASSES, MAX_PHRASE_SHARE, MIN_AGENTS, find_tokens

FACTS_VERSION = 3
TARGET_SECONDS = 300
WIKI_FILES = ("revisions.jsonl.gz", "pages.jsonl.gz", "labels.jsonl.gz")
CORE_KEYS = ("input", "params", "load", "tokens", "cascade", "trees", "families", "groups",
             "risky", "outputs", "nullmodel", "scores")
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
    null: Optional[N.NullResult] = None  # the label-shuffle baseline


# ---------------------------------------------------------------- small helpers


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


OUTPUT_FILES = ("facts.json", "trees.jsonl", "top_trees.json", "report.html")


def check_out_paths(out_dir, inputs):
    """Refuse an --out that is a file, is an input, sits inside an input folder, or
    would hold an output file with the same path as an input (plain sentences)."""
    guard_out(out_dir, inputs, "--out")
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        raise ValueError("--out names %s, which is a file; give a folder for the four outputs" % out_dir)
    for name in OUTPUT_FILES:
        guard_out(os.path.join(out_dir, name), inputs, "--out")


def input_files(adapter, path):
    """[{name, bytes, sha256}] for the files the adapter reads (base names only)."""
    if adapter == "wiki":
        paths = [os.path.join(path, n) for n in WIKI_FILES if os.path.isfile(os.path.join(path, n))]
    else:
        paths = [path]
    # The name is shown, not opened again: IP-shaped text masked, escaped and cut like any data string.
    return [{"name": short(scrub_ips(os.path.basename(p))[0]), "bytes": os.path.getsize(p),
             "sha256": _sha256_file(p)} for p in paths]


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
        given_texts=(), shuffles=N.DEFAULT_SHUFFLES, seed=N.DEFAULT_SEED, scope=N.SCOPE_DAY,
        field_map=None):
    """Load, find tokens, build spread trees, run the shuffle. Returns a Result (in memory).

    ``field_map`` (generic adapter only) maps field names to the log's own columns.
    """
    seconds = {}
    t0 = time.perf_counter()
    rows, load_stats = load(adapter, input_path, field_map)
    seconds["load"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    tokens, token_stats = find_tokens(rows, min_agents=min_agents, given_texts=given_texts,
                                      max_phrase_share=max_phrase_share)
    seconds["tokens"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    trees, tree_stats = build_trees(rows, tokens)
    seconds["cascade"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    null = N.run(rows, tokens, shuffles=shuffles, seed=seed, scope=scope)
    seconds["nullmodel"] = time.perf_counter() - t0
    token_stats = {k: v for k, v in token_stats.items() if k != "seconds"}
    tree_stats = {k: v for k, v in tree_stats.items() if k != "seconds"}
    return Result(rows, load_stats, tokens, token_stats, trees, tree_stats, seconds, null)


def _top(counter, n):
    return sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[:n]


def tree_record(tree, token, rows, shuffle=(None, None)):
    """Summary fields of one spread tree (see the module docstring).

    ``shuffle`` is (mean named links under the shuffle, shuffles reaching the
    observed count) from ``nullmodel.per_token``.
    """
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
    speed = S.tree_speed(tree, big_root)
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
        "seed_branch": speed["seed_branch"],
        "seed_min_to_3rd": speed["seed_min_to_3rd"],
        "seed_min_to_10th": speed["seed_min_to_10th"],
        "branch_min_to_3rd": speed["branch_min_to_3rd"],
        "branch_min_to_10th": speed["branch_min_to_10th"],
        "shuffle_links_mean": shuffle[0],
        "shuffle_ge_observed": shuffle[1],
    }


def tree_records(result):
    """Records for every tree, biggest first (agents, then named links, then token key)."""
    shuffle = N.per_token(result.null) if result.null is not None else {}
    out = []
    for key, tree in result.trees.items():
        if tree.adoptions:
            out.append((tree, tree_record(tree, result.tokens[key], result.rows,
                                          shuffle.get(key, (None, None)))))
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


def _null_slot(null_facts):
    """Fact-sheet slot F5 from the facts.json "nullmodel" section."""
    links = null_facts["links_named"]
    share = null_facts["linked_share_of_later_adopters"]
    adopt = null_facts["adopters"]
    return {
        "links_observed": links["observed"],
        "links_expected_mean": links["mean"],
        "links_expected_sd": links["sd"],
        "links_expected_p05": links["p05"],
        "links_expected_p95": links["p95"],
        "ratio_observed_to_expected": links["ratio"],
        "shuffles_ge_observed": links["shuffles_ge_observed"],
        "shuffles": null_facts["shuffles"],
        "seed": null_facts["seed"],
        "scope": null_facts["scope"],
        "adopters_observed": adopt["observed"],
        "adopters_expected_mean": adopt["mean"],
        "linked_share_observed": share["observed"],
        "linked_share_expected_mean": share["mean"],
        "linked_share_ratio": share["ratio"],
        "trees_beating_every_shuffle": null_facts["trees_beating_every_shuffle"],
        "chance_beats_upper_bound": null_facts["chance_beats_upper_bound"],
    }


def _speed_slot(speed):
    """Fact-sheet slot F7 from the "scores.speed" section."""
    to10 = speed["seed_to_10th"]
    to3 = speed["seed_to_3rd"]
    fastest = to10["fastest"] or {}
    return {
        "fastest_minutes_to_10th": fastest.get("minutes"),
        "fastest_token_id": fastest.get("token_id"),
        "fastest_cls": fastest.get("cls"),
        "fastest_seed_branch": fastest.get("seed_branch"),
        "trees_reaching_10_in_seed_branch": to10["trees"],
        "median_minutes_to_10th": to10["median_minutes"],
        "trees_reaching_3_in_seed_branch": to3["trees"],
        "median_minutes_to_3rd": to3["median_minutes"],
        "fastest_minutes_to_3rd": (to3["fastest"] or {}).get("minutes"),
    }


def build_facts(result, records, files, params, trees_meta, code, started, total_seconds,
                scores=None, risky=None, outputs=None):
    """The facts.json object (counts, plus agent labels in the influence table)."""
    ts = result.token_stats
    cs = result.tree_stats
    trees = _trees_table(records)
    if risky is None:
        risky = S.risky_table(records, result.tokens, result.null)
    if scores is None:
        scores = S.build_scores(result, records)
    null_facts = (N.facts_section(result.null, cs["links_named"]) if result.null is not None
                  else None)
    with_seed = trees["with_effective_seed"]
    shuffles = result.null.shuffles if result.null is not None else 0
    facts = {
        "tool": "patient-zero",
        "facts_version": FACTS_VERSION,
        "stage": "J7 full run: load, tokens, spread trees, shuffle baseline, scores, report",
        "input": dict({"adapter": params["adapter"], "files": files}, **params.get("input_note", {})),
        "params": {
            "min_agents": params["min_agents"],
            "max_phrase_share": params["max_phrase_share"],
            "given_texts": params["given_texts"],
            "lead": params.get("lead"),
            "shuffles": shuffles,
            "seed": result.null.seed if result.null is not None else None,
            "shuffle_scope": result.null.scope if result.null is not None else None,
            "randomness": ("label shuffle only: random.Random(seed), %d shuffles" % shuffles
                           if shuffles else "none (no shuffles)"),
        },
        "load": safe_keys(result.load_stats),
        "tokens": safe_keys(ts),
        "cascade": safe_keys(cs),
        "trees": trees,
        "families": _family_table(result),
        "groups": _group_table(result),
        "risky": risky,
        "outputs": outputs if outputs is not None else {"trees.jsonl": trees_meta},
        "nullmodel": null_facts,
        "scores": scores,
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
        "F5": _null_slot(null_facts) if null_facts is not None else None,
        "F7": _speed_slot(scores["speed"]),
        "F12": {"trees_with_type": risky["trees_with_type"],
                "trees_with_narrow_type": risky["trees_with_narrow_type"],
                "narrow_named_links_vs_shuffle": risky.get("named_links_vs_shuffle", {}).get("narrow")},
        "F14": {"runtime_seconds": round(total_seconds, 1), "code_lines": code["code_lines"],
                "test_functions": code["test_functions"]},
        "pending": {
            "F1": "lead finding: picked from the trees, shuffle baseline and scores (J10)",
            "F2": "lead tree page family: with F1",
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


def write_outputs(out_dir, result, params, started_clock, started_utc, report=True):
    """Write trees.jsonl, top_trees.json, facts.json and report.html to out_dir. Returns the facts object.

    ``params["lead"]`` (a token_id or None) picks the report's lead tree.
    """
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    records = tree_records(result)
    lines = "".join(json.dumps(r, ensure_ascii=True, separators=(",", ":")) + "\n" for r in records)
    trees_path = os.path.join(out_dir, "trees.jsonl")
    _write_atomic(trees_path, lines)
    trees_meta = {"lines": len(records), "sha256": hashlib.sha256(lines.encode("ascii")).hexdigest()}
    detail = R.detail_set(result, records, lead=params.get("lead"))
    detail_text = R.detail_text(detail)
    _write_atomic(os.path.join(out_dir, "top_trees.json"), detail_text)
    outputs = {
        "trees.jsonl": trees_meta,
        "top_trees.json": {"trees": len(detail["trees"]), "lead": detail["lead"],
                           "lead_rule": detail["lead_rule"], "scale": detail["scale"],
                           "sha256": hashlib.sha256(detail_text.encode("utf-8")).hexdigest()},
    }
    result.seconds["records"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = S.build_scores(result, records)
    risky = S.risky_table(records, result.tokens, result.null)
    result.seconds["scores"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    files = input_files(params["adapter"], params["input"])
    code = code_info()
    result.seconds["hash_and_code_info"] = time.perf_counter() - t0
    total = time.perf_counter() - started_clock
    facts = build_facts(result, records, files, params, trees_meta, code, started_utc, total,
                        scores=scores, risky=risky, outputs=outputs)
    _write_atomic(os.path.join(out_dir, "facts.json"), json.dumps(facts, indent=1, ensure_ascii=True) + "\n")
    if report:
        R.write_report(out_dir, facts, records, detail)
    return facts


# ---------------------------------------------------------------- command line


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(
        prog="cli.py",
        description="Run patient-zero end to end and write facts.json, trees.jsonl, top_trees.json and "
                    "report.html (counts and summary fields only; no row text).")
    ap.add_argument("--adapter", choices=sorted(ADAPTERS), required=True,
                    help="wiki: the collusion.wiki export folder; generic: one JSON Lines, CSV or TSV "
                         "log (see examples/fake_log.ndjson and examples/fake_log.csv)")
    ap.add_argument("--input", required=True,
                    help="export folder (wiki) or a JSON Lines, CSV or TSV file (generic)")
    ap.add_argument("--map", help=MAP_HELP)
    ap.add_argument("--out", required=True, help="output folder: outside the repo, or the repo's git-ignored out/")
    ap.add_argument("--min-agents", type=int, default=MIN_AGENTS,
                    help="keep tokens used by at least this many agents (default %d)" % MIN_AGENTS)
    ap.add_argument("--max-phrase-share", type=float, default=MAX_PHRASE_SHARE,
                    help="drop 3-word phrases found in more than this share of agent rows, as templates "
                         "(default %s; see patient_zero/tokens.py)" % MAX_PHRASE_SHARE)
    ap.add_argument("--given", action="append", default=[],
                    help="a shared goal or prompt text file; its tokens are dropped (repeatable)")
    ap.add_argument("--shuffles", type=int, default=N.DEFAULT_SHUFFLES,
                    help="label shuffles for the chance baseline (default %d; 0 skips it)"
                    % N.DEFAULT_SHUFFLES)
    ap.add_argument("--seed", type=int, default=N.DEFAULT_SEED,
                    help="seed for the shuffles (default %d)" % N.DEFAULT_SEED)
    ap.add_argument("--shuffle-scope", choices=N.SCOPES, default=N.SCOPE_DAY,
                    help="permute labels within each UTC day (default) or within each place and day")
    ap.add_argument("--lead", help="token_id of the report's lead tree (default: see patient_zero/report.py)")
    args = ap.parse_args(argv)
    started_clock = time.perf_counter()
    started_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        check_out_dir(args.out)
        check_out_paths(args.out, [args.input] + list(args.given))
        field_map = parse_map(args.map)
        given = []
        for path in args.given:
            with open(path, "r", encoding="utf-8") as fh:
                given.append(fh.read())
        result = run(args.adapter, args.input, min_agents=args.min_agents,
                     max_phrase_share=args.max_phrase_share, given_texts=given,
                     shuffles=args.shuffles, seed=args.seed, scope=args.shuffle_scope, field_map=field_map)
        params = {"adapter": args.adapter, "input": args.input, "min_agents": args.min_agents,
                  "max_phrase_share": args.max_phrase_share, "given_texts": len(given), "lead": args.lead,
                  "input_note": input_note(args.adapter, args.input, field_map)}
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
    f5 = s["F5"]
    if f5 and f5["shuffles"]:
        print("shuffle (%s, %d x, seed %d): expected named links %.1f (sd %.1f), observed / expected %s; "
              "shuffles reaching the observed count %d" % (
                  f5["scope"], f5["shuffles"], f5["seed"], f5["links_expected_mean"],
                  f5["links_expected_sd"], f5["ratio_observed_to_expected"], f5["shuffles_ge_observed"]),
              file=out)
        word, _sentence = R.chance_verdict(f5)
        print("against the shuffle: %s (observed %s the shuffle's 5th to 95th percentile, %.1f to %.1f)" % (
            word, {"above chance": "over", "below chance": "under"}.get(word, "inside"),
            f5["links_expected_p05"], f5["links_expected_p95"]), file=out)
    else:
        print("against the shuffle: no shuffle in this run, so no chance baseline", file=out)
    f7 = s["F7"]
    if f7["trees_reaching_10_in_seed_branch"]:
        print("speed: %d trees reach a 10th agent in the seed's branch; fastest %s min, median %s min" % (
            f7["trees_reaching_10_in_seed_branch"], f7["fastest_minutes_to_10th"], f7["median_minutes_to_10th"]),
            file=out)
    else:
        print("speed: no tree reaches a 10th agent in the seed's branch", file=out)
    print("runtime: %.1f s (target under %d s)" % (facts["runtime"]["total_seconds"], TARGET_SECONDS),
          file=out)
    print("wrote: %s" % ", ".join(os.path.join(args.out, n) for n in (
        "facts.json", "trees.jsonl", "top_trees.json", "report.html")), file=out)
    top = facts["outputs"]["top_trees.json"]
    print("lead tree: %s (%s)" % (top["lead"], top["lead_rule"]), file=out)
    print("core_sha256: %s" % facts["core_sha256"], file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
