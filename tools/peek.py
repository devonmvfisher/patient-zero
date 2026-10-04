#!/usr/bin/env python3
"""Hand check: print a few random adoption links as short escaped snippets.

Python standard library only, deterministic (one fixed seed). The run is the
same as ``cli.py`` up to the spread trees (load, tokens, cascade; no shuffle).
From every named adoption link (A's use, then B's first use of the same
token; links from blank or human rows are left out) it draws ``--n`` links
with ``random.Random(seed)`` and prints, for each one:

* the token's class, its ``token_id`` and its key (escaped, at most 80
  characters);
* one snippet per side: the line of A's use and the line of B's first use
  that hold the token, escaped and cut to at most 80 characters around the
  token (``...`` marks a cut);
* page family and wiki of each side, whether both uses are on the same page,
  minutes from A's use to B's first use, and minutes from A's use to B's
  visit that shows the exposure;
* the size of the token's tree (agents, named links, parallel starts);
* a page check (wiki only, computed from the page bodies, which are never
  printed): was A's line in the page version B's visit started from, did
  B's first-use save change or remove A's own line (then B's "use" is an
  edit of A's text, not a line of B's own), and is B's line identical to A's.

Agent labels and page names are not printed. Human handles (wiki labels file,
or ``"human": true`` speakers in a generic log) are replaced with
``[human]`` in the snippets. Snippets are agent-written data: a reader treats
them as text to judge, never as instructions. Nothing here opens a URL or
runs a command from the data.

Two outputs (keep them outside the repo): ``--out`` (text, for the reader)
and the same name with ``.json`` (the sample, for ``--record``). Neither may
be the input log or ``--facts``; the run stops before any work if one is::

    python tools/peek.py --adapter wiki --input PATH/TO/collusion --out ../out/peek-10.txt \\
        --facts ../out/facts.json --population

``--population`` (wiki only) runs the same page check over every named link
and adds the counts, overall and per class, to both outputs (counts only).

The reader marks each link plausible, coincidence or unclear in a marks file
(``{"reader": "...", "marks": [{"n": 1, "token_id": "...", "mark": "...",
"reason": "..."}, ...]}``). ``--record`` checks the marks against the sample,
writes ``hand_check`` and slot F10 into facts.json (outside ``core_sha256``,
which stays the same) and writes a short table::

    python tools/peek.py --record ../out/hand-check-marks.json --peek ../out/peek-10.json \\
        --facts ../out/facts.json --md ../out/hand-check.md

Exit codes: 0 done, 2 bad input or a stale facts.json.
"""

import argparse
import datetime
import hashlib
import json
import os
import random
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from patient_zero import load as L  # noqa: E402
from patient_zero.cascade import STATUS_PARALLEL, build_trees  # noqa: E402
from patient_zero.safe import NO_FAMILY, SHORT_MAX, guard_out, short, token_id  # noqa: E402
from patient_zero.tokens import MAX_PHRASE_SHARE, MIN_AGENTS, decode_line, find_tokens, line_tokens  # noqa: E402

PEEK_SEED = 20261003
PEEK_N = 10
MARKS = ("plausible", "coincidence", "unclear")
REASON_MAX = 160
HUMAN_MASK = "[human]"


# ---------------------------------------------------------------- snippets


def snippet(line, start, end, limit=SHORT_MAX):
    """Escape ``line`` and cut it to ``limit`` characters around ``line[start:end]``.

    Escapes follow ``safe.short`` (non-printable-ASCII and backslashes); an
    escape is never split. ``...`` at either end marks a cut, and counts
    toward ``limit``.
    """
    pieces = [short(ch) for ch in line]
    if sum(len(p) for p in pieces) <= limit:
        return "".join(pieces)
    budget = limit - 6  # room for "..." on both sides
    start = max(0, min(start, len(pieces)))
    end = max(start, min(end, len(pieces)))
    i = j = start
    size = 0
    while j < end and size + len(pieces[j]) <= budget:  # the token first
        size += len(pieces[j])
        j += 1
    grew = True
    while grew:  # then context, one character each side in turn
        grew = False
        if i > 0 and size + len(pieces[i - 1]) <= budget:
            i -= 1
            size += len(pieces[i])
            grew = True
        if j < len(pieces) and size + len(pieces[j]) <= budget:
            size += len(pieces[j])
            j += 1
            grew = True
    return ("..." if i > 0 else "") + "".join(pieces[i:j]) + ("..." if j < len(pieces) else "")


def find_span(line, token):
    """(start, end) of the token in ``line``, or None when it cannot be placed."""
    value = token.value
    if token.cls == "phrase":  # words were letters only and case-folded
        words = value.split(" ")
        pattern = (r"(?<![^\W\d_])" + r"[\W\d_]+".join(re.escape(w) for w in words)
                   + r"(?![^\W\d_])")
        m = re.search(pattern, line, re.IGNORECASE)
    elif token.cls == "command":
        verb = value.split(" ")[0]
        m = re.search(r"(?<![\w-])" + re.escape(verb) + r"(?![\w-])", line)
    elif token.cls == "code" or token.kind == "control":
        m = re.search(re.escape(value), line) or re.search(re.escape(value), line, re.IGNORECASE)
    else:  # host, file, prefix, mailbox (lower-cased when found)
        m = re.search(re.escape(value), line, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def mask_humans(line, humans):
    """Replace whole-word human handles (any case) with [human]. Returns (line, count)."""
    count = 0
    for handle in humans:
        line, n = re.subn(r"(?<!\w)" + re.escape(handle) + r"(?!\w)", HUMAN_MASK, line,
                          flags=re.IGNORECASE)
        count += n
    return line, count


def token_line(text, token):
    """(raw line, prepared line) of the first line of ``text`` that holds the token.

    Lines are prepared as ``tokens.row_tokens`` prepares them (percent-escapes
    decoded when present), so the line found is the one the token came from.
    The raw line (stripped) is what the page body holds. (None, None) when no
    line holds it.
    """
    for raw in text.split("\n"):
        if not raw.strip():
            continue
        line = decode_line(raw) if "%" in raw else raw
        found, _stopped = line_tokens(line)
        if any(key == token.key for key, _c, _k, _v in found):
            return raw.strip(), line
    return None, None


def locate(text, token, humans=()):
    """The line of ``text`` that holds the token, as a snippet around the token.

    Returns (snippet, placed, humans_masked); ``placed`` is False when the
    token's position could not be found and the line start is shown instead.
    """
    _raw, line = token_line(text, token)
    if line is None:
        return "(token not found in this row)", False, 0
    line, masked = mask_humans(line, humans)
    line = line.strip()
    span = find_span(line, token)
    if span is None:
        return snippet(line, 0, 0), False, masked
    return snippet(line, span[0], span[1]), True, masked


def wiki_saves(path, rev_ids):
    """{rev_id: (diff_base, body lines)} for ``rev_ids`` and their diff bases (wiki only).

    Bodies are read by this code only, as sets of stripped lines; nothing from
    them is printed. Two passes over revisions.jsonl.gz: bases first, then bodies.
    """
    rev_path = os.path.join(path, "revisions.jsonl.gz")
    keys = frozenset(("rev_id", "diff_base", "body"))
    want = set(rev_ids)
    bases = {}
    for _n, s in L._read_json_lines(rev_path, keys):
        rid = str(s.get("rev_id"))
        if rid in want:
            bases[rid] = None if s.get("diff_base") is None else str(s.get("diff_base"))
    want |= {b for b in bases.values() if b is not None}
    out = {}
    for _n, s in L._read_json_lines(rev_path, keys):
        rid = str(s.get("rev_id"))
        if rid in want:
            base = None if s.get("diff_base") is None else str(s.get("diff_base"))
            out[rid] = (base, frozenset(x.strip() for x in (s.get("body") or "").split("\n")))
    return out


def page_check(link, rows, a_line, saves):
    """Two page facts for one link, from the wiki bodies (None when unknown).

    ``a_line_on_page_at_visit``: A's token line was in the page version B's
    visit save started from (B's editor showed it). ``b_save_changed_a_line``:
    B's first use is on A's page, A's line was in the version that save
    started from and is gone after it, so B's first use is an edit of A's own
    line (or its removal), not a line of B's own.
    """
    if not saves or a_line is None:
        return None, None
    visit = saves.get(rows[link.visit_row].row_id)
    on_page = None
    if visit is not None:
        base = saves.get(visit[0]) if visit[0] is not None else None
        on_page = bool(base and a_line in base[1])
    changed = None
    target = saves.get(rows[link.target_row].row_id)
    if target is not None:
        if rows[link.target_row].place != rows[link.source_row].place:
            changed = False
        else:
            base = saves.get(target[0]) if target[0] is not None else None
            changed = bool(base and a_line in base[1] and a_line not in target[1])
    return on_page, changed


def human_handles(adapter, path, field_map=None):
    """Human handles the loader hides, so snippets can mask them too."""
    humans = set()
    if adapter == "wiki":
        label_path = os.path.join(path, "labels.jsonl.gz")
        if os.path.isfile(label_path):
            for _n, lab in L._read_json_lines(label_path, L._LABEL_KEYS):
                if lab.get("is_human_handle") is True and isinstance(lab.get("label"), str):
                    if lab["label"].strip():
                        humans.add(lab["label"].strip())
    else:  # read with the same columns and codec as the loader (--map, CSV, BOM), so every hidden handle is masked
        humans = L.generic_human_labels(path, field_map)
    return sorted(humans, key=lambda h: (-len(h), h))  # longest first


# ---------------------------------------------------------------- the sample


def named_links(trees):
    """[(token key, Link)] for every named link, in token-key then adoption order."""
    return [(key, link) for key in sorted(trees) for link in trees[key].links if link.source is not None]


def _minutes(seconds):
    return round(seconds / 60.0, 2)


def link_row_ids(rows, links):
    """Row ids of every target and visit row of ``links`` ([(key, Link)])."""
    ids = set()
    for _key, link in links:
        ids.update((rows[link.target_row].row_id, rows[link.visit_row].row_id))
    return sorted(ids)


POPULATION_FIELDS = ("links", "a_line_found", "a_line_on_page_at_visit", "b_save_changed_a_line",
                     "same_line", "on_page_not_changed_not_identical")


def population_check(rows, tokens, trees, saves):
    """The page check over every named link: counts only, overall and per class.

    ``on_page_not_changed_not_identical`` is the mechanical upper bound on
    links that look like B adding a line of its own after A's line was in
    front of it. Lines are matched exactly, so a line someone else edited in
    between counts as not on the page.
    """
    lines = {}

    def line_of(idx, key):
        if (idx, key) not in lines:
            lines[(idx, key)] = token_line(rows[idx].text, tokens[key])[0]
        return lines[(idx, key)]

    total = dict.fromkeys(POPULATION_FIELDS, 0)
    by_class = {}
    for key, link in named_links(trees):
        counts = by_class.setdefault(trees[key].cls, dict.fromkeys(POPULATION_FIELDS, 0))
        a_line, b_line = line_of(link.source_row, key), line_of(link.target_row, key)
        on_page, changed = page_check(link, rows, a_line, saves)
        same = a_line is not None and a_line == b_line
        for c in (total, counts):
            c["links"] += 1
            c["a_line_found"] += a_line is not None
            c["a_line_on_page_at_visit"] += bool(on_page)
            c["b_save_changed_a_line"] += bool(changed)
            c["same_line"] += same
            c["on_page_not_changed_not_identical"] += bool(on_page and not changed and not same)
    return {"total": total, "by_class": dict(sorted(by_class.items()))}


def sample(rows, tokens, trees, n=PEEK_N, seed=PEEK_SEED, humans=(), bodies=None):
    """Draw ``n`` named links with ``random.Random(seed)``. Returns (items, population).

    ``bodies`` (optional) maps a list of row ids to ``wiki_saves`` output, for
    the page checks; without it those fields are None.
    """
    links = named_links(trees)
    rng = random.Random(seed)
    picks = rng.sample(range(len(links)), min(n, len(links)))
    saves = {}
    if bodies is not None and picks:
        saves = bodies(link_row_ids(rows, [links[idx] for idx in picks]))
    items = []
    for number, idx in enumerate(picks, 1):
        key, link = links[idx]
        token, tree = tokens[key], trees[key]
        src, dst, visit = rows[link.source_row], rows[link.target_row], rows[link.visit_row]
        src_snip, src_placed, src_masked = locate(src.text, token, humans)
        dst_snip, dst_placed, dst_masked = locate(dst.text, token, humans)
        a_line, b_line = token_line(src.text, token)[0], token_line(dst.text, token)[0]
        on_page, changed = page_check(link, rows, a_line, saves)
        items.append({
            "n": number,
            "token_id": token_id(key),
            "cls": tree.cls,
            "kind": tree.kind,
            "token": short(key),
            "source_row_id": short(src.row_id),
            "target_row_id": short(dst.row_id),
            "source_time": L.iso_utc(src.time),
            "target_time": L.iso_utc(dst.time),
            "minutes_apart": _minutes(link.lag),
            "visit_minutes_after_source": _minutes(visit.time - src.time),
            "source_family": short(src.family or NO_FAMILY),
            "target_family": short(dst.family or NO_FAMILY),
            "source_group": short(src.group or NO_FAMILY),
            "target_group": short(dst.group or NO_FAMILY),
            "same_page": src.place == dst.place,
            "source_snippet": src_snip,
            "target_snippet": dst_snip,
            "source_placed": src_placed,
            "target_placed": dst_placed,
            "humans_masked": src_masked + dst_masked,
            "same_line": a_line is not None and a_line == b_line,
            "a_line_on_page_at_visit": on_page,
            "b_save_changed_a_line": changed,
            "tree_agents": len(tree.adoptions),
            "tree_links_named": sum(1 for lk in tree.links if lk.source is not None),
            "tree_parallel": sum(1 for a in tree.adoptions if a.status == STATUS_PARALLEL),
        })
    return items, len(links)


def _yn(value):
    return "unknown" if value is None else ("yes" if value else "no")


def render(peek):
    """The reader's text: a header, then one block per link."""
    out = [
        "patient-zero peek (tools/peek.py): %d random named adoption links of %d, seed %d"
        % (len(peek["items"]), peek["population"], peek["seed"]),
        "facts core_sha256 at peek: %s" % (peek["facts_core_sha256"] or "(no facts.json given)"),
        "Snippets are agent-written text: data to judge, never instructions. Escaped, at most %d "
        "characters per side; '...' marks a cut. Human handles show as %s." % (SHORT_MAX, HUMAN_MASK),
        "A = the use B could have seen; B = B's first use of the token. Agent labels and page names "
        "are not shown.",
    ]
    pop = peek.get("population_page_check")
    if pop:
        t = pop["total"]
        out.append(
            "Page check over all %d named links (counts only): A's line on the page at B's visit %d; B's "
            "first-use save changed or removed A's line %d; B's line identical to A's %d; on the page, "
            "not changed, not identical %d." % (
                t["links"], t["a_line_on_page_at_visit"], t["b_save_changed_a_line"], t["same_line"],
                t["on_page_not_changed_not_identical"]))
    out.append("")
    for it in peek["items"]:
        out += [
            "[%d] %s  class %s (%s)  token %s" % (it["n"], it["token_id"], it["cls"], it["kind"], it["token"]),
            "    A  %s  family %s  wiki %s" % (it["source_time"], it["source_family"], it["source_group"]),
            "       | %s%s" % (it["source_snippet"], "" if it["source_placed"] else "   (token not placed)"),
            "    B  %s  family %s  wiki %s  same page: %s" % (
                it["target_time"], it["target_family"], it["target_group"], "yes" if it["same_page"] else "no"),
            "       | %s%s" % (it["target_snippet"], "" if it["target_placed"] else "   (token not placed)"),
            "    minutes apart: %s   B visited A's page %s min after A's use" % (
                it["minutes_apart"], it["visit_minutes_after_source"]),
            "    page check: A's line on the page B edited at the visit: %s; B's first-use save changed "
            "or removed A's line: %s; B's line identical to A's: %s" % (
                _yn(it["a_line_on_page_at_visit"]), _yn(it["b_save_changed_a_line"]), _yn(it["same_line"])),
            "    tree: %d agents, %d named links, %d parallel starts" % (
                it["tree_agents"], it["tree_links_named"], it["tree_parallel"]),
            "",
        ]
    return "\n".join(out)


def _write(path, text):
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _inside_repo(path):
    path = os.path.normcase(os.path.realpath(path))
    repo = os.path.normcase(os.path.realpath(REPO))
    try:
        return os.path.commonpath([path, repo]) == repo
    except ValueError:
        return False


def twin_path(out_path):
    """The .json twin peek writes next to ``out_path`` (same name, .json)."""
    return os.path.splitext(out_path)[0] + ".json"


def guard_peek_out(out_path, inputs):
    """Refuse an --out (or its .json twin) that is an input; refuse an --out ending in .json."""
    twin = twin_path(out_path)
    if os.path.normcase(os.path.abspath(twin)) == os.path.normcase(os.path.abspath(out_path)):
        raise ValueError("--out names %s; it must not end in .json, because the .json twin is "
                         "written under that name. Use .txt." % out_path)
    guard_out(out_path, inputs, "--out")
    guard_out(twin, inputs, "the .json twin of --out")


def peek(adapter, input_path, out_path, facts_path=None, n=PEEK_N, seed=PEEK_SEED,
         min_agents=MIN_AGENTS, max_phrase_share=MAX_PHRASE_SHARE, population=False, field_map=None):
    """Run load, tokens and cascade, draw the sample, write ``out_path`` and its .json twin.

    ``population`` (wiki only) also runs the page check over every named link
    (counts only) and adds it to both outputs. Refuses, before any work, an
    ``out_path`` or twin that is the input log or ``facts_path``.
    """
    guard_peek_out(out_path, [input_path, facts_path])
    facts = None
    if facts_path:
        with open(facts_path, "r", encoding="utf-8") as fh:
            facts = json.load(fh)
    rows, _load_stats = L.load(adapter, input_path, field_map)
    tokens, token_stats = find_tokens(rows, min_agents=min_agents, max_phrase_share=max_phrase_share)
    trees, tree_stats = build_trees(rows, tokens)
    if facts is not None:  # the sample must come from the run facts.json describes
        want = (facts["tokens"]["kept_total"], facts["cascade"]["links_named"])
        have = (token_stats["kept_total"], tree_stats["links_named"])
        if want != have:
            raise ValueError("this run (tokens %d, named links %d) differs from facts.json "
                             "(tokens %d, named links %d); rerun cli.py first" % (have + want))
    bodies = (lambda ids: wiki_saves(input_path, ids)) if adapter == "wiki" else None
    page_counts = None
    if population and bodies is not None:
        all_saves = bodies(link_row_ids(rows, named_links(trees)))
        page_counts = population_check(rows, tokens, trees, all_saves)
        bodies = lambda _ids: all_saves  # noqa: E731 (one read serves both)
    items, population_size = sample(rows, tokens, trees, n=n, seed=seed,
                                    humans=human_handles(adapter, input_path, field_map), bodies=bodies)
    result = {
        "tool": "patient-zero peek",
        "seed": seed,
        "n": len(items),
        "population": population_size,
        "population_rule": "named adoption links (source is an agent), token-key then adoption order",
        "params": {"adapter": adapter, "min_agents": min_agents, "max_phrase_share": max_phrase_share},
        "facts_core_sha256": facts["core_sha256"] if facts is not None else None,
        "population_page_check": page_counts,
        "items": items,
    }
    text = render(result)
    _write(out_path, text + "\n")
    result["text_sha256"] = hashlib.sha256((text + "\n").encode("utf-8")).hexdigest()
    _write(twin_path(out_path), json.dumps(result, indent=1, ensure_ascii=True) + "\n")
    return result


# ---------------------------------------------------------------- recording marks


def check_marks(peek_obj, marks_obj):
    """Validate the reader's marks against the sample. Returns the marks list."""
    marks = marks_obj.get("marks")
    items = peek_obj["items"]
    if not isinstance(marks, list) or len(marks) != len(items):
        raise ValueError("need exactly %d marks" % len(items))
    for item, mark in zip(items, marks):
        if mark.get("n") != item["n"] or mark.get("token_id") != item["token_id"]:
            raise ValueError("mark %r does not match sample item %d (%s)" % (mark.get("n"), item["n"],
                                                                            item["token_id"]))
        if mark.get("mark") not in MARKS:
            raise ValueError("mark %d: %r is not one of %s" % (item["n"], mark.get("mark"), ", ".join(MARKS)))
        reason = mark.get("reason")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > REASON_MAX \
                or short(reason, REASON_MAX + 10) != reason:
            raise ValueError("mark %d: reason must be printable ASCII, 1 to %d characters"
                             % (item["n"], REASON_MAX))
    return marks


def hand_check_section(peek_obj, marks_obj, marks):
    counts = {m: sum(1 for x in marks if x["mark"] == m) for m in MARKS}
    n = len(marks)
    return {
        "precision": "%d of %d" % (counts["plausible"], n),
        "plausible": counts["plausible"],
        "coincidence": counts["coincidence"],
        "unclear": counts["unclear"],
        "sample": n,
        "seed": peek_obj["seed"],
        "population": peek_obj["population"],
        "population_rule": peek_obj["population_rule"],
        "reader": short(marks_obj.get("reader") or "unnamed reader", 120),
        "tool": "tools/peek.py",
        "peek_text_sha256": peek_obj.get("text_sha256"),
        "facts_core_sha256_at_check": peek_obj["facts_core_sha256"],
        "population_page_check": peek_obj.get("population_page_check"),
        "page_check_counts": {
            "a_line_on_page_at_visit": sum(1 for it in peek_obj["items"] if it.get("a_line_on_page_at_visit")),
            "b_save_changed_a_line": sum(1 for it in peek_obj["items"] if it.get("b_save_changed_a_line")),
            "same_line": sum(1 for it in peek_obj["items"] if it.get("same_line")),
        },
        "items": [{"n": it["n"], "token_id": it["token_id"], "cls": it["cls"],
                   "minutes_apart": it["minutes_apart"], "same_page": it["same_page"],
                   "a_line_on_page_at_visit": it.get("a_line_on_page_at_visit"),
                   "b_save_changed_a_line": it.get("b_save_changed_a_line"),
                   "same_line": it.get("same_line"),
                   "mark": mk["mark"], "reason": mk["reason"]}
                  for it, mk in zip(peek_obj["items"], marks)],
    }


def render_md(section, peek_obj):
    lines = [
        "# Hand check (J9): precision %s" % section["precision"],
        "",
        "Plausible %d, coincidence %d, unclear %d. Sample: %d named adoption links drawn with seed %d "
        "from %d (`tools/peek.py`). Reader: %s." % (
            section["plausible"], section["coincidence"], section["unclear"], section["sample"],
            section["seed"], section["population"], section["reader"]),
        "",
        "Marks: **plausible** = B's use reads as picked up from A's (same distinctive wording, "
        "context or page); **coincidence** = not an adoption: generic, template or source wording "
        "both could write on their own, or B's save only edits or puts back A's own line; "
        "**unclear** = the snippets cannot tell.",
        "",
        "Snippets (agent-written text, escaped, at most 80 characters per side) are in "
        "`peek-10.txt` next to this file; this table carries no snippet text.",
        "",
        "Page check (from the page bodies, by code): **on page** = A's line was in the page version "
        "B's visit started from; **B edited A** = B's first-use save changed or removed A's own line; "
        "**same** = B's line is identical to A's.",
        "",
        "| # | token_id | class | family A -> B | min apart | same page | on page | B edited A | same "
        "| mark | reason |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    by_n = {it["n"]: it for it in peek_obj["items"]}
    for row in section["items"]:
        it = by_n[row["n"]]
        fam = it["source_family"] if it["source_family"] == it["target_family"] else \
            "%s -> %s" % (it["source_family"], it["target_family"])
        lines.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row["n"], row["token_id"], row["cls"], fam.replace("|", "\\|"), row["minutes_apart"],
            _yn(row["same_page"]), _yn(row["a_line_on_page_at_visit"]), _yn(row["b_save_changed_a_line"]),
            _yn(row["same_line"]), row["mark"], row["reason"].replace("|", "\\|")))
    pc = section["page_check_counts"]
    lines += [
        "",
        "Page check counts: A's line on the page at the visit %d of %d; B's first-use save changed or "
        "removed A's line %d; B's line identical to A's %d." % (
            pc["a_line_on_page_at_visit"], section["sample"], pc["b_save_changed_a_line"], pc["same_line"]),
    ]
    pop = section.get("population_page_check")
    if pop:
        lines += [
            "",
            "The same page check over every named link, by code (counts only; lines matched exactly, so a "
            "line someone edited in between counts as not on the page):",
            "",
            "| class | links | A's line on page at visit | B edited A's line | B's line identical "
            "| on page, not edited, not identical |",
            "|---|---|---|---|---|---|",
        ]
        for name, c in [("all", pop["total"])] + list(pop["by_class"].items()):
            lines.append("| %s | %d | %d | %d | %d | %d |" % (
                name, c["links"], c["a_line_on_page_at_visit"], c["b_save_changed_a_line"], c["same_line"],
                c["on_page_not_changed_not_identical"]))
    lines += [
        "",
        "Limits: one reader, 10 links, judged from one line per side. A plausible mark says the "
        "wording fits copying; it does not prove B read A.",
        "",
        "facts core_sha256 at check: `%s`. Recorded %s." % (
            section["facts_core_sha256_at_check"],
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        "",
    ]
    return "\n".join(lines)


def record(marks_path, peek_path, facts_path, md_path):
    """Write hand_check and slot F10 into facts.json and the table to ``md_path``.

    ``md_path`` may not be the marks file, the peek .json or facts.json.
    """
    guard_out(md_path, [marks_path, peek_path, facts_path], "--md")
    with open(peek_path, "r", encoding="utf-8") as fh:
        peek_obj = json.load(fh)
    with open(marks_path, "r", encoding="utf-8-sig") as fh:  # hand-edited: a byte-order mark is fine
        marks_obj = json.load(fh)
    with open(facts_path, "r", encoding="utf-8") as fh:
        facts = json.load(fh)
    if peek_obj.get("facts_core_sha256") != facts.get("core_sha256"):
        raise ValueError("facts.json core_sha256 differs from the one at peek time; rerun peek.py")
    marks = check_marks(peek_obj, marks_obj)
    section = hand_check_section(peek_obj, marks_obj, marks)
    facts["hand_check"] = section
    slots = facts.setdefault("slots", {})
    slots["F10"] = {"precision": section["precision"], "plausible": section["plausible"],
                    "coincidence": section["coincidence"], "unclear": section["unclear"],
                    "sample": section["sample"], "seed": section["seed"]}
    slots.get("pending", {}).pop("F10", None)
    _write(facts_path, json.dumps(facts, indent=1, ensure_ascii=True) + "\n")
    _write(md_path, render_md(section, peek_obj))
    return section


# ---------------------------------------------------------------- command line


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="Print random adoption links as short escaped snippets, "
                                             "or record a reader's marks (--record).")
    ap.add_argument("--adapter", choices=sorted(L.ADAPTERS))
    ap.add_argument("--input", help="export folder (wiki) or a JSON Lines, CSV or TSV file (generic)")
    ap.add_argument("--map", help=L.MAP_HELP)
    ap.add_argument("--out", help="text file for the reader, outside the repo (a .json twin is written too)")
    ap.add_argument("--facts", help="facts.json of the same run (checked; written by --record)")
    ap.add_argument("--n", type=int, default=PEEK_N)
    ap.add_argument("--seed", type=int, default=PEEK_SEED)
    ap.add_argument("--min-agents", type=int, default=MIN_AGENTS)
    ap.add_argument("--max-phrase-share", type=float, default=MAX_PHRASE_SHARE)
    ap.add_argument("--population", action="store_true",
                    help="wiki only: also run the page check over every named link (counts only)")
    ap.add_argument("--record", metavar="MARKS", help="marks file to record into facts.json")
    ap.add_argument("--peek", help="with --record: the .json the peek wrote")
    ap.add_argument("--md", help="with --record: table to write")
    args = ap.parse_args(argv)
    try:
        if args.record:
            if not (args.peek and args.facts and args.md):
                raise ValueError("--record needs --peek, --facts and --md")
            for path in (args.facts, args.md):
                if _inside_repo(path):
                    raise ValueError("outputs must be outside the repo: %s" % path)
            section = record(args.record, args.peek, args.facts, args.md)
            print("hand check: precision %s (coincidence %d, unclear %d); wrote %s and %s" % (
                section["precision"], section["coincidence"], section["unclear"], args.facts, args.md),
                file=out)
            return 0
        if not (args.adapter and args.input and args.out):
            raise ValueError("need --adapter, --input and --out (or --record)")
        if _inside_repo(args.out):
            raise ValueError("--out must be outside the repo")
        result = peek(args.adapter, args.input, args.out, facts_path=args.facts, n=args.n, seed=args.seed,
                      min_agents=args.min_agents, max_phrase_share=args.max_phrase_share,
                      population=args.population, field_map=L.parse_map(args.map))
    except (OSError, ValueError, KeyError) as exc:
        print("peek: ERROR: %s" % exc, file=out)
        return 2
    print("peek: %d of %d named links (seed %d); wrote %s" % (
        result["n"], result["population"], result["seed"], args.out), file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
