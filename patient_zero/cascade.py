"""Who started each trick, who picked it up from whom: spread trees.

Python standard library only, deterministic (no randomness). Input is the
time-sorted rows from ``patient_zero.load`` and the tokens from
``patient_zero.tokens.find_tokens``. Output is one ``Tree`` per token.

Method
------
First use
    For each token, each agent's earliest row that holds it. Agents are
    ordered by first use (row order, which is time then row id).

Exposure per page
    An agent *visits* a place (wiki page, chat room) with every row it writes
    there, whether or not the row adds text. Agent B is exposed to a use of
    token X when that use sits on a place B visited after the use and no later
    than B's own first use of X. Uses by blank or human rows (exposure-only)
    can expose others; they are never credited as users.

Clock-aware links
    A use U (time ``t_u``, uncertainty ``u_u``) and a later row V are in clock
    order only when ``t_v - t_u > u_u + u_v``. A link from U to B counts only
    when B's first use is in clock order after U and at least one of B's
    visits to U's place is in clock order after U. Credit goes to the most
    recent earlier use B could have seen that passes this rule.

    Counted separately: ``dropped`` (B was exposed, but no exposing use passes
    the clock rule; B gets no parent) and ``shifted`` (the most recent exposing
    use failed the rule and credit moved to an older one).

Status of each adopter, in first-use order
    ``first``          the first user ("patient zero"), when not exposed to an
                       earlier exposure-only use
    ``linked``         parent is a named agent
    ``unnamed``        the exposing use was a blank or human row: B saw it,
                       but there is nobody to credit
    ``clock_dropped``  exposed, but no exposing use is in clock order
    ``parallel``       a parallel start: B began with no exposure to any
                       earlier use of the token

Spread tree, first user and effective seed
    Nodes are the token's agents; edges are the links. ``first_user`` is the
    earliest agent to use the token. ``effective_seed`` is the earliest agent
    with at least one adopter (a direct child), or ``None`` when the token has
    no links.

Limits (stated, not hidden): a visit counts as exposure even if the text was
deleted from the page before the visit; exposure-only rows have no identity,
so ``unnamed`` adoptions end a chain.

Usage::

    rows, _ = load("wiki", path)
    tokens, _ = find_tokens(rows)
    trees, stats = build_trees(rows, tokens)

    python -m patient_zero.cascade --adapter wiki --input PATH --summary out/cascade-summary.json

The command line and the summary file hold counts only, never token text,
agent names or row text.
"""

import argparse
import json
import os
import sys
import time
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from statistics import median
from typing import NamedTuple, Optional, Tuple

from patient_zero.load import ADAPTERS, load
from patient_zero.safe import guard_out
from patient_zero.tokens import MAX_PHRASE_SHARE, MIN_AGENTS, find_tokens, in_clock_order

STATUS_FIRST = "first"
STATUS_LINKED = "linked"
STATUS_UNNAMED = "unnamed"
STATUS_CLOCK = "clock_dropped"
STATUS_PARALLEL = "parallel"
STATUSES = (STATUS_FIRST, STATUS_LINKED, STATUS_UNNAMED, STATUS_CLOCK, STATUS_PARALLEL)


class Link(NamedTuple):
    """B picked up the token from the use in ``source_row``."""

    token: str
    source: Optional[str]  # agent of the source row; None when it is exposure-only
    source_row: int  # the use B could have seen (index into rows)
    target: str  # agent B
    target_row: int  # B's first use
    visit_row: int  # B's row on the source row's place that shows the exposure
    lag: float  # seconds from the source use to B's first use


class Adoption(NamedTuple):
    """One agent's first use of a token and where it came from."""

    agent: str
    row: int  # first use (index into rows)
    time: float
    status: str  # one of STATUSES
    parent: Optional[str]  # credited agent (status "linked" only)
    link: Optional[Link]  # set for "linked" and "unnamed"


class Tree(NamedTuple):
    """The spread tree of one token."""

    token: str
    cls: str
    kind: str
    adoptions: Tuple[Adoption, ...]  # first-use order
    links: Tuple[Link, ...]  # named and unnamed sources, in adoption order
    first_user: str
    effective_seed: Optional[str]
    parallel_starts: Tuple[str, ...]
    dropped: int  # adopters left without a parent by the clock rule
    shifted: int  # adopters whose credit moved to an older use by the clock rule
    depth: int  # longest chain of named links (a lone first user has depth 0)
    places: int  # distinct places holding a use (all roles)


# ---------------------------------------------------------------- exposure


def build_visits(rows):
    """Agent -> place -> sorted row indices of that agent's rows on that place.

    Exposure-only rows (blank, human) visit nothing: they have no identity.
    """
    visits = defaultdict(lambda: defaultdict(list))
    for idx, row in enumerate(rows):
        if not row.exposure_only:
            visits[row.agent][row.place].append(idx)  # rows are in order, so lists stay sorted
    return {agent: dict(places) for agent, places in visits.items()}


def first_uses(rows, use_rows):
    """[(agent, first-use row index), ...] in first-use order, named agents only."""
    seen = {}
    for idx in use_rows:
        row = rows[idx]
        if not row.exposure_only and row.agent not in seen:
            seen[row.agent] = idx
    return sorted(seen.items(), key=lambda kv: kv[1])


def _clock_visit(rows, use_idx, place_visits, last_idx):
    """B's latest visit in (use_idx, last_idx] that is in clock order after the use.

    Returns (exposed, visit index or None). ``exposed`` is True when any visit
    falls in the range at all (clock order or not).
    """
    lo = bisect_right(place_visits, use_idx)
    hi = bisect_right(place_visits, last_idx)
    if lo >= hi:
        return False, None
    u = rows[use_idx]
    for k in range(hi - 1, lo - 1, -1):
        v = rows[place_visits[k]]
        if in_clock_order(u.time, u.uncertainty, v.time, v.uncertainty):
            return True, place_visits[k]
    return True, None


def adopt(rows, use_rows, agent, first_idx, visits):
    """Classify one agent's first use. Returns (status, link or None, shifted).

    ``use_rows`` is the token's sorted row indices (all roles).
    """
    b = rows[first_idx]
    mine = visits.get(agent, {})
    exposed = False
    shifted = False
    stop = bisect_left(use_rows, first_idx)
    for pos in range(stop - 1, -1, -1):  # most recent earlier use first
        u_idx = use_rows[pos]
        u = rows[u_idx]
        if not u.exposure_only and u.agent == agent:
            continue  # cannot happen before a first use; kept as a guard
        place_visits = mine.get(u.place)
        if not place_visits:
            continue
        seen, visit_idx = _clock_visit(rows, u_idx, place_visits, first_idx)
        if not seen:
            continue
        if visit_idx is not None and in_clock_order(u.time, u.uncertainty, b.time, b.uncertainty):
            link = Link(
                token="",
                source=None if u.exposure_only else u.agent,
                source_row=u_idx,
                target=agent,
                target_row=first_idx,
                visit_row=visit_idx,
                lag=b.time - u.time,
            )
            status = STATUS_UNNAMED if u.exposure_only else STATUS_LINKED
            return status, link, shifted
        exposed = True
        shifted = True  # a more recent exposing use failed the clock rule
    if exposed:
        return STATUS_CLOCK, None, False
    return STATUS_PARALLEL, None, False


def _depth(adoptions):
    level = {}
    for a in adoptions:  # parents always come earlier in first-use order
        level[a.agent] = level[a.parent] + 1 if a.parent in level else 0
    return max(level.values()) if level else 0


def build_tree(rows, token, visits):
    """Spread tree for one ``patient_zero.tokens.Token``."""
    use_rows = sorted(token.rows)
    adoptions = []
    links = []
    parallel = []
    dropped = shifted = 0
    for n, (agent, first_idx) in enumerate(first_uses(rows, use_rows)):
        status, link, moved = adopt(rows, use_rows, agent, first_idx, visits)
        if link is not None:
            link = link._replace(token=token.key)
            links.append(link)
        if n == 0 and status in (STATUS_PARALLEL, STATUS_CLOCK):
            status = STATUS_FIRST
        if status == STATUS_PARALLEL:
            parallel.append(agent)
        elif status == STATUS_CLOCK:
            dropped += 1
        if moved and link is not None:
            shifted += 1
        adoptions.append(Adoption(
            agent=agent,
            row=first_idx,
            time=rows[first_idx].time,
            status=status,
            parent=link.source if link is not None and status == STATUS_LINKED else None,
            link=link,
        ))
    children = Counter(a.parent for a in adoptions if a.parent is not None)
    seed = next((a.agent for a in adoptions if children[a.agent]), None)
    return Tree(
        token=token.key,
        cls=token.cls,
        kind=token.kind,
        adoptions=tuple(adoptions),
        links=tuple(links),
        first_user=adoptions[0].agent if adoptions else None,
        effective_seed=seed,
        parallel_starts=tuple(parallel),
        dropped=dropped,
        shifted=shifted,
        depth=_depth(adoptions),
        places=len(set(rows[i].place for i in use_rows)),
    )


def children_of(tree):
    """Parent agent -> list of child agents (named links only), in adoption order."""
    out = defaultdict(list)
    for a in tree.adoptions:
        if a.parent is not None:
            out[a.parent].append(a.agent)
    return dict(out)


def parents_of(tree):
    """Agent -> credited parent agent (None for roots), for every adopter."""
    return {a.agent: a.parent for a in tree.adoptions}


# ---------------------------------------------------------------- all tokens


def _size_bucket(n):
    if n < 5:
        return "3-4"
    if n < 10:
        return "5-9"
    if n < 50:
        return "10-49"
    return "50+"


def build_trees(rows, tokens):
    """Spread trees for every token. Returns (trees, stats).

    ``trees`` maps token key -> Tree, sorted by key; ``stats`` holds counts only.
    """
    started = time.time()
    visits = build_visits(rows)
    trees = {}
    for key in sorted(tokens):
        trees[key] = build_tree(rows, tokens[key], visits)

    status = Counter()
    per_class = defaultdict(Counter)
    lags = []
    named_links = unnamed_links = 0
    seed_moved = no_seed = 0
    biggest = deepest = 0
    with_links = Counter()
    for tree in trees.values():
        for a in tree.adoptions:
            status[a.status] += 1
        for link in tree.links:
            if link.source is None:
                unnamed_links += 1
            else:
                named_links += 1
                lags.append(link.lag)
        n_named = sum(1 for a in tree.adoptions if a.status == STATUS_LINKED)
        pc = per_class[tree.cls]
        pc["trees"] += 1
        pc["named_links"] += n_named
        pc["parallel_starts"] += len(tree.parallel_starts)
        pc["dropped_by_clock"] += tree.dropped
        if n_named:
            pc["trees_with_links"] += 1
            with_links[_size_bucket(len(tree.adoptions))] += 1
        if tree.effective_seed is None:
            no_seed += 1
        elif tree.effective_seed != tree.first_user:
            seed_moved += 1
        biggest = max(biggest, len(tree.adoptions))
        deepest = max(deepest, tree.depth)

    stats = {
        "trees": len(trees),
        "trees_with_named_links": sum(1 for t in trees.values() if any(l.source for l in t.links)),
        "adoptions": sum(status.values()),
        "status": {s: status[s] for s in STATUSES},
        "links_named": named_links,
        "links_from_exposure_only_rows": unnamed_links,
        "links_dropped_by_clock": status[STATUS_CLOCK],
        "links_shifted_by_clock": sum(t.shifted for t in trees.values()),
        "parallel_starts": status[STATUS_PARALLEL],
        "trees_seed_is_not_first_user": seed_moved,
        "trees_without_effective_seed": no_seed,
        "largest_tree_agents": biggest,
        "deepest_tree": deepest,
        "trees_with_links_by_agent_count": {b: with_links[b] for b in ("3-4", "5-9", "10-49", "50+")},
        "named_link_lag_seconds": {
            "median": round(median(lags), 1) if lags else None,
            "min": round(min(lags), 1) if lags else None,
            "max": round(max(lags), 1) if lags else None,
        },
        "classes": {c: dict(sorted(v.items())) for c, v in sorted(per_class.items())},
        "seconds": round(time.time() - started, 1),
    }
    return trees, stats


# ---------------------------------------------------------------- command line


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="Build spread trees; print counts only (no text, no names).")
    ap.add_argument("--adapter", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--input", required=True, help="export folder (wiki) or JSON Lines file (generic)")
    ap.add_argument("--summary", help="write the counts as JSON here (keep it outside the repo)")
    ap.add_argument("--min-agents", type=int, default=MIN_AGENTS)
    ap.add_argument("--max-phrase-share", type=float, default=MAX_PHRASE_SHARE)
    ap.add_argument("--given", action="append", default=[],
                    help="a shared goal or prompt text file; its tokens are dropped (repeatable)")
    args = ap.parse_args(argv)
    try:
        guard_out(args.summary, [args.input] + list(args.given), "--summary")
        rows, load_stats = load(args.adapter, args.input)
        given = []
        for path in args.given:
            with open(path, "r", encoding="utf-8") as fh:
                given.append(fh.read())
    except (OSError, ValueError) as exc:
        print("cascade: ERROR: %s" % exc, file=out)
        return 2
    tokens, token_stats = find_tokens(rows, min_agents=args.min_agents, given_texts=given,
                                      max_phrase_share=args.max_phrase_share)
    _trees, stats = build_trees(rows, tokens)
    summary = {
        "job": "J4 cascade",
        "adapter": args.adapter,
        "load": {k: load_stats[k] for k in ("rows", "agents", "group_count", "time_span") if k in load_stats},
        "tokens_kept": token_stats["kept_total"],
        "token_seconds": token_stats["seconds"],
    }
    summary.update(stats)
    text = json.dumps(summary, indent=1, ensure_ascii=True)
    if args.summary:
        folder = os.path.dirname(os.path.abspath(args.summary))
        os.makedirs(folder, exist_ok=True)
        with open(args.summary, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
    out.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
