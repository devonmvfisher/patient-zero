"""Scores on the spread trees: influence, speed, risky flags, page-family flow.

Python standard library only, deterministic. Inputs are the rows from
``patient_zero.load``, the trees from ``patient_zero.cascade.build_trees``
and, for comparisons against chance, the ``NullResult`` from
``patient_zero.nullmodel.run``. Strings from the data (agent labels, page
families) leave this module only through ``safe.short``: escaped, at most 80
characters. Blank and human rows have no label, so they are never named.

Influence (per agent)
    ``adopters``         named links credited to the agent (direct adopters)
    ``agents_reached``   distinct agents among those adopters, over all tokens
    ``downstream``       agents below the agent in its trees (any depth), summed
    ``trees_seeded``     trees where the agent is the effective seed
    ``trees_first``      trees where the agent is the first user
    ``trees_joined``     trees the agent is in (any status)
    ``parallel_starts``  trees the agent began with no exposure to an earlier use
    Ranked by adopters, then downstream, then trees seeded, then label.

Speed (per tree)
    Minutes from the effective seed's first use to the first use of the 3rd
    and of the 10th agent in the seed's branch: the agents linked below the
    seed at any depth, in time order, the seed itself not counted. The same
    for the root of the tree's largest branch. None when the branch is
    smaller than that.

Risky flags (type names only, never the surrounding text)
    Trees per risky-word type, and named links in risky trees against the
    shuffle: for trees with a narrow type (a named mechanism), trees with
    broad types only (answer, share, relay), and trees with no type.

Page-family flow
    Named links by (page family of the source use, page family of the
    adopter's first use), with the mean count under the shuffle; links within
    one family against links across families; links out of each family
    against the shuffle.
"""

from collections import Counter, defaultdict
from math import ceil
from statistics import median

from patient_zero import nullmodel as N
from patient_zero.safe import NO_FAMILY, short
from patient_zero.tokens import RISKY_TYPES

# Risky types that name a mechanism; "answer", "share" and "relay" are broad words.
NARROW_RISKY = ("bypass", "workaround", "blocked", "captcha", "rate_limit", "proxy", "cors")
SPEED_KS = (3, 10)
SPEED_BUCKETS = ((10, "le_10_min"), (60, "le_60_min"), (360, "le_6_h"), (1440, "le_24_h"))
SPEED_OVER = "over_24_h"
INFLUENCE_TOP = 20
FLOW_TOP = 25
FAMILY_TOP = 15


# ---------------------------------------------------------------- branches and speed


def branch_sizes(tree):
    """Root agent -> agents in its branch (the root plus everyone linked below it)."""
    size = {a.agent: 1 for a in tree.adoptions}
    for a in reversed(tree.adoptions):  # parents come earlier in first-use order
        if a.parent is not None:
            size[a.parent] += size[a.agent]
    return {a.agent: size[a.agent] for a in tree.adoptions if a.parent is None}


def subtree_sizes(tree):
    """Agent -> agents in its subtree (itself plus everyone linked below it), every adopter."""
    size = {a.agent: 1 for a in tree.adoptions}
    for a in reversed(tree.adoptions):
        if a.parent is not None:
            size[a.parent] += size[a.agent]
    return size


def branch_minutes(tree, root, ks=SPEED_KS):
    """(branch size, {k: minutes from the root's first use to the k-th agent below it, or None})."""
    children = defaultdict(list)
    start = None
    for a in tree.adoptions:
        if a.parent is not None:
            children[a.parent].append(a)
        if a.agent == root:
            start = a.time
    if start is None:
        raise ValueError("root is not an adopter of this tree")
    below = []
    stack = [root]
    while stack:
        for child in children.get(stack.pop(), ()):
            below.append(child.time)
            stack.append(child.agent)
    below.sort()
    minutes = {k: (round((below[k - 1] - start) / 60.0, 2) if len(below) >= k else None) for k in ks}
    return len(below) + 1, minutes


def tree_speed(tree, branch_root=None):
    """Speed fields of one tree: from the effective seed, and from the largest branch's root."""
    out = {"seed_branch": 0, "seed_min_to_3rd": None, "seed_min_to_10th": None,
           "branch_min_to_3rd": None, "branch_min_to_10th": None}
    if tree.effective_seed is not None:
        size, minutes = branch_minutes(tree, tree.effective_seed)
        out["seed_branch"] = size
        out["seed_min_to_3rd"] = minutes[3]
        out["seed_min_to_10th"] = minutes[10]
    if branch_root is not None:
        _size, minutes = branch_minutes(tree, branch_root)
        out["branch_min_to_3rd"] = minutes[3]
        out["branch_min_to_10th"] = minutes[10]
    return out


def _bucket(minutes):
    for limit, name in SPEED_BUCKETS:
        if minutes <= limit:
            return name
    return SPEED_OVER


def _speed_block(records, field):
    found = sorted((r[field], r["token_id"], n) for n, r in enumerate(records) if r[field] is not None)
    buckets = Counter(_bucket(m) for m, _t, _n in found)
    block = {
        "trees": len(found),
        "median_minutes": round(median(m for m, _t, _n in found), 2) if found else None,
        "buckets": {name: buckets[name] for name in [b for _l, b in SPEED_BUCKETS] + [SPEED_OVER]},
        "fastest": None,
    }
    if found:
        minutes, _tid, n = found[0]
        rec = records[n]
        block["fastest"] = {"token_id": rec["token_id"], "cls": rec["cls"], "kind": rec["kind"],
                            "minutes": minutes, "agents": rec["agents"],
                            "seed_branch": rec["seed_branch"], "largest_branch": rec["largest_branch"]}
    return block


def speed_table(records):
    """Speed over all trees, from the tree records (fields from ``tree_speed``)."""
    return {
        "definition": "minutes from the root's first use to the first use of the k-th agent "
                      "linked below it (any depth; the root not counted)",
        "seed_to_3rd": _speed_block(records, "seed_min_to_3rd"),
        "seed_to_10th": _speed_block(records, "seed_min_to_10th"),
        "largest_branch_to_3rd": _speed_block(records, "branch_min_to_3rd"),
        "largest_branch_to_10th": _speed_block(records, "branch_min_to_10th"),
    }


# ---------------------------------------------------------------- influence


def influence_table(trees, top=INFLUENCE_TOP):
    """Per-agent influence over all trees: summary counts and the top agents (labels escaped)."""
    per = defaultdict(Counter)
    reached = defaultdict(set)
    for tree in trees.values():
        if not tree.adoptions:
            continue
        size = subtree_sizes(tree)
        for a in tree.adoptions:
            mine = per[a.agent]
            mine["trees_joined"] += 1
            mine["downstream"] += size[a.agent] - 1
            if a.status == "parallel":
                mine["parallel_starts"] += 1
            if a.parent is not None:
                per[a.parent]["adopters"] += 1
                reached[a.parent].add(a.agent)
        per[tree.first_user]["trees_first"] += 1
        if tree.effective_seed is not None:
            per[tree.effective_seed]["trees_seeded"] += 1
    ranked = sorted(per.items(), key=lambda kv: (-kv[1]["adopters"], -kv[1]["downstream"],
                                                 -kv[1]["trees_seeded"], kv[0]))
    total = sum(c["adopters"] for c in per.values())
    with_adopters = [c["adopters"] for _a, c in ranked if c["adopters"] > 0]
    top_pct = max(1, ceil(len(ranked) / 100.0)) if ranked else 0

    def share(n):
        return round(sum(c["adopters"] for _a, c in ranked[:n]) / total, 4) if total else None

    fields = ("adopters", "downstream", "trees_seeded", "trees_first", "trees_joined", "parallel_starts")
    table = []
    for rank, (agent, c) in enumerate(ranked[:top], 1):
        row = {"rank": rank, "agent": short(agent)}
        row.update({f: c[f] for f in fields})
        row["agents_reached"] = len(reached[agent])
        table.append(row)
    return {
        "agents": len(ranked),
        "agents_with_adopters": len(with_adopters),
        "named_links": total,
        "top_1_share": share(1),
        "top_10_share": share(10),
        "top_1pct_agents": top_pct,
        "top_1pct_share": share(top_pct),
        "median_adopters_of_agents_with_adopters": median(with_adopters) if with_adopters else None,
        "top": table,
    }


# ---------------------------------------------------------------- risky flags


def _token_groups(tokens, keys):
    """Index positions for the risky groups and for each type."""
    narrow, broad_only, none = [], [], []
    by_type = defaultdict(list)
    for t, key in enumerate(keys):
        kinds = set(tokens[key].risky)
        for name in kinds:
            by_type[name].append(t)
        if kinds & set(NARROW_RISKY):
            narrow.append(t)
        elif kinds:
            broad_only.append(t)
        else:
            none.append(t)
    return narrow, broad_only, none, by_type


def risky_table(records, tokens=None, null=None):
    """Risky-word types per tree (type names only); against the shuffle when ``null`` is given."""
    by_type = Counter()
    links_by_type = Counter()
    linked_by_type = Counter()
    for rec in records:
        for t in rec["risky"]:
            by_type[t] += 1
            links_by_type[t] += rec["links_named"]
            if rec["links_named"]:
                linked_by_type[t] += 1
    table = {
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
    if null is not None and tokens is not None:
        narrow, broad_only, none, by_type_idx = _token_groups(tokens, null.index.keys)
        table["named_links_vs_shuffle"] = {
            "narrow": N.brief(N.group_summary(null, narrow)),
            "broad_only": N.brief(N.group_summary(null, broad_only)),
            "no_type": N.brief(N.group_summary(null, none)),
            "by_type": {t: N.brief(N.group_summary(null, by_type_idx.get(t, ()))) for t in RISKY_TYPES},
        }
    return table


# ---------------------------------------------------------------- page-family flow


def _ratio(observed, expected):
    return round(observed / expected, 3) if expected else None


def family_flow(rows, trees, null=None, top=FLOW_TOP, family_top=FAMILY_TOP):
    """Named links by (source family, target family), with shuffle means when ``null`` is given."""
    observed = Counter()
    for tree in trees.values():
        for link in tree.links:
            if link.source is not None:
                observed[(rows[link.source_row].family or NO_FAMILY,
                          rows[link.target_row].family or NO_FAMILY)] += 1
    expected = Counter()
    n = null.shuffles if null is not None else 0
    if n:
        names = null.index.families
        for (src, dst), count in null.pairs_sum.items():
            expected[(names[src], names[dst])] += count
    matches = None
    if null is not None:
        names = null.index.families
        from_null = Counter({(names[s], names[d]): c for (s, d), c in null.pairs_observed.items()})
        matches = from_null == observed

    def mean(count):
        return round(count / n, 3) if n else None

    within = sum(c for (s, d), c in observed.items() if s == d)
    within_exp = sum(c for (s, d), c in expected.items() if s == d)
    total = sum(observed.values())
    total_exp = sum(expected.values())
    pairs = sorted(observed.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    out_obs, out_exp = Counter(), Counter()
    for (s, _d), c in observed.items():
        out_obs[s] += c
    for (s, _d), c in expected.items():
        out_exp[s] += c
    families = sorted(out_obs.items(), key=lambda kv: (-kv[1], kv[0]))[:family_top]
    return {
        "named_links": total,
        "pairs_with_links": len(observed),
        "observed_matches_nullmodel": matches,
        "within_family": {"observed": within, "shuffle_mean": mean(within_exp),
                          "ratio": _ratio(within, mean(within_exp))},
        "across_families": {"observed": total - within, "shuffle_mean": mean(total_exp - within_exp),
                            "ratio": _ratio(total - within, mean(total_exp - within_exp))},
        "within_share": round(within / total, 4) if total else None,
        "within_share_shuffle": round(within_exp / total_exp, 4) if total_exp else None,
        "top_pairs": [{"from": short(s), "to": short(d), "named_links": c,
                       "shuffle_mean": mean(expected[(s, d)]), "ratio": _ratio(c, mean(expected[(s, d)]))}
                      for (s, d), c in pairs],
        "links_out_of_family": [{"family": short(f), "named_links": c, "shuffle_mean": mean(out_exp[f]),
                                 "ratio": _ratio(c, mean(out_exp[f]))} for f, c in families],
    }


# ---------------------------------------------------------------- all scores


def build_scores(result, records):
    """The facts.json "scores" section: influence, speed and page-family flow.

    Risky flags go to the facts.json "risky" section (``risky_table``).
    """
    return {
        "influence": influence_table(result.trees),
        "speed": speed_table(records),
        "family_flow": family_flow(result.rows, result.trees, result.null),
    }

