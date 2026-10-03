"""Chance baseline for adoption links: shuffle agent labels within each day.

Python standard library only. Deterministic: one ``random.Random(seed)``
drives every shuffle in a fixed order, so the same rows, tokens, seed and
number of shuffles always give the same numbers.

Method
------
Shuffle
    Every agent row keeps its time, place and text; only its agent label
    moves. Within each UTC day, the labels of that day's agent rows are put in
    random order and dealt back to the same rows. Blank and human rows are
    exposure-only and never move. So every agent keeps its number of rows per
    day, every page keeps its saves, and every token keeps its uses, times and
    places. What the shuffle breaks is the tie between the pages an agent
    visited and the tokens that same agent wrote later.

    It also scatters each agent's rows of a day over that day's pages, so an
    agent with many rows reaches many more distinct pages than it really did.
    That raises chance exposure. ``agent_place_pairs`` (distinct agent and
    page pairs, observed against shuffled) shows how much.

    ``scope="place_day"`` is a stricter option (not the default): labels are
    permuted only among agent rows on the same place and UTC day, so every
    agent keeps exactly its pages per day and only the order of saves within
    a page and day is random.

Count
    The token set is the observed one, held fixed. Under each shuffle every
    token's adopters are classified with the cascade rules unchanged: first
    use, exposure per page, the clock rule, credit to the most recent visible
    use (see ``patient_zero.cascade``). ``count_links`` is a faster
    re-implementation of ``cascade.adopt`` for counting only. It walks the
    places an adopter visited instead of every earlier use; with the true
    labels it gives every adopter the same status and source as the cascade
    (tests check this on every fixture and on the real data).

Compare
    Observed against the shuffles: mean, standard deviation, 5th and 95th
    percentile, min and max, ratio observed / mean, z, and how many shuffles
    reached the observed count (empirical p-value (k + 1) / (N + 1)). For
    named links in total, per class, per token, for trees with links, and as
    mean counts per page-family pair.

Limits (stated, not hidden): the shuffle keeps each day's activity, so links
that a busy day makes by itself count as chance; the number of distinct
agents per token is not held fixed; page-family pairs get a mean only.

Usage::

    null = run(rows, tokens, shuffles=100, seed=DEFAULT_SEED, scope="day")
    section = facts_section(null, cascade_links_named)   # counts only
    python cli.py ... --shuffles 100 --seed 20261003 --shuffle-scope day
"""

import random
import time
from array import array
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from statistics import stdev
from typing import NamedTuple, Optional, Tuple

from patient_zero.safe import NO_FAMILY

DEFAULT_SHUFFLES = 100
DEFAULT_SEED = 20261003
DAY_SECONDS = 86400
STATUS_KEYS = ("first", "linked", "unnamed", "clock_dropped", "parallel")
SCOPE_DAY = "day"
SCOPE_PLACE_DAY = "place_day"
SCOPES = (SCOPE_DAY, SCOPE_PLACE_DAY)
METHOD = {
    SCOPE_DAY: "agent labels permuted among the agent rows of each UTC day (blank and human rows "
               "fixed); token set fixed; cascade rules unchanged; one random.Random(seed) for all "
               "shuffles",
    SCOPE_PLACE_DAY: "agent labels permuted among the agent rows of each place and UTC day (blank "
                     "and human rows fixed); token set fixed; cascade rules unchanged; one "
                     "random.Random(seed) for all shuffles",
}


class Index(NamedTuple):
    """Rows and tokens as flat integer tables, built once per run."""

    keys: Tuple[str, ...]  # token keys, sorted
    classes: Tuple[str, ...]  # token class per key
    times: Tuple[float, ...]  # per row
    unc: Tuple[float, ...]  # per row
    place: Tuple[int, ...]  # place id per row
    family: Tuple[int, ...]  # family id per row
    families: Tuple[str, ...]  # family id -> name ("(none)" when missing)
    agent: Tuple[int, ...]  # true agent id per row; -1 for exposure-only rows
    agent_names: Tuple[str, ...]  # agent id -> label
    agent_rows: Tuple[int, ...]  # row indices of agent rows, ascending
    days: Tuple[Tuple[int, ...], ...]  # agent rows per UTC day, days ascending
    uses: Tuple[Tuple[int, ...], ...]  # per token: sorted row indices (all roles)
    uses_by_place: Tuple[dict, ...]  # per token: place id -> sorted row indices


class Count(NamedTuple):
    """One labelling's links, counted with the cascade rules."""

    named: array  # per token (Index order): named links
    status: dict  # status -> adopters
    trees_with_named: int
    agent_place_pairs: int  # distinct (agent, place) pairs under this labelling
    detail: Optional[tuple]  # per token: ((agent id, status, source row or -1), ...) when asked


class NullResult(NamedTuple):
    """The observed count and every shuffle's count."""

    index: Index
    shuffles: int
    seed: int
    scope: str
    groups: int  # shuffle groups (days, or place-days) holding agent rows
    observed: Count
    named: Tuple[array, ...]  # per shuffle: per-token named links
    status: Tuple[dict, ...]  # per shuffle: status -> adopters
    trees_with_named: Tuple[int, ...]  # per shuffle
    agent_place_pairs: Tuple[int, ...]  # per shuffle
    pairs_observed: Counter  # (from family id, to family id) -> named links, true labels
    pairs_sum: Counter  # the same, summed over all shuffles
    seconds: float


# ---------------------------------------------------------------- index


def build_index(rows, tokens):
    """Flat tables for ``count_links``. ``rows`` are load()'s time-sorted rows."""
    place_ids, family_ids, agent_ids = {}, {}, {}
    times, unc, place, family, agent, agent_rows = [], [], [], [], [], []
    by_day = defaultdict(list)
    for i, row in enumerate(rows):
        times.append(float(row.time))
        unc.append(float(row.uncertainty))
        place.append(place_ids.setdefault(row.place, len(place_ids)))
        family.append(family_ids.setdefault(row.family or NO_FAMILY, len(family_ids)))
        if row.exposure_only:
            agent.append(-1)
        else:
            agent.append(agent_ids.setdefault(row.agent, len(agent_ids)))
            agent_rows.append(i)
            by_day[int(row.time // DAY_SECONDS)].append(i)
    keys = tuple(sorted(tokens))
    uses, uses_by_place = [], []
    for key in keys:
        use_rows = tuple(sorted(tokens[key].rows))
        groups = defaultdict(list)
        for i in use_rows:
            groups[place[i]].append(i)
        uses.append(use_rows)
        uses_by_place.append({p: tuple(v) for p, v in groups.items()})
    return Index(
        keys=keys,
        classes=tuple(tokens[k].cls for k in keys),
        times=tuple(times),
        unc=tuple(unc),
        place=tuple(place),
        family=tuple(family),
        families=tuple(sorted(family_ids, key=family_ids.get)),
        agent=tuple(agent),
        agent_names=tuple(sorted(agent_ids, key=agent_ids.get)),
        agent_rows=tuple(agent_rows),
        days=tuple(tuple(by_day[d]) for d in sorted(by_day)),
        uses=tuple(uses),
        uses_by_place=tuple(uses_by_place),
    )


# ---------------------------------------------------------------- counting


def count_links(index, labels, pairs=None, detail=False):
    """Classify every adopter of every token under ``labels`` (row -> agent id, -1 exposure-only).

    Same rules as ``cascade.adopt``: B's exposure to an earlier use U needs a
    visit by B to U's place after U and no later than B's first use F; the
    visit and F must both be in clock order after U; credit goes to the most
    recent such U. ``pairs`` (a Counter) gets +1 per named link keyed by
    (source family id, target family id).
    """
    times, unc, place, family = index.times, index.unc, index.place, index.family
    visits = {}
    for i in index.agent_rows:
        mine = visits.get(labels[i])
        if mine is None:
            mine = visits[labels[i]] = {}
        lst = mine.get(place[i])
        if lst is None:
            mine[place[i]] = [i]
        else:
            lst.append(i)  # agent_rows ascend, so every list stays sorted
    named = array("i", [0]) * len(index.uses)
    n_first = n_linked = n_unnamed = n_clock = n_parallel = 0
    with_named = 0
    details = [] if detail else None
    for t, use_rows in enumerate(index.uses):
        by_place = index.uses_by_place[t]
        firsts = {}
        for i in use_rows:
            a = labels[i]
            if a >= 0 and a not in firsts:
                firsts[a] = i  # dict order is first-use order
        tok_named = 0
        tok_detail = [] if detail else None
        lead = True
        for a, f in firsts.items():
            tf, uf = times[f], unc[f]
            mine = visits[a]
            if len(by_place) < len(mine):
                common = ((U, mine.get(p)) for p, U in by_place.items())
            else:
                common = ((by_place.get(p), V) for p, V in mine.items())
            best = -1
            exposed = False
            for U, V in common:
                if U is None or V is None:
                    continue
                k = bisect_left(U, f)  # uses strictly before F on this place
                if k == 0:
                    continue
                hv = bisect_right(V, f)  # visits no later than F
                if hv == 0:
                    continue
                last_v = V[hv - 1]
                if U[0] >= last_v:
                    continue  # no use before the latest visit: not exposed here
                exposed = True
                for j in range(bisect_left(U, last_v, 0, k) - 1, -1, -1):
                    u = U[j]
                    if u <= best:
                        break  # an equal or more recent use already qualified
                    tu, uu = times[u], unc[u]
                    if not tf - tu > uu + uf:
                        continue  # F is not in clock order after U
                    m = hv - 1
                    while m >= 0 and V[m] > u:
                        v = V[m]
                        if times[v] - tu > uu + unc[v]:
                            best = u
                            break
                        m -= 1
                    if best == u:
                        break
            if best >= 0:
                if labels[best] >= 0:
                    status = "linked"
                    n_linked += 1
                    tok_named += 1
                    if pairs is not None:
                        pairs[(family[best], family[f])] += 1
                else:
                    status = "unnamed"
                    n_unnamed += 1
            elif lead:
                status = "first"
                n_first += 1
            elif exposed:
                status = "clock_dropped"
                n_clock += 1
            else:
                status = "parallel"
                n_parallel += 1
            lead = False
            if detail:
                tok_detail.append((a, status, best))
        named[t] = tok_named
        with_named += tok_named > 0
        if detail:
            details.append(tuple(tok_detail))
    status = {"first": n_first, "linked": n_linked, "unnamed": n_unnamed,
              "clock_dropped": n_clock, "parallel": n_parallel}
    return Count(named=named, status=status, trees_with_named=with_named,
                 agent_place_pairs=sum(len(mine) for mine in visits.values()),
                 detail=tuple(details) if detail else None)


def shuffle_groups(index, scope=SCOPE_DAY):
    """Groups of agent rows whose labels are permuted together, in a fixed order."""
    if scope == SCOPE_DAY:
        return index.days
    if scope == SCOPE_PLACE_DAY:
        groups = {}
        for day_rows in index.days:
            for i in day_rows:
                groups.setdefault((index.place[i], int(index.times[i] // DAY_SECONDS)), []).append(i)
        return tuple(tuple(g) for g in groups.values())  # first-row order: deterministic
    raise ValueError("scope must be one of %s" % ", ".join(SCOPES))


def shuffle_labels(index, rng, groups=None):
    """New row -> agent id labels: each group's agent labels dealt back in random order.

    ``groups`` defaults to the UTC days (``shuffle_groups(index, "day")``).
    """
    labels = list(index.agent)
    for group in (index.days if groups is None else groups):
        dealt = [labels[i] for i in group]
        rng.shuffle(dealt)
        for i, a in zip(group, dealt):
            labels[i] = a
    return labels


def run(rows, tokens, shuffles=DEFAULT_SHUFFLES, seed=DEFAULT_SEED, scope=SCOPE_DAY, index=None):
    """Observed links plus ``shuffles`` label shuffles from ``seed``. Returns a NullResult."""
    if shuffles < 0:
        raise ValueError("shuffles must be 0 or more")
    started = time.perf_counter()
    index = index or build_index(rows, tokens)
    groups = shuffle_groups(index, scope)
    pairs_observed = Counter()
    observed = count_links(index, index.agent, pairs=pairs_observed)
    rng = random.Random(seed)
    pairs_sum = Counter()
    named, status, with_named, agent_places = [], [], [], []
    for _ in range(shuffles):
        count = count_links(index, shuffle_labels(index, rng, groups), pairs=pairs_sum)
        named.append(count.named)
        status.append(count.status)
        with_named.append(count.trees_with_named)
        agent_places.append(count.agent_place_pairs)
    return NullResult(
        index=index,
        shuffles=shuffles,
        seed=seed,
        scope=scope,
        groups=len(groups),
        observed=observed,
        named=tuple(named),
        status=tuple(status),
        trees_with_named=tuple(with_named),
        agent_place_pairs=tuple(agent_places),
        pairs_observed=pairs_observed,
        pairs_sum=pairs_sum,
        seconds=time.perf_counter() - started,
    )


# ---------------------------------------------------------------- summaries


def percentile(sorted_values, q):
    """Linear-interpolated percentile of an ascending list (q in [0, 1])."""
    if not sorted_values:
        return None
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def summarize(values, observed):
    """Observed against shuffle values: mean, spread, ratio, z, shuffles reaching it."""
    vals = sorted(values)
    n = len(vals)
    out = {"observed": observed, "shuffles": n}
    if not n:
        out.update({"mean": None, "sd": None, "min": None, "p05": None, "median": None,
                    "p95": None, "max": None, "ratio": None, "z": None,
                    "shuffles_ge_observed": None, "p_value": None})
        return out
    m = sum(vals) / n
    sd = float(stdev(vals)) if n > 1 else 0.0
    ge = sum(1 for v in vals if v >= observed)
    digits = 3 if all(isinstance(v, int) for v in vals) else 5
    out.update({
        "mean": round(m, digits),
        "sd": round(sd, digits),
        "min": round(vals[0], digits),
        "p05": round(percentile(vals, 0.05), digits),
        "median": round(percentile(vals, 0.5), digits),
        "p95": round(percentile(vals, 0.95), digits),
        "max": round(vals[-1], digits),
        "ratio": round(observed / m, 3) if m > 0 else None,
        "z": round((observed - m) / sd, 2) if sd > 0 else None,
        "shuffles_ge_observed": ge,
        "p_value": round((ge + 1) / (n + 1), 4),
    })
    return out


def brief(summary):
    """The short form of a summary: observed, mean, sd, ratio, shuffles reaching it."""
    return {k: summary[k] for k in ("observed", "mean", "sd", "ratio", "shuffles_ge_observed")}


def group_values(null, token_indices):
    """Per-shuffle named links summed over a group of tokens (Index positions)."""
    idx = list(token_indices)
    return [sum(arr[i] for i in idx) for arr in null.named]


def group_summary(null, token_indices):
    idx = list(token_indices)
    return summarize(group_values(null, idx), sum(null.observed.named[i] for i in idx))


def per_token(null):
    """Token key -> (mean named links under the shuffle, shuffles reaching the observed count)."""
    out = {}
    n = null.shuffles
    for t, key in enumerate(null.index.keys):
        obs = null.observed.named[t]
        if n:
            total = sum(arr[t] for arr in null.named)
            ge = sum(1 for arr in null.named if arr[t] >= obs)
            out[key] = (round(total / n, 3), ge)
        else:
            out[key] = (None, None)
    return out


def adopters(status):
    """All adopters under one labelling (every status)."""
    return sum(status[k] for k in STATUS_KEYS)


def linked_share(status):
    """Named links per later adopter (every adopter but each token's first user)."""
    later = adopters(status) - status["first"]
    return status["linked"] / later if later else 0.0


def facts_section(null, cascade_links_named=None):
    """The facts.json "nullmodel" section (counts only).

    ``adopters`` and ``linked_share_of_later_adopters`` matter when reading
    ``links_named``: the shuffle splits an agent's repeated uses of a token
    over several agents, so it makes more adopters, and more chances to link.
    ``trees_beating_every_shuffle`` comes with ``chance_beats_upper_bound``:
    if the shuffle were the true process, each tree would beat all N shuffles
    with probability at most 1 / (N + 1), so at most tokens / (N + 1) trees
    would do so by luck.
    """
    ix = null.index
    obs = null.observed
    by_class = defaultdict(list)
    for t, cls in enumerate(ix.classes):
        by_class[cls].append(t)
    section = {
        "method": METHOD[null.scope],
        "scope": null.scope,
        "shuffles": null.shuffles,
        "seed": null.seed,
        "days": len(ix.days),
        "shuffle_groups": null.groups,
        "agent_rows_shuffled": len(ix.agent_rows),
        "tokens": len(ix.keys),
        "observed_matches_cascade": (obs.status["linked"] == cascade_links_named
                                     if cascade_links_named is not None else None),
        "links_named": summarize([s["linked"] for s in null.status], obs.status["linked"]),
        "trees_with_named_links": summarize(list(null.trees_with_named), obs.trees_with_named),
        "adopters": summarize([adopters(s) for s in null.status], adopters(obs.status)),
        "linked_share_of_later_adopters": summarize([linked_share(s) for s in null.status],
                                                    round(linked_share(obs.status), 5)),
        "links_from_exposure_only_rows": summarize([s["unnamed"] for s in null.status],
                                                   obs.status["unnamed"]),
        "parallel_starts": summarize([s["parallel"] for s in null.status], obs.status["parallel"]),
        "links_dropped_by_clock": summarize([s["clock_dropped"] for s in null.status],
                                            obs.status["clock_dropped"]),
        "by_class": {cls: brief(group_summary(null, by_class[cls])) for cls in sorted(by_class)},
        "agent_place_pairs": summarize(list(null.agent_place_pairs), obs.agent_place_pairs),
    }
    tok = per_token(null)
    linked_tokens = [k for t, k in enumerate(ix.keys) if obs.named[t] > 0]
    n = null.shuffles
    section["trees_beating_every_shuffle"] = sum(1 for k in linked_tokens if tok[k][1] == 0) if n else None
    section["trees_with_links_reached_by_a_shuffle"] = (
        sum(1 for k in linked_tokens if tok[k][1] > 0) if n else None)
    section["chance_beats_upper_bound"] = round(len(ix.keys) / (n + 1), 2) if n else None
    return section
