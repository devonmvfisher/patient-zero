"""Detection limit by replicated injection: how much copying can a log hide from the order null?

Python standard library only. Deterministic: every plant set draws from its
own ``random.Random(seed)``, and every plant set is scored against the same
order shuffles (one ``random.Random(seed)``, as in ``patient_zero.checks``).
Opt-in: the main run (``cli.py``, facts.json) does not call this module.
Output is counts only: no row text, no agent labels, no token keys.

Why
---
"Observed links sit at chance" only means something once we know what the
test can see. This module plants synthetic copying on the log's own rows,
adds it to the real tokens and asks whether the order null now flags the
total. One planted draw per amount can be lucky or unlucky, so the draw is
repeated for several named seeds. For each amount ``k`` (planted cascades)
it reports how many draws the test detected, and the smallest ``k`` detected
in most draws: the minimum detectable carry, in cascades and in links.

Carry cascade (``plant_carry``)
    Start at a random agent row. A later visitor to the page of an infected
    use (in clock order, within 3 days) is infected with probability 0.35.
    Its use of the token goes on its own next save on another page (in clock
    order, within 3 days of the visit), and only on a page the cascade has
    not used yet. A cascade stops at 40 agents and is kept when it has 4 or
    more adopters. A planted token is a set of real rows: no row, time or
    text is changed. Every planted link is a cross-page link: the adopter saw
    the token on one page and used it on another.

Statistics
    ``aggregate``: every named link (real tokens plus the first ``k``
    planted cascades). ``cross_page``: only links whose adopter's first use
    is on a different page from the credited source use.

Null
    The order null of ``patient_zero.checks``: every agent keeps its rows and
    its own times per UTC day; only which row came at which time is random.
    Its random stream depends on the rows only, so adding planted tokens
    leaves every real token's shuffled counts unchanged, and every plant set
    meets the same shuffles.

Detected
    ``z >= 3`` and at most 1 shuffle reaching the observed count (defaults).

Limits (stated, not hidden): this is a detection limit for one synthetic
mechanism on one background, never an upper bound on real copying. A log
whose links already sit below the null needs more planted copying before
the test moves, and real copying need not look like the planted kind.

Usage::

    python -m patient_zero.inject --adapter wiki --input PATH --out FILE.json \\
        --plant-seeds 20261041,20261042,20261043,20261044,20261045 --ks 40,60,80,100
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from typing import NamedTuple, Tuple

from patient_zero import checks as K
from patient_zero import nullmodel as N
from patient_zero.load import ADAPTERS, load
from patient_zero.tokens import MAX_PHRASE_SHARE, MIN_AGENTS, find_tokens

CARRY_Q = 0.35
CARRY_CAP = 40
CARRY_HORIZON = 3 * 86400
MIN_ADOPTERS = 4
MAX_TRIES = 20000
DETECT_Z = 3.0
DETECT_MAX_REACHED = 1
DEFAULT_KS = (40, 60, 80, 100)
STATS = ("aggregate", "cross_page")
CELL_KEYS = ("observed", "mean", "sd", "p05", "p95", "ratio", "z", "shuffles_ge_observed")


class PlantedToken(NamedTuple):
    """A planted token: a set of real rows, shaped like ``tokens.Token`` for the index."""

    key: str
    cls: str
    rows: Tuple[int, ...]


class Layout(NamedTuple):
    """Row lookups the generator needs, built once per log."""

    agent_rows: Tuple[int, ...]  # agent rows, ascending
    by_place: dict  # place -> all row indices on it, ascending
    by_agent: dict  # agent -> its row indices, ascending


class Plant(NamedTuple):
    """One plant set: its name, the (seed, count) segments drawn in order, and its token keys."""

    name: str
    segments: Tuple[Tuple[int, int], ...]
    keys: Tuple[str, ...]  # injection order: the first k are planted at amount k


class InjectionNull(NamedTuple):
    """Observed and shuffled counts for the real tokens and every planted token."""

    shuffles: int
    seed: int
    scope: str
    real_observed: dict  # stat -> count
    real_sims: dict  # stat -> per-shuffle counts
    token_observed: dict  # planted key -> (links, cross-page links)
    token_sims: tuple  # per shuffle: planted key -> (links, cross-page links)
    seconds: float


# ---------------------------------------------------------------- planting


def layout(rows):
    agent_rows, by_place, by_agent = [], defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        by_place[row.place].append(i)
        if not row.exposure_only:
            agent_rows.append(i)
            by_agent[row.agent].append(i)
    return Layout(tuple(agent_rows), dict(by_place), dict(by_agent))


def _clock_after(rows, u, v):
    return rows[v].time - rows[u].time > rows[u].uncertainty + rows[v].uncertainty


def plant_carry(rows, lay, rng, seed_row, carry_p=1.0, q=CARRY_Q, cap=CARRY_CAP, horizon=CARRY_HORIZON):
    """One cascade from ``seed_row``. Returns (use rows, {adopter: parent agent}).

    With probability ``carry_p`` an adopter's use moves to its next save on
    another page (carry); otherwise it stays on the page where it saw the
    token. ``carry_p=1.0`` is the carry cascade used for the detection limit.
    """
    infected = {rows[seed_row].agent}
    use_rows = [seed_row]
    parent = {}
    frontier = [seed_row]
    while frontier and len(infected) < cap:
        u = frontier.pop(0)
        for v in lay.by_place[rows[u].place]:
            if v <= u or rows[v].exposure_only or not _clock_after(rows, u, v):
                continue
            if rows[v].time - rows[u].time > horizon:
                break
            a = rows[v].agent
            if a in infected or rng.random() >= q:
                continue
            adopt = v
            if rng.random() < carry_p:
                nxt = [w for w in lay.by_agent[a] if w > v and rows[w].place != rows[v].place
                       and _clock_after(rows, v, w) and rows[w].time - rows[v].time < horizon]
                if not nxt:
                    continue
                adopt = nxt[0]
                if any(rows[x].place == rows[adopt].place for x in use_rows):
                    continue
            infected.add(a)
            parent[a] = rows[u].agent
            use_rows.append(adopt)
            frontier.append(adopt)
            if len(infected) >= cap:
                break
    return use_rows, parent


def draw_pool(rows, lay, seed, count, carry_p=1.0, min_adopters=MIN_ADOPTERS, max_tries=MAX_TRIES):
    """Up to ``count`` cascades (sorted use rows) from ``random.Random(seed)``, in draw order."""
    rng = random.Random(seed)
    pool = []
    tries = 0
    while len(pool) < count and tries < max_tries and lay.agent_rows:
        tries += 1
        start = rng.choice(lay.agent_rows)
        use_rows, parent = plant_carry(rows, lay, rng, start, carry_p)
        if len(parent) >= min_adopters:
            pool.append(tuple(sorted(set(use_rows))))
    return pool


def plant_set(rows, name, segments, lay=None, min_adopters=MIN_ADOPTERS):
    """A plant set from (seed, count) segments drawn in order. Returns (Plant, {key: PlantedToken})."""
    lay = lay or layout(rows)
    tokens, keys = {}, []
    for seed, count in segments:
        for use_rows in draw_pool(rows, lay, seed, count, min_adopters=min_adopters):
            key = "planted:%s:%03d" % (name, len(keys))
            tokens[key] = PlantedToken(key, "planted_carry", use_rows)
            keys.append(key)
    return Plant(name, tuple((int(s), int(c)) for s, c in segments), tuple(keys)), tokens


# ---------------------------------------------------------------- counting


def split_counts(index, labels=None, strict=False):
    """Per token (index order): (same-page links, cross-page links) under ``labels``."""
    labels = index.agent if labels is None else labels
    count = N.count_links(index, labels, detail=True, strict=strict)
    same = [0] * len(index.keys)
    cross = [0] * len(index.keys)
    place = index.place
    for t, det in enumerate(count.detail):
        firsts = {}
        for i in index.uses[t]:
            a = labels[i]
            if a >= 0 and a not in firsts:
                firsts[a] = i
        if len(firsts) != len(det):
            raise RuntimeError("detail and first uses differ in length")
        for (a, status, src), (a2, f) in zip(det, firsts.items()):
            if a != a2:
                raise RuntimeError("detail order differs from first-use order")
            if status == "linked":
                if place[src] == place[f]:
                    same[t] += 1
                else:
                    cross[t] += 1
    return same, cross


def injection_null(rows, real_tokens, planted_tokens, shuffles=N.DEFAULT_SHUFFLES, seed=N.DEFAULT_SEED,
                   scope=K.SCOPE_AGENT_DAY, progress=None):
    """Real and planted counts, observed and under ``shuffles`` order shuffles from ``seed``."""
    if shuffles < 1:
        raise ValueError("shuffles must be 1 or more")
    clash = set(real_tokens) & set(planted_tokens)
    if clash:
        raise ValueError("%d planted keys clash with real token keys" % len(clash))
    started = time.perf_counter()
    alltok = dict(real_tokens)
    alltok.update(planted_tokens)

    def tally(index):
        same, cross = split_counts(index)
        real = {"aggregate": 0, "cross_page": 0}
        tok = {}
        for t, key in enumerate(index.keys):
            if key in planted_tokens:
                tok[key] = (same[t] + cross[t], cross[t])
            else:
                real["aggregate"] += same[t] + cross[t]
                real["cross_page"] += cross[t]
        return real, tok

    real_obs, tok_obs = tally(N.build_index(rows, alltok))
    rng = random.Random(seed)
    real_sims = {stat: [] for stat in STATS}
    tok_sims = []
    for s in range(shuffles):
        new_rows, new_tokens = K.reorder(rows, alltok, rng, scope)
        real, tok = tally(N.build_index(new_rows, new_tokens))
        for stat in STATS:
            real_sims[stat].append(real[stat])
        tok_sims.append(tok)
        if progress is not None:
            progress(s + 1)
    return InjectionNull(shuffles, seed, scope, real_obs, real_sims, tok_obs, tuple(tok_sims),
                         time.perf_counter() - started)


# ---------------------------------------------------------------- cells and rates


def is_detected(summary, min_z=DETECT_Z, max_reached=DETECT_MAX_REACHED):
    """Detected: z at or above ``min_z`` and at most ``max_reached`` shuffles reaching the observed count."""
    z = summary.get("z")
    return z is not None and z >= min_z and summary["shuffles_ge_observed"] <= max_reached


def cells(null, plant, ks, min_z=DETECT_Z, max_reached=DETECT_MAX_REACHED):
    """{stat: [cell per k]} for one plant set: real tokens plus its first k cascades."""
    out = {stat: [] for stat in STATS}
    n = null.shuffles
    for k in ks:
        sel = plant.keys[:k]
        links = sum(null.token_observed[key][0] for key in sel)
        cross = sum(null.token_observed[key][1] for key in sel)
        for p, stat in enumerate(STATS):
            observed = null.real_observed[stat] + sum(null.token_observed[key][p] for key in sel)
            values = [null.real_sims[stat][i] + sum(null.token_sims[i][key][p] for key in sel) for i in range(n)]
            sm = N.summarize(values, observed)
            cell = {"k": k, "cascades": len(sel), "planted_links": links, "planted_cross_links": cross}
            cell.update({key: sm[key] for key in CELL_KEYS})
            cell["detected"] = is_detected(sm, min_z, max_reached)
            if len(sel) < k:
                cell["short"] = True  # the pool ran out of tries before k cascades
            out[stat].append(cell)
    return out


def default_need(replicates):
    """Draws needed to call an amount detectable: 4 of 5 (80%, rounded up)."""
    return max(1, math.ceil(0.8 * replicates))


def rate_summary(table, names, ks, extension_ks=(), need=None, reference_links=None):
    """Detection rate per k, the minimum detectable carry, and every non-monotone step, per statistic.

    ``table``: {plant name: {stat: [cell per k]}} (``cells`` output). ``names``:
    the replicate plant sets that count. ``reference_links``: the denominator
    for the minimum detectable carry as a share (the real aggregate links).
    """
    need = default_need(len(names)) if need is None else need
    grid = list(ks) + list(extension_ks)
    out = {"replicates": len(names), "need": need, "rule": "detected in at least %d of %d draws" % (need, len(names))}
    for stat in STATS:
        by_k = {}
        for name in names:
            for cell in table[name][stat]:
                by_k.setdefault(cell["k"], []).append(cell)
        rates = []
        for k in grid:
            cs = by_k.get(k, [])
            links = [c["planted_links"] for c in cs]
            hits = sum(1 for c in cs if c["detected"])
            rates.append({"k": k, "extension": k not in ks, "detected": hits, "of": len(cs),
                          "rate": round(hits / len(cs), 3) if cs else None,
                          "planted_links_median": statistics.median(links) if links else None,
                          "planted_links_min": min(links) if links else None,
                          "planted_links_max": max(links) if links else None,
                          "z_min": min((c["z"] for c in cs if c["z"] is not None), default=None),
                          "z_max": max((c["z"] for c in cs if c["z"] is not None), default=None)})

        def first_reaching(rows_):
            return next((r for r in rows_ if r["of"] and r["detected"] >= need), None)

        hit = first_reaching([r for r in rates if not r["extension"]])
        ext = None if hit else first_reaching([r for r in rates if r["extension"]])
        mdc = {"k": hit["k"] if hit else None, "reached_in_grid": hit is not None,
               "beyond_k": None if hit else max(ks)}
        src = hit or ext
        if src is not None:
            mdc.update({"planted_links_median": src["planted_links_median"],
                        "planted_links_min": src["planted_links_min"],
                        "planted_links_max": src["planted_links_max"]})
            if reference_links:
                mdc["pct_of_reference_median"] = round(100.0 * src["planted_links_median"] / reference_links, 2)
                mdc["pct_of_reference_min"] = round(100.0 * src["planted_links_min"] / reference_links, 2)
                mdc["pct_of_reference_max"] = round(100.0 * src["planted_links_max"] / reference_links, 2)
        if ext is not None:
            mdc["extension_k"] = ext["k"]
        drops = [{"from_k": a["k"], "to_k": b["k"], "detected_from": a["detected"], "detected_to": b["detected"]}
                 for a, b in zip(rates, rates[1:]) if a["of"] and b["of"] and b["detected"] < a["detected"]]
        draw_drops = []
        per_draw = {}
        for name in names:
            seq = sorted(table[name][stat], key=lambda c: c["k"])
            seq = [c for c in seq if c["k"] in grid]
            per_draw[name] = next((c["k"] for c in seq if c["detected"]), None)
            for i, c in enumerate(seq):
                if c["detected"]:
                    for later in seq[i + 1:]:
                        if not later["detected"]:
                            draw_drops.append({"draw": name, "detected_at_k": c["k"], "missed_at_k": later["k"]})
                    break
        out[stat] = {"rates": rates, "minimum_detectable_carry": mdc, "monotone": not drops,
                     "non_monotone_steps": drops, "draw_drops": draw_drops, "first_detected_k_per_draw": per_draw}
    return out


def run(rows, real_tokens, plant_seeds, ks=DEFAULT_KS, extension_ks=(), anchors=(), shuffles=N.DEFAULT_SHUFFLES,
        seed=N.DEFAULT_SEED, scope=K.SCOPE_AGENT_DAY, need=None, min_z=DETECT_Z, max_reached=DETECT_MAX_REACHED,
        min_adopters=MIN_ADOPTERS, progress=None):
    """The detection limit. ``anchors``: (name, segments) plant sets reported beside, never counted.

    Returns a counts-only dict: one entry per plant set with every cell, the
    rates over the replicate seeds, and seconds per stage.
    """
    ks, extension_ks = tuple(ks), tuple(extension_ks)
    if not ks or any(k < 0 for k in ks + extension_ks):
        raise ValueError("ks must be a non-empty list of amounts, 0 or more")
    if len(set(plant_seeds)) != len(plant_seeds):
        raise ValueError("plant seeds must differ")
    t0 = time.perf_counter()
    top = max(ks + extension_ks)
    lay = layout(rows)
    plants, planted = [], {}
    for name, segments in anchors:
        p, toks = plant_set(rows, name, segments, lay, min_adopters)
        plants.append(("anchor", p))
        planted.update(toks)
    names = []
    for s in plant_seeds:
        name = "seed_%d" % s
        p, toks = plant_set(rows, name, ((s, top),), lay, min_adopters)
        plants.append(("replicate", p))
        planted.update(toks)
        names.append(name)
    t_plant = time.perf_counter() - t0
    null = injection_null(rows, real_tokens, planted, shuffles, seed, scope, progress)
    t1 = time.perf_counter()
    grid = (0,) + tuple(sorted(set(ks + extension_ks) - {0}))
    table = {p.name: cells(null, p, grid, min_z, max_reached) for _role, p in plants}
    real_cells = {}
    for stat in STATS:
        sm = N.summarize(null.real_sims[stat], null.real_observed[stat])
        real_cells[stat] = {key: sm[key] for key in CELL_KEYS}
        real_cells[stat]["shuffles_le_observed"] = sum(1 for v in null.real_sims[stat]
                                                       if v <= null.real_observed[stat])
    out = {
        "tool": "patient-zero inject",
        "method": "carry cascades planted on real rows (q %.2f, cap %d, horizon %d s, %d+ adopters); the first k "
                  "added to the real tokens; order null, %s scope, %d shuffles, seed %d; detected = z >= %g and at "
                  "most %d shuffles reaching the observed count" % (CARRY_Q, CARRY_CAP, CARRY_HORIZON, min_adopters,
                                                                   scope, shuffles, seed, min_z, max_reached),
        "rows": len(rows),
        "real_tokens": len(real_tokens),
        "shuffles": shuffles,
        "seed": seed,
        "scope": scope,
        "ks": list(ks),
        "extension_ks": list(extension_ks),
        "real": real_cells,
        "plant_sets": [{"name": p.name, "role": role, "segments": [list(s_) for s_ in p.segments],
                        "cascades_drawn": len(p.keys), "cascades_asked": sum(c for _s, c in p.segments),
                        "cells": table[p.name]} for role, p in plants],
        "rates": rate_summary(table, names, ks, extension_ks, need, null.real_observed["aggregate"]),
        "seconds": {"plant": round(t_plant, 1), "shuffles": round(null.seconds, 1),
                    "cells": round(time.perf_counter() - t1, 1), "total": round(time.perf_counter() - t0, 1)},
    }
    return out


# ---------------------------------------------------------------- command line


def _ints(text):
    return tuple(int(x) for x in text.split(",") if x.strip())


def _anchor(text):
    """NAME=SEED:COUNT[,SEED:COUNT...]"""
    name, _eq, rest = text.partition("=")
    if not name or not rest:
        raise argparse.ArgumentTypeError("--anchor wants NAME=SEED:COUNT[,SEED:COUNT]")
    segs = []
    for part in rest.split(","):
        seed, _c, count = part.partition(":")
        segs.append((int(seed), int(count)))
    return name, tuple(segs)


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="Detection limit by replicated injection (counts only).")
    ap.add_argument("--adapter", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--input", required=True, help="export folder (wiki) or JSON Lines file (generic)")
    ap.add_argument("--out", help="write the counts as JSON here (outside the repo, or in its out/ folder)")
    ap.add_argument("--plant-seeds", type=_ints, required=True, help="comma-separated seeds, one draw each")
    ap.add_argument("--ks", type=_ints, default=DEFAULT_KS, help="planted cascades per amount (default 40,60,80,100)")
    ap.add_argument("--extension-ks", type=_ints, default=(), help="larger amounts, reported apart")
    ap.add_argument("--anchor", type=_anchor, action="append", default=[],
                    help="a plant set reported beside, never counted: NAME=SEED:COUNT[,SEED:COUNT]")
    ap.add_argument("--shuffles", type=int, default=N.DEFAULT_SHUFFLES)
    ap.add_argument("--seed", type=int, default=N.DEFAULT_SEED)
    ap.add_argument("--scope", choices=K.SCOPES, default=K.SCOPE_AGENT_DAY)
    ap.add_argument("--need", type=int, default=None, help="draws needed per amount (default 80%%, rounded up)")
    ap.add_argument("--min-agents", type=int, default=MIN_AGENTS)
    ap.add_argument("--max-phrase-share", type=float, default=MAX_PHRASE_SHARE)
    args = ap.parse_args(argv)
    try:
        if args.out:
            from patient_zero.pipeline import check_out_dir  # late import: pipeline imports report
            check_out_dir(os.path.dirname(os.path.abspath(args.out)))
        if args.shuffles < 1:
            raise ValueError("--shuffles must be 1 or more")
        if not args.plant_seeds:
            raise ValueError("--plant-seeds needs at least one seed")
        rows, _stats = load(args.adapter, args.input)
        tokens, _ = find_tokens(rows, min_agents=args.min_agents, max_phrase_share=args.max_phrase_share)
        result = run(rows, tokens, args.plant_seeds, args.ks, args.extension_ks, args.anchor, args.shuffles,
                     args.seed, args.scope, args.need)
    except (OSError, ValueError) as exc:
        print("inject: ERROR: %s" % exc, file=out)
        return 2
    text = json.dumps(result, indent=1, ensure_ascii=True)
    if args.out:
        folder = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(folder, exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
    out.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
