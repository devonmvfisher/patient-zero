"""Robustness checks for adoption links: a fairer null and stricter exposure rules.

Python standard library only. Deterministic: one ``random.Random(seed)`` drives
every shuffle in a fixed order. Opt-in: the main run (``cli.py``, facts.json)
does not call this module. Output is counts only: no row text, no agent
labels, no token keys (tokens appear as ``token_id`` handles).

Why these checks exist
----------------------
The day shuffle in ``patient_zero.nullmodel`` deals each agent's rows of a
day to many labels. One agent that writes the same token on several pages
becomes several adopters under the shuffle, and each later one has usually
"seen" the token on the same page, so the shuffle makes links no real agent
could make (on the wiki: 93,718 adopters under the shuffle against 73,176
observed). That null leans toward more links than chance, so "observed is
below the shuffle" does not show that agents avoid each other's tricks, and
"observed beats the shuffle" is harder than it should be.

Checks
------
Order null (``order_null``)
    Every agent keeps all of its rows (page and text, so the same tokens and
    the same adopters per token) and its own set of times in each UTC day
    (``scope="agent_day"``) or over all days (``scope="agent"``). Only which
    of its rows came at which of its own times is random. Blank and human
    rows never move. If agents pick a trick up after seeing it, observed
    links beat this null. If they bring it with them on arrival, observed
    sits at or below it. Agents with one row in a group cannot move, so their
    links are the same under every shuffle: the ratio is pulled toward 1 and
    ``z`` is the better guide.

Strict exposure (``strict=True``)
    The adoption save itself does not count as the visit. A save's time is
    when the text was sent, not when the page was read, so an agent that
    saved on a page seconds after another agent may never have seen that
    text. Exposure then needs an earlier save by the adopter on the source's
    page.

Minimum lag (``min_lag``)
    Every link and every exposing visit must come more than ``min_lag``
    seconds after the source use, on top of the clock rule.

Activity (``without_top``)
    The most active agents (a share of agents, ranked by rows) become
    exposure-only: they still expose others, but are never counted as
    adopters or credited as sources.

Usage::

    python -m patient_zero.checks --adapter wiki --input PATH --out ../out/checks.json
    python -m patient_zero.checks --adapter generic --input examples/fake_log.ndjson --shuffles 20
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

from patient_zero import nullmodel as N
from patient_zero.load import ADAPTERS, MAP_HELP, load, parse_map
from patient_zero.safe import guard_out, token_id
from patient_zero.tokens import MAX_PHRASE_SHARE, MIN_AGENTS, find_tokens

DAY_SECONDS = N.DAY_SECONDS
SCOPE_AGENT_DAY = "agent_day"
SCOPE_AGENT = "agent"
SCOPES = (SCOPE_AGENT_DAY, SCOPE_AGENT)
DEFAULT_LAGS = (0, 30, 60, 120, 300)
DEFAULT_TOP_SHARES = (0.01, 0.05)
SUMMARY_KEYS = ("observed", "mean", "sd", "p05", "p95", "ratio", "z", "shuffles_ge_observed", "p_value")


# ---------------------------------------------------------------- building blocks


def with_min_lag(index, min_lag):
    """The index with every row's uncertainty raised by ``min_lag / 2`` (a wider clock rule)."""
    if not min_lag:
        return index
    half = float(min_lag) / 2.0
    return index._replace(unc=tuple(u + half for u in index.unc))


def reorder(rows, tokens, rng, scope=SCOPE_AGENT_DAY):
    """One order shuffle. Returns (rows, tokens) re-sorted by the new times.

    Each agent's times are dealt back to its own rows in random order, per
    UTC day (``agent_day``) or over all its rows (``agent``). Rows keep their
    place, text and row id; token rows are re-pointed at the moved rows.
    """
    if scope not in SCOPES:
        raise ValueError("scope must be one of %s" % ", ".join(SCOPES))
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row.exposure_only:
            continue
        day = int(row.time // DAY_SECONDS) if scope == SCOPE_AGENT_DAY else 0
        groups[(row.agent, day)].append(i)
    new_time = [row.time for row in rows]
    for key in sorted(groups):  # fixed order: the same seed gives the same shuffle
        idx = groups[key]
        if len(idx) < 2:
            continue
        times = [rows[i].time for i in idx]
        rng.shuffle(times)
        for i, t in zip(idx, times):
            new_time[i] = t
    order = sorted(range(len(rows)), key=lambda i: (new_time[i], rows[i].row_id))
    where = {old: new for new, old in enumerate(order)}
    new_rows = [rows[i]._replace(time=new_time[i]) for i in order]
    new_tokens = {key: tok._replace(rows=tuple(sorted(where[i] for i in tok.rows)))
                  for key, tok in tokens.items()}
    return new_rows, new_tokens


def _later_share(status):
    later = N.adopters(status) - status["first"]
    return round(status["linked"] / later, 5) if later else None


def _brief(summary):
    return {k: summary[k] for k in SUMMARY_KEYS}


def _lead_position(index, lead):
    if lead is None:
        return None
    for t, key in enumerate(index.keys):
        if token_id(key) == lead:
            return t
    return None


# ---------------------------------------------------------------- the checks


def observed_variants(index, lags=DEFAULT_LAGS, lead_t=None):
    """Observed named links under each minimum lag, with and without strict exposure."""
    out = {}
    for strict in (False, True):
        for lag in lags:
            ix = with_min_lag(index, lag)
            count = N.count_links(ix, ix.agent, strict=strict)
            entry = {"links_named": count.status["linked"],
                     "linked_share_of_later_adopters": _later_share(count.status)}
            if lead_t is not None:
                entry["lead_links_named"] = count.named[lead_t]
            out["%s_min_lag_%ds" % ("strict" if strict else "default", lag)] = entry
    return out


def without_top(rows, index, shares=DEFAULT_TOP_SHARES):
    """Observed counts with the most active agents (by rows) made exposure-only."""
    per_agent = Counter(row.agent for row in rows if not row.exposure_only)
    ranked = [a for a, _n in sorted(per_agent.items(), key=lambda kv: (-kv[1], kv[0]))]
    ids = {name: i for i, name in enumerate(index.agent_names)}
    total_rows = sum(per_agent.values())
    out = {}
    for share in shares:
        n = max(1, int(len(ranked) * share))
        drop = {ids[a] for a in ranked[:n]}
        labels = [(-1 if a in drop else a) for a in index.agent]
        count = N.count_links(index, labels)
        out["top_%g_pct" % (share * 100)] = {
            "agents_made_exposure_only": n,
            "their_share_of_agent_rows": round(sum(per_agent[a] for a in ranked[:n]) / total_rows, 4)
            if total_rows else None,
            "links_named": count.status["linked"],
            "links_from_exposure_only_rows": count.status["unnamed"],
            "adopters": N.adopters(count.status),
            "linked_share_of_later_adopters": _later_share(count.status),
        }
    return out


def order_null(rows, tokens, shuffles=N.DEFAULT_SHUFFLES, seed=N.DEFAULT_SEED, scope=SCOPE_AGENT_DAY,
               min_lag=0, strict=False, lead=None, index=None):
    """Observed links against ``shuffles`` order shuffles. Returns a counts-only dict."""
    if shuffles < 1:
        raise ValueError("shuffles must be 1 or more")
    started = time.perf_counter()
    index = with_min_lag(index or N.build_index(rows, tokens), min_lag)
    observed = N.count_links(index, index.agent, strict=strict)
    rng = random.Random(seed)
    linked, shares, per_token = [], [], []
    for _ in range(shuffles):
        new_rows, new_tokens = reorder(rows, tokens, rng, scope)
        ix = with_min_lag(N.build_index(new_rows, new_tokens), min_lag)
        count = N.count_links(ix, ix.agent, strict=strict)
        linked.append(count.status["linked"])
        shares.append(_later_share(count.status) or 0.0)
        per_token.append(count.named)
    obs_named = observed.named
    n_tok = len(obs_named)
    beats = sum(1 for t in range(n_tok) if obs_named[t] > 0 and all(p[t] < obs_named[t] for p in per_token))
    below = sum(1 for t in range(n_tok) if all(p[t] > obs_named[t] for p in per_token))
    out = {
        "scope": scope,
        "shuffles": shuffles,
        "seed": seed,
        "min_lag_seconds": min_lag,
        "strict_exposure": strict,
        "adopters": N.adopters(observed.status),
        "links_named": _brief(N.summarize(linked, observed.status["linked"])),
        "linked_share_of_later_adopters": _brief(N.summarize(shares, _later_share(observed.status) or 0.0)),
        "trees_above_every_shuffle": beats,
        "trees_below_every_shuffle": below,
        "chance_bound_each_way": round(n_tok / (shuffles + 1), 2),
    }
    by_class = defaultdict(list)
    for t, cls in enumerate(index.classes):
        by_class[cls].append(t)
    out["by_class"] = {}
    for cls in sorted(by_class):
        ts = by_class[cls]
        values = [sum(p[t] for t in ts) for p in per_token]
        out["by_class"][cls] = _brief(N.summarize(values, sum(obs_named[t] for t in ts)))
    lead_t = _lead_position(index, lead)
    if lead_t is not None:
        values = [p[lead_t] for p in per_token]
        out["lead"] = {"token_id": lead, "observed": obs_named[lead_t],
                       "mean": round(sum(values) / len(values), 3),
                       "shuffles_ge_observed": sum(1 for v in values if v >= obs_named[lead_t])}
    out["seconds"] = round(time.perf_counter() - started, 1)
    return out


def run_checks(rows, tokens, shuffles=N.DEFAULT_SHUFFLES, seed=N.DEFAULT_SEED, lead=None,
               variant_shuffles=None, lags=DEFAULT_LAGS):
    """Every check on one set of rows and tokens. Returns a counts-only dict."""
    index = N.build_index(rows, tokens)
    lead_t = _lead_position(index, lead)
    small = variant_shuffles if variant_shuffles is not None else max(1, shuffles // 3)
    return {
        "tool": "patient-zero checks",
        "rows": len(rows),
        "tokens": len(tokens),
        "observed": observed_variants(index, lags, lead_t),
        "without_most_active": without_top(rows, index),
        "order_null": order_null(rows, tokens, shuffles, seed, SCOPE_AGENT_DAY, lead=lead, index=index),
        "order_null_all_days": order_null(rows, tokens, small, seed, SCOPE_AGENT, lead=lead, index=index),
        "order_null_min_lag_120s": order_null(rows, tokens, small, seed, SCOPE_AGENT_DAY, min_lag=120,
                                              lead=lead, index=index),
        "order_null_strict": order_null(rows, tokens, small, seed, SCOPE_AGENT_DAY, strict=True,
                                        lead=lead, index=index),
    }


# ---------------------------------------------------------------- command line


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="Robustness checks for adoption links (counts only).")
    ap.add_argument("--adapter", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--input", required=True,
                    help="export folder (wiki) or a JSON Lines, CSV or TSV file (generic)")
    ap.add_argument("--map", help=MAP_HELP)
    ap.add_argument("--out", help="write the counts as JSON here (outside the repo, or in its out/ folder; "
                                  "never the input or inside the input folder)")
    ap.add_argument("--shuffles", type=int, default=N.DEFAULT_SHUFFLES)
    ap.add_argument("--variant-shuffles", type=int, default=None,
                    help="shuffles for the three variant nulls (default: a third of --shuffles)")
    ap.add_argument("--seed", type=int, default=N.DEFAULT_SEED)
    ap.add_argument("--lead", help="token_id of a tree to follow through every check")
    ap.add_argument("--min-agents", type=int, default=MIN_AGENTS)
    ap.add_argument("--max-phrase-share", type=float, default=MAX_PHRASE_SHARE)
    args = ap.parse_args(argv)
    try:
        if args.out:
            from patient_zero.pipeline import check_out_dir  # late import: pipeline imports report
            check_out_dir(os.path.dirname(os.path.abspath(args.out)))
            if os.path.isdir(args.out):
                raise ValueError("--out names %s, which is a folder; give a file name such as "
                                 "checks.json" % args.out)
            guard_out(args.out, [args.input], "--out")
        if args.shuffles < 1:
            raise ValueError("--shuffles must be 1 or more")
        rows, _stats = load(args.adapter, args.input, parse_map(args.map))
    except (OSError, ValueError) as exc:
        print("checks: ERROR: %s" % exc, file=out)
        return 2
    tokens, _ = find_tokens(rows, min_agents=args.min_agents, max_phrase_share=args.max_phrase_share)
    result = run_checks(rows, tokens, args.shuffles, args.seed, args.lead, args.variant_shuffles)
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
