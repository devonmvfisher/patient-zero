"""Cross-page grid: cross-page links under the order null, by token class and by exposure horizon.

Python standard library only. Deterministic: one ``random.Random(seed)`` drives
every order shuffle (the same permutations as ``patient_zero.checks``), and
every control set has its own fixed seed. Opt-in: the main run (``cli.py``,
facts.json) and ``checks.py`` never call this module. Output is counts only:
no row text, no agent labels, no token keys.

Why it exists
-------------
Under the order null (``checks.order_null``) the cross-page part of the links
reads below chance. A careful reader asks two things: is that only phrases (the
biggest class), and is it only the exposure rule we chose? This module answers
both at once, without picking: every token class, every horizon, every cell
with its count and z, and random-row controls that show whether the statistic
itself is unbiased at each horizon.

Definitions
-----------
Cross-page link
    A named link whose credited source row is on a different page from the
    adopter's first-use row (same-page links are reported beside).
Exposure horizon
    The tracer itself has no horizon: it credits the most recent visible
    earlier use, however old (column ``none``). A link counts at horizon H when
    the adopter's first use comes no more than H seconds after the credited
    source use. The credited use is the most recent one that qualifies, so an
    older use can never qualify when it fails: the filter never changes who is
    credited, and the cells nest (24 h <= 72 h <= 7 d <= none) in the observed
    count and in every shuffle.
Classes
    The tracer's six token classes (``tokens.CLASSES``). The class cells sum to
    the pooled total at every horizon.
Controls
    Token sets that are not real tokens, counted under the same shuffles:
    ``calibration`` sets (random agent rows, no structure: an unbiased statistic
    gives |z| < 2) and ``power`` sets (planted carry cascades: a statistic with
    power gives z >= 3). A horizon is calibrated only when every calibration
    set passes there.

Readings (descriptive; they never change a pre-registered headline)
    Each cell: "above" if z >= 3 and at most 1 shuffle in 100 reaches the
    observed count; "below" if z <= -3 and at most 1 in 100 is at or below it;
    otherwise "at chance". Breadth: the class with the largest expected count
    (L) against all other classes pooled (REST), see ``breadth``.

Usage::

    python -m patient_zero.crossgrid --adapter wiki --input PATH --out ../out/crossgrid.json
    python -m patient_zero.crossgrid --adapter generic --input examples/fake_log.ndjson --shuffles 20
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from statistics import stdev
from typing import NamedTuple, Tuple

from patient_zero import checks as K
from patient_zero import nullmodel as N
from patient_zero.load import ADAPTERS, load
from patient_zero.tokens import CLASSES, MAX_PHRASE_SHARE, MIN_AGENTS, find_tokens

HOUR = 3600.0
HORIZONS = (("24h", 24 * HOUR), ("72h", 72 * HOUR), ("7d", 168 * HOUR), ("none", None))
NO_HORIZON = "none"  # the tracer's own rule: no age limit on the credited use
CONTROL_PREFIX = "control:"
ROLE_CALIBRATION = "calibration"
ROLE_POWER = "power"
ROLES = (ROLE_CALIBRATION, ROLE_POWER)
# (name, tokens, rows per token, seed): C1 is the E7c "random" set, C2 a larger one.
RANDOM_CONTROLS = (("random_c1", 40, 10, 20261031), ("random_c2", 400, 10, 20261041))
# The 40 carry cascades of E7b/E7c (max/prong-b exp2 generator, first loop of its seed).
CARRY_CONTROL = ("carry_k", 40, 20261023)
SUMMARY_KEYS = ("observed", "mean", "sd", "p05", "p95", "ratio", "z", "shuffles_ge_observed",
                "shuffles_le_observed")


class ControlToken(NamedTuple):
    """A control token: rows only (no text). ``cls`` starts with ``control:``."""

    key: str
    cls: str
    rows: Tuple[int, ...]


# ---------------------------------------------------------------- control sets


def random_row_tokens(rows, name, count, size, seed):
    """``count`` tokens, each on ``size`` agent rows drawn at random (E7c's sampler).

    Rows are drawn with ``random.Random(seed).sample`` over the agent rows in
    load order, then de-duplicated and sorted, exactly as E7c drew its
    "random" set. Exposure-only rows are never drawn.
    """
    agent_rows = [i for i, r in enumerate(rows) if not r.exposure_only]
    if len(agent_rows) < size:
        raise ValueError("need at least %d agent rows for a random control" % size)
    rng = random.Random(seed)
    out = {}
    for j in range(count):
        key = "%s%s:%03d" % (CONTROL_PREFIX, name, j)
        out[key] = ControlToken(key, CONTROL_PREFIX + name, tuple(sorted(set(rng.sample(agent_rows, size)))))
    return out


def planted_carry_tokens(rows, name, count, seed, q=0.35, cap=40, horizon=72 * HOUR, max_tries=20000):
    """``count`` planted carry cascades: each adopter sees the token on a page, then first uses it elsewhere.

    The generator of max/prong-b exp2.py with carry probability 1 (the same
    logic as kirk/cards/km-run E7b and E7c, so the same seed plants the same
    cascades). A cascade spreads from a random agent row to later agents on the
    same page (probability ``q`` each, within ``horizon``, clock rule
    respected); each adopter's use is moved to its own next row on another page
    within ``horizon``. Kept when it reaches at least 4 adopters.
    """
    agent_rows = [i for i, r in enumerate(rows) if not r.exposure_only]
    if not agent_rows:
        raise ValueError("no agent rows for a planted control")
    by_place = defaultdict(list)
    by_agent = defaultdict(list)
    for i, r in enumerate(rows):
        by_place[r.place].append(i)
        if not r.exposure_only:
            by_agent[r.agent].append(i)

    def clock_after(u, v):
        return rows[v].time - rows[u].time > rows[u].uncertainty + rows[v].uncertainty

    def plant(rng, seed_row):
        infected = {rows[seed_row].agent}
        use_rows = [seed_row]
        parents = 0
        frontier = [seed_row]
        while frontier and len(infected) < cap:
            u = frontier.pop(0)
            for v in by_place[rows[u].place]:
                if v <= u or rows[v].exposure_only or not clock_after(u, v):
                    continue
                if rows[v].time - rows[u].time > horizon:
                    break
                a = rows[v].agent
                if a in infected or rng.random() >= q:
                    continue
                rng.random()  # exp2's carry draw (carry probability 1 always carries)
                nxt = [w for w in by_agent[a] if w > v and rows[w].place != rows[v].place
                       and clock_after(v, w) and rows[w].time - rows[v].time < horizon]
                if not nxt:
                    continue
                adopt = nxt[0]
                if any(rows[x].place == rows[adopt].place for x in use_rows):
                    continue
                infected.add(a)
                parents += 1
                use_rows.append(adopt)
                frontier.append(adopt)
                if len(infected) >= cap:
                    break
        return use_rows, parents

    rng = random.Random(seed)
    out = {}
    made = tries = 0
    while made < count and tries < max_tries:
        tries += 1
        use_rows, parents = plant(rng, rng.choice(agent_rows))
        if parents >= 4:
            key = "%s%s:%03d" % (CONTROL_PREFIX, name, made)
            out[key] = ControlToken(key, CONTROL_PREFIX + name, tuple(sorted(set(use_rows))))
            made += 1
    return out


def default_controls(rows):
    """[(name, role, tokens)]: the two random-row sets and the planted carry set."""
    out = [(name, ROLE_CALIBRATION, random_row_tokens(rows, name, count, size, seed))
           for name, count, size, seed in RANDOM_CONTROLS]
    name, count, seed = CARRY_CONTROL
    out.append((name, ROLE_POWER, planted_carry_tokens(rows, name, count, seed)))
    return out


# ---------------------------------------------------------------- counting


def link_lags(index, labels):
    """Per token: (cross-page lags, same-page lags, adopters) of its named links under ``labels``.

    A lag is the adopter's first-use time minus the credited source use's time
    (seconds), on ``index``'s own times.
    """
    count = N.count_links(index, labels, detail=True)
    times, place = index.times, index.place
    out = []
    for t, det in enumerate(count.detail):
        firsts = []
        seen = set()
        for i in index.uses[t]:
            a = labels[i]
            if a >= 0 and a not in seen:
                seen.add(a)
                firsts.append(i)
        if len(firsts) != len(det):
            raise RuntimeError("detail does not match first uses")
        cross, same = [], []
        for (a, status, src), f in zip(det, firsts):
            if a != labels[f]:
                raise RuntimeError("detail order does not match first uses")
            if status != "linked":
                continue
            lag = times[f] - times[src]
            (cross if place[src] != place[f] else same).append(lag)
        out.append((cross, same, len(det)))
    return out, count


def _within(lags, seconds):
    if seconds is None:
        return len(lags)
    return sum(1 for x in lags if x <= seconds)


def cell_counts(index, labels, horizons=HORIZONS):
    """{group: {"cross": [per horizon], "same": [per horizon], "tokens", "adopters"}} for one labelling.

    The group is the token's class (controls: their ``control:`` class). Also
    returns, for the real classes only, the pooled cross and same counts per
    horizon computed in a separate pass, and the tracer's own named-link total.
    """
    lags, count = link_lags(index, labels)
    groups = {}
    pooled_cross = [0] * len(horizons)
    pooled_same = [0] * len(horizons)
    named_real = 0
    for t, (cross, same, adopters) in enumerate(lags):
        cls = index.classes[t]
        g = groups.get(cls)
        if g is None:
            g = groups[cls] = {"cross": [0] * len(horizons), "same": [0] * len(horizons),
                               "tokens": 0, "adopters": 0}
        g["tokens"] += 1
        g["adopters"] += adopters
        for h, (_name, seconds) in enumerate(horizons):
            g["cross"][h] += _within(cross, seconds)
            g["same"][h] += _within(same, seconds)
    for t, (cross, same, _adopters) in enumerate(lags):  # separate pass, no class grouping
        if index.classes[t].startswith(CONTROL_PREFIX):
            continue
        named_real += count.named[t]
        for h, (_name, seconds) in enumerate(horizons):
            pooled_cross[h] += _within(cross, seconds)
            pooled_same[h] += _within(same, seconds)
    return {"groups": groups, "pooled_cross": pooled_cross, "pooled_same": pooled_same,
            "named_real": named_real}


# ---------------------------------------------------------------- summaries and readings


def summary(values, observed):
    """Observed against shuffle values (``nullmodel.summarize``) plus shuffles at or below it."""
    sm = N.summarize(values, observed)
    out = {k: sm.get(k) for k in SUMMARY_KEYS if k != "shuffles_le_observed"}
    out["shuffles_le_observed"] = sum(1 for v in values if v <= observed) if values else None
    return out


def reading(sm):
    """The PREREG-v2 V rule on one summary: above, below or at chance."""
    z = sm.get("z")
    n = sm.get("shuffles_ge_observed")
    if z is None or n is None:
        return "at chance (no spread)"
    if z >= 3 and sm["shuffles_ge_observed"] <= 1:
        return "above"
    if z <= -3 and sm["shuffles_le_observed"] <= 1:
        return "below"
    return "at chance"


def calibration_pass(sm):
    """|z| < 2; with no spread, a pass only when observed equals the null mean."""
    z = sm.get("z")
    if z is None:
        return sm.get("mean") is not None and sm["observed"] == sm["mean"]
    return abs(z) < 2


def breadth(obs_total, sim_total, obs_rest, sim_rest):
    """Is REST's shortfall proportional to the total's? Returns z_null, z_prop and a reading.

    R = total observed / total mean. z_null = (REST observed - REST mean) / REST sd;
    z_prop = (REST observed - R * REST mean) / REST sd. First match wins:
    REST above chance (z_null >= 2), REST carries more (z_prop <= -2), broad
    (z_null <= -2, |z_prop| < 2), partly broad (z_null <= -2, z_prop >= 2),
    driven by L (z_prop >= 2), undetermined (low power) otherwise.
    """
    n = len(sim_total)
    if n < 2:
        return {"R": None, "z_null": None, "z_prop": None, "reading": "undetermined (too few shuffles)"}
    mean_t = sum(sim_total) / n
    mean_r = sum(sim_rest) / n
    sd = stdev(sim_rest)
    R = obs_total / mean_t if mean_t > 0 else None
    if R is None or sd <= 0:
        return {"R": None if R is None else round(R, 4), "z_null": None, "z_prop": None,
                "reading": "undetermined (no spread)"}
    z_null = (obs_rest - mean_r) / sd
    z_prop = (obs_rest - R * mean_r) / sd
    if z_null >= 2:
        label = "REST above chance"
    elif z_prop <= -2:
        label = "REST carries more"
    elif z_null <= -2 and abs(z_prop) < 2:
        label = "broad"
    elif z_null <= -2:
        label = "partly broad"
    elif z_prop >= 2:
        label = "driven by L"
    else:
        label = "undetermined (low power)"
    return {"R": round(R, 4), "rest_expected_if_proportional": round(R * mean_r, 2),
            "z_null": round(z_null, 2), "z_prop": round(z_prop, 2), "reading": label}


# ---------------------------------------------------------------- the grid


def grid(rows, tokens, controls=(), shuffles=N.DEFAULT_SHUFFLES, seed=N.DEFAULT_SEED,
         scope=K.SCOPE_AGENT_DAY, horizons=HORIZONS):
    """Cross-page links by class and horizon against ``shuffles`` order shuffles. Counts only.

    ``controls`` is a sequence of (name, role, tokens) with role "calibration"
    or "power"; control tokens must carry a ``control:`` class.
    """
    if shuffles < 1:
        raise ValueError("shuffles must be 1 or more")
    names = [h[0] for h in horizons]
    if len(set(names)) != len(names):
        raise ValueError("horizon names must be distinct")
    started = time.perf_counter()
    alltok = dict(tokens)
    roles = {}
    for name, role, toks in controls:
        if role not in ROLES:
            raise ValueError("control role must be one of %s" % ", ".join(ROLES))
        cls = CONTROL_PREFIX + name
        for key, tok in toks.items():
            if tok.cls != cls or key in alltok:
                raise ValueError("control tokens need class %r and keys of their own" % cls)
        roles[cls] = (name, role, len(toks))
        alltok.update(toks)
    for tok in tokens.values():
        if tok.cls.startswith(CONTROL_PREFIX):
            raise ValueError("real tokens may not use the control class prefix")

    index = N.build_index(rows, alltok)
    obs = cell_counts(index, index.agent, horizons)
    rng = random.Random(seed)
    sims = []
    for _ in range(shuffles):
        new_rows, new_tokens = K.reorder(rows, alltok, rng, scope)
        ix = N.build_index(new_rows, new_tokens)
        sims.append(cell_counts(ix, ix.agent, horizons))

    def series(group, side, h, source=None):
        src = obs if source is None else source
        g = src["groups"].get(group)
        return g[side][h] if g else 0

    real_classes = list(CLASSES)
    out = {
        "tool": "patient-zero cross-page grid",
        "null": {"kind": "order", "scope": scope, "shuffles": shuffles, "seed": seed},
        "horizons": [{"name": n, "seconds": s} for n, s in horizons],
        "horizon_rule": "a link counts at horizon H when the adopter's first use is no more than H seconds "
                        "after the credited source use; 'none' is the tracer's own rule (no limit)",
        "rows": len(rows),
        "tokens": len(tokens),
        "classes": {},
    }

    # checks: class cells sum to the pooled total (separate pass), nesting, tracer total
    sums_ok, nest_ok = {}, True
    for h, name in enumerate(names):
        ok = True
        for src in [obs] + sims:
            cross_sum = sum(series(c, "cross", h, src) for c in real_classes)
            same_sum = sum(series(c, "same", h, src) for c in real_classes)
            ok = ok and cross_sum == src["pooled_cross"][h] and same_sum == src["pooled_same"][h]
        sums_ok[name] = ok
    order = sorted(range(len(horizons)), key=lambda h: (horizons[h][1] is None, horizons[h][1] or 0))
    for src in [obs] + sims:
        for g in src["groups"].values():
            for side in ("cross", "same"):
                vals = [g[side][h] for h in order]
                nest_ok = nest_ok and all(a <= b for a, b in zip(vals, vals[1:]))
    none_h = names.index(NO_HORIZON) if NO_HORIZON in names else None
    tracer_ok = (none_h is None or all(src["pooled_cross"][none_h] + src["pooled_same"][none_h]
                                       == src["named_real"] for src in [obs] + sims))
    out["checks"] = {"class_cells_sum_to_total": sums_ok, "horizons_nest": nest_ok,
                     "cross_plus_same_equals_tracer_links": tracer_ok}

    # cells per class
    for cls in real_classes:
        g = obs["groups"].get(cls, {"tokens": 0, "adopters": 0})
        entry = {"tokens": g["tokens"], "adopters": g["adopters"], "cross_page": {}, "same_page": {}}
        for h, name in enumerate(names):
            for side, label in (("cross", "cross_page"), ("same", "same_page")):
                sm = summary([series(cls, side, h, s) for s in sims], series(cls, side, h))
                if side == "cross":
                    sm["reading"] = reading(sm)
                entry[label][name] = sm
        out["classes"][cls] = entry

    total = {"cross_page": {}, "same_page": {}}
    for h, name in enumerate(names):
        for side, label in (("cross", "cross_page"), ("same", "same_page")):
            key = "pooled_" + side
            sm = summary([s[key][h] for s in sims], obs[key][h])
            if side == "cross":
                sm["reading"] = reading(sm)
            total[label][name] = sm
    total["tracer_named_links"] = obs["named_real"]
    out["total"] = total

    # breadth: the class with the largest expected cross-page count (no-horizon column) against the rest
    ref_h = none_h if none_h is not None else len(names) - 1
    means = {c: out["classes"][c]["cross_page"][names[ref_h]]["mean"] or 0 for c in real_classes}
    largest = max(real_classes, key=lambda c: (means[c], c))
    out["breadth"] = {"L": largest, "L_chosen_by": "largest null-mean cross-page count in column '%s'"
                      % names[ref_h], "by_horizon": {}, "drop_one": {}}
    for h, name in enumerate(names):
        sim_total = [s["pooled_cross"][h] for s in sims]
        sim_rest = [s["pooled_cross"][h] - series(largest, "cross", h, s) for s in sims]
        obs_rest = obs["pooled_cross"][h] - series(largest, "cross", h)
        b = breadth(obs["pooled_cross"][h], sim_total, obs_rest, sim_rest)
        rest_sm = summary(sim_rest, obs_rest)
        rest_sm["reading"] = reading(rest_sm)
        b["rest"] = rest_sm
        out["breadth"]["by_horizon"][name] = b
        drops = {}
        for cls in real_classes:
            vals = [s["pooled_cross"][h] - series(cls, "cross", h, s) for s in sims]
            sm = summary(vals, obs["pooled_cross"][h] - series(cls, "cross", h))
            sm["reading"] = reading(sm)
            drops["without_" + cls] = {k: sm[k] for k in ("observed", "mean", "sd", "ratio", "z", "reading")}
        out["breadth"]["drop_one"][name] = drops

    # controls
    out["controls"] = {}
    calibrated = {name: True for name in names}
    for cls, (cname, role, n_tok) in roles.items():
        g = obs["groups"].get(cls, {"tokens": 0, "adopters": 0})
        entry = {"role": role, "tokens": n_tok, "adopters": g["adopters"], "cross_page": {}, "same_page": {}}
        for h, name in enumerate(names):
            sm = summary([series(cls, "cross", h, s) for s in sims], series(cls, "cross", h))
            if role == ROLE_CALIBRATION:
                sm["pass"] = calibration_pass(sm)
                calibrated[name] = calibrated[name] and sm["pass"]
            else:
                sm["powered"] = sm["z"] is not None and sm["z"] >= 3
            entry["cross_page"][name] = sm
            entry["same_page"][name] = summary([series(cls, "same", h, s) for s in sims], series(cls, "same", h))
        if none_h is not None:  # every link of the control under the tracer's own rule, shuffle for shuffle
            entry["all_links"] = summary([series(cls, "cross", none_h, s) + series(cls, "same", none_h, s)
                                          for s in sims],
                                         series(cls, "cross", none_h) + series(cls, "same", none_h))
        out["controls"][cname] = entry
    has_cal = any(role == ROLE_CALIBRATION for _c, role, _n in roles.values())
    out["horizon_calibrated"] = calibrated if has_cal else {name: None for name in names}
    out["seconds"] = round(time.perf_counter() - started, 1)
    return out


# ---------------------------------------------------------------- text view


def text_table(result):
    """A plain table of the cross-page cells: n, observed, mean, z and reading per class and horizon."""
    names = [h["name"] for h in result["horizons"]]
    lines = ["cross-page links under the order null (%d shuffles, seed %d), by class and horizon"
             % (result["null"]["shuffles"], result["null"]["seed"])]
    lines.append("%-11s %6s | %s" % ("class", "tokens", " | ".join("%-27s" % n for n in names)))

    def cell(sm):
        z = "%.2f" % sm["z"] if sm["z"] is not None else "-"
        return "%-27s" % ("%d vs %.1f z %s" % (sm["observed"], sm["mean"] or 0.0, z))

    for cls, entry in result["classes"].items():
        lines.append("%-11s %6d | %s" % (cls, entry["tokens"],
                                         " | ".join(cell(entry["cross_page"][n]) for n in names)))
    lines.append("%-11s %6d | %s" % ("TOTAL", result["tokens"],
                                     " | ".join(cell(result["total"]["cross_page"][n]) for n in names)))
    lines.append("reading:    %6s | %s" % ("", " | ".join("%-27s" % result["total"]["cross_page"][n]["reading"]
                                                         for n in names)))
    lines.append("breadth (L = %s): %s" % (result["breadth"]["L"], ", ".join(
        "%s %s" % (n, result["breadth"]["by_horizon"][n]["reading"]) for n in names)))
    for cname, entry in result["controls"].items():
        lines.append("control %s (%s, %d tokens): %s" % (cname, entry["role"], entry["tokens"], ", ".join(
            "%s z %s" % (n, entry["cross_page"][n]["z"]) for n in names)))
    lines.append("calibrated: %s" % ", ".join("%s %s" % (n, result["horizon_calibrated"][n]) for n in names))
    lines.append("checks: %s" % json.dumps(result["checks"], sort_keys=True))
    return "\n".join(lines)


# ---------------------------------------------------------------- command line


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="Cross-page links by class and exposure horizon under the order "
                                             "null, with random-row and planted-carry controls (counts only).")
    ap.add_argument("--adapter", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--input", required=True, help="export folder (wiki) or JSON Lines file (generic)")
    ap.add_argument("--out", help="write the counts as JSON here (outside the repo, or in its out/ folder)")
    ap.add_argument("--shuffles", type=int, default=N.DEFAULT_SHUFFLES)
    ap.add_argument("--seed", type=int, default=N.DEFAULT_SEED)
    ap.add_argument("--no-controls", action="store_true", help="skip the random-row and planted-carry controls")
    ap.add_argument("--min-agents", type=int, default=MIN_AGENTS)
    ap.add_argument("--max-phrase-share", type=float, default=MAX_PHRASE_SHARE)
    args = ap.parse_args(argv)
    try:
        if args.out:
            from patient_zero.pipeline import check_out_dir  # late import: pipeline imports report
            check_out_dir(os.path.dirname(os.path.abspath(args.out)))
        if args.shuffles < 1:
            raise ValueError("--shuffles must be 1 or more")
        rows, _stats = load(args.adapter, args.input)
        tokens, _ = find_tokens(rows, min_agents=args.min_agents, max_phrase_share=args.max_phrase_share)
        controls = () if args.no_controls else default_controls(rows)
    except (OSError, ValueError) as exc:
        print("crossgrid: ERROR: %s" % exc, file=out)
        return 2
    result = grid(rows, tokens, controls, args.shuffles, args.seed)
    text = json.dumps(result, indent=1, ensure_ascii=True)
    if args.out:
        folder = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(folder, exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
    out.write(text_table(result) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
