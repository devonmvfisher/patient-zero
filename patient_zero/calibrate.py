"""Calibration bench: plant known kinds of spread on a log's own rows and see which null catches each.

Python standard library only. Deterministic: every planted kind has its own fixed seed and every null uses
one ``random.Random(seed)`` in a fixed order. Output is counts only: no row text, no token text, no agent
label.

Why it exists
-------------
A null that cannot see a kind of spread reads "at chance" whether that spread is there or not. So every
verdict needs the null's power measured on the same background, with known answers. This module plants
known kinds on real rows (real agents, pages and times; only the token placement is planted) and scores
each kind under every null. It is the repo port of the KM red-team run E7 (kirk/cards/km-run/
e7_calibrate.py); the E6 and exp2 generators are re-implemented line for line, with their seeds, so the
plantings and every matrix cell reproduce.

Planted kinds (40 tokens each)
------------------------------
contagion         E6: an infected agent's save on page P infects any later writer on P within 72 h,
                  each with probability 0.35 (cap 40 agents). Every adoption sits on one page.
page_instruction  NEW (added after E7, labelled so): a page-wide shared cause. An instruction appears on
                  page P at time t0; every writer on P in [t0, t0 + 72 h] follows it with probability 0.35,
                  each on its own coin, with no dependence on anyone's earlier use. It is built to be
                  the twin of same-page contagion that has no contact at all.
same, mixed, carry  exp2: same-page spread, half carried, or always carried (seen on P, first used on Q).
decoy_fam, decoy_fam2  shared cause: 10 agents of one page-family day, each on its first row that day.
decoy_bcast       shared cause: 10 agents of one busy day, each on its first row that day.
random            10 random agent rows, no structure.
repeat            every row of 4 busy agent-days (4 agents), no contact.
pool_contagion, pool_carry  160 more cascades of each mechanism, for the power curve only.

Nulls
-----
N1 label day, N2 label day x page family (``patient_zero.nullmodel``); N3 order null per agent-day,
N4 the same with strict exposure, N5 the same with a 120 s minimum lag (``patient_zero.checks``); and the
reversed clock (forward against reversed links per token; no shuffle).

The reversed clock gets a z too: for each token, d = forward links - reversed links, and z is the mean of d
over the kind's tokens divided by its standard error (a paired t over tokens, the independent units). When
every token has the same d, z is 0 if d is 0 and unbounded (``None``) otherwise.
"""

import hashlib
import math
import random
import time
from collections import Counter, defaultdict
from statistics import stdev
from typing import NamedTuple, Tuple

from patient_zero import checks as K
from patient_zero import nullmodel as N

SEED = N.DEFAULT_SEED  # 20261003: every shuffle
DAY = N.DAY_SECONDS
Q = 0.35
CAP = 40
HORIZON = 3 * 86400
PER_KIND = 40
POOL = 160
POWER_K = (0, 10, 20, 40, 80, 120, 160, 200)

PAGE_INSTRUCTION_SEED = 20261035
PAGE_INSTRUCTION_MIN_FOLLOWERS = 5  # E6 keeps a cascade with its seed plus at least 4 adopters: 5 users
PAGE_INSTRUCTION_TRIES = 20000

NULL_NAMES = ("N1_label_day", "N2_label_day_x_family", "N3_order_agent_day", "N4_order_strict",
              "N5_order_lag120")
# Matrix rows in display order: the page-wide shared cause sits directly under same-page contagion.
MATRIX_ORDER = ("contagion", "page_instruction", "same", "mixed", "carry", "decoy_fam", "decoy_fam2",
                "decoy_bcast", "random", "repeat")
KIND_LABELS = {
    "contagion": "Same-page contagion (E6)",
    "page_instruction": "Page instruction: page-wide shared cause, no contact (NEW)",
    "same": "Same-page (exp2 \"same\")",
    "mixed": "Mixed: half carried (exp2)",
    "carry": "Carry: seen on page P, used on page Q (exp2)",
    "decoy_fam": "Shared cause, family-day first rows (E6 decoy)",
    "decoy_fam2": "Shared cause, family-day (exp2 decoy)",
    "decoy_bcast": "Shared cause, broadcast to one day (exp2 decoy)",
    "random": "Random rows",
    "repeat": "Self-repeat, no contact",
}
SUMMARY_KEYS = ("observed", "mean", "sd", "p05", "p95", "ratio", "z", "shuffles_ge_observed")


class PTok(NamedTuple):
    """A planted token: the parts ``nullmodel.build_index`` and ``checks.reorder`` read."""

    key: str
    cls: str
    rows: Tuple[int, ...]


class Background:
    """The log's own rows, indexed for planting."""

    def __init__(self, rows):
        self.rows = rows
        self.agent_rows = [i for i, r in enumerate(rows) if not r.exposure_only]
        self.by_place = defaultdict(list)
        self.by_agent = defaultdict(list)
        for i, r in enumerate(rows):
            self.by_place[r.place].append(i)
            if not r.exposure_only:
                self.by_agent[r.agent].append(i)

    def clock_after(self, u, v):
        rows = self.rows
        return rows[v].time - rows[u].time > rows[u].uncertainty + rows[v].uncertainty


# ---------------------------------------------------------------- generators


def plant_contagion(bg, rng, seed_row, q=Q, cap=CAP, horizon=HORIZON):
    """E6 (max/prong-b/exp.py): the adoption row is the later save on an infected row's page."""
    rows, by_place = bg.rows, bg.by_place
    infected = {rows[seed_row].agent: (seed_row, None)} if rows[seed_row].agent else {}
    use_rows = [seed_row]
    parent = {}
    frontier = [seed_row]
    while frontier and len(infected) < cap:
        u = frontier.pop(0)
        for v in by_place[rows[u].place]:
            if v <= u or rows[v].exposure_only or not bg.clock_after(u, v):
                continue
            if rows[v].time - rows[u].time > horizon:
                break
            a = rows[v].agent
            if a in infected:
                continue
            if rng.random() < q:
                infected[a] = (v, rows[u].agent)
                parent[a] = rows[u].agent
                use_rows.append(v)
                frontier.append(v)
                if len(infected) >= cap:
                    break
    return use_rows, parent


def plant_exp2(bg, rng, seed_row, carry_p, q=Q, cap=CAP, horizon=HORIZON):
    """exp2 (max/prong-b/exp2.py): with probability carry_p the adopter first uses it on another page, later."""
    rows, by_place, by_agent = bg.rows, bg.by_place, bg.by_agent
    infected = {rows[seed_row].agent}
    use_rows = [seed_row]
    parent = {}
    frontier = [seed_row]
    while frontier and len(infected) < cap:
        u = frontier.pop(0)
        for v in by_place[rows[u].place]:
            if v <= u or rows[v].exposure_only or not bg.clock_after(u, v):
                continue
            if rows[v].time - rows[u].time > horizon:
                break
            a = rows[v].agent
            if a in infected or rng.random() >= q:
                continue
            adopt = v
            if rng.random() < carry_p:
                nxt = [w for w in by_agent[a] if w > v and rows[w].place != rows[v].place
                       and bg.clock_after(v, w) and rows[w].time - rows[v].time < horizon]
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


def follows(key_seed, place, agent, q=Q):
    """One agent's own coin for one page's instruction: a hash of (seed, page, agent), nothing else.

    No other agent, no order and no earlier use enters it, so who wrote first cannot change who follows.
    """
    digest = hashlib.sha256(("%d\x1f%s\x1f%s" % (key_seed, place, agent)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64 < q


def plant_page_instruction(bg, place, t0, key_seed=PAGE_INSTRUCTION_SEED, q=Q, horizon=HORIZON):
    """Page-wide shared cause: an instruction on ``place`` from ``t0``; each writer follows on its own coin.

    Writers are the agents with a save on the page in [t0, t0 + horizon]. A follower's use row is its first
    save on the page in that window. Returns (use rows ascending, followers sorted by label).
    """
    rows = bg.rows
    first = {}
    for v in bg.by_place[place]:  # time order
        r = rows[v]
        if r.exposure_only or r.time < t0:
            continue
        if r.time - t0 > horizon:
            break
        first.setdefault(r.agent, v)
    followers = sorted(a for a in first if follows(key_seed, place, a, q))
    return sorted(first[a] for a in followers), followers


# ---------------------------------------------------------------- the planted set


class Planted(NamedTuple):
    tokens: dict  # key -> PTok
    kind_of: dict  # key -> kind
    truth: dict  # key -> (seed agent, {adopter: parent}); held in memory only, never written
    info: dict  # counts only


def plant_all(rows, page_instruction=True):
    """Every planted kind on ``rows``, in E7's order and seeds; the page instruction has its own seed.

    The page instruction draws from its own generator, so adding it leaves every E7 planting unchanged.
    """
    bg = Background(rows)
    agent_rows = bg.agent_rows
    tokens, truth, kind_of = {}, {}, {}

    def add(kind, j, use_rows, tr=None, dedup=False):
        k = "planted:%s:%03d" % (kind, j)
        rr = tuple(sorted(set(use_rows))) if dedup else tuple(sorted(use_rows))
        tokens[k] = PTok(k, "planted_" + kind, rr)
        kind_of[k] = kind
        if tr is not None:
            truth[k] = tr

    fam_day = defaultdict(list)
    day_rows = defaultdict(list)
    for i in agent_rows:
        fam_day[(rows[i].family, int(rows[i].time // DAY))].append(i)
        day_rows[int(rows[i].time // DAY)].append(i)
    fam_cands = sorted([k for k, v in fam_day.items() if len({rows[i].agent for i in v}) >= 12], key=str)

    if agent_rows:
        # E6 (exp.py): rng SEED+6, contagion first, then the family decoy
        rng = random.Random(SEED + 6)
        made = tries = 0
        while made < PER_KIND and tries < 5000:
            tries += 1
            s = rng.choice(agent_rows)
            ur, par = plant_contagion(bg, rng, s)
            if len(par) >= 4:
                add("contagion", made, ur, (rows[s].agent, par))
                made += 1
        for j in range(min(PER_KIND, len(fam_cands))):
            firsts = {}
            for i in fam_day[fam_cands[(j * 7919) % len(fam_cands)]]:
                firsts.setdefault(rows[i].agent, i)
            add("decoy_fam", j, rng.sample(sorted(firsts.values()), 10))

        # exp2: rng SEED+20, carry, mixed, same, then two decoys
        rng = random.Random(SEED + 20)
        for mode, carry_p in (("carry", 1.0), ("mixed", 0.5), ("same", 0.0)):
            made = tries = 0
            while made < PER_KIND and tries < 20000:
                tries += 1
                s = rng.choice(agent_rows)
                ur, par = plant_exp2(bg, rng, s, carry_p)
                if len(par) >= 4:
                    add(mode, made, ur, (rows[s].agent, par), dedup=True)
                    made += 1
        for j in range(min(PER_KIND, len(fam_cands))):
            firsts = {}
            for i in fam_day[fam_cands[(j * 7919) % len(fam_cands)]]:
                firsts.setdefault(rows[i].agent, i)
            add("decoy_fam2", j, rng.sample(sorted(firsts.values()), 10), dedup=True)
        bdays = sorted(d for d, v in day_rows.items() if len({rows[i].agent for i in v}) >= 30)
        for j in range(PER_KIND if bdays else 0):
            firsts = {}
            for i in day_rows[bdays[j % len(bdays)]]:
                firsts.setdefault(rows[i].agent, i)
            add("decoy_bcast", j, rng.sample(sorted(firsts.values()), 10), dedup=True)

        # random rows, no structure
        rng = random.Random(20261031)
        for j in range(PER_KIND):
            add("random", j, rng.sample(agent_rows, min(10, len(agent_rows))), dedup=True)

        # self-repeat decoy: every row of 4 busy agent-days, 4 different agents, no planted contact
        agent_day = defaultdict(list)
        for i in agent_rows:
            agent_day[(rows[i].agent, int(rows[i].time // DAY))].append(i)
        busy = sorted(k for k, v in agent_day.items() if len(v) >= 5)
        rng = random.Random(20261032)
        if len({a for a, _d in busy}) >= 4:
            for j in range(PER_KIND):
                while True:
                    pick = rng.sample(busy, 4)
                    if len({a for a, _d in pick}) == 4:
                        break
                add("repeat", j, [i for key in pick for i in agent_day[key]], dedup=True)

        # pools for the power curve
        rng = random.Random(20261033)
        made = tries = 0
        while made < POOL and tries < 20000:
            tries += 1
            s = rng.choice(agent_rows)
            ur, par = plant_contagion(bg, rng, s)
            if len(par) >= 4:
                add("pool_contagion", made, ur, (rows[s].agent, par))
                made += 1
        rng = random.Random(20261034)
        made = tries = 0
        while made < POOL and tries < 20000:
            tries += 1
            s = rng.choice(agent_rows)
            ur, par = plant_exp2(bg, rng, s, 1.0)
            if len(par) >= 4:
                add("pool_carry", made, ur, (rows[s].agent, par), dedup=True)
                made += 1

    info = {}
    if page_instruction and agent_rows:
        # NEW, added after E7: own seed; 40 distinct real pages; each page tried once.
        rng = random.Random(PAGE_INSTRUCTION_SEED)
        tried = set()
        made = tries = 0
        while made < PER_KIND and tries < PAGE_INSTRUCTION_TRIES:
            tries += 1
            s = rng.choice(agent_rows)
            place = rows[s].place
            if place in tried:
                continue
            tried.add(place)
            ur, followers = plant_page_instruction(bg, place, rows[s].time)
            if len(followers) >= PAGE_INSTRUCTION_MIN_FOLLOWERS:
                add("page_instruction", made, ur, dedup=True)
                made += 1
        info["page_instruction"] = {"tokens": made, "pages_tried": len(tried), "draws": tries,
                                    "followers": sum(len(tokens["planted:page_instruction:%03d" % j].rows)
                                                     for j in range(made))}
    info["planted_tokens"] = dict(Counter(kind_of.values()))
    return Planted(tokens, kind_of, truth, info)


# ---------------------------------------------------------------- scoring


def per_kind(named, kinds_list):
    out = Counter()
    for t, kd in enumerate(kinds_list):
        out[kd] += named[t]
    return out


def summarize_kinds(obs, sims):
    res = {}
    for kd in sorted(obs):
        vals = [s_[kd] for s_ in sims]
        sm = N.summarize(vals, obs[kd])
        res[kd] = {key: sm.get(key) for key in SUMMARY_KEYS}
        res[kd]["excess_share"] = round((obs[kd] - sm["mean"]) / obs[kd], 4) if obs[kd] else None
    return res


def paired_z(diffs):
    """Mean of per-token differences over its standard error; 0 or None when every difference is equal."""
    n = len(diffs)
    if n == 0:
        return None
    m = sum(diffs) / n
    sd = float(stdev(diffs)) if n > 1 else 0.0
    if sd == 0:
        return 0.0 if m == 0 else None
    return round(m / (sd / math.sqrt(n)), 2)


def accuracy(ixp, cp, kind_of, truth):
    """Per kind: tokens, adopters, links, statuses; for kinds with truth, seed and parent accuracy."""
    acc = defaultdict(Counter)
    names = ixp.agent_names
    for t, det in enumerate(cp.detail):
        k = ixp.keys[t]
        R = acc[kind_of[k]]
        R["tokens"] += 1
        R["adopters"] += len(det)
        R["links"] += cp.named[t]
        for _a, st, _b in det:
            R["status_" + st] += 1
        if k in truth:
            seed_true, par = truth[k]
            srcs = {ixp.agent[b] for (a, s_, b) in det if s_ == "linked"}
            seed = next((names[a] for (a, s_, b) in det if a in srcs), None)
            R["seed_exact"] += seed == seed_true
            for (a, s_, b) in det:
                nm = names[a]
                if nm in par and par[nm] is not None:
                    R["true_adopters"] += 1
                    if s_ == "linked":
                        R["true_linked"] += 1
                        R["parent_exact"] += names[ixp.agent[b]] == par[nm]
    return {k: dict(v) for k, v in sorted(acc.items())}


def h1_counts(acc):
    """The E7 H1 reproduction counts (E6 and exp2 plantings)."""
    def get(kind, key):
        return acc.get(kind, {}).get(key, 0)
    return {"contagion_true_linked": get("contagion", "true_linked"),
            "contagion_true_adopters": get("contagion", "true_adopters"),
            "contagion_parent_exact": get("contagion", "parent_exact"),
            "decoy_fam_links": get("decoy_fam", "links"), "carry_links": get("carry", "links"),
            "same_links": get("same", "links"), "decoy_bcast_links": get("decoy_bcast", "links"),
            "decoy_fam2_links": get("decoy_fam2", "links")}


def time_reversal(rows, planted, kind_of, ixp, cp):
    """Forward against reversed links per kind, with the paired z over tokens."""
    n = len(rows)
    rev_rows = [r._replace(time=-r.time) for r in reversed(rows)]
    rev_planted = {k: t._replace(rows=tuple(sorted(n - 1 - i for i in t.rows))) for k, t in planted.items()}
    ixr = N.build_index(rev_rows, rev_planted)
    cr = N.count_links(ixr, ixr.agent)
    rk = {k: t for t, k in enumerate(ixr.keys)}
    rev = defaultdict(Counter)
    diffs = defaultdict(list)
    for t, k in enumerate(ixp.keys):
        kd = kind_of[k]
        f, r = cp.named[t], cr.named[rk[k]]
        rev[kd]["forward"] += f
        rev[kd]["reversed"] += r
        diffs[kd].append(f - r)
    return {k: dict(v, ratio=round(v["forward"] / v["reversed"], 3) if v["reversed"] else None,
                    z_paired=paired_z(diffs[k]))
            for k, v in sorted(rev.items())}


def calibration(nulls):
    """E7 PREREG C, unchanged: which nulls are calibrated on this background."""
    cal = {}
    for name, res in nulls.items():
        kd = res["kinds"]

        def zof(k):
            z = kd.get(k, {}).get("z")
            return z if z is not None else 0.0

        def ge(k):
            v = kd.get(k, {}).get("shuffles_ge_observed")
            return v if v is not None else 10 ** 9
        a = zof("contagion") >= 3 and ge("contagion") <= 1
        b = zof("carry") >= 3 and ge("carry") <= 1
        c = all(zof(k) < 2 for k in ("decoy_fam", "decoy_fam2", "decoy_bcast"))
        d = zof("random") < 2
        e = zof("repeat") < 2
        cal[name] = {"a_contagion": a, "b_carry": b, "c_common_decoys": c, "d_random": d, "e_repeat": e,
                     "calibrated": a and b and c and d and e}
    return cal


def run(rows, real, shuffles=100, seed=SEED, log=None, page_instruction=True):
    """Plant every kind on ``rows`` and score it under every null. Returns a counts-only dict.

    ``real`` is the log's own token set (the ``find_tokens`` result): the order null N3 runs on real and
    planted tokens together, under the same permutations, so its real-data numbers come out too (H0).
    """
    log = log or (lambda *a: None)
    started = time.perf_counter()
    P = plant_all(rows, page_instruction=page_instruction)
    planted, kind_of, truth = P.tokens, P.kind_of, P.truth
    out = {"shuffles": shuffles, "seed": seed, "rows": len(rows), "real_tokens": len(real),
           "planted_tokens": P.info["planted_tokens"]}
    if "page_instruction" in P.info:
        out["page_instruction_planting"] = P.info["page_instruction"]
    log("planted", out["planted_tokens"])

    ixp = N.build_index(rows, planted)
    pkind = [kind_of[k] for k in ixp.keys]
    cp = N.count_links(ixp, ixp.agent, detail=True)
    acc = accuracy(ixp, cp, kind_of, truth)
    out["accuracy"] = acc
    out["H1"] = h1_counts(acc)
    out["time_reversal"] = time_reversal(rows, planted, kind_of, ixp, cp)
    log("accuracy and reversal done")

    nulls = {}
    gfam = defaultdict(list)
    for i in ixp.agent_rows:
        gfam[(int(ixp.times[i] // DAY), ixp.family[i])].append(i)
    groups_dayfam = tuple(tuple(gfam[k]) for k in sorted(gfam, key=str))
    obs_default = per_kind(cp.named, pkind)
    for name, groups in (("N1_label_day", ixp.days), ("N2_label_day_x_family", groups_dayfam)):
        r_ = random.Random(seed)
        sims = []
        for _ in range(shuffles):
            lab = N.shuffle_labels(ixp, r_, groups)
            sims.append(per_kind(N.count_links(ixp, lab).named, pkind))
        nulls[name] = {"groups": len(groups), "kinds": summarize_kinds(obs_default, sims)}
        log(name, "done")

    for name, strict, lag in (("N4_order_strict", True, 0), ("N5_order_lag120", False, 120)):
        ix0 = K.with_min_lag(ixp, lag)
        obs = per_kind(N.count_links(ix0, ix0.agent, strict=strict).named, pkind)
        r_ = random.Random(seed)
        sims = []
        for _ in range(shuffles):
            nr, nt = K.reorder(rows, planted, r_, K.SCOPE_AGENT_DAY)
            ix = K.with_min_lag(N.build_index(nr, nt), lag)
            kl = [kind_of[k] for k in ix.keys]
            sims.append(per_kind(N.count_links(ix, ix.agent, strict=strict).named, kl))
        nulls[name] = {"kinds": summarize_kinds(obs, sims)}
        log(name, "done")

    alltok = dict(real)
    alltok.update(planted)
    ixa = N.build_index(rows, alltok)
    is_real = [k not in kind_of for k in ixa.keys]
    ca = N.count_links(ixa, ixa.agent)
    obs_real = sum(ca.named[t] for t in range(len(ixa.keys)) if is_real[t])
    pkeys = [k for k in ixa.keys if k in kind_of]
    pos = {k: t for t, k in enumerate(ixa.keys)}
    obs_tok = {k: ca.named[pos[k]] for k in pkeys}
    r_ = random.Random(seed)
    sim_real, sim_tok = [], []
    for s_i in range(shuffles):
        nr, nt = K.reorder(rows, alltok, r_, K.SCOPE_AGENT_DAY)
        ix = N.build_index(nr, nt)
        c_ = N.count_links(ix, ix.agent)
        tot = 0
        tok = {}
        for t, k in enumerate(ix.keys):
            if k in kind_of:
                tok[k] = c_.named[t]
            else:
                tot += c_.named[t]
        sim_real.append(tot)
        sim_tok.append(tok)
        if s_i % 10 == 9:
            log("N3 shuffle", s_i + 1)
    sm_real = N.summarize(sim_real, obs_real)
    out["H0"] = {"observed": obs_real, "mean": sm_real["mean"], "sd": sm_real["sd"]}
    obs3 = Counter()
    for k in pkeys:
        obs3[kind_of[k]] += obs_tok[k]
    sims3 = []
    for tok in sim_tok:
        c3 = Counter()
        for k, v in tok.items():
            c3[kind_of[k]] += v
        sims3.append(c3)
    nulls["N3_order_agent_day"] = {"kinds": summarize_kinds(obs3, sims3),
                                   "real": {key: sm_real.get(key) for key in SUMMARY_KEYS}}
    out["nulls"] = nulls
    out["calibration"] = calibration(nulls)

    power = {}
    for mech, first, pool in (("contagion", "contagion", "pool_contagion"), ("carry", "carry", "pool_carry")):
        order = sorted(k for k in pkeys if kind_of[k] == first) + sorted(k for k in pkeys if kind_of[k] == pool)
        rowsout = []
        for kk in POWER_K:
            sel = order[:kk]
            o = obs_real + sum(obs_tok[k] for k in sel)
            vals = [sim_real[i] + sum(sim_tok[i][k] for k in sel) for i in range(shuffles)]
            sm = N.summarize(vals, o)
            rowsout.append({"k": kk, "planted_links": sum(obs_tok[k] for k in sel), "observed": o,
                            "mean": sm["mean"], "sd": sm["sd"], "z": sm["z"], "ratio": sm["ratio"],
                            "shuffles_ge_observed": sm["shuffles_ge_observed"]})
        thr = next((r["k"] for r in rowsout if r["k"] > 0 and r["z"] is not None and r["z"] >= 3
                    and r["shuffles_ge_observed"] <= 1), None)
        power[mech] = {"curve": rowsout, "threshold_k": thr}
    out["power_N3"] = power
    out["matrix"] = matrix(out)
    if page_instruction:
        out["page_instruction_lines"] = page_instruction_lines(out)
    out["seconds"] = round(time.perf_counter() - started, 1)
    return out


# ---------------------------------------------------------------- the matrix and the pre-registered lines


def matrix(result):
    """Matrix rows in display order: per null z and shuffles at or above; the reversed clock."""
    rows_out = []
    for kind in MATRIX_ORDER:
        if kind not in result["accuracy"]:
            continue
        cells = {}
        for name in NULL_NAMES:
            c = result["nulls"].get(name, {}).get("kinds", {}).get(kind)
            if c is not None:
                cells[name] = {"z": c["z"], "shuffles_ge_observed": c["shuffles_ge_observed"],
                               "observed": c["observed"], "mean": c["mean"]}
        tr = result["time_reversal"].get(kind, {})
        rows_out.append({"kind": kind, "label": KIND_LABELS[kind], "nulls": cells,
                         "reversed_clock": {"ratio": tr.get("ratio"), "z_paired": tr.get("z_paired")}})
    return rows_out


def _no_reading(cell):
    """|z| < 2 on a shuffled null; z is None only when every shuffle gave the same count."""
    z = cell.get("z")
    if z is None:
        return cell.get("sd") == 0 and cell.get("mean") == cell.get("observed")
    return abs(z) < 2


def page_instruction_lines(result):
    """The two pre-registered lines for the page instruction, each PASS or FAIL (never retuned).

    L1: both label nulls flag it as spread (z >= 3 under N1 and under N2).
    L2: the order null N3 and the reversed clock read nothing (|z| < 2 for each).
    """
    kinds = {name: result["nulls"][name]["kinds"].get("page_instruction", {}) for name in NULL_NAMES}
    z1, z2 = kinds["N1_label_day"].get("z"), kinds["N2_label_day_x_family"].get("z")
    l1 = z1 is not None and z1 >= 3 and z2 is not None and z2 >= 3
    n3 = kinds["N3_order_agent_day"]
    zr = result["time_reversal"].get("page_instruction", {}).get("z_paired")
    n3_ok = bool(n3) and _no_reading(n3)
    rev_ok = zr is not None and abs(zr) < 2
    return {
        "L1_label_nulls_flag_it": {"N1_z": z1, "N2_z": z2, "rule": "z >= 3 under N1 and under N2",
                                   "result": "PASS" if l1 else "FAIL"},
        "L2_order_null_and_reversed_clock_read_nothing": {
            "N3_z": n3.get("z"), "reversed_clock_z_paired": zr, "rule": "|z| < 2 under N3 and on the reversed clock",
            "N3_part": "PASS" if n3_ok else "FAIL", "reversed_clock_part": "PASS" if rev_ok else "FAIL",
            "result": "PASS" if (n3_ok and rev_ok) else "FAIL"},
    }


def withhold(result, kind="page_instruction", reason="harness gate failed"):
    """The result with every cell of ``kind`` removed (used when the harness does not reproduce)."""
    out = dict(result)
    note = "WITHHELD: %s" % reason
    for sec in ("accuracy", "time_reversal"):
        if sec in out:
            out[sec] = {k: v for k, v in out[sec].items() if k != kind}
    if "nulls" in out:
        out["nulls"] = {name: dict(res, kinds={k: v for k, v in res["kinds"].items() if k != kind})
                        for name, res in out["nulls"].items()}
    if "matrix" in out:
        out["matrix"] = [r for r in out["matrix"] if r["kind"] != kind]
    if kind == "page_instruction" and "page_instruction_lines" in out:
        out["page_instruction_lines"] = note
    out[kind + "_cells"] = note
    return out


def compare_expected(result, expected, sections=("nulls", "time_reversal", "accuracy", "power_N3", "calibration")):
    """Paths where ``result`` differs from an earlier run's JSON, for every leaf that run holds."""
    bad = []

    def walk(exp, got, path):
        if isinstance(exp, dict):
            if not isinstance(got, dict):
                bad.append(path)
                return
            for k, v in exp.items():
                if k not in got:
                    bad.append(path + "/" + str(k) + " (missing)")
                else:
                    walk(v, got[k], path + "/" + str(k))
        elif isinstance(exp, list):
            if not isinstance(got, (list, tuple)) or len(got) != len(exp):
                bad.append(path)
                return
            for i, (a, b) in enumerate(zip(exp, got)):
                walk(a, b, "%s[%d]" % (path, i))
        elif exp != got:
            bad.append(path)

    for sec in sections:
        if sec in expected:
            walk(expected[sec], result.get(sec), sec)
    return bad


def markdown_table(result):
    """The matrix as a Markdown table: z (shuffles at or above observed); reversed clock ratio and z."""
    def fmt(c):
        if c is None:
            return "-"
        z = "n/a" if c["z"] is None else "%.2f" % c["z"]
        return "%s (%s)" % (z, c["shuffles_ge_observed"])
    head = ("| Planted kind (40 tokens each) | N1 label day | N2 label day x family | N3 order agent-day | "
            "N4 order, strict | N5 order, 120 s lag | Reversed clock fwd/rev, z |")
    lines = [head, "|" + "---|" * 7]
    for r in result["matrix"]:
        rc = r["reversed_clock"]
        zr = "unbounded" if rc["z_paired"] is None else "%.2f" % rc["z_paired"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s, %s |" % (
            r["label"], *(fmt(r["nulls"].get(n)) for n in NULL_NAMES),
            "-" if rc["ratio"] is None else "%.2f" % rc["ratio"], zr))
    return "\n".join(lines)
