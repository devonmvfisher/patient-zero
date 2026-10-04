#!/usr/bin/env python3
"""Facts sheet: fill slots F1 to F14 and three candidate lead findings from a run.

Python standard library only. Reads a run's output folder (keep it outside the
repo):

* ``facts.json`` (required, from ``cli.py``), with the hand check
  (``tools/peek.py --record``) and stretch S1 (``patient_zero.offwiki``) when
  they were merged into it for the same ``core_sha256``;
* ``top_trees.json`` (required for the lead tree), checked against the
  sha256 that facts.json holds for it;
* ``checks.json`` (optional, ``python -m patient_zero.checks``): the order
  null, strict exposure and minimum-lag counts. It is used only when its row,
  token and link counts match facts.json.

It also re-runs the planted fixtures of ``tests/test_planted.py`` (made-up
names, a few rows) for slot F11.

Writes ``FACTS-<label>.md`` (slot table, candidate findings A, B and C, each
with its numbers, its observed-to-shuffle ratio and one limit) and, with
``--html``, one HTML page: a question at the top, one screen per finding, the
lead figure (``docs/lead.svg``) inline, the slot table and, with
``--outline``, the outline section of a Markdown plan. Neither output may
name a file the sheet reads (facts.json above all: it holds the merged hand
check and stretch S1); the run stops before any work if one does.

Everything written is a count, a ratio, a ``token_id`` or a data string
passed through ``patient_zero.safe.short`` (escaped, at most 80 characters).
The sheet gives facts and short labels. It writes no sentences for anyone's
write-up.

Rules for the numbers, built in:

* F4 links always come with the strict and 120 s counts when checks.json has them.
* F5 shows the planned label shuffle next to its adopter counts, and the order
  null labelled as added after the planned null's result.
* F7 gives medians with their rule; the fastest values are withheld, because
  they look like burst editing until recomputed under a stricter rule.
* F13 is labelled unverified while S1 has no timing baseline.
* The lead tree is labelled as picked from all tokens.

Usage (from the repo folder)::

    python tools/facts_sheet.py --out ../out --label v1
    python tools/facts_sheet.py --out ../out --label v1 --html PAGE.html \\
        --outline PLAN.md --ask "Lead finding: A, B or C?" --note "..." \\
        --slot-note F8="..."

Exit codes: 0 done, 2 bad input or a stale file.
"""

import argparse
import collections
import datetime
import hashlib
import html
import json
import os
import re
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from patient_zero.safe import guard_out, same_file, short  # noqa: E402

MEANINGS = (
    ("F1", "Lead finding: token type, effective seed (agent name), adopters, minutes to 3rd and 10th adopter"),
    ("F2", "Page family where the lead tree lived"),
    ("F3", "Candidate tokens; tokens used by 3+ agents, per class"),
    ("F4", "Adoption links observed"),
    ("F5", "Links expected under the shuffle (mean and spread); ratio observed / expected"),
    ("F6", "Spread trees with 5+ agents; with 10+"),
    ("F7", "Fastest tree: minutes to 10th adopter"),
    ("F8", "Parallel starts found; share of trees where first user is not the effective seed"),
    ("F9", "Links dropped by the clock rule"),
    ("F10", "Hand check result, X of 10 plausible"),
    ("F11", "Planted test: cascades found (of 3); false links from inherited text"),
    ("F12", "Risky-spread trees (by type only)"),
    ("F13", "Stretch result (S1 off-wiki reuse count and median lag), or \"not done\""),
    ("F14", "Runtime, lines of code, number of tests"),
)
SLOT_IDS = tuple(s for s, _ in MEANINGS)
DEFAULT_ASK = "Lead finding: A, B or C?"
STATUS_WORDS = {"first": "first use", "linked": "linked use", "parallel": "parallel start",
                "unnamed": "use after an unnamed row", "clock_dropped": "use dropped by the clock rule"}


class SheetError(Exception):
    """Bad or stale input; the message says what to rerun."""


# ---------------------------------------------------------------- small helpers


def g(obj, *keys, default=None):
    """obj[k1][k2]... or ``default`` when any step is missing."""
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
        elif isinstance(obj, (list, tuple)) and isinstance(k, int) and -len(obj) <= k < len(obj):
            obj = obj[k]
        else:
            return default
    return obj


def num(x, digits=1):
    """1234 -> '1,234'; 0.4313 (digits=3) -> '0.431'; None -> 'n/a'."""
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return "{:,}".format(x)
    if float(x).is_integer() and digits <= 1 and abs(x) >= 1000:
        return "{:,}".format(int(x))
    return "{:,.{d}f}".format(x, d=digits)


def pct(part, whole, digits=1):
    if not whole or part is None:
        return "n/a"
    return "%.*f%%" % (digits, 100.0 * part / whole)


def tiny(x):
    """A small probability in plain decimals, 2 significant digits (6.5e-06 -> '0.0000065')."""
    if not x:
        return "0"
    return ("%.12f" % float("%.2g" % x)).rstrip("0").rstrip(".")


def status_word(status):
    return STATUS_WORDS.get(status, short(status, 20))


def ratio(obs, mean):
    if obs is None or not mean:
        return None
    return obs / mean


def verdict(stat):
    """'above chance' / 'below chance' / 'within chance' against a shuffle's 5th to 95th percentile."""
    if not stat or stat.get("p05") is None or stat.get("p95") is None:
        return None
    if stat["observed"] > stat["p95"]:
        return "above chance"
    if stat["observed"] < stat["p05"]:
        return "below chance"
    return "within chance"


def token_text(key):
    """'phrase:to answered correct' -> 'to answered correct' (escaped, at most 60 characters)."""
    if key is None:
        return "n/a"
    value = key.split(":", 1)[1] if ":" in key else key
    if value.startswith(("prefix:", "mailbox:", "control:")):
        value = value.split(":", 1)[1]
    return short(value, 60)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_atomic(path, text):
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


RUN_FILES = ("facts.json", "top_trees.json", "checks.json")


def input_paths(out_dir, svg_path=None, outline_path=None):
    """Every file the sheet reads: the run's three files, the figure and the outline."""
    return [os.path.join(out_dir, name) for name in RUN_FILES] + [svg_path, outline_path]


def guard_outputs(md_path, html_path, inputs):
    """Refuse --md or --html when it names an input (facts.json above all), or both name one file."""
    guard_out(md_path, inputs, "--md")
    if html_path:
        guard_out(html_path, inputs, "--html")
        if same_file(md_path, html_path):
            raise ValueError("--md and --html both name %s; give them different paths" % html_path)


# ---------------------------------------------------------------- inputs


def checks_problem(checks, facts):
    """Why checks.json does not belong to this facts.json, or None when it does."""
    pairs = (
        ("rows", checks.get("rows"), g(facts, "load", "rows")),
        ("tokens", checks.get("tokens"), g(facts, "tokens", "kept_total")),
        ("named links", g(checks, "observed", "default_min_lag_0s", "links_named"),
         g(facts, "slots", "F4", "links_named")),
    )
    for name, have, want in pairs:
        if have is None or want is None or have != want:
            return "its %s (%s) differ from facts.json (%s)" % (name, have, want)
    return None


def load_inputs(out_dir):
    """(facts, top, checks, warnings). Raises SheetError on a missing or stale required file."""
    facts_path = os.path.join(out_dir, "facts.json")
    if not os.path.isfile(facts_path):
        raise SheetError("no facts.json in %s: run cli.py first" % out_dir)
    with open(facts_path, encoding="utf-8") as fh:
        facts = json.load(fh)
    warnings = []

    top = None
    top_path = os.path.join(out_dir, "top_trees.json")
    want = g(facts, "outputs", "top_trees.json", "sha256")
    if os.path.isfile(top_path):
        if want and sha256_file(top_path) != want:
            raise SheetError("top_trees.json does not match the sha256 in facts.json (stale): rerun cli.py")
        with open(top_path, encoding="utf-8") as fh:
            top = json.load(fh)
    else:
        warnings.append("no top_trees.json: F1, F2 and finding B are empty")

    checks = None
    checks_path = os.path.join(out_dir, "checks.json")
    if os.path.isfile(checks_path):
        with open(checks_path, encoding="utf-8") as fh:
            candidate = json.load(fh)
        problem = checks_problem(candidate, facts)
        if problem:
            warnings.append("checks.json not used: " + problem)
        else:
            checks = candidate
    else:
        warnings.append("no checks.json: findings use the planned label shuffle only")

    core = facts.get("core_sha256")
    for key, stamp in (("hand_check", "facts_core_sha256_at_check"), ("stretch_s1", "facts_core_sha256_at_write")):
        part = facts.get(key)
        if part and part.get(stamp) not in (None, core):
            warnings.append("%s not used: written for another core_sha256" % key)
            facts = dict(facts)
            facts.pop(key)
    return facts, top, checks, warnings


def lead_record(top):
    if not top:
        return None
    lead = top.get("lead")
    return next((t for t in top.get("trees", ()) if t.get("token_id") == lead), None)


def lead_check(checks, section, lead_id):
    """checks[section]['lead'] when it is about the same lead tree, else None."""
    part = g(checks, section, "lead")
    if part and part.get("token_id") == lead_id:
        return part
    return None


def planted_check():
    """Re-run the planted fixtures of tests/test_planted.py and count what comes back (slot F11)."""
    try:
        from patient_zero import cascade as C
        import tests.test_planted as TP
    except Exception as exc:  # the tests folder is missing or broken
        return {"error": "planted fixtures not importable: %s" % short(exc, 60)}
    out = {"cascades_planted": len(TP.PLANTED)}
    try:
        TP.ChatPlanted.setUpClass()
        try:
            trees = TP.ChatPlanted.trees
            found = 0
            for key, want in TP.PLANTED.items():
                tree = trees.get(key)
                if (tree is not None and C.parents_of(tree) == want["parents"]
                        and tree.first_user == want["first_user"]
                        and tree.effective_seed == want["effective_seed"]):
                    found += 1
            out["cascades_found"] = found
        finally:
            TP.ChatPlanted.tearDownClass()
        TP.WikiInheritedText.setUpClass()
        try:
            tree = TP.WikiInheritedText.trees.get(TP.INHERITED)
            out["inherited_token_live"] = tree is not None
            out["inherited_false_links"] = len(tree.links) if tree is not None else None
            out["wiki_fixture_links"] = TP.WikiInheritedText.stats.get("links_named")
        finally:
            TP.WikiInheritedText.tearDownClass()
        case = TP.ParallelStart("test_parallel_starts_flagged")
        case.setUp()
        out["parallel_flagged"] = len(case.tree.parallel_starts)
        out["parallel_planted"] = 2
    except Exception as exc:
        out["error"] = "planted fixtures failed: %s" % short(exc, 60)
    return out


def code_now():
    try:
        from patient_zero.pipeline import code_info
        return code_info(REPO_DIR)
    except Exception:
        return {}


# ---------------------------------------------------------------- slots


def slot_values(facts, top, checks, planted, now, slot_notes=None):
    """{slot: (value text, source text)} for F1 to F14."""
    s = facts.get("slots") or {}
    rec = lead_record(top)
    v = {}

    # F1, F2: the lead tree drawn in docs/lead.svg
    if rec:
        nodes = rec.get("nodes") or []
        seed = g(nodes, rec.get("seed")) if rec.get("seed") is not None else None
        root = g(nodes, rec.get("branch_root")) if rec.get("branch_root") is not None else None
        st = rec.get("status") or {}
        parts = ['%s token "%s" (token_id %s)' % (rec.get("cls"), token_text(rec.get("token")), rec.get("token_id"))]
        if seed:
            parts.append("effective seed %s%s" % (short(seed.get("agent"), 60),
                                                   " (also the first user)" if rec.get("seed") == rec.get("first")
                                                   else ""))
        parts.append("%s agents used it (%s linked, %s parallel starts)"
                     % (num(rec.get("agents")), num(st.get("linked")), num(st.get("parallel"))))
        tenth = rec.get("seed_min_to_10th")
        parts.append("seed branch %s agents: 3rd adopter after %s min, 10th %s"
                     % (num(rec.get("seed_branch")), num(rec.get("seed_min_to_3rd")),
                        "after %s min" % num(tenth) if tenth is not None else "not reached"))
        if root:
            parts.append("largest branch %s agents, root %s (a %s, %s h after the first use): "
                         "3rd after %s min, 10th after %s min"
                         % (num(rec.get("largest_branch")), short(root.get("agent"), 60),
                            status_word(root.get("status")), num((root.get("t_min") or 0) / 60.0),
                            num(rec.get("branch_min_to_3rd")), num(rec.get("branch_min_to_10th"))))
        parts.append("picked by: %s" % (top.get("lead_rule") or "n/a"))
        v["F1"] = ("; ".join(parts), "top_trees.json (lead)")
        fams = collections.Counter(short(n.get("family"), 60) for n in nodes)
        first_fam = short(g(nodes, rec.get("first"), "family"), 60)
        fam_text = ", ".join("%s (%s of %s first uses)" % (f, num(c), num(len(nodes)))
                             for f, c in fams.most_common(3))
        v["F2"] = ("%s; the first use was on %s" % (fam_text, first_fam), "top_trees.json (lead nodes)")
    else:
        v["F1"] = ("n/a (no lead tree)", "top_trees.json")
        v["F2"] = ("n/a", "top_trees.json")

    f3 = s.get("F3") or {}
    by = f3.get("tokens_3plus_agents_by_class") or {}
    v["F3"] = ("%s candidate tokens; %s used by 3+ agents: %s"
               % (num(f3.get("candidate_tokens")), num(f3.get("tokens_3plus_agents")),
                  ", ".join("%s %s" % (k, num(c)) for k, c in sorted(by.items(), key=lambda kv: -kv[1]))),
               "facts.json slots.F3")

    f4 = s.get("F4") or {}
    text = ("%s named links (candidates), plus %s from exposure-only rows"
            % (num(f4.get("links_named")), num(f4.get("links_from_exposure_only_rows"))))
    strict = g(checks, "observed", "strict_min_lag_0s", "links_named")
    lag120 = g(checks, "observed", "default_min_lag_120s", "links_named")
    src = "facts.json slots.F4"
    if strict is not None and lag120 is not None:
        text += ("; stricter rules: %s when the adopting save does not count as a visit, %s with a 120 s "
                 "minimum lag" % (num(strict), num(lag120)))
        src += "; checks.json observed"
    v["F4"] = (text, src)

    f5 = s.get("F5") or {}
    text = ("Planned label shuffle (%s scope, %s shuffles, seed %s): mean %s, sd %s, 5th to 95th percentile "
            "%s to %s; ratio %s; %s of %s shuffles at or above. It is biased toward more links: it makes %s "
            "adopters against %s observed (linked share %s observed, %s under it)"
            % (f5.get("scope"), num(f5.get("shuffles")), f5.get("seed"), num(f5.get("links_expected_mean")),
               num(f5.get("links_expected_sd")), num(f5.get("links_expected_p05"), 0),
               num(f5.get("links_expected_p95"), 0), num(f5.get("ratio_observed_to_expected"), 3),
               num(f5.get("shuffles_ge_observed")), num(f5.get("shuffles")),
               num(f5.get("adopters_expected_mean"), 0), num(f5.get("adopters_observed")),
               num(f5.get("linked_share_observed"), 3), num(f5.get("linked_share_expected_mean"), 3)))
    src = "facts.json slots.F5"
    on = g(checks, "order_null", "links_named")
    if on:
        text += (". Order null (added after the planned null's result; each agent keeps its own saves and its "
                 "own times in each UTC day; %s shuffles): mean %s, sd %s; ratio %s; z %s; %s of %s at or above"
                 % (num(g(checks, "order_null", "shuffles")), num(on.get("mean")), num(on.get("sd")),
                    num(on.get("ratio"), 3), num(on.get("z"), 2), num(on.get("shuffles_ge_observed")),
                    num(g(checks, "order_null", "shuffles"))))
        src += "; checks.json order_null"
    v["F5"] = (text, src)

    f6 = s.get("F6") or {}
    v["F6"] = ("Tokens with 5+ agents %s, with 10+ %s (all adopters, parallel starts included). Connected "
               "branches of named links with 5+ agents %s, with 10+ %s; largest branch %s agents"
               % (num(f6.get("tokens_5plus_agents")), num(f6.get("tokens_10plus_agents")),
                  num(f6.get("branches_5plus_agents")), num(f6.get("branches_10plus_agents")),
                  num(g(facts, "trees", "largest_branch_max"))), "facts.json slots.F6, trees")

    f7 = s.get("F7") or {}
    v["F7"] = ("Median minutes from the effective seed to its 10th linked agent: %s (%s trees reach it); to "
               "its 3rd: %s (%s trees). Default rules. Fastest values withheld: recompute them under the "
               "strict or 120 s rule before quoting"
               % (num(f7.get("median_minutes_to_10th")), num(f7.get("trees_reaching_10_in_seed_branch")),
                  num(f7.get("median_minutes_to_3rd")), num(f7.get("trees_reaching_3_in_seed_branch"))),
               "facts.json slots.F7")

    f8 = s.get("F8") or {}
    v["F8"] = ("%s parallel starts; the first user is not the effective seed in %s of %s trees (%s)"
               % (num(f8.get("parallel_starts")), num(f8.get("seed_is_not_first_user")),
                  num(f8.get("trees_with_effective_seed")),
                  pct(f8.get("seed_is_not_first_user"), f8.get("trees_with_effective_seed"))),
               "facts.json slots.F8")

    f9 = s.get("F9") or {}
    v["F9"] = ("%s links dropped, %s links shifted to an older use; %s token dropped by the clock rule"
               % (num(f9.get("links_dropped_by_clock")), num(f9.get("links_shifted_by_clock")),
                  num(f9.get("tokens_dropped_by_clock_rule"))), "facts.json slots.F9")

    hc = facts.get("hand_check")
    if hc:
        v["F10"] = ("%s of %s plausible (%s coincidence, %s unclear); %s links drawn with seed %s from %s "
                    "named links" % (num(hc.get("plausible")), num(hc.get("sample")), num(hc.get("coincidence")),
                                     num(hc.get("unclear")), num(hc.get("sample")), hc.get("seed"),
                                     num(hc.get("population"))), "facts.json hand_check")
    else:
        v["F10"] = ("not done", "facts.json (no hand_check)")

    if planted.get("error"):
        v["F11"] = (planted["error"], "tests/test_planted.py fixtures")
    else:
        v["F11"] = ("%s of %s planted cascades found; inherited wiki text makes %s false links (token still "
                    "live: %s); %s of %s planted parallel starts flagged"
                    % (num(planted.get("cascades_found")), num(planted.get("cascades_planted")),
                       num(planted.get("inherited_false_links")), num(planted.get("inherited_token_live")),
                       num(planted.get("parallel_flagged")), num(planted.get("parallel_planted"))),
                    "tests/test_planted.py fixtures, re-run now")

    risky = facts.get("risky") or {}
    with_type = risky.get("trees_with_type") or {}
    narrow = risky.get("narrow_types") or []
    narrow_txt = ", ".join("%s %s" % (t, num(with_type.get(t))) for t in narrow)
    broad_txt = ", ".join("%s %s" % (t, num(c)) for t, c in with_type.items() if t not in narrow)
    nv = g(risky, "named_links_vs_shuffle", "narrow") or {}
    v["F12"] = ("Trees by narrow type: %s (%s trees with a narrow type, %s with named links). Broad types: %s. "
                "Named links in narrow-type trees %s against a planned-shuffle mean of %s (ratio %s; same "
                "bias as F5)"
                % (narrow_txt, num(risky.get("trees_with_narrow_type")),
                   num(risky.get("trees_with_narrow_type_and_named_links")), broad_txt,
                   num(nv.get("observed")), num(nv.get("mean")), num(nv.get("ratio"), 3)),
                "facts.json risky")

    s1 = facts.get("stretch_s1")
    if s1 and s1.get("status") == "done":
        h = s1.get("headline") or {}
        lag = h.get("lag_wiki_first_to_off_wiki") or {}
        npz = h.get("non_phrase") or {}
        rec_span = g(s1, "published_dates_vs_wiki_span", "records") or {}
        v["F13"] = ("UNVERIFIED (no timing baseline). S1: %s of %s kept tokens seen off the wiki, %s with a "
                    "published off-wiki date: %s on the wiki first then off it, %s off it first; median lag %s h "
                    "(%s days, quartiles %s to %s days). Without phrases: %s wiki first, median lag %s days. %s of "
                    "%s dated off-wiki records fall inside the wiki's own time span"
                    % (num(h.get("seen_off_wiki")), num(g(facts, "tokens", "kept_total")),
                       num(h.get("with_published_date")), num(h.get("wiki_first_then_off")),
                       num(h.get("off_first")), num(lag.get("median_hours"), 2), num(lag.get("median_days"), 2),
                       num(lag.get("p25_days"), 2), num(lag.get("p75_days"), 2),
                       num(npz.get("wiki_first_then_off")), num(g(npz, "lag_wiki_first_to_off_wiki", "median_days"), 2),
                       num(rec_span.get("during_wiki_span")), num(rec_span.get("published_events"))),
                    "facts.json stretch_s1")
    else:
        v["F13"] = ("not done", "facts.json (no stretch_s1)")

    f14 = s.get("F14") or {}
    text = ("Run %s s (target %s s) at commit %s%s: %s code lines, %s test functions"
            % (num(f14.get("runtime_seconds")), num(g(facts, "runtime", "target_seconds")),
               g(facts, "code", "commit"), " (clean)" if g(facts, "code", "dirty") is False else "",
               num(f14.get("code_lines")), num(f14.get("test_functions"))))
    if now.get("commit"):
        text += ("; at %s now%s: %s code lines, %s test functions"
                 % (now["commit"], " (uncommitted changes)" if now.get("dirty") else "",
                    num(now.get("code_lines")), num(now.get("test_functions"))))
    v["F14"] = (text, "facts.json slots.F14; git checkout now")

    for slot, note in (slot_notes or {}).items():
        value, src = v[slot]
        v[slot] = ("%s. Note: %s" % (value, note), src + "; caller note")
    return v


# ---------------------------------------------------------------- findings


def finding_a(facts, checks, planted):
    s = facts.get("slots") or {}
    f5 = s.get("F5") or {}
    links = g(s, "F4", "links_named")
    on = g(checks, "order_null", "links_named")
    nums = ["Named links (candidates): %s" % num(links)]
    ratios = []
    if on:
        word = verdict(on)
        n = g(checks, "order_null", "shuffles")
        nums.append("Order null (added after the planned null's result; each agent keeps its own saves and its own "
                    "times in each UTC day): mean %s, sd %s, 5th to 95th percentile %s to %s; %s of %s shuffles at "
                    "or above the observed count; z %s"
                    % (num(on["mean"]), num(on["sd"]), num(on["p05"], 0), num(on["p95"], 0),
                       num(on["shuffles_ge_observed"]), num(n), num(on["z"], 2)))
        classes = g(checks, "order_null", "by_class") or {}
        nums.append("By class against the order null: " + ", ".join(
            "%s %s" % (k, num(c["ratio"], 3)) for k, c in sorted(classes.items()) if c.get("ratio") is not None))
        nums.append("Single trees: %s above every order shuffle, %s below every one; chance allows up to %s either way"
                    % (num(g(checks, "order_null", "trees_above_every_shuffle")),
                       num(g(checks, "order_null", "trees_below_every_shuffle")),
                       num(int(round(g(checks, "order_null", "chance_bound_each_way", default=0))))))
        ratios.append(("order null", on["ratio"]))
        limit = ("The order null was built after the planned null's result was seen (a forking path), and the "
                 "planned label shuffle is biased, so neither is a pre-registered test.")
    else:
        word = verdict({"observed": f5.get("links_observed"), "p05": f5.get("links_expected_p05"),
                        "p95": f5.get("links_expected_p95")})
        limit = ("Only the planned label shuffle ran, and it is biased toward more links (it makes extra "
                 "adopters), so 'below' it does not mean agents avoid each other's tricks.")
    if f5:
        nums.append("Planned label shuffle (biased: %s adopters against %s observed): mean %s, ratio %s"
                    % (num(f5.get("adopters_expected_mean"), 0), num(f5.get("adopters_observed")),
                       num(f5.get("links_expected_mean")), num(f5.get("ratio_observed_to_expected"), 3)))
        ratios.append(("planned label shuffle, biased", f5.get("ratio_observed_to_expected")))
    all_found = (not planted.get("error") and planted.get("cascades_found") == planted.get("cascades_planted"))
    if not planted.get("error"):
        nums.append("Planted test: %s of %s cascades found; inherited wiki text makes %s false links"
                    % (num(planted.get("cascades_found")), num(planted.get("cascades_planted")),
                       num(planted.get("inherited_false_links"))))
    hc = facts.get("hand_check")
    if hc:
        nums.append("Hand check: %s of %s links plausible (%s coincidence, %s unclear)"
                    % (num(hc.get("plausible")), num(hc.get("sample")), num(hc.get("coincidence")),
                       num(hc.get("unclear"))))
    title = "Self-check: %sthe wiki's adoption links are %s" % (
        "the tool finds planted spread; " if all_found else "", word or "untested")
    return {"key": "A", "title": title, "ratios": ratios, "primary": ratios[0] if ratios else (None, None),
            "numbers": nums, "limit": limit}


def finding_b(facts, top, checks):
    rec = lead_record(top)
    if not rec:
        return {"key": "B", "title": "Example tree: none (no lead tree)", "ratios": [], "primary": (None, None),
                "numbers": [], "limit": "No tree has a named link."}
    lead_id = rec.get("token_id")
    obs = rec.get("links_named")
    shuffles = top.get("shuffles")
    nodes = rec.get("nodes") or []
    on = lead_check(checks, "order_null", lead_id)
    strict = lead_check(checks, "order_null_strict", lead_id)
    lag = lead_check(checks, "order_null_min_lag_120s", lead_id)
    ratios = []
    if on:
        ratios.append(("order null", ratio(obs, on.get("mean"))))
    ratios.append(("planned label shuffle, biased", ratio(obs, rec.get("shuffle_links_mean"))))
    nums = ['%s token "%s" (token_id %s): %s agents, %s uses, %s places, span %s h'
            % (rec.get("cls"), token_text(rec.get("token")), lead_id, num(rec.get("agents")), num(rec.get("uses")),
               num(rec.get("places")), num(rec.get("span_hours")))]
    text = "Named links %s:" % num(obs)
    if on:
        text += (" order null mean %s (%s of %s shuffles at or above);"
                 % (num(on.get("mean")), num(on.get("shuffles_ge_observed")), num(g(checks, "order_null", "shuffles"))))
    text += (" planned shuffle mean %s (%s of %s at or above)"
             % (num(rec.get("shuffle_links_mean")), num(rec.get("shuffle_ge_observed")), num(shuffles)))
    nums.append(text)
    if strict and lag:
        nums.append("Stricter rules: adopting save not a visit %s against %s (%s of %s at or above); 120 s minimum "
                    "lag %s against %s (%s of %s)"
                    % (num(strict.get("observed")), num(strict.get("mean")), num(strict.get("shuffles_ge_observed")),
                       num(g(checks, "order_null_strict", "shuffles")), num(lag.get("observed")), num(lag.get("mean")),
                       num(lag.get("shuffles_ge_observed")), num(g(checks, "order_null_min_lag_120s", "shuffles"))))
    seed = g(nodes, rec.get("seed")) if rec.get("seed") is not None else None
    if seed:
        tenth = rec.get("seed_min_to_10th")
        nums.append("Effective seed %s%s; its branch %s agents: 3rd adopter after %s min, 10th %s"
                    % (short(seed.get("agent"), 60), " (also the first user)" if rec.get("seed") == rec.get("first")
                       else "", num(rec.get("seed_branch")), num(rec.get("seed_min_to_3rd")),
                       "after %s min" % num(tenth) if tenth is not None else "not reached"))
    root = g(nodes, rec.get("branch_root")) if rec.get("branch_root") is not None else None
    if root:
        nums.append("Largest branch %s agents, root %s: a %s %s h after the first use; 3rd after %s min, "
                    "10th after %s min; depth %s"
                    % (num(rec.get("largest_branch")), short(root.get("agent"), 60), status_word(root.get("status")),
                       num((root.get("t_min") or 0) / 60.0), num(rec.get("branch_min_to_3rd")),
                       num(rec.get("branch_min_to_10th")), num(rec.get("depth"))))
    fams = collections.Counter(short(n.get("family"), 60) for n in nodes)
    if fams:
        fam, count = fams.most_common(1)[0]
        nums.append("%s parallel starts; page family %s (%s of %s first uses); risky flags: %s"
                    % (num(g(rec, "status", "parallel")), fam, num(count), num(len(nodes)),
                       ", ".join(short(r, 20) for r in rec.get("risky") or ()) or "none"))
    beats = (on.get("shuffles_ge_observed") == 0) if on else (rec.get("shuffle_ge_observed") == 0)
    title = 'Example tree: %s token "%s", %s agents, %s' % (
        rec.get("cls"), token_text(rec.get("token")), num(rec.get("agents")),
        "beats every shuffle" if beats else "reached by some shuffles")
    nm = facts.get("nullmodel") or {}
    tokens = nm.get("tokens") or g(facts, "tokens", "kept_total")
    n = (g(checks, "order_null", "shuffles") if on else shuffles) or 0
    limit = ("Picked from %s tokens: %s trees beat every planned shuffle where chance alone allows up to %s, and "
             "one tree's p of 1/%s is far above a Bonferroni line of %s. An example, not proof."
             % (num(tokens), num(nm.get("trees_beating_every_shuffle")),
                num(int(round(nm.get("chance_beats_upper_bound") or 0))), n + 1,
                tiny(0.05 / tokens) if tokens else "n/a"))
    return {"key": "B", "title": title, "ratios": ratios, "primary": ratios[0], "numbers": nums, "limit": limit}


def finding_c(facts, checks):
    st = g(facts, "cascade", "status") or {}
    adopters = g(facts, "cascade", "adoptions")
    later = (adopters - st.get("first", 0)) if adopters is not None else None
    nums = ["Later adopters %s: parallel starts (no visible earlier use) %s (%s); linked to a named earlier user "
            "%s (%s)" % (num(later), num(st.get("parallel")), pct(st.get("parallel"), later),
                         num(st.get("linked")), pct(st.get("linked"), later))]
    strict = g(checks, "order_null_strict", "links_named")
    lag = g(checks, "order_null_min_lag_120s", "links_named")
    ratios = []
    if strict:
        nums.append("Adopting save not counted as a visit: %s links against an order-null mean of %s; ratio %s; "
                    "z %s; %s of %s shuffles at or above"
                    % (num(strict["observed"]), num(strict["mean"]), num(strict["ratio"], 3), num(strict["z"], 2),
                       num(strict["shuffles_ge_observed"]), num(g(checks, "order_null_strict", "shuffles"))))
        ratios.append(("order null, strict exposure", strict["ratio"]))
    if lag:
        nums.append("120 s minimum lag: %s against %s; ratio %s; z %s; %s of %s at or above"
                    % (num(lag["observed"]), num(lag["mean"]), num(lag["ratio"], 3), num(lag["z"], 2),
                       num(lag["shuffles_ge_observed"]), num(g(checks, "order_null_min_lag_120s", "shuffles"))))
        ratios.append(("order null, 120 s lag", lag["ratio"]))
    pc = g(facts, "hand_check", "population_page_check", "total")
    if pc:
        nums.append("Page check of all %s links: B's save changed A's own line in %s (%s); A's line was on the page "
                    "at B's visit in %s (%s)"
                    % (num(pc.get("links")), num(pc.get("b_save_changed_a_line")),
                       pct(pc.get("b_save_changed_a_line"), pc.get("links")), num(pc.get("a_line_on_page_at_visit")),
                       pct(pc.get("a_line_on_page_at_visit"), pc.get("links"))))
    top5 = g(checks, "without_most_active", "top_5_pct")
    if top5:
        nums.append("Most active 5%% of agents (%s, with %s of agent saves) made exposure-only: linked share %s, "
                    "from %s"
                    % (num(top5.get("agents_made_exposure_only")), pct(top5.get("their_share_of_agent_rows"), 1),
                       num(top5.get("linked_share_of_later_adopters"), 3),
                       num(g(facts, "slots", "F5", "linked_share_observed"), 3)))
    if strict:
        word = verdict(strict)
        title = "Stricter exposure: links %s%s" % (
            word, " (tricks show up in early saves, not after visits)" if word == "below chance" else "")
        limit = ("These rules and this shuffle were chosen after the planned null's result (opt-in checks, %s "
                 "shuffles each); the same counts fit burst editing, not only agents arriving with a trick."
                 % num(g(checks, "order_null_strict", "shuffles")))
    else:
        f5 = g(facts, "slots", "F5") or {}
        ratios.append(("planned label shuffle, linked share", f5.get("linked_share_ratio")))
        title = "Parallel starts outnumber links"
        limit = "Only the planned label shuffle ran, and it is biased toward more links."
    return {"key": "C", "title": title, "ratios": ratios, "primary": ratios[0] if ratios else (None, None),
            "numbers": nums, "limit": limit}


def findings(facts, top, checks, planted):
    return [finding_a(facts, checks, planted), finding_b(facts, top, checks), finding_c(facts, checks)]


def silence_pick(found):
    """The finding with the highest primary observed-to-shuffle ratio (ties go to the earlier letter)."""
    best = None
    for f in found:
        r = f["primary"][1]
        if r is not None and (best is None or r > best["primary"][1]):
            best = f
    return best


def ratio_text(f):
    return "; ".join("%s %s" % (num(r, 3), label) for label, r in f["ratios"] if r is not None) or "no shuffle"


# ---------------------------------------------------------------- markdown


def sources(facts, top, checks, now):
    out = ["facts.json: stage '%s', facts_version %s, core_sha256 %s, run commit %s"
           % (facts.get("stage"), facts.get("facts_version"), facts.get("core_sha256"), g(facts, "code", "commit"))]
    if top:
        out.append("top_trees.json: sha256 %s (matches facts.json), lead %s" %
                   (g(facts, "outputs", "top_trees.json", "sha256"), top.get("lead")))
    if checks:
        out.append("checks.json: %s (opt-in; %s rows, %s tokens, order null %s shuffles, seed %s)"
                   % (checks.get("tool"), num(checks.get("rows")), num(checks.get("tokens")),
                      num(g(checks, "order_null", "shuffles")), g(checks, "order_null", "seed")))
    if facts.get("hand_check"):
        out.append("hand check: %s" % g(facts, "hand_check", "tool"))
    if facts.get("stretch_s1"):
        out.append("stretch S1: patient_zero.offwiki")
    out.append("planted fixtures: tests/test_planted.py at %s" % (now.get("commit") or "n/a"))
    return out


def render_md(label, stamp, ask, found, pick, slots, src, warnings, notes):
    lines = ["# FACTS %s: patient-zero" % label, "",
             "Filled by tools/facts_sheet.py at %s. Numbers and short labels only; the write-up is its "
             "author's own and this sheet has no sentences for it." % stamp, ""]
    lines += ["## Question", "", "%s" % ask, ""]
    if pick:
        lines.append("Highest observed-to-shuffle ratio: %s (%s, %s)." % (pick["key"], num(pick["primary"][1], 3),
                                                                          pick["primary"][0]))
    for n in notes:
        lines.append("Note: %s" % n)
    lines += ["", "## Candidate lead findings", ""]
    for f in found:
        lines += ["### %s. %s" % (f["key"], f["title"]), "", "- Observed / shuffle: %s" % ratio_text(f)]
        lines += ["- %s" % n for n in f["numbers"]]
        lines += ["- Limit: %s" % f["limit"], ""]
    lines += ["## Slots F1 to F14 (%s)" % label, "", "| Slot | Meaning | %s | Source |" % label, "|---|---|---|---|"]
    for slot, meaning in MEANINGS:
        value, where = slots[slot]
        lines.append("| %s | %s | %s | %s |" % (slot, meaning, value.replace("|", "/"), where))
    lines += ["", "## Sources", ""] + ["- %s" % s for s in src]
    if warnings:
        lines += ["", "## Warnings", ""] + ["- %s" % w for w in warnings]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- outline (Markdown subset)


def outline_section(path, marker="outline"):
    """(heading, lines) of the first Markdown heading that contains ``marker``, up to the next one at its level."""
    with open(path, encoding="utf-8-sig") as fh:  # hand-edited; a BOM would hide a first heading
        lines = fh.read().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("#") and marker in line.lower()), None)
    if start is None:
        raise SheetError("no heading containing '%s' in %s" % (marker, path))
    level = len(lines[start]) - len(lines[start].lstrip("#"))
    end = len(lines)
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if line.startswith("#") and len(line) - len(line.lstrip("#")) <= level:
            end = j
            break
    return lines[start].lstrip("#").strip(), lines[start + 1:end]


def inline_md(text):
    out = html.escape(text, quote=False)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return out


def md_blocks_html(lines):
    parts, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "---":
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                head = "".join("<th>%s</th>" % inline_md(c) for c in rows[0])
                body = "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % inline_md(c) for c in r) for r in rows[1:])
                parts.append('<div class="wrap"><table class="outline"><thead><tr>%s</tr></thead><tbody>%s</tbody>'
                             '</table></div>' % (head, body))
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append("<li>%s</li>" % inline_md(lines[i].strip()[2:]))
                i += 1
            parts.append("<ul>%s</ul>" % "".join(items))
            continue
        parts.append("<p>%s</p>" % inline_md(line))
        i += 1
    return "\n".join(parts)


# ---------------------------------------------------------------- HTML

WORDS = (
    ("ratio", "links seen divided by the links a shuffle makes by luck. 1.0 means luck explains it. Above 1 is "
              "more than luck. Below 1 is fewer."),
    ("named link", "agent B used a trick after B could see it on a page where agent A used it first. A candidate, "
                   "not proof."),
    ("parallel start", "B used the trick with no earlier use B could have seen."),
    ("planned label shuffle", "the plan's check: agent names mixed within each day. It is biased: one agent's "
                              "repeat uses turn into several agents, so it makes extra links."),
    ("order null", "the fairer check, added after the planned result. Each agent keeps its own saves, pages and "
                   "times. Only which save came when is mixed, within each day."),
    ("strict exposure", "the save where B uses the trick does not count as B seeing the page."),
)

CSS = """
:root{--bg:#f6f5f1;--surface:#ffffff;--ink:#121212;--muted:#56544e;--rule:#dcdad2;--accent:#1f63b8;
--chip:#e8f0fb;--chip-ink:#123d73;--warn-bg:#fff4e5;--warn-ink:#7a3d00;--fig:#fcfcfb}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111214;--surface:#1a1b1e;
--ink:#f2f1ed;--muted:#aeaca5;--rule:#34353b;--accent:#7fb2ff;--chip:#1d2a3d;--chip-ink:#cfe1ff;
--warn-bg:#33240f;--warn-ink:#ffd29a;--fig:#fcfcfb}}
:root[data-theme="dark"]{--bg:#111214;--surface:#1a1b1e;--ink:#f2f1ed;--muted:#aeaca5;--rule:#34353b;
--accent:#7fb2ff;--chip:#1d2a3d;--chip-ink:#cfe1ff;--warn-bg:#33240f;--warn-ink:#ffd29a;--fig:#fcfcfb}
*{box-sizing:border-box}
html{scroll-snap-type:y proximity;scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:400 clamp(19px,2.1vw,23px)/1.5 system-ui,-apple-system,
"Segoe UI",sans-serif;overflow-x:hidden}
nav{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--rule);display:flex;
gap:4px 16px;flex-wrap:wrap;align-items:center;padding:8px 16px;font-size:16px}
nav a{color:var(--accent);text-decoration:none;font-weight:600}
nav button{margin-left:auto;font:inherit;font-size:15px;background:var(--surface);color:var(--ink);
border:1px solid var(--rule);border-radius:8px;padding:4px 10px;cursor:pointer}
.screen{min-height:100svh;max-width:1080px;margin:0 auto;padding:40px 16px 32px;display:flex;flex-direction:column;
justify-content:center;scroll-snap-align:start}
.screen>*{min-width:0;max-width:100%}
.kicker{color:var(--muted);font-size:16px;margin:0 0 12px}
h1{font-size:clamp(34px,5.4vw,56px);line-height:1.1;margin:0 0 20px}
h2{font-size:clamp(26px,3.4vw,38px);line-height:1.15;margin:0 0 14px}
.letter{color:var(--accent);font-weight:800;margin-right:.35em}
.pick{font-size:1.1em;margin:0 0 16px}
.ratio{display:inline-block;background:var(--chip);color:var(--chip-ink);border-radius:10px;padding:6px 12px;
font-weight:600;margin:0 0 12px}
.nums{margin:0 0 14px;padding-left:1.1em;font-size:.9em;line-height:1.42}.nums li{margin:0 0 6px}
.limit{background:var(--warn-bg);color:var(--warn-ink);border-radius:10px;padding:12px 16px;margin:0}
.yours{border-left:6px solid var(--accent);background:var(--surface);padding:14px 18px;border-radius:8px;margin:18px 0}
.note{color:var(--muted);margin:0 0 8px}
.choices{padding-left:0;list-style:none;margin:8px 0 0}.choices li{margin:0 0 10px}
.choices a{color:var(--ink);text-decoration:none}.choices b{color:var(--accent);margin-right:8px}
.fig{background:var(--fig);border-radius:12px;padding:8px;border:1px solid var(--rule)}
.fig svg{display:block;width:100%;height:auto;max-height:78svh}
.wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:0 0 16px}
table{border-collapse:collapse;width:100%;font-size:17px;background:var(--surface)}
th,td{border:1px solid var(--rule);padding:8px 10px;text-align:left;vertical-align:top}
th{background:var(--chip);color:var(--chip-ink)}
table.slots{min-width:640px}table.outline{min-width:860px}
td.slot{font-weight:700;white-space:nowrap}
.words dt{font-weight:700;color:var(--accent);margin:14px 0 2px}.words dd{margin:0}
.small{font-size:16px;color:var(--muted)}
code{font-size:.9em}
footer{max-width:1080px;margin:0 auto;padding:24px 16px 64px;font-size:15px;color:var(--muted);
overflow-wrap:anywhere}
.screen p,.screen li{overflow-wrap:break-word}
"""

THEME_JS = """
(function(){var b=document.getElementById('theme');if(!b)return;b.addEventListener('click',function(){
var r=document.documentElement,c=r.getAttribute('data-theme'),d=c?c==='dark':
window.matchMedia('(prefers-color-scheme: dark)').matches;r.setAttribute('data-theme',d?'light':'dark');});})();
"""


def inline_svg(path):
    """docs/lead.svg as inline markup: XML declaration dropped, fixed width and height dropped (viewBox kept)."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    text = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", text)
    m = re.search(r"<svg\b[^>]*>", text)
    if not m:
        raise SheetError("%s is not an SVG file" % path)
    tag = re.sub(r'\s(width|height)="[^"]*"', "", m.group(0), count=2)
    return text[:m.start()] + tag + text[m.end():]


def esc(text):
    return html.escape(str(text), quote=True)


def render_html(label, stamp, ask, found, pick, slots, src, warnings, notes, svg=None, outline=None):
    e = esc
    nav = ['<a href="#ask">Question</a>', '<a href="#words">Words</a>'] + ['<a href="#%s">%s</a>' % (f["key"], f["key"]) for f in found]
    if svg:
        nav.append('<a href="#figure">Figure</a>')
    nav += ['<a href="#slots">Numbers</a>']
    if outline:
        nav.append('<a href="#outline">Outline</a>')
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           '<title>SWARM facts %s</title><style>%s</style></head><body>' % (e(label), CSS),
           '<nav>%s<button id="theme" type="button">Light / dark</button></nav>' % " ".join(nav)]
    out.append('<section class="screen" id="ask"><p class="kicker">patient-zero &middot; FACTS %s &middot; %s</p>'
               '<h1>%s</h1>' % (e(label), e(stamp), e(ask)))
    if pick:
        out.append('<p class="pick">Silence picks <b>%s</b>: ratio %s (%s).</p>'
                   % (e(pick["key"]), e(num(pick["primary"][1], 3)), e(pick["primary"][0])))
    for n in notes:
        out.append('<p class="note">%s</p>' % e(n))
    out.append('<div class="yours"><b>The write-up is yours to write.</b> Every sentence in it is yours. This page '
               'has numbers and short labels only. It has no sample sentences for the write-up.</div>')
    out.append('<ul class="choices">%s</ul></section>' % "".join(
        '<li><a href="#%s"><b>%s</b>%s <span class="small">(%s)</span></a></li>'
        % (e(f["key"]), e(f["key"]), e(f["title"]), e(ratio_text(f))) for f in found))
    out.append('<section class="screen" id="words"><h2>Words on this page</h2><dl class="words">%s</dl></section>'
               % "".join("<dt>%s</dt><dd>%s</dd>" % (e(w), e(d)) for w, d in WORDS))
    for f in found:
        out.append('<section class="screen" id="%s"><h2><span class="letter">%s</span>%s</h2>'
                   '<p><span class="ratio">Observed / shuffle: %s</span></p><ul class="nums">%s</ul>'
                   '<p class="limit"><b>Limit:</b> %s</p></section>'
                   % (e(f["key"]), e(f["key"]), e(f["title"]), e(ratio_text(f)),
                      "".join("<li>%s</li>" % e(n) for n in f["numbers"]), e(f["limit"])))
    if svg:
        out.append('<section class="screen" id="figure"><h2>The lead figure</h2>'
                   '<p class="small">docs/lead.svg, the tree in finding B. It has no agent names and no token text.'
                   '</p><div class="fig">%s</div></section>' % svg)
    rows = "".join('<tr><td class="slot">%s</td><td>%s</td><td>%s</td><td class="small">%s</td></tr>'
                   % (e(slot), e(meaning), e(slots[slot][0]), e(slots[slot][1])) for slot, meaning in MEANINGS)
    out.append('<section class="screen" id="slots"><h2>Numbers: slots F1 to F14 (%s)</h2>'
               '<div class="wrap"><table class="slots"><thead><tr><th>Slot</th><th>Meaning</th><th>%s</th>'
               '<th>Source</th></tr></thead><tbody>%s</tbody></table></div></section>' % (e(label), e(label), rows))
    if outline:
        heading, lines = outline
        out.append('<section class="screen" id="outline"><h2>%s</h2>%s</section>'
                   % (e(heading), md_blocks_html(lines)))
    foot = "".join("<li>%s</li>" % e(s) for s in src)
    warn = "".join("<li>%s</li>" % e(w) for w in warnings)
    out.append('<footer><p>Made by tools/facts_sheet.py at %s.</p><ul>%s</ul>%s</footer>'
               % (e(stamp), foot, ("<p>Warnings:</p><ul>%s</ul>" % warn) if warn else ""))
    out.append("<script>%s</script></body></html>" % THEME_JS)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- main


def parse_slot_notes(items):
    notes = {}
    for item in items or ():
        slot, sep, text = item.partition("=")
        slot = slot.strip().upper()
        if not sep or slot not in SLOT_IDS or not text.strip():
            raise SheetError("--slot-note wants SLOT=TEXT with SLOT in F1..F14, got %r" % item)
        notes[slot] = text.strip()
    return notes


def build(out_dir, label="v1", ask=DEFAULT_ASK, notes=(), slot_notes=None, stamp=None, want_html=False,
          svg_path=None, outline_path=None, planted=None, now=None):
    """(markdown, html or None, summary dict). Raises SheetError on bad input."""
    facts, top, checks, warnings = load_inputs(out_dir)
    planted = planted_check() if planted is None else planted
    now = code_now() if now is None else now
    stamp = stamp or datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z")
    slots = slot_values(facts, top, checks, planted, now, slot_notes)
    found = findings(facts, top, checks, planted)
    pick = silence_pick(found)
    src = sources(facts, top, checks, now)
    page = None
    if want_html:
        svg = inline_svg(svg_path) if svg_path and os.path.isfile(svg_path) else None
        if svg_path and svg is None:
            warnings.append("no figure at %s" % svg_path)
        outline = outline_section(outline_path) if outline_path else None
        page = render_html(label, stamp, ask, found, pick, slots, src, warnings, list(notes), svg, outline)
    md = render_md(label, stamp, ask, found, pick, slots, src, warnings, list(notes))
    summary = {"label": label, "pick": pick["key"] if pick else None,
               "ratios": {f["key"]: f["primary"][1] for f in found},
               "titles": {f["key"]: f["title"] for f in found}, "warnings": warnings,
               "planted": planted, "core_sha256": facts.get("core_sha256")}
    return md, page, summary


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="Fill the facts sheet (slots F1-F14, findings A-C) from a run.")
    ap.add_argument("--out", required=True, help="the run's output folder (facts.json, top_trees.json, checks.json)")
    ap.add_argument("--label", default="v1", help="sheet label, for example v1 or v2 (default v1)")
    ap.add_argument("--md", default=None, help="Markdown path (default OUT/FACTS-<label>.md)")
    ap.add_argument("--html", default=None, help="also write one HTML page here")
    ap.add_argument("--svg", default=os.path.join(REPO_DIR, "docs", "lead.svg"),
                    help="lead figure shown inline in the HTML (default docs/lead.svg)")
    ap.add_argument("--outline", default=None, help="Markdown file whose 'Outline' section is shown in the HTML")
    ap.add_argument("--ask", default=DEFAULT_ASK, help="the question at the top")
    ap.add_argument("--note", action="append", default=[], help="a line shown under the question (repeatable)")
    ap.add_argument("--slot-note", action="append", default=[], help="SLOT=TEXT appended to one slot (repeatable)")
    args = ap.parse_args(argv)
    try:
        slot_notes = parse_slot_notes(args.slot_note)
        md_path = args.md or os.path.join(args.out, "FACTS-%s.md" % args.label)
        guard_outputs(md_path, args.html, input_paths(args.out, args.svg, args.outline))
        md, page, summary = build(args.out, args.label, args.ask, args.note, slot_notes,
                                  want_html=bool(args.html), svg_path=args.svg, outline_path=args.outline)
        write_atomic(md_path, md)
        if args.html:
            write_atomic(args.html, page)
    except (SheetError, OSError, ValueError) as exc:
        out.write("facts_sheet: ERROR %s\n" % exc)
        return 2
    out.write("facts_sheet: wrote %s%s\n" % (md_path, (" and " + args.html) if args.html else ""))
    for key in sorted(summary["ratios"]):
        r = summary["ratios"][key]
        out.write("  %s ratio %s: %s\n" % (key, num(r, 3), summary["titles"][key]))
    out.write("  highest ratio (silence pick): %s\n" % summary["pick"])
    for w in summary["warnings"]:
        out.write("  warning: %s\n" % w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
