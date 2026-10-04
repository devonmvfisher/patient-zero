"""Stretch S1: do wiki tokens show up later off the wiki?

Python standard library only, deterministic (no randomness). The run is the
same as ``cli.py`` up to the token finder (load, tokens; no cascade, no
shuffle). It then reads the other files of the same export folder and looks
for the kept wiki tokens (used by 3 or more agents on the wiki) there:

``records.jsonl.gz``
    Agent-related texts found on other sites (pastebins, URL shorteners,
    other wikis, package registries). Each record lists its ``origins``
    (site, kind, date). Records whose origins are all one of the loaded
    wikis are copies of wiki saves and are skipped; every other record is
    one off-wiki event.
``links.jsonl.gz``
    Links found in those texts. Only the ``host`` of a link attached to an
    off-wiki record is used (it is present even when the URL is withheld).
    No link is followed and the URL field is never read.
``shortener-logs.json.gz``
    A URL shortener's admin log: one event per short link (destination URL
    and title, creation time).
``other-wikis.json.gz``
    Added lines on three other public wikis. Reported on its own, not in
    the headline, because the plan's S1 names only the three files above.

Tokens are found with the same extractor as on the wiki
(``patient_zero.tokens.row_tokens``: percent-decoding, stop lists, IP mask),
so a wiki token and an off-wiki token match only when their keys are equal.

Dates
-----
An off-wiki event's date comes from the origin's ``source_date_literal``
(records; POSIX seconds, ISO-8601, or a 14-digit capture stamp), the
shortener log's ``time``, or the other wiki's revision ``time``. The export
does not document these fields, so the reading here is an assumption:

* **published**: the time the text was posted on that site. Origin kinds
  ``paste_candidate``, ``prior_paste``, ``investigator_public_revision_export``
  and ``prior_revision`` (paste creation times and wiki revision times), the
  shortener log's creation time and other-wiki revision times.
* **seen by**: archive captures and investigator fetches (``wayback_capture``,
  ``commoncrawl_capture``, ``saved_primary_projection`` and the like). The text
  existed by then but may be older, so these dates never give a lag; they
  are counted on their own.
* **undated**: no parseable date (empty, ``current``, revision numbers,
  package versions). Most shortener records are undated.

A links event takes the earliest published date of its off-wiki records.

The question, per kept token
----------------------------
``wiki first use``: the token's earliest row on the wiki (any role, so a
blank or human row counts; the token was on the wiki from then), with that
save's clock uncertainty. ``first off-wiki``: the token's earliest published
off-wiki event (off-wiki stamps are to the second: uncertainty 1 s). With
the same clock rule as cascade links:

* ``wiki_first_then_off``: first off-wiki is later than wiki first use by more
  than both uncertainties. **Lag** = first off-wiki minus wiki first use.
* ``off_first``: first off-wiki is earlier by more than both uncertainties.
* ``unresolved``: inside the uncertainty.

Limits (also written into the output): off-wiki records carry no agent
label, so a later off-wiki use cannot be tied to an agent who saw the wiki;
phrase tokens may be shared language (non-phrase classes are reported on
their own); the dates are as the export gives them.

Output
------
A ``stretch_s1`` section (counts, dates, token_ids and classes; no text, no
labels, no URLs), merged into facts.json outside ``core_sha256`` together
with fact-sheet slot F13, the way ``tools/peek.py`` adds ``hand_check``.
A full ``cli.py`` run rewrites facts.json without it, so run this again
after a full run::

    python -m patient_zero.offwiki --input PATH/TO/collusion --facts ../out/facts.json \\
        --json ../out/stretch_s1.json

The command checks that facts.json comes from the same wiki files and token
parameters, and prints counts only. Nothing here opens a URL or runs a
command from the data.
"""

import argparse
import datetime
import gzip
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from statistics import median
from typing import FrozenSet, NamedTuple, Optional

from patient_zero import load as L
from patient_zero.pipeline import _sha256_file, check_out_dir, input_files
from patient_zero.safe import guard_out, short, token_id
from patient_zero.tokens import (CLASSES, MAX_PHRASE_SHARE, MIN_AGENTS, _common_host, _norm_host,
                                 find_tokens, row_tokens)

RECORDS = "records.jsonl.gz"
LINKS = "links.jsonl.gz"
SHORTENER = "shortener-logs.json.gz"
OTHER_WIKIS = "other-wikis.json.gz"
OFF_FILES = (RECORDS, LINKS, SHORTENER, OTHER_WIKIS)

SRC_RECORDS = "records"
SRC_LINKS = "links"
SRC_SHORTENER = "shortener_log"
SRC_OTHER = "other_wikis"
HEADLINE_SOURCES = (SRC_RECORDS, SRC_LINKS, SRC_SHORTENER)
SOURCES = HEADLINE_SOURCES + (SRC_OTHER,)

OFF_UNCERTAINTY = 1.0  # off-wiki stamps are to the second
PUBLISHED_KINDS = frozenset(("paste_candidate", "prior_paste", "investigator_public_revision_export",
                             "prior_revision"))
GRADE_PUBLISHED = "published"
GRADE_SEEN = "seen_by"
GRADE_NONE = "undated"
GRADES = (GRADE_PUBLISHED, GRADE_SEEN, GRADE_NONE)
VERDICTS = ("wiki_first_then_off", "off_first", "unresolved")
NON_PHRASE = tuple(c for c in CLASSES if c != "phrase")
EARLIEST = 1577836800.0  # 2020-01-01: older (or newer than LATEST) stamps are not dates
LATEST = 1830297600.0  # 2028-01-01
EXAMPLES = 10

# Whitelists: only these keys survive parsing (nested keys included).
_RECORD_KEYS = frozenset(("id", "text", "origins", "kind", "site", "source_date_literal"))
_LINK_KEYS = frozenset(("host", "record_ids", "url_withheld"))
_SHORTENER_KEYS = frozenset(("sites", "site", "links", "url", "title", "time"))
_OTHER_KEYS = frozenset(("pages", "wiki", "revisions", "added", "time"))

_POSIX_RE = re.compile(r"^\d{10}$")
_CAPTURE_RE = re.compile(r"^\d{14}$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


class OffEvent(NamedTuple):
    """One off-wiki appearance place: a record, a link host, a short link or a revision."""

    source: str  # one of SOURCES
    time: Optional[float]  # published time (POSIX seconds) or None
    seen_by: Optional[float]  # earliest capture or fetch time, only when there is no published time
    text: str  # text to search (IPs masked); empty for links
    keys: FrozenSet[str]  # token keys given directly (links: the host)


# ---------------------------------------------------------------- reading


def _plausible(secs):
    return secs is not None and EARLIEST <= secs < LATEST


def parse_stamp(literal):
    """POSIX seconds from a date literal (10-digit POSIX, 14-digit capture, ISO-8601), or None."""
    lit = (literal or "").strip() if isinstance(literal, str) else ""
    secs = None
    try:
        if _POSIX_RE.match(lit):
            secs = float(lit)
        elif _CAPTURE_RE.match(lit):
            secs = datetime.datetime.strptime(lit, "%Y%m%d%H%M%S").replace(
                tzinfo=datetime.timezone.utc).timestamp()
        elif _ISO_RE.match(lit):
            secs = L.parse_time(lit)
    except ValueError:
        secs = None
    return secs if _plausible(secs) else None


def origin_time(kind, literal):
    """(POSIX seconds or None, grade) for one record origin (see the module docstring)."""
    secs = parse_stamp(literal)
    if secs is None:
        return None, GRADE_NONE
    return secs, (GRADE_PUBLISHED if kind in PUBLISHED_KINDS else GRADE_SEEN)


def is_wiki_site(site, groups):
    """True when a record origin's site is one of the loaded wikis.

    ``prowiki.org/dse`` matches group ``dse`` by its last path segment; a bare
    host such as ``dorfwiki.org`` matches group ``dorfwiki`` by its first label.
    """
    s = (site or "").strip().lower().rstrip("/")
    if not s:
        return False
    groups = {g.lower() for g in groups if g}
    if "/" in s:
        return s.rsplit("/", 1)[-1] in groups
    return s.split(".", 1)[0] in groups


def _read_lines(path, keys):
    return (obj for _n, obj in L._read_json_lines(path, keys))


def _read_json(path, keys):
    with L._open_text(path) as fh:
        return json.load(fh, object_pairs_hook=L._keep(keys))


def _text(value):
    return L.scrub_ips(value if isinstance(value, str) else "")


def load_off_wiki(data_dir, groups):
    """Read the off-wiki files. Returns (events, stats); stats hold counts and dates only.

    Missing files are counted under ``missing_files``; at least the records
    file is needed (ValueError otherwise).
    """
    stats = {"missing_files": [n for n in OFF_FILES if not os.path.isfile(os.path.join(data_dir, n))]}
    if RECORDS in stats["missing_files"]:
        raise ValueError("%s not found in the export folder" % RECORDS)
    events = []
    ip_masked = Counter()

    # records
    rec = Counter()
    by_kind = defaultdict(Counter)
    by_site = Counter()
    wiki_sites = set()
    off_time = {}  # record id -> published time or None (off-wiki records only)
    off_seen = {}
    for obj in _read_lines(os.path.join(data_dir, RECORDS), _RECORD_KEYS):
        rec["records"] += 1
        origins = [o for o in obj.get("origins") or [] if isinstance(o, dict)]
        off = []
        for o in origins:
            site = o.get("site") if isinstance(o.get("site"), str) else ""
            if is_wiki_site(site, groups):
                wiki_sites.add(site)
            else:
                off.append(o)
        if not off:
            rec["wiki_only_skipped"] += 1
            continue
        rec["off_wiki"] += 1
        if len(off) < len(origins):
            rec["off_wiki_also_on_wiki"] += 1
        published, seen = [], []
        for o in off:
            kind = o.get("kind") if isinstance(o.get("kind"), str) else ""
            secs, grade = origin_time(kind, o.get("source_date_literal"))
            by_kind[kind][grade] += 1
            by_site[o.get("site") if isinstance(o.get("site"), str) else ""] += 1
            if grade == GRADE_PUBLISHED:
                published.append(secs)
            elif grade == GRADE_SEEN:
                seen.append(secs)
        when = min(published) if published else None
        seen_by = min(seen) if (seen and when is None) else None
        rec[GRADE_PUBLISHED if when is not None else GRADE_SEEN if seen_by is not None else GRADE_NONE] += 1
        text, n_ip = _text(obj.get("text"))
        ip_masked[SRC_RECORDS] += n_ip
        rid = obj.get("id")
        if isinstance(rid, str):
            off_time[rid] = when
            off_seen[rid] = seen_by
        events.append(OffEvent(SRC_RECORDS, when, seen_by, text, frozenset()))
    stats[SRC_RECORDS] = {
        "records": rec["records"],
        "wiki_only_skipped": rec["wiki_only_skipped"],
        "off_wiki": rec["off_wiki"],
        "off_wiki_also_on_wiki": rec["off_wiki_also_on_wiki"],
        "off_wiki_by_date": {g: rec[g] for g in GRADES},
        "origins_by_kind": {short(k): {g: c[g] for g in GRADES}
                            for k, c in sorted(by_kind.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))},
        "origins_by_site": {short(s): n for s, n in sorted(by_site.items(), key=lambda kv: (-kv[1], kv[0]))},
        "wiki_sites_skipped": sorted(short(s) for s in wiki_sites),
    }

    # links: host of each link attached to an off-wiki record
    lk = Counter()
    if LINKS not in stats["missing_files"]:
        for obj in _read_lines(os.path.join(data_dir, LINKS), _LINK_KEYS):
            lk["links"] += 1
            ids = [i for i in obj.get("record_ids") or [] if isinstance(i, str) and i in off_time]
            if not ids:
                lk["on_wiki_records_only"] += 1
                continue
            lk["on_off_wiki_records"] += 1
            if obj.get("url_withheld") is True:
                lk["url_withheld"] += 1
            raw = obj.get("host") if isinstance(obj.get("host"), str) else ""
            host = None if L.is_ip_shaped(raw) else _norm_host(raw)
            if not host or _common_host(host):
                lk["host_skipped"] += 1
                continue
            times = [off_time[i] for i in ids if off_time[i] is not None]
            seens = [off_seen[i] for i in ids if off_seen[i] is not None]
            when = min(times) if times else None
            seen_by = min(seens) if (seens and when is None) else None
            lk[GRADE_PUBLISHED if when is not None else GRADE_SEEN if seen_by is not None else GRADE_NONE] += 1
            events.append(OffEvent(SRC_LINKS, when, seen_by, "", frozenset(("host:" + host,))))
    stats[SRC_LINKS] = {k: lk[k] for k in ("links", "on_wiki_records_only", "on_off_wiki_records",
                                           "url_withheld", "host_skipped")}
    stats[SRC_LINKS]["by_date"] = {g: lk[g] for g in GRADES}

    # shortener log
    sh = Counter()
    sites = []
    if SHORTENER not in stats["missing_files"]:
        obj = _read_json(os.path.join(data_dir, SHORTENER), _SHORTENER_KEYS)
        for site in (obj.get("sites") or []) if isinstance(obj, dict) else []:
            if not isinstance(site, dict):
                continue
            sites.append(short(site.get("site")))
            for link in site.get("links") or []:
                if not isinstance(link, dict):
                    continue
                sh["links"] += 1
                when = parse_stamp(link.get("time"))
                sh[GRADE_PUBLISHED if when is not None else GRADE_NONE] += 1
                text, n_ip = _text("%s\n%s" % (link.get("url") or "", link.get("title") or ""))
                ip_masked[SRC_SHORTENER] += n_ip
                events.append(OffEvent(SRC_SHORTENER, when, None, text, frozenset()))
    stats[SRC_SHORTENER] = {"sites": sites, "links": sh["links"],
                            "by_date": {g: sh[g] for g in GRADES}}

    # other wikis (reported on its own)
    ow = Counter()
    if OTHER_WIKIS not in stats["missing_files"]:
        obj = _read_json(os.path.join(data_dir, OTHER_WIKIS), _OTHER_KEYS)
        for page in (obj.get("pages") or []) if isinstance(obj, dict) else []:
            if not isinstance(page, dict):
                continue
            ow["pages"] += 1
            for rev in page.get("revisions") or []:
                if not isinstance(rev, dict):
                    continue
                ow["revisions"] += 1
                added = [a for a in rev.get("added") or [] if isinstance(a, str)]
                when = parse_stamp(rev.get("time"))
                ow[GRADE_PUBLISHED if when is not None else GRADE_NONE] += 1
                text, n_ip = _text("\n".join(added))
                ip_masked[SRC_OTHER] += n_ip
                events.append(OffEvent(SRC_OTHER, when, None, text, frozenset()))
    stats[SRC_OTHER] = {"pages": ow["pages"], "revisions": ow["revisions"],
                        "by_date": {g: ow[g] for g in GRADES}}
    stats["ip_masked_in_text"] = {s: ip_masked[s] for s in SOURCES}
    return events, stats


# ---------------------------------------------------------------- matching


def find_hits(events, tokens):
    """{token key: [(source, published time or None, seen-by time or None, event index), ...]}.

    Only kept ``tokens`` are looked for; the event index points into ``events``.
    """
    kept = set(tokens)
    hits = defaultdict(list)
    for idx, ev in enumerate(events):
        keys = set(ev.keys & kept) if ev.keys else set()
        if ev.text:
            found, _stopped = row_tokens(ev.text)
            keys.update(k for k in found if k in kept)
        for key in sorted(keys):
            hits[key].append((ev.source, ev.time, ev.seen_by, idx))
    return dict(hits)


def verdict(t_wiki, u_wiki, t_off):
    """Clock-rule verdict for the earliest published off-wiki time against wiki first use."""
    gap = u_wiki + OFF_UNCERTAINTY
    if t_off - t_wiki > gap:
        return "wiki_first_then_off"
    if t_wiki - t_off > gap:
        return "off_first"
    return "unresolved"


def _quantile(sorted_values, q):
    """Linear-interpolated quantile of a sorted list (deterministic)."""
    if not sorted_values:
        return None
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def lag_stats(lags):
    """Distribution of lags in seconds: count, median (s, h, days), quartiles and range in days."""
    if not lags:
        return {"n": 0, "median_seconds": None, "median_hours": None, "median_days": None,
                "p25_days": None, "p75_days": None, "min_seconds": None, "max_days": None}
    s = sorted(lags)
    mid = median(s)
    return {
        "n": len(s),
        "median_seconds": round(mid, 1),
        "median_hours": round(mid / 3600.0, 2),
        "median_days": round(mid / 86400.0, 2),
        "p25_days": round(_quantile(s, 0.25) / 86400.0, 2),
        "p75_days": round(_quantile(s, 0.75) / 86400.0, 2),
        "min_seconds": round(s[0], 1),
        "max_days": round(s[-1] / 86400.0, 2),
    }


def summarise(tokens, rows, hits, sources, events=None):
    """Per-token verdicts and counts for off-wiki appearances from ``sources`` only.

    With ``events`` (the list ``hits`` indexes into), also reports how
    concentrated the first off-wiki appearances are: many tokens can first
    show up off the wiki in one long text, so the lag is also given with each
    such text counted once.
    """
    per_class = {c: Counter() for c in CLASSES}
    lags = defaultdict(list)  # "all", class, "non_phrase" -> lags of wiki_first_then_off tokens
    total = Counter()
    any_later = 0
    examples = []
    first_event = defaultdict(list)  # event index -> lags of the tokens it first shows off the wiki
    only_source = Counter()
    for key in sorted(tokens):
        apps = [a for a in hits.get(key, ()) if a[0] in sources]
        if not apps:
            continue
        token = tokens[key]
        cls = token.cls
        total["seen_off_wiki"] += 1
        per_class[cls]["seen_off_wiki"] += 1
        srcs = {a[0] for a in apps}
        if len(srcs) == 1:
            only_source[next(iter(srcs))] += 1
        published = sorted((a[1], a[3]) for a in apps if a[1] is not None)
        if not published:
            grade = GRADE_SEEN if any(a[2] is not None for a in apps) else GRADE_NONE
            total["only_" + grade] += 1
            per_class[cls]["only_" + grade] += 1
            continue
        total["with_published_date"] += 1
        per_class[cls]["with_published_date"] += 1
        first = rows[token.rows[0]]
        t_off, idx = published[0]
        v = verdict(first.time, first.uncertainty, t_off)
        total[v] += 1
        per_class[cls][v] += 1
        if any(verdict(first.time, first.uncertainty, t) == "wiki_first_then_off" for t, _i in published):
            any_later += 1
        if v == "wiki_first_then_off":
            lag = t_off - first.time
            lags["all"].append(lag)
            lags[cls].append(lag)
            if cls != "phrase":
                lags["non_phrase"].append(lag)
            first_event[idx].append(lag)
            examples.append((cls != "phrase", len(published), key, cls, token.agents, lag))
    examples.sort(key=lambda e: (-e[0], -e[1], token_id(e[2])))
    keys = ("seen_off_wiki", "with_published_date") + VERDICTS + ("only_" + GRADE_SEEN, "only_" + GRADE_NONE)
    out = {k: total[k] for k in keys}
    out["any_published_appearance_after_wiki_first_use"] = any_later
    out["seen_off_wiki_from_one_source_only"] = {s: only_source[s] for s in SOURCES if s in sources}
    out["lag_wiki_first_to_off_wiki"] = lag_stats(lags["all"])
    biggest = max(first_event.items(), key=lambda kv: (len(kv[1]), -kv[0]), default=(None, []))
    out["first_off_wiki_texts"] = {
        "distinct": len(first_event),
        "largest_tokens": len(biggest[1]),
        "largest_source": events[biggest[0]].source if (events is not None and biggest[0] is not None) else None,
        "largest_time": (L.iso_utc(events[biggest[0]].time)
                         if (events is not None and biggest[0] is not None) else None),
        "lag_one_per_text": lag_stats([median(v) for v in first_event.values()]),
    }
    out["non_phrase"] = {k: sum(per_class[c][k] for c in NON_PHRASE) for k in keys}
    out["non_phrase"]["lag_wiki_first_to_off_wiki"] = lag_stats(lags["non_phrase"])
    out["by_class"] = {}
    for c in CLASSES:
        row = {k: per_class[c][k] for k in keys}
        lag = lag_stats(lags[c])
        row["lag_n"] = lag["n"]
        row["lag_median_days"] = lag["median_days"]
        out["by_class"][c] = row
    out["examples"] = [{"token_id": token_id(key), "cls": cls, "wiki_agents": agents,
                        "off_wiki_published_appearances": n, "lag_hours": round(lag / 3600.0, 2)}
                       for _np, n, key, cls, agents, lag in examples[:EXAMPLES]]
    return out


def _date_context(events, wiki_span):
    """Published off-wiki events before, during and after the wiki's own time span, per source."""
    start, end = wiki_span
    table = {}
    for src in SOURCES:
        times = sorted(e.time for e in events if e.source == src and e.time is not None)
        table[src] = {
            "published_events": len(times),
            "before_wiki_start": sum(1 for t in times if t < start),
            "during_wiki_span": sum(1 for t in times if start <= t <= end),
            "after_wiki_end": sum(1 for t in times if t > end),
            "first": L.iso_utc(times[0]) if times else None,
            "last": L.iso_utc(times[-1]) if times else None,
        }
    return table


def stretch_s1(rows, tokens, data_dir, groups=None):
    """The ``stretch_s1`` section for kept ``tokens`` of time-sorted wiki ``rows``.

    Returns {"status": "skipped", "reason": ...} when the records file is missing.
    """
    started = time.perf_counter()
    if groups is None:
        groups = sorted({r.group for r in rows if r.group})
    try:
        events, load_stats = load_off_wiki(data_dir, groups)
    except ValueError as exc:
        return {"status": "skipped", "reason": str(exc)}
    hits = find_hits(events, tokens)
    wiki_span = (min(r.time for r in rows), max(r.time for r in rows)) if rows else (0.0, 0.0)
    inputs = [{"name": n, "bytes": os.path.getsize(os.path.join(data_dir, n)),
               "sha256": _sha256_file(os.path.join(data_dir, n))}
              for n in OFF_FILES if os.path.isfile(os.path.join(data_dir, n))]
    headline = summarise(tokens, rows, hits, HEADLINE_SOURCES, events)
    section = {
        "status": "done",
        "question": "do wiki tokens (3+ agents on the wiki) show up later off the wiki?",
        "inputs": inputs,
        "params": {
            "kept_wiki_tokens": len(tokens),
            "headline_sources": list(HEADLINE_SOURCES),
            "extra_sources": [SRC_OTHER],
            "off_wiki_uncertainty_seconds": OFF_UNCERTAINTY,
            "published_kinds": sorted(PUBLISHED_KINDS),
            "wiki_groups": [short(g) for g in groups],
            "wiki_first_use": "earliest wiki row holding the token (any role), with its uncertainty",
            "lag": "earliest published off-wiki appearance minus wiki first use",
        },
        "sources": load_stats,
        "wiki_span": [L.iso_utc(wiki_span[0]), L.iso_utc(wiki_span[1])],
        "published_dates_vs_wiki_span": _date_context(events, wiki_span),
        "headline": headline,
        "by_source": {src: {k: v for k, v in summarise(tokens, rows, hits, (src,), events).items()
                            if k not in ("by_class", "examples", "seen_off_wiki_from_one_source_only")}
                      for src in SOURCES},
        "limits": [
            "Off-wiki records carry no agent label: a later off-wiki use cannot be tied to an agent "
            "who saw the wiki, and the same agents may have posted in both places.",
            "Phrase tokens may be shared language; non_phrase reports hosts, files, commands, code "
            "words and conventions on their own.",
            "Dates are read from undocumented export fields: published = paste creation and wiki "
            "revision times, shortener creation times; captures and fetches are seen_by and give no lag.",
            "Most shortener records are undated and count only in seen_off_wiki.",
            "Matching uses the wiki token extractor on off-wiki text; a token that is reworded off "
            "the wiki is missed.",
        ],
        "seconds": round(time.perf_counter() - started, 2),
    }
    return section


def f13_slot(section):
    """Fact-sheet slot F13 from a ``stretch_s1`` section."""
    if section.get("status") != "done":
        return "skipped: %s" % section.get("reason", "unknown")
    h = section["headline"]
    np_ = h["non_phrase"]
    lag = h["lag_wiki_first_to_off_wiki"]
    np_lag = np_["lag_wiki_first_to_off_wiki"]
    return {
        "kept_wiki_tokens": section["params"]["kept_wiki_tokens"],
        "tokens_seen_off_wiki": h["seen_off_wiki"],
        "tokens_with_published_off_wiki_date": h["with_published_date"],
        "tokens_wiki_first_then_off": h["wiki_first_then_off"],
        "tokens_off_first": h["off_first"],
        "tokens_unresolved": h["unresolved"],
        "median_lag_days": lag["median_days"],
        "median_lag_hours": lag["median_hours"],
        "lag_p25_days": lag["p25_days"],
        "lag_p75_days": lag["p75_days"],
        "first_off_wiki_texts": h["first_off_wiki_texts"]["distinct"],
        "largest_text_tokens": h["first_off_wiki_texts"]["largest_tokens"],
        "median_lag_days_one_per_text": h["first_off_wiki_texts"]["lag_one_per_text"]["median_days"],
        "non_phrase_tokens_wiki_first_then_off": np_["wiki_first_then_off"],
        "non_phrase_median_lag_days": np_lag["median_days"],
        "sources": section["params"]["headline_sources"],
    }


# ---------------------------------------------------------------- facts.json


def check_facts(facts, data_dir, kept, min_agents, max_phrase_share, given):
    """Raise ValueError unless facts.json comes from the same wiki files and token parameters."""
    if facts.get("input", {}).get("adapter") != "wiki":
        raise ValueError("facts.json is not from a wiki run")
    params = facts.get("params", {})
    if params.get("min_agents") != min_agents or params.get("max_phrase_share") != max_phrase_share:
        raise ValueError("facts.json token parameters differ from this run")
    if params.get("given_texts") != given:
        raise ValueError("facts.json used %s given text(s); this run used %d"
                         % (params.get("given_texts"), given))
    if facts.get("tokens", {}).get("kept_total") != kept:
        raise ValueError("facts.json kept %s tokens; this run kept %d"
                         % (facts.get("tokens", {}).get("kept_total"), kept))
    want = {f["name"]: f["sha256"] for f in facts.get("input", {}).get("files", [])}
    have = {f["name"]: f["sha256"] for f in input_files("wiki", data_dir)}
    if want != have:
        raise ValueError("facts.json input files differ from the export folder given")


def merge_into_facts(facts_path, section):
    """Write ``stretch_s1`` and slot F13 into facts.json (outside core_sha256, which is unchanged)."""
    with open(facts_path, "r", encoding="utf-8") as fh:
        facts = json.load(fh)
    section = dict(section)
    section["facts_core_sha256_at_write"] = facts.get("core_sha256")
    facts["stretch_s1"] = section
    slots = facts.setdefault("slots", {})
    slots["F13"] = f13_slot(section)
    slots.get("pending", {}).pop("F13", None)
    _write(facts_path, json.dumps(facts, indent=1, ensure_ascii=True) + "\n")
    return facts


def _write(path, text):
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------- command line


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(
        prog="python -m patient_zero.offwiki",
        description="Stretch S1: count wiki tokens that show up later off the wiki (records, links, "
                    "shortener log of the same export). Prints counts only.")
    ap.add_argument("--input", required=True, help="the wiki export folder")
    ap.add_argument("--facts", help="facts.json of the same run: stretch_s1 and slot F13 are merged in")
    ap.add_argument("--json", help="also write the stretch_s1 section alone here (outside the repo)")
    ap.add_argument("--min-agents", type=int, default=MIN_AGENTS)
    ap.add_argument("--max-phrase-share", type=float, default=MAX_PHRASE_SHARE)
    ap.add_argument("--given", action="append", default=[],
                    help="a shared goal or prompt text file, as given to cli.py (repeatable)")
    args = ap.parse_args(argv)
    try:
        for path in (args.facts, args.json):
            if path:
                check_out_dir(os.path.dirname(os.path.abspath(path)))
        guard_out(args.json, [args.input, args.facts] + list(args.given), "--json")
        if args.facts:  # facts.json is merged in place by design; it may not sit in the export folder
            guard_out(args.facts, [args.input] + list(args.given), "--facts")
        given = []
        for path in args.given:
            with open(path, "r", encoding="utf-8") as fh:
                given.append(fh.read())
        rows, _load_stats = L.load("wiki", args.input)
        tokens, _token_stats = find_tokens(rows, min_agents=args.min_agents, given_texts=given,
                                           max_phrase_share=args.max_phrase_share)
        if args.facts:
            with open(args.facts, "r", encoding="utf-8") as fh:
                check_facts(json.load(fh), args.input, len(tokens), args.min_agents,
                            args.max_phrase_share, len(given))
        section = stretch_s1(rows, tokens, args.input)
        if args.json:
            _write(args.json, json.dumps(section, indent=1, ensure_ascii=True) + "\n")
        if args.facts:
            merge_into_facts(args.facts, section)
    except (OSError, ValueError) as exc:
        print("offwiki: ERROR: %s" % exc, file=out)
        return 2
    if section["status"] != "done":
        print("offwiki: skipped: %s" % section["reason"], file=out)
        return 0
    s = section["sources"]
    h = section["headline"]
    lag = h["lag_wiki_first_to_off_wiki"]
    np_ = h["non_phrase"]
    print("off-wiki events: records %d (of %d; %d published-dated), link hosts %d, shortener links %d, "
          "other-wiki revisions %d" % (
              s[SRC_RECORDS]["off_wiki"], s[SRC_RECORDS]["records"],
              s[SRC_RECORDS]["off_wiki_by_date"][GRADE_PUBLISHED],
              s[SRC_LINKS]["on_off_wiki_records"] - s[SRC_LINKS]["host_skipped"],
              s[SRC_SHORTENER]["links"], s[SRC_OTHER]["revisions"]), file=out)
    print("kept wiki tokens %d: seen off the wiki %d, with a published off-wiki date %d" % (
        section["params"]["kept_wiki_tokens"], h["seen_off_wiki"], h["with_published_date"]), file=out)
    print("wiki first, then off the wiki: %d (median lag %s days, %s h; quartiles %s to %s days); "
          "off the wiki first: %d; unresolved: %d" % (
              h["wiki_first_then_off"], lag["median_days"], lag["median_hours"], lag["p25_days"],
              lag["p75_days"], h["off_first"], h["unresolved"]), file=out)
    ft = h["first_off_wiki_texts"]
    print("those first show up in %d distinct off-wiki texts; the largest holds %d of them (%s, %s); "
          "median lag with each text once: %s days" % (
              ft["distinct"], ft["largest_tokens"], ft["largest_source"], ft["largest_time"],
              ft["lag_one_per_text"]["median_days"]), file=out)
    print("non-phrase classes: wiki first then off %d (median lag %s days), off first %d" % (
        np_["wiki_first_then_off"], np_["lag_wiki_first_to_off_wiki"]["median_days"], np_["off_first"]),
        file=out)
    print("stretch seconds: %.1f" % section["seconds"], file=out)
    if args.facts:
        print("merged stretch_s1 and slot F13 into %s (core_sha256 unchanged)" % args.facts, file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
