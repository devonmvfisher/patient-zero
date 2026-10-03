"""Load multi-agent logs into rows of (agent, time, place, text).

Python standard library only. Two adapters:

``wiki``
    The collusion.wiki export folder: ``revisions.jsonl.gz`` (one save per
    line), ``pages.jsonl.gz`` (page family per page) and ``labels.jsonl.gz``
    (which labels are human handles). One save becomes one row:

    * ``text`` is only what the save added: the new-side lines of its
      ``insert`` and ``replace`` hunks. If a save has no hunks, the loader
      diffs it against its ``diff_base`` save with ``difflib``; a save with
      neither is a page creation and adds its whole body.
    * ``place`` is the page id, ``family`` its page family, ``group`` the wiki.
    * ``time`` is POSIX seconds (UTC), with ``uncertainty`` (seconds) and
      ``time_grade`` copied from the save.

``generic``
    JSON Lines (``.jsonl``, ``.ndjson``, optionally gzipped), one event per
    line with ``agent``, ``time``, ``place`` and ``text``. Optional fields:
    ``id``, ``uncertainty_seconds``, ``group``, ``family``, ``time_grade`` and
    ``human`` (true marks a human speaker). ``time`` is an ISO-8601 string
    (no zone means UTC) or POSIX seconds.

Privacy rules applied at load, in both adapters:

* Fields are read from a fixed whitelist while parsing; every other field
  (including the export's half-IP field) is discarded before a row exists.
* IP-shaped strings (IPv4, IPv4 with ``x``/``*`` wildcards, IPv6) in text and
  place names are replaced with ``[ip]``. An IP-shaped agent label is treated
  as blank. Two-octet fragments cannot be told from decimals and are not
  touched; dropping the half-IP field is the guard for those.
* Blank labels and human handles are exposure-only: the row keeps its text,
  time and place (so others can be exposed to it), but ``agent`` is ``None``
  and ``role`` says ``"blank"`` or ``"human"``. Human handles never leave the
  loader.

Usage::

    rows, stats = load("wiki", "path/to/collusion")
    rows, stats = load("generic", "examples/fake_log.ndjson")

    python -m patient_zero.load --adapter wiki --input path/to/collusion

The command line prints counts only, never text or names.
"""

import argparse
import datetime
import difflib
import gzip
import ipaddress
import json
import os
import re
import sys
from collections import Counter
from typing import NamedTuple, Optional

ADDED_OPS = ("insert", "replace")
IP_MASK = "[ip]"

ROLE_AGENT = "agent"
ROLE_BLANK = "blank"
ROLE_HUMAN = "human"

# Whitelists: only these keys survive parsing (nested hunk keys included).
_REVISION_KEYS = frozenset(
    ("rev_id", "page_id", "wiki", "label", "time", "time_grade", "uncertainty_seconds",
     "body", "hunks", "diff_base", "op", "a0", "a1", "b0", "b1")
)
_PAGE_KEYS = frozenset(("page_id", "page_family"))
_LABEL_KEYS = frozenset(("label", "is_human_handle"))
_GENERIC_KEYS = frozenset(
    ("id", "agent", "time", "place", "text", "uncertainty_seconds", "group", "family",
     "time_grade", "human")
)
_GENERIC_REQUIRED = ("agent", "time", "place", "text")


class Row(NamedTuple):
    """One event: who (agent, role), when (time, uncertainty), where, and what."""

    row_id: str
    agent: Optional[str]  # None when the row is exposure-only
    role: str  # "agent", "blank" or "human"
    time: float  # POSIX seconds, UTC
    uncertainty: float  # seconds either side of time
    time_grade: Optional[str]
    place: str
    family: Optional[str]
    group: Optional[str]
    text: str

    @property
    def exposure_only(self):
        return self.role != ROLE_AGENT


# ---------------------------------------------------------------- IP scrub

_IPV4_RE = re.compile(r"(?<![0-9.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![0-9]|\.[0-9])")
_IPV4_WILD_RE = re.compile(
    r"(?<![0-9.])(\d{1,3})\.(\d{1,3})\.(?:x{1,3}|\*)\.(?:x{1,3}|\*)(?![0-9a-z])", re.IGNORECASE
)
_IPV6_RE = re.compile(r"(?<![0-9a-z:])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![0-9a-z:])", re.IGNORECASE)


def _octets_ok(groups):
    return all(0 <= int(g) <= 255 for g in groups)


def _ipv6_ok(candidate):
    try:
        ipaddress.IPv6Address(candidate)
    except ValueError:
        return False
    groups = [g for g in candidate.split(":") if g]
    if "::" not in candidate:
        return len(groups) == 8
    return len(groups) >= 2 and (len(groups) >= 3 or any(len(g) >= 3 for g in groups))


def scrub_ips(text):
    """Return (text with IP-shaped strings replaced by [ip], number replaced)."""
    count = 0

    def v4(m):
        nonlocal count
        if _octets_ok(m.groups()):
            count += 1
            return IP_MASK
        return m.group(0)

    def v6(m):
        nonlocal count
        if _ipv6_ok(m.group(0)):
            count += 1
            return IP_MASK
        return m.group(0)

    if any(ch.isdigit() for ch in text):
        text = _IPV4_RE.sub(v4, text)
        text = _IPV4_WILD_RE.sub(v4, text)
    if ":" in text:
        text = _IPV6_RE.sub(v6, text)
    return text, count


def is_ip_shaped(value):
    """True when the whole value (trimmed) is an IP-shaped string."""
    value = value.strip()
    return bool(value) and scrub_ips(value)[0] == IP_MASK


# ---------------------------------------------------------------- helpers


def _keep(keys):
    """json object_pairs_hook that drops every key outside the whitelist."""

    def hook(pairs):
        return {k: v for k, v in pairs if k in keys}

    return hook


def _open_text(path):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _read_json_lines(path, keys):
    """Yield (line_number, whitelisted dict) for each non-empty line."""
    hook = _keep(keys)
    with _open_text(path) as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line, object_pairs_hook=hook)
                except json.JSONDecodeError as exc:
                    raise ValueError("%s line %d: not valid JSON (%s)" % (path, number, exc.msg)) from None
                if not isinstance(obj, dict):
                    raise ValueError("%s line %d: expected a JSON object" % (path, number))
                yield number, obj


def parse_time(value):
    """ISO-8601 string (Z or offset; none means UTC) or POSIX seconds -> float."""
    if isinstance(value, bool):
        raise ValueError("time must be a string or a number")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("time must be a string or a number")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    stamp = datetime.datetime.fromisoformat(text)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=datetime.timezone.utc)
    return stamp.timestamp()


def iso_utc(seconds):
    """POSIX seconds -> 'YYYY-MM-DDTHH:MM:SSZ'."""
    stamp = datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def added_text(body, hunks, base_body):
    """Text a save added: new-side lines of insert/replace hunks, joined by newlines.

    Lines are ``body.split("\\n")``, matching the export's line numbers. With no
    hunks, diff against ``base_body``; with no base either, the whole body.
    """
    lines = body.split("\n")
    if hunks:
        ops = [(h.get("op"), h.get("b0"), h.get("b1")) for h in hunks if isinstance(h, dict)]
    elif base_body is None:
        ops = [("insert", 0, len(lines))] if body else []
    else:
        matcher = difflib.SequenceMatcher(None, base_body.split("\n"), lines, autojunk=False)
        ops = [(op, b0, b1) for op, _a0, _a1, b0, b1 in matcher.get_opcodes()]
    parts = []
    for op, b0, b1 in ops:
        if op in ADDED_OPS and isinstance(b0, int) and isinstance(b1, int) and b1 > b0:
            parts.extend(lines[b0:b1])
    return "\n".join(parts)


def _sorted(rows):
    rows.sort(key=lambda r: (r.time, r.row_id))
    return rows


# ---------------------------------------------------------------- wiki adapter


def load_wiki(data_dir):
    """Load the collusion.wiki export folder. Returns (rows, stats)."""
    rev_path = os.path.join(data_dir, "revisions.jsonl.gz")
    page_path = os.path.join(data_dir, "pages.jsonl.gz")
    label_path = os.path.join(data_dir, "labels.jsonl.gz")
    for path in (rev_path, page_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    families = {}
    for _n, p in _read_json_lines(page_path, _PAGE_KEYS):
        families[p.get("page_id")] = p.get("page_family") or None

    humans = set()
    if os.path.isfile(label_path):
        for _n, lab in _read_json_lines(label_path, _LABEL_KEYS):
            if lab.get("is_human_handle") is True and isinstance(lab.get("label"), str):
                humans.add(lab["label"].strip())

    saves = [obj for _n, obj in _read_json_lines(rev_path, _REVISION_KEYS)]
    bodies = {s.get("rev_id"): s.get("body") or "" for s in saves}

    stats = Counter()
    per_group, per_grade = Counter(), Counter()
    labels, agents = set(), set()
    rows = []
    for s in saves:
        row_id = str(s.get("rev_id"))
        body = s.get("body") or ""
        hunks = s.get("hunks") or []
        base_id = s.get("diff_base")
        base_body = None
        if hunks:
            stats["text_from_hunks"] += 1
        elif base_id is not None and base_id in bodies:
            base_body = bodies[base_id]
            stats["text_from_difflib"] += 1
        elif base_id is not None:
            stats["base_missing_whole_body"] += 1
        else:
            stats["page_created_whole_body"] += 1
        text, n_ip = scrub_ips(added_text(body, hunks, base_body))
        stats["ip_masked_in_text"] += n_ip
        if not text.strip():
            stats["saves_without_added_text"] += 1

        label = s.get("label")
        label = label.strip() if isinstance(label, str) else ""
        if label and is_ip_shaped(label):
            stats["ip_shaped_labels_blanked"] += 1
            label = ""
        if not label:
            agent, role = None, ROLE_BLANK
            stats["blank_saves"] += 1
        elif label in humans:
            agent, role = None, ROLE_HUMAN
            labels.add(label)
            stats["human_saves"] += 1
        else:
            agent, role = label, ROLE_AGENT
            labels.add(label)
            agents.add(label)

        page_id = s.get("page_id")
        if page_id not in families:
            stats["saves_on_unknown_page"] += 1
        place, n_ip = scrub_ips(str(page_id))
        stats["ip_masked_in_place"] += n_ip
        group = s.get("wiki") or None
        grade = s.get("time_grade") or None
        unc = s.get("uncertainty_seconds")
        per_group[group] += 1
        per_grade[grade] += 1
        rows.append(Row(
            row_id=row_id,
            agent=agent,
            role=role,
            time=parse_time(s.get("time")),
            uncertainty=float(unc) if isinstance(unc, (int, float)) and not isinstance(unc, bool) else 0.0,
            time_grade=grade,
            place=place,
            family=families.get(page_id),
            group=group,
            text=text,
        ))
    del saves, bodies

    out = {
        "adapter": "wiki",
        "rows": len(rows),
        "labels_with_text": len(labels),
        "agents": len(agents),
        "human_labels": len(humans & labels),
        "blank_saves": stats["blank_saves"],
        "human_saves": stats["human_saves"],
        "groups": dict(sorted(per_group.items(), key=lambda kv: (-kv[1], str(kv[0])))),
        "group_count": len(per_group),
        "pages": len(families),
        "families": len(set(f for f in families.values() if f)),
        "time_grades": dict(sorted(per_grade.items(), key=lambda kv: (-kv[1], str(kv[0])))),
    }
    for key in ("text_from_hunks", "text_from_difflib", "page_created_whole_body",
                "base_missing_whole_body", "saves_without_added_text", "saves_on_unknown_page",
                "ip_masked_in_text", "ip_masked_in_place", "ip_shaped_labels_blanked"):
        out[key] = stats[key]
    if rows:
        out["time_span"] = [iso_utc(min(r.time for r in rows)), iso_utc(max(r.time for r in rows))]
    return _sorted(rows), out


# ---------------------------------------------------------------- generic adapter


def load_generic(path):
    """Load a JSON Lines log with agent, time, place, text. Returns (rows, stats)."""
    stats = Counter()
    per_group = Counter()
    agents = set()
    rows = []
    seen_ids = set()
    for number, obj in _read_json_lines(path, _GENERIC_KEYS):
        missing = [k for k in _GENERIC_REQUIRED if k not in obj]
        if missing:
            raise ValueError("%s line %d: missing field(s) %s" % (path, number, ", ".join(missing)))
        try:
            when = parse_time(obj["time"])
        except ValueError as exc:
            raise ValueError("%s line %d: bad time (%s)" % (path, number, exc)) from None
        text = obj["text"]
        if not isinstance(text, str):
            raise ValueError("%s line %d: text must be a string" % (path, number))
        if obj["place"] is None or isinstance(obj["place"], (dict, list)):
            raise ValueError("%s line %d: place must be a string" % (path, number))
        row_id = str(obj["id"]) if obj.get("id") is not None else "line-%d" % number
        if row_id in seen_ids:
            raise ValueError("%s line %d: duplicate id" % (path, number))
        seen_ids.add(row_id)

        label = obj["agent"]
        label = str(label).strip() if label is not None else ""
        if label and is_ip_shaped(label):
            stats["ip_shaped_labels_blanked"] += 1
            label = ""
        if not label:
            agent, role = None, ROLE_BLANK
            stats["blank_rows"] += 1
        elif obj.get("human") is True:
            agent, role = None, ROLE_HUMAN
            stats["human_rows"] += 1
        else:
            agent, role = label, ROLE_AGENT
            agents.add(label)

        text, n_ip = scrub_ips(text)
        stats["ip_masked_in_text"] += n_ip
        place, n_ip = scrub_ips(str(obj["place"]))
        stats["ip_masked_in_place"] += n_ip
        unc = obj.get("uncertainty_seconds", 0)
        if isinstance(unc, bool) or not isinstance(unc, (int, float)) or unc < 0:
            raise ValueError("%s line %d: uncertainty_seconds must be a number >= 0" % (path, number))
        group = str(obj["group"]) if obj.get("group") is not None else None
        per_group[group] += 1
        rows.append(Row(
            row_id=row_id,
            agent=agent,
            role=role,
            time=when,
            uncertainty=float(unc),
            time_grade=str(obj["time_grade"]) if obj.get("time_grade") is not None else None,
            place=place,
            family=str(obj["family"]) if obj.get("family") is not None else None,
            group=group,
            text=text,
        ))

    out = {
        "adapter": "generic",
        "rows": len(rows),
        "agents": len(agents),
        "blank_rows": stats["blank_rows"],
        "human_rows": stats["human_rows"],
        "places": len(set(r.place for r in rows)),
        "groups": dict(sorted(per_group.items(), key=lambda kv: (-kv[1], str(kv[0])))),
        "ip_masked_in_text": stats["ip_masked_in_text"],
        "ip_masked_in_place": stats["ip_masked_in_place"],
        "ip_shaped_labels_blanked": stats["ip_shaped_labels_blanked"],
    }
    if rows:
        out["time_span"] = [iso_utc(min(r.time for r in rows)), iso_utc(max(r.time for r in rows))]
    return _sorted(rows), out


ADAPTERS = {"wiki": load_wiki, "generic": load_generic}


def load(adapter, path):
    """Load path with the named adapter ('wiki' or 'generic'). Returns (rows, stats)."""
    try:
        fn = ADAPTERS[adapter]
    except KeyError:
        raise ValueError("unknown adapter %r (choose from %s)" % (adapter, ", ".join(sorted(ADAPTERS))))
    return fn(path)


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="Load a log and print row counts (no text, no names).")
    ap.add_argument("--adapter", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--input", required=True, help="export folder (wiki) or JSON Lines file (generic)")
    args = ap.parse_args(argv)
    try:
        _rows, stats = load(args.adapter, args.input)
    except (OSError, ValueError) as exc:
        print("load: ERROR: %s" % exc, file=out)
        return 2
    json.dump(stats, out, indent=1)
    out.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
