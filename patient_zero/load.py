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
    (no zone means UTC) or POSIX seconds (a number, or a string of digits
    such as Slack's ``"1768039200.000100"``).

    A ``.csv`` or ``.tsv`` file (optionally gzipped) is read with the
    standard ``csv`` module: a header line, then one event per line. Empty
    cells in optional columns count as absent; ``human`` is true for
    ``true``, ``yes`` or ``1``.

    ``field_map`` (``--map agent=user,time=ts,place=channel,text=body``)
    names the log's own column for each field, so an export need not be
    renamed first. In JSON a dotted path walks into nested objects
    (``author.name``); a key that itself holds a dot is matched as-is first.
    Fields left out of the map are read under their own names. A mapped
    column that the log does not have is an error that names the column.

Privacy rules applied at load, in both adapters:

* Fields are read from a fixed whitelist while parsing; every other field
  (including the export's half-IP field) is discarded before a row exists.
  With ``field_map`` the whitelist is the mapped columns (and each part of
  a dotted path) plus the unmapped fields' own names.
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
    rows, stats = load("generic", "examples/fake_log.csv",
                       parse_map("agent=user,time=ts,place=channel,text=body"))

    python -m patient_zero.load --adapter wiki --input path/to/collusion
    python -m patient_zero.load --adapter generic --input chat.csv --map agent=user,text=body

The command line prints counts only, never text or names.
"""

import argparse
import contextlib
import csv
import datetime
import difflib
import gzip
import ipaddress
import json
import math
import os
import re
import sys
from collections import Counter
from typing import NamedTuple, Optional

from patient_zero.safe import short

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
_GENERIC_FIELDS = ("agent", "time", "place", "text", "id", "group", "family", "uncertainty_seconds",
                   "time_grade", "human")
_GENERIC_KEYS = frozenset(_GENERIC_FIELDS)
_GENERIC_REQUIRED = ("agent", "time", "place", "text")
_MISSING = object()  # a field the event does not have
_CSV_TRUE = frozenset(("true", "yes", "1"))
_CSV_FALSE = frozenset(("false", "no", "0", ""))
_CSV_FIELD_LIMIT = 1 << 24  # characters per cell (the csv default, 131072, is too small for long posts)
_COLUMNS_SHOWN = 20


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


_SECONDS_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)\Z")


def parse_time(value):
    """ISO-8601 string (Z or offset; none means UTC) or POSIX seconds -> float.

    POSIX seconds may be a number or a string of digits with an optional
    fraction (CSV cells, Slack's ``ts``). Milliseconds are not detected.
    """
    if isinstance(value, bool):
        raise ValueError("time must be a string or a number")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("time must be a string or a number")
    text = value.strip()
    if _SECONDS_RE.match(text):
        return float(text)
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
            raise FileNotFoundError(
                "%s not found. --input must be the unpacked collusion.wiki export folder holding "
                "revisions.jsonl.gz and pages.jsonl.gz (labels.jsonl.gz too, for agent names); "
                "README.md, section \"Run it on the German wiki data\", says where to get it." % path)

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


def parse_map(spec):
    """``"agent=user,time=ts,place=channel,text=body"`` -> ``{"agent": "user", ...}``.

    Keys are patient-zero's field names; values are the log's own column
    names, or dotted paths into nested JSON objects (``author.name``). An empty
    or missing spec gives ``{}``. Column names cannot hold ``,`` or ``=``.
    """
    if spec is None or not spec.strip():
        return {}
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        field, sep, source = part.partition("=")
        field, source = field.strip(), source.strip()
        if not sep or not field or not source:
            raise ValueError("--map entry %r should look like field=column, for example agent=user"
                             % short(part))
        if field not in _GENERIC_KEYS:
            raise ValueError("--map: %r is not a field patient-zero reads; the fields are %s"
                             % (short(field), ", ".join(_GENERIC_FIELDS)))
        if field in out:
            raise ValueError("--map names %s twice" % field)
        if any(not piece for piece in source.split(".")):
            raise ValueError("--map %s=%s: a dotted path cannot have an empty part" % (field, short(source)))
        out[field] = source
    return out


def generic_format(path):
    """'csv', 'tsv' or 'jsonl' (anything else), from the file name; a .gz ending is ignored."""
    name = path.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".tsv"):
        return "tsv"
    return "jsonl"


def _sources(field_map):
    """Field -> the column it is read from (its own name unless the map says otherwise)."""
    return {f: (field_map or {}).get(f, f) for f in _GENERIC_FIELDS}


def _resolve(obj, source):
    """obj[source] if that key exists, else walk a dotted path; _MISSING when absent."""
    if source in obj:
        return obj[source]
    value = obj
    for piece in source.split("."):
        if not isinstance(value, dict) or piece not in value:
            return _MISSING
        value = value[piece]
    return value


def _column_list(header):
    names = [short(h, 40) for h in header[:_COLUMNS_SHOWN]]
    more = len(header) - len(names)
    return ", ".join(names) + (" and %d more" % more if more > 0 else "")


def _no_column(path, where, field, source, field_map, header=None):
    """The plain sentence for a column the log does not have."""
    if field in (field_map or {}):
        msg = "%s%s has no column '%s' (--map %s=%s)." % (path, where, short(source), field, short(source))
    else:
        msg = ("%s%s has no column '%s'; if the log calls it something else, name that column with "
               "--map %s=COLUMN." % (path, where, field, field))
    if header is not None:
        msg += " Its columns are: %s." % (_column_list(header) or "(none)")
    return msg


def _json_records(path, sources):
    """Yield ('line N', N, {field: value or _MISSING}) from a JSON Lines log."""
    keep = set()
    for source in sources.values():
        keep.add(source)
        keep.update(source.split("."))
    for number, obj in _read_json_lines(path, frozenset(keep)):
        yield "line %d" % number, number, {f: _resolve(obj, s) for f, s in sources.items()}


def _csv_records(path, sources, field_map, delimiter):
    """Yield ('line N', N, {field: string or _MISSING}) from a CSV or TSV log with a header line.

    Only the needed columns are copied out of each line. Empty cells in
    optional columns are _MISSING.
    """
    old_limit = csv.field_size_limit()
    csv.field_size_limit(max(old_limit, _CSV_FIELD_LIMIT))
    opener = gzip.open if path.lower().endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            try:
                header = [h.strip() for h in next(reader)]
            except StopIteration:
                return
            for f in list(_GENERIC_REQUIRED) + [f for f in _GENERIC_FIELDS if f in (field_map or {})]:
                if sources[f] not in header:
                    raise ValueError(_no_column(path, "", f, sources[f], field_map, header))
            cols = {}
            for f, source in sources.items():
                if source in header:
                    if header.count(source) > 1:
                        raise ValueError("%s: column '%s' appears %d times in the header line; rename one "
                                         "of them" % (path, short(source), header.count(source)))
                    cols[f] = header.index(source)
            for cells in reader:
                if not any(cell.strip() for cell in cells):
                    continue
                record = {}
                for f in _GENERIC_FIELDS:
                    i = cols.get(f)
                    value = cells[i] if i is not None and i < len(cells) else _MISSING
                    if f not in _GENERIC_REQUIRED and value is not _MISSING and not value.strip():
                        value = _MISSING
                    record[f] = value
                yield "line %d" % reader.line_num, reader.line_num, record
    except csv.Error as exc:
        raise ValueError("%s: not valid CSV (%s)" % (path, exc)) from None
    finally:
        csv.field_size_limit(old_limit)


def _csv_human(value, path, where):
    word = value.strip().lower()
    if word in _CSV_TRUE:
        return True
    if word in _CSV_FALSE:
        return False
    raise ValueError("%s %s: human must be true, yes, 1, false, no, 0 or empty" % (path, where))


def _human_flag(value, path, where):
    """The human flag from a CSV cell or a JSON value. Anything unclear is an error (fail closed).

    JSON ``true``/``false``, the CSV words (true, yes, 1, false, no, 0, empty)
    as strings, and the numbers 1 and 0 are understood, so a mapped column
    that writes ``1`` or ``"yes"`` never lets a human speaker be named.
    """
    if value is _MISSING or value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _csv_human(value, path, where)
    if isinstance(value, (int, float)) and value in (0, 1):
        return value == 1
    raise ValueError("%s %s: human must be true, yes, 1, false, no, 0 or empty" % (path, where))


def _csv_seconds(value, path, where):
    try:
        unc = float(value.strip())
    except ValueError:
        unc = -1.0
    if not math.isfinite(unc) or unc < 0:
        raise ValueError("%s %s: uncertainty_seconds must be a number >= 0" % (path, where))
    return unc


def _generic_records(path, field_map):
    """(sources, record iterator, cells) for a generic log; cells is True for CSV/TSV (all strings)."""
    for field in field_map:
        if field not in _GENERIC_KEYS:
            raise ValueError("--map: %r is not a field patient-zero reads" % short(field))
    sources = _sources(field_map)
    fmt = generic_format(path)
    if fmt == "jsonl":
        return sources, _json_records(path, sources), False
    return sources, _csv_records(path, sources, field_map, "\t" if fmt == "tsv" else ","), True


def generic_human_labels(path, field_map=None):
    """Labels of the human speakers in a generic log, read the way ``load_generic`` reads them.

    For masking human handles in snippets (tools/peek.py). Never written out.
    """
    field_map = dict(field_map or {})
    _sources_, records, _cells = _generic_records(path, field_map)
    humans = set()
    with contextlib.closing(records):
        for where, _number, obj in records:
            human, label = _human_flag(obj["human"], path, where), obj["agent"]
            if human is True and isinstance(label, str) and label.strip():
                humans.add(label.strip())
    return humans


def load_generic(path, field_map=None):
    """Load a JSON Lines, CSV or TSV log with agent, time, place, text. Returns (rows, stats).

    ``field_map`` maps field names to the log's own columns (see ``parse_map``).
    """
    field_map = dict(field_map or {})
    sources, records, cells = _generic_records(path, field_map)
    stats = Counter()
    per_group = Counter()
    agents = set()
    rows = []
    seen_ids = set()
    found = set()  # mapped optional fields seen on at least one line
    with contextlib.closing(records):  # closes the file at once, even when a line is bad
        for where, number, obj in records:
            missing = [k for k in _GENERIC_REQUIRED if obj[k] is _MISSING]
            if missing:
                mapped = [k for k in missing if k in field_map]
                if mapped:
                    raise ValueError(_no_column(path, " " + where, mapped[0], sources[mapped[0]], field_map))
                raise ValueError("%s %s: missing field(s) %s; if the log calls a field something else, name its "
                                 "column with --map, for example --map %s=COLUMN"
                                 % (path, where, ", ".join(missing), missing[0]))
            found.update(f for f in field_map if obj[f] is not _MISSING)
            try:
                when = parse_time(obj["time"])
            except ValueError as exc:
                raise ValueError("%s %s: bad time (%s)" % (path, where, exc)) from None
            text = obj["text"]
            if not isinstance(text, str):
                raise ValueError("%s %s: text must be a string" % (path, where))
            if obj["place"] is None or isinstance(obj["place"], (dict, list)):
                raise ValueError("%s %s: place must be a string" % (path, where))
            for field in ("agent", "id", "group", "family", "time_grade"):  # str(dict) could carry a handle
                if isinstance(obj[field], (dict, list)):
                    raise ValueError("%s %s: %s must be a string or a number, not a JSON object or list "
                                     "(with --map, point at a field inside it, such as author.name)"
                                     % (path, where, field))
            row_id = (str(obj["id"]) if obj["id"] is not _MISSING and obj["id"] is not None
                      else "line-%d" % number)
            if row_id in seen_ids:
                raise ValueError("%s %s: duplicate id" % (path, where))
            seen_ids.add(row_id)

            label = obj["agent"]
            label = str(label).strip() if label is not None else ""
            if label and is_ip_shaped(label):
                stats["ip_shaped_labels_blanked"] += 1
                label = ""
            human = _human_flag(obj["human"], path, where)
            if not label:
                agent, role = None, ROLE_BLANK
                stats["blank_rows"] += 1
            elif human is True:
                agent, role = None, ROLE_HUMAN
                stats["human_rows"] += 1
            else:
                agent, role = label, ROLE_AGENT
                agents.add(label)

            text, n_ip = scrub_ips(text)
            stats["ip_masked_in_text"] += n_ip
            place, n_ip = scrub_ips(str(obj["place"]))
            stats["ip_masked_in_place"] += n_ip
            unc = obj["uncertainty_seconds"]
            if unc is _MISSING:
                unc = 0
            elif cells:
                unc = _csv_seconds(unc, path, where)
            if isinstance(unc, bool) or not isinstance(unc, (int, float)) or unc < 0:
                raise ValueError("%s %s: uncertainty_seconds must be a number >= 0" % (path, where))
            opt = {k: (str(obj[k]) if obj[k] is not _MISSING and obj[k] is not None else None)
                   for k in ("group", "family", "time_grade")}
            per_group[opt["group"]] += 1
            rows.append(Row(
                row_id=row_id,
                agent=agent,
                role=role,
                time=when,
                uncertainty=float(unc),
                time_grade=opt["time_grade"],
                place=place,
                family=opt["family"],
                group=opt["group"],
                text=text,
            ))
    if rows and not cells:  # a CSV header was checked up front; JSON keys can differ per line
        for field in _GENERIC_FIELDS:
            if field in field_map and field not in found:
                raise ValueError("%s: no line has column '%s' (--map %s=%s)."
                                 % (path, short(field_map[field]), field, short(field_map[field])))

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


def load(adapter, path, field_map=None):
    """Load path with the named adapter ('wiki' or 'generic'). Returns (rows, stats).

    ``field_map`` (generic only) maps field names to the log's own columns.
    """
    try:
        fn = ADAPTERS[adapter]
    except KeyError:
        raise ValueError("unknown adapter %r (choose from %s)" % (adapter, ", ".join(sorted(ADAPTERS))))
    if field_map:
        if adapter != "generic":
            raise ValueError("--map works only with --adapter generic; the %s adapter reads fixed "
                             "file and field names" % adapter)
        return fn(path, field_map)
    return fn(path)


MAP_HELP = ("generic only: the log's own column for each field, for example "
            "agent=user,time=ts,place=channel,text=body; dotted paths reach into nested JSON "
            "(agent=author.name); fields left out are read under their own names")


def input_note(adapter, path, field_map=None):
    """Extra facts.json "input" keys for a non-default generic run: {format?, map?} ({} otherwise).

    A JSON Lines run with no map adds nothing, so its facts (and core hash)
    are the same as before ``--map`` existed.
    """
    note = {}
    if adapter == "generic":
        fmt = generic_format(path)
        if fmt != "jsonl":
            note["format"] = fmt
        if field_map:
            note["map"] = {f: short(field_map[f]) for f in _GENERIC_FIELDS if f in field_map}
    return note


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="Load a log and print row counts (no text, no names).")
    ap.add_argument("--adapter", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--input", required=True,
                    help="export folder (wiki) or a JSON Lines, CSV or TSV file (generic)")
    ap.add_argument("--map", help=MAP_HELP)
    args = ap.parse_args(argv)
    try:
        _rows, stats = load(args.adapter, args.input, parse_map(args.map))
    except (OSError, ValueError) as exc:
        print("load: ERROR: %s" % exc, file=out)
        return 2
    json.dump(stats, out, indent=1)
    out.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
