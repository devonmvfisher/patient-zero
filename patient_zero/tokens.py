"""Find candidate "tricks": distinctive tokens that several agents use.

Python standard library only, deterministic (no randomness). Input is the
time-sorted rows from ``patient_zero.load``; output is one ``Token`` per
candidate, keyed ``"<class>:<value>"`` (conventions: ``"convention:<kind>:<value>"``).

Token classes (PLAN section 4.5)
--------------------------------
``host``
    Host names from URLs (``scheme://host``) and bare domains on a short list
    of top-level domains. Lower-cased, ``www.`` and port dropped. E-mail
    addresses are skipped. IPs were already masked by the loader.
``file``
    File names with a known extension (``notes.txt``, ``run.sh``), also inside
    URL paths. Lower-cased.
``command``
    A line (or a backtick span) that starts with a known command name, after
    an optional ``$``/``%``/``>`` prompt and ``sudo``. The token is the command
    name plus its sub-command when there is one (``git clone``, ``pip install``).
    Nothing is ever run.
``code``
    Code words: ``snake_case`` and ``UPPER_SNAKE`` names, ``lowerCamelCase``
    names, dotted calls (``mod.func(`` gives ``mod.func``) and one-word
    backtick spans.
``phrase``
    Word 3-grams inside one line (letters only, case-folded) with at least two
    content words (not stop words or common web words, 3+ letters). "Rare":
    phrases found in more than ``max_phrase_share`` of agent rows, and in more
    than 25 of them (``PHRASE_CAP_FLOOR``), are template or common language
    and are dropped.
``convention`` (coordination conventions), three kinds:
    ``prefix``  naming prefixes: the first segment of a multi-part name such
    as ``relay-node-7`` or ``team_alpha`` (gives ``relay-``, ``team_``); a
    hyphen name needs a digit in a later segment, so ``e-mail`` is not one.
    ``mailbox`` mailbox or inbox style names (inbox, outbox, mailbox,
    postbox, dead drop, message board, German Postfach and Briefkasten).
    ``control`` all-caps control words: a fixed list (HOLD, VETO, STOP, ...),
    any all-caps word of 5+ letters, or a 3-4 letter one used as a signal (at
    line start, in brackets, or before ``:``/``!``). Common acronyms are
    skipped.

Before extraction, percent-escapes are decoded (``%2F`` to ``/``, two rounds) and
IP-shaped values revealed by decoding are masked again.

Filters, in order
-----------------
1. Stop lists of common web words, common hosts and files, and acronyms.
2. ``given_texts``: tokens found in a shared goal or prompt text are dropped
   (handed down, not spread).
3. Phrases above ``max_phrase_share`` of agent rows, and above the floor of
   25 agent rows, are dropped.
4. Keep tokens used by ``min_agents`` (3) or more distinct agents. Blank and
   human rows are exposure-only: their rows stay in ``Token.rows`` (so others
   can be exposed to them) but they never count as users.
5. Clock rule: take each of the first 3 distinct agents' first use. Two uses
   are in clock order only when the later is later by more than the sum of
   both saves' ``uncertainty`` seconds (the same rule as cascade links). If no
   pair of those 3 first uses is in clock order, nobody can say who was first,
   and the token is dropped.

Risky-spread words
------------------
Each token also carries the risky-word *types* found on the same lines as the
token (bypass, workaround, blocked, captcha, rate_limit, answer, share, proxy,
cors, relay). Types only; no text around them is kept.

Usage::

    rows, _ = load("wiki", path)
    tokens, stats = find_tokens(rows)

    python -m patient_zero.tokens --adapter wiki --input PATH --summary out/tokens-summary.json

The command line and the summary file hold counts only, never token text,
agent names or row text.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from typing import NamedTuple, Tuple
from urllib.parse import unquote

from patient_zero.load import ADAPTERS, load, scrub_ips
from patient_zero.safe import guard_out

CLASSES = ("host", "file", "command", "code", "phrase", "convention")
CONVENTION_KINDS = ("prefix", "mailbox", "control")
MIN_AGENTS = 3
FIRST_USERS = 3
MAX_PHRASE_SHARE = 0.02
PHRASE_CAP_FLOOR = 25  # never call a phrase common below this many agent rows


class Token(NamedTuple):
    """One candidate trick and every row that holds it."""

    key: str  # "host:ferry.example.org", "convention:control:HOLD"
    cls: str  # one of CLASSES
    kind: str  # cls, or the convention kind
    value: str
    rows: Tuple[int, ...]  # indices into the time-sorted rows, all roles
    agents: int  # distinct agents (exposure-only rows never count)
    first_users: Tuple[Tuple[str, int], ...]  # (agent, row index) of the first 3 agents
    risky: Tuple[str, ...]  # risky-word types seen on the token's lines


# ---------------------------------------------------------------- word lists

STOPWORDS = frozenset("""
a an and or but if then else of to in on at by for with from as is are was were be been
being it its this that these those there here we you he she they i me my our your their his
her them us not no yes do does did done have has had will would can could should may might
must shall so than too very just also only all any some each more most other such into out
up down over under about after before again what which who whom when where why how one two
new use used using via per etc get got let lets see now well like make made way even much
many own same both few off our ours
der die das den dem des ein eine einer eines einem einen und oder aber wenn dann als wie im
am an auf aus bei mit nach von vor zu zum zur fur fuer für uber ueber über unter durch gegen
ohne um ist sind war waren sein seine seiner wird werden wurde wurden hat haben hatte kann
konnen können muss soll nicht kein keine auch noch nur schon sehr es er sie wir ihr ich du
man sich dass daß diese dieser dieses diesem hier dort da was wer wo alle mehr bis seit ja
nein mal doch ab so wird
""".split())

WEB_WORDS = frozenset("""
http https www com org net de html htm php index page pages seite seiten link links click
klick home wiki edit bearbeiten url urls site web website online email mail login logout
user users benutzer name title titel text content image images bild file files datei
category kategorie back next previous top menu search suche help hilfe view show read more
source quelle sources quellen post posts comment comments reply main start
""".split())

COMMON_HOSTS = (
    "google.com", "google.de", "wikipedia.org", "wikimedia.org", "wikidata.org", "github.com",
    "youtube.com", "youtu.be", "twitter.com", "x.com", "facebook.com", "instagram.com",
    "linkedin.com", "reddit.com", "w3.org", "schema.org", "mozilla.org", "apple.com",
    "microsoft.com", "amazon.com", "archive.org", "stackoverflow.com", "gmail.com", "bing.com",
    "duckduckgo.com", "creativecommons.org", "gnu.org",
)
EXACT_ONLY_HOSTS = frozenset(("example.com", "example.org", "example.net"))  # placeholders

COMMON_FILES = frozenset((
    "index.html", "index.htm", "index.php", "robots.txt", "sitemap.xml", "readme.md",
    "readme.txt", "license.txt", "license.md", "package.json", "requirements.txt", "style.css",
))

BARE_TLDS = (
    "com", "org", "net", "io", "ai", "co", "dev", "app", "xyz", "info", "re", "ly", "gl", "gd",
    "cc", "link", "site", "online", "top", "ru", "cn", "uk", "eu", "ch", "fr", "nl", "us", "tk",
    "ml", "ga", "cf", "gq", "pw", "ws", "wiki", "page", "biz", "tv", "fm", "gg", "de", "es",
    "se", "edu", "gov", "onion", "run", "cloud", "tech", "store", "club", "space", "live",
    "pro", "lol", "art", "zone",
)

FILE_EXTS = (
    "py", "js", "mjs", "cjs", "ts", "tsx", "jsx", "sh", "bash", "ps1", "bat", "json", "ndjson",
    "yaml", "yml", "toml", "ini", "cfg", "conf", "env", "txt", "md", "csv", "tsv", "xml",
    "html", "htm", "php", "rb", "java", "kt", "swift", "cpp", "sql", "db", "sqlite", "log",
    "pdf", "png", "jpg", "jpeg", "gif", "svg", "webp", "zip", "tar", "gz", "tgz", "7z", "rar",
    "exe", "dll", "apk", "msi", "deb", "rpm", "ipynb", "pem", "crt", "lock", "wasm", "css",
    "scss", "docx", "xlsx", "pptx", "mp3", "mp4", "wav",
)

COMMAND_VERBS = frozenset((
    "curl", "wget", "git", "pip", "pip3", "npm", "npx", "yarn", "pnpm", "python", "python3",
    "node", "deno", "bun", "bash", "sh", "zsh", "ssh", "scp", "rsync", "docker", "kubectl",
    "helm", "terraform", "chmod", "chown", "apt", "apt-get", "brew", "cargo", "rustc", "gcc",
    "java", "javac", "ls", "cat", "grep", "sed", "awk", "echo", "export", "cd", "mkdir", "rm",
    "mv", "cp", "tar", "unzip", "crontab", "systemctl", "service", "nohup", "screen", "tmux",
    "openssl", "base64", "ping", "dig", "nslookup", "whois", "traceroute", "pwsh", "powershell",
    "gh", "heroku", "vercel", "netlify", "wrangler", "flyctl", "ollama", "uv", "pipx", "conda",
))

CODE_STOP = frozenset(("iphone", "ipad", "ipod", "ebay", "macos", "javascript", "typescript"))

CONTROL_WORDS = frozenset((
    "HOLD", "VETO", "STOP", "GO", "NOGO", "WAIT", "ACK", "NACK", "DONE", "CLAIM", "CLAIMED",
    "LOCK", "LOCKED", "UNLOCK", "RELEASE", "RELEASED", "ABORT", "HALT", "PAUSE", "RESUME",
    "READY", "BUSY", "URGENT", "PING", "PONG", "HANDOFF", "ESCALATE", "YIELD", "OVERRIDE",
    "DENY", "APPROVE", "APPROVED", "REJECT", "REJECTED",
))

CAPS_STOP = frozenset("""
HTTP HTTPS HTML URL URI API JSON XML CSS PDF FAQ USA UK EU UN AI KI ID OK PS DE EN CEO CTO
IT TV PC SQL PHP CSV SSH SSL TLS DNS IP TCP UDP GPU CPU RAM USB NOTE INFO WARN WARNING ERROR
DEBUG TRACE FATAL TODO FIXME XXX NB PPS UTC CET CEST GMT AM PM ISBN DOI GNU MIT BSD GPL
ASCII UTF RSS LLM GPT AGI NASA GDPR DSGVO FTP SMTP IMAP POP RFC ISO CVE SHA MD AWS GCP CLI
GUI SDK IDE OS YAML TOML README LICENSE MP JPG PNG GIF SVG ZIP EXE NEU ENDE TEIL
""".split())

PREFIX_STOP = frozenset("""
http https mailto ftp file wiki user benutzer kategorie category datei image bild vorlage
template spezial special diskussion talk hilfe help portal iso utf sha md rfc cve covid
windows win x e v de en u s
""".split())

RISKY_PATTERNS = (
    ("bypass", r"\bbypass\w*|\bumgeh\w*|\bumgang\w*"),
    ("workaround", r"\bwork-?arounds?\b"),
    ("blocked", r"\bblock(?:ed|s|ing)?\b|\bgesperrt\b|\bblockiert\w*"),
    ("captcha", r"captcha"),
    ("rate_limit", r"\brate[\s_-]?limit\w*|\bthrottl\w*"),
    ("answer", r"\banswer\w*"),
    ("share", r"\bshar(?:e|ed|es|ing)\b"),
    ("proxy", r"\bprox(?:y|ies|ied)\b"),
    ("cors", r"\bcors\b"),
    ("relay", r"relay"),
)
RISKY_TYPES = tuple(name for name, _ in RISKY_PATTERNS)
_RISKY_RES = tuple((name, re.compile(pat, re.IGNORECASE)) for name, pat in RISKY_PATTERNS)

# ---------------------------------------------------------------- patterns

_URL_HOST_RE = re.compile(r"\b(?:https?|ftp)://([^\s/?#<>\"'`\[\]{}|\\^()]+)", re.IGNORECASE)
_BARE_HOST_RE = re.compile(
    r"(?<![\w@./:-])((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:"
    + "|".join(BARE_TLDS) + r"))(?![\w-]|\.[a-z0-9]|\()",
    re.IGNORECASE,
)
_HOST_OK_RE = re.compile(
    r"^(?:[0-9a-zÀ-￿](?:[0-9a-zÀ-￿-]{0,61}[0-9a-zÀ-￿])?\.)+"
    r"(?:[a-zÀ-￿]{2,24}|xn--[a-z0-9-]{1,59})$"
)
_FILE_RE = re.compile(
    r"(?<![\w.-])([\w-](?:[\w.-]{0,78}[\w-])?\.(?:" + "|".join(FILE_EXTS) + r"))(?![\w-]|\.[A-Za-z0-9])",
    re.IGNORECASE,
)
_PROMPT_RE = re.compile(r"^\s*(?:[$%>]\s+)?(?:sudo\s+)?([a-z][a-z0-9._+-]*)(?:\s+(\S+))?")
_SUBCOMMAND_RE = re.compile(r"^[a-z][a-z0-9-]{1,24}$")
_BACKTICK_RE = re.compile(r"`([^`\n]{1,120})`")
_SNAKE_RE = re.compile(r"(?<![\w-])([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)(?![\w-]|\.\w)")
_CAMEL_RE = re.compile(r"(?<![\w-])([a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+)(?![\w-])")
_CALL_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\(")
_ONE_WORD_CODE_RE = re.compile(r"^[A-Za-z_][\w.:-]{2,59}$")
_WORD_RE = re.compile(r"[^\W\d_]+")
_PREFIX_RE = re.compile(
    r"(?<![\w:-])([A-Za-z][A-Za-z0-9]{1,15})([-_:])((?:[A-Za-z0-9]+[-_:])*[A-Za-z0-9]+)(?![\w-]|\.\w|:\w)"
)
_MAILBOX_RE = re.compile(
    r"(?<![\w-])([\w-]*(?:inbox|outbox|mailbox|postbox|letterbox|dead[-_]?drop|postfach|"
    r"briefkasten|message[-_]?board|notice[-_]?board|mailslot)[\w-]*)(?![\w-])",
    re.IGNORECASE,
)
_CAPS_RE = re.compile(r"(?<![\w-])([A-ZÄÖÜ]{2,12})(?![\w-])")
_LINE_START_RE = re.compile(r"^\W{0,3}$")
_SIGNAL_AFTER_RE = re.compile(r"^\s*(?:[:!\]]|$)")


# ---------------------------------------------------------------- extraction


def _norm_host(raw):
    host = raw.rsplit("@", 1)[-1].split(":", 1)[0].strip(".").casefold()
    if host.startswith("www."):
        host = host[4:]
    return host if _HOST_OK_RE.match(host) else None


def _common_host(host):
    if host in EXACT_ONLY_HOSTS:
        return True
    return any(host == h or host.endswith("." + h) for h in COMMON_HOSTS)


def _command(span):
    m = _PROMPT_RE.match(span)
    if not m or m.group(1) not in COMMAND_VERBS:
        return None
    verb, arg = m.group(1), m.group(2)
    if arg is None:
        return None  # a bare word is prose, not a command
    if _SUBCOMMAND_RE.match(arg) and arg not in STOPWORDS:
        return verb + " " + arg
    return verb


def _is_content(word):
    return len(word) >= 3 and word not in STOPWORDS and word not in WEB_WORDS


def line_tokens(line):
    """Tokens on one line, as (key, cls, kind, value), stop lists applied.

    Returns (tokens, stoplisted) where stoplisted counts per class the tokens
    a stop list removed.
    """
    found = []
    stopped = Counter()

    def add(cls, kind, value):
        key = cls + ":" + value if cls != "convention" else "convention:" + kind + ":" + value
        found.append((key, cls, kind, value))

    # hosts
    for m in _URL_HOST_RE.finditer(line):
        host = _norm_host(m.group(1))
        if host:
            if _common_host(host):
                stopped["host"] += 1
            else:
                add("host", "host", host)
    if "." in line:
        for m in _BARE_HOST_RE.finditer(line):
            host = _norm_host(m.group(1))
            if host:
                if _common_host(host):
                    stopped["host"] += 1
                else:
                    add("host", "host", host)
        # files
        for m in _FILE_RE.finditer(line):
            name = m.group(1).casefold()
            if name in COMMON_FILES:
                stopped["file"] += 1
            else:
                add("file", "file", name)

    # commands: the line itself, then backtick spans
    spans = [line] + [m.group(1) for m in _BACKTICK_RE.finditer(line)]
    for span in spans:
        cmd = _command(span)
        if cmd:
            add("command", "command", cmd)

    # mailbox names first, so code words skip them
    mailbox = set()
    for m in _MAILBOX_RE.finditer(line):
        name = m.group(1).casefold()
        if 3 <= len(name) <= 48:
            mailbox.add(name)
            add("convention", "mailbox", name)

    # code words
    def code(word):
        if word.casefold() in mailbox:
            return
        if word.casefold() in CODE_STOP or "http" in word.casefold():
            stopped["code"] += 1
            return
        add("code", "code", word)

    for m in _SNAKE_RE.finditer(line):
        if len(m.group(1)) >= 5:
            code(m.group(1))
    for m in _CAMEL_RE.finditer(line):
        word = m.group(1)
        if len(word) >= 6 and sum(ch.islower() for ch in word) >= 2:
            code(word)
    for m in _CALL_RE.finditer(line):
        if len(m.group(1)) >= 5:
            code(m.group(1))
    for m in _BACKTICK_RE.finditer(line):
        span = m.group(1).strip()
        if _ONE_WORD_CODE_RE.match(span) and not _command(span):
            code(span)

    # naming prefixes
    for m in _PREFIX_RE.finditer(line):
        head, sep, rest = m.group(1), m.group(2), m.group(3)
        if sep == "-" and not any(ch.isdigit() for ch in rest):
            continue
        if head.casefold() in PREFIX_STOP or head.casefold() in STOPWORDS:
            stopped["convention"] += 1
            continue
        add("convention", "prefix", head.casefold() + sep)

    # all-caps control words
    for m in _CAPS_RE.finditer(line):
        word = m.group(1)
        if word in CAPS_STOP or word.casefold() in STOPWORDS:
            stopped["convention"] += 1
            continue
        before, after = line[:m.start()], line[m.end():]
        signal = (
            word in CONTROL_WORDS
            or len(word) >= 5
            or (len(word) >= 3 and (
                _LINE_START_RE.match(before)
                or (before.endswith("[") and after.startswith("]"))
                or _SIGNAL_AFTER_RE.match(after)
            ))
        )
        if signal:
            add("convention", "control", word)

    # rare 3-word phrases, never across or inside URLs, host or file names
    for segment in _phrase_segments(line):
        words = [w.casefold() for w in _WORD_RE.findall(segment)]
        for i in range(len(words) - 2):
            tri = words[i:i + 3]
            if sum(1 for w in tri if _is_content(w)) >= 2:
                add("phrase", "phrase", " ".join(tri))
            else:
                stopped["phrase"] += 1
    return found, stopped


_URL_SPAN_RE = re.compile(r"\b(?:https?|ftp)://\S+", re.IGNORECASE)


def _phrase_segments(line):
    """Parts of a line outside URL, bare host and file-name spans."""
    spans = [m.span() for m in _URL_SPAN_RE.finditer(line)]
    if "." in line:
        spans += [m.span(1) for m in _BARE_HOST_RE.finditer(line)]
        spans += [m.span(1) for m in _FILE_RE.finditer(line)]
    if not spans:
        return [line]
    spans.sort()
    parts, pos = [], 0
    for start, end in spans:
        if start > pos:
            parts.append(line[pos:start])
        pos = max(pos, end)
    parts.append(line[pos:])
    return parts


def risky_types(text):
    """Risky-word types found in text (type names only, in RISKY_TYPES order)."""
    return tuple(name for name, rx in _RISKY_RES if rx.search(text))


_PCT_RE = re.compile(r"%[0-9A-Fa-f]{2}")


def decode_line(line):
    """Undo percent-escapes (two rounds, for double encoding) and re-mask IPs.

    ``%2Fwww.host.org`` would otherwise give the host ``2fwww.host.org``.
    Decoding can reveal an IP-shaped value, so the loader's mask runs again.
    """
    for _ in range(2):
        if not _PCT_RE.search(line):
            break
        line = unquote(line, errors="replace")
    line = line.replace("\r", " ").replace("\n", " ")
    return scrub_ips(line)[0]


def row_tokens(text):
    """Tokens in one row's text: ({key: (cls, kind, value, risky types)}, stoplisted)."""
    out = {}
    stopped = Counter()
    for line in text.split("\n"):
        if not line.strip():
            continue
        if "%" in line:
            line = decode_line(line)
        found, stop = line_tokens(line)
        stopped.update(stop)
        if not found:
            continue
        risky = risky_types(line)
        for key, cls, kind, value in found:
            if key in out:
                if risky:
                    prev = out[key]
                    out[key] = (cls, kind, value, prev[3] | frozenset(risky))
            else:
                out[key] = (cls, kind, value, frozenset(risky))
    return out, stopped


def given_keys(texts):
    """Token keys found in shared goal or prompt texts (dropped as handed down)."""
    keys = set()
    for text in texts or ():
        keys.update(row_tokens(text)[0])
    return keys


# ---------------------------------------------------------------- clock rule


def in_clock_order(t_a, u_a, t_b, u_b):
    """True when use B is later than use A by more than both clocks' uncertainty."""
    return t_b - t_a > u_a + u_b


def clock_resolved(first_uses):
    """first_uses: [(time, uncertainty), ...] sorted by time. True if any pair is in order."""
    for i in range(len(first_uses)):
        for j in range(i + 1, len(first_uses)):
            (t_a, u_a), (t_b, u_b) = first_uses[i], first_uses[j]
            if in_clock_order(t_a, u_a, t_b, u_b):
                return True
    return False


# ---------------------------------------------------------------- main finder


def _agents_bucket(n):
    if n < 5:
        return "3-4"
    if n < 10:
        return "5-9"
    if n < 50:
        return "10-49"
    return "50+"


def find_tokens(rows, min_agents=MIN_AGENTS, given_texts=(), max_phrase_share=MAX_PHRASE_SHARE):
    """Find candidate tricks in time-sorted rows.

    Returns (tokens, stats): tokens maps key -> Token, sorted by key; stats
    holds counts only.
    """
    started = time.time()
    seen_rows = {}  # key -> list of row indices
    meta = {}  # key -> (cls, kind, value)
    risky = {}  # key -> set of risky types
    agents_of = {}  # key -> {agent: first row index}
    stopped = Counter()
    agent_rows = 0
    rows_risky = Counter()
    for idx, row in enumerate(rows):
        is_agent = not row.exposure_only
        agent_rows += is_agent
        if not row.text:
            continue
        found, stop = row_tokens(row.text)
        stopped.update(stop)
        if is_agent:
            for name in risky_types(row.text):
                rows_risky[name] += 1
        for key, (cls, kind, value, rk) in found.items():
            if key not in meta:
                meta[key] = (cls, kind, value)
                seen_rows[key] = []
                risky[key] = set()
                agents_of[key] = {}
            seen_rows[key].append(idx)
            if rk:
                risky[key].update(rk)
            if is_agent and row.agent not in agents_of[key]:
                agents_of[key][row.agent] = idx

    given = given_keys(given_texts)
    phrase_cap = max(PHRASE_CAP_FLOOR, max_phrase_share * agent_rows)
    per_class = {c: Counter() for c in CLASSES}
    per_kind = Counter()
    tokens = {}
    for key in sorted(meta):
        cls, kind, value = meta[key]
        counts = per_class[cls]
        counts["distinct_seen"] += 1
        if key in given:
            counts["given_dropped"] += 1
            continue
        users = agents_of[key]
        if cls == "phrase":
            agent_row_count = sum(1 for i in seen_rows[key] if not rows[i].exposure_only)
            if agent_row_count > phrase_cap:
                counts["common_dropped"] += 1
                continue
        if len(users) < min_agents:
            counts["under_min_agents"] += 1
            continue
        firsts = sorted(users.items(), key=lambda kv: (rows[kv[1]].time, kv[1]))[:FIRST_USERS]
        uses = [(rows[i].time, rows[i].uncertainty) for _a, i in firsts]
        if not clock_resolved(uses):
            counts["clock_dropped"] += 1
            continue
        counts["kept"] += 1
        if cls == "convention":
            per_kind[kind] += 1
        tokens[key] = Token(
            key=key,
            cls=cls,
            kind=kind,
            value=value,
            rows=tuple(seen_rows[key]),
            agents=len(users),
            first_users=tuple(firsts),
            risky=tuple(t for t in RISKY_TYPES if t in risky[key]),
        )

    by_bucket = Counter(_agents_bucket(t.agents) for t in tokens.values())
    risky_tokens = Counter(r for t in tokens.values() for r in t.risky)
    stats = {
        "rows": len(rows),
        "agent_rows": agent_rows,
        "agents": len(set(r.agent for r in rows if not r.exposure_only)),
        "params": {
            "min_agents": min_agents,
            "first_users_for_clock_rule": FIRST_USERS,
            "max_phrase_share": max_phrase_share,
            "phrase_cap_floor": PHRASE_CAP_FLOOR,
            "phrase_cap_agent_rows": round(phrase_cap, 2),
            "given_texts": len(given_texts or ()),
        },
        "classes": {
            c: {k: per_class[c][k] for k in ("distinct_seen", "given_dropped", "common_dropped",
                                             "under_min_agents", "clock_dropped", "kept")}
            for c in CLASSES
        },
        "stoplisted_occurrences": {c: stopped[c] for c in CLASSES},
        "convention_kinds_kept": {k: per_kind[k] for k in CONVENTION_KINDS},
        "kept_total": len(tokens),
        "kept_by_agent_count": {b: by_bucket[b] for b in ("3-4", "5-9", "10-49", "50+")},
        "kept_tokens_with_risky_type": {t: risky_tokens[t] for t in RISKY_TYPES},
        "agent_rows_with_risky_type": {t: rows_risky[t] for t in RISKY_TYPES},
        "seconds": round(time.time() - started, 1),
    }
    return tokens, stats


# ---------------------------------------------------------------- command line


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="Find candidate tricks; print counts only (no text, no names).")
    ap.add_argument("--adapter", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--input", required=True, help="export folder (wiki) or JSON Lines file (generic)")
    ap.add_argument("--summary", help="write the counts as JSON here (keep it outside the repo)")
    ap.add_argument("--min-agents", type=int, default=MIN_AGENTS)
    ap.add_argument("--max-phrase-share", type=float, default=MAX_PHRASE_SHARE)
    ap.add_argument("--given", action="append", default=[],
                    help="a shared goal or prompt text file; its tokens are dropped (repeatable)")
    args = ap.parse_args(argv)
    try:
        guard_out(args.summary, [args.input] + list(args.given), "--summary")
        rows, load_stats = load(args.adapter, args.input)
        given = []
        for path in args.given:
            with open(path, "r", encoding="utf-8") as fh:
                given.append(fh.read())
    except (OSError, ValueError) as exc:
        print("tokens: ERROR: %s" % exc, file=out)
        return 2
    _tokens, stats = find_tokens(rows, min_agents=args.min_agents, given_texts=given,
                                 max_phrase_share=args.max_phrase_share)
    summary = {"job": "J3 tokens", "adapter": args.adapter, "load": {
        k: load_stats[k] for k in ("rows", "agents", "group_count", "time_span") if k in load_stats
    }}
    summary.update(stats)
    text = json.dumps(summary, indent=1, ensure_ascii=True)
    if args.summary:
        folder = os.path.dirname(os.path.abspath(args.summary))
        os.makedirs(folder, exist_ok=True)
        with open(args.summary, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
    out.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
