"""Can it see, did it see: one test, one null, one axis (docs/can-it-see.svg).

Python standard library only. Deterministic: the same two input files give
the same bytes. Draws one public figure from two counts-only files a run
leaves outside the repo:

``calibration.json`` (schema ``pz-calibration/0``, from the power panel)
    Planted kinds scored on the real background under one null, plus the
    real data split into cross-page and same-page links under that null.
``checks.json`` (from ``python -m patient_zero.checks``)
    The order null on the real data (``order_null.links_named``).

Every row sits on one shared axis, observed ÷ null mean, with a grey band
from the null's 5th to 95th percentile (each divided by the same mean) and one
accent dot. Rows:

* Can it see? (planted on the real wiki)
    - planted carry: seen on one page, used later on another;
    - planted same-page spread (E6): the blind spot, shown on purpose;
    - page-instruction decoy: drawn as "not run yet" until calibration.json
      holds ``planted.page_instruction``.
* Did it see? (the real wiki)
    - all links (from checks.json), and under it the same no-spread control on
      all links (``controls.random_400.all_links``);
    - cross-page links (from calibration.json);
    - the no-spread control for that row: random rows scored the same way
      (``controls.random_400.cross_page``), drawn as "not run yet" until
      calibration.json holds it. Inside the band reads "unbiased"; outside it
      reads "biased", and then the real row beside it is not evidence either way.

Every row must use the same null, or nothing is drawn: the null id built
from each file's null fields must match, calibration.json's ``null.h0``
(the real all-links count, mean and sd under its null) must equal
checks.json's ``order_null.links_named``, so both files used the same
shuffles, and the real data's same-page and cross-page counts and null means
must add up to ``null.h0``, so the cross-page row used those shuffles too.
There is no z on the figure and no second axis. The footer prints the null's
tag (the first 12 hex of the SHA-256 of its id), its seed and shuffle count,
and the SHA-256 of both input files. ``--skip-pending`` leaves out a row whose
numbers do not exist yet.

The row labels and section titles say "real wiki", which is right for the
published figure and for no other log. ``--subject "this log"`` (or any short
plain name) puts that name there instead; ``tools/can_it_see.py`` always passes
it, so "real wiki" is never printed on a log that is not the wiki export. The
default leaves every byte of the published figure as it is.

Visible words were checked against the wiki export with
``tools/labelscan.py``: a few plain English words that some agent also used as
its label or as a control word are avoided (the scan finds 0), so the figure
says "pairs" for what the rest of the repo calls named links, and the footer
prints a hashed tag instead of the readable null id.

Each row's group in the SVG carries its source file, JSON path, null id and
the raw numbers (``data-*``), so ``tools/figcheck.py --sources`` can check
every number, position and word against the JSON without trusting this module.

calibration.json contract (``pz-calibration/0``)::

    {"schema": "pz-calibration/0",
     "null": {"kind": "order", "scope": "agent_day", "shuffles": 100, "seed": 20261003,
              "min_lag_seconds": 0, "strict_exposure": false,
              "h0": {"observed": ..., "mean": ..., "sd": ...}},
     "planted": {"<kind>": {"contact": true|false, "tokens": 40,
                            "all_links": {"observed", "mean", "sd", "p05", "p95", ...}}},
     "real": {"cross_page": {...}, "same_page": {...}, "all_links": {...}}}

Usage::

    python -m patient_zero.canisee                       # ../out/*.json -> docs/can-it-see.svg
    python -m patient_zero.canisee --calibration PATH --checks PATH --svg docs/can-it-see.svg
"""

import argparse
import hashlib
import json
import math
import os
import sys
from collections import namedtuple

from patient_zero.report import AXIS, FONT, INK, INK2, MUTED, SERIES, SURFACE, _c, _e

SCHEMA = "pz-calibration/0"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(REPO), "out")

W = 600
X0, X1 = 24.0, 576.0
PHONE_WIDTH = 358  # px: a small phone's content width (tools/figcheck.py wants text of 10 px or more there)
SIZE_TITLE, SIZE_SECTION, SIZE_LABEL, SIZE_SMALL = 26, 19, 18, 17
ACCENT = SERIES[0]
BAND_FILL = "#d6d4cc"  # grey, darker than the grid marks so a narrow band still shows
TRACK_GRID = "#ecebe6"
DOT_R = 6.0
BAND_H = 20  # taller than the dot, so a band narrower than the dot shows above and below it
MIN_BAND = 2.0  # px: a band narrower than this is drawn 2 px wide, centred
ROW_H = 82

Row = namedtuple("Row", "key section source path label required")
ROWS = (
    Row("carry", "can", "calibration.json", ("planted", "carry", "all_links"),
        "planted: carried to another page", True),
    Row("same_page", "can", "calibration.json", ("planted", "contagion", "all_links"),
        "planted: spread on one page", True),
    Row("page_instruction", "can", "calibration.json", ("planted", "page_instruction", "all_links"),
        "decoy: the page tells every writer", False),
    Row("real_all", "did", "checks.json", ("order_null", "links_named"), "real wiki: all pairs", True),
    Row("random_all", "did", "calibration.json", ("controls", "random_400", "all_links"),
        "control: random rows, all pairs", False),
    Row("real_cross", "did", "calibration.json", ("real", "cross_page"), "real wiki: cross-page pairs", True),
    Row("random_cross", "did", "calibration.json", ("controls", "random_400", "cross_page"),
        "control: random rows, cross-page", False),
)
SECTIONS = (("can", "Can it see? Planted on the real wiki"), ("did", "Did it see? The real wiki"))
SUBJECT = "real wiki"  # what the labels above call the log: right for the published figure and no other log
SUBJECT_MAX = 40
WORDS = {
    "contact": {"above": "seen", "inside": "blind spot", "below": "reads the wrong way"},
    "decoy": {"above": "wrongly flagged", "inside": "not flagged", "below": "not flagged"},
    "control": {"above": "biased high", "inside": "unbiased", "below": "biased low"},
    "real": {},
}
CELL_KEYS = ("observed", "mean", "p05", "p95")
SPLIT_TOL = 0.011  # two means rounded to 0.01 can miss their rounded sum by 0.01


class CanISeeError(ValueError):
    """The figure cannot be drawn honestly from these inputs."""


def null_id(block, kind=None):
    """Compact id of a null from its fields; equal ids mean the same null settings."""
    kind = kind or block.get("kind")
    for name in ("scope", "shuffles", "seed", "min_lag_seconds", "strict_exposure"):
        if name not in block:
            raise CanISeeError("null block has no %r" % name)
    return "%s:%s:lag%s:%s:seed%s:n%s" % (kind, block["scope"], block["min_lag_seconds"],
                                          "strict" if block["strict_exposure"] else "default",
                                          block["seed"], block["shuffles"])


def null_tag(ident):
    """What the footer prints for a null id: the first 12 hex of its SHA-256."""
    return hashlib.sha256(ident.encode("utf-8")).hexdigest()[:12]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _get(doc, path):
    for part in path:
        if not isinstance(doc, dict) or part not in doc:
            return None
        doc = doc[part]
    return doc


def where(observed, p05, p95):
    """'above', 'inside' or 'below' the null's 5th to 95th percentile band."""
    if observed > p95:
        return "above"
    if observed < p05:
        return "below"
    return "inside"


def role_of(row, calibration):
    if row.path[0] == "controls":  # tokens with no structure, scored like the real data beside them
        kind = _get(calibration, row.path[:2]) or {}
        if kind.get("contact") is not False:
            raise CanISeeError("calibration.json %s must say contact: false" % ".".join(row.path[:2]))
        return "control"
    if row.section == "did":
        return "real"
    kind = _get(calibration, row.path[:2]) or {}
    if "contact" not in kind:
        raise CanISeeError("calibration.json %s has no 'contact' flag" % ".".join(row.path[:2]))
    return "contact" if kind["contact"] else "decoy"


def check_subject(subject):
    """A short plain name for the log (letters, digits, spaces, hyphens). Raises CanISeeError."""
    ok = isinstance(subject, str) and 0 < len(subject) <= SUBJECT_MAX and subject == subject.strip()
    if not ok or any(not (ch.isalnum() or ch in " -") for ch in subject):
        raise CanISeeError("--subject must be 1 to %d letters, digits, spaces or hyphens" % SUBJECT_MAX)
    return subject


def label_for(row, subject=SUBJECT):
    """The row's label with the log called ``subject``; the default leaves every label as it is."""
    if subject == SUBJECT or not row.label.startswith(SUBJECT + ": "):
        return row.label
    return subject + row.label[len(SUBJECT):]


def sections_for(subject=SUBJECT):
    """The two section titles with the log called ``subject``; the default leaves them as they are."""
    if subject == SUBJECT:
        return SECTIONS
    return (("can", "Can it see? Planted on %s" % subject),
            ("did", "Did it see? %s%s" % (subject[:1].upper(), subject[1:])))


def verdict(role, position):
    word = WORDS[role].get(position)
    return "%s the band%s" % (position, (": " + word) if word else "")


def build_rows(calibration, checks, subject=SUBJECT):
    """The rows to draw, after the same-null checks. Raises CanISeeError.

    ``subject`` names the log on the two rows and the section titles that say "real wiki" by default.
    """
    check_subject(subject)
    if calibration.get("schema") != SCHEMA:
        raise CanISeeError("calibration.json schema is %r, not %r" % (calibration.get("schema"), SCHEMA))
    cal_null = calibration.get("null") or {}
    order = checks.get("order_null")
    if not order:
        raise CanISeeError("checks.json has no order_null (run python -m patient_zero.checks)")
    ident = null_id(cal_null)
    if null_id(order, kind="order") != ident:
        raise CanISeeError("calibration.json and checks.json used different nulls: %s vs %s"
                           % (ident, null_id(order, kind="order")))
    h0, real_all = cal_null.get("h0") or {}, order.get("links_named") or {}
    if any(h0.get(k) != real_all.get(k) for k in ("observed", "mean", "sd")):
        raise CanISeeError("same null settings but different shuffles: calibration null.h0 does not equal "
                           "checks.json order_null.links_named")
    split = [(calibration.get("real") or {}).get(k) for k in ("same_page", "cross_page")]
    if all(split) and (split[0]["observed"] + split[1]["observed"] != h0["observed"]
                       or abs(split[0]["mean"] + split[1]["mean"] - h0["mean"]) > SPLIT_TOL):
        raise CanISeeError("calibration real.same_page + real.cross_page do not add up to null.h0: the split did "
                           "not use the same shuffles")
    docs = {"calibration.json": calibration, "checks.json": checks}
    rows = []
    for row in ROWS:
        cell = _get(docs[row.source], row.path)
        entry = {"row": row, "null": ident, "role": role_of(row, calibration) if cell else None,
                 "label": label_for(row, subject)}
        if cell is None:
            if row.required:
                raise CanISeeError("%s has no %s" % (row.source, ".".join(row.path)))
            entry["status"] = "pending"
            rows.append(entry)
            continue
        missing = [k for k in CELL_KEYS if not isinstance(cell.get(k), (int, float))]
        if missing or cell["mean"] <= 0:
            raise CanISeeError("%s %s needs observed, mean > 0, p05 and p95" % (row.source, ".".join(row.path)))
        obs, mean, p05, p95 = (cell[k] for k in CELL_KEYS)
        entry.update(status="drawn", cell={k: cell[k] for k in CELL_KEYS}, ratio=obs / mean,
                     lo=p05 / mean, hi=p95 / mean, position=where(obs, p05, p95))
        entry["verdict"] = verdict(entry["role"], entry["position"])
        rows.append(entry)
    return rows


def axis(rows):
    """(lo, hi, label step, grid step) in tenths-safe floats, covering every dot, band and 1."""
    values = [1.0]
    for r in rows:
        if r["status"] == "drawn":
            values += [r["ratio"], r["lo"], r["hi"]]
    lo_t = int(math.floor((min(values) - 0.02) * 10))
    hi_t = int(math.ceil((max(values) + 0.02) * 10))
    span = hi_t - lo_t
    label_t = 1 if span <= 6 else (2 if span <= 12 else 5)
    if label_t > 1:  # labels on multiples of the step, so 1 always carries a label
        lo_t -= lo_t % label_t
        hi_t += (-hi_t) % label_t
    grid_t = 1 if (hi_t - lo_t) <= 12 else label_t
    return lo_t, hi_t, label_t, grid_t


def _fmt(x):
    return "{:,}".format(x)


def _tick(t):
    return ("%.1f" % (t / 10.0)).rstrip("0").rstrip(".")


def _text(x, y, s, size, fill, anchor="start", weight=None, cls=None, ident=None):
    attrs = ['x="%s"' % _c(x), 'y="%s"' % _c(y), 'font-size="%s"' % _c(size), 'fill="%s"' % fill]
    if anchor != "start":
        attrs.append('text-anchor="%s"' % anchor)
    if weight:
        attrs.append('font-weight="%s"' % weight)
    if cls:
        attrs.append('class="%s"' % cls)
    if ident:
        attrs.append('id="%s"' % ident)
    return "<text %s>%s</text>" % (" ".join(attrs), _e(s))


def draw(calibration, checks, hashes, skip_pending=False, subject=SUBJECT):
    """The figure as an SVG string. hashes: {"calibration.json": sha256, "checks.json": sha256}.

    skip_pending leaves out rows whose numbers do not exist yet instead of drawing them as "not run yet".
    subject names the log (default "real wiki", the published figure; any other log needs its own name).
    """
    rows = [r for r in build_rows(calibration, checks, subject)
            if not (skip_pending and r["status"] == "pending")]
    lo_t, hi_t, label_t, grid_t = axis(rows)
    lo, hi = lo_t / 10.0, hi_t / 10.0

    def X(v):
        return X0 + (v - lo) / (hi - lo) * (X1 - X0)

    seed, shuffles = calibration["null"]["seed"], calibration["null"]["shuffles"]
    title = "Can it see? Did it see?"
    parts = [_text(24, 42, title, SIZE_TITLE, INK, weight="600", cls="cis-title", ident="cis-title"),
             _text(24, 72, "Dot: pairs (earlier → later user) ÷ order-null mean.", SIZE_SMALL, INK2, cls="cis-sub"),
             _text(24, 96, "Grey band: 5th to 95th percentile of %d shuffles." % shuffles, SIZE_SMALL, INK2,
                   cls="cis-sub")]
    y = 112
    for section, heading in sections_for(subject):
        y += 34
        parts.append(_text(24, y, heading, SIZE_SECTION, INK, weight="600", cls="cis-section"))
        y += 8
        for r in (r for r in rows if r["row"].section == section):
            parts.append(_row(r, y, X, lo_t, hi_t, grid_t))
            y += ROW_H
    y += 6
    ticks = []
    t = lo_t
    while t <= hi_t:
        if (t - lo_t) % label_t == 0:
            ticks.append('<line class="cis-tick" data-v="%s" x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" '
                         'stroke-width="1"/>' % (_tick(t), _c(X(t / 10.0)), _c(y), _c(X(t / 10.0)), _c(y + 6), AXIS))
            ticks.append(_text(X(t / 10.0), y + 26, _tick(t), SIZE_SMALL, MUTED, anchor="middle",
                               cls="cis-tick-label"))
        t += 1
    parts.append('<g class="cis-axis" data-lo="%s" data-hi="%s" data-x0="%s" data-x1="%s">'
                 '<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1"/>%s</g>'
                 % (_tick(lo_t), _tick(hi_t), _c(X0), _c(X1), _c(X0), _c(y), _c(X1), _c(y), AXIS, "".join(ticks)))
    parts.append(_text((X0 + X1) / 2.0, y + 52, "pairs ÷ order-null mean", SIZE_SMALL, INK2, anchor="middle",
                       cls="cis-x-label", ident="cis-x-label"))
    y += 88
    ident = rows[0]["null"]
    parts.append(_text(24, y, "null id %s: seed %s, %s shuffles" % (null_tag(ident), seed, shuffles), SIZE_SMALL,
                       MUTED, cls="cis-null-id"))
    for name in ("calibration.json", "checks.json"):
        y += 24
        parts.append(_text(24, y, "%s sha256 %s" % (name, hashes[name][:16]), SIZE_SMALL, MUTED, cls="cis-hash"))
    height = int(math.ceil(y + 20))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" role="img" '
            'aria-labelledby="cis-title-el" font-family="%s"><title id="cis-title-el">%s</title>'
            '<rect width="%d" height="%d" fill="%s"/>%s</svg>\n'
            % (W, height, W, height, FONT, _e(title), W, height, SURFACE, "".join(parts)))


def _row(r, top, X, lo_t, hi_t, grid_t):
    row = r["row"]
    attrs = ['class="cis-row"', 'id="cis-row-%s"' % row.key, 'data-row="%s"' % row.key,
             'data-source="%s"' % row.source, 'data-path="%s"' % ".".join(row.path), 'data-null="%s"' % r["null"],
             'data-status="%s"' % r["status"]]
    if r["status"] == "drawn":
        attrs.append('data-role="%s"' % r["role"])
        attrs += ['data-%s="%s"' % (k, _e(repr(r["cell"][k]))) for k in CELL_KEYS]
    centre = top + 40
    out = [_text(24, top + 18, r["label"], SIZE_LABEL, INK, weight="600", cls="cis-label")]
    t = lo_t
    while t <= hi_t:  # short grid marks inside the track, on the shared axis
        if (t - lo_t) % grid_t == 0:
            out.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="%s"/>'
                       % (_c(X(t / 10.0)), _c(centre - 14), _c(X(t / 10.0)), _c(centre + 14),
                          INK2 if t == 10 else TRACK_GRID, "1.5" if t == 10 else "1"))
        t += 1
    if r["status"] == "pending":
        out.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1.5" stroke-dasharray="4 5"/>'
                   % (_c(X0), _c(centre), _c(X1), _c(centre), MUTED))
        out.append(_text(24, top + 70, "not run yet", SIZE_SMALL, MUTED, cls="cis-where"))
    else:
        cell = r["cell"]
        left, right = X(r["lo"]), X(r["hi"])
        if right - left < MIN_BAND:
            mid = (left + right) / 2.0
            left, right = mid - MIN_BAND / 2.0, mid + MIN_BAND / 2.0
        out.append('<rect class="cis-band" x="%s" y="%s" width="%s" height="%d" rx="3" fill="%s"/>'
                   % (_c(left), _c(centre - BAND_H / 2.0), _c(right - left), BAND_H, BAND_FILL))
        out.append('<circle class="cis-dot" cx="%s" cy="%s" r="%s" fill="%s" stroke="%s" stroke-width="2"/>'
                   % (_c(X(r["ratio"])), _c(centre), _c(DOT_R), ACCENT, SURFACE))
        out.append(_text(X1, top + 18, "%s vs %s" % (_fmt(cell["observed"]), _fmt(cell["mean"])), SIZE_SMALL, INK2,
                         anchor="end", cls="cis-value"))
        out.append(_text(24, top + 70, "%.3f · %s" % (r["ratio"], r["verdict"]), SIZE_SMALL, INK2, cls="cis-where"))
    return "<g %s>%s</g>" % (" ".join(attrs), "".join(out))


def phone_px(svg_width, smallest_font):
    """Smallest text in px once a figure of svg_width is scaled to a phone's width."""
    return smallest_font * PHONE_WIDTH / float(svg_width)


def summary(rows):
    lines = []
    for r in rows:
        if r["status"] == "pending":
            lines.append("%s: not run yet" % r["label"])
        else:
            lines.append("%s: %s vs %s, %.3f, %s" % (r["label"], _fmt(r["cell"]["observed"]),
                                                    _fmt(r["cell"]["mean"]), r["ratio"], r["verdict"]))
    return lines


def _write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(prog="python -m patient_zero.canisee",
                                 description="Draw docs/can-it-see.svg: one null, one axis, can it see / did it see.")
    ap.add_argument("--calibration", default=os.path.join(OUT, "calibration.json"))
    ap.add_argument("--checks", default=os.path.join(OUT, "checks.json"))
    ap.add_argument("--svg", default=os.path.join(REPO, "docs", "can-it-see.svg"))
    ap.add_argument("--skip-pending", action="store_true",
                    help="leave out rows with no numbers yet (default: draw them as 'not run yet')")
    ap.add_argument("--subject", default=SUBJECT,
                    help="what the labels call the log (default '%s', right only for the German wiki export; "
                         "for any other log give its own name, such as 'this log')" % SUBJECT)
    args = ap.parse_args(argv)
    try:
        if not args.svg.lower().endswith(".svg"):
            raise CanISeeError("--svg must name a .svg file")
        check_subject(args.subject)
        with open(args.calibration, encoding="utf-8") as fh:
            calibration = json.load(fh)
        with open(args.checks, encoding="utf-8") as fh:
            checks = json.load(fh)
        hashes = {"calibration.json": sha256_file(args.calibration), "checks.json": sha256_file(args.checks)}
        svg = draw(calibration, checks, hashes, skip_pending=args.skip_pending, subject=args.subject)
        rows = [r for r in build_rows(calibration, checks, args.subject)
                if not (args.skip_pending and r["status"] == "pending")]
        os.makedirs(os.path.dirname(os.path.abspath(args.svg)), exist_ok=True)
        _write_atomic(args.svg, svg)
    except (OSError, ValueError) as exc:
        print("patient-zero canisee: ERROR: %s" % exc, file=out)
        return 2
    for line in summary(rows):
        print(line, file=out)
    print("null id: %s (every row)" % rows[0]["null"], file=out)
    print("wrote: %s" % args.svg, file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
