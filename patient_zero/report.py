"""Write the offline report (report.html) and the public lead figure (lead.svg).

Python standard library only. Two outputs:

``report.html`` (in the run's output folder)
    One static page: inline SVG and inline CSS, no scripts, no external
    files, so it opens offline. Sections: key numbers; the verdict against
    the label shuffle in plain words (``chance_verdict``); the lead spread tree
    (time left to right, coloured by page family, with a legend); the largest
    tree for scale; spread speed; the agent influence table; observed against
    the label shuffle; parallel starts; page-family flow; wikis; risky-word
    types against the shuffle; the top 20 trees; method and limits. Every
    chart has a table view. Agent labels and token keys appear escaped and cut
    to 80 characters (``safe.short``), then HTML-escaped. It is a local
    working view, not a repo file.

``lead.svg`` (only with ``--svg``, normally ``docs/lead.svg``)
    The lead tree alone, summary data only: no agent labels and no token text
    (the token is named by its class and token_id), page families, and minutes
    since the first use, plus a line saying how the tree was picked and how
    many trees chance alone would let through (``selection_lines``), so the
    figure stays honest when shown on its own. Meant to be committed and shown
    in the README.

Both are built from three files a run writes to its output folder:
facts.json, trees.jsonl and top_trees.json. top_trees.json holds node-level
detail for a few trees (the lead, the largest, and the top 20 by largest
branch overall and among trees that beat every shuffle); ``detail_set``
builds it and ``pipeline.write_outputs`` writes it. ``cli.py`` writes
report.html at the end of every run. This module's command line re-renders
the page, writes the public figure or switches the lead tree, without
re-running::

    python -m patient_zero.report --out PATH/TO/out
    python -m patient_zero.report --out PATH/TO/out --svg docs/lead.svg
    python -m patient_zero.report --out PATH/TO/out --lead TOKEN_ID

Lead tree rule (default): the largest branch among trees whose named links
beat every shuffle; with no shuffle, or no such tree, the largest branch among
trees with named links. Ties go to more named links, then the token_id.
``--lead TOKEN_ID`` picks another tree; ``cli.py --lead TOKEN_ID`` also adds a
tree that is not yet in top_trees.json.

Chart colours: the first three categorical slots of the dataviz reference
palette (blue, orange, aqua; validated for all pairs on the light surface),
then one grey for "other". Nothing here opens a URL or runs a command from
the data.
"""

import argparse
import hashlib
import html
import json
import math
import os
import sys
from collections import Counter, defaultdict

from patient_zero.safe import NO_FAMILY, short, token_id
from patient_zero.score import branch_sizes

DETAIL_VERSION = 1
DETAIL_TOP = 20
MAX_DRAW = 160
TOP_TREES_TABLE = 20

FONT = "system-ui, -apple-system, Segoe UI, sans-serif"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WASH = "#e1e0d9"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
OTHER = "#b4b2aa"

RULE_GIVEN = "given with --lead"
RULE_BEAT = "largest branch among trees whose named links beat every shuffle"
RULE_LARGEST = "largest branch among trees with named links (no tree beat every shuffle, or no shuffle ran)"
RULE_NONE = "no tree has a named link"

STATUS_NAMES = {
    "first": "first user",
    "linked": "linked: credited to a named earlier user",
    "parallel": "parallel start: no earlier use seen",
    "unnamed": "source unnamed (blank or human row)",
    "clock_dropped": "dropped by the clock rule",
}
BUCKETS = (("le_10_min", "≤ 10 min"), ("le_60_min", "10–60 min"), ("le_6_h", "1–6 h"),
           ("le_24_h", "6–24 h"), ("over_24_h", "> 24 h"))
SHUFFLE_ROWS = (("links_named", "named links"), ("trees_with_named_links", "trees with named links"),
                ("adopters", "adopters (all statuses)"),
                ("linked_share_of_later_adopters", "linked share of later adopters"),
                ("parallel_starts", "parallel starts"), ("agent_place_pairs", "agent-page pairs"))
STATUS_ROWS = (("linked", "linked to a named earlier user", "links_named"),
               ("parallel", "parallel start", "parallel_starts"),
               ("unnamed", "source unnamed", "links_from_exposure_only_rows"),
               ("clock_dropped", "dropped by the clock rule", "links_dropped_by_clock"))
LOG_TICKS = ((0, "0"), (1, "1 min"), (10, "10 min"), (60, "1 h"), (360, "6 h"), (1440, "1 day"),
             (10080, "1 week"), (43200, "30 days"), (525600, "1 year"))


# ---------------------------------------------------------------- picking and node detail


def _record_key(rec):
    return (-rec["largest_branch"], -rec["links_named"], rec["token_id"])


def beats_every_shuffle(rec):
    """True when a tree has named links and no shuffle reached its count."""
    return (rec["links_named"] > 0 and rec["shuffle_links_mean"] is not None
            and rec["shuffle_ge_observed"] == 0)


def _linked(records):
    return sorted((r for r in records if r["links_named"] > 0), key=_record_key)


def pick_lead(records, lead=None):
    """(token_id of the lead tree or None, the rule that picked it)."""
    if lead is not None:
        if not any(r["token_id"] == lead for r in records):
            raise ValueError("--lead %s is not a tree of this run" % short(lead, 20))
        return lead, RULE_GIVEN
    linked = _linked(records)
    beat = [r for r in linked if beats_every_shuffle(r)]
    if beat:
        return beat[0]["token_id"], RULE_BEAT
    if linked:
        return linked[0]["token_id"], RULE_LARGEST
    return None, RULE_NONE


def pick_scale(records):
    """token_id of the tree with the largest branch (None without named links)."""
    linked = _linked(records)
    return linked[0]["token_id"] if linked else None


def detail_ids(records, lead_id):
    """Trees that get node detail: lead, largest, top by branch overall and among shuffle-beaters."""
    ids = []

    def add(tid):
        if tid is not None and tid not in ids:
            ids.append(tid)

    add(lead_id)
    add(pick_scale(records))
    linked = _linked(records)
    for rec in linked[:DETAIL_TOP]:
        add(rec["token_id"])
    for rec in [r for r in linked if beats_every_shuffle(r)][:DETAIL_TOP]:
        add(rec["token_id"])
    return ids


DETAIL_FIELDS = ("token_id", "token", "cls", "kind", "agents", "status", "links_named", "links_unnamed",
                 "depth", "largest_branch", "uses", "places", "first_use", "span_hours", "risky",
                 "seed_branch", "seed_min_to_3rd", "seed_min_to_10th", "branch_min_to_3rd",
                 "branch_min_to_10th", "shuffle_links_mean", "shuffle_ge_observed")


def tree_detail(tree, rows, rec):
    """Node-level detail of one tree: one node per adopter, in first-use order.

    Node fields: n, agent (escaped label), t_min (minutes after the tree's
    first use), family, group, status, parent (node index or None), lag_min.
    ``first`` is node 0; ``seed`` and ``branch_root`` are node indices.
    """
    start = tree.adoptions[0].time
    index = {a.agent: n for n, a in enumerate(tree.adoptions)}
    big_root, big_size = None, 0
    for root, size in branch_sizes(tree).items():  # first-use order: ties go to the earlier root
        if size > big_size:
            big_root, big_size = root, size
    nodes = []
    for n, a in enumerate(tree.adoptions):
        row = rows[a.row]
        named = a.parent is not None
        nodes.append({
            "n": n,
            "agent": short(a.agent),
            "t_min": round((a.time - start) / 60.0, 3),
            "family": short(row.family or NO_FAMILY),
            "group": short(row.group or NO_FAMILY),
            "status": a.status,
            "parent": index[a.parent] if named else None,
            "lag_min": round(a.link.lag / 60.0, 2) if named and a.link is not None else None,
        })
    out = {k: rec[k] for k in DETAIL_FIELDS}
    out["first"] = 0
    out["seed"] = index.get(tree.effective_seed) if tree.effective_seed is not None else None
    out["branch_root"] = index[big_root] if big_root is not None else 0
    out["nodes"] = nodes
    return out


def detail_set(result, records, lead=None):
    """The top_trees.json object for one run (see the module docstring)."""
    lead_id, rule = pick_lead(records, lead)
    ids = detail_ids(records, lead_id)
    by_id = {r["token_id"]: r for r in records}
    keys = {token_id(k): k for k in result.trees}
    trees = [tree_detail(result.trees[keys[tid]], result.rows, by_id[tid]) for tid in ids]
    return {
        "version": DETAIL_VERSION,
        "lead": lead_id,
        "lead_rule": rule,
        "scale": pick_scale(records),
        "shuffles": result.null.shuffles if result.null is not None else 0,
        "trees": trees,
    }


# ---------------------------------------------------------------- small SVG helpers


def _e(value):
    return html.escape("" if value is None else str(value), quote=True)


def _c(v):
    s = "%.1f" % v
    return s[:-2] if s.endswith(".0") else s


def _num(x, digits=1):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return "{:,}".format(x)
    return "{:,.{d}f}".format(x, d=digits)


def _clip(s, n):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n - 3] + "..."


def _tw(s, size):
    """Rough text width in px (system sans)."""
    return len(s) * size * 0.58


def _text(x, y, s, size=12, fill=INK2, anchor="start", weight=None, ident=None, rotate=False):
    attrs = ['x="%s"' % _c(x), 'y="%s"' % _c(y), 'font-size="%s"' % _c(size), 'fill="%s"' % fill]
    if anchor != "start":
        attrs.append('text-anchor="%s"' % anchor)
    if weight:
        attrs.append('font-weight="%s"' % weight)
    if ident:
        attrs.append('id="%s"' % ident)
    if rotate:
        attrs.append('transform="rotate(-90 %s %s)"' % (_c(x), _c(y)))
    return "<text %s>%s</text>" % (" ".join(attrs), _e(s))


def _svg(width, height, prefix, label, parts, standalone):
    ns = ' xmlns="http://www.w3.org/2000/svg"' if standalone else ""
    head = '<?xml version="1.0" encoding="UTF-8"?>\n' if standalone else ""
    width, height = int(math.ceil(width)), int(math.ceil(height))
    return (head + '<svg%s width="%d" height="%d" viewBox="0 0 %d %d" role="img" aria-labelledby="%stitle-el" '
            'font-family="%s"><title id="%stitle-el">%s</title><rect width="%d" height="%d" fill="%s"/>%s</svg>'
            % (ns, width, height, width, height, prefix, FONT, prefix, _e(label), width, height, SURFACE,
               "".join(parts)) + ("\n" if standalone else ""))


def _legend(items, x0, y0, max_x, size=12):
    """Lay out legend keys left to right, wrapping. items: (kind, colour, label). Returns (parts, bottom)."""
    parts, x, y = [], x0, y0
    for kind, colour, label in items:
        width = 18 + _tw(label, size) + 22
        if x > x0 and x + width > max_x:
            x, y = x0, y + 22
        cy = y - 4
        if kind == "swatch":
            parts.append('<rect class="legend-swatch" x="%s" y="%s" width="12" height="12" rx="2" fill="%s"/>'
                         % (_c(x), _c(cy - 6), colour))
        elif kind == "dot":
            parts.append('<circle cx="%s" cy="%s" r="4.5" fill="%s"/>' % (_c(x + 6), _c(cy), colour))
        elif kind == "ring":
            parts.append('<circle cx="%s" cy="%s" r="4" fill="%s" stroke="%s" stroke-width="2"/>'
                         % (_c(x + 6), _c(cy), SURFACE, colour))
        elif kind == "first":
            parts.append('<circle cx="%s" cy="%s" r="4" fill="%s"/><circle cx="%s" cy="%s" r="7" fill="none" '
                         'stroke="%s" stroke-width="1.5"/>' % (_c(x + 6), _c(cy), colour, _c(x + 6), _c(cy), INK))
        elif kind == "square":
            parts.append('<rect x="%s" y="%s" width="8" height="8" fill="%s" stroke="%s" stroke-width="2"/>'
                         % (_c(x + 2), _c(cy - 4), SURFACE, colour))
        elif kind == "range":
            parts.append('<rect x="%s" y="%s" width="14" height="10" rx="3" fill="%s"/>'
                         % (_c(x - 1), _c(cy - 5), colour))
        elif kind == "tick":
            parts.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1.5"/>'
                         % (_c(x + 6), _c(cy - 7), _c(x + 6), _c(cy + 7), colour))
        parts.append(_text(x + 18, y, label, size=size, fill=INK2))
        x += width
    return parts, y + 8


def _nice_step(vmax, ticks=5):
    raw = max(vmax, 1) / float(ticks)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            step = m * mag
            return int(step) if step >= 1 and float(step).is_integer() else step
    return 10 * mag


def _col_path(x, base, w, h, r=4.0):
    """Column growing up from ``base``: rounded data end, square at the baseline."""
    r = min(r, h, w / 2.0)
    top = base - h
    return ("M%s %sV%sQ%s %s %s %sH%sQ%s %s %s %sV%sZ"
            % (_c(x), _c(base), _c(top + r), _c(x), _c(top), _c(x + r), _c(top), _c(x + w - r),
               _c(x + w), _c(top), _c(x + w), _c(top + r), _c(base)))


def _bar_path(x0, y, w, h, r=4.0):
    """Bar growing right from ``x0``: rounded data end, square at the baseline."""
    r = min(r, w, h / 2.0)
    right = x0 + w
    return ("M%s %sH%sQ%s %s %s %sV%sQ%s %s %s %sH%sZ"
            % (_c(x0), _c(y), _c(right - r), _c(right), _c(y), _c(right), _c(y + r), _c(y + h - r),
               _c(right), _c(y + h), _c(right - r), _c(y + h), _c(x0)))


# ---------------------------------------------------------------- the tree figure


def drawn_nodes(detail, limit=MAX_DRAW):
    """(node indices to draw, "all" or "branch"): every adopter, or the largest branch when too many."""
    nodes = detail["nodes"]
    if len(nodes) <= limit:
        return list(range(len(nodes))), "all"
    root = detail["branch_root"]
    keep = {root}
    for nd in nodes:  # parents come earlier in first-use order
        if nd["parent"] is not None and nd["parent"] in keep:
            keep.add(nd["n"])
    return sorted(keep)[:limit], "branch"


def time_axis(times):
    """(t -> 0..1, ticks [(minutes, label)], axis label) for minutes since the first use.

    Linear in minutes, hours or days; log(1 + minutes) when most uses sit in the
    first tenth of the span, so a burst and a long tail both stay readable.
    """
    tmax = max(max(times) if times else 0.0, 1.0)
    mid = sorted(times)[len(times) // 2] if times else 0.0
    if tmax >= 60 and mid < 0.1 * tmax:
        end = next((m for m, _l in LOG_TICKS if m >= tmax), tmax)
        top = math.log10(1 + end)
        ticks = [(m, label) for m, label in LOG_TICKS if m <= end]
        return (lambda t: math.log10(1 + max(t, 0.0)) / top), ticks, "time since the first use (log scale)"
    if tmax <= 120:
        unit, size, name = "min", 1.0, "minutes"
    elif tmax <= 72 * 60:
        unit, size, name = "h", 60.0, "hours"
    else:
        unit, size, name = "d", 1440.0, "days"
    span = tmax / size
    step = next((s for s in (1, 2, 3, 5, 6, 10, 12, 15, 20, 24, 30, 50, 60, 100, 200, 500) if span / s <= 8),
                1000)
    count = max(int(math.ceil(span / step - 1e-9)), 1)
    end = count * step * size
    ticks = [(k * step * size, ("%g %s" % (k * step, unit)) if k else "0") for k in range(count + 1)]
    return (lambda t: t / end), ticks, "time since the first use (%s)" % name


def tree_colours(detail, picked):
    """(family -> colour, legend items): the 3 most used families in the drawing, then grey."""
    counts = Counter(detail["nodes"][i]["family"] for i in picked)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    colour, legend = {}, []
    for k, (fam, n) in enumerate(ranked):
        if k < len(SERIES):
            colour[fam] = SERIES[k]
            legend.append(("swatch", SERIES[k], "%s (%d)" % (_clip(fam, 40), n)))
        else:
            colour[fam] = OTHER
    rest = ranked[len(SERIES):]
    if rest:
        legend.append(("swatch", OTHER, "%d other page families (%d)" % (len(rest), sum(n for _f, n in rest))))
    return colour, legend


def svg_tree(detail, shuffles, public=True, prefix="lead-", heading="Lead spread tree", standalone=True,
             note_lines=()):
    """One spread tree, time left to right, one row per agent, coloured by page family.

    ``public`` leaves out agent labels (no tooltips) and the token text.
    ``note_lines`` (from ``selection_lines``) go under the subtitle, so a copy
    of the figure on its own still says how the tree was picked.
    """
    nodes = detail["nodes"]
    picked, mode = drawn_nodes(detail)
    drawn = set(picked)
    children, roots = defaultdict(list), []
    for i in picked:
        p = nodes[i]["parent"]
        (children[p] if p is not None and p in drawn else roots).append(i)
    order, stack = [], list(reversed(roots))
    while stack:
        i = stack.pop()
        order.append(i)
        stack.extend(reversed(children[i]))
    lane = {i: k for k, i in enumerate(order)}
    colour, fam_legend = tree_colours(detail, picked)

    W, ML, MR = 960, 72, 180
    PW = W - ML - MR
    parts = []
    title = "%s: %s token %s" % (heading, detail["cls"], detail["token_id"])
    parts.append(_text(24, 32, title, size=18, fill=INK, weight="600", ident=prefix + "title"))
    st = detail["status"]
    lines = ["%s agents used it: %s linked to a named earlier user, %s parallel starts; "
             "largest branch %s agents, depth %s"
             % (_num(detail["agents"]), _num(st["linked"]), _num(st["parallel"]),
                _num(detail["largest_branch"]), _num(detail["depth"]))]
    if detail["shuffle_links_mean"] is not None and shuffles:
        lines.append("Named links %s; under the label shuffle the mean is %s, and %d of %d shuffles reached %s"
                     % (_num(detail["links_named"]), _num(detail["shuffle_links_mean"]),
                        detail["shuffle_ge_observed"], shuffles, _num(detail["links_named"])))
    else:
        lines.append("Named links %s; no label shuffle in this run" % _num(detail["links_named"]))
    lines.extend(note_lines)
    if not public:
        lines.append("token: %s" % detail["token"])
    y = 54
    for line in lines:
        parts.append(_text(24, y, line, size=12.5, fill=INK2))
        y += 18
    shape_items = [("dot", INK2, "linked"), ("first", INK2, "first user")]
    present = set(nodes[i]["status"] for i in picked)
    if "parallel" in present:
        shape_items.append(("ring", INK2, "parallel start (no earlier use seen)"))
    if present & {"unnamed", "clock_dropped"}:
        shape_items.append(("square", INK2, "source unnamed or dropped by the clock rule"))
    leg, y = _legend(fam_legend, 24, y + 14, W - 24)
    parts.extend(leg)
    leg, y = _legend(shape_items, 24, y + 14, W - 24)
    parts.extend(leg)

    n = len(order)
    lane_h = min(40.0, max(14.0, 200.0 / max(n, 1))) if n <= 40 else max(5.0, min(14.0, 560.0 / n))
    r = 4.0 if lane_h >= 10 else 3.0
    top = y + 24
    PH = lane_h * max(n, 1)
    bottom = top + PH
    scale, ticks, xlabel = time_axis([nodes[i]["t_min"] for i in picked])

    def X(t):
        return ML + scale(t) * PW

    def Y(i):
        return top + (lane[i] + 0.5) * lane_h

    for t, label in ticks:
        x = X(t)
        parts.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1"/>'
                     % (_c(x), _c(top - 6), _c(x), _c(bottom), GRID))
        parts.append(_text(x, bottom + 17, label, size=11, fill=MUTED, anchor="middle"))
    parts.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1"/>'
                 % (_c(ML), _c(bottom), _c(ML + PW), _c(bottom), AXIS))
    parts.append(_text(ML + PW / 2.0, bottom + 40, xlabel, size=12, fill=INK2, anchor="middle",
                       ident=prefix + "x-label"))
    parts.append(_text(30, top + PH / 2.0, "agents, one row each", size=12, fill=INK2, anchor="middle",
                       ident=prefix + "y-label", rotate=True))

    for i in order:  # links first, so the marks sit on top
        p = nodes[i]["parent"]
        if p is not None and p in drawn:
            parts.append('<path d="M%s %sV%sH%s" fill="none" stroke="%s" stroke-width="1.25"/>'
                         % (_c(X(nodes[p]["t_min"])), _c(Y(p)), _c(Y(i)), _c(X(nodes[i]["t_min"])), AXIS))
    labels = defaultdict(list)
    labels[detail["first"]].append("first user")
    if detail["seed"] is not None:
        labels[detail["seed"]].append("effective seed")
    if detail["branch_root"] not in (detail["first"], detail["seed"]):
        labels[detail["branch_root"]].append("largest branch root")
    for i in order:
        nd = nodes[i]
        x, yy, col = X(nd["t_min"]), Y(i), colour[nd["family"]]
        tip = ""
        if not public:
            src = (" | source: %s" % nodes[nd["parent"]]["agent"]) if nd["parent"] is not None else ""
            tip = "<title>%s | %s min after the first use | %s | %s%s</title>" % (
                _e(nd["agent"]), _e(_num(nd["t_min"])), _e(nd["family"]), _e(STATUS_NAMES[nd["status"]]), _e(src))
        attrs = 'class="node" data-family="%s"' % _e(nd["family"])
        if nd["status"] == "parallel":
            parts.append('<circle %s cx="%s" cy="%s" r="%s" fill="%s" stroke="%s" stroke-width="2">%s</circle>'
                         % (attrs, _c(x), _c(yy), _c(r), SURFACE, col, tip))
        elif nd["status"] in ("unnamed", "clock_dropped"):
            parts.append('<rect %s x="%s" y="%s" width="%s" height="%s" fill="%s" stroke="%s" stroke-width="2">'
                         '%s</rect>' % (attrs, _c(x - r), _c(yy - r), _c(2 * r), _c(2 * r), SURFACE, col, tip))
        else:
            parts.append('<circle %s cx="%s" cy="%s" r="%s" fill="%s" stroke="%s" stroke-width="1.5">%s</circle>'
                         % (attrs, _c(x), _c(yy), _c(r), col, SURFACE, tip))
        if i == detail["first"]:
            parts.append('<circle cx="%s" cy="%s" r="%s" fill="none" stroke="%s" stroke-width="1.5"/>'
                         % (_c(x), _c(yy), _c(r + 3), INK))
        if i in labels:
            parts.append(_text(x + r + 7, yy + 4, ", ".join(labels[i]), size=11, fill=INK2))
    if mode == "all":
        note = "Drawn: all %s adopters." % _num(len(nodes))
    else:
        note = ("Drawn: the largest branch, %s of %s adopters; the other %s are not drawn."
                % (_num(len(picked)), _num(len(nodes)), _num(len(nodes) - len(picked))))
    note += " Each mark is one agent's first use, coloured by the page family of that use."
    parts.append(_text(24, bottom + 64, note, size=11, fill=MUTED))
    return _svg(W, bottom + 80, prefix, title, parts, standalone)


# ---------------------------------------------------------------- the other charts


def svg_speed(facts, prefix="speed-"):
    """Columns: trees per time bucket, to the 3rd and to the 10th agent linked below the seed."""
    sp = facts["scores"]["speed"]
    series = (("seed_to_3rd", "3rd agent", SERIES[0]), ("seed_to_10th", "10th agent", SERIES[1]))
    W, H, ML, MR = 760, 380, 64, 24
    PW = W - ML - MR
    parts = []
    title = "Spread speed: time to the 3rd and 10th linked agent"
    parts.append(_text(24, 32, title, size=17, fill=INK, weight="600", ident=prefix + "title"))
    parts.append(_text(24, 54, "Trees by the time from the effective seed's first use to its k-th agent "
                                "linked below it (any depth)", size=12.5, fill=INK2))
    items = []
    for field, name, col in series:
        block = sp[field]
        items.append(("swatch", col, "%s: %s trees, median %s min" % (name, _num(block["trees"]),
                                                                      _num(block["median_minutes"]))))
    leg, y = _legend(items, 24, 80, W - 24)
    parts.extend(leg)
    top, bottom = y + 30, H - 70
    PH = bottom - top
    vmax = max([sp[f]["buckets"][k] for f, _n, _c2 in series for k, _l in BUCKETS] + [1])
    step = _nice_step(vmax)
    ymax = int(math.ceil(vmax / float(step))) * step
    k = 0
    while k * step <= ymax:
        yy = bottom - (k * step) / float(ymax) * PH
        parts.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1"/>'
                     % (_c(ML), _c(yy), _c(ML + PW), _c(yy), GRID if k else AXIS))
        parts.append(_text(ML - 8, yy + 4, _num(k * step), size=11, fill=MUTED, anchor="end"))
        k += 1
    band = PW / float(len(BUCKETS))
    bar = min(24.0, (band - 24) / 2.0)
    for b, (key, label) in enumerate(BUCKETS):
        cx = ML + band * (b + 0.5)
        x0 = cx - (2 * bar + 2) / 2.0
        for s, (field, name, col) in enumerate(series):
            v = sp[field]["buckets"][key]
            h = v / float(ymax) * PH
            x = x0 + s * (bar + 2)
            if v > 0:
                parts.append('<path d="%s" fill="%s"><title>%s, %s: %s trees</title></path>'
                             % (_col_path(x, bottom, bar, h), col, _e(name), _e(label), v))
            parts.append(_text(x + bar / 2.0, bottom - h - 5, _num(v), size=11, fill=MUTED, anchor="middle"))
        parts.append(_text(cx, bottom + 18, label, size=11.5, fill=INK2, anchor="middle"))
    parts.append(_text(ML + PW / 2.0, bottom + 42, "time to reach the k-th agent", size=12, fill=INK2,
                       anchor="middle", ident=prefix + "x-label"))
    parts.append(_text(20, top + PH / 2.0, "trees", size=12, fill=INK2, anchor="middle",
                       ident=prefix + "y-label", rotate=True))
    return _svg(W, H, prefix, title, parts, False)


def shuffle_rows(facts):
    """[(label, observed, mean, p05, p95, shuffles at or above observed, shuffles)] for the shuffle chart."""
    nm = facts.get("nullmodel")
    if not nm or not nm.get("shuffles"):
        return []
    rows = []
    for key, label in SHUFFLE_ROWS:
        s = nm.get(key)
        if not s or not s.get("mean"):
            continue
        rows.append((label, s["observed"], s["mean"], s["p05"], s["p95"], s["shuffles_ge_observed"],
                     s["shuffles"]))
    return rows


def _fmt_metric(v):
    return _num(v, 3) if isinstance(v, float) and v < 10 else _num(round(v, 1) if isinstance(v, float) else v)


def svg_shuffle(facts, prefix="shuffle-"):
    """Observed / shuffle mean per measure, with the shuffle's 5th-95th percentile band, on one axis."""
    rows = shuffle_rows(facts)
    if not rows:
        return None
    nm = facts["nullmodel"]
    ratios = [1.0]
    for _l, obs, mean, p05, p95, _g, _n in rows:
        ratios += [obs / mean, p05 / mean, p95 / mean]
    lo = max(0.0, math.floor((min(ratios) - 0.05) * 10) / 10.0)
    hi = math.ceil((max(ratios) + 0.05) * 10) / 10.0
    step = 0.1 if hi - lo <= 1.0 else (0.25 if hi - lo <= 2.5 else 0.5)
    W, LC, RC = 920, 230, 210
    PW = W - LC - RC - 24
    parts = []
    title = "Observed against %d label shuffles" % nm["shuffles"]
    parts.append(_text(24, 32, title, size=17, fill=INK, weight="600", ident=prefix + "title"))
    scope = "page and UTC day" if nm.get("scope") == "place_day" else "UTC day"
    parts.append(_text(24, 54, "Agent labels permuted within each %s, seed %s. Below 1 means fewer than "
                               "the shuffle gives." % (scope, nm["seed"]), size=12.5, fill=INK2))
    leg, y = _legend([("dot", SERIES[0], "observed"), ("range", WASH, "shuffle 5th to 95th percentile"),
                      ("tick", INK2, "shuffle mean (1)")], 24, 80, W - 24)
    parts.extend(leg)
    top = y + 30
    row_h = 40
    bottom = top + row_h * len(rows)

    def X(v):
        return LC + (v - lo) / (hi - lo) * PW

    k = 0
    while lo + k * step <= hi + 1e-9:
        v = lo + k * step
        parts.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1"/>'
                     % (_c(X(v)), _c(top - 6), _c(X(v)), _c(bottom), GRID))
        parts.append(_text(X(v), bottom + 17, ("%.2f" % v).rstrip("0").rstrip("."), size=11, fill=MUTED,
                           anchor="middle"))
        k += 1
    parts.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1.5"/>'
                 % (_c(X(1.0)), _c(top - 10), _c(X(1.0)), _c(bottom), INK2))
    for n, (label, obs, mean, p05, p95, ge, shuffles) in enumerate(rows):
        yy = top + row_h * (n + 0.5)
        parts.append(_text(LC - 14, yy + 4, label, size=12, fill=INK2, anchor="end"))
        parts.append('<rect x="%s" y="%s" width="%s" height="10" rx="3" fill="%s"/>'
                     % (_c(X(p05 / mean)), _c(yy - 5), _c(max(X(p95 / mean) - X(p05 / mean), 2)), WASH))
        parts.append('<circle cx="%s" cy="%s" r="5" fill="%s" stroke="%s" stroke-width="2"><title>%s: observed %s, '
                     'shuffle mean %s</title></circle>' % (_c(X(obs / mean)), _c(yy), SERIES[0], SURFACE,
                                                            _e(label), _e(_fmt_metric(obs)), _e(_fmt_metric(mean))))
        parts.append(_text(W - RC, yy - 2, "%s vs %s" % (_fmt_metric(obs), _fmt_metric(mean)), size=12, fill=INK))
        parts.append(_text(W - RC, yy + 13, "%d of %d shuffles at or above" % (ge, shuffles), size=11, fill=MUTED))
    parts.append(_text(LC + PW / 2.0, bottom + 40, "observed ÷ shuffle mean", size=12, fill=INK2,
                       anchor="middle", ident=prefix + "x-label"))
    return _svg(W, bottom + 56, prefix, title, parts, False)


def status_rows(facts):
    """[(label, observed, shuffle mean or None)] per adopter status (not the first user)."""
    nm = facts.get("nullmodel") or {}
    status = facts["cascade"]["status"]
    rows = []
    for key, label, null_key in STATUS_ROWS:
        s = nm.get(null_key) if nm.get("shuffles") else None
        rows.append((label, status.get(key, 0), s["mean"] if s else None))
    return rows


def svg_status(facts, prefix="status-"):
    """Bars: adopters by status, observed and the shuffle mean."""
    rows = status_rows(facts)
    has_null = any(m is not None for _l, _o, m in rows)
    W, LC, MR = 920, 230, 120
    PW = W - LC - MR
    parts = []
    title = "Adopters by status, observed and under the shuffle"
    parts.append(_text(24, 32, title, size=17, fill=INK, weight="600", ident=prefix + "title"))
    parts.append(_text(24, 54, "Every adopter after a token's first user is linked, a parallel start, "
                                "unnamed or dropped by the clock rule", size=12.5, fill=INK2))
    items = [("swatch", SERIES[0], "observed")]
    if has_null:
        items.append(("swatch", OTHER, "shuffle mean"))
    leg, y = _legend(items, 24, 80, W - 24)
    parts.extend(leg)
    top = y + 26
    bar, gap = 12.0, 2.0
    row_h = 2 * bar + gap + 20 if has_null else bar + 22
    bottom = top + row_h * len(rows)
    vmax = max([o for _l, o, _m in rows] + [m or 0 for _l, _o, m in rows] + [1])
    step = _nice_step(vmax)
    xmax = int(math.ceil(vmax / float(step))) * step
    k = 0
    while k * step <= xmax:
        x = LC + (k * step) / float(xmax) * PW
        parts.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1"/>'
                     % (_c(x), _c(top - 6), _c(x), _c(bottom), GRID if k else AXIS))
        parts.append(_text(x, bottom + 17, _num(k * step), size=11, fill=MUTED, anchor="middle"))
        k += 1
    for n, (label, obs, mean) in enumerate(rows):
        y0 = top + row_h * n + 8
        block = 2 * bar + gap if has_null else bar
        parts.append(_text(LC - 14, y0 + block / 2.0 + 4, label, size=12, fill=INK2, anchor="end"))
        values = [(obs, SERIES[0], "observed")] + ([(mean, OTHER, "shuffle mean")] if has_null else [])
        for s, (v, col, name) in enumerate(values):
            yy = y0 + s * (bar + gap)
            w = (v or 0) / float(xmax) * PW
            if v:
                parts.append('<path d="%s" fill="%s"><title>%s, %s: %s</title></path>'
                             % (_bar_path(LC, yy, w, bar), col, _e(label), name, _e(_num(v))))
            parts.append(_text(LC + w + 6, yy + bar - 2, _num(v) if v is not None else "n/a", size=11,
                               fill=MUTED))
    parts.append(_text(LC + PW / 2.0, bottom + 40, "adopters", size=12, fill=INK2, anchor="middle",
                       ident=prefix + "x-label"))
    return _svg(W, bottom + 56, prefix, title, parts, False)


# ---------------------------------------------------------------- the page


CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: %(page)s; color: %(ink)s; font: 15px/1.5 %(font)s; }
main { max-width: 1000px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 26px; margin: 8px 0 4px; }
h2 { font-size: 19px; margin: 0 0 8px; }
p { margin: 6px 0; max-width: 78ch; }
.meta, .note { color: %(ink2)s; font-size: 13px; }
section { background: %(surface)s; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px;
          padding: 18px 18px 14px; margin: 18px 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
.tile { background: %(surface)s; border: 1px solid rgba(11,11,11,0.10); border-radius: 8px; padding: 10px 12px; }
.tile .label { color: %(ink2)s; font-size: 12.5px; }
.tile .value { font-size: 22px; font-weight: 600; }
figure { margin: 10px 0; }
svg { display: block; max-width: 100%%; height: auto; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; margin: 8px 0; }
th, td { padding: 4px 10px 4px 0; text-align: left; vertical-align: top; border-bottom: 1px solid %(grid)s; }
th { color: %(ink2)s; font-weight: 600; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
code { font-size: 12.5px; background: %(page)s; padding: 1px 4px; border-radius: 4px; word-break: break-all; }
details { margin: 6px 0; }
summary { cursor: pointer; color: %(ink2)s; font-size: 13px; }
ul { margin: 6px 0; padding-left: 20px; }
li { margin: 3px 0; max-width: 80ch; }
footer { color: %(ink2)s; font-size: 12.5px; margin-top: 24px; }
.verdict { max-width: none; margin: 14px 0 0; padding: 10px 14px; background: %(surface)s;
           border: 1px solid rgba(11,11,11,0.10); border-left: 4px solid %(ink2)s; border-radius: 8px; }
""" % {"page": PAGE, "ink": INK, "ink2": INK2, "font": FONT, "surface": SURFACE, "grid": GRID}


def _table(headers, rows, numeric=()):
    """HTML table; every cell escaped. ``numeric`` holds column indices to right-align."""
    out = ['<div class="scroll"><table><thead><tr>']
    for k, h in enumerate(headers):
        out.append('<th%s>%s</th>' % (' class="n"' if k in numeric else "", _e(h)))
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for k, cell in enumerate(row):
            out.append('<td%s>%s</td>' % (' class="n"' if k in numeric else "", _e(_cell(cell))))
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _cell(v):
    """Table cell text: ints with commas, small floats to 3 places, others to 1."""
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return "{:,}".format(v)
    if isinstance(v, float):
        return _num(v, 3) if abs(v) < 10 else _num(v, 1)
    return v


def _details(summary, body):
    return "<details><summary>%s</summary>%s</details>" % (_e(summary), body)


def _tiles(items):
    out = ['<div class="tiles">']
    for label, value in items:
        out.append('<div class="tile"><div class="label">%s</div><div class="value">%s</div></div>'
                   % (_e(label), _e(value)))
    out.append("</div>")
    return "".join(out)


def _find(detail, tid):
    if tid is None:
        return None
    return next((t for t in detail["trees"] if t["token_id"] == tid), None)


def beat_bound(nm):
    """tokens / (shuffles + 1): the most trees that beat every shuffle by luck, on average.

    Recomputed from the counts when they are there, because facts.json rounds it
    to 2 places, and a small run must never read "up to 0.0 trees".
    """
    if nm.get("shuffles") and nm.get("tokens") is not None:
        return nm["tokens"] / (nm["shuffles"] + 1)
    return nm.get("chance_beats_upper_bound")


def _small(x):
    """A probability or bound under 1 to 2 significant figures, never in e-notation."""
    text = "%.2g" % x
    if "e" in text:
        text = ("%.10f" % float(text)).rstrip("0").rstrip(".")
    return text


def bound_text(bound):
    """The chance bound as printed: whole from 10, 1 decimal from 1, 2 decimals under 1.

    Under 1 it never prints "0.0" or "0.00": a bound that small gets 2
    significant figures instead (0.003).
    """
    if bound is None:
        return "n/a"
    if bound >= 10:
        return _num(int(round(bound)))
    if bound >= 1:
        return _num(float(bound))
    text = "%.2f" % bound
    if text == "0.00" and bound > 0:
        text = _small(bound)
    return text


def beats_beyond_chance(nm):
    """(beats, bound, luck) when more trees beat every shuffle than the chance bound, else None.

    ``luck`` is bound / beats. Each tree beats all N shuffles by luck with
    probability at most 1 / (N + 1), so the expected number of such trees is at
    most ``bound``, and by Markov's inequality luck alone gives ``beats`` or more
    of them with probability at most ``luck``. Only then does a caption drop its
    "not proof" line; the number stays printed, so a narrow win reads as narrow.
    """
    beats, bound = nm.get("trees_beating_every_shuffle"), beat_bound(nm)
    if not nm.get("shuffles") or beats is None or bound is None or beats <= bound:
        return None
    return beats, bound, bound / beats


def _lead_text(rule, facts):
    nm = facts.get("nullmodel") or {}
    text = "How this tree was picked: %s." % rule
    if rule == RULE_BEAT and nm.get("shuffles"):
        text += (" %s trees beat every one of the %d shuffles; chance alone would let up to %s trees do that "
                 "(tokens / (shuffles + 1))" % (_num(nm["trees_beating_every_shuffle"]), nm["shuffles"],
                                                bound_text(beat_bound(nm))))
        beyond = beats_beyond_chance(nm)
        if beyond:
            text += (" on average, so luck alone gives %s or more with a chance of at most %s (Markov's "
                     "inequality). Each named link is still a candidate: hand-check a sample before calling "
                     "it spread." % (_num(beyond[0]), _small(beyond[2])))
        else:
            text += ", so this tree shows how the method draws a spread, not proof that spread beat chance."
    text += (" Another tree: python -m patient_zero.report --out OUT --lead TOKEN_ID (a tree in "
             "top_trees.json), or cli.py --lead TOKEN_ID.")
    return text


SCOPE_WORDS = {"day": "within each UTC day", "place_day": "within each place and UTC day"}


def chance_verdict(f5):
    """(word, sentences): observed named links against the label shuffle, in plain words.

    The report prints "Against the label shuffle: <word>." in bold, then the
    sentences. ``f5`` is facts["slots"]["F5"]. The word is "above chance" when the
    observed count is over the shuffle's 95th percentile, "below chance" when
    it is under the 5th, and "within chance" otherwise. Returns (None, None)
    when the run had no shuffle.
    """
    if not f5 or not f5.get("shuffles"):
        return None, None
    obs, mean = f5["links_observed"], f5["links_expected_mean"]
    lo, hi = f5["links_expected_p05"], f5["links_expected_p95"]
    if obs > hi:
        word = "above chance"
    elif obs < lo:
        word = "below chance"
    else:
        word = "within chance"
    scope = SCOPE_WORDS.get(f5.get("scope"), "within each shuffle group")
    text = ("Named links: %s observed against a shuffle mean of %s (5th to 95th percentile %s to %s); "
            "%d of %d shuffles reached the observed count."
            % (_num(obs), _num(mean), _num(lo), _num(hi), f5["shuffles_ge_observed"], f5["shuffles"]))
    if f5.get("linked_share_observed") is not None and f5.get("linked_share_expected_mean") is not None:
        text += (" Share of later adopters linked to a named earlier user (this allows for the shuffle making "
                 "more adopters): %s observed, %s under the shuffle."
                 % (_num(f5["linked_share_observed"], 3), _num(f5["linked_share_expected_mean"], 3)))
    if word == "below chance":
        text += (" Fewer adoptions follow a visit to an earlier user's page than when agent labels are shuffled "
                 "%s, so these links are not evidence of spread beyond chance." % scope)
        text += (" Falling below this shuffle does not show that agents avoid each other's tricks: the shuffle "
                 "deals one agent's repeated uses of a token to several labels, which adds adopters and links. "
                 "The order null in patient_zero.checks keeps every agent's own rows and is the fairer baseline.")
    elif word == "within chance":
        text += (" About as many adoptions follow a visit to an earlier user's page as when agent labels are "
                 "shuffled %s, so these links are not evidence of spread beyond chance." % scope)
    else:
        text += (" More adoptions follow a visit to an earlier user's page than when agent labels are shuffled "
                 "%s. Each link is still a candidate: hand-check a sample before calling it spread." % scope)
    return word, text


def selection_lines(rule, facts):
    """One or two short lines saying how the drawn tree was picked, for inside the figure itself."""
    nm = (facts or {}).get("nullmodel") or {}
    if rule == RULE_BEAT and nm.get("shuffles"):
        first = ("Picked as the largest of %s trees that beat every shuffle; chance alone allows up to %s such "
                 "trees among %s." % (_num(nm["trees_beating_every_shuffle"]), bound_text(beat_bound(nm)),
                                      _num(nm["tokens"])))
        beyond = beats_beyond_chance(nm)
        if beyond:
            return [first, "More trees beat every shuffle than luck gives on average; luck alone gives %s or more "
                           "with a chance of at most %s." % (_num(beyond[0]), _small(beyond[2]))]
        return [first, "So it shows how the method draws a spread; it is not proof that spread beat chance."]
    if rule == RULE_BEAT:
        return ["Picked as the largest tree that beat every shuffle."]
    if rule == RULE_LARGEST:
        return ["Picked as the largest branch among trees with named links; no tree beat every shuffle."]
    if rule == RULE_GIVEN:
        return ["Picked by hand (--lead)."]
    return []


def render_report(facts, records, detail, lead=None):
    """The report.html text. ``lead`` (a token_id in ``detail``) overrides the stored lead tree."""
    lead_id = lead or detail["lead"]
    rule = RULE_GIVEN if lead else detail["lead_rule"]
    lead_tree = _find(detail, lead_id)
    if lead_id is not None and lead_tree is None:
        raise ValueError("tree %s is not in top_trees.json; rerun cli.py with --lead %s"
                         % (short(lead_id, 20), short(lead_id, 20)))
    scale_tree = _find(detail, detail["scale"])
    shuffles = detail["shuffles"]
    s = facts["slots"]
    nm = facts.get("nullmodel") or {}
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width, initial-scale=1">'
           '<title>patient-zero report</title><style>%s</style></head><body><main>' % CSS]

    files = ", ".join("%s (sha256 %s...)" % (f["name"], f["sha256"][:12]) for f in facts["input"]["files"])
    if facts["input"].get("format"):
        files += ", read as %s" % facts["input"]["format"]
    if facts["input"].get("map"):
        files += ", --map %s" % ",".join("%s=%s" % kv for kv in facts["input"]["map"].items())
    code = facts.get("code") or {}
    out.append("<header><h1>patient-zero report</h1>")
    out.append('<p class="meta">Built by patient_zero/report.py from facts.json, trees.jsonl and top_trees.json. '
               "Opens offline: no scripts, no external files. Input (%s adapter): %s. Run started %s; "
               "commit %s%s. core_sha256 %s.</p>"
               % (_e(facts["input"]["adapter"]), _e(files), _e(facts["runtime"]["started_utc"]),
                  _e(code.get("commit")), " (uncommitted changes)" if code.get("dirty") else "",
                  _e(facts["core_sha256"])))
    out.append("</header>")

    f5 = s.get("F5") or {}
    out.append(_tiles([
        ("saves (rows)", _num(facts["load"]["rows"])),
        ("agents", _num(facts["load"]["agents"])),
        ("tokens used by 3+ agents", _num(s["F3"]["tokens_3plus_agents"])),
        ("trees with named links", _num(facts["trees"]["with_named_links"])),
        ("named links", _num(s["F4"]["links_named"])),
        ("observed ÷ shuffle mean", _num(f5.get("ratio_observed_to_expected"), 3)
         if f5.get("ratio_observed_to_expected") is not None else "n/a"),
        ("parallel starts", _num(s["F8"]["parallel_starts"])),
    ]))
    word, verdict = chance_verdict(f5)
    if verdict is None:
        out.append('<p class="verdict">No label shuffle in this run (--shuffles 0), so there is no chance '
                   "baseline: the links below are not compared with anything.</p>")
    else:
        out.append('<p class="verdict"><strong>Against the label shuffle: %s.</strong> %s</p>'
                   % (_e(word), _e(verdict)))

    out.append("<section><h2>Lead spread tree</h2>")
    if lead_tree is None:
        out.append("<p>No tree in this run has a named link, so there is no tree to draw.</p>")
    else:
        out.append("<p>%s</p>" % _e(_lead_text(rule, facts)))
        out.append("<figure>%s</figure>" % svg_tree(lead_tree, shuffles, public=False, prefix="lead-",
                                                      standalone=False, note_lines=selection_lines(rule, facts)))
        node_rows = []
        for nd in lead_tree["nodes"]:
            parent = nd["parent"]
            node_rows.append([nd["n"], nd["agent"], nd["t_min"], nd["family"], nd["group"],
                              STATUS_NAMES[nd["status"]], "" if parent is None else str(parent),
                              "" if nd["lag_min"] is None else nd["lag_min"]])
        out.append(_details("Table view: every adopter of the lead tree", _table(
            ["#", "agent", "minutes after first use", "page family", "wiki", "status", "source #",
             "lag (min)"], node_rows, numeric=(0, 2, 7))))
    out.append("</section>")

    if scale_tree is not None and scale_tree is not lead_tree:
        out.append("<section><h2>Largest tree, for scale</h2>")
        out.append("<p>The tree with the largest connected branch of named links. Its shuffle numbers are in "
                   "the subtitle: compare them before reading it as spread.</p>")
        out.append("<figure>%s</figure>" % svg_tree(scale_tree, shuffles, public=False, prefix="scale-",
                                                      heading="Largest tree", standalone=False))
        out.append("</section>")

    sp = facts["scores"]["speed"]
    out.append("<section><h2>Spread speed</h2>")
    out.append("<figure>%s</figure>" % svg_speed(facts))
    fast10 = sp["seed_to_10th"]["fastest"]
    if fast10:
        out.append('<p class="note">Fastest to a 10th agent: %s min (tree %s, %s class). Very fast spreads can be '
                   "burst editing (many labels within seconds); hand-check them before calling them spread.</p>"
                   % (_e(_num(fast10["minutes"], 2)), _e(fast10["token_id"]), _e(fast10["cls"])))
    speed_rows = [[label] + [sp[f]["buckets"][key] for f in ("seed_to_3rd", "seed_to_10th",
                                                             "largest_branch_to_3rd", "largest_branch_to_10th")]
                  for key, label in BUCKETS]
    speed_rows.append(["median minutes"] + [sp[f]["median_minutes"] for f in (
        "seed_to_3rd", "seed_to_10th", "largest_branch_to_3rd", "largest_branch_to_10th")])
    out.append(_details("Table view: trees per bucket (from the effective seed, and from the largest branch's root)",
                        _table(["time", "seed to 3rd", "seed to 10th", "branch root to 3rd",
                                "branch root to 10th"], speed_rows, numeric=(1, 2, 3, 4))))
    out.append("</section>")

    inf = facts["scores"]["influence"]
    out.append("<section><h2>Agent influence</h2>")
    out.append("<p>Top %d agents by adopters credited to them (named links out). Downstream counts everyone "
               "below the agent in its trees. %s of %s agents have at least one adopter; the top agent holds "
               "%s of named links and the top 10 hold %s.</p>"
               % (len(inf["top"]), _e(_num(inf["agents_with_adopters"])), _e(_num(inf["agents"])),
                  _e(_pct(inf["top_1_share"])), _e(_pct(inf["top_10_share"]))))
    out.append(_table(["#", "agent", "adopters", "downstream", "trees seeded", "trees first", "trees joined",
                       "parallel starts", "agents reached"],
                      [[r["rank"], r["agent"], r["adopters"], r["downstream"], r["trees_seeded"],
                        r["trees_first"], r["trees_joined"], r["parallel_starts"], r["agents_reached"]]
                       for r in inf["top"]], numeric=(0, 2, 3, 4, 5, 6, 7, 8)))
    out.append("</section>")

    out.append("<section><h2>Observed against the shuffle</h2>")
    chart = svg_shuffle(facts)
    if chart is None:
        out.append("<p>No label shuffle ran (--shuffles 0), so there is no chance baseline.</p>")
    else:
        out.append("<figure>%s</figure>" % chart)
        out.append(_details("Table view: observed and shuffle summary", _table(
            ["measure", "observed", "shuffle mean", "5th pct", "95th pct", "shuffles at or above", "shuffles"],
            [[label, _fmt_metric(o), _fmt_metric(m), _fmt_metric(a), _fmt_metric(b), g, n]
             for label, o, m, a, b, g, n in shuffle_rows(facts)], numeric=(1, 2, 3, 4, 5, 6))))
        out.append("<p>%s trees beat every shuffle (chance bound %s). Named links by token class:</p>"
                   % (_e(_num(nm["trees_beating_every_shuffle"])), _e(bound_text(beat_bound(nm)))))
        out.append(_table(["class", "observed", "shuffle mean", "ratio", "shuffles at or above"],
                          [[c, b["observed"], b["mean"], b["ratio"], b["shuffles_ge_observed"]]
                           for c, b in sorted(nm["by_class"].items())], numeric=(1, 2, 3, 4)))
    out.append("</section>")

    out.append("<section><h2>Parallel starts</h2>")
    f8 = s["F8"]
    out.append(_tiles([
        ("parallel starts", _num(f8["parallel_starts"])),
        ("linked adoptions", _num(facts["cascade"]["status"]["linked"])),
        ("trees with a parallel start", _num(facts["trees"]["with_parallel_starts"])),
        ("trees where the seed is not the first user", "%s (%s)" % (_num(f8["seed_is_not_first_user"]),
                                                                    _pct(f8["share_seed_is_not_first_user"]))),
    ]))
    out.append("<figure>%s</figure>" % svg_status(facts))
    out.append(_details("Table view: adopters by status", _table(
        ["status", "observed", "shuffle mean"],
        [[label, o, "n/a" if m is None else m] for label, o, m in status_rows(facts)], numeric=(1, 2))))
    out.append("</section>")

    flow = facts["scores"]["family_flow"]
    out.append("<section><h2>Page-family flow</h2>")
    out.append("<p>Named links within one page family: %s (shuffle mean %s); across families: %s (shuffle "
               "mean %s). Top pairs:</p>"
               % (_e(_num(flow["within_family"]["observed"])), _e(_num(flow["within_family"]["shuffle_mean"])),
                  _e(_num(flow["across_families"]["observed"])),
                  _e(_num(flow["across_families"]["shuffle_mean"]))))
    out.append(_table(["from family", "to family", "named links", "shuffle mean", "ratio"],
                      [[p["from"], p["to"], p["named_links"], p["shuffle_mean"], p["ratio"]]
                       for p in flow["top_pairs"][:15]], numeric=(2, 3, 4)))
    out.append("<h2>Wikis</h2>")
    out.append(_table(["wiki", "saves", "trees first used here", "named links from here",
                       "named links to another wiki"],
                      [[g["group"], g["rows"], g["trees_first_used_here"], g["named_links_from_here"],
                        g["named_links_to_other_group"]] for g in facts["groups"]], numeric=(1, 2, 3, 4)))
    out.append("</section>")

    risky = facts["risky"]
    vs = risky.get("named_links_vs_shuffle") or {}
    out.append("<section><h2>Risky-word types</h2>")
    out.append("<p>Trees whose token sits on a line with a risky word, by type name only. 'answer', 'share' "
               "and 'relay' are broad; the narrow types name a mechanism (%s).</p>"
               % _e(", ".join(risky["narrow_types"])))
    by_type = vs.get("by_type") or {}
    rows = []
    for t in risky["types"]:
        b = by_type.get(t) or {}
        rows.append([t, risky["trees_with_type"][t], risky["trees_with_type_and_named_links"][t],
                     risky["named_links_in_trees_with_type"][t], b.get("mean", "n/a"), b.get("ratio", "n/a")])
    out.append(_table(["type", "trees", "trees with named links", "named links", "shuffle mean", "ratio"],
                      rows, numeric=(1, 2, 3, 4, 5)))
    out.append("</section>")

    out.append("<section><h2>Top %d trees by largest branch</h2>" % TOP_TREES_TABLE)
    top = _linked(records)[:TOP_TREES_TABLE]
    out.append(_table(["token_id", "token", "class", "agents", "largest branch", "named links", "shuffle mean",
                       "shuffles at or above", "parallel starts", "top page family"],
                      [[r["token_id"], r["token"], r["cls"], r["agents"], r["largest_branch"], r["links_named"],
                        "n/a" if r["shuffle_links_mean"] is None else r["shuffle_links_mean"],
                        "n/a" if r["shuffle_ge_observed"] is None else r["shuffle_ge_observed"],
                        r["status"]["parallel"], r["families"][0][0] if r["families"] else ""]
                       for r in top], numeric=(3, 4, 5, 6, 7, 8)))
    out.append("</section>")

    out.append("<section><h2>Method and limits</h2><ul>")
    for item in METHOD_NOTES:
        out.append("<li>%s</li>" % _e(item))
    out.append("<li>%s</li>" % _e("Shuffle: %s; %d shuffles, seed %s." % (
        nm.get("method", "none"), nm.get("shuffles") or 0, nm.get("seed"))))
    out.append("</ul></section>")
    out.append("<footer><p>Local working view. Token text and agent labels on this page are escaped data from "
               "the run: review before sharing a screenshot. docs/lead.svg is the public figure (no agent "
               "labels, no token text).</p></footer>")
    out.append("</main></body></html>\n")
    return "".join(out)


METHOD_NOTES = (
    "A use is a token in a save's added lines (diff hunks), so text a page already held is not a new use.",
    "Agent B is linked to A when A used the token first on a page B later saved to, before B's own first use, "
    "with both saves' clock uncertainty cleared; credit goes to the most recent use B could have seen.",
    "A parallel start is a later user with no exposure to any earlier use of the token.",
    "A visit counts as exposure even if the token's text was deleted from the page before the visit.",
    "The row-level shuffle spreads an agent's saves over more pages and splits repeated uses of a token across "
    "agents, which makes more adopters and so more chances to link; the adopters and agent-page pairs rows "
    "show the size of that effect.",
    "Speeds count minutes to the k-th linked agent; very fast spreads can be burst editing.",
    "Risky-word flags are type names only and mark a token's lines, not intent.",
)


def _pct(x):
    return "n/a" if x is None else "%.1f%%" % (100.0 * x)


# ---------------------------------------------------------------- files


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def detail_text(detail):
    """Canonical top_trees.json text (sorted keys, ASCII)."""
    return json.dumps(detail, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"


def write_report(out_dir, facts, records, detail, lead=None):
    """Write report.html to out_dir. Returns its path."""
    path = os.path.join(out_dir, "report.html")
    _write_atomic(path, render_report(facts, records, detail, lead=lead))
    return path


def public_svg(detail, lead=None, facts=None):
    """The lead tree as a standalone SVG: summary data only (no agent labels, no token text).

    With ``facts`` the figure also says how the tree was picked and how many
    trees chance alone would let through (``selection_lines``).
    """
    tid = lead or detail["lead"]
    tree = _find(detail, tid)
    if tree is None:
        raise ValueError("no lead tree to draw" if tid is None else
                         "tree %s is not in top_trees.json" % short(tid, 20))
    rule = RULE_GIVEN if lead else detail["lead_rule"]
    return svg_tree(tree, detail["shuffles"], public=True, prefix="lead-", standalone=True,
                    note_lines=selection_lines(rule, facts))


def load_outputs(out_dir):
    """(facts, records, detail) from a run folder; refuses files that do not match facts.json."""
    with open(os.path.join(out_dir, "facts.json"), encoding="utf-8") as fh:
        facts = json.load(fh)
    with open(os.path.join(out_dir, "trees.jsonl"), encoding="utf-8") as fh:
        trees_text = fh.read()
    with open(os.path.join(out_dir, "top_trees.json"), encoding="utf-8") as fh:
        detail_raw = fh.read()
    outputs = facts.get("outputs", {})
    if _sha256_text(trees_text) != outputs.get("trees.jsonl", {}).get("sha256"):
        raise ValueError("trees.jsonl does not match facts.json: rerun cli.py")
    if _sha256_text(detail_raw) != outputs.get("top_trees.json", {}).get("sha256"):
        raise ValueError("top_trees.json does not match facts.json: rerun cli.py")
    records = [json.loads(line) for line in trees_text.splitlines() if line.strip()]
    return facts, records, json.loads(detail_raw)


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(
        prog="python -m patient_zero.report",
        description="Re-render report.html from a run folder, and optionally write the public lead figure.")
    ap.add_argument("--out", required=True, help="the run folder (facts.json, trees.jsonl, top_trees.json)")
    ap.add_argument("--lead", help="token_id of the tree to lead with (must be in top_trees.json)")
    ap.add_argument("--svg", help="also write the lead tree here as a standalone SVG (summary data only)")
    args = ap.parse_args(argv)
    try:
        facts, records, detail = load_outputs(args.out)
        path = write_report(args.out, facts, records, detail, lead=args.lead)
        print("wrote: %s" % path, file=out)
        if args.svg:
            if not args.svg.lower().endswith(".svg"):
                raise ValueError("--svg must name a .svg file")
            folder = os.path.dirname(os.path.abspath(args.svg))
            os.makedirs(folder, exist_ok=True)
            _write_atomic(args.svg, public_svg(detail, lead=args.lead, facts=facts))
            print("wrote: %s" % args.svg, file=out)
    except (OSError, ValueError) as exc:
        print("patient-zero report: ERROR: %s" % exc, file=out)
        return 2
    print("lead tree: %s (%s)" % (args.lead or detail["lead"], RULE_GIVEN if args.lead else detail["lead_rule"]),
          file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
