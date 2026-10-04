#!/usr/bin/env python3
"""Render check for the public lead figure (docs/lead.svg).

Python standard library only, plus a local Chrome or Edge run headless from
its command line (no window, a throwaway profile, nothing from the data is
opened). Three checks, written to a JSON file before the render (status
"pending") and filled in after it:

``title_present``
    The SVG has a non-empty <title> and a visible title text, and the title
    text really draws: the pixels in its box change when all text is hidden.
``axis_labels_present``
    The x and y axis label texts exist and draw, by the same text-hidden diff.
``legend_colours_match``
    Every page-family colour on a mark has a legend swatch and every swatch
    colour is used by a mark (in the SVG source); in the PNG, each swatch's
    centre pixel is its declared colour and that colour also appears in the
    plot area below the legend.

The PNG is low resolution (default half size) and is shown through an <img>
tag, the way a README shows an SVG. Nobody, person or model, has to look at
it: every check is computed from pixels. Without a browser the verdict is
UNVERIFIED and the source checks are still written.

    python tools/figcheck.py                      # docs/lead.svg -> ../out/
    python tools/figcheck.py --svg docs/lead.svg --out-dir ../out --scale 0.5

Data mode (``--sources``, for docs/can-it-see.svg): no browser, only
arithmetic on the SVG source and the JSON files it was drawn from. Every row
group (``class="cis-row"``) names its source file, JSON path and null id.
Eight checks (``DATA_PLANNED``): every number in every text equals its source
to the printed precision and every ``data-*`` value equals it exactly; every
row cites one null id, the id each source's null fields give, and the two
sources share the same real-data result under that null (``h0``); dots and
bands sit where the numbers put them on the one axis; position words follow
from the numbers; no z, sd or sigma and exactly one axis; printed hashes are
the SHA-256 prefixes of the named files; and the phone check, smallest font
px × 358 / canvas width ≥ 10 (the figure stays readable at 10 px on a phone
that shows 358 px); and the five rows are all there, in order, so no
inconvenient row can be dropped (only the decoy may wait for its numbers)::

    python tools/figcheck.py --svg docs/can-it-see.svg \\
        --sources ../out/calibration.json ../out/checks.json --out-dir ../out

Exit codes: 0 PASS, 1 FAIL, 2 bad input, 3 UNVERIFIED (no render; lead mode only).
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import zlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SVG_NS = "{http://www.w3.org/2000/svg}"
SURFACE = "#fcfcfb"
BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
PLANNED = {
    "title_present": "SVG <title> and visible title text are non-empty, and the title draws (text-hidden diff)",
    "axis_labels_present": "x and y axis label texts are non-empty, and both draw (text-hidden diff)",
    "legend_colours_match": "mark colours equal swatch colours in the source; each swatch's centre pixel is "
                            "its colour and the colour also appears in the plot area",
}
COLOUR_TOLERANCE = 18  # per channel, for anti-aliasing and colour management
DIFF_LEVEL = 48  # a pixel "changed" when some channel moves by more than this
MIN_CHANGED = 20  # changed pixels a drawn text box must show

PHONE_WIDTH = 358  # px a small phone shows
PHONE_MIN_PX = 10.0  # smallest text there
DEFAULT_FONT = 16.0  # an SVG text with no font-size anywhere up its tree
DATA_PLANNED = {
    "numbers_match_sources": "every number in every text equals its JSON source to the printed precision, every "
                             "data-* value equals it exactly, and each drawn row prints its observed, mean and ratio",
    "same_null": "every row cites one null id, the id its source's null fields give; the footer prints its tag "
                 "(first 12 hex of the id's SHA-256); calibration null.h0 equals checks order_null.links_named; "
                 "the real same-page and cross-page counts and means add up to null.h0",
    "geometry_matches": "each dot sits at observed / mean and each band spans p05 / mean to p95 / mean on the one "
                        "axis; tick labels sit at their values; one accent colour for dots, a grey band",
    "words_match": "each row's position words (above, inside or below the band, and seen / blind spot / not "
                   "flagged / wrongly flagged) follow from its numbers and its planted kind",
    "no_z": "no z, sd or sigma in any text and exactly one axis",
    "hashes_match": "every hash printed is the SHA-256 prefix of the source file named beside it, and every "
                    "source's hash is printed",
    "phone_check": "smallest font px x %d / canvas width >= %g" % (PHONE_WIDTH, PHONE_MIN_PX),
    "rows_complete": "the five rows are there in order (carry, same-page, decoy, real all, real cross-page); only the "
                     "decoy may be pending or left out, and only while its source has no numbers",
}
# Rows of docs/can-it-see.svg, in order: (key, source, JSON path, may be missing).
DATA_ROWS = (
    ("carry", "calibration.json", "planted.carry.all_links", False),
    ("same_page", "calibration.json", "planted.contagion.all_links", False),
    ("page_instruction", "calibration.json", "planted.page_instruction.all_links", True),
    ("real_all", "checks.json", "order_null.links_named", False),
    ("real_cross", "calibration.json", "real.cross_page", False),
)
GEOM_TOL = 0.11  # px: coordinates are written to 0.1
MIN_BAND = 2.0  # px: a narrower band is drawn this wide, centred
BAND_PERCENTILES = (5.0, 95.0)  # the band's definition (p05, p95), allowed in a text that says "percentile"
NUMBER_RE = re.compile(r"(?<![\d.,])\d[\d,]*(?:\.\d+)?")
HEX_RE = re.compile(r"(?<![0-9a-z])[0-9a-f]{12,64}(?![0-9a-z])")
Z_RE = re.compile(r"(?i)(?<![a-z])(?:z|sd|sigma|std|z-score|stdev)(?![a-z])|σ")
POSITION_WORDS = {
    "contact": {"above": "seen", "inside": "blind spot", "below": "reads the wrong way"},
    "decoy": {"above": "wrongly flagged", "inside": "not flagged", "below": "not flagged"},
    "real": {},
}
CELL_KEYS = ("observed", "mean", "p05", "p95")
SPLIT_TOL = 0.011  # two means rounded to 0.01 can miss their rounded sum by 0.01


# ---------------------------------------------------------------- PNG (8-bit RGB or RGBA, not interlaced)


def read_png(path):
    """(width, height, bytes per pixel, rows) of an 8-bit RGB or RGBA PNG."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos, idat, head = 8, [], None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            head = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if head is None:
        raise ValueError("PNG without a header")
    width, height, depth, colour, _comp, _filt, interlace = head
    if depth != 8 or interlace != 0 or colour not in (2, 6):
        raise ValueError("unsupported PNG (depth %d, colour type %d, interlace %d)" % (depth, colour, interlace))
    bpp = 3 if colour == 2 else 4
    raw = zlib.decompress(b"".join(idat))
    stride = width * bpp
    rows, prev, i = [], bytearray(stride), 0
    for _y in range(height):
        f = raw[i]
        line = bytearray(raw[i + 1:i + 1 + stride])
        i += 1 + stride
        if f == 1:
            for x in range(bpp, stride):
                line[x] = (line[x] + line[x - bpp]) & 255
        elif f == 2:
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 255
        elif f == 3:
            for x in range(stride):
                left = line[x - bpp] if x >= bpp else 0
                line[x] = (line[x] + ((left + prev[x]) >> 1)) & 255
        elif f == 4:
            for x in range(stride):
                a = line[x - bpp] if x >= bpp else 0
                b = prev[x]
                c = prev[x - bpp] if x >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[x] = (line[x] + pred) & 255
        elif f != 0:
            raise ValueError("bad PNG filter %d" % f)
        rows.append(bytes(line))
        prev = line
    return width, height, bpp, rows


def pixel(img, x, y):
    width, height, bpp, rows = img
    x = min(max(int(x), 0), width - 1)
    y = min(max(int(y), 0), height - 1)
    k = x * bpp
    return tuple(rows[y][k:k + 3])


def hex_rgb(colour):
    colour = colour.strip().lstrip("#")
    return tuple(int(colour[k:k + 2], 16) for k in (0, 2, 4))


def near(rgb, target, tol=COLOUR_TOLERANCE):
    return all(abs(a - b) <= tol for a, b in zip(rgb, target))


# ---------------------------------------------------------------- the SVG


def _tag(el):
    return el.tag[len(SVG_NS):] if el.tag.startswith(SVG_NS) else el.tag


def text_box(el):
    """Rough box (x0, y0, x1, y1) of a <text> in SVG units; handles rotate(-90 x y)."""
    x, y = float(el.get("x", 0)), float(el.get("y", 0))
    size = float(el.get("font-size", 12))
    width = len(el.text or "") * size * 0.58
    anchor = el.get("text-anchor", "start")
    start = {"middle": -width / 2.0, "end": -width}.get(anchor, 0.0)
    if "rotate(-90" in (el.get("transform") or ""):
        return (x - size * 0.9, y - (start + width), x + size * 0.3, y - start)
    return (x + start, y - size * 0.9, x + start + width, y + size * 0.3)


def parse_svg(path):
    """Facts the checks need from the SVG source."""
    root = ET.parse(path).getroot()
    if _tag(root) != "svg":
        raise ValueError("not an SVG")
    width, height = float(root.get("width")), float(root.get("height"))
    info = {"width": width, "height": height, "title_el": "", "texts": {}, "swatches": [], "mark_colours": set(),
            "legend_bottom": 0.0}
    for el in root.iter():
        tag = _tag(el)
        ident = el.get("id") or ""
        if tag == "title" and el is root.find(SVG_NS + "title"):
            info["title_el"] = (el.text or "").strip()
        if tag == "text" and ident:
            for name in ("title", "x-label", "y-label"):
                if ident.endswith(name) and not ident.endswith("title-el"):
                    info["texts"][name] = {"text": (el.text or "").strip(), "box": text_box(el)}
        cls = el.get("class") or ""
        if tag == "rect" and "legend-swatch" in cls:
            x, y = float(el.get("x")), float(el.get("y"))
            w, h = float(el.get("width")), float(el.get("height"))
            info["swatches"].append({"colour": el.get("fill").lower(), "box": (x, y, x + w, y + h)})
            info["legend_bottom"] = max(info["legend_bottom"], y + h)
        if "node" in cls.split():
            fill = (el.get("fill") or "").lower()
            colour = (el.get("stroke") or "").lower() if fill == SURFACE else fill
            info["mark_colours"].add(colour)
    return info


def source_checks(info):
    texts = info["texts"]
    swatch_colours = set(s["colour"] for s in info["swatches"])
    return {
        "title_present": {
            "svg_title_element": info["title_el"],
            "visible_title": texts.get("title", {}).get("text", ""),
            "pass": bool(info["title_el"]) and bool(texts.get("title", {}).get("text")),
        },
        "axis_labels_present": {
            "x_label": texts.get("x-label", {}).get("text", ""),
            "y_label": texts.get("y-label", {}).get("text", ""),
            "pass": bool(texts.get("x-label", {}).get("text")) and bool(texts.get("y-label", {}).get("text")),
        },
        "legend_colours_match": {
            "swatch_colours": sorted(swatch_colours),
            "mark_colours": sorted(info["mark_colours"]),
            "pass": bool(swatch_colours) and swatch_colours == info["mark_colours"],
        },
    }


# ---------------------------------------------------------------- rendering


def find_browser(given=None):
    for path in ([given] if given else []) + list(BROWSERS):
        if path and os.path.isfile(path):
            return path
    for name in ("chrome", "google-chrome", "chromium", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    return None


def render(browser, svg_path, png_path, width, height, scale, workdir):
    """Screenshot an <img> of the SVG at ``scale`` with a headless browser. Returns seconds taken."""
    shown_w, shown_h = int(round(width * scale)), int(round(height * scale))
    name = os.path.basename(svg_path)
    page = os.path.join(workdir, name + ".html")
    with open(page, "w", encoding="utf-8") as fh:
        fh.write('<!doctype html><html><head><meta charset="utf-8"></head>'
                 '<body style="margin:0;background:#ffffff"><img src="%s" width="%d" height="%d" '
                 'style="display:block"></body></html>' % (name, shown_w, shown_h))
    profile = os.path.join(workdir, "profile")
    cmd = [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
           "--no-default-browser-check", "--disable-extensions", "--disable-sync", "--mute-audio",
           "--force-device-scale-factor=1", "--user-data-dir=" + profile,
           "--window-size=%d,%d" % (shown_w, shown_h + 40), "--screenshot=" + png_path,
           pathlib.Path(page).resolve().as_uri()]
    kwargs = {"capture_output": True, "timeout": 120}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW: no console pops up
    t0 = time.perf_counter()
    subprocess.run(cmd, **kwargs)
    if not os.path.isfile(png_path):
        raise RuntimeError("the browser wrote no screenshot")
    return time.perf_counter() - t0


def hide_text(svg_text):
    """The same SVG with every text hidden (for the text-draws diff)."""
    return re.sub(r"(<svg\b[^>]*>)", r"\1<style>text{visibility:hidden}</style>", svg_text, count=1)


def changed_in_box(full, blank, box, scale):
    x0, y0, x1, y1 = [v * scale for v in box]
    count = 0
    for y in range(max(int(y0), 0), min(int(y1) + 1, full[1])):
        for x in range(max(int(x0), 0), min(int(x1) + 1, full[0])):
            a, b = pixel(full, x, y), pixel(blank, x, y)
            if max(abs(p - q) for p, q in zip(a, b)) > DIFF_LEVEL:
                count += 1
    return count


def colour_count(img, colour, y_from, scale):
    target = hex_rgb(colour)
    width, height = img[0], img[1]
    return sum(1 for y in range(int(y_from * scale), height) for x in range(width)
               if near(pixel(img, x, y), target))


def rendered_checks(info, full, blank, scale):
    texts = info["texts"]
    title_changed = changed_in_box(full, blank, texts["title"]["box"], scale) if "title" in texts else 0
    x_changed = changed_in_box(full, blank, texts["x-label"]["box"], scale) if "x-label" in texts else 0
    y_changed = changed_in_box(full, blank, texts["y-label"]["box"], scale) if "y-label" in texts else 0
    swatches = []
    for s in info["swatches"]:
        x0, y0, x1, y1 = s["box"]
        got = pixel(full, (x0 + x1) / 2.0 * scale, (y0 + y1) / 2.0 * scale)
        in_plot = colour_count(full, s["colour"], info["legend_bottom"] + 4, scale)
        swatches.append({"colour": s["colour"], "centre_pixel": "#%02x%02x%02x" % got,
                         "centre_matches": near(got, hex_rgb(s["colour"])), "pixels_in_plot": in_plot})
    background = hex_rgb(SURFACE)
    width, height = full[0], full[1]
    inked = sum(1 for y in range(height) for x in range(width)
                if not near(pixel(full, x, y), background, 6) and not near(pixel(full, x, y), (255, 255, 255), 6))
    return {
        "inked_pixels": inked,
        "title_present": {"changed_pixels": title_changed, "pass": title_changed >= MIN_CHANGED},
        "axis_labels_present": {"x_changed_pixels": x_changed, "y_changed_pixels": y_changed,
                                "pass": x_changed >= MIN_CHANGED and y_changed >= MIN_CHANGED},
        "legend_colours_match": {"swatches": swatches,
                                 "pass": bool(swatches) and all(s["centre_matches"] and s["pixels_in_plot"] > 0
                                                                for s in swatches)},
    }


# ---------------------------------------------------------------- data mode (--sources): arithmetic only


def _sha256_path(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def font_sizes(root):
    """Font size (px) of every <text>, taken from the element or the nearest ancestor that sets one."""
    sizes = []

    def walk(el, inherited):
        size = el.get("font-size")
        here = float(re.sub(r"px$", "", size)) if size else inherited
        if _tag(el) == "text":
            sizes.append(here)
        for child in el:
            walk(child, here)

    walk(root, DEFAULT_FONT)
    return sizes


def phone_check(root):
    width = float(root.get("width"))
    sizes = font_sizes(root)
    smallest = min(sizes) if sizes else DEFAULT_FONT
    px = smallest * PHONE_WIDTH / width
    return {"canvas_width": width, "smallest_font": smallest, "phone_px": round(px, 3),
            "rule": "smallest font x %d / canvas width >= %g" % (PHONE_WIDTH, PHONE_MIN_PX),
            "pass": px >= PHONE_MIN_PX}


def _null_block(doc):
    """(kind, block) of a source's null: calibration.json's 'null', or checks.json's 'order_null'."""
    if isinstance(doc.get("null"), dict):
        return doc["null"].get("kind"), doc["null"]
    if isinstance(doc.get("order_null"), dict):
        return "order", doc["order_null"]
    return None, None


def expected_null_id(doc):
    kind, block = _null_block(doc)
    if block is None or any(k not in block for k in ("scope", "shuffles", "seed", "min_lag_seconds",
                                                     "strict_exposure")):
        return None
    return "%s:%s:lag%s:%s:seed%s:n%s" % (kind, block["scope"], block["min_lag_seconds"],
                                          "strict" if block["strict_exposure"] else "default", block["seed"],
                                          block["shuffles"])


def _resolve(doc, dotted):
    for part in dotted.split("."):
        if not isinstance(doc, dict) or part not in doc:
            return None
        doc = doc[part]
    return doc


def _printed_matches(printed, value):
    """A printed number equals a source value to the printed precision."""
    digits = len(printed.split(".", 1)[1]) if "." in printed else 0
    return abs(float(printed.replace(",", "")) - float(value)) <= 0.5 * 10 ** (-digits) + 1e-9


def _texts(el):
    return [(t, "".join(t.itertext())) for t in el.iter() if _tag(t) == "text"]


def _role_expected(row, sources):
    path = row.get("data-path", "")
    if not path.startswith("planted."):
        return "real"
    kind = _resolve(sources.get(row.get("data-source")) or {}, ".".join(path.split(".")[:2])) or {}
    if "contact" not in kind:
        return None
    return "contact" if kind["contact"] else "decoy"


def data_checks(svg_path, source_paths):
    """The eight data checks (DATA_PLANNED) of a can-it-see figure against its JSON sources."""
    root = ET.parse(svg_path).getroot()
    if _tag(root) != "svg":
        raise ValueError("not an SVG")
    sources, hashes = {}, {}
    for path in source_paths:
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            sources[name] = json.load(fh)
        hashes[name] = _sha256_path(path)
    rows = [el for el in root.iter() if "cis-row" in (el.get("class") or "").split()]
    axes = [el for el in root.iter() if "cis-axis" in (el.get("class") or "").split()]
    problems = {name: [] for name in DATA_PLANNED}

    def bad(check, message):
        problems[check].append(message)

    # The one axis.
    if len(axes) != 1:
        bad("no_z", "%d axes (want exactly 1)" % len(axes))
    X = None
    if axes:
        ax = axes[0]
        lo, hi = float(ax.get("data-lo")), float(ax.get("data-hi"))
        x0, x1 = float(ax.get("data-x0")), float(ax.get("data-x1"))

        def X(v):
            return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    # The null every row must share.
    ids = {name: expected_null_id(doc) for name, doc in sources.items()}
    row_ids = set(r.get("data-null") for r in rows)
    if len(row_ids) != 1:
        bad("same_null", "rows cite %d different null ids" % len(row_ids))
    the_id = next(iter(row_ids)) if len(row_ids) == 1 else None
    cal, chk = sources.get("calibration.json"), sources.get("checks.json")
    if cal is not None and chk is not None:
        h0 = (cal.get("null") or {}).get("h0") or {}
        real = (chk.get("order_null") or {}).get("links_named") or {}
        if not h0 or any(h0.get(k) != real.get(k) for k in ("observed", "mean", "sd")):
            bad("same_null", "calibration null.h0 differs from checks order_null.links_named")
        split = [(cal.get("real") or {}).get(k) for k in ("same_page", "cross_page")]
        if h0 and all(split) and (split[0].get("observed", 0) + split[1].get("observed", 0) != h0.get("observed")
                                  or abs(split[0].get("mean", 0) + split[1].get("mean", 0) - h0.get("mean", 0))
                                  > SPLIT_TOL):
            bad("same_null", "calibration real same_page + cross_page do not add up to null.h0")
    shuffles = set()
    for doc in sources.values():
        _kind, block = _null_block(doc)
        if block is not None and "shuffles" in block:
            shuffles.add(block["shuffles"])

    drawn, dots, bands = 0, [], []
    for row in rows:
        key = row.get("data-row") or "?"
        src_name, dotted = row.get("data-source"), row.get("data-path") or ""
        if src_name not in sources:
            bad("numbers_match_sources", "row %s: source %s not given" % (key, src_name))
            continue
        if ids.get(src_name) is None:
            bad("same_null", "row %s: %s has no null fields" % (key, src_name))
        elif row.get("data-null") != ids[src_name]:
            bad("same_null", "row %s: cites %s, its source gives %s" % (key, row.get("data-null"), ids[src_name]))
        cell = _resolve(sources[src_name], dotted)
        texts = _texts(row)
        if row.get("data-status") == "pending":
            if cell is not None:
                bad("numbers_match_sources", "row %s: drawn as pending but %s has %s" % (key, src_name, dotted))
            if any(_tag(el) in ("circle", "rect") for el in row.iter()):
                bad("geometry_matches", "row %s: a pending row draws a dot or band" % key)
            for _t, s in texts:
                if NUMBER_RE.search(s):
                    bad("numbers_match_sources", "row %s: a pending row prints a number" % key)
            continue
        if not isinstance(cell, dict) or any(not isinstance(cell.get(k), (int, float)) for k in CELL_KEYS):
            bad("numbers_match_sources", "row %s: %s has no full cell at %s" % (key, src_name, dotted))
            continue
        drawn += 1
        for k in CELL_KEYS:
            got = row.get("data-" + k)
            if got is None or float(got) != float(cell[k]):
                bad("numbers_match_sources", "row %s: data-%s %s != %s" % (key, k, got, cell[k]))
        obs, mean, p05, p95 = (float(cell[k]) for k in CELL_KEYS)
        ratio = obs / mean
        # Every number printed in the row is its observed, mean or ratio; each of the three is printed.
        seen = set()
        for _t, s in texts:
            for m in NUMBER_RE.finditer(s):
                hit = [name for name, v in (("observed", obs), ("mean", mean), ("ratio", ratio))
                       if _printed_matches(m.group(0), v)]
                if not hit:
                    bad("numbers_match_sources", "row %s: printed %s matches none of observed %s, mean %s, "
                                                 "ratio %.6f" % (key, m.group(0), cell["observed"], cell["mean"],
                                                                 ratio))
                seen.update(hit)
        for name in ("observed", "mean", "ratio"):
            if name not in seen:
                bad("numbers_match_sources", "row %s: %s is not printed" % (key, name))
        # Words.
        position = "above" if obs > p95 else ("below" if obs < p05 else "inside")
        role = _role_expected(row, sources)
        if role is None:
            bad("words_match", "row %s: %s has no contact flag for its planted kind" % (key, src_name))
        elif row.get("data-role") != role:
            bad("words_match", "row %s: role %s, source says %s" % (key, row.get("data-role"), role))
        where_texts = [s for t, s in texts if "cis-where" in (t.get("class") or "")]
        wanted = "%s the band" % position
        word = POSITION_WORDS.get(role or "real", {}).get(position)
        if not where_texts or wanted not in where_texts[0]:
            bad("words_match", "row %s: position should read '%s'" % (key, wanted))
        elif word and not where_texts[0].endswith(word):
            bad("words_match", "row %s: should end '%s'" % (key, word))
        elif not word and not where_texts[0].endswith(wanted):
            bad("words_match", "row %s: should end '%s'" % (key, wanted))
        for other in ("above the band", "inside the band", "below the band"):
            if other != wanted and any(other in s for s in where_texts):
                bad("words_match", "row %s: also says '%s'" % (key, other))
        # Geometry on the one axis.
        circles = [el for el in row.iter() if "cis-dot" in (el.get("class") or "").split()]
        rects = [el for el in row.iter() if "cis-band" in (el.get("class") or "").split()]
        if len(circles) != 1 or len(rects) != 1:
            bad("geometry_matches", "row %s: %d dots and %d bands (want 1 and 1)" % (key, len(circles), len(rects)))
        elif X is not None:
            dot, band = circles[0], rects[0]
            dots.append(dot)
            bands.append(band)
            if abs(float(dot.get("cx")) - X(ratio)) > GEOM_TOL:
                bad("geometry_matches", "row %s: dot at %s, numbers put it at %.2f" % (key, dot.get("cx"), X(ratio)))
            left, right = X(p05 / mean), X(p95 / mean)
            if right - left < MIN_BAND:
                mid = (left + right) / 2.0
                left, right = mid - MIN_BAND / 2.0, mid + MIN_BAND / 2.0
            bx, bw = float(band.get("x")), float(band.get("width"))
            if abs(bx - left) > GEOM_TOL or abs(bx + bw - right) > 2 * GEOM_TOL:
                bad("geometry_matches", "row %s: band %.1f-%.1f, numbers give %.2f-%.2f"
                    % (key, bx, bx + bw, left, right))
            if not (x0 - GEOM_TOL <= float(dot.get("cx")) <= x1 + GEOM_TOL and x0 - GEOM_TOL <= bx
                    and bx + bw <= x1 + 2 * GEOM_TOL):
                bad("geometry_matches", "row %s: drawn outside the axis" % key)
    if drawn == 0:
        bad("numbers_match_sources", "no drawn row")
    if dots:
        fills = set((d.get("fill") or "").lower() for d in dots)
        band_fills = set((b.get("fill") or "").lower() for b in bands)
        if len(fills) != 1:
            bad("geometry_matches", "dots use %d colours (want one accent)" % len(fills))
        if fills & band_fills:
            bad("geometry_matches", "a band has the accent colour")
        for fill in band_fills:
            r, g, b = hex_rgb(fill)
            if max(r, g, b) - min(r, g, b) > 24:
                bad("geometry_matches", "band colour %s is not grey" % fill)

    # Every text outside the rows: the null id, hashes, ticks, the shuffle count, the band's definition.
    in_rows = set(t for row in rows for t in row.iter())  # row texts were checked against their own row
    printed_hashes = set()
    title = root.find(SVG_NS + "title")
    outside = [(t, "".join(t.itertext())) for t in root.iter() if _tag(t) == "text" and t not in in_rows]
    if title is not None:
        outside.append((title, title.text or ""))
    all_text = [s for _t, s in _texts(root)] + ([title.text or ""] if title is not None else [])
    id_printed = False
    tag = "null id " + hashlib.sha256(the_id.encode("utf-8")).hexdigest()[:12] if the_id else None
    seeds = set()
    for doc in sources.values():
        _kind, block = _null_block(doc)
        if block is not None and "seed" in block:
            seeds.add(block["seed"])
    for el, s in outside:
        if tag and tag in s:
            id_printed = True
            s = s.replace(tag, " ")
        rest = s
        for m in HEX_RE.finditer(s):
            named = [n for n in hashes if n in s]
            if len(named) == 1 and hashes[named[0]].startswith(m.group(0)) and "sha256 " + m.group(0) in s:
                printed_hashes.add(named[0])
            else:
                bad("hashes_match", "hash %s... matches no source named beside it" % m.group(0)[:8])
            # A hash is judged here, not as a number; the algorithm's name goes with it.
            rest = rest.replace("sha256 " + m.group(0), " ").replace(m.group(0), " ")
        is_tick = "cis-tick-label" in (el.get("class") or "")
        for m in NUMBER_RE.finditer(rest):
            value = float(m.group(0).replace(",", ""))
            if is_tick and X is not None:
                if abs(float(el.get("x")) - X(value)) > GEOM_TOL:
                    bad("geometry_matches", "tick label %s at x %s, the axis puts it at %.2f"
                        % (m.group(0), el.get("x"), X(value)))
                continue
            if value in shuffles and "shuffles" in rest:
                continue
            if value in seeds and "seed" in rest:
                continue
            if value in BAND_PERCENTILES and "percentile" in rest:
                continue
            bad("numbers_match_sources", "unexplained number %s outside the rows" % m.group(0))
    if the_id and not id_printed:
        bad("same_null", "the null tag is not printed")
    for name in hashes:
        if name not in printed_hashes:
            bad("hashes_match", "%s's hash is not printed" % name)
    for s in all_text:
        if Z_RE.search(s):
            bad("no_z", "a text names z, sd or sigma")
    for el in root.iter():
        if any(k.startswith("data-z") or k == "data-sd" for k in el.attrib):
            bad("no_z", "a data-z or data-sd attribute")
    keys = [r.get("data-row") for r in rows]
    wanted = [k for k, _s, _p, _o in DATA_ROWS]
    if [k for k in wanted if k in keys] != keys:
        bad("rows_complete", "rows %s are not the expected rows in order" % ",".join(str(k) for k in keys))
    by_key = dict((r.get("data-row"), r) for r in rows)
    for key, src_name, dotted, optional in DATA_ROWS:
        row = by_key.get(key)
        if row is not None and (row.get("data-source"), row.get("data-path")) != (src_name, dotted):
            bad("rows_complete", "row %s reads %s %s, not %s %s" % (key, row.get("data-source"),
                                                                    row.get("data-path"), src_name, dotted))
        has_numbers = src_name in sources and _resolve(sources[src_name], dotted) is not None
        if row is None and (not optional or has_numbers):
            bad("rows_complete", "row %s is missing%s" % (key, " although %s has its numbers" % src_name
                                                         if optional else ""))
        if row is not None and row.get("data-status") != "drawn" and not optional:
            bad("rows_complete", "row %s must be drawn" % key)
    phone = phone_check(root)
    if not phone["pass"]:
        bad("phone_check", "phone px %.2f < %g" % (phone["phone_px"], PHONE_MIN_PX))
    checks = {name: {"pass": not problems[name], "problems": problems[name][:20]} for name in DATA_PLANNED}
    checks["phone_check"].update(phone)
    checks["same_null"]["null_id"] = the_id
    checks["numbers_match_sources"]["rows_drawn"] = drawn
    checks["numbers_match_sources"]["rows_pending"] = sum(1 for r in rows if r.get("data-status") == "pending")
    return {"checks": checks, "source_sha256": hashes}


def data_main(args, out):
    """--sources: arithmetic checks only, no browser."""
    json_path = os.path.join(args.out_dir, args.json or (os.path.splitext(os.path.basename(args.svg))[0]
                                                         + "-checks.json"))
    try:
        with open(args.svg, "r", encoding="utf-8") as fh:
            svg_text = fh.read()
        result = data_checks(args.svg, args.sources)
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        print("figcheck: ERROR: %s" % exc, file=out)
        return 2
    os.makedirs(args.out_dir, exist_ok=True)
    checks = result["checks"]
    report = {"tool": "tools/figcheck.py --sources", "figure": os.path.basename(args.svg),
              "figure_sha256": hashlib.sha256(svg_text.encode("utf-8")).hexdigest(),
              "sources": {os.path.basename(p): result["source_sha256"][os.path.basename(p)] for p in args.sources},
              "planned_checks": DATA_PLANNED, "checks": checks,
              "viewer": "mechanical: arithmetic on the SVG source and the JSON sources; no browser, nobody looked",
              "status": "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"}
    _write_json(json_path, report)
    print("figcheck: %s (%s)" % (report["status"], ", ".join(
        "%s=%s" % (k, "ok" if c["pass"] else "FAIL") for k, c in checks.items())), file=out)
    for name, c in checks.items():
        for p in c["problems"]:
            print("  %s: %s" % (name, p), file=out)
    print("phone px %.2f (smallest font %g at width %g); null id %s; rows drawn %d, pending %d"
          % (checks["phone_check"]["phone_px"], checks["phone_check"]["smallest_font"],
             checks["phone_check"]["canvas_width"], checks["same_null"]["null_id"],
             checks["numbers_match_sources"]["rows_drawn"], checks["numbers_match_sources"]["rows_pending"]),
          file=out)
    print("wrote: %s" % json_path, file=out)
    return 0 if report["status"] == "PASS" else 1


# ---------------------------------------------------------------- main


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="Render check for the public lead figure.")
    ap.add_argument("--svg", default=os.path.join(REPO, "docs", "lead.svg"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(REPO), "out"))
    ap.add_argument("--png", default="lead-check.png")
    ap.add_argument("--json", default=None, help="report name (default lead-checks.json, or <figure>-checks.json "
                                                 "with --sources)")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--browser", help="path to chrome or msedge (default: look in the usual places)")
    ap.add_argument("--sources", nargs="+", metavar="JSON",
                    help="data mode: the JSON files the figure was drawn from (no browser; arithmetic only)")
    args = ap.parse_args(argv)
    if args.sources:
        return data_main(args, out)
    png_path = os.path.join(args.out_dir, args.png)
    json_path = os.path.join(args.out_dir, args.json or "lead-checks.json")
    try:
        with open(args.svg, "r", encoding="utf-8") as fh:
            svg_text = fh.read()
        info = parse_svg(args.svg)
    except (OSError, ValueError, ET.ParseError) as exc:
        print("figcheck: ERROR: %s" % exc, file=out)
        return 2
    os.makedirs(args.out_dir, exist_ok=True)
    report = {
        "tool": "tools/figcheck.py",
        "figure": os.path.basename(args.svg),
        "figure_sha256": hashlib.sha256(svg_text.encode("utf-8")).hexdigest(),
        "planned_checks": PLANNED,
        "status": "pending",
        "viewer": "mechanical: every check is computed from the SVG source and the PNG pixels; nobody looked at "
                  "the PNG",
    }
    _write_json(json_path, report)  # the checks exist before the render
    src = source_checks(info)
    report["source"] = src
    browser = find_browser(args.browser)
    rendered, error, seconds = None, None, None
    if browser is None:
        error = "no Chrome or Edge found"
    else:
        work = tempfile.mkdtemp(prefix="pz-figcheck-")
        try:
            shutil.copyfile(args.svg, os.path.join(work, "lead.svg"))
            with open(os.path.join(work, "lead-notext.svg"), "w", encoding="utf-8") as fh:
                fh.write(hide_text(svg_text))
            if os.path.exists(png_path):
                os.remove(png_path)
            seconds = render(browser, os.path.join(work, "lead.svg"), png_path, info["width"], info["height"],
                             args.scale, work)
            blank_png = os.path.join(work, "notext.png")
            render(browser, os.path.join(work, "lead-notext.svg"), blank_png, info["width"], info["height"],
                   args.scale, work)
            full, blank = read_png(png_path), read_png(blank_png)
            rendered = rendered_checks(info, full, blank, args.scale)
            report["render"] = {"browser": os.path.basename(browser), "png": args.png,
                                "png_size": [full[0], full[1]], "scale": args.scale,
                                "seconds": round(seconds, 1), "inked_pixels": rendered["inked_pixels"]}
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, zlib.error) as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    checks = {}
    for name in PLANNED:
        entry = {"source": src[name]}
        if rendered is not None:
            entry["rendered"] = rendered[name]
            entry["pass"] = src[name]["pass"] and rendered[name]["pass"]
        else:
            entry["pass"] = None
        checks[name] = entry
    report["checks"] = checks
    if rendered is None:
        report["status"] = "UNVERIFIED"
        report["render_error"] = error
        code = 3
    elif rendered["inked_pixels"] < 500:
        report["status"] = "FAIL"
        report["render_error"] = "the render is nearly blank"
        code = 1
    else:
        report["status"] = "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"
        code = 0 if report["status"] == "PASS" else 1
    _write_json(json_path, report)
    print("figcheck: %s (%s)" % (report["status"], ", ".join(
        "%s=%s" % (k, {True: "ok", False: "FAIL", None: "unverified"}[c["pass"]]) for k, c in checks.items())),
        file=out)
    print("wrote: %s%s" % (json_path, (" and " + png_path) if rendered is not None else ""), file=out)
    return code


if __name__ == "__main__":
    sys.exit(main())
