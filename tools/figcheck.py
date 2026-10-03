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

Exit codes: 0 PASS, 1 FAIL, 2 bad input, 3 UNVERIFIED (no render).
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
    ap.add_argument("--json", default="lead-checks.json")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--browser", help="path to chrome or msedge (default: look in the usual places)")
    args = ap.parse_args(argv)
    png_path = os.path.join(args.out_dir, args.png)
    json_path = os.path.join(args.out_dir, args.json)
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
