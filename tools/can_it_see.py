#!/usr/bin/env python3
"""Can it see? The whole self-check in one command, on a made-up log or on your own.

Python standard library only. Deterministic: one fixed seed writes the made-up log and one fixed seed drives
every shuffle. Counts only.

    python tools/can_it_see.py --demo
    python tools/can_it_see.py --input LOG.ndjson --out out/mylog

It runs the code that made ``docs/can-it-see.svg``, each step exactly as its own command would, in this order:

1. ``python -m patient_zero.checks``     the order null on the log's own tokens;
2. ``python tools/calibrate.py``         known kinds planted on the log's own rows, scored under every null;
3. ``python -m patient_zero.crossgrid``  cross-page links and the random-row controls, same shuffles;
4. ``python tools/make_calibration.py``  joins 2 and 3, and refuses when they used different shuffles;
5. ``python -m patient_zero.canisee``    the card and the figure, with the log called "this log".

Then it prints the card: one line per row with the count, the order-null mean, their ratio and where the
count sits against the null's 5th to 95th percentile band. Three rows are planted, so their answers are
known: carried spread and same-page spread are real contact, and the page-instruction decoy has none. The
words are the figure's own: ``seen``, ``blind spot``, ``not flagged``.

``--demo`` needs no data. It first writes a made-up log into the output folder (default
``out/can-it-see-demo``): ``made_up_log`` below, plain words by invented agents on invented boards, every row
in the group ``made-up``. No spread is planted in that log itself: who writes, where, when and which words
are independent draws from one seed. The known kinds are planted on its rows by step 2, the way they are
planted on the wiki's rows for the published figure. ``tests/test_can_it_see.py`` pins the log's SHA-256 and
the three planted verdicts.

``--input`` takes your own log in the README's JSON Lines format, read by the generic adapter under its own
field names (this command has no ``--map`` and no ``--given``). A small log cannot hold every planted kind:
a row that could not be planted prints ``not run yet``, and when carried or same-page spread cannot be
planted at all the run stops with one sentence.

Exit codes: 0 the card was printed, 2 a step stopped (its own message is printed) or the setup is wrong.
"""

import argparse
import hashlib
import io
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _path in (HERE, ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import calibrate as calibrate_tool  # noqa: E402  (tools/calibrate.py)
import make_calibration  # noqa: E402  (tools/make_calibration.py)
from patient_zero import canisee, checks, crossgrid  # noqa: E402
from patient_zero import nullmodel as N  # noqa: E402
from patient_zero.safe import guard_out  # noqa: E402

SUBJECT = "this log"  # never "real wiki": that name belongs to the published figure alone
OUTPUTS = ("checks.json", "calibrate.json", "crossgrid.json", "calibration.json", "can-it-see.svg")

# The made-up demo log. Change any of these and the pinned hash in tests/test_can_it_see.py must change too.
DEMO_SEED = 20261004
DEMO_ROWS = 600
DEMO_AGENTS = 36
DEMO_PLACES = 24
DEMO_DAYS = 3
DEMO_NAME = "made_up_board.ndjson"
DEMO_OUT = os.path.join(ROOT, "out", "can-it-see-demo")
DEMO_START = 1768003200  # 2026-01-10T00:00:00Z, POSIX seconds
DEMO_NOTE_SHARE = 0.25  # share of rows that also name one of the made-up note files
WORDS = (
    "amber", "anchor", "aspen", "basalt", "birch", "bramble", "canyon", "cedar", "cinder", "clover", "cobalt",
    "copper", "coral", "cypress", "delta", "drift", "ember", "fennel", "fern", "flint", "garnet", "glacier",
    "granite", "grove", "harbor", "hazel", "heron", "hollow", "indigo", "iris", "jasper", "juniper", "kelp",
    "kestrel", "lagoon", "larch", "ledger", "lichen", "linden", "maple", "marble", "meadow", "moss", "nectar",
    "nettle", "onyx", "orchid", "osprey", "pebble", "pewter", "plume", "quartz", "quill", "raven", "ridge",
    "river", "saffron", "sage", "spruce", "thistle", "timber", "umber", "violet", "willow",
)
NOTES = tuple("notes-%02d.md" % i for i in range(1, 13))

STEPS = ("order null on the log's own tokens", "known kinds planted on the log's own rows",
         "cross-page links and random-row controls", "one calibration file from steps 2 and 3",
         "the card and the figure")


def made_up_log(seed=DEMO_SEED, rows=DEMO_ROWS, agents=DEMO_AGENTS, places=DEMO_PLACES, days=DEMO_DAYS,
                note_share=DEMO_NOTE_SHARE):
    """The made-up log as bytes (JSON Lines, LF). Every draw is ``random.Random(seed).random()`` alone.

    Each row: a time drawn evenly over ``days`` days, an agent and a board drawn evenly, 8 to 13 words drawn
    evenly from WORDS, and for ``note_share`` of the rows one of the NOTES file names. Nothing depends on any
    earlier row, so the log holds no spread by construction.
    """
    rng = random.Random(seed)

    def pick(n):
        return int(rng.random() * n)

    times = sorted(DEMO_START + pick(days * 86400) for _ in range(rows))
    lines = []
    for i, t in enumerate(times):
        words = [WORDS[pick(len(WORDS))] for _ in range(8 + pick(6))]
        if rng.random() < note_share:
            words.insert(pick(len(words) + 1), NOTES[pick(len(NOTES))])
        lines.append(json.dumps({"id": "m%04d" % i, "agent": "Agent%02d" % pick(agents), "time": t,
                                 "place": "board/p%02d" % pick(places), "text": " ".join(words),
                                 "group": "made-up", "uncertainty_seconds": 1}, ensure_ascii=True))
    return ("\n".join(lines) + "\n").encode("ascii")


def _step(number, func, argv, show):
    """Run one step's own ``main`` with its output captured. Returns (exit code, captured text)."""
    buf = io.StringIO()
    started = time.perf_counter()
    try:
        code = func(argv, out=buf)
    except SystemExit as exc:  # argparse, or a step that refuses its input
        code = exc.code if isinstance(exc.code, int) else 2
    except Exception as exc:  # a step met a log it cannot score: one sentence, never a traceback
        code = 2
        print("ERROR: %s (%s)" % (exc, type(exc).__name__), file=buf)
    show("[%d/%d] %s: %s (%.1f s)" % (number, len(STEPS), STEPS[number - 1], "done" if code == 0 else "STOPPED",
                                     time.perf_counter() - started))
    return code, buf.getvalue()


def run(log_path, out_dir, shuffles=N.DEFAULT_SHUFFLES, seed=N.DEFAULT_SEED, out=None):
    """The five steps on one generic log. Returns (exit code, card lines); prints progress and the card."""
    out = out or sys.stdout

    def show(text):
        print(text, file=out, flush=True)

    def path(name):
        return os.path.join(out_dir, name)

    null = ["--shuffles", str(shuffles), "--seed", str(seed)]
    log = ["--adapter", "generic", "--input", log_path]
    chk, cal, grid, joined, svg = (path(name) for name in OUTPUTS)
    plan = (
        (checks.main, log + null + ["--out", chk]),
        (calibrate_tool.main, log + null + ["--out", cal]),
        (crossgrid.main, log + null + ["--out", grid]),
        (make_calibration.main, ["--calibrate", cal, "--crossgrid", grid, "--checks", chk, "--out", joined]),
        (canisee.main, ["--calibration", joined, "--checks", chk, "--svg", svg, "--subject", SUBJECT]),
    )
    text = ""
    for number, (func, argv) in enumerate(plan, 1):
        code, text = _step(number, func, argv, show)
        if code != 0:
            errors = [line for line in text.splitlines() if "ERROR" in line]
            for line in errors or text.splitlines()[-3:]:
                show("  " + line)
            why = ("This command reads the README's field names as they are: it has no --map." if number == 1
                   else "A log with too few rows, places or writers cannot hold the planted kinds.")
            show("can_it_see: stopped at step %d of %d; no card. %s" % (number, len(STEPS), why))
            return 2, []
    card = [line for line in text.splitlines() if not line.startswith(("null id:", "wrote:"))]
    show("")
    show("Can it see? Did it see? (%s; order null, seed %d, %d shuffles)" % (SUBJECT, seed, shuffles))
    for line in card:
        show("  " + line)
    if any(line.endswith(": not run yet") for line in card):
        show("A row that says 'not run yet' could not be planted on this log: too few rows, places or writers.")
    show("Each row: count vs the order-null mean, their ratio, and where the count sits against the null's "
         "5th to 95th percentile band.")
    show("The planted rows have known answers: carried and same-page spread are real contact; the decoy has none.")
    show("wrote: %s" % out_dir)
    return 0, card


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="Can it see? Plant known kinds of spread on a log's own rows and "
                                             "print what the order null sees, what it misses and what it "
                                             "does not flag (counts only).")
    ap.add_argument("--demo", action="store_true",
                    help="run on a made-up log written by this command (no data needed)")
    ap.add_argument("--input", help="your own log: one JSON Lines file in the README's format (generic adapter)")
    ap.add_argument("--out", help="output folder: outside the repo, or inside its git-ignored out/ "
                                  "(default with --demo: out/can-it-see-demo)")
    ap.add_argument("--shuffles", type=int, default=N.DEFAULT_SHUFFLES,
                    help="order-null shuffles (default %d, as in the published figure)" % N.DEFAULT_SHUFFLES)
    ap.add_argument("--seed", type=int, default=N.DEFAULT_SEED,
                    help="seed for the shuffles (default %d, as in the published figure)" % N.DEFAULT_SEED)
    args = ap.parse_args(argv)
    try:
        if args.demo == bool(args.input):
            raise ValueError("give --demo, or --input LOG with --out DIR (one of the two)")
        if args.input and not args.out:
            raise ValueError("--input needs --out DIR (outside the repo, or inside its out/ folder)")
        if args.shuffles < 2:
            raise ValueError("--shuffles must be 2 or more")
        out_dir = os.path.abspath(args.out or DEMO_OUT)
        from patient_zero.pipeline import check_out_dir  # late import: pipeline imports report
        check_out_dir(out_dir)
        log_path = args.input
        if args.demo:
            data = made_up_log()
            os.makedirs(out_dir, exist_ok=True)
            log_path = os.path.join(out_dir, DEMO_NAME)
            with open(log_path, "wb") as fh:
                fh.write(data)
            print("made-up log: %s rows of plain words, %d invented agents, %d invented boards, %d days "
                  "(seed %d, sha256 %s); no spread is planted in it"
                  % ("{:,}".format(DEMO_ROWS), DEMO_AGENTS, DEMO_PLACES, DEMO_DAYS, DEMO_SEED,
                     hashlib.sha256(data).hexdigest()[:16]), file=out)
        else:
            if not os.path.isfile(log_path):
                raise ValueError("--input names %s, which is not a file" % log_path)
            for name in OUTPUTS:  # no step may write over the log it reads
                guard_out(os.path.join(out_dir, name), [log_path], "--out")
    except (OSError, ValueError) as exc:
        print("can_it_see: ERROR: %s" % exc, file=out)
        return 2
    code, _card = run(log_path, out_dir, args.shuffles, args.seed, out=out)
    return code


if __name__ == "__main__":
    sys.exit(main())
