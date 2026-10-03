# patient-zero

*Code docs drafted by Claude (Anthropic). The findings and the story are in the write-up, `WRITEUP.md`, which is the author's own text.*

Give patient-zero a multi-agent log (who wrote what, when and where). It finds the tokens that several agents used, such as hosts, file names, code words, naming conventions and rare phrases. For each token it names the first user, links each later user to the most recent earlier use they could have seen, and draws the spread tree. Then it counts how many of those links a random relabelling of agents would also produce. It is Python standard library only and fully deterministic: no packages, no API keys, no network.

![Lead spread tree from a run on the German wiki export](docs/lead.svg)

*`docs/lead.svg`: one tree from a run on the German wiki export. It holds summary data only: no agent names and no token text. The figure itself says how the tree was picked and how many trees chance alone would let through.*

## Run it in two commands (no data needed)

```
python -m unittest
python cli.py --adapter generic --input examples/fake_log.ndjson --out out/example
```

Then open `out/example/report.html` in a browser. It works offline, with no scripts and no external files. Tested on Python 3.12, on Windows. The example log is **made up**: 14 lines, 5 invented agents and one human speaker.

## Run it on the German wiki data

The data is **not** in this repo and never will be. It is the public export of the German wiki incident from [collusion.wiki](https://collusion.wiki/explorer/download), with no login needed. The site shows no licence, so this repo does not republish any of it.

1. Download the export and unpack it into a folder **outside** this repo. The loader needs `revisions.jsonl.gz` and `pages.jsonl.gz`, plus `labels.jsonl.gz` for agent names.
2. Run:

```
python cli.py --adapter wiki --input PATH/TO/collusion --out out/wiki
```

A full run takes 1 to 5 minutes on one laptop. We measured 45 s to 299 s depending on machine load, against a 300 s target. `facts.json` records the SHA-256 of every input file, so two runs can be compared exactly. Runs are deterministic: the same input and seed give the same `core_sha256`.

## What a run writes

`--out` must be outside the repo, or inside the repo's git-ignored `out/` folder.

| File | What it holds |
|---|---|
| `facts.json` | Counts only. Input names, sizes and hashes; parameters; loader, token and tree counts; the shuffle baseline; scores; run times; `core_sha256` |
| `trees.jsonl` | One summary line per spread tree, biggest first. Agent labels are escaped and cut to 80 characters; no row text |
| `top_trees.json` | Node-level detail for the lead tree, the largest tree and the top trees |
| `report.html` | The offline report. Under the key numbers it gives a plain-words verdict against the shuffle: above, within or below chance |

`report.html` is a **local working view**: it shows escaped agent labels and token text, so review it before sharing a screenshot. `docs/lead.svg` is the public figure. To change the lead tree or redraw the figure without re-running:

```
python -m patient_zero.report --out out/wiki --lead TOKEN_ID --svg docs/lead.svg
```

## Your own log (generic adapter)

Write JSON Lines (`.jsonl` or `.ndjson`, gzipped or not), one event per line:

```
{"id": "e01", "agent": "Wren", "time": "2026-01-10T09:00:00Z", "place": "board/plans", "text": "..."}
```

- **Required:** `agent`, `time` (an ISO-8601 string, where no zone means UTC, or POSIX seconds), `place` (a page, room or channel) and `text`.
- **Optional:** `id`, `group`, `family` (used for colours and the flow table), `uncertainty_seconds`, `time_grade`, and `human: true` for a human speaker.

Human speakers and blank labels are exposure-only: others can see what they wrote, but they are never named and never credited. If you have a shared goal or prompt text, pass it with `--given FILE`: tokens from it are dropped, because they were handed down, not spread. `python cli.py --help` lists every option.

## How it works

- **Load** (`patient_zero/load.py`). For the wiki, a save's text is only what it **added**: the new side of its diff hunks, so text a page already held is never counted as a new use. Fields are read from a whitelist, so the half-IP field never exists in memory. IP-shaped strings are masked.
- **Tokens** (`tokens.py`). There are six classes: hosts, file names, commands, code words, coordination conventions (naming prefixes, mailbox names and all-caps control words) and rare 3-word phrases. Common web words are stop-listed, and template phrases above `--max-phrase-share` of rows are dropped. A token is kept only if `--min-agents` (default 3) distinct agents used it.
- **Links** (`cascade.py`). Agent B is linked to agent A when A used the token first, on a page B saved to later, before B's own first use. Both saves' clock uncertainty must be cleared. Credit goes to the most recent use B could have seen. A **parallel start** is a later user with no exposure to any earlier use. The **effective seed** is the earliest user with at least one adopter, which is not always the first user.
- **Chance baseline** (`nullmodel.py`). Agent labels are permuted among the agent rows of each UTC day, 100 times with one fixed seed (default 20261003). The token set and the link rules stay the same. The run reports observed against expected links, overall, per class and per tree.
- **Scores** (`score.py`). These cover influence (adopters credited to each agent), speed (minutes from the seed to the 3rd and 10th linked agent), page-family flow, and risky-word types. Risky words are reported by type name only and mark a token's lines, not intent.
- **Report** (`report.py`). This writes `report.html` and `docs/lead.svg`. Every chart has a table view.

**Known limits.** A visit counts as exposure even if the token was deleted from the page before the visit. The row-level shuffle spreads an agent's saves over more pages and so makes more adopters. The report prints the adopter counts and the linked share next to the link counts, so you can see the size of that effect. Very fast spreads can be burst editing. Links are candidates, not proof: hand-check a sample before calling one spread.

## Tests

```
python -m unittest
```

The suite runs in about 10 seconds. `tests/test_planted.py` plants three known cascades in a made-up chat log, an inherited-text trap in a wiki fixture and a parallel start. All three cascades must be found and inherited text must make no links. `tests/test_nullmodel.py` checks that no shuffle reaches the planted chain or star. Tests that need the real export skip unless `PZ_DATA` points at its folder. The browser render check of the figure runs only with `PZ_FIGCHECK=1`.

## Data handling

The German wiki text was written by AI agents for other agents, and some of it is hostile. Only deterministic code reads it. Nothing here opens a URL or runs a command found in the data. Strings taken from the data are escaped and cut to 80 characters before they reach any output.

Maintainer tool: `tools/leakcheck.py` runs before every commit. It fails on any `.gz`, `.jsonl` or `.zip` file in the repo, on IP-shaped strings and on the half-IP field name. It also fails on any 12-word run copied from the data. It needs an index built from the data (`--build-index --data DIR`). In a fresh clone without that index it fails closed with exit code 2. It also scans the git-ignored `out/` folder, so it fails on the `.jsonl` files that a demo run leaves there. Keep run outputs outside the repo when you run it.

## Citations

- German wiki incident data: collusion.wiki (Nightingale Collective). The export was generated 2026-09-03 and downloaded 2026-10-02.
- Built for the AI Swarm Dynamics Hackathon (AI Village / AI Digest and Grove Research), October 2026.

## Licence

Dual-licensed under MIT or Apache-2.0, at your option: [LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE).

Code and code docs were drafted by Claude (Anthropic) in Claude Code. Every commit carries a `Co-Authored-By` line.
