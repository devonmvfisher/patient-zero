#!/usr/bin/env python3
"""patient-zero: one command runs the whole pipeline.

    python cli.py --adapter wiki --input PATH/TO/collusion --out PATH/TO/out
    python cli.py --adapter generic --input examples/fake_log.ndjson --out out/example

Writes facts.json (counts only) and trees.jsonl (one summary line per spread
tree) to --out. See patient_zero/pipeline.py for the fields.
"""

import sys

from patient_zero.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
