"""Output-safe strings and paths: escape and cut data strings, stable token
handles, and output paths that never overwrite an input.

Python standard library only. Every string taken from the data (agent labels,
page families, token keys) passes through ``short`` before it is written to
an output file: anything outside printable ASCII and every backslash is
escaped, and the result is cut to 80 characters. Every command that writes a
file checks the path with ``guard_out`` first.
"""

import hashlib
import os

SHORT_MAX = 80
NO_FAMILY = "(none)"


def short(value, limit=SHORT_MAX):
    """Escape a data string (non-printable-ASCII and backslashes) and cut it to ``limit``."""
    if value is None:
        return None
    pieces = []
    for ch in str(value):
        code = ord(ch)
        if ch == "\\":
            pieces.append("\\\\")
        elif 32 <= code < 127:
            pieces.append(ch)
        elif code < 0x10000:
            pieces.append("\\u%04x" % code)
        else:
            pieces.append("\\U%08x" % code)
    if sum(len(p) for p in pieces) <= limit:
        return "".join(pieces)
    kept, size = [], 0
    for piece in pieces:  # never cut inside an escape
        if size + len(piece) > limit - 3:
            break
        kept.append(piece)
        size += len(piece)
    return "".join(kept) + "..."


def token_id(key):
    """Stable 12-hex handle for a token key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _real(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def same_file(a, b):
    """True when paths ``a`` and ``b`` name the same file or folder.

    Same real path (links resolved, case folded on Windows), or the same file
    on disk (``os.path.samefile``: hard links, short names). False when either
    does not exist and the real paths differ.
    """
    if _real(a) == _real(b):
        return True
    try:
        return os.path.samefile(a, b)
    except (OSError, ValueError):
        return False


def _inside(path, folder):
    try:
        return os.path.commonpath([_real(path), _real(folder)]) == _real(folder)
    except ValueError:  # different drives
        return False


def guard_out(output, inputs, flag="--out"):
    """Refuse an output path that would overwrite an input; call it before any work.

    ``inputs`` holds the file and folder paths the command reads (empty
    entries are skipped). The output may not be an input, may not sit inside
    an input folder, and its ``.tmp`` twin (the writers here write
    ``PATH.tmp`` first, then rename it) may not be an input. Raises ValueError
    with a plain sentence; nothing has been written when it does.
    """
    if not output:
        return
    for item in inputs:
        if not item:
            continue
        if same_file(output, item):
            raise ValueError("%s names %s, which is also an input; writing there would destroy it. "
                             "Choose another path." % (flag, output))
        if os.path.isdir(item) and _inside(output, item):
            raise ValueError("%s names %s, which is inside the input folder %s. Write outputs "
                             "somewhere else." % (flag, output, item))
        if same_file(output + ".tmp", item):
            raise ValueError("%s names %s, whose temporary file %s.tmp is an input. Choose another "
                             "path." % (flag, output, output))
