"""Output-safe strings: escape and cut data strings, stable token handles.

Python standard library only. Every string taken from the data (agent labels,
page families, token keys) passes through ``short`` before it is written to
an output file: anything outside printable ASCII and every backslash is
escaped, and the result is cut to 80 characters.
"""

import hashlib

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
