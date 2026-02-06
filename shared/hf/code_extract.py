from __future__ import annotations

import re


_CODE_BLOCK_RE = re.compile(
    r"```(?:[a-zA-Z0-9_+-]+)?\n(?P<code>.*?)(?:\n```|```)",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_python_code(text: str) -> str:
    """
    Extract Python code from a model completion.

    We try to be permissive (handle fenced code blocks) but avoid heavy post-processing:
    reward is computed by executing the returned code, so we want the executed string
    to be as close as possible to what the model produced.
    """
    if not isinstance(text, str):
        return ""

    m = _CODE_BLOCK_RE.search(text)
    if m:
        return (m.group("code") or "").strip()
    return text.strip()
