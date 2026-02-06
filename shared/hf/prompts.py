from __future__ import annotations

from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful coding assistant.\n"
    "Write a correct Python solution.\n"
    "Output ONLY Python code. Do not use Markdown."
)

DEFAULT_SQL_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "Write a correct SQL query.\n"
    "Output ONLY the SQL query. Do not use Markdown."
)


def build_code_prompt(
    observation: str,
    tokenizer: Any,
    *,
    use_chat_template: bool = True,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    user_prompt = observation.strip() + "\n\nReturn ONLY the Python code."

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback: plain-text prompt
    return system_prompt.strip() + "\n\n" + user_prompt + "\n\n# Python code\n"


def build_sql_prompt(
    observation: str,
    tokenizer: Any,
    *,
    use_chat_template: bool = True,
    system_prompt: str = DEFAULT_SQL_SYSTEM_PROMPT,
) -> str:
    user_prompt = observation.strip() + "\n\nReturn ONLY the SQL query."

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return system_prompt.strip() + "\n\n" + user_prompt + "\n\n-- SQL\n"
