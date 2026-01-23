"""
Output formatting helpers for feature-to-prompt conversion results.

The test suite expects `format_output()` to exist and support:
- json
- text
- markdown
- category
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def _safe_list(result: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    val = result.get(key, [])
    return val if isinstance(val, list) else []


def format_output(result: Dict[str, Any], *, format_type: str = "text") -> str:
    """
    Format converter output into a human-readable string.

    Args:
        result: Dict returned by `convert_features_to_prompts`
        format_type: One of: json, text, markdown, category

    Returns:
        Formatted string.
    """

    fmt_type = str(format_type or "").strip().lower()

    # Always include something useful even if `combined_prompt` is empty
    all_instructions = _safe_list(result, "all_instructions")
    combined_prompt = str(result.get("combined_prompt", "") or "").strip()

    if fmt_type == "json":
        return json.dumps(result, ensure_ascii=False, indent=2)

    if fmt_type == "text":
        if combined_prompt:
            return combined_prompt + "\n"
        if all_instructions:
            return (
                "\n".join(
                    str(x.get("instruction", "")).strip()
                    for x in all_instructions
                ).strip()
                + "\n"
            )
        return "No instructions generated.\n"

    if fmt_type == "markdown":
        lines: List[str] = []
        lines.append("## Feature Instructions")
        if combined_prompt:
            lines.append("")
            lines.append("```")
            lines.append(combined_prompt)
            lines.append("```")
            lines.append("")
            return "\n".join(lines).strip() + "\n"

        if all_instructions:
            lines.append("")
            for x in all_instructions:
                inst = str(x.get("instruction", "")).strip()
                if inst:
                    lines.append(f"- {inst}")
            return "\n".join(lines).strip() + "\n"

        return "## Feature Instructions\n\nNo instructions generated.\n"

    if fmt_type == "category":
        category_prompts = result.get("category_prompts") or {}
        if isinstance(category_prompts, dict) and category_prompts:
            lines = []
            for cat, txt in category_prompts.items():
                text = str(txt or "").strip()
                if not text:
                    continue
                lines.append(f"{str(cat).upper()} REQUIREMENTS:")
                lines.append(text)
                lines.append("")
            return "\n".join(lines).strip() + "\n"

        # Fallback: behave like markdown
        return format_output(result, format_type="markdown")

    raise ValueError(f"Unsupported format_type: {format_type}")
