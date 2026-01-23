"""Format recommendations as prompts for image generation."""

import csv
import json
from typing import Any, Dict, List, Optional


def format_recs_as_prompts(
    recommendations: List[Dict[str, Any]],
    style: Optional[str] = None,
    include_metadata: bool = True,
) -> List[str]:
    """Format recommendations as image generation prompts.

    Args:
        recommendations: List of recommendation dictionaries
        style: Optional style descriptor for prompts
        include_metadata: Whether to include metadata in prompts

    Returns:
        List of formatted prompt strings
    """
    if not recommendations:
        return []

    prompts = []
    for rec in recommendations:
        prompt = _build_single_prompt(rec, style, include_metadata)
        if prompt:
            prompts.append(prompt)

    return prompts


def _build_single_prompt(
    rec: Dict[str, Any],
    style: Optional[str],
    include_metadata: bool,
) -> Optional[str]:
    """Build a single prompt from a recommendation.

    Args:
        rec: Recommendation dictionary
        style: Optional style descriptor
        include_metadata: Whether to include metadata

    Returns:
        Formatted prompt string or None if invalid
    """
    if not rec or not isinstance(rec, dict):
        return None

    parts = _extract_prompt_parts(rec)
    if not parts:
        return None

    prompt = _combine_prompt_parts(parts, style)
    if include_metadata:
        prompt = _add_metadata(prompt, rec)

    return prompt


def _extract_prompt_parts(rec: Dict[str, Any]) -> Dict[str, str]:
    """Extract key parts from recommendation for prompt building.

    Args:
        rec: Recommendation dictionary

    Returns:
        Dictionary with prompt parts
    """
    parts = {}
    parts["action"] = rec.get("action", "")
    parts["description"] = rec.get("description", "")
    parts["category"] = rec.get("category", "")
    parts["priority"] = rec.get("priority", "")

    return parts


def _combine_prompt_parts(
    parts: Dict[str, str],
    style: Optional[str],
) -> str:
    """Combine prompt parts into a single string.

    Args:
        parts: Dictionary of prompt parts
        style: Optional style descriptor

    Returns:
        Combined prompt string
    """
    main_parts = []
    if parts.get("action"):
        main_parts.append(parts["action"])
    if parts.get("description"):
        main_parts.append(parts["description"])
    if parts.get("category"):
        main_parts.append(f"Category: {parts['category']}")

    prompt = ", ".join(main_parts) if main_parts else "Generic recommendation"

    if style:
        prompt = f"{prompt}, Style: {style}"

    if parts.get("priority"):
        prompt = f"{prompt}, Priority: {parts['priority']}"

    return prompt


def _add_metadata(prompt: str, rec: Dict[str, Any]) -> str:
    """Add metadata to prompt if available.

    Args:
        prompt: Base prompt string
        rec: Recommendation dictionary

    Returns:
        Prompt with metadata appended
    """
    metadata_parts = []
    if rec.get("adset_id"):
        metadata_parts.append(f"Adset: {rec['adset_id']}")
    if rec.get("campaign_id"):
        metadata_parts.append(f"Campaign: {rec['campaign_id']}")

    if metadata_parts:
        prompt = f"{prompt} ({', '.join(metadata_parts)})"

    return prompt


def export_prompts_batch(
    prompts: List[str],
    output_path: str,
    format_type: str = "txt",
) -> bool:
    """Export prompts for batch processing.

    Args:
        prompts: List of prompt strings
        output_path: Path to output file
        format_type: Output format (txt, json, csv)

    Returns:
        True if successful, False otherwise
    """
    if not prompts:
        return False

    if format_type == "txt":
        return _export_txt(prompts, output_path)
    if format_type == "json":
        return _export_json(prompts, output_path)
    if format_type == "csv":
        return _export_csv(prompts, output_path)

    return False


def _export_txt(prompts: List[str], output_path: str) -> bool:
    """Export prompts as text file.

    Args:
        prompts: List of prompt strings
        output_path: Path to output file

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for prompt in prompts:
                f.write(f"{prompt}\n")
        return True
    except (IOError, OSError):
        return False


def _export_json(prompts: List[str], output_path: str) -> bool:
    """Export prompts as JSON file.

    Args:
        prompts: List of prompt strings
        output_path: Path to output file

    Returns:
        True if successful, False otherwise
    """
    try:
        data = {"prompts": prompts, "count": len(prompts)}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except (IOError, OSError, ValueError):
        return False


def _export_csv(prompts: List[str], output_path: str) -> bool:
    """Export prompts as CSV file.

    Args:
        prompts: List of prompt strings
        output_path: Path to output file

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "prompt"])
            for idx, prompt in enumerate(prompts, start=1):
                writer.writerow([idx, prompt])
        return True
    except (IOError, OSError):
        return False
