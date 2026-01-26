"""Markdown I/O for recommendations.

Export and load recommendations in .md format (DOs / DON'Ts) so the
creative generator can consume the ad/recommender output file.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

SECTION_DO = "## DOs"
SECTION_DONT = "## DON'Ts"
SECTION_DOS_ALT = "## Do's"
# New comprehensive format sections
SECTION_DO_COMPREHENSIVE = "## DO - Positive Patterns to Implement"
SECTION_DONT_COMPREHENSIVE = "## DON'T - Anti-Patterns to Avoid"
BULLET_RE = re.compile(r"^\s*[-*]\s+(.+)$")
# Pattern for numbered feature headings: "### 1. Feature Name"
FEATURE_HEADING_RE = re.compile(r"^###\s+\d+\.\s+(.+)$")
# Pattern for Value/Confidence/Opportunity Size/Suggestion lines
VALUE_LINE_RE = re.compile(r"^- \*\*Value\*\*:\s*`(.+?)`")
CONFIDENCE_LINE_RE = re.compile(r"^- \*\*Confidence\*\*:\s*(.+)$")
OPPORTUNITY_SIZE_RE = re.compile(r"^- \*\*Opportunity Size\*\*:\s*([\d.]+)$")
SUGGESTION_LINE_RE = re.compile(r"^- \*\*Suggestion\*\*:\s*(.+)$")


def export_recommendations_md(data: Dict[str, Any], path: Path | str) -> None:
    """Write recommendations to a Markdown file (DOs / DON'Ts / Opportunities).

    Args:
        data: Dict with "recommendations" list. Each item has "feature",
            "recommended", "type" ("improvement" -> DO, "anti_pattern" -> DON'T),
            optional "confidence", "reason", "potential_impact" (opportunity_size).
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    recs = data.get("recommendations") or []
    
    # Build structured DO and DON'T lists with opportunity_size
    dos_map: Dict[tuple, Dict[str, Any]] = {}  # Key: (feature, value) -> data
    donts_map: Dict[tuple, Dict[str, Any]] = {}
    
    for r in recs:
        feat = r.get("feature", "").strip()
        rec = r.get("recommended", "").strip()
        current = r.get("current", "").strip()
        typ = r.get("type", "improvement")
        confidence = r.get("confidence", "medium")
        reason = r.get("reason", "")
        potential_impact = r.get("potential_impact", 0.0)
        
        if not feat:
            continue
        
        # Cap infinite opportunity sizes at 100
        opportunity_size = potential_impact
        if opportunity_size == float("inf") or opportunity_size > 100:
            opportunity_size = 100.0
        
        if typ == "anti_pattern":
            # For DON'Ts, use current value (what to avoid)
            display_val = current if current else rec.replace("NOT ", "")
            key = (feat, display_val)
            # Keep highest opportunity_size for duplicates
            if key not in donts_map or opportunity_size > donts_map[key].get("opportunity_size", 0):
                donts_map[key] = {
                    "feature": feat,
                    "value": display_val,
                    "confidence": confidence,
                    "opportunity_size": opportunity_size,
                    "reason": reason,
                }
        else:
            # For DOs, use recommended value
            display_val = rec
            key = (feat, display_val)
            # Keep highest opportunity_size for duplicates
            if key not in dos_map or opportunity_size > dos_map[key].get("opportunity_size", 0):
                dos_map[key] = {
                    "feature": feat,
                    "value": display_val,
                    "confidence": confidence,
                    "opportunity_size": opportunity_size,
                    "reason": reason,
                }
    
    # Convert maps to sorted lists (by opportunity_size descending)
    dos = sorted(
        dos_map.values(), 
        key=lambda x: x["opportunity_size"], 
        reverse=True
    )
    donts = sorted(
        donts_map.values(), 
        key=lambda x: x["opportunity_size"], 
        reverse=True
    )
    
    # Generate suggestions (simplified version of creative scorer's _generate_suggestion)
    def _generate_suggestion(feature: str, value: str, rec_type: str) -> str:
        """Generate human-readable suggestion for a feature-value pair."""
        # Simple mapping for common features
        suggestions = {
            "direction": {
                "Overhead": "Shoot from directly above (bird's eye view looking down)",
                "Front": "Front-facing product shot",
                "Side": "Side angle, product profile",
            },
            "lighting_style": {
                "studio": "Use professional studio lighting setup",
                "natural": "Use natural window or outdoor lighting",
            },
            "visual_prominence": {
                "dominant": "Make product the dominant, largest element in the frame",
                "balanced": "Balance product with other elements evenly",
            },
            "lighting_type": {
                "Artificial": "Use studio/artificial lighting setup instead of natural light",
                "Natural": "Use natural daylight lighting",
            },
        }
        
        if feature in suggestions and value in suggestions[feature]:
            suggestion = suggestions[feature][value]
        else:
            # Fallback: use value as suggestion
            value_clean = value.replace("NOT ", "").strip()
            suggestion = f"Use '{value_clean}'"
        
        if rec_type == "dont":
            return f"Avoid {suggestion.lower()}"
        return suggestion
    
    lines = ["# Creative recommendations", ""]
    
    # Add comprehensive Opportunities section (all DOs and DON'Ts with opportunity sizes)
    # This matches the creative scorer format - shows all opportunities prioritized by impact
    all_opportunities = dos + donts
    all_opportunities.sort(key=lambda x: x["opportunity_size"], reverse=True)
    
    if all_opportunities:
        lines.append("## Opportunities")
        lines.append("")
        lines.append("All recommendations prioritized by potential ROAS improvement:")
        lines.append("")
        for i, opp in enumerate(all_opportunities, 1):  # Show all opportunities
            feat_name = opp["feature"].replace("_", " ").title()
            value = opp["value"]
            conf = opp["confidence"].title()
            opp_size = opp["opportunity_size"]
            rec_type = "DO" if opp in dos else "DON'T"
            lines.append(f"{i}. **{feat_name}**: `{value}` ({conf}, {rec_type}) — Opportunity Size: {opp_size:.2f}")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # DO section with numbered items and opportunity sizes (comprehensive format)
    lines.append("## DO - Positive Patterns to Implement")
    lines.append("")
    lines.append("These features are associated with top-performing creatives:")
    lines.append("")
    
    if dos:
        for i, do in enumerate(dos, 1):
            suggestion = _generate_suggestion(do["feature"], do["value"], "do")
            lines.append(f"### {i}. {do['feature'].replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"- **Value**: `{do['value']}`")
            lines.append(f"- **Confidence**: {do['confidence'].title()}")
            lines.append(f"- **Opportunity Size**: {do['opportunity_size']:.2f}")
            lines.append(f"- **Suggestion**: {suggestion}")
            lines.append("")
    else:
        lines.append("- *(none)*")
        lines.append("")

    lines.append("---")
    lines.append("")
    
    # DON'T section with numbered items and opportunity sizes (comprehensive format)
    lines.append("## DON'T - Anti-Patterns to Avoid")
    lines.append("")
    lines.append("These features are associated with lower-performing creatives:")
    lines.append("")
    
    if donts:
        for i, dont in enumerate(donts, 1):
            suggestion = _generate_suggestion(dont["feature"], dont["value"], "dont")
            lines.append(f"### {i}. {dont['feature'].replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"- **Value**: `{dont['value']}`")
            lines.append(f"- **Confidence**: {dont['confidence'].title()}")
            lines.append(f"- **Opportunity Size**: {dont['opportunity_size']:.2f}")
            lines.append(f"- **Suggestion**: {suggestion}")
            lines.append("")
    else:
        lines.append("- *(none)*")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total DOs**: {len(dos)}")
    lines.append(f"- **Total DON'Ts**: {len(donts)}")
    lines.append(f"- **Total Opportunities**: {len(dos) + len(donts)}")

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    logger.info(
        "Wrote recommendations to %s (%d DOs, %d DON'Ts, %d total opportunities)",
        path, len(dos), len(donts), len(dos) + len(donts)
    )


def _parse_bullet(line: str) -> str | None:
    m = BULLET_RE.match(line)
    if not m:
        return None
    s = m.group(1).strip()
    if not s or s.startswith("*(none)*"):
        return None
    return s


def _parse_do_line(s: str) -> Dict[str, Any] | None:
    # "**feature**: value (confidence) — reason" or "**feature**: value"
    s = s.strip()
    match = re.match(r"\*\*(.+?)\*\*:\s*(.+)$", s)
    if not match:
        return None
    feat = match.group(1).strip()
    rest = match.group(2).strip()
    confidence = ""
    reason = ""
    if " — " in rest:
        rest, reason = rest.split(" — ", 1)
        reason = reason.strip()
    if "(" in rest and ")" in rest:
        m = re.search(r"\(([^)]+)\)\s*$", rest)
        if m:
            confidence = m.group(1).strip()
            rest = rest[: m.start()].strip()
    recommended = rest.strip()
    if not feat or not recommended:
        return None
    return {
        "source": "md",
        "feature": feat,
        "recommended": recommended,
        "type": "improvement",
        "confidence": confidence or "medium",
        "reason": reason,
    }


def _parse_dont_line(s: str) -> Dict[str, Any] | None:
    # "**feature**: value" -> avoid value; store current=value, recommended="NOT value"
    out = _parse_do_line(s)
    if not out:
        return None
    out["type"] = "anti_pattern"
    val = out["recommended"]
    out["current"] = val
    if not val.upper().startswith("NOT "):
        out["recommended"] = f"NOT {val}"
    return out


def load_recommendations_md(path: Path | str) -> Dict[str, Any]:
    """Load recommendations from a Markdown file (DOs / DON'Ts).

    Supports two formats:
    1. Legacy format: "## DOs" / "## DON'Ts" with bullet lines
    2. Comprehensive format: "## DO - Positive Patterns..." with numbered ### headings

    Returns a dict compatible with format_recs_as_prompts:
    {"recommendations": [...], "creative_id": "aggregated", "confidence_scores": {...}}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Recommendations file not found: {path}")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    recs: List[Dict[str, Any]] = []
    section: str | None = None
    
    # Try comprehensive format first (new format with ### headings)
    recs = _load_comprehensive_format(lines)
    
    # If comprehensive format found nothing, fall back to legacy format
    if not recs:
        section = None
        for line in lines:
            if line.strip() == SECTION_DO or line.strip() == SECTION_DOS_ALT:
                section = "do"
                continue
            if line.strip() == SECTION_DONT:
                section = "dont"
                continue
            bullet = _parse_bullet(line)
            if not bullet:
                continue
            if section == "do":
                parsed = _parse_do_line(bullet)
                if parsed:
                    recs.append(parsed)
            elif section == "dont":
                parsed = _parse_dont_line(bullet)
                if parsed:
                    recs.append(parsed)

    logger.info("Loaded %d recommendations from %s", len(recs), path)
    return {
        "recommendations": recs,
        "creative_id": "aggregated",
        "current_roas": 0.0,
        "predicted_roas": 0.0,
        "confidence_scores": {"combined_confidence": 0.7},
    }


def _load_comprehensive_format(lines: List[str]) -> List[Dict[str, Any]]:
    """Load recommendations from comprehensive format with ### headings.
    
    Format:
    ### 1. Feature Name
    - **Value**: `value`
    - **Confidence**: High/Medium/Low
    - **Opportunity Size**: 100.00
    - **Suggestion**: suggestion text
    """
    recs: List[Dict[str, Any]] = []
    section: str | None = None
    current_feature: Dict[str, Any] = {}
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Detect section headers
        if line_stripped == SECTION_DO_COMPREHENSIVE:
            section = "do"
            continue
        if line_stripped == SECTION_DONT_COMPREHENSIVE:
            section = "dont"
            continue
        
        # Skip if not in a section
        if not section:
            continue
        
        # Detect feature heading: "### 1. Feature Name"
        heading_match = FEATURE_HEADING_RE.match(line_stripped)
        if heading_match:
            # Save previous feature if complete
            if current_feature.get("feature") and current_feature.get("recommended"):
                rec = _build_recommendation_from_comprehensive(current_feature, section)
                if rec:
                    recs.append(rec)
            # Start new feature
            feature_name = heading_match.group(1).strip()
            # Convert "Feature Name" back to "feature_name" (snake_case)
            feature_name = feature_name.lower().replace(" ", "_")
            current_feature = {
                "feature": feature_name,
                "section": section,
            }
            continue
        
        # Parse value line: "- **Value**: `value`"
        value_match = VALUE_LINE_RE.match(line_stripped)
        if value_match:
            current_feature["recommended"] = value_match.group(1).strip()
            continue
        
        # Parse confidence line: "- **Confidence**: High/Medium/Low"
        confidence_match = CONFIDENCE_LINE_RE.match(line_stripped)
        if confidence_match:
            current_feature["confidence"] = confidence_match.group(1).strip().lower()
            continue
        
        # Parse opportunity size: "- **Opportunity Size**: 100.00"
        opp_match = OPPORTUNITY_SIZE_RE.match(line_stripped)
        if opp_match:
            try:
                current_feature["potential_impact"] = float(opp_match.group(1))
            except ValueError:
                pass
            continue
        
        # Parse suggestion line: "- **Suggestion**: suggestion text"
        suggestion_match = SUGGESTION_LINE_RE.match(line_stripped)
        if suggestion_match:
            current_feature["reason"] = suggestion_match.group(1).strip()
            continue
    
    # Save last feature if complete
    if current_feature.get("feature") and current_feature.get("recommended"):
        rec = _build_recommendation_from_comprehensive(current_feature, section or "do")
        if rec:
            recs.append(rec)
    
    return recs


def _build_recommendation_from_comprehensive(
    feature_data: Dict[str, Any], section: str
) -> Dict[str, Any] | None:
    """Build recommendation dict from comprehensive format data."""
    feature = feature_data.get("feature", "").strip()
    recommended = feature_data.get("recommended", "").strip()
    
    if not feature or not recommended:
        return None
    
    rec: Dict[str, Any] = {
        "source": "md",
        "feature": feature,
        "recommended": recommended,
        "type": "improvement" if section == "do" else "anti_pattern",
        "confidence": feature_data.get("confidence", "medium").lower(),
        "reason": feature_data.get("reason", ""),
    }
    
    if "potential_impact" in feature_data:
        rec["potential_impact"] = feature_data["potential_impact"]
    
    if section == "dont":
        rec["current"] = recommended
        if not recommended.upper().startswith("NOT "):
            rec["recommended"] = f"NOT {recommended}"
    
    return rec


def load_recommendations_file(path: Path | str) -> Dict[str, Any]:
    """Load recommendations from .json or .md (ad/recommender output).

    Returns dict with "recommendations" list, compatible with
    format_recs_as_prompts and prompt conversion.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Recommendations file not found: {path}")
    suf = path.suffix.lower()
    if suf == ".md":
        return load_recommendations_md(path)
    if suf == ".json":
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {
                "recommendations": data,
                "creative_id": "aggregated",
                "current_roas": 0.0,
                "predicted_roas": 0.0,
                "confidence_scores": {"combined_confidence": 0.7},
            }
        return data
    raise ValueError(f"Unsupported recommendations format: {suf}. Use .json or .md")
