"""
Adapter to convert ad/miner patterns to visual formula format.

The ad generator expects recommendations in the creative scorer format:
- entrance_features: High-performance baseline features
- headroom_features: High ROAS, low penetration features

The ad miner outputs:
- patterns.yaml with positive_patterns and negative_patterns

This adapter bridges the gap by converting ad/miner YAML format to visual formula format.
"""

from __future__ import annotations

import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.meta.ad.generator.orchestrator.recommendation_mapping import (
    get_placeholder_for_feature,
    is_mapped_feature,
    transform_feature_value,
)

logger = logging.getLogger(__name__)


def convert_patterns_to_visual_formula(
    patterns_data: Dict[str, Any],
    min_confidence: str = "medium",
    min_high_performer_pct: float = 0.25,
) -> Dict[str, Any]:
    """
    Convert ad/miner patterns to visual formula format.

    Args:
        patterns_data: Dict from patterns.yaml with "positive_patterns" and "negative_patterns"
        min_confidence: Minimum confidence level ("high", "medium", "low")
        min_high_performer_pct: Minimum high_performer_pct to include

    Returns:
        Visual formula dict with entrance_features and headroom_features
    """
    # Process positive patterns
    positive_patterns = patterns_data.get("positive_patterns", {})
    negative_patterns = patterns_data.get("negative_patterns", {})

    # Flatten positive patterns into a list
    improvements = []
    for feature_name, patterns in positive_patterns.items():
        if isinstance(patterns, list):
            for pattern in patterns:
                if pattern.get("confidence", "medium") in _get_confidence_levels(min_confidence):
                    improvements.append({
                        "feature": feature_name,
                        "recommended": pattern.get("value"),
                        "confidence": pattern.get("confidence"),
                        "opportunity_size": pattern.get("opportunity_size", 0),
                        "suggestion": pattern.get("suggestion"),
                        "type": "improvement"
                    })

    # Flatten negative patterns
    anti_patterns = []
    for feature_name, patterns in negative_patterns.items():
        if isinstance(patterns, list):
            for pattern in patterns:
                if pattern.get("confidence", "medium") in _get_confidence_levels(min_confidence):
                    anti_patterns.append({
                        "feature": feature_name,
                        "recommended": pattern.get("value"),
                        "confidence": pattern.get("confidence"),
                        "opportunity_size": pattern.get("opportunity_size", 0),
                        "suggestion": pattern.get("suggestion"),
                        "type": "anti_pattern"
                    })

    # Convert improvements to entrance_features (high confidence, high prevalence)
    # and headroom_features (lower prevalence but high ROAS potential)
    entrance_features = []
    headroom_features = []
    
    for rec in improvements:
        original_feature_name = rec.get("feature", "")
        original_feature_value = rec.get("recommended", "")
        
        # Map recommendation feature to template placeholder
        placeholder_name = get_placeholder_for_feature(original_feature_name)
        
        # Skip if feature doesn't map to any placeholder (e.g., text_elements, cta_visuals)
        if not placeholder_name:
            logger.debug(
                "Skipping unmapped feature: %s (no placeholder mapping)",
                original_feature_name
            )
            continue
        
        # Transform value using mapping function
        transformed_value = transform_feature_value(original_feature_name, original_feature_value)
        
        # Try multiple fields for high_performer_pct (MD format may not have it)
        high_pct = (
            rec.get("high_performer_pct", 0) or
            rec.get("high_pct", 0) or
            _extract_pct_from_reason(rec.get("reason", ""))
        )
        confidence = rec.get("confidence", "medium")
        potential_impact = rec.get("potential_impact", 0)
        
        # Skip if confidence too low
        if confidence not in _get_confidence_levels(min_confidence):
            continue
        
        # If no high_pct available, use confidence as proxy
        # High confidence = assume higher prevalence
        if high_pct == 0:
            high_pct = 0.5 if confidence == "high" else 0.3 if confidence == "medium" else 0.25
        
        # Skip if prevalence too low
        if high_pct < min_high_performer_pct:
            continue
        
        # Estimate ROAS from potential impact and confidence
        # High confidence + high prevalence = higher ROAS estimate
        if potential_impact == float("inf"):
            # High confidence features get higher ROAS estimates
            avg_roas = 4.0 if confidence == "high" else 3.0
        else:
            avg_roas = min(potential_impact, 5.0)
        
        # Adjust ROAS based on prevalence (higher prevalence = more reliable)
        if high_pct >= 0.5:
            avg_roas *= 1.2  # Boost for high prevalence
        elif high_pct >= 0.33:
            avg_roas *= 1.1  # Slight boost for medium-high prevalence
        
        # Use mapped placeholder name as feature_name, transformed value as feature_value
        feature_dict = {
            "feature_name": placeholder_name,  # Use placeholder name (e.g., "global_view_definition")
            "feature_value": transformed_value,  # Use transformed value
            "avg_roas": avg_roas,
            "penetration_pct": high_pct * 100,  # Convert to percentage
            "sample_count": int(high_pct * 100),  # Estimate from percentage
            "roas_lift_pct": 0.0,  # Not available in ad/miner format
            # Store original for reference
            "_original_feature": original_feature_name,
            "_original_value": original_feature_value,
        }
        
        # High prevalence (>= 50%) or high confidence -> entrance_features
        # Lower prevalence but medium+ confidence -> headroom_features
        if high_pct >= 0.5 or confidence == "high":
            entrance_features.append(feature_dict)
        else:
            headroom_features.append(feature_dict)
    
    # Convert anti_patterns to negative_guidance
    negative_guidance = []
    for rec in anti_patterns:
        feature_name = rec.get("feature", "")
        bad_value = rec.get("current", "") or rec.get("recommended", "").replace("NOT ", "")
        low_pct = rec.get("low_performer_pct", 0)
        
        if not feature_name or not bad_value:
            continue
        
        negative_guidance.append({
            "feature_name": feature_name,
            "feature_value": bad_value,
            "low_performer_pct": low_pct * 100,
            "reason": rec.get("reason", f"Present in {low_pct:.1%} of bottom performers"),
        })

    # Build visual formula
    current_roas = patterns_data.get("current_roas", 0.0)
    predicted_roas = patterns_data.get("predicted_roas", current_roas)

    visual_formula = {
        "entrance_features": entrance_features,
        "headroom_features": headroom_features,
        "negative_guidance": negative_guidance,
        "account_avg_roas": current_roas,
        "winning_threshold_roas": predicted_roas * 1.2,  # 20% above predicted
        "expected_roas_improvement_pct": (
            ((predicted_roas - current_roas) / current_roas * 100) 
            if current_roas > 0 else 0.0
        ),
        "synergy_pairs": [],  # Not available in ad/miner format
        "generation_instructions": {
            "must_include": [f.get("feature_name") for f in entrance_features[:5]],
            "prioritize": [f.get("feature_name") for f in headroom_features[:5]],
            "avoid": [g.get("feature_name") for g in negative_guidance[:5]],
        },
    }
    
    logger.info(
        "Converted recommendations: %d entrance, %d headroom, %d negative",
        len(entrance_features),
        len(headroom_features),
        len(negative_guidance),
    )
    
    return visual_formula


def _get_confidence_levels(min_confidence: str) -> List[str]:
    """Get list of confidence levels >= min_confidence."""
    levels = ["low", "medium", "high"]
    min_idx = levels.index(min_confidence) if min_confidence in levels else 1
    return levels[min_idx:]


def _extract_pct_from_reason(reason: str) -> float:
    """Extract percentage from reason string like 'Present in 25.0% of top performers'."""
    import re
    match = re.search(r"(\d+\.?\d*)%", reason)
    if match:
        return float(match.group(1)) / 100.0
    return 0.0


def load_patterns_as_visual_formula(
    patterns_path: Path | str,
    min_confidence: str = "medium",
    min_high_performer_pct: float = 0.25,
) -> Dict[str, Any]:
    """
    Load patterns from ad/miner and convert to visual formula format.

    Args:
        patterns_path: Path to patterns.yaml
        min_confidence: Minimum confidence level
        min_high_performer_pct: Minimum high_performer_pct to include

    Returns:
        Visual formula dict compatible with PromptBuilder
    """
    path = Path(patterns_path)
    if not path.exists():
        raise FileNotFoundError(f"Patterns file not found: {path}")

    # Load YAML patterns file
    with open(path) as f:
        patterns_data = yaml.safe_load(f)

    logger.info(f"Loaded patterns from {path}")
    logger.info(f"  Positive patterns: {sum(len(v) if isinstance(v, list) else 1 for v in patterns_data.get('positive_patterns', {}).values())}")
    logger.info(f"  Negative patterns: {sum(len(v) if isinstance(v, list) else 1 for v in patterns_data.get('negative_patterns', {}).values())}")

    # Convert to visual formula format
    visual_formula = convert_patterns_to_visual_formula(
        patterns_data,
        min_confidence=min_confidence,
        min_high_performer_pct=min_high_performer_pct,
    )

    return visual_formula


# Backward compatibility aliases
load_recommendations_as_visual_formula = load_patterns_as_visual_formula
convert_recommendations_to_visual_formula = convert_patterns_to_visual_formula
