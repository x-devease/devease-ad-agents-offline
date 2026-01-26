"""
Confidence-Weighted Features Module.

Prioritizes and orders features based on confidence levels to ensure
high-confidence features appear prominently in generated prompts.

Confidence levels:
- HIGH (0.8-1.0): Critical features, must be prominent
- MEDIUM (0.5-0.8): Important features, should be included
- LOW (0.2-0.5): Optional features, include if space permits
"""

import logging
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


# Confidence level definitions
CONFIDENCE_LEVELS = {
    "HIGH": {
        "range": (0.8, 1.0),
        "description": "Critical feature - must be prominently featured",
        "placement": "First 25% of prompt",
        "emphasis": "Maximum detail and specificity",
        "repetition": "Can be referenced 2-3 times",
        "priority": 1,
    },
    "MEDIUM": {
        "range": (0.5, 0.8),
        "description": "Important feature - should be included",
        "placement": "Middle 50% of prompt",
        "emphasis": "Good detail and clarity",
        "repetition": "Can be referenced 1-2 times",
        "priority": 2,
    },
    "LOW": {
        "range": (0.2, 0.5),
        "description": "Optional feature - include if space permits",
        "placement": "Last 25% of prompt",
        "emphasis": "Brief mention",
        "repetition": "Reference once max",
        "priority": 3,
    },
    "VERY_LOW": {
        "range": (0.0, 0.2),
        "description": "Speculative feature - omit if constrained",
        "placement": "End of prompt or omit",
        "emphasis": "Minimal or omit",
        "repetition": "Omit preferred",
        "priority": 4,
    },
}


def get_confidence_level(confidence: float) -> str:
    """
    Get confidence level category from confidence score.

    Args:
        confidence: Confidence score (0.0 to 1.0)

    Returns:
        Confidence level name (HIGH, MEDIUM, LOW, VERY_LOW)

    Example:
        >>> level = get_confidence_level(0.85)
        >>> # Returns "HIGH"
    """
    if confidence >= 0.8:
        return "HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    elif confidence >= 0.2:
        return "LOW"
    else:
        return "VERY_LOW"


def prioritize_features(
    features: List[Dict],
    confidence_key: str = "confidence",
) -> List[Dict]:
    """
    Prioritize features by confidence level.

    Args:
        features: List of feature dictionaries
        confidence_key: Key name for confidence value in feature dict

    Returns:
        Prioritized list of features (HIGH → MEDIUM → LOW → VERY_LOW)

    Example:
        >>> prioritized = prioritize_features([
        ...     {"feature_name": "USB-C Port", "confidence": 0.9},
        ...     {"feature_name": "LED Indicator", "confidence": 0.6},
        ...     {"feature_name": "Sticker", "confidence": 0.3},
        ... ])
        >>> # Returns in order: USB-C (HIGH), LED (MEDIUM), Sticker (LOW)
    """
    # Assign priority score based on confidence
    for feature in features:
        confidence = feature.get(confidence_key, 0.5)
        level = get_confidence_level(confidence)
        priority = CONFIDENCE_LEVELS[level]["priority"]
        feature["_priority"] = priority
        feature["_confidence_level"] = level

    # Sort by priority (1 = highest)
    prioritized = sorted(features, key=lambda f: f["_priority"])

    logger.info(
        "Prioritized %d features: %d HIGH, %d MEDIUM, %d LOW, %d VERY_LOW",
        len(features),
        sum(1 for f in prioritized if f["_confidence_level"] == "HIGH"),
        sum(1 for f in prioritized if f["_confidence_level"] == "MEDIUM"),
        sum(1 for f in prioritized if f["_confidence_level"] == "LOW"),
        sum(1 for f in prioritized if f["_confidence_level"] == "VERY_LOW"),
    )

    return prioritized


def weight_feature_value(
    feature: Dict,
    confidence: float,
    max_detail_level: str = "high",
) -> str:
    """
    Generate weighted feature description based on confidence.

    Args:
        feature: Feature dictionary with feature_name and feature_value
        confidence: Confidence score (0.0 to 1.0)
        max_detail_level: Maximum detail level (high, medium, low)

    Returns:
        Weighted feature description string

    Example:
        >>> desc = weight_feature_value(
        ...     {"feature_name": "USB-C Port", "feature_value": "Fast charging"},
        ...     confidence=0.9
        ... )
        >>> # Returns "prominently featured USB-C Port with fast charging capability"
    """
    level = get_confidence_level(confidence)
    level_config = CONFIDENCE_LEVELS[level]

    feature_name = feature.get("feature_name", "")
    feature_value = feature.get("feature_value", "")

    # Adjust detail based on confidence
    if level == "HIGH":
        # Maximum emphasis
        if max_detail_level == "high":
            description = f"prominently featured {feature_name} with {feature_value}"
        else:
            description = f"{feature_name} with {feature_value}"
    elif level == "MEDIUM":
        # Good detail
        description = f"{feature_name} with {feature_value}"
    elif level == "LOW":
        # Brief mention
        description = f"{feature_name}"
        if feature_value:
            description += f" ({feature_value})"
    else:  # VERY_LOW
        # Minimal or omit
        if feature_value and len(feature_value) < 20:
            description = f"{feature_value}"
        else:
            description = f"{feature_name}"

    return description


def create_confidence_section(
    features: List[Dict],
    confidence_level: str,
    section_type: str = "entrance",
) -> str:
    """
    Create a prompt section for features of a specific confidence level.

    Args:
        features: List of features (already filtered by confidence)
        confidence_level: Confidence level (HIGH, MEDIUM, LOW, VERY_LOW)
        section_type: Type of section (entrance or headroom)

    Returns:
        Formatted section string

    Example:
        >>> high_conf_features = [f for f in features if get_confidence_level(f['confidence']) == 'HIGH']
        >>> section = create_confidence_section(high_conf_features, "HIGH")
    """
    if not features:
        return ""

    level_config = CONFIDENCE_LEVELS[confidence_level]
    section_parts = []

    # Section header
    if confidence_level == "HIGH":
        header = f"[Critical Features - {level_config['description']}]"
    elif confidence_level == "MEDIUM":
        header = f"[Important Features - {level_config['description']}]"
    elif confidence_level == "LOW":
        header = f"[Additional Features - {level_config['description']}]"
    else:
        header = f"[Optional Features - {level_config['description']}]"

    section_parts.append(header)

    # Add features
    for feature in features:
        confidence = feature.get("confidence", 0.5)
        weighted_desc = weight_feature_value(feature, confidence)
        section_parts.append(f"- {weighted_desc}")

    return "\n".join(section_parts)


def structure_prompt_by_confidence(
    entrance_features: List[Dict],
    headroom_features: List[Dict],
    max_length: int = 2000,
) -> str:
    """
    Structure prompt with confidence-weighted feature ordering.

    Args:
        entrance_features: List of entrance features
        headroom_features: List of headroom features
        max_length: Maximum prompt length

    Returns:
        Structured prompt with confidence-weighted sections

    Example:
        >>> prompt = structure_prompt_by_confidence(
        ...     entrance_features=[
        ...         {"feature_name": "Handle", "feature_value": "Ergonomic grip", "confidence": 0.95},
        ...     ],
        ...     headroom_features=[
        ...         {"feature_name": "Logo", "feature_value": "Moprobo branding", "confidence": 0.9},
        ...     ]
        ... )
    """
    sections = []

    # Prioritize all features
    prioritized_entrance = prioritize_features(entrance_features)
    prioritized_headroom = prioritize_features(headroom_features)

    # Combine and categorize by confidence level
    all_features = []

    for feature in prioritized_entrance:
        feature["section"] = "entrance"
        all_features.append(feature)

    for feature in prioritized_headroom:
        feature["section"] = "headroom"
        all_features.append(feature)

    # Group by confidence level
    high_conf = [f for f in all_features if f["_confidence_level"] == "HIGH"]
    medium_conf = [f for f in all_features if f["_confidence_level"] == "MEDIUM"]
    low_conf = [f for f in all_features if f["_confidence_level"] == "LOW"]
    very_low_conf = [f for f in all_features if f["_confidence_level"] == "VERY_LOW"]

    # Build sections
    if high_conf:
        entrance_high = [f for f in high_conf if f["section"] == "entrance"]
        headroom_high = [f for f in high_conf if f["section"] == "headroom"]

        if entrance_high:
            sections.append(create_confidence_section(entrance_high, "HIGH", "entrance"))
        if headroom_high:
            sections.append(create_confidence_section(headroom_high, "HIGH", "headroom"))

    if medium_conf:
        entrance_medium = [f for f in medium_conf if f["section"] == "entrance"]
        headroom_medium = [f for f in medium_conf if f["section"] == "headroom"]

        if entrance_medium:
            sections.append(create_confidence_section(entrance_medium, "MEDIUM", "entrance"))
        if headroom_medium:
            sections.append(create_confidence_section(headroom_medium, "MEDIUM", "headroom"))

    # Include LOW if space permits
    current_length = sum(len(s) for s in sections)

    if low_conf and current_length < max_length * 0.8:
        entrance_low = [f for f in low_conf if f["section"] == "entrance"]
        headroom_low = [f for f in low_conf if f["section"] == "headroom"]

        if entrance_low:
            sections.append(create_confidence_section(entrance_low, "LOW", "entrance"))
        if headroom_low:
            sections.append(create_confidence_section(headroom_low, "LOW", "headroom"))

        current_length = sum(len(s) for s in sections)

    # Include VERY_LOW if significant space remains
    if very_low_conf and current_length < max_length * 0.6:
        entrance_very_low = [f for f in very_low_conf if f["section"] == "entrance"]
        headroom_very_low = [f for f in very_low_conf if f["section"] == "headroom"]

        if entrance_very_low:
            sections.append(create_confidence_section(entrance_very_low, "VERY_LOW", "entrance"))
        if headroom_very_low:
            sections.append(create_confidence_section(headroom_very_low, "VERY_LOW", "headroom"))

    return "\n\n".join(sections)


def get_confidence_summary(features: List[Dict]) -> Dict:
    """
    Get confidence summary statistics for features.

    Args:
        features: List of feature dictionaries

    Returns:
        Dictionary with confidence statistics

    Example:
        >>> summary = get_confidence_summary(features)
        >>> # Returns {"HIGH": 5, "MEDIUM": 3, "LOW": 2, "VERY_LOW": 1}
    """
    summary = {
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
        "VERY_LOW": 0,
        "total": len(features),
        "avg_confidence": 0.0,
    }

    if not features:
        return summary

    total_confidence = 0.0

    for feature in features:
        confidence = feature.get("confidence", 0.5)
        level = get_confidence_level(confidence)
        summary[level] += 1
        total_confidence += confidence

    summary["avg_confidence"] = total_confidence / len(features)

    return summary


def enforce_minimum_confidence(
    features: List[Dict],
    min_confidence: float = 0.3,
) -> List[Dict]:
    """
    Filter features by minimum confidence threshold.

    Args:
        features: List of feature dictionaries
        min_confidence: Minimum confidence threshold

    Returns:
        Filtered list of features

    Example:
        >>> filtered = enforce_minimum_confidence(features, min_confidence=0.5)
        >>> # Returns only features with confidence >= 0.5
    """
    filtered = [f for f in features if f.get("confidence", 0.0) >= min_confidence]

    removed = len(features) - len(filtered)
    if removed > 0:
        logger.info(
            "Filtered out %d features below confidence threshold %.2f",
            removed,
            min_confidence,
        )

    return filtered


def adjust_feature_counts_by_confidence(
    features: List[Dict],
    max_features: int = 20,
    confidence_distribution: Dict[str, float] = None,
) -> List[Dict]:
    """
    Adjust feature counts based on confidence distribution targets.

    Args:
        features: List of features (already prioritized)
        max_features: Maximum total features to include
        confidence_distribution: Target distribution (e.g., {"HIGH": 0.5, "MEDIUM": 0.3, "LOW": 0.2})

    Returns:
        Adjusted list of features

    Example:
        >>> adjusted = adjust_feature_counts_by_confidence(
        ...     features,
        ...     max_features=15,
        ...     confidence_distribution={"HIGH": 0.5, "MEDIUM": 0.3, "LOW": 0.2}
        ... )
    """
    if confidence_distribution is None:
        # Default: prioritize high confidence
        confidence_distribution = {
            "HIGH": 0.5,
            "MEDIUM": 0.3,
            "LOW": 0.15,
            "VERY_LOW": 0.05,
        }

    # Calculate target counts
    high_count = int(max_features * confidence_distribution["HIGH"])
    medium_count = int(max_features * confidence_distribution["MEDIUM"])
    low_count = int(max_features * confidence_distribution.get("LOW", 0.0))
    very_low_count = int(max_features * confidence_distribution.get("VERY_LOW", 0.0))

    # Select features by confidence level
    selected = []

    high_features = [f for f in features if f.get("_confidence_level") == "HIGH"]
    medium_features = [f for f in features if f.get("_confidence_level") == "MEDIUM"]
    low_features = [f for f in features if f.get("_confidence_level") == "LOW"]
    very_low_features = [f for f in features if f.get("_confidence_level") == "VERY_LOW"]

    selected.extend(high_features[:high_count])
    selected.extend(medium_features[:medium_count])
    selected.extend(low_features[:low_count])
    selected.extend(very_low_features[:very_low_count])

    logger.info(
        "Selected %d features: %d HIGH, %d MEDIUM, %d LOW, %d VERY_LOW",
        len(selected),
        len([f for f in selected if f.get("_confidence_level") == "HIGH"]),
        len([f for f in selected if f.get("_confidence_level") == "MEDIUM"]),
        len([f for f in selected if f.get("_confidence_level") == "LOW"]),
        len([f for f in selected if f.get("_confidence_level") == "VERY_LOW"]),
    )

    return selected


# Confidence-based template adjustments
CONFIDENCE_TEMPLATE_ADJUSTMENTS = {
    "HIGH": {
        "prefix": "CRITICAL - Must include: ",
        "emphasis": "prominently featured, detailed specification",
        "placement": "beginning of prompt",
    },
    "MEDIUM": {
        "prefix": "IMPORTANT - Include: ",
        "emphasis": "clear description with key details",
        "placement": "middle of prompt",
    },
    "LOW": {
        "prefix": "OPTIONAL - If space: ",
        "emphasis": "brief mention",
        "placement": "end of prompt",
    },
    "VERY_LOW": {
        "prefix": "",
        "emphasis": "omit unless critical",
        "placement": "omit",
    },
}


def get_confidence_template_adjustment(confidence: float) -> Dict:
    """
    Get template adjustment for a given confidence level.

    Args:
        confidence: Confidence score (0.0 to 1.0)

    Returns:
        Template adjustment dictionary

    Example:
        >>> adjustment = get_confidence_template_adjustment(0.9)
        >>> # Returns {"prefix": "CRITICAL - Must include: ", ...}
    """
    level = get_confidence_level(confidence)
    return CONFIDENCE_TEMPLATE_ADJUSTMENTS[level]
