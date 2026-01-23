"""
Advanced Feature-to-Prompt Converter with Full Feature Utilization

Optimized to use all available data from recommendations.json:
- Feature interactions
- Confidence weighting
- ROAS context
- Multiple positive signals as alternatives
- Statistical context
- Data quality filtering
"""

# flake8: noqa
import logging
from typing import Any, Dict, List, Optional

from .constants import CATEGORIES
from .constants import get_feature_category as _get_feature_category
from .feature_descriptions import get_feature_value_description
from .recommendations_loader import normalize_value


logger = logging.getLogger(__name__)


def _calculate_feature_weight(importance: float, confidence: str) -> float:
    """
    Calculate weighted importance considering confidence.

    Args:
        importance: Importance score
        confidence: Confidence level ("high", "medium", "low")

    Returns:
        Weighted importance score
    """
    confidence_weights = {"high": 1.0, "medium": 0.7, "low": 0.4}
    conf_weight = confidence_weights.get(confidence.lower(), 0.4)
    return importance * conf_weight


def _effective_importance(
    rec: Dict[str, Any],
    recommended_model: Optional[str],
) -> float:
    """
    Choose the importance field based on scorer's recommended_model.
    Falls back to importance_score.
    """
    rm = (recommended_model or "").strip().lower()
    if rm == "classification":
        val = rec.get("importance_classification", None)
        if val is not None:
            return float(val)
    if rm == "regression":
        val = rec.get("importance_regression", None)
        if val is not None:
            return float(val)
    return float(rec.get("importance_score", 0.0) or 0.0)


def _extract_correlation_groups(
    feature_selection: Optional[Dict[str, Any]],
) -> list[list[str]]:
    if not isinstance(feature_selection, dict):
        return []
    groups = feature_selection.get("correlation_groups", [])
    if not isinstance(groups, list):
        return []
    out: list[list[str]] = []
    for g in groups:
        if isinstance(g, list) and all(isinstance(x, str) for x in g):
            out.append(list(g))
    return out


def _apply_correlation_dedupe(
    filtered_features: list[dict],
    *,
    correlation_groups: list[list[str]],
) -> list[dict]:
    """
    Keep only one representative feature per correlation group to reduce
    redundant constraints and prompt conflicts.
    Representative selection: max weighted_importance.
    """
    if not correlation_groups:
        return filtered_features

    by_name = {
        str(f.get("feature")): f
        for f in filtered_features
        if f.get("feature") is not None
    }
    drop: set[str] = set()
    for group in correlation_groups:
        present = [by_name.get(x) for x in group if x in by_name]
        present = [p for p in present if p is not None]
        if len(present) <= 1:
            continue
        present.sort(
            key=lambda x: float(x.get("weighted_importance", 0.0)),
            reverse=True,
        )
        keep_name = str(present[0].get("feature"))
        for p in present[1:]:
            name = str(p.get("feature"))
            if name and name != keep_name:
                drop.add(name)

    return [f for f in filtered_features if str(f.get("feature")) not in drop]


def _select_interaction_overrides(
    interaction_optimized: Optional[List[Dict[str, Any]]],
    *,
    min_improvement_pct: float = 30.0,
    max_prob_std: float = 0.15,
    min_prediction_samples: int = 5,
) -> tuple[Dict[str, str], List[str]]:
    """
    Select interaction-optimized value overrides when they show meaningful
    improvement with acceptable uncertainty.
    Returns:
      - overrides: feature -> value
      - notes: human-readable bullets for audit
    """
    overrides: Dict[str, str] = {}
    notes: List[str] = []
    if not interaction_optimized:
        return overrides, notes
    # Collect candidates first, then pick a non-overlapping subset.
    candidates: List[Dict[str, Any]] = []
    for rec in interaction_optimized:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("type", "")).strip() != "feature_interaction_optimized":
            continue

        f1 = str(rec.get("feature1", "") or "").strip()
        f2 = str(rec.get("feature2", "") or "").strip()
        strategies = rec.get("recommendation_strategies", [])
        if not f1 or not f2 or not isinstance(strategies, list):
            continue

        best = None
        for s in strategies:
            if (
                isinstance(s, dict)
                and s.get("strategy") == "interaction_optimized"
            ):
                best = s
                break
        if not isinstance(best, dict):
            continue

        imp_pct = float(best.get("improvement_percentage", 0.0) or 0.0)
        if imp_pct < min_improvement_pct:
            continue
        # Look up uncertainty in all_combinations_evaluated (rank 1)
        combos = rec.get("all_combinations_evaluated", [])
        prob_std = None
        pred_samples = None
        if isinstance(combos, list):
            for c in combos:
                if (
                    isinstance(c, dict)
                    and int(c.get("rank", 9999) or 9999) == 1
                ):
                    prob_std = c.get(
                        "predicted_high_performance_probability_std",
                        None,
                    )
                    pred_samples = c.get("prediction_samples", None)
                    break

        if prob_std is not None and float(prob_std) > max_prob_std:
            continue
        if (
            pred_samples is not None
            and int(pred_samples) < min_prediction_samples
        ):
            continue

        vals = best.get("values", {})
        if not isinstance(vals, dict):
            continue
        v1 = vals.get(f1)
        v2 = vals.get(f2)
        if v1 is None or v2 is None:
            continue

        candidates.append(
            {
                "f1": f1,
                "f2": f2,
                "v1": normalize_value(str(v1)),
                "v2": normalize_value(str(v2)),
                "imp_pct": imp_pct,
                "prob_std": float(prob_std) if prob_std is not None else 0.0,
                "pred_samples": (
                    int(pred_samples) if pred_samples is not None else 0
                ),
            }
        )
    # Sort: higher improvement first, then lower uncertainty, then more samples.
    candidates.sort(
        key=lambda x: (
            float(x.get("imp_pct", 0.0)),
            -float(x.get("prob_std", 0.0)),
            float(x.get("pred_samples", 0.0)),
        ),
        reverse=True,
    )

    for c in candidates:
        f1 = str(c["f1"])
        f2 = str(c["f2"])
        if f1 in overrides or f2 in overrides:
            # Avoid overwriting a feature with multiple interaction blocks.
            continue
        overrides[f1] = str(c["v1"])
        overrides[f2] = str(c["v2"])
        notes.append(
            f"- Apply interaction_optimized for {f1} × {f2}: "
            f"{f1}={overrides[f1]}, {f2}={overrides[f2]} "
            f"(improvement {float(c.get('imp_pct', 0.0)):.1f}%)"
        )

    return overrides, notes


def _is_high_quality(feature_rec: Dict[str, Any]) -> bool:
    """
    Check if feature recommendation has sufficient data quality.

    Args:
        feature_rec: Feature recommendation dict

    Returns:
        True if high quality, False otherwise
    """
    dq = feature_rec.get("data_quality", {})
    sample_size = dq.get("sample_size", 0)
    power = dq.get("statistical_power", "low")
    # Require minimum sample size
    if sample_size < 5:
        return False
    # Prefer higher statistical power
    if power == "low" and sample_size < 10:
        return False

    return True


def _get_roas_context(feature_rec: Dict[str, Any], value: str) -> str:
    """
    Get ROAS comparison context for a feature value.

    Args:
        feature_rec: Feature recommendation dict
        value: Feature value

    Returns:
        ROAS context string (empty if not available)
    """
    value_comp = feature_rec.get("value_comparison", {})
    normalized_value = normalize_value(str(value))

    if normalized_value not in value_comp:
        return ""

    roas_data = value_comp[normalized_value]
    mean_roas = roas_data.get("mean_roas", 0)
    count = roas_data.get("count", 0)
    # Find best alternative for comparison
    best_alt_roas = 0
    for alt_value, alt_data in value_comp.items():
        if alt_value != normalized_value:
            alt_roas = alt_data.get("mean_roas", 0)
            best_alt_roas = max(best_alt_roas, alt_roas)

    if 0 < best_alt_roas < mean_roas:
        return f" (ROAS: {mean_roas:.2f} vs {best_alt_roas:.2f}, n={count})"
    if mean_roas > 0:
        return f" (ROAS: {mean_roas:.2f}, n={count})"

    return ""


def _get_statistical_context(feature_rec: Dict[str, Any]) -> str:
    """
    Get statistical significance context.

    Args:
        feature_rec: Feature recommendation dict

    Returns:
        Statistical context string (empty if not significant)
    """
    p_value = feature_rec.get("p_value_1a")
    cramers_v = feature_rec.get("cramers_v_1a")
    confidence = feature_rec.get("confidence", "low")

    if confidence == "high" and p_value and p_value < 0.05:
        if cramers_v:
            if cramers_v > 0.3:
                effect_size = "large"
            elif cramers_v > 0.2:
                effect_size = "medium"
            else:
                effect_size = "small"
            return (
                f" [statistically significant, p<{p_value:.3f}, "
                f"{effect_size} effect]"
            )
        return f" [statistically significant, p<{p_value:.3f}]"

    return ""


def _get_value_alternatives(feature_rec: Dict[str, Any]) -> List[str]:
    """
    Get alternative values if multiple positive signals exist.

    Args:
        feature_rec: Feature recommendation dict

    Returns:
        List of alternative values
    """
    pos_signals = feature_rec.get("positive_signals", [])
    recommended = feature_rec.get("recommended_value", "")

    if not recommended:
        return []
    # Normalize
    normalized_rec = normalize_value(str(recommended))
    alternatives = []
    value_comp = feature_rec.get("value_comparison", {})

    for signal in pos_signals:
        normalized_signal = normalize_value(str(signal))
        if (
            normalized_signal != normalized_rec
            and normalized_signal in value_comp
        ):
            # Check if alternative has good ROAS
            alt_data = value_comp[normalized_signal]
            alt_roas = alt_data.get("mean_roas", 0)
            if alt_roas > 1.5:  # Only include if decent ROAS
                alternatives.append(normalized_signal)

    return alternatives[:2]  # Limit to top 2 alternatives


def _format_feature_description_advanced(
    feature_name: str,
    value: str,
    feature_rec: Dict[str, Any],
    is_negative: bool = False,
    include_roas_in_prompt: bool = True,
    include_reason_in_prompt: bool = True,
    include_alternatives_in_prompt: bool = True,
    include_stats_in_prompt: bool = True,
) -> str:
    """
    Format a feature as rich natural language description with context.

    Args:
        feature_name: Name of the feature
        value: Feature value
        feature_rec: Full feature recommendation dict
        is_negative: Whether this is a negative instruction

    Returns:
        Formatted natural language description with context
    """
    # Get rich description of the value
    value_desc = get_feature_value_description(feature_name, value)
    # Build description
    if is_negative:
        if value_desc:
            desc = f"AVOID: {value_desc}"
        else:
            value_display = value.replace("_", " ").title()
            desc = f"AVOID: {value_display}"
    else:
        if value_desc:
            desc = value_desc
        else:
            value_display = value.replace("_", " ").title()
            desc = value_display
        # Add ROAS context (optional; can hurt edit-model consistency)
        if include_roas_in_prompt:
            roas_ctx = _get_roas_context(feature_rec, value)
            if roas_ctx:
                desc += roas_ctx
        # Add statistical context
        if include_stats_in_prompt:
            stat_ctx = _get_statistical_context(feature_rec)
            if stat_ctx:
                desc += stat_ctx
        # Add reason field if available (concise explanation)
        if include_reason_in_prompt:
            reason = feature_rec.get("reason", "")
            if reason and not is_negative:
                # Keep reason concise (max 80 chars) to avoid long prompts
                reason_short = (
                    reason[:80] + "..." if len(reason) > 80 else reason
                )
                desc += f" (Reason: {reason_short})"
        # Add alternatives if available
        if include_alternatives_in_prompt:
            alternatives = _get_value_alternatives(feature_rec)
            if alternatives and not is_negative:
                alt_str = " or ".join(
                    [a.replace("_", " ").title() for a in alternatives]
                )
                desc += f" (Alternative: {alt_str})"

    return desc


def _build_interaction_section(
    interactions: List[Dict[str, Any]], feature_values: Dict[str, str]
) -> List[str]:
    """
    Build section for feature interactions.

    Args:
        interactions: List of interaction dicts
        feature_values: Dict of current feature values

    Returns:
        List of interaction section lines
    """
    relevant_interactions = []
    for interaction in interactions:
        feat1 = interaction.get("feature1")
        feat2 = interaction.get("feature2")
        if feat1 in feature_values and feat2 in feature_values:
            relevant_interactions.append(interaction)

    if not relevant_interactions:
        return []
    # Sort by strength (descending)
    relevant_interactions.sort(key=lambda x: x.get("strength", 0), reverse=True)

    lines = ["🔗 FEATURE INTERACTIONS:"]
    for interaction in relevant_interactions:
        recommendation = interaction.get("recommendation", "")
        strength = interaction.get("strength", 0)
        if recommendation:
            lines.append(f"  - {recommendation}")
        else:
            feat1 = interaction.get("feature1", "")
            feat2 = interaction.get("feature2", "")
            lines.append(
                f"  - {feat1} × {feat2}: "
                f"strong interaction (strength: {strength:.3f})"
            )

    return lines


def _build_critical_constraints_section(
    features: List[Dict[str, Any]],
    feature_recs: Dict[str, Dict[str, Any]],
) -> List[str]:
    """
    Build critical constraints section (high-weighted features).

    Args:
        features: List of feature dicts with 'feature', 'value', 'instruction'
        feature_recs: Dict mapping feature names to full recommendation dicts

    Returns:
        List of constraint lines
    """
    # Calculate weighted importance for each feature
    weighted_features = []
    for feat in features:
        feature_name = feat["feature"]
        feature_rec = feature_recs.get(feature_name, {})
        importance = feature_rec.get("importance_score", 0.0)
        confidence = feature_rec.get("confidence", "low")
        weight = _calculate_feature_weight(importance, confidence)

        if weight >= 10.0:  # High weighted importance
            weighted_features.append((weight, feat))

    if not weighted_features:
        return []
    # Sort by weighted importance
    weighted_features.sort(key=lambda x: x[0], reverse=True)

    lines = ["CRITICAL REQUIREMENTS:"]
    for weight, feat in weighted_features:
        lines.append(f"  {feat['instruction']}")

    return lines


def _build_category_section(
    category: str,
    features: List[Dict[str, Any]],
) -> List[str]:
    """
    Build a category section with natural language descriptions.

    Args:
        category: Category name (lighting, composition, etc.)
        features: List of feature dicts

    Returns:
        List of section lines
    """
    if not features:
        return []
    category_title = category.replace("_", " ").title()
    title = f"{category_title.upper()} REQUIREMENTS:"

    lines = [title]

    for feat in features:
        lines.append(f"  {feat['instruction']}")

    return lines


def _build_negative_constraints_section(
    negative_features: List[Dict[str, Any]],
) -> List[str]:
    """
    Build negative constraints section (what to avoid).

    Args:
        negative_features: List of negative feature dicts

    Returns:
        List of constraint lines
    """
    if not negative_features:
        return []

    lines = ["AVOID THESE ELEMENTS:"]

    for feat in negative_features:
        lines.append(f"  {feat['instruction']}")

    return lines


def _build_technical_specs_section() -> List[str]:
    """
    Build technical photography specifications section.

    Returns:
        List of technical spec lines
    """
    return [
        "TECHNICAL PHOTOGRAPHY SPECIFICATIONS:",
        "  - Shot with a full-frame camera, 50mm or 85mm lens",
        "  - Aperture: f/2.8–f/4 with natural depth of field",
        "  - Realistic lighting with believable direction and falloff",
        "  - Consistent shadows matching light position",
        "  - Natural material imperfections and surface textures",
        "  - Professional product photography quality",
    ]


def convert_features_to_advanced_prompt(
    feature_recommendations: List[Dict[str, Any]],
    interactions: Optional[List[Dict[str, Any]]] = None,
    interaction_optimized: Optional[List[Dict[str, Any]]] = None,
    necessary_conditions: Optional[List[Dict[str, Any]]] = None,
    feature_selection: Optional[Dict[str, Any]] = None,
    recommended_model: Optional[str] = None,
    min_importance: float = 0.0,
    min_weighted_importance: float = 0.0,
    filter_by_quality: bool = True,
    include_technical_specs: bool = True,
    include_roas_in_prompt: bool = True,
    include_reason_in_prompt: bool = True,
    include_alternatives_in_prompt: bool = True,
    include_stats_in_prompt: bool = True,
) -> Dict[str, Any]:
    """
    Convert feature recommendations to advanced structured prompt.

    Uses all available data: interactions, confidence, ROAS, statistics.

    Args:
        feature_recommendations: List of full feature recommendation dicts
        interactions: Optional list of feature interaction dicts
        min_importance: Minimum importance score to include
        min_weighted_importance: Minimum weighted importance (after confidence)
        filter_by_quality: Whether to filter by data quality
        include_technical_specs: Whether to include technical photography specs

    Returns:
        Dict with advanced prompt sections
    """
    # Filter by quality if requested
    if filter_by_quality:
        feature_recommendations = [
            rec for rec in feature_recommendations if _is_high_quality(rec)
        ]

    correlation_groups = _extract_correlation_groups(feature_selection)
    overrides, override_notes = _select_interaction_overrides(
        interaction_optimized
    )
    # Build feature_recs dict for quick lookup
    feature_recs = {}
    for rec in feature_recommendations:
        feature_name = rec.get("feature")
        if feature_name:
            feature_recs[feature_name] = rec
    # Filter and calculate weighted importance
    filtered_features = []
    for rec in feature_recommendations:
        feature_name = rec.get("feature")
        if not feature_name:
            continue

        importance = _effective_importance(rec, recommended_model)
        if importance < min_importance:
            continue

        confidence = rec.get("confidence", "low")
        weighted_importance = _calculate_feature_weight(importance, confidence)

        if weighted_importance < min_weighted_importance:
            continue
        # Use interaction-optimized overrides when available
        recommended_value = overrides.get(feature_name) or rec.get(
            "recommended_value"
        )
        if not recommended_value:
            # Try positive_signals
            pos_signals = rec.get("positive_signals", [])
            if pos_signals:
                recommended_value = pos_signals[0]
            else:
                continue

        normalized_value = normalize_value(str(recommended_value))

        filtered_features.append(
            {
                "feature": feature_name,
                "value": normalized_value,
                "importance": importance,
                "weighted_importance": weighted_importance,
                "confidence": confidence,
                "rec": rec,
            }
        )
    # Sort by weighted importance (descending)
    filtered_features.sort(key=lambda f: f["weighted_importance"], reverse=True)
    # De-duplicate correlated features
    filtered_features = _apply_correlation_dedupe(
        filtered_features, correlation_groups=correlation_groups
    )
    # Organize features by category
    category_features = {cat: [] for cat in CATEGORIES}
    category_features["other"] = []
    all_feature_dicts = []
    # Process positive features
    for feat_data in filtered_features:
        feature_name = feat_data["feature"]
        value = feat_data["value"]
        feature_rec = feat_data["rec"]
        category = _get_feature_category(feature_name)

        instruction = _format_feature_description_advanced(
            feature_name,
            value,
            feature_rec,
            is_negative=False,
            include_roas_in_prompt=include_roas_in_prompt,
            include_reason_in_prompt=include_reason_in_prompt,
            include_alternatives_in_prompt=include_alternatives_in_prompt,
            include_stats_in_prompt=include_stats_in_prompt,
        )

        feat_dict = {
            "feature": feature_name,
            "value": value,
            "instruction": instruction,
            "weighted_importance": feat_data["weighted_importance"],
        }

        category_features[category].append(feat_dict)
        all_feature_dicts.append(feat_dict)
    # Process negative features
    negative_feature_dicts = []
    for rec in feature_recommendations:
        feature_name = rec.get("feature")
        if not feature_name:
            continue

        negative_signals = rec.get("negative_signals", [])
        if not negative_signals:
            continue
        # Use first negative signal (advanced converter stays conservative here;
        # loader already can compute negative_feature_values via value_comparison).
        negative_value = normalize_value(str(negative_signals[0]))

        instruction = _format_feature_description_advanced(
            feature_name,
            negative_value,
            rec,
            is_negative=True,
            include_roas_in_prompt=include_roas_in_prompt,
            include_reason_in_prompt=include_reason_in_prompt,
            include_alternatives_in_prompt=include_alternatives_in_prompt,
            include_stats_in_prompt=include_stats_in_prompt,
        )

        feat_dict = {
            "feature": feature_name,
            "value": negative_value,
            "instruction": instruction,
        }
        negative_feature_dicts.append(feat_dict)

    def _necessary_conditions_section(
        items: Optional[List[Dict[str, Any]]],
    ) -> List[str]:
        if not items:
            return []
        rows = [x for x in items if isinstance(x, dict)]
        if not rows:
            return []
        out: List[str] = []
        out.append("NECESSARY CONDITIONS (HIGH PERFORMERS):")
        out.append(
            "- Predictive, high-coverage conditions found in top performers."
        )
        for it in rows[:10]:
            feat = str(it.get("feature", "") or "").strip()
            val = str(it.get("value", "") or "").strip()
            strength = str(it.get("strength", "") or "").strip()
            if feat and val:
                suffix = f" ({strength})" if strength else ""
                out.append(f"- MUST INCLUDE: {feat} = {val}{suffix}")
        out.append("")
        return out

    # Build structured prompt sections
    prompt_sections = []
    # 0. Necessary conditions (if provided)
    prompt_sections.extend(_necessary_conditions_section(necessary_conditions))
    # 0.5 Interaction overrides (audit)
    if override_notes:
        prompt_sections.append("🧩 INTERACTION OPTIMIZATION APPLIED:")
        prompt_sections.extend([f"  {n}" for n in override_notes])
        prompt_sections.append("")
    # 1. Critical constraints (high-weighted features)
    critical_section = _build_critical_constraints_section(
        all_feature_dicts, feature_recs
    )
    if critical_section:
        prompt_sections.extend(critical_section)
        prompt_sections.append("")  # Blank line separator
    # 2. Feature interactions
    interaction_section = []
    if interactions:
        interaction_section = _build_interaction_section(
            interactions, {f["feature"]: f["value"] for f in all_feature_dicts}
        )
        if interaction_section:
            prompt_sections.extend(interaction_section)
            prompt_sections.append("")  # Blank line separator
    # 3. Category-based sections
    category_order = [
        "lighting",
        "composition",
        "content",
        "visual_style",
        "product",
        "background",
        "other",
    ]
    for category in category_order:
        if category_features[category]:
            section = _build_category_section(
                category, category_features[category]
            )
            if section:
                prompt_sections.extend(section)
                prompt_sections.append("")  # Blank line separator
    # 4. Negative constraints
    if negative_feature_dicts:
        negative_section = _build_negative_constraints_section(
            negative_feature_dicts
        )
        if negative_section:
            prompt_sections.extend(negative_section)
            prompt_sections.append("")  # Blank line separator
    # 5. Technical specifications
    if include_technical_specs:
        tech_section = _build_technical_specs_section()
        prompt_sections.extend(tech_section)
    # Combine into final prompt
    advanced_prompt = "\n".join(prompt_sections)

    return {
        "advanced_prompt": advanced_prompt,
        "sections": {
            "critical": critical_section if critical_section else [],
            "interactions": interaction_section if interactions else [],
            "categories": {
                cat: category_features[cat]
                for cat in category_order
                if category_features.get(cat)
            },
            "negative": negative_feature_dicts,
            "technical": (
                (
                    _build_technical_specs_section()
                    if include_technical_specs
                    else []
                )
            ),
        },
        "all_features": all_feature_dicts,
        "negative_features": negative_feature_dicts,
        "stats": {
            "total_features": len(all_feature_dicts),
            "high_confidence": sum(
                1 for f in filtered_features if f["confidence"] == "high"
            ),
            "medium_confidence": sum(
                1 for f in filtered_features if f["confidence"] == "medium"
            ),
            "low_confidence": sum(
                1 for f in filtered_features if f["confidence"] == "low"
            ),
        },
    }
