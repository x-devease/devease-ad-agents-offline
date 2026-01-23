"""
Variant builder for recommendations.json.

Goal:
- Produce multiple internally consistent feature-value combinations
  ("variants") from one recommendations.json.
- Use scorer signals:
  - value_comparison (multiple strong values per feature)
  - feature_interaction_optimized (pairwise optimized bundles)
  - conflicts + alternatives (explicit compatibility repair moves)

All content is intentionally in English.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .recommendations_loader import normalize_value


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariantSpec:
    """A concrete feature assignment + audit metadata."""

    name: str
    assignments: Dict[str, str]  # feature -> normalized value
    notes: List[str]


def _parse_diversity_features(values: Optional[Iterable[str]]) -> List[str]:
    out: List[str] = []
    if not values:
        return out
    for val in values:
        if val is None:
            continue
        # Support comma-separated or repeated flags.
        parts = [p.strip() for p in str(val).split(",") if p.strip()]
        out.extend(parts)
    # Preserve order, drop duplicates.
    seen = set()
    uniq: List[str] = []
    for f in out:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq


def _best_value_candidates(
    rec: Dict[str, Any],
    *,
    min_count: int,
    max_values: int,
) -> List[str]:
    """
    Build a ranked list of candidate values for one feature.
    Returned values are normalized.
    """
    candidates: List[Tuple[float, str]] = []
    # 1) recommended_value / positive_signals
    rec_val = rec.get("recommended_value", None)
    if rec_val is None:
        pos = rec.get("positive_signals", [])
        if isinstance(pos, list) and pos:
            rec_val = pos[0]
    if rec_val is not None:
        candidates.append((float("inf"), normalize_value(str(rec_val))))
    # 2) value_comparison: sort by best_metric desc with count >= min_count
    val_comp = rec.get("value_comparison", None)
    if isinstance(val_comp, dict):
        for raw_val, stats in val_comp.items():
            if not isinstance(stats, dict):
                continue
            try:
                cnt = int(stats.get("count", 0) or 0)
            except Exception:  # pylint: disable=broad-exception-caught
                cnt = 0
            if cnt < min_count:
                continue
            best_metric = stats.get("best_metric", None)
            if best_metric is None:
                continue
            try:
                score = float(best_metric)
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            candidates.append((score, normalize_value(str(raw_val))))
    # 3) positive_signals (as additional options)
    pos2 = rec.get("positive_signals", [])
    if isinstance(pos2, list):
        for p in pos2:
            if p is None:
                continue
            # Lower priority than explicit value_comparison; still useful.
            candidates.append((float("-inf"), normalize_value(str(p))))
    # Dedupe while preserving best score for each value
    best: Dict[str, float] = {}
    for score, val in candidates:
        if val not in best or score > best[val]:
            best[val] = score
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return [v for v, _ in ranked][:max_values]


def summarize_available_combinations(
    *,
    feature_recommendations: List[Dict[str, Any]],
    interaction_optimized: Optional[List[Dict[str, Any]]] = None,
    value_comparison_min_count: int = 5,
    per_feature_max_values: int = 5,
) -> Dict[str, Any]:
    """
    Offline-only summary of what combinations are available.

    This does NOT call any APIs. It only inspects recommendations.json content.
    """
    per_feature: Dict[str, Any] = {}
    for rec in feature_recommendations:
        feat = str(rec.get("feature", "") or "").strip()
        if not feat:
            continue
        candidates = _best_value_candidates(
            rec,
            min_count=value_comparison_min_count,
            max_values=per_feature_max_values,
        )
        per_feature[feat] = {
            "recommended_value": normalize_value(
                str(rec.get("recommended_value", "") or "")
            ),
            "candidates": candidates,
            "confidence": rec.get("confidence"),
            "importance_score": rec.get("importance_score"),
            "has_conflicts": bool(rec.get("has_conflicts")),
        }

    conflicts = _collect_conflicts(feature_recommendations)

    interactions_out: List[Dict[str, Any]] = []
    if isinstance(interaction_optimized, list):
        for rec in interaction_optimized:
            if not isinstance(rec, dict):
                continue
            if (
                str(rec.get("type", "")).strip()
                != "feature_interaction_optimized"
            ):
                continue
            f1 = str(rec.get("feature1", "") or "").strip()
            f2 = str(rec.get("feature2", "") or "").strip()
            if not f1 or not f2:
                continue
            best = None
            strategies = rec.get("recommendation_strategies", [])
            if isinstance(strategies, list):
                for s in strategies:
                    if (
                        isinstance(s, dict)
                        and s.get("strategy") == "interaction_optimized"
                    ):
                        best = s
                        break
            if isinstance(best, dict):
                vals = (best or {}).get("values", {})
            else:
                vals = {}
            interactions_out.append(
                {
                    "feature1": f1,
                    "feature2": f2,
                    "improvement_percentage": (best or {}).get(
                        "improvement_percentage"
                    ),
                    "values": (
                        {
                            f1: normalize_value(str(vals.get(f1, "") or "")),
                            f2: normalize_value(str(vals.get(f2, "") or "")),
                        }
                        if isinstance(vals, dict)
                        else {}
                    ),
                }
            )

    return {
        "per_feature_candidates": per_feature,
        "interaction_optimized_pairs": interactions_out,
        "conflicts": conflicts,
    }


def _collect_conflicts(
    feature_recommendations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in feature_recommendations:
        conflicts = rec.get("conflicts", [])
        if not isinstance(conflicts, list):
            continue
        for c in conflicts:
            if isinstance(c, dict):
                out.append(c)
    return out


def _conflict_matches(
    c: Dict[str, Any],
    assignments: Dict[str, str],
) -> bool:
    f1 = str(c.get("feature1", "") or "").strip()
    f2 = str(c.get("feature2", "") or "").strip()
    v1 = normalize_value(str(c.get("value1", "") or ""))
    v2 = normalize_value(str(c.get("value2", "") or ""))
    if not f1 or not f2 or not v1 or not v2:
        return False
    return assignments.get(f1) == v1 and assignments.get(f2) == v2


def _choose_conflict_repair(
    c: Dict[str, Any],
    *,
    locked_features: set[str],
) -> Optional[Tuple[str, str, str]]:
    """
    Return (feature, new_value, note) if a repair is possible.
    Chooses between alternative1/alternative2 using mean_roas, then count.
    """
    a1 = c.get("alternative1", None)
    a2 = c.get("alternative2", None)
    alts = []
    for a in (a1, a2):
        if not isinstance(a, dict):
            continue
        feat = str(a.get("feature", "") or "").strip()
        val = a.get("value", None)
        if not feat or val is None:
            continue
        if feat in locked_features:
            continue
        try:
            mean_roas = float(a.get("mean_roas", 0.0) or 0.0)
        except Exception:  # pylint: disable=broad-exception-caught
            mean_roas = 0.0
        try:
            count = int(a.get("count", 0) or 0)
        except Exception:  # pylint: disable=broad-exception-caught
            count = 0
        alts.append(
            (
                mean_roas,
                count,
                feat,
                normalize_value(str(val)),
                a,
            )
        )

    if not alts:
        return None

    alts.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = alts[0]
    desc = str(best[4].get("description", "") or "").strip()
    note = desc or str(c.get("recommendation", "") or "").strip()
    if not note:
        note = "Conflict repaired."
    return best[2], best[3], note


def _resolve_conflicts(
    assignments: Dict[str, str],
    conflicts: List[Dict[str, Any]],
    *,
    locked_features: set[str],
    max_passes: int = 20,
) -> List[str]:
    """
    Iteratively apply scorer-provided alternatives until no conflicts remain
    or max passes reached.
    """
    notes: List[str] = []
    for _ in range(max_passes):
        any_change = False
        for c in conflicts:
            if not _conflict_matches(c, assignments):
                continue
            repair = _choose_conflict_repair(
                c,
                locked_features=locked_features,
            )
            if not repair:
                continue
            feat, new_val, note = repair
            if assignments.get(feat) == new_val:
                continue
            assignments[feat] = new_val
            notes.append(f"- Conflict repair: set {feat}={new_val}. {note}")
            any_change = True
        if not any_change:
            break
    return notes


def apply_variant_overrides_to_feature_recommendations(
    feature_recommendations: List[Dict[str, Any]],
    *,
    assignments: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Return a deep-copied list of feature recommendations where
    recommended_value is replaced by the variant assignment (when present).
    """
    recs = copy.deepcopy(feature_recommendations)
    for rec in recs:
        feat = rec.get("feature")
        if feat and feat in assignments:
            rec["recommended_value"] = assignments[feat]
            rec["variant_forced_value"] = True
    return recs


def build_feature_variants(
    *,
    feature_recommendations: List[Dict[str, Any]],
    interaction_optimized: Optional[List[Dict[str, Any]]] = None,
    k: int = 3,
    diversity_features: Optional[Iterable[str]] = None,
    locked_features: Optional[Iterable[str]] = None,
    value_comparison_min_count: int = 5,
    per_feature_max_values: int = 5,
) -> List[VariantSpec]:
    """
    Produce K variant feature assignments.

    Current strategy (practical + robust):
    - Variant 1: baseline (recommended values), then conflict repair.
    - Variants 2..K: change one or two "diversity" features to their next-best
      candidate values (from value_comparison), then conflict repair.

    Note: interaction_optimized is not yet used to *enumerate* variants here;
    it is handled downstream by the advanced converter. This keeps variant
    generation deterministic and avoids producing contradictory overrides.
    """
    if k <= 0:
        return []

    div = _parse_diversity_features(diversity_features) or [
        "human_elements",
        "product_visibility",
        "color_balance",
        "primary_colors",
        "visual_impact",
    ]
    locked = set(_parse_diversity_features(locked_features))
    if interaction_optimized:
        logger.debug(
            "Received %d interaction_optimized blocks; "
            "variants currently vary "
            "per-feature values and rely on the advanced converter for "
            "interaction text.",
            len(interaction_optimized),
        )
    # Baseline assignment from recommended_value (normalized).
    baseline: Dict[str, str] = {}
    candidates_by_feature: Dict[str, List[str]] = {}
    for rec in feature_recommendations:
        feat = rec.get("feature")
        if not feat:
            continue
        cands = _best_value_candidates(
            rec,
            min_count=value_comparison_min_count,
            max_values=per_feature_max_values,
        )
        if not cands:
            continue
        candidates_by_feature[str(feat)] = cands
        baseline[str(feat)] = cands[0]

    conflicts = _collect_conflicts(feature_recommendations)

    base_assign = dict(baseline)
    base_notes = _resolve_conflicts(
        base_assign,
        conflicts,
        locked_features=locked,
    )
    variants: List[VariantSpec] = [
        VariantSpec(
            name="variant_001_baseline",
            assignments=base_assign,
            notes=["- Baseline assignment (recommended values)."] + base_notes,
        )
    ]
    # Build additional variants by perturbing diversity features.
    seen_signatures = {tuple((f, base_assign.get(f, "")) for f in div)}
    # Round-robin over diversity features, try next candidates.
    attempt = 0
    while len(variants) < k and attempt < 200:
        attempt += 1

        idx = len(variants)  # 1-based offset from baseline
        # Choose a primary feature to vary
        primary = div[(idx - 1) % len(div)]
        cands = candidates_by_feature.get(primary, [])
        if len(cands) < 2:
            continue
        # pick next-best candidate (cycle if k > number of candidates)
        pick_i = 1 + ((idx - 1) // len(div)) % (len(cands) - 1)
        new_val = cands[pick_i]

        assign = dict(base_assign)
        assign[primary] = new_val
        notes = [
            f"- Diversity: set {primary}={new_val} "
            f"(candidate rank {pick_i+1})."
        ]
        # Optionally vary a second diversity feature to increase distinctness.
        secondary = div[(idx) % len(div)]
        if secondary != primary:
            sc = candidates_by_feature.get(secondary, [])
            if len(sc) >= 2:
                s_pick = 1
                s_val = sc[s_pick]
                if s_val != assign.get(secondary):
                    assign[secondary] = s_val
                    notes.append(
                        f"- Diversity: set {secondary}={s_val} "
                        f"(candidate rank {s_pick+1})."
                    )

        notes.extend(
            _resolve_conflicts(assign, conflicts, locked_features=locked)
        )
        sig = tuple((f, assign.get(f, "")) for f in div)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        variants.append(
            VariantSpec(
                name=f"variant_{len(variants)+1:03d}",
                assignments=assign,
                notes=notes,
            )
        )

    if len(variants) < k:
        logger.warning(
            "Requested %s variants but only built %s "
            "(insufficient diversity candidates).",
            k,
            len(variants),
        )

    return variants[:k]
