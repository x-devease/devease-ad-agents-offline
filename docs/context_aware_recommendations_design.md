# Context-Aware Ad Miner: Design Document

**Author:** Claude
**Date:** 2026-01-27
**Status:** Design Proposal
**Branch:** ad-reviewer

---

## Problem Statement

Current ad miner outputs **generic recommendations** that don't account for:
- Campaign goals (awareness vs conversion vs traffic)
- Product identity (Power Station vs MoProBo vs others)
- Branch-specific patterns (regional aesthetic preferences)

**Result:** One-size-fits-all recommendations that may not apply to specific contexts.

---

## Design Goals

1. **Context-aware**: Recommendations tailored to campaign goals, products, and branches
2. **Data-driven**: Only use segmented patterns when statistically significant
3. **Graceful degradation**: Fall back to broader aggregations when insufficient data
4. **Backward compatible**: Existing workflows continue to work
5. **Queryable**: Easy to retrieve recommendations for any context level

---

## Architecture Overview

### Current Architecture (Generic)
```
Input: all_creatives.csv (single dataset)
         ↓
Analysis: Top 25% vs Bottom 25% (all data combined)
         ↓
Output: config/ad/miner/{customer}/{platform}/recommendations.md
```

### Proposed Architecture (Segmented)
```
Input: all_creatives.csv (with metadata columns: goal, product, branch)
         ↓
Segmentation: Split by (customer, product, branch, goal)
         ↓
Analysis: Per-segment pattern detection
         ↓
Aggregation: Multi-level hierarchy with fallback
         ↓
Output: config/ad/miner/{customer}/{product}/{branch}/{goal}/recommendations.json
```

---

## Granularity Levels

### Level 1: Most Specific
**Key:** `(customer, product, branch, campaign_goal)`

**Example:** `moprobo + Power Station + US + conversion`

**Minimum sample size:** 200 creatives per segment

**Use case:** Highly tailored recommendations for specific product-market-goal combinations

---

### Level 2: Product-Goal Specific
**Key:** `(customer, product, campaign_goal)`

**Example:** `moprobo + Power Station + conversion` (all branches)

**Minimum sample size:** 100 creatives per segment

**Use case:** Product-specific patterns that apply across branches

---

### Level 3: Goal-Specific
**Key:** `(customer, campaign_goal)`

**Example:** `moprobo + conversion` (all products, all branches)

**Minimum sample size:** 50 creatives per segment

**Use case:** Goal-specific patterns (e.g., "conversion campaigns always need product prominence")

---

### Level 4: Generic Fallback
**Key:** `(customer)`

**Example:** `moprobo` (all products, branches, goals)

**Minimum sample size:** None (always available)

**Use case:** Current generic recommendations (backward compatibility)

---

## Data Structure Changes

### 1. Input CSV: Add Metadata Columns

**Current schema:**
```csv
creative_id, filename, roas, direction, lighting_style, ...
```

**New schema:**
```csv
creative_id, filename, roas, campaign_goal, product, branch, direction, lighting_style, ...
```

**Values:**
- `campaign_goal`: `awareness | conversion | traffic | lead_generation | app_installs`
- `product`: `Power Station | MoProBo | Smart Watch | ...`
- `branch`: `US | EU | APAC | LATAM | Global`

---

### 2. Output: Structured JSON with Metadata

**File path:** `config/ad/miner/{customer}/{product}/{branch}/{goal}/recommendations.json`

**Structure:**
```json
{
  "metadata": {
    "customer": "moprobo",
    "product": "Power Station",
    "branch": "US",
    "campaign_goal": "conversion",
    "granularity_level": 1,
    "sample_size": 342,
    "min_threshold": 200,
    "analysis_date": "2026-01-27",
    "fallback_used": false
  },
  "high_impact_recommendations": [
    {
      "feature": "product_position",
      "value": "bottom-right",
      "roas_lift_multiple": 2.8,
      "roas_lift_pct": 180,
      "top_quartile_prevalence": 0.67,
      "confidence": "high",
      "type": "DO",
      "goal_specific": true,
      "reason": "For conversion campaigns, products positioned bottom-right show 2.8x higher ROAS. Present in 67% of top performers."
    }
  ],
  "negative_guidance": [...],
  "low_priority_insights": [...]
}
```

---

### 3. Segment Index File

**Path:** `config/ad/miner/{customer}/segment_index.json`

**Purpose:** Track available segments and their granularity levels.

**Structure:**
```json
{
  "customer": "moprobo",
  "last_updated": "2026-01-27",
  "segments": [
    {
      "product": "Power Station",
      "branch": "US",
      "campaign_goal": "conversion",
      "granularity_level": 1,
      "sample_size": 342,
      "file_path": "config/ad/miner/moprobo/Power_Station/US/conversion/recommendations.json"
    },
    {
      "product": "Power Station",
      "branch": "EU",
      "campaign_goal": "conversion",
      "granularity_level": 2,
      "sample_size": 145,
      "file_path": "config/ad/miner/moprobo/Power_Station/conversion/recommendations.json"
    },
    {
      "product": "MoProBo",
      "branch": "US",
      "campaign_goal": "awareness",
      "granularity_level": 3,
      "sample_size": 78,
      "file_path": "config/ad/miner/moprobo/awareness/recommendations.json"
    }
  ]
}
```

---

## Algorithm Design

### Pattern Detection: Goal-Specific Analysis

**Hypothesis:** Different goals have different optimal creative patterns.

**Example:**
- **Conversion campaigns**: Product prominence, clear CTA, studio lighting
- **Awareness campaigns**: Lifestyle context, emotional storytelling, natural lighting
- **Traffic campaigns**: Bold colors, high contrast, eye-catching visuals

**Algorithm:**
```python
def detect_goal_specific_patterns(df, goal_column, features):
    """
    Detect patterns that are specific to campaign goals.

    For each feature, test if:
    1. Feature value distribution differs significantly by goal (chi-square)
    2. Feature-ROAS relationship differs by goal (interaction test)
    3. Feature value is uniquely high-performing for a specific goal
    """
    goal_specific_patterns = {}

    for goal in df[goal_column].unique():
        goal_df = df[df[goal_column] == goal]

        # Find features that are especially important for this goal
        for feature in features:
            # Test: Is this feature more predictive for this goal vs others?
            goal_lift = calculate_goal_specific_lift(
                df, goal, feature, goal_column
            )

            if goal_lift > 1.5:  # Threshold
                goal_specific_patterns[(goal, feature)] = {
                    "lift": goal_lift,
                    "confidence": calculate_significance(df, goal, feature),
                    "reason": f"For {goal} campaigns, this feature shows {goal_lift}x higher impact"
                }

    return goal_specific_patterns
```

---

### Pattern Detection: Product-Specific Analysis

**Hypothesis:** Different products have different optimal creative treatments.

**Example:**
- **Power Station**: Technical product → needs clear features, professional lighting
- **MoProBo**: Consumer gadget → benefits from lifestyle context, people using it
- **Smart Watch**: Fashion accessory → emphasizes aesthetics, color coordination

**Algorithm:**
```python
def detect_product_specific_patterns(df, product_column, features):
    """
    Detect patterns that are specific to products.
    """
    product_patterns = {}

    for product in df[product_column].unique():
        product_df = df[df[product_column] == product]

        # Find features that work uniquely well for this product
        for feature in features:
            # Compare: Feature performance for this product vs others
            product_lift = calculate_product_specific_lift(
                df, product, feature, product_column
            )

            if product_lift > 1.3:  # Lower threshold (products are more similar)
                product_patterns[(product, feature)] = {
                    "lift": product_lift,
                    "confidence": calculate_significance(df, product, feature),
                    "reason": f"For {product}, this feature shows {product_lift}x higher impact"
                }

    return product_patterns
```

---

### Pattern Detection: Branch-Specific Analysis

**Hypothesis:** Regional/cultural differences affect creative performance.

**Example:**
- **US**: Bold, direct, high contrast, people smiling
- **EU**: Minimalist, subtle, clean aesthetics
- **APAC**: Group scenes, harmony, family contexts

**Algorithm:**
```python
def detect_branch_specific_patterns(df, branch_column, features):
    """
    Detect patterns that are specific to branches/regions.
    """
    branch_patterns = {}

    for branch in df[branch_column].unique():
        branch_df = df[df[branch_column] == branch]

        # Find features that work uniquely well for this branch
        for feature in features:
            # Compare: Feature performance for this branch vs others
            branch_lift = calculate_branch_specific_lift(
                df, branch, feature, branch_column
            )

            if branch_lift > 1.2:  # Lowest threshold (regional differences are subtle)
                branch_patterns[(branch, feature)] = {
                    "lift": branch_lift,
                    "confidence": calculate_significance(df, branch, feature),
                    "reason": f"For {branch} branch, this feature shows {branch_lift}x higher impact"
                }

    return branch_patterns
```

---

### Granularity Selection Algorithm

**Decision tree for choosing granularity level:**

```python
def select_granularity_level(
    customer: str,
    product: str | None = None,
    branch: str | None = None,
    campaign_goal: str | None = None,
    min_sample_size_map = {
        1: 200,  # Level 1
        2: 100,  # Level 2
        3: 50,   # Level 3
    }
) -> int:
    """
    Select the most specific granularity level with sufficient data.

    Returns: granularity_level (1-4)
    """
    segment_counts = get_segment_sample_sizes(customer, product, branch, campaign_goal)

    # Try Level 1: Most specific
    if product and branch and campaign_goal:
        count = segment_counts.get((product, branch, campaign_goal), 0)
        if count >= min_sample_size_map[1]:
            return 1

    # Try Level 2: Product + Goal
    if product and campaign_goal:
        count = segment_counts.get((product, campaign_goal), 0)
        if count >= min_sample_size_map[2]:
            return 2

    # Try Level 3: Goal only
    if campaign_goal:
        count = segment_counts.get((campaign_goal,), 0)
        if count >= min_sample_size_map[3]:
            return 3

    # Fallback to Level 4: Generic
    return 4
```

---

## API Changes

### 1. Analysis: New Parameters

**Current:**
```python
def generate_recommendations(
    features_df: pd.DataFrame,
    top_pct: float = 0.25,
    bottom_pct: float = 0.25,
) -> Dict[str, Any]:
    ...
```

**New:**
```python
def generate_recommendations(
    features_df: pd.DataFrame,
    customer: str,
    product: str | None = None,
    branch: str | None = None,
    campaign_goal: str | None = None,
    top_pct: float = 0.25,
    bottom_pct: float = 0.25,
    enable_segmentation: bool = True,
) -> Dict[str, Any]:
    """
    Generate recommendations with optional segmentation.

    Args:
        features_df: DataFrame with features + metadata columns
        customer: Customer name
        product: Optional product filter (enables Level 2/1 analysis)
        branch: Optional branch filter (enables Level 1 analysis)
        campaign_goal: Optional goal filter (enables Level 3/2/1 analysis)
        enable_segmentation: If True, use segmented analysis; if False, use generic
    """
    if not enable_segmentation:
        # Backward compatible: generic analysis
        return generate_generic_recommendations(features_df, top_pct, bottom_pct)

    # Determine granularity level
    granularity = select_granularity_level(
        customer, product, branch, campaign_goal
    )

    # Segment the data
    segmented_df = segment_data(features_df, product, branch, campaign_goal)

    # Generate recommendations for this segment
    recommendations = analyze_segment(segmented_df, granularity)

    # Add metadata
    recommendations["metadata"] = {
        "customer": customer,
        "product": product or "all",
        "branch": branch or "all",
        "campaign_goal": campaign_goal or "all",
        "granularity_level": granularity,
        "sample_size": len(segmented_df),
    }

    return recommendations
```

---

### 2. Query: Retrieve Recommendations with Fallback

**New function:**
```python
def load_recommendations_with_fallback(
    customer: str,
    product: str | None = None,
    branch: str | None = None,
    campaign_goal: str | None = None,
    format: str = "json"
) -> Dict[str, Any]:
    """
    Load recommendations, falling back to less specific levels if needed.

    Search order:
    1. (customer, product, branch, campaign_goal) - Level 1
    2. (customer, product, campaign_goal) - Level 2
    3. (customer, campaign_goal) - Level 3
    4. (customer) - Level 4 (generic fallback)

    Returns:
        Recommendations dict + metadata about which level was used
    """
    # Load segment index
    index = load_segment_index(customer)

    # Try Level 1
    if product and branch and campaign_goal:
        recs = try_load_segment(index, product, branch, campaign_goal)
        if recs:
            recs["metadata"]["fallback_used"] = False
            return recs

    # Try Level 2
    if product and campaign_goal:
        recs = try_load_segment(index, product, None, campaign_goal)
        if recs:
            recs["metadata"]["fallback_used"] = True
            recs["metadata"]["fallback_level"] = 2
            return recs

    # Try Level 3
    if campaign_goal:
        recs = try_load_segment(index, None, None, campaign_goal)
        if recs:
            recs["metadata"]["fallback_used"] = True
            recs["metadata"]["fallback_level"] = 3
            return recs

    # Level 4: Generic fallback (always exists)
    recs = load_generic_recommendations(customer)
    recs["metadata"]["fallback_used"] = True
    recs["metadata"]["fallback_level"] = 4
    return recs
```

---

## Storage Strategy

### File Hierarchy

```
config/ad/miner/
├── moprobo/
│   ├── segment_index.json
│   ├── Power_Station/
│   │   ├── US/
│   │   │   ├── conversion/
│   │   │   │   └── recommendations.json  (Level 1)
│   │   │   ├── awareness/
│   │   │   │   └── recommendations.json  (Level 1)
│   │   │   └── traffic/
│   │   │       └── recommendations.json  (Level 1)
│   │   ├── EU/
│   │   │   └── conversion/
│   │   │       └── recommendations.json  (Level 1)
│   │   └── conversion/
│   │       └── recommendations.json      (Level 2, EU merged)
│   ├── MoProBo/
│   │   ├── US/
│   │   │   └── awareness/
│   │   │       └── recommendations.json  (Level 1)
│   │   └── awareness/
│   │       └── recommendations.json      (Level 2)
│   ├── conversion/
│   │   └── recommendations.json          (Level 3)
│   ├── awareness/
│   │   └── recommendations.json          (Level 3)
│   └── recommendations.json              (Level 4, generic fallback)
└── customer_b/
    └── ...
```

---

### Metadata File Format

**`segment_index.json`** (as shown above)

**Purpose:**
- Fast lookup of available segments
- Avoid filesystem searches
- Enable efficient fallback logic

**Update on every analysis run:**
- Add new segments
- Update sample sizes
- Track last analysis date

---

## Example Workflows

### Workflow 1: Generate Context-Aware Recommendations

**Scenario:** Analyze creatives for Power Station, US branch, conversion goal.

```python
from src.meta.ad.miner.recommendations import generate_recommendations

# Load features with metadata
features_df = load_features_with_metadata("data/creative_features.csv")

# Generate Level 1 recommendations (most specific)
recommendations = generate_recommendations(
    features_df=features_df,
    customer="moprobo",
    product="Power Station",
    branch="US",
    campaign_goal="conversion",
    enable_segmentation=True
)

# Save to appropriate path
save_recommendations(
    recommendations,
    path="config/ad/miner/moprobo/Power_Station/US/conversion/recommendations.json"
)

# Output:
# {
#   "metadata": {
#     "customer": "moprobo",
#     "product": "Power Station",
#     "branch": "US",
#     "campaign_goal": "conversion",
#     "granularity_level": 1,
#     "sample_size": 342,
#     "fallback_used": false
#   },
#   "high_impact_recommendations": [
#     {
#       "feature": "product_position",
#       "value": "bottom-right",
#       "roas_lift_multiple": 2.8,
#       "goal_specific": true,
#       "reason": "For conversion campaigns with Power Station in US, ..."
#     }
#   ]
# }
```

---

### Workflow 2: Query with Automatic Fallback

**Scenario:** Request recommendations for a new product/goal combination with limited data.

```python
from src.meta.ad.miner.queries import load_recommendations_with_fallback

# Request Level 1 (most specific)
recommendations = load_recommendations_with_fallback(
    customer="moprobo",
    product="NewProduct",
    branch="US",
    campaign_goal="conversion"
)

# System tries:
# 1. moprobo/NewProduct/US/conversion → Not found (insufficient data)
# 2. moprobo/NewProduct/conversion → Not found (insufficient data)
# 3. moprobo/conversion → Found (Level 3)
#
# Returns:
# {
#   "metadata": {
#     "customer": "moprobo",
#     "product": "NewProduct",  # Preserved request
#     "branch": "US",
#     "campaign_goal": "conversion",
#     "granularity_level": 3,
#     "sample_size": 156,
#     "fallback_used": true,
#     "fallback_level": 3,
#     "reason": "Insufficient data for Level 1 or 2. Using goal-specific recommendations."
#   },
#   "high_impact_recommendations": [...]
# }
```

---

### Workflow 3: Ad Generator Integration

**Scenario:** Ad generator requests recommendations for a specific campaign.

```python
from src.meta.ad.miner.queries import load_recommendations_with_fallback
from src.meta.ad.generator.pipeline.ad_miner_adapter import convert_recommendations_to_visual_formula

# Load context-aware recommendations
miner_recs = load_recommendations_with_fallback(
    customer="moprobo",
    product="Power Station",
    branch="US",
    campaign_goal="conversion"
)

# Convert to visual formula format
visual_formula = convert_recommendations_to_visual_formula(miner_recs)

# Generate prompts using context-aware recommendations
prompt = prompt_builder.generate_p0_prompt(
    visual_formula=visual_formula,
    product_context={"product_name": "Power Station", "market": "US"}
)

# Result: Prompts are tailored for conversion campaigns with Power Station in US
```

---

## Migration Path

### Phase 1: Add Metadata (Week 1)
1. Update feature extraction to add `campaign_goal`, `product`, `branch` columns
2. Backfill existing data with default values:
   - `campaign_goal`: "unknown"
   - `product`: "unknown"
   - `branch`: "global"
3. Update CSV schema documentation

### Phase 2: Implement Segmentation (Week 2-3)
1. Add segmentation functions to `rule_engine.py`
2. Implement granularity selection algorithm
3. Add goal/product/branch specific pattern detection
4. Unit tests for segmentation logic

### Phase 3: Update Storage (Week 3-4)
1. Create new file hierarchy structure
2. Implement `segment_index.json` generation
3. Update `save_recommendations()` to support hierarchy
4. Migration script to reorganize existing files

### Phase 4: Update Query API (Week 4-5)
1. Implement `load_recommendations_with_fallback()`
2. Update ad generator to use new query function
3. Add CLI support for context parameters:
   ```bash
   python run.py recommend --product "Power Station" --branch "US" --goal "conversion"
   ```

### Phase 5: Backward Compatibility (Week 5)
1. Keep generic `recommendations.md` as fallback
2. Deprecation warnings for old API
3. Documentation updates
4. Training materials

---

## Success Metrics

1. **Coverage:** % of campaigns that can use Level 1-3 recommendations (target: >80%)
2. **Lift improvement:** ROAS lift from context-aware vs generic (target: +20%)
3. **Query latency:** Time to load recommendations with fallback (target: <100ms)
4. **Data quality:** % of segments with sufficient sample size (target: >60% at Level 1-2)

---

## Open Questions

1. **Sample size thresholds:** Are 200/100/50 the right numbers? Need A/B testing.
2. **Feature interactions:** Should we detect 2-way interactions (e.g., "product + goal")?
3. **Temporal drift:** How often to re-analyze segments? (Weekly? Monthly?)
4. **Cold start:** New products with no history → use "similar product" patterns?
5. **Multi-goal campaigns:** Campaigns with multiple goals → merge recommendations?

---

## Appendix: Code Skeleton

### New File: `src/meta/ad/miner/segmentation.py`

```python
"""Context-aware segmentation for recommendation generation."""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SegmentAnalyzer:
    """Analyze creative data by context segments."""

    MIN_SAMPLE_SIZES = {
        1: 200,  # (customer, product, branch, goal)
        2: 100,  # (customer, product, goal)
        3: 50,   # (customer, goal)
        4: 0,    # (customer) - generic fallback
    }

    def __init__(self, features_df: pd.DataFrame):
        """Initialize with features + metadata."""
        self.df = features_df
        self._validate_metadata_columns()

    def _validate_metadata_columns(self):
        """Ensure required metadata columns exist."""
        required = ["campaign_goal", "product", "branch"]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing metadata columns: {missing}")

    def get_available_segments(self) -> Dict[Tuple, int]:
        """Get all segments with their sample sizes."""
        segments = {}

        for (product, branch, goal), group in self.df.groupby(["product", "branch", "campaign_goal"]):
            segments[(product, branch, goal)] = len(group)

        return segments

    def select_granularity_level(
        self,
        product: Optional[str] = None,
        branch: Optional[str] = None,
        campaign_goal: Optional[str] = None
    ) -> int:
        """Select most specific granularity level with sufficient data."""
        # Implementation shown earlier
        pass

    def segment_data(
        self,
        product: Optional[str] = None,
        branch: Optional[str] = None,
        campaign_goal: Optional[str] = None
    ) -> pd.DataFrame:
        """Filter data to specified segment."""
        df = self.df.copy()

        if product:
            df = df[df["product"] == product]
        if branch:
            df = df[df["branch"] == branch]
        if campaign_goal:
            df = df[df["campaign_goal"] == campaign_goal]

        return df

    def detect_goal_specific_patterns(
        self,
        feature: str,
        min_lift: float = 1.5
    ) -> Dict[str, Dict]:
        """Detect goal-specific patterns for a feature."""
        # Implementation shown earlier
        pass

    def detect_product_specific_patterns(
        self,
        feature: str,
        min_lift: float = 1.3
    ) -> Dict[str, Dict]:
        """Detect product-specific patterns for a feature."""
        # Implementation shown earlier
        pass

    def detect_branch_specific_patterns(
        self,
        feature: str,
        min_lift: float = 1.2
    ) -> Dict[str, Dict]:
        """Detect branch-specific patterns for a feature."""
        # Implementation shown earlier
        pass
```

---

### New File: `src/meta/ad/miner/queries.py`

```python
"""Query API for retrieving context-aware recommendations."""

from pathlib import Path
from typing import Any, Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)


def load_segment_index(customer: str) -> Dict:
    """Load segment index for a customer."""
    index_path = Path(f"config/ad/miner/{customer}/segment_index.json")

    if not index_path.exists():
        return {"segments": []}

    with open(index_path) as f:
        return json.load(f)


def try_load_segment(
    index: Dict,
    product: Optional[str],
    branch: Optional[str],
    campaign_goal: Optional[str]
) -> Optional[Dict]:
    """Try to load a specific segment's recommendations."""
    # Search index for matching segment
    for segment in index.get("segments", []):
        if (
            segment.get("product") == product and
            segment.get("branch") == branch and
            segment.get("campaign_goal") == campaign_goal
        ):
            # Load from file
            path = Path(segment["file_path"])
            if path.exists():
                with open(path) as f:
                    return json.load(f)

    return None


def load_recommendations_with_fallback(
    customer: str,
    product: Optional[str] = None,
    branch: Optional[str] = None,
    campaign_goal: Optional[str] = None
) -> Dict[str, Any]:
    """Load recommendations with automatic fallback."""
    # Implementation shown earlier
    pass
```

---

## Summary

This design enables ad miner to generate **context-aware recommendations** while maintaining:

1. **Statistical rigor**: Only use segmented patterns when sample sizes are sufficient
2. **Graceful degradation**: Automatic fallback to broader aggregations
3. **Backward compatibility**: Existing workflows continue to work
4. **Scalability**: Hierarchical storage and efficient querying

**Key innovation:** Multi-level granularity with automatic selection based on data availability.
