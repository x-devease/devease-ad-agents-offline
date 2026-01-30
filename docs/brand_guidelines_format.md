# Brand Guidelines Format Specification

## Overview

This specification defines a structured, machine-readable format for brand guidelines that supports:
- **Automated ad review** - Machine-checkable rules and thresholds
- **Human curation** - Clear, editable format
- **Extensibility** - Easy to add new brands or modify existing ones
- **Multi-level specificity** - Required rules vs. optional recommendations
- **Cultural context** - Market-specific variations and sensitivities

## Format Choice: YAML

**Why YAML?**
- Human-readable and editable
- Supports nested structures
- Type-safe (can be validated against schema)
- Comment support for documentation
- Widely used in config files

**Alternative: JSON** - More strict but less human-friendly
**Alternative: Python dict** - Fastest for code but not externalizable

---

## Complete Schema

```yaml
brand:
  # Basic identification
  id: string                    # Unique identifier (e.g., "moprobo")
  name: string                  # Display name
  website: string               # Official website
  industries: [string]          # e.g., ["consumer_electronics", "outdoor_gear"]

  # Brand tier for validation strictness
  tier: enum                    # enterprise | midmarket | startup
  verification_level: enum      # full | partial | provisional

# ============================================================================
# VISUAL IDENTITY
# ============================================================================

visual_identity:
  # ---------------------------------------------------------------------------
  # COLOR PALETTE
  # ---------------------------------------------------------------------------
  colors:
    # Primary colors (MUST be used)
    primary:
      - name: "Primary Red"
        hex: "#FF0000"
        rgb: [255, 0, 0]
        lab: [54.0, 80.0, 70.0]           # For Delta E calculations
        usage: "Main brand color, CTAs, key highlights"
        restrictions:
          - "Never use in full backgrounds"

      - name: "Black"
        hex: "#000000"
        rgb: [0, 0, 0]
        usage: "Text, borders, contrast"

    # Secondary colors (MAY be used for accents)
    secondary:
      - name: "White"
        hex: "#FFFFFF"
        rgb: [255, 255, 255]
        usage: "Backgrounds, negative space"

      - name: "Dark Gray"
        hex: "#333333"
        rgb: [51, 51, 51]
        usage: "Secondary text, subtle elements"

    # Accent colors (Use sparingly)
    accents:
      - name: "Highlight Blue"
        hex: "#0066CC"
        usage: "Links, hover states, subtle highlights"

    # Forbidden colors (Explicitly NOT allowed)
    forbidden:
      - name: "Neon Green"
        reason: "Conflicts with eco-competitor branding"

    # Color matching tolerance (for automated checking)
    color_matching:
      delta_e_tolerance: 5.0              # CIE76 ΔE threshold
      strict_mode: false                  # If true, tolerance = 2.0
      allow_variations: true              # Allow lighter/darker variants
      variation_tolerance: 10.0           # ΔE for variants

  # ---------------------------------------------------------------------------
  # LOGO SPECIFICATIONS
  # ---------------------------------------------------------------------------
  logo:
    # Logo files
    assets:
      primary: "assets/logos/moprobo_primary.svg"
      secondary: "assets/logos/moprobo_secondary.svg"
      icon_only: "assets/logos/moprobo_icon.svg"
      dark_background: "assets/logos/moprobo_dark.svg"
      light_background: "assets/logos/moprobo_light.svg"

    # Size requirements (percentage of image width)
    size:
      min_percent: 5.0                    # Minimum 5% of image width
      max_percent: 8.0                    # Maximum 8% of image width
      default_percent: 6.5                # Recommended default

      # Special cases
      icon_only:
        min_percent: 3.0
        max_percent: 5.0

    # Placement rules
    placement:
      allowed_positions:
        - "top_right"
        - "top_left"
        - "centered_top"

      default_position: "top_right"

      forbidden_positions:
        - "bottom_right"                 # Too easy to crop
        - "bottom_left"

      # Clear space requirements (2x logo width on all sides)
      clear_space:
        multiplier: 2.0                  # 2x logo width
        min_padding_pixels: 20            # For small images

    # Quality requirements
    quality:
      max_blur_score: 0.05               # Near-zero blur allowed
      min_sharpness: 0.95                # Edge detection threshold
      require_exact_colors: true         # No color shift allowed
      color_delta_e_limit: 1.0           # Stricter than general colors

      # Integrity checks
      must_be_intact: true               # No cropping, cutting, or overlapping
      no_modifications:
        - "stretching"
        - "skewing"
        - "color_overlay"
        - "drop_shadow"
        - "gradient_overlay"

    # Logo usage in text overlays
    in_text:
      allowed: false
      alternative: "Use brand name text instead"

  # ---------------------------------------------------------------------------
  # TYPOGRAPHY
  # ---------------------------------------------------------------------------
  typography:
    primary_font:
      family: "Helvetica Neue"
      fallback: "Arial, sans-serif"
      weights: [700, 900]                # Bold and heavy only
      style: "sans_serif"

    secondary_font:
      family: null                       # None specified
      use_primary: true

    # Text specifications
    text_overlays:
      allowed: true

      # Size requirements
      min_size_percent: 3.0              # Minimum 3% of image height
      readable_at_zoom: "100%"           # Must be readable at 100%

      # Color and contrast
      min_contrast_ratio: 4.5            # WCAG AA standard
      preferred_contrast_ratio: 7.0      # WCAG AAA

      # Weight requirements
      min_font_weight: 400               # Normal weight minimum
      preferred_font_weight: 700         # Bold preferred

      # Text integrity
      must_be_complete: true             # No cut-off text
      must_be_sharp: true                # No blurry text
      max_blur_score: 0.1

      # Forbidden text treatments
      forbidden_effects:
        - "text_outline"                # No stroke/outline
        -text_shadow"                   # No drop shadows
        - "gradient_text"               # No gradients
        - "distortion"                  # No warping/skewing

    # Character limits per ad format
    character_limits:
      headline: 25
      subheadline: 40
      body_text: 100
      total_text: 125

  # ---------------------------------------------------------------------------
  # VISUAL STYLE & COMPOSITION
  # ---------------------------------------------------------------------------
  style:
    # Overall aesthetic
    aesthetic:
      primary: "minimalist"
      secondary: ["tech", "modern", "clean"]
      mood: "confident, professional, straightforward"

    # Composition preferences
    composition:
      balance: "asymmetrical"
      rule_of_thirds: true
      negative_space: "generous"
      focal_point: "product or hero element"

    # Image treatments
    image_treatments:
      allowed:
        - "color_correction"
        - "cropping"
        - "sharpening"

      forbidden:
        - "heavy_filters"               # No Instagram-style filters
        - "vignette"                    # No dark edges
        - "overlay_gradients"           # No color overlays
        - "blur_background"             # Keep backgrounds sharp

    # Product photography
    product_photography:
      background: "clean or contextual"
      lighting: "bright, even"
      angle: "front_angle or slight_3_4_view"
      shadows: "minimal, natural"

    # Human subjects
    people:
      diversity: "required"              # Must show diversity
      demographics:
        - "age_range: 25-45"
        - "ethnicity: diverse"
        - "gender: balanced"

      expressions:
        allowed: ["confident", "friendly", "determined"]
        forbidden: ["overly_dramatic", "seductive", "controversial"]

      clothing:
        style: "casual to professional"
        avoid: ["logos_of_competitors", "offensive_symbols"]

# ============================================================================
# BRAND VOICE & MESSAGING
# ============================================================================

voice:
  # Tone and personality
  tone:
    primary: "professional"
    secondary: ["confident", "straightforward", "technical"]

    forbidden:
      - "humor"                          # Avoid jokes
      - "slang"                          # No casual language
      - "hyperbole"                      # No exaggeration
      - "fear_based"                     # No fear-mongering

  # Messaging principles
  messaging:
    focus: "product_benefits"            # Feature-benefit structure
    proof_points: "required"             # Must back up claims
    specificity: "high"                  # Be specific, not vague

  # Power words (brand-specific terminology)
  power_words:
    use:
      - "reliable"
      - "precision"
      - "performance"
      - "engineered"

    avoid:
      - "cheap"
      - "value"                          # Implies low quality
      - "best_in_class"                 # Requires proof
      - "guarantee"                     # Legal implications

  # Claims and promises
  claims:
    require_evidence: true               # All claims must be provable
    forbidden_claims:
      - "unlimited"
      - "forever"
      - "never"
      - "always"
      - "guarantee"

    # Regulatory compliance
    disclaimers:
      required_when:
        - "making_performance_claims"
        - "showing_test_results"
        - "comparing_to_competitors"

# ============================================================================
# USAGE RULES & RESTRICTIONS
# ============================================================================

usage_rules:
  # Platform-specific rules
  platforms:
    meta:
      text_to_image_ratio: "20% text maximum"
      cta_button_style: "brand_color_with_white_text"
      link_handling: "short_links_only"

    google:
      character_limits:
        headline: 30
        description: 90

    tiktok:
      video_duration: "15-60 seconds"
      trending_audio: "allowed_but_monitor"

  # Context restrictions
  forbidden_contexts:
    - "political_content"
    - "religious_content"
    - "controversial_topics"
    - "competitor_comparison"
    - "negative_advertising"

  # Geographic restrictions
  geographic:
    restricted_regions:
      - region: "CN"
        reason: "separate_brand_guidelines apply"
        alternative_brand: "moprobo_china"

    # Market-specific adaptations
    market_adaptations:
      - market: "DE"
        cultural_notes: "Germans prefer technical specs over emotional appeals"
        tone_adjustment: "more_technical_less_emotional"

      - market: "JP"
        cultural_notes: "Japanese prefer subtlety and minimal text"
        max_text_percent: 10

  # Seasonal considerations
  seasonal:
    holiday_seasons:
      - season: "christmas"
        usage: "secular_only"
        forbidden: ["religious_imagery", "religious_messaging"]

      - season: "ramadan"
        usage: "respectful_acknowledgment_only"
        forbidden: ["festive_imagery", "partying", "alcohol_references"]

# ============================================================================
# COMPLIANCE & LEGAL
# ============================================================================

compliance:
  # Industry regulations
  industry_specific:
    consumer_electronics:
      certifications:
        required_when_visible:
          - "CE"
          - "FCC"
          - "RoHS"
          - "UL"

      safety_claims:
        require_disclaimer: true
        standard_phrases: "Use as directed. Not a toy."

  # Legal review requirements
  legal_review:
    required_for:
      - "health_claims"
      - "safety_promises"
      - "comparative_advertising"
      - "price_promotions"

  # Intellectual property
  intellectual_property:
    trademark_notice: "Use ™ on first brand mention"
    copyright_notice: "© 2024 Moprobo Inc. All rights reserved."
    user_content: "respect_user_privacy"

# ============================================================================
# QUALITY THRESHOLDS (for automated review)
# ============================================================================

quality_thresholds:
  # Minimum scores for auto-approval
  auto_approval:
    min_overall_score: 85.0
    min_brand_score: 90.0
    min_culture_score: 80.0
    min_technical_score: 85.0

  # Critical violations (instant rejection)
  critical_violations:
    - "logo_not_present"
    - "logo_manipulated"
    - "forbidden_color_used"
    - "text_cut_off"
    - "competitor_logo_visible"
    - "explicit_content"
    - "misleading_claim"

  # Warning violations (flag for review)
  warning_violations:
    - "logo_size_out_of_range"
    - "color_tolerance_exceeded"
    - "text_contrast_below_preferred"
    - "font_weight_below_preferred"

  # Scoring weights
  scoring_weights:
    brand_compliance: 0.40
    culture_fit: 0.30
    technical_quality: 0.20
    compliance: 0.10

# ============================================================================
# CULTURAL CONTEXT
# ============================================================================

cultural_context:
  # Brand values
  values:
    - "innovation"
    - "quality"
    - "reliability"
    - "customer_focus"

  # Taboos (never associate brand with)
  taboos:
    - "environmental_harm"
    - "labor_exploitation"
    - "political_extremism"
    - "discrimination"

  # Sensitivities (approach with caution)
  sensitivities:
    - topic: "environmental_impact"
      stance: "acknowledge_transparency"
      guidance: "Be honest about sustainability efforts, avoid greenwashing"

    - topic: "pricing"
      stance: "value_focused"
      guidance: "Emphasize quality and features, avoid discount messaging"

  # Diversity & inclusion
  diversity_inclusion:
    representation:
      race_ethnicity: "reflect_demographics_or_more_diverse"
      gender: "balanced_representation"
      age: "include_adults_25_55"
      ability: "include_people_with_disabilities"

    stereotypes:
      avoid:
        - "gender_roles"
        - "racial_stereotypes"
        - "age_stereotypes"

# ============================================================================
# REFERENCE & ASSETS
# ============================================================================

references:
  # Example creatives (what good looks like)
  approved_examples:
    - path: "examples/approved/campaign_1.jpg"
      notes: "Excellent brand compliance, perfect logo placement"

    - path: "examples/approved/campaign_2.jpg"
      notes: "Great product photography, minimal text"

  # Examples to avoid
  forbidden_examples:
    - path: "examples/forbidden/logo_too_small.jpg"
      violation: "Logo size below 5% threshold"

    - path: "examples/forbidden/competitor_logo.jpg"
      violation: "Competitor logo visible in frame"

  # Brand assets location
  asset_library:
    base_path: "assets/brands/moprobo/"
    logos: "assets/brands/moprobo/logos/"
    fonts: "assets/brands/moprobo/fonts/"
    templates: "assets/brands/moprobo/templates/"
    guidelines_pdf: "assets/brands/moprobo/brand_guidelines.pdf"

# ============================================================================
# METADATA
# ============================================================================

metadata:
  version: "1.0"
  last_updated: "2024-01-15"
  updated_by: "brand_team@company.com"
  review_frequency: "quarterly"
  next_review_date: "2024-04-15"

  # Change tracking
  changes:
    - version: "1.0"
      date: "2024-01-15"
      changes:
        - "Initial brand guidelines definition"
        - "Based on official brand guidelines v3.2"

    - version: "0.9"
      date: "2023-12-01"
      changes:
        - "Draft version extracted from website"
        - "Pending brand team review"

  # Validation status
  validation:
    status: "verified"                   # draft | pending_review | verified | deprecated
    verified_by: "brand_manager"
    verified_date: "2024-01-15"
    confidence_score: 1.0               # 0-1, 1.0 = fully verified

  # Data sources
  sources:
    - type: "official_guidelines"
      url: "https://brand.moprobo.com/guidelines"
      confidence: 1.0

    - type: "website"
      url: "https://moprobo.com"
      confidence: 0.6

    - type: "manual_curation"
      curator: "brand_team"
      confidence: 1.0
```

---

## Example: Moprobo Brand Guidelines

```yaml
# config/ad/reviewer/brand_guidelines/moprobo.yaml

brand:
  id: "moprobo"
  name: "Moprobo"
  website: "https://moprobo.com"
  industries: ["consumer_electronics", "portable_power", "outdoor_gear"]
  tier: "midmarket"
  verification_level: "full"

visual_identity:
  colors:
    primary:
      - name: "Moprobo Red"
        hex: "#FF0000"
        rgb: [255, 0, 0]
        lab: [54.0, 80.0, 70.0]
        usage: "Primary CTAs, brand accents, key highlights"
        restrictions:
          - "Never use as full background"
          - "Use sparingly - max 20% of image area"

      - name: "Tech Black"
        hex: "#1A1A1A"
        rgb: [26, 26, 26]
        usage: "Text, product edges, technical elements"

    secondary:
      - name: "Pure White"
        hex: "#FFFFFF"
        rgb: [255, 255, 255]
        usage: "Backgrounds, negative space"

      - name: "Cool Gray"
        hex: "#E8E8E8"
        rgb: [232, 232, 232]
        usage: "Subtle backgrounds, borders"

    accents:
      - name: "Electric Blue"
        hex: "#0066FF"
        usage: "Power indicators, technology accents"

    color_matching:
      delta_e_tolerance: 5.0
      strict_mode: false
      allow_variations: true

  logo:
    assets:
      primary: "assets/brands/moprobo/logos/moprobo_primary.svg"
      secondary: "assets/brands/moprobo/logos/moprobo_stack.svg"
      icon_only: "assets/brands/moprobo/logos/moprobo_icon.svg"
      light_bg: "assets/brands/moprobo/logos/moprobo_light.svg"
      dark_bg: "assets/brands/moprobo/logos/moprobo_dark.svg"

    size:
      min_percent: 5.0
      max_percent: 8.0
      default_percent: 6.5
      icon_only:
        min_percent: 3.0
        max_percent: 5.0

    placement:
      allowed_positions: ["top_right", "top_left"]
      default_position: "top_right"
      forbidden_positions: ["bottom_right", "bottom_left", "centered"]
      clear_space:
        multiplier: 2.0
        min_padding_pixels: 20

    quality:
      max_blur_score: 0.05
      min_sharpness: 0.95
      require_exact_colors: true
      color_delta_e_limit: 1.0
      must_be_intact: true
      no_modifications:
        - "stretching"
        - "skewing"
        - "color_overlay"
        - "drop_shadow"

  typography:
    primary_font:
      family: "Inter"
      fallback: "Helvetica Neue, Arial, sans-serif"
      weights: [600, 700, 900]
      style: "sans_serif"

    text_overlays:
      allowed: true
      min_size_percent: 3.5
      readable_at_zoom: "100%"
      min_contrast_ratio: 4.5
      preferred_contrast_ratio: 7.0
      min_font_weight: 600
      preferred_font_weight: 700
      must_be_complete: true
      must_be_sharp: true
      max_blur_score: 0.1
      forbidden_effects:
        - "text_outline"
        - "text_shadow"
        - "gradient_text"
        - "distortion"

    character_limits:
      headline: 25
      subheadline: 30
      body_text: 80
      total_text: 100

  style:
    aesthetic:
      primary: "minimalist"
      secondary: ["technical", "modern", "clean"]
      mood: "confident, precise, engineered"

    composition:
      balance: "asymmetrical_dynamic"
      rule_of_thirds: true
      negative_space: "generous"
      focal_point: "product_hero"

    image_treatments:
      allowed:
        - "color_correction"
        - "cropping"
        - "minor_retouching"

      forbidden:
        - "heavy_filters"
        - "vignette"
        - "overlay_gradients"
        - "blur_background"

    product_photography:
      background: "clean_white_or_contextual_outdoor"
      lighting: "bright_even"
      angle: "front_angle_or_slight_3_4"
      shadows: "minimal_natural"

    people:
      diversity: "required"
      demographics:
        - "age_range: 28-45"
        - "ethnicity: diverse"
        - "gender: balanced"
      expressions:
        allowed: ["confident", "adventurous", "prepared"]
        forbidden: ["overly_dramatic", "seductive", "fearful"]

voice:
  tone:
    primary: "professional"
    secondary: ["confident", "technical", "straightforward"]
    forbidden:
      - "humor"
      - "slang"
      - "hyperbole"
      - "emotional_manipulation"

  messaging:
    focus: "product_benefits_and_specifications"
    proof_points: "required"
    specificity: "high"

  power_words:
    use:
      - "precision"
      - "power"
      - "reliable"
      - "engineered"
      - "adventure"

    avoid:
      - "cheap"
      - "value"
      - "unbelievable"
      - "best"

  claims:
    require_evidence: true
    forbidden_claims:
      - "unlimited"
      - "forever"
      - "never_runs_out"
      - "guarantee"

usage_rules:
  platforms:
    meta:
      text_to_image_ratio: "20% text maximum"
      cta_button_style: "red_with_white_text"

    google:
      character_limits:
        headline: 30
        description: 90

  forbidden_contexts:
    - "political_content"
    - "religious_content"
    - "competitor_comparison"
    - "environmental_claims_without_proof"

  geographic:
    market_adaptations:
      - market: "DE"
        cultural_notes: "Germans prefer technical specifications"
        tone_adjustment: "more_technical"

      - market: "JP"
        cultural_notes: "Minimalist aesthetics preferred"
        max_text_percent: 10

compliance:
  industry_specific:
    consumer_electronics:
      certifications:
        required_when_visible: ["CE", "FCC", "RoHS"]
      safety_claims:
        require_disclaimer: true

quality_thresholds:
  auto_approval:
    min_overall_score: 85.0
    min_brand_score: 90.0
    min_culture_score: 80.0

  critical_violations:
    - "logo_not_present"
    - "logo_manipulated"
    - "forbidden_color_used"
    - "competitor_brand_visible"

  warning_violations:
    - "logo_size_out_of_range"
    - "color_tolerance_exceeded"

  scoring_weights:
    brand_compliance: 0.40
    culture_fit: 0.30
    technical_quality: 0.20
    compliance: 0.10

cultural_context:
  values:
    - "innovation"
    - "quality"
    - "adventure"
    - "reliability"

  taboos:
    - "environmental_harm"
    - "cheap_quality"
    - "unreliability"

  diversity_inclusion:
    representation:
      race_ethnicity: "diverse_or_inclusive"
      gender: "balanced"
      age: "28_50_primary"
    stereotypes:
      avoid:
        - "gender_stereotypes"
        - "racial_stereotypes"

references:
  approved_examples:
    - path: "examples/moprobo/approved/campaign_001.jpg"
      notes: "Perfect logo placement, clean composition"

  forbidden_examples:
    - path: "examples/moprobo/forbidden/logo_small.jpg"
      violation: "Logo only 3% of image width"

  asset_library:
    base_path: "assets/brands/moprobo/"
    logos: "assets/brands/moprobo/logos/"
    fonts: "assets/brands/moprobo/fonts/"

metadata:
  version: "1.0"
  last_updated: "2024-01-15"
  updated_by: "brand_team@moprobo.com"
  review_frequency: "quarterly"
  next_review_date: "2024-04-15"

  changes:
    - version: "1.0"
      date: "2024-01-15"
      changes:
        - "Initial comprehensive guidelines"

  validation:
    status: "verified"
    verified_by: "sarah_brand_manager"
    verified_date: "2024-01-15"
    confidence_score: 1.0

  sources:
    - type: "official_guidelines"
      url: "https://brand.moprobo.com/guidelines-v3"
      confidence: 1.0
    - type: "manual_curation"
      curator: "brand_team"
      confidence: 1.0
```

---

## Example: EcoFlow Brand Guidelines (Contrast)

```yaml
# config/ad/reviewer/brand_guidelines/ecoflow.yaml

brand:
  id: "ecoflow"
  name: "EcoFlow"
  website: "https://ecoflow.com"
  industries: ["consumer_electronics", "green_energy", "outdoor_power"]
  tier: "midmarket"
  verification_level: "full"

visual_identity:
  colors:
    primary:
      - name: "Eco Green"
        hex: "#00A651"
        rgb: [0, 166, 81]
        lab: [60.0, -50.0, 30.0]
        usage: "Primary brand color, CTAs, eco-accents"

      - name: "Leaf Dark Green"
        hex: "#2D5A27"
        rgb: [45, 90, 39]
        usage: "Text, serious applications"

    secondary:
      - name: "Sky Blue"
        hex: "#87CEEB"
        rgb: [135, 206, 235]
        usage: "Energy indicators, clean power messaging"

      - name: "Warm White"
        hex: "#FFF8E7"
        rgb: [255, 248, 231]
        usage: "Backgrounds, warm natural feel"

    # EcoFlow allows MORE color flexibility than Moprobo
    color_matching:
      delta_e_tolerance: 10.0            # More lenient
      allow_variations: true
      variation_tolerance: 15.0

  logo:
    # Similar structure but different values
    size:
      min_percent: 4.0                   # Slightly smaller allowed
      max_percent: 10.0                  # More flexibility
      default_percent: 7.0

    placement:
      allowed_positions: ["top_right", "top_left", "centered_top", "bottom_right"]
      default_position: "top_right"
      # Note: bottom_right ALLOWED (unlike Moprobo)

  style:
    aesthetic:
      primary: "natural"
      secondary: ["outdoorsy", "adventure", "sustainable"]
      mood: "empowering, free, eco_conscious"

    # EcoFlow is MORE relaxed about image treatments
    image_treatments:
      allowed:
        - "color_correction"
        - "cropping"
        - "warm_filters"                 # Unlike Moprobo
        - "subtle_vignette"              # Allowed for drama

      forbidden:
        - "neon_filters"
        - "heavy_overlays"

voice:
  tone:
    primary: "empowering"
    secondary: ["adventurous", "optimistic", "sustainability_focused"]
    # EcoFlow allows some emotion, unlike Moprobo
    forbidden:
      - "fear_based"
      - "guilt_trip"                     # No climate guilt
      - "political"

  power_words:
    use:
      - "freedom"
      - "sustainable"
      - "adventure"
      - "clean_energy"
      - "off_grid"

    avoid:
      - "cheap"
      - "emergency"                      # Too negative

usage_rules:
  forbidden_contexts:
    - "political_content"
    # Note: EcoFlow allows environmental claims (their brand purpose)
    # unlike Moprobo which requires proof

cultural_context:
  values:
    - "sustainability"
    - "freedom"
    - "adventure"
    - "environmental_stewardship"

  # EcoFlow has DIFFERENT taboos than Moprobo
  taboos:
    - "environmental_harm"               # Critical for them
    - "wastefulness"
    - "fossil_fuel_dependence"

metadata:
  version: "1.0"
  last_updated: "2024-01-15"
  validation:
    status: "verified"
    confidence_score: 1.0
```

---

## Implementation: Python Data Classes

```python
# src/meta/ad/reviewer/models/brand_guidelines.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal
from pathlib import Path

@dataclass
class ColorSpec:
    """Color specification."""
    name: str
    hex: str
    rgb: Tuple[int, int, int]
    lab: Optional[Tuple[float, float, float]] = None
    usage: Optional[str] = None
    restrictions: List[str] = field(default_factory=list)

@dataclass
class ColorMatchingRules:
    """Color matching tolerance rules."""
    delta_e_tolerance: float = 5.0
    strict_mode: bool = False
    allow_variations: bool = True
    variation_tolerance: float = 10.0

@dataclass
class LogoSizeRules:
    """Logo size specifications."""
    min_percent: float
    max_percent: float
    default_percent: float

@dataclass
class LogoPlacementRules:
    """Logo placement specifications."""
    allowed_positions: List[str]
    default_position: str
    forbidden_positions: List[str] = field(default_factory=list)
    clear_space_multiplier: float = 2.0
    min_padding_pixels: int = 20

@dataclass
class LogoQualityRules:
    """Logo quality requirements."""
    max_blur_score: float = 0.05
    min_sharpness: float = 0.95
    require_exact_colors: bool = True
    color_delta_e_limit: float = 1.0
    must_be_intact: bool = True
    no_modifications: List[str] = field(default_factory=list)

@dataclass
class LogoSpec:
    """Complete logo specifications."""
    assets: Dict[str, str]
    size: LogoSizeRules
    placement: LogoPlacementRules
    quality: LogoQualityRules

@dataclass
class TypographyRules:
    """Typography specifications."""
    primary_font: Dict[str, any]
    text_overlays: Dict[str, any]
    character_limits: Dict[str, int]

@dataclass
class StyleRules:
    """Visual style and composition rules."""
    aesthetic: Dict[str, any]
    composition: Dict[str, any]
    image_treatments: Dict[str, List[str]]
    product_photography: Optional[Dict[str, any]] = None
    people: Optional[Dict[str, any]] = None

@dataclass
class VoiceRules:
    """Brand voice and messaging rules."""
    tone: Dict[str, any]
    messaging: Dict[str, any]
    power_words: Dict[str, List[str]]
    claims: Dict[str, any]

@dataclass
class UsageRules:
    """Usage rules and restrictions."""
    platforms: Dict[str, any]
    forbidden_contexts: List[str]
    geographic: Optional[Dict[str, any]] = None

@dataclass
class ComplianceRules:
    """Compliance and legal requirements."""
    industry_specific: Optional[Dict[str, any]] = None
    legal_review: Optional[Dict[str, any]] = None

@dataclass
class QualityThresholds:
    """Quality thresholds for automated review."""
    auto_approval: Dict[str, float]
    critical_violations: List[str]
    warning_violations: List[str]
    scoring_weights: Dict[str, float]

@dataclass
class CulturalContext:
    """Cultural context and sensitivities."""
    values: List[str]
    taboos: List[str]
    sensitivities: Optional[List[Dict]] = None
    diversity_inclusion: Optional[Dict[str, any]] = None

@dataclass
class References:
    """Reference examples and assets."""
    approved_examples: List[Dict[str, str]] = field(default_factory=list)
    forbidden_examples: List[Dict[str, str]] = field(default_factory=list)
    asset_library: Optional[Dict[str, str]] = None

@dataclass
class Metadata:
    """Metadata and validation info."""
    version: str
    last_updated: str
    validation: Dict[str, any]
    sources: List[Dict[str, any]]

@dataclass
class BrandGuidelines:
    """Complete brand guidelines specification."""
    # Basic info
    id: str
    name: str
    website: str
    industries: List[str]
    tier: Literal["enterprise", "midmarket", "startup"]
    verification_level: Literal["full", "partial", "provisional"]

    # Visual identity
    visual_identity: Dict[str, any]

    # Voice
    voice: VoiceRules

    # Usage rules
    usage_rules: UsageRules

    # Compliance
    compliance: ComplianceRules

    # Quality thresholds
    quality_thresholds: QualityThresholds

    # Cultural context
    cultural_context: CulturalContext

    # References
    references: Optional[References] = None

    # Metadata
    metadata: Optional[Metadata] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BrandGuidelines':
        """Load brand guidelines from YAML file."""
        import yaml

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse nested objects
        visual_identity = data['visual_identity']

        return cls(
            id=data['brand']['id'],
            name=data['brand']['name'],
            website=data['brand']['website'],
            industries=data['brand']['industries'],
            tier=data['brand']['tier'],
            verification_level=data['brand']['verification_level'],
            visual_identity=visual_identity,
            voice=VoiceRules(**data['voice']),
            usage_rules=UsageRules(**data['usage_rules']),
            compliance=ComplianceRules(**data.get('compliance', {})),
            quality_thresholds=QualityThresholds(**data['quality_thresholds']),
            cultural_context=CulturalContext(**data['cultural_context']),
            references=References(**data.get('references', {})),
            metadata=Metadata(**data['metadata']) if 'metadata' in data else None
        )

    def get_color_palette(self) -> Dict[str, List[ColorSpec]]:
        """Extract color palette as ColorSpec objects."""
        colors = {}
        for category in ['primary', 'secondary', 'accents']:
            if category in self.visual_identity['colors']:
                colors[category] = [
                    ColorSpec(**c) for c in self.visual_identity['colors'][category]
                ]
        return colors

    def get_logo_spec(self) -> LogoSpec:
        """Get logo specifications."""
        logo_data = self.visual_identity['logo']
        return LogoSpec(
            assets=logo_data['assets'],
            size=LogoSizeRules(**logo_data['size']),
            placement=LogoPlacementRules(**logo_data['placement']),
            quality=LogoQualityRules(**logo_data['quality'])
        )

    def is_critical_violation(self, violation_type: str) -> bool:
        """Check if violation type is critical."""
        return violation_type in self.quality_thresholds.critical_violations
```

---

## Validation Schema (JSON Schema)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Brand Guidelines Specification",
  "type": "object",
  "required": ["brand", "visual_identity", "voice", "quality_thresholds"],
  "properties": {
    "brand": {
      "type": "object",
      "required": ["id", "name", "website", "tier"],
      "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "website": {"type": "string", "format": "uri"},
        "industries": {"type": "array", "items": {"type": "string"}},
        "tier": {"enum": ["enterprise", "midmarket", "startup"]}
      }
    },
    "visual_identity": {
      "type": "object",
      "required": ["colors", "logo"],
      "properties": {
        "colors": {
          "type": "object",
          "required": ["primary"],
          "properties": {
            "primary": {
              "type": "array",
              "items": {
                "type": "object",
                "required": ["name", "hex", "rgb"],
                "properties": {
                  "name": {"type": "string"},
                  "hex": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
                  "rgb": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 255}}
                }
              }
            }
          }
        }
      }
    }
  }
}
```

---

## Migration from Current Format

```python
# Migration script from current brand_guidelines.py to new YAML format

def migrate_brand_guidelines_to_yaml():
    """Migrate existing brand_guidelines.py to new YAML format."""

    # Current format in generator/orchestrator/brand_identity.py
    BRAND_GUIDELINES = {
        "moprobo": {
            "name": "Moprobo",
            "primary_colors": ["#FF0000", "#000000"],
            # ... existing structure
        }
    }

    # Convert to new format
    for brand_id, guidelines in BRAND_GUIDELINES.items():
        new_guidelines = convert_to_new_format(guidelines)
        yaml_path = f"config/ad/reviewer/brand_guidelines/{brand_id}.yaml"

        with open(yaml_path, 'w') as f:
            yaml.dump(new_guidelines, f, default_flow_style=False)

        print(f"Migrated {brand_id} to {yaml_path}")
```

---

## Summary

This brand guidelines format provides:

1. **Comprehensive Coverage** - All aspects needed for automated review
2. **Machine-Checkable** - Specific thresholds, tolerances, and rules
3. **Human-Readable** - Clear YAML structure with comments
4. **Validatable** - Can use JSON Schema or Pydantic for validation
5. **Extensible** - Easy to add new brands or modify existing ones
6. **Traceable** - Metadata tracks sources, confidence, and changes
7. **Context-Rich** - Cultural considerations and market adaptations

**Key advantages over current `brand_guidelines.py`:**
- External configuration (no code changes needed)
- Richer metadata and validation
- Better organization of complex rules
- Support for market-specific adaptations
- Clear provenance and confidence tracking
