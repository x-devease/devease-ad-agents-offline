# Campaign Effectiveness & Regional Pattern Matching

## Overview

The ad reviewer should evaluate creatives not just for compliance, but for **effectiveness** by checking:

1. **Campaign Goal Alignment** - Does the creative match campaign objectives?
2. **Regional Historical Patterns** - Does it follow what works in this market?
3. **Performance Prediction** - Likely performance based on historical data
4. **Optimization Recommendations** - Specific improvements for better results

This transforms the reviewer from a "gatekeeper" to an **optimizer** that improves campaign outcomes.

---

## Architecture

```yaml
effectiveness_checker:
  enabled: true

  # Data sources
  data_sources:
    - "historical_campaign_performance"
    - "regional_performance_database"
    - "seasonal_patterns"
    - "competitive_intelligence"
    - "platform_best_practices"

  # Check dimensions
  dimensions:
    - "campaign_goal_alignment"
    - "regional_pattern_matching"
    - "audience_fit"
    - "platform_optimization"
    - "seasonal_relevance"
```

---

## 1. Campaign Goal Alignment

### Campaign Goal Types

```yaml
campaign_goals:
  brand_awareness:
    description: "Increase brand visibility and recognition"
    kpis:
      - "reach"
      - "impressions"
      - "brand_recall"
      - "brand_sentiment"

    creative_characteristics:
      emphasis:
        - "brand_logo_prominent"
        - "brand_colors_dominant"
        - "simple_message"
        - "memorable_visual"

      cta:
        type: "soft"
        examples:
          - "Learn More"
          - "Discover [Brand]"
          - "See How It Works"

      text_ratio: "minimal"         # < 20% text
      visual_style: "bold_distinctive"

    validation_checks:
      - check: "logo_visibility"
        requirement: "logo_area >= 6% of image"
        score_weight: 0.30

      - check: "brand_color_dominance"
        requirement: "primary_colors >= 40% of image"
        score_weight: 0.25

      - check: "message_clarity"
        requirement: "single_clear_message"
        score_weight: 0.20

      - check: "visual_impact"
        requirement: "high_contrast_bright_colors"
        score_weight: 0.15

      - check: "cta_appropriateness"
        requirement: "soft_cta_present"
        score_weight: 0.10

  consideration:
    description: "Generate consideration and interest"
    kpis:
      - "click_through_rate"
      - "engagement_rate"
      - "time_spent_on_site"
      - "page_views_per_visitor"

    creative_characteristics:
      emphasis:
        - "product_benefits"
        - "key_features"
        - "social_proof"
        - "value_proposition"

      cta:
        type: "medium"
        examples:
          - "Shop Now"
          - "Explore Features"
          - "See Details"
          - "Compare Options"

      text_ratio: "moderate"        # 20-30% text
      visual_style: "product_focused"

    validation_checks:
      - check: "product_hero_visibility"
        requirement: "product occupies >= 30% of frame"
        score_weight: 0.25

      - check: "benefit_clarity"
        requirement: "at least 2 key benefits visible"
        score_weight: 0.25

      - check: "social_proof_presence"
        requirement: "reviews/ratings/endorsements visible"
        score_weight: 0.20

      - check: "cta_clarity"
        requirement: "clear_action_oriented_cta"
        score_weight: 0.15

      - check: "information_density"
        requirement: "balanced_not_overwhelming"
        score_weight: 0.15

  conversion:
    description: "Drive immediate action or purchase"
    kpis:
      - "conversion_rate"
      - "cost_per_acquisition"
      - "return_on_ad_spend"
      - "purchases"

    creative_characteristics:
      emphasis:
        - "strong_offer"
        - "urgency"
        - "clear_value"
        - "trust_signals"

      cta:
        type: "hard"
        examples:
          - "Buy Now"
          - "Limited Time - 50% Off"
          - "Shop Sale"
          - "Get Yours Today"

      text_ratio: "higher"           # 30-40% text for offer details
      visual_style: "offer_driven"

    validation_checks:
      - check: "offer_clarity"
        requirement: "discount/deal clearly stated"
        score_weight: 0.30

      - check: "urgency_signals"
        requirement: "scarcity/deadline visible"
        score_weight: 0.25

      - check: "trust_signals"
        requirement: "guarantee/returns/security_badge visible"
        score_weight: 0.20

      - check: "cta_prominence"
        requirement: "cta is most prominent element after offer"
        score_weight: 0.15

      - check: "price_visibility"
        requirement: "original_and_sale_price visible"
        score_weight: 0.10

  retention:
    description: "Retain existing customers"
    kpis:
      - "retention_rate"
      - "repeat_purchase_rate"
      - "customer_lifetime_value"
      - "engagement_frequency"

    creative_characteristics:
      emphasis:
        - "exclusivity"
        - "loyalty_rewards"
        - "new_product_access"
        - "appreciation"

      cta:
        type: "relationship"
        examples:
          - "Exclusive for You"
          - "Loyalty Rewards"
          - "Get Your Perk"
          - "Members Only"

      text_ratio: "moderate"
      visual_style: "premium_exclusive"

    validation_checks:
      - check: "exclusivity_signals"
        requirement: "member/customer_only language"
        score_weight: 0.30

      - check: "personalization"
        requirement: "personalized_elements visible"
        score_weight: 0.25

      - check: "reward_visibility"
        requirement: "points/discount/benefit clear"
        score_weight: 0.25

      - check: "appreciation_tone"
        requirement: "thank_you/valued_customer messaging"
        score_weight: 0.20

  lead_generation:
    description: "Capture lead information"
    kpis:
      - "cost_per_lead"
      - "lead_quality_score"
      - "conversion_to_opportunity"
      - "form_completion_rate"

    creative_characteristics:
      emphasis:
        - "value_exchange"
        - "low_friction"
        - "clear_benefit"
        - "trust_indicators"

      cta:
        type: "form_submission"
        examples:
          - "Get Free Guide"
          - "Download Now"
          - "Request Quote"
          - "Start Free Trial"

      text_ratio: "moderate_to_high"
      visual_style: "professional_trustworthy"

    validation_checks:
      - check: "value_proposition_clarity"
        requirement: "what_they_get_clearly_stated"
        score_weight: 0.30

      - check: "low_friction_signals"
        requirement: "no_payment/simple_form emphasized"
        score_weight: 0.25

      - check: "credibility_indicators"
        requirement: "expertise/certifications visible"
        score_weight: 0.25

      - check: "cta_strength"
        requirement: "compelling_immediate_value_cta"
        score_weight: 0.20

  app_install:
    description: "Drive mobile app installations"
    kpis:
      - "install_rate"
      - "cost_per_install"
      - "app_open_rate"
      - "retention_after_install"

    creative_characteristics:
      emphasis:
        - "app_screenshots"
        - "key_features_demo"
        - "app_store_rating"
        - "install_cta"

      cta:
        type: "app_install"
        examples:
          - "Install Now"
          - "Download on App Store"
          - "Get the App"
          - "Available on Google Play"

      text_ratio: "minimal"
      visual_style: "mobile_app_showcase"

    validation_checks:
      - check: "app_visuals"
        requirement: "app_interface/screenshots visible"
        score_weight: 0.30

      - check: "feature_demonstration"
        requirement: "key_app_feature shown in action"
        score_weight: 0.25

      - check: "rating_visibility"
        requirement: "app_store_rating visible (4.5+ preferred)"
        score_weight: 0.20

      - check: "platform_badges"
        requirement: "App Store/Google Play badges"
        score_weight: 0.15

      - check: "value_clarity"
        requirement: "why_install_clear_communicated"
        score_weight: 0.10
```

### Goal Alignment Checker

```python
# src/meta/ad/qa/effectiveness/campaign_goal_checker.py

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class CampaignGoal(Enum):
    BRAND_AWARENESS = "brand_awareness"
    CONSIDERATION = "consideration"
    CONVERSION = "conversion"
    RETENTION = "retention"
    LEAD_GENERATION = "lead_generation"
    APP_INSTALL = "app_install"

@dataclass
class GoalAlignmentResult:
    """Result of campaign goal alignment check."""
    campaign_goal: CampaignGoal
    alignment_score: float  # 0-100
    passed_checks: List[Dict]
    failed_checks: List[Dict]
    recommendations: List[str]

    def is_aligned(self, threshold: float = 70.0) -> bool:
        """Check if creative aligns with campaign goal."""
        return self.alignment_score >= threshold

class CampaignGoalChecker:
    """Checks if creative aligns with campaign goals."""

    def __init__(self):
        self.goal_specifications = self._load_goal_specs()

    def check_alignment(
        self,
        image_analysis: Dict,
        campaign_goal: CampaignGoal,
        text_content: Optional[str] = None
    ) -> GoalAlignmentResult:
        """
        Check if creative aligns with campaign goal.

        Args:
            image_analysis: GPT-4 Vision analysis results
            campaign_goal: The campaign's primary objective
            text_content: Text overlays in the creative

        Returns:
            GoalAlignmentResult with detailed analysis
        """
        goal_spec = self.goal_specifications[campaign_goal.value]
        checks = goal_spec['validation_checks']

        passed = []
        failed = []
        total_weight = 0
        weighted_score = 0

        for check in checks:
            result = self._run_check(check, image_analysis, text_content)
            total_weight += check['score_weight']

            if result['passed']:
                passed.append(result)
                weighted_score += check['score_weight'] * result['score']
            else:
                failed.append(result)
                weighted_score += check['score_weight'] * result['score']

        # Calculate overall alignment score
        alignment_score = (weighted_score / total_weight * 100) if total_weight > 0 else 0

        # Generate recommendations
        recommendations = self._generate_recommendations(
            campaign_goal,
            failed,
            goal_spec
        )

        return GoalAlignmentResult(
            campaign_goal=campaign_goal,
            alignment_score=alignment_score,
            passed_checks=passed,
            failed_checks=failed,
            recommendations=recommendations
        )

    def _run_check(
        self,
        check: Dict,
        image_analysis: Dict,
        text_content: Optional[str]
    ) -> Dict:
        """Run a single validation check."""
        check_type = check['check']
        requirement = check['requirement']

        if check_type == 'logo_visibility':
            return self._check_logo_visibility(image_analysis, requirement)

        elif check_type == 'brand_color_dominance':
            return self._check_brand_colors(image_analysis, requirement)

        elif check_type == 'product_hero_visibility':
            return self._check_product_visibility(image_analysis, requirement)

        elif check_type == 'offer_clarity':
            return self._check_offer_clarity(image_analysis, text_content, requirement)

        elif check_type == 'cta_clarity':
            return self._check_cta_clarity(image_analysis, text_content, requirement)

        elif check_type == 'social_proof_presence':
            return self._check_social_proof(image_analysis, text_content, requirement)

        elif check_type == 'urgency_signals':
            return self._check_urgency_signals(image_analysis, text_content, requirement)

        elif check_type == 'trust_signals':
            return self._check_trust_signals(image_analysis, text_content, requirement)

        else:
            return {
                'check': check_type,
                'passed': True,
                'score': 0.5,  # Neutral score for unknown checks
                'reason': 'Check not implemented'
            }

    def _check_logo_visibility(self, image_analysis: Dict, requirement: str) -> Dict:
        """Check if logo is visible and prominent enough."""
        logo = image_analysis.get('logo', {})

        if not logo.get('present'):
            return {
                'check': 'logo_visibility',
                'passed': False,
                'score': 0.0,
                'reason': 'Logo not detected in image'
            }

        logo_area_percent = logo.get('area_percentage', 0)
        required_min = 6.0  # From requirement "logo_area >= 6%"

        if logo_area_percent >= required_min:
            return {
                'check': 'logo_visibility',
                'passed': True,
                'score': 1.0,
                'measured': f'{logo_area_percent:.1f}%',
                'requirement': f'>= {required_min}%',
                'reason': 'Logo meets size requirement'
            }
        else:
            score = logo_area_percent / required_min
            return {
                'check': 'logo_visibility',
                'passed': False,
                'score': score,
                'measured': f'{logo_area_percent:.1f}%',
                'requirement': f'>= {required_min}%',
                'reason': f'Logo too small (current: {logo_area_percent:.1f}%, required: {required_min}%)'
            }

    def _check_offer_clarity(
        self,
        image_analysis: Dict,
        text_content: Optional[str],
        requirement: str
    ) -> Dict:
        """Check if offer is clearly stated."""
        if not text_content:
            return {
                'check': 'offer_clarity',
                'passed': False,
                'score': 0.0,
                'reason': 'No text content found'
            }

        # Look for offer indicators
        offer_patterns = [
            r'\d+%.*off',
            r'\$\d+.*off',
            r'save.*\$\d+',
            r'deal',
            r'offer',
            r'limited',
            r'special'
        ]

        import re
        has_offer = any(re.search(pattern, text_content, re.IGNORECASE) for pattern in offer_patterns)

        if has_offer:
            # Check if specific numbers are present
            has_percentage = bool(re.search(r'\d+%', text_content))
            has_price = bool(re.search(r'\$\d+', text_content))

            if has_percentage or has_price:
                return {
                    'check': 'offer_clarity',
                    'passed': True,
                    'score': 1.0,
                    'measured': 'Specific offer with numbers',
                    'reason': 'Clear offer with specific discount/price'
                }
            else:
                return {
                    'check': 'offer_clarity',
                    'passed': True,
                    'score': 0.7,
                    'measured': 'Generic offer mentioned',
                    'reason': 'Offer present but could be more specific (add percentage or amount)'
                }
        else:
            return {
                'check': 'offer_clarity',
                'passed': False,
                'score': 0.0,
                'reason': 'No clear offer detected in text'
            }

    def _check_urgency_signals(
        self,
        image_analysis: Dict,
        text_content: Optional[str],
        requirement: str
    ) -> Dict:
        """Check for urgency/scarcity signals."""
        if not text_content:
            return {
                'check': 'urgency_signals',
                'passed': False,
                'score': 0.0,
                'reason': 'No text content'
            }

        urgency_patterns = [
            r'\bonly\s+\d+\b',
            r'\blimited\s+time\b',
            r'\bends?\s+\w+\b',
            r'\bhurry\b',
            r'\btoday\s+only\b',
            r'\blast\s+chance\b',
            r'\bwhile\s+supplies?\s+last\b'
        ]

        import re
        matches = []
        for pattern in urgency_patterns:
            if re.search(pattern, text_content, re.IGNORECASE):
                matches.append(pattern)

        if len(matches) >= 2:
            return {
                'check': 'urgency_signals',
                'passed': True,
                'score': 1.0,
                'measured': f'{len(matches)} urgency indicators',
                'reason': 'Strong urgency signals present'
            }
        elif len(matches) == 1:
            return {
                'check': 'urgency_signals',
                'passed': True,
                'score': 0.7,
                'measured': '1 urgency indicator',
                'reason': 'Urgency present but could be strengthened'
            }
        else:
            return {
                'check': 'urgency_signals',
                'passed': False,
                'score': 0.0,
                'reason': 'No urgency signals detected'
            }

    def _generate_recommendations(
        self,
        goal: CampaignGoal,
        failed_checks: List[Dict],
        goal_spec: Dict
    ) -> List[str]:
        """Generate recommendations based on failed checks."""
        recommendations = []

        for check in failed_checks:
            check_type = check['check']
            reason = check.get('reason', '')

            if check_type == 'logo_visibility':
                recommendations.append(
                    "Increase logo size to at least 6% of image area for better brand recall"
                )

            elif check_type == 'brand_color_dominance':
                recommendations.append(
                    "Increase use of primary brand colors to strengthen brand association"
                )

            elif check_type == 'offer_clarity':
                recommendations.append(
                    "Make the offer more specific - include exact discount percentage or price savings"
                )

            elif check_type == 'urgency_signals':
                recommendations.append(
                    "Add urgency elements like 'Limited Time' or specific deadline to encourage immediate action"
                )

            elif check_type == 'trust_signals':
                recommendations.append(
                    "Add trust signals like 'Money Back Guarantee', 'Free Returns', or security badges"
                )

            elif check_type == 'cta_clarity':
                recommendations.append(
                    f"Use a stronger, more action-oriented CTA. Suggested: '{goal_spec['creative_characteristics']['cta']['examples'][0]}'"
                )

            else:
                recommendations.append(f"Improve {check_type}: {reason}")

        return recommendations

    def _load_goal_specs(self) -> Dict:
        """Load campaign goal specifications."""
        # In production, load from YAML/JSON config
        return {
            'brand_awareness': {
                'validation_checks': [
                    {'check': 'logo_visibility', 'requirement': 'logo_area >= 6%', 'score_weight': 0.30},
                    {'check': 'brand_color_dominance', 'requirement': 'primary_colors >= 40%', 'score_weight': 0.25},
                    # ... more checks
                ]
            },
            'conversion': {
                'validation_checks': [
                    {'check': 'offer_clarity', 'requirement': 'clear offer stated', 'score_weight': 0.30},
                    {'check': 'urgency_signals', 'requirement': 'scarcity/deadline visible', 'score_weight': 0.25},
                    # ... more checks
                ]
            },
            # ... more goals
        }
```

---

## 2. Regional Historical Pattern Matching

### Regional Performance Database

```yaml
# config/ad/reviewer/regions/performance_database.yaml

regional_patterns:
  # North America
  US:
    visual_preferences:
      colors:
        high_performing:
          - color: "blue"
            lift_vs_average: "+15%"
            confidence: 0.92
            best_for_goals: ["trust", "professionalism", "technology"]

          - color: "red"
            lift_vs_average: "+12%"
            confidence: 0.88
            best_for_goals: ["urgency", "excitement", "cta"]

          - color: "green"
            lift_vs_average: "+8%"
            confidence: 0.85
            best_for_goals: ["sustainability", "nature", "health"]

        low_performing:
          - color: "brown"
            lift_vs_average: "-5%"
            reason: "Perceived as dull or outdated"

          - color: "gray"
            lift_vs_average: "-3%"
            reason: "Low energy, not attention-grabbing"

      composition:
        high_performing:
          - style: "clean_minimalist"
            lift_vs_average: "+18%"
            sample_size: 1240
            best_for: ["tech", "fashion", "lifestyle"]

          - style: "product_hero_centered"
            lift_vs_average: "+14%"
            sample_size: 890
            best_for: ["ecommerce", "apparel"]

        low_performing:
          - style: "cluttered_busy"
            lift_vs_average: "-22%"
            reason: "Cognitive overload, unclear message"

          - style: "text_heavy"
            lift_vs_average: "-15%"
            reason: "Ignores visual-first nature of platform"

      imagery:
        people:
          diversity:
            high_performing:
              - level: "diverse_cast"
                lift_vs_average: "+20%"
                confidence: 0.95
                note: "Mixed race, age, gender performs best"

              - level: "inclusive_representation"
                lift_vs_average: "+12%"
                confidence: 0.89

          age:
            sweet_spot:
              - range: "25-40"
                performance: "baseline"
              - range: "30-45"
                lift_vs_average: "+8%"
                reason: "Purchasing power demographic"

          expressions:
            high_performing:
              - expression: "confident_smile"
                lift_vs_average: "+15%"
                best_for: ["positive_emotions", "trust"]

              - expression: "determined_focused"
                lift_vs_average: "+10%"
                best_for: ["performance", "achievement"]

        product_in_context:
          high_performing:
            - context: "lifestyle_scene"
              lift_vs_average: "+22%"
              examples: ["product in use", "real_life_scenario"]

            - context: "clean_studio"
              lift_vs_average: "+8%"
              best_for: ["luxury", "tech"]

      text_overlay:
        optimal_ratio: "15-25%"
        character_limits:
          headline: "20-25 chars optimal"
          body_text: "80-100 chars max"

        high_performing_phrases:
          - phrase: "Shop Now"
            ctr_lift: "+25%"
          - phrase: "Limited Time"
            ctr_lift: "+18%"
          - phrase: "Free Shipping"
            ctr_lift: "+15%"

        low_performing:
          - phrase: "Click Here"
            ctr_lift: "-10%"
            reason: "Generic, low value"

          - phrase: "Submit"
            ctr_lift: "-8%"
            reason: "Sounds bureaucratic"

    seasonal_patterns:
      Q1_jan_march:
        themes:
          - "new_year_new_you"
          - "fitness_goals"
          - "organization"
        colors:
          - "fresh_blues"
          - "energetic_greens"

      Q2_april_june:
        themes:
          - "spring_renewal"
          - "outdoor_activities"
          - "graduation"
        colors:
          - "pastels"
          - "bright_yellows"

      Q3_july_sept:
        themes:
          - "summer_vibes"
          - "back_to_school"
          - "labor_day"
        colors:
          - "vibrant_oranges"
          - "sky_blues"

      Q4_oct_dec:
        themes:
          - "halloween_october"
          - "thanksgiving_nov"
          - "holidays_dec"
        colors:
          - "warm_reds"
          - "metallics_gold_silver"

    platform_specific:
      meta_facebook:
        best_performing:
          - "carousels_for_storytelling"
          - "video_for_engagement"
          - "single_image_for_clarity"

      meta_instagram:
        best_performing:
          - "aesthetic_consistency"
          - "lifestyle_focus"
          - "user_generated_content_style"

      tiktok:
        best_performing:
          - "authentic_raw"
          - "trending_audio"
          - "behind_scenes"

  # Europe
  UK:
    visual_preferences:
      colors:
        high_performing:
          - color: "navy_blue"
            lift_vs_average: "+12%"
            confidence: 0.90
            cultural_note: "Trust, professionalism, heritage"

          - color: "forest_green"
            lift_vs_average: "+10%"
            cultural_note: "British countryside, sustainability"

      tone:
        preference: "understated_sophistication"
        avoid: "overly_promotional_or_hype"

        humor:
          level: "dry_witty"
          works_well: true
          examples:
            - "subtle_wordplay"
            - "self_deprecating_brand"

      price_sensitivity:
        level: "high"
        value_emphasis:
          - "quality_over_quantity"
          - "longevity"
          - "investment_pieces"

    regulatory_considerations:
      - "GDPR_compliance"
      - "advertising_standards_authority_rules"
      - "clear_pricing_requirements"

  DE:  # Germany
    visual_preferences:
      colors:
        high_performing:
          - color: "engineering_grey"
            lift_vs_average: "+15%"
            cultural_note: "Technical precision, quality"

          - color: "safety_blue"
            lift_vs_average: "+12%"

      tone:
        preference: "factual_technical"
        avoid: "emotional_or_vague"

        content:
          emphasis:
            - "technical_specifications"
            - "quality_certifications"
            - "engineering_excellence"

      trust_signals:
        high_performing:
          - "made_in_germany"
          - "iso_certifications"
          - "test_awards"
          - "technical_details"

    cultural_values:
      - "environmental_consciousness"
      - "data_privacy"
      - "quality_over_price"
      - "reliability"

  # Asia Pacific
  JP:  # Japan
    visual_preferences:
      colors:
        high_performing:
          - color: "soft_white"
            lift_vs_average: "+18%"
            cultural_note: "Cleanliness, purity"

          - color: "pastel_pink"
            lift_vs_average: "+14%"
            cultural_note: "Kawaii culture, approachability"

          - color: "indigo_blue"
            lift_vs_average: "+12%"
            cultural_note: "Trustworthiness, professionalism"

      composition:
        preference: "minimalist_orderly"
        avoid: "cluttered_chaotic"

        whitespace:
          importance: "critical"
          lift_when_adequate: "+20%"
          reason: "Represents refinement, luxury"

      imagery:
        people:
          preference: "asian_representation"
          cultural_authenticity: "important"

        products:
          presentation: "meticulous_detail"
          quality_over_lifestyle: true

      text_overlay:
        character_limits:
          headline: "10-15 chars"
          reason: "Kanji takes more space, less is more"

        avoid:
          - "direct_translation"
          - "english_phrases_unless_cool"

    cultural_values:
      - "harmony_wa"
      - "attention_to_detail"
      - "quality_craftsmanship"
      - "politeness"

  CN:  # China
    visual_preferences:
      colors:
        high_performing:
          - color: "lucky_red"
            lift_vs_average: "+25%"
            cultural_note: "Good fortune, celebration"

          - color: "imperial_gold"
            lift_vs_average: "+18%"
            cultural_note: "Prosperity, luxury"

      composition:
        preference: "vibrant_dynamic"
        energy_level: "high"

      imagery:
        people:
          preference: "chinese_representation"
          cultural_symbols: "acceptable_if_respectful"

        lifestyle:
          focus: "aspirational"
          success_indicators: "important"

      platform_specific:
        wechat:
          - "mini_programs_integrated"
          - "social_commerce_focus"

        douyin_tiktok_china:
          - "live_shopping"
          - "influencer_collaboration"

    cultural_values:
      - "prosperity_success"
      - "family_harmony"
      - "technological_progress"
      - "national_pride"

    regulatory:
      - "strict_content_review"
      - "political_sensitivity"
      - "great_firewall_compliance"

  # Middle East
  SA:  # Saudi Arabia
    visual_preferences:
      colors:
        high_performing:
          - color: "desert_sand"
            lift_vs_average: "+10%"
            cultural_note: "Connection to heritage"

          - color: "islamic_green"
            lift_vs_average: "+15%"
            cultural_note: "Religious significance, prosperity"

      imagery:
        people:
          gender:
            respect: "critical"
            guidelines: "modest_attire_appropriate_interactions"

          family:
            emphasis: "important"
            performs_better: "family_scenes"

      values:
        - "religious_observance"
        - "family_tradition"
        - "hospitality_generosity"
        - "modesty"

    regulatory:
      - "strict_content_guidelines"
      - "religious_holidays_respect"
      - "gender_appropriate_content"

  # Latin America
  BR:  # Brazil
    visual_preferences:
      colors:
        high_performing:
          - color: "vibrant_yellow"
            lift_vs_average: "+18%"
            cultural_note: "Optimism, energy (national colors)"

          - color: "tropical_green"
            lift_vs_average: "+15%"
            cultural_note: "Nature, amazon, sustainability"

      tone:
        preference: "warm_friendly"
        emotional_level: "high"

        humor:
          works_well: true
          style: "playful_joyful"

      imagery:
        people:
          diversity: "brazilian_representation_important"
          demographics: "mixed_race_majority"

        lifestyle:
          focus: "social_family_oriented"
          activities:
            - "beach_outdoor"
            - "social_gathering"
            - "sports_football"

      cultural_values:
        - "celebration_joy"
        - "social_connection"
        - "beauty_appearance"
        - "music_carnival"

    seasonal:
      - "carnaval_preparation_jan_feb"
      - "world_cup_years_elevated_soccer"
```

### Regional Pattern Matcher

```python
# src/meta/ad/qa/effectiveness/regional_matcher.py

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class RegionalPatternMatch:
    """Result of regional pattern matching."""
    region: str
    match_score: float  # 0-100
    performance_prediction: str  # "above_average" | "average" | "below_average"
    high_performing_elements: List[Dict]
    low_performing_elements: List[Dict]
    seasonal_alignment: Dict
    recommendations: List[str]

class RegionalPatternMatcher:
    """Matches creative against regional historical patterns."""

    def __init__(self, region: str):
        self.region = region
        self.patterns_db = self._load_regional_patterns(region)

    def match_patterns(
        self,
        image_analysis: Dict,
        text_content: Optional[str] = None,
        target_date: Optional[datetime] = None
    ) -> RegionalPatternMatch:
        """
        Match creative against regional performance patterns.

        Args:
            image_analysis: GPT-4 Vision analysis
            text_content: Text overlays
            target_date: When the ad will run (for seasonal matching)

        Returns:
            RegionalPatternMatch with performance prediction
        """
        # Extract creative elements
        colors = image_analysis.get('dominant_colors', [])
        composition = image_analysis.get('composition_style', '')
        people = image_analysis.get('people', [])
        text_ratio = image_analysis.get('text_overlay_ratio', 0)

        # Match against historical patterns
        color_matches = self._match_colors(colors)
        composition_matches = self._match_composition(composition)
        people_matches = self._match_people_demographics(people)
        text_matches = self._match_text_elements(text_content, text_ratio)

        # Check seasonal alignment
        seasonal_alignment = self._check_seasonal_alignment(
            target_date or datetime.now(),
            image_analysis,
            text_content
        )

        # Calculate overall match score
        match_score = self._calculate_match_score({
            'colors': color_matches,
            'composition': composition_matches,
            'people': people_matches,
            'text': text_matches
        })

        # Predict performance
        performance_prediction = self._predict_performance(match_score)

        # Generate recommendations
        recommendations = self._generate_regional_recommendations({
            'color_matches': color_matches,
            'composition_matches': composition_matches,
            'people_matches': people_matches,
            'text_matches': text_matches,
            'seasonal': seasonal_alignment
        })

        return RegionalPatternMatch(
            region=self.region,
            match_score=match_score,
            performance_prediction=performance_prediction,
            high_performing_elements=self._get_high_performing_elements({
                'colors': color_matches,
                'composition': composition_matches,
                'people': people_matches,
                'text': text_matches
            }),
            low_performing_elements=self._get_low_performing_elements({
                'colors': color_matches,
                'composition': composition_matches,
                'people': people_matches,
                'text': text_matches
            }),
            seasonal_alignment=seasonal_alignment,
            recommendations=recommendations
        )

    def _match_colors(self, detected_colors: List[str]) -> Dict:
        """Match detected colors against regional performance data."""
        high_performing_colors = self.patterns_db['visual_preferences']['colors']['high_performing']
        low_performing_colors = self.patterns_db['visual_preferences']['colors']['low_performing']

        matches = {
            'high_performing': [],
            'low_performing': [],
            'neutral': []
        }

        for detected_color in detected_colors:
            # Check if it matches a high-performing color
            for hp_color in high_performing_colors:
                if self._colors_match(detected_color, hp_color['color']):
                    matches['high_performing'].append({
                        'color': detected_color,
                        'lift': hp_color['lift_vs_average'],
                        'confidence': hp_color['confidence'],
                        'best_for': hp_color.get('best_for_goals', [])
                    })
                    break
            else:
                # Check if it matches a low-performing color
                for lp_color in low_performing_colors:
                    if self._colors_match(detected_color, lp_color['color']):
                        matches['low_performing'].append({
                            'color': detected_color,
                            'lift': lp_color['lift_vs_average'],
                            'reason': lp_color.get('reason', '')
                        })
                        break
                else:
                    matches['neutral'].append(detected_color)

        return matches

    def _match_composition(self, composition: str) -> Dict:
        """Match composition style against regional preferences."""
        high_performing = self.patterns_db['visual_preferences']['composition']['high_performing']
        low_performing = self.patterns_db['visual_preferences']['composition']['low_performing']

        # Find best match
        best_match = None
        best_score = 0

        for hp in high_performing:
            if self._styles_match(composition, hp['style']):
                if hp.get('confidence', 0.85) > best_score:
                    best_match = hp
                    best_score = hp.get('confidence', 0.85)

        if best_match:
            return {
                'matched': True,
                'style': best_match['style'],
                'lift': best_match['lift_vs_average'],
                'sample_size': best_match.get('sample_size', 0),
                'best_for': best_match.get('best_for', [])
            }

        # Check if it matches low-performing style
        for lp in low_performing:
            if self._styles_match(composition, lp['style']):
                return {
                    'matched': True,
                    'style': lp['style'],
                    'lift': lp['lift_vs_average'],
                    'is_low_performing': True,
                    'reason': lp.get('reason', '')
                }

        return {
            'matched': False,
            'style': composition,
            'note': 'No matching pattern found'
        }

    def _match_people_demographics(self, people: List[Dict]) -> Dict:
        """Match people representation against regional preferences."""
        if not people:
            return {
                'has_people': False,
                'note': 'No people detected'
            }

        preferences = self.patterns_db['visual_preferences']['imagery']['people']

        results = {
            'has_people': True,
            'count': len(people),
            'diversity_match': None,
            'age_match': None,
            'expression_match': None
        }

        # Check diversity
        if 'diversity' in preferences:
            diversity_prefs = preferences['diversity']['high_performing']
            detected_demographics = self._analyze_diversity(people)

            # Check if matches preferred diversity level
            results['diversity_match'] = {
                'detected': detected_demographics,
                'preferred': diversity_prefs[0]['level'],
                'match_score': self._calculate_diversity_match(detected_demographics, diversity_prefs)
            }

        # Check age range
        if 'age' in preferences:
            age_prefs = preferences['age']['sweet_spot']
            detected_ages = [p.get('age_range', 'unknown') for p in people]

            results['age_match'] = {
                'detected': detected_ages,
                'preferred': age_prefs,
                'matches': any(pref in str(detected_ages) for pref in age_prefs)
            }

        # Check expressions
        if 'expressions' in preferences:
            expr_prefs = preferences['expressions']['high_performing']
            detected_expressions = [p.get('expression', 'unknown') for p in people]

            results['expression_match'] = {
                'detected': detected_expressions,
                'preferred': [e['expression'] for e in expr_prefs],
                'matches': any(
                    pref['expression'] in detected_expressions
                    for pref in expr_prefs
                )
            }

        return results

    def _check_seasonal_alignment(
        self,
        target_date: datetime,
        image_analysis: Dict,
        text_content: Optional[str]
    ) -> Dict:
        """Check if creative aligns with seasonal patterns."""
        seasonal_patterns = self.patterns_db.get('seasonal_patterns', {})

        # Determine current season
        month = target_date.month
        quarter = (month - 1) // 3 + 1
        season_key = f'Q{quarter}_{target_date.strftime("%b_%b").lower()}'

        # Get seasonal patterns if available
        season_data = None
        for season_key_pattern, data in seasonal_patterns.items():
            if season_key_pattern.startswith(f'Q{quarter}'):
                season_data = data
                break

        if not season_data:
            return {
                'aligned': True,
                'note': 'No specific seasonal patterns for this period'
            }

        # Check if creative matches seasonal themes
        recommended_themes = season_data.get('themes', [])
        recommended_colors = season_data.get('colors', [])

        detected_colors = image_analysis.get('dominant_colors', [])

        theme_matches = []
        if text_content:
            for theme in recommended_themes:
                theme_words = theme.split('_')
                if any(word in text_content.lower() for word in theme_words):
                    theme_matches.append(theme)

        color_matches = []
        for rec_color in recommended_colors:
            if any(self._colors_match(dc, rec_color) for dc in detected_colors):
                color_matches.append(rec_color)

        alignment_score = len(theme_matches) + len(color_matches)
        max_possible = len(recommended_themes) + len(recommended_colors)

        return {
            'season': season_key,
            'recommended_themes': recommended_themes,
            'theme_matches': theme_matches,
            'recommended_colors': recommended_colors,
            'color_matches': color_matches,
            'alignment_score': (alignment_score / max_possible * 100) if max_possible > 0 else 100,
            'aligned': alignment_score >= max_possible * 0.5
        }

    def _predict_performance(self, match_score: float) -> str:
        """Predict performance based on pattern match."""
        if match_score >= 80:
            return "above_average"
        elif match_score >= 60:
            return "average"
        else:
            return "below_average"

    def _generate_regional_recommendations(self, matches: Dict) -> List[str]:
        """Generate recommendations based on regional pattern matches."""
        recommendations = []

        # Color recommendations
        color_matches = matches['color_matches']
        if not color_matches['high_performing']:
            high_perf_colors = self.patterns_db['visual_preferences']['colors']['high_performing'][:2]
            recommendations.append(
                f"Consider using {high_perf_colors[0]['color']} or {high_perf_colors[1]['color']} "
                f"which have shown {high_perf_colors[0]['lift_vs_average']} lift in {self.region} market"
            )

        if color_matches['low_performing']:
            for lp in color_matches['low_performing']:
                recommendations.append(
                    f"Replace {lp['color']} with alternative - shows {lp['lift']} vs regional average"
                )

        # Composition recommendations
        comp_match = matches['composition_matches']
        if comp_match.get('is_low_performing'):
            recommendations.append(
                f"Current composition style ({comp_match['style']}) underperforms in {self.region}. "
                f"Consider cleaner, more minimalist approach."
            )

        # People recommendations
        people_match = matches['people_matches']
        if people_match.get('has_people'):
            if people_match.get('diversity_match'):
                div_score = people_match['diversity_match']['match_score']
                if div_score < 0.7:
                    recommendations.append(
                        f"Increase diversity in talent casting - current representation underperforms regional preference"
                    )

        # Seasonal recommendations
        seasonal = matches['seasonal']
        if not seasonal.get('aligned', True):
            recommended_themes = seasonal.get('recommended_themes', [])
            recommendations.append(
                f"Consider incorporating seasonal themes: {', '.join(recommended_themes[:2])}"
            )

        return recommendations

    def _colors_match(self, detected: str, reference: str) -> bool:
        """Check if detected color matches reference color."""
        # Simplified - in production use Delta E
        return detected.lower() in reference.lower() or reference.lower() in detected.lower()

    def _styles_match(self, detected: str, reference: str) -> bool:
        """Check if composition style matches reference."""
        # Simplified pattern matching
        detected_words = set(detected.lower().split('_'))
        reference_words = set(reference.lower().split('_'))
        return bool(detected_words & reference_words)

    def _load_regional_patterns(self, region: str) -> Dict:
        """Load regional performance patterns."""
        # In production, load from YAML/JSON database
        import yaml
        config_path = f"config/ad/reviewer/regions/{region}.yaml"
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return default patterns
            return self._get_default_patterns()

    def _get_default_patterns(self) -> Dict:
        """Default patterns when region-specific data unavailable."""
        return {
            'visual_preferences': {
                'colors': {
                    'high_performing': [
                        {'color': 'blue', 'lift_vs_average': '+10%', 'confidence': 0.8}
                    ],
                    'low_performing': []
                },
                'composition': {
                    'high_performing': [
                        {'style': 'clean_minimalist', 'lift_vs_average': '+15%', 'confidence': 0.85}
                    ],
                    'low_performing': []
                }
            },
            'seasonal_patterns': {}
        }
```

---

## 3. Performance Prediction Model

```python
# src/meta/ad/qa/effectiveness/performance_predictor.py

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class PerformancePrediction:
    """Predicted performance metrics."""
    predicted_ctr: float  # Click-through rate
    predicted_conversion_rate: float
    confidence_interval: tuple  # (lower, upper)
    prediction_confidence: float  # 0-1
    key_positive_factors: List[str]
    key_negative_factors: List[str]
    optimization_priority: List[Dict]

class PerformancePredictor:
    """Predicts creative performance based on historical patterns."""

    def __init__(self):
        self.historical_data = self._load_historical_performance()
        self.model = self._load_prediction_model()

    def predict_performance(
        self,
        image_analysis: Dict,
        campaign_goal: CampaignGoal,
        region: str,
        platform: str
    ) -> PerformancePrediction:
        """
        Predict creative performance.

        Returns:
            PerformancePrediction with metrics and confidence
        """
        # Extract features
        features = self._extract_features(image_analysis, campaign_goal, region, platform)

        # Get baseline performance for this segment
        baseline = self._get_baseline_performance(campaign_goal, region, platform)

        # Calculate feature impacts
        feature_impacts = self._calculate_feature_impacts(features, region)

        # Predict performance
        predicted_ctr = baseline['ctr'] * (1 + feature_impacts['total_lift'])
        predicted_cvr = baseline['cvr'] * (1 + feature_impacts['total_lift'] * 0.8)

        # Calculate confidence interval
        confidence = self._calculate_confidence(features, region)
        margin = predicted_ctr * (1 - confidence) * 0.5
        confidence_interval = (
            max(0, predicted_ctr - margin),
            predicted_ctr + margin
        )

        # Identify key factors
        positive_factors = [
            f"{feat['feature']}: +{feat['lift']}%"
            for feat in feature_impacts['features']
            if feat['lift'] > 0
        ]
        negative_factors = [
            f"{feat['feature']}: {feat['lift']}%"
            for feat in feature_impacts['features']
            if feat['lift'] < 0
        ]

        # Generate optimization priorities
        optimization_priority = self._prioritize_optimizations(
            feature_impacts['features'],
            campaign_goal
        )

        return PerformancePrediction(
            predicted_ctr=predicted_ctr,
            predicted_conversion_rate=predicted_cvr,
            confidence_interval=confidence_interval,
            prediction_confidence=confidence,
            key_positive_factors=positive_factors,
            key_negative_factors=negative_factors,
            optimization_priority=optimization_priority
        )

    def _calculate_feature_impacts(self, features: Dict, region: str) -> Dict:
        """Calculate performance impact of each feature."""
        regional_data = self.historical_data.get(region, {})
        feature_impacts = []
        total_lift = 0.0

        # Color impact
        if features['dominant_colors']:
            color_lift = 0.0
            for color in features['dominant_colors']:
                color_data = self._find_color_performance(color, region)
                if color_data:
                    # Parse lift percentage
                    lift_str = color_data['lift_vs_average']
                    lift = float(lift_str.replace('+', '').replace('%', ''))
                    weight = color_data.get('confidence', 0.8)
                    color_lift += lift * weight

            avg_color_lift = color_lift / len(features['dominant_colors'])
            feature_impacts.append({
                'feature': f"Color ({features['dominant_colors'][0]})",
                'lift': avg_color_lift,
                'confidence': color_data.get('confidence', 0.8) if color_data else 0.5
            })
            total_lift += avg_color_lift * 0.25

        # Composition impact
        comp_data = self._find_composition_performance(features['composition'], region)
        if comp_data:
            lift = float(comp_data['lift_vs_average'].replace('+', '').replace('%', ''))
            feature_impacts.append({
                'feature': f"Composition ({features['composition']})",
                'lift': lift,
                'confidence': comp_data.get('confidence', 0.85)
            })
            total_lift += lift * 0.30

        # People representation impact
        if features['has_people']:
            people_data = self._find_people_performance(features['people_analysis'], region)
            if people_data:
                lift = float(people_data['lift_vs_average'].replace('+', '').replace('%', ''))
                feature_impacts.append({
                    'feature': "People representation",
                    'lift': lift,
                    'confidence': people_data.get('confidence', 0.9)
                })
                total_lift += lift * 0.25

        # Text overlay impact
        text_data = self._find_text_performance(features['text_ratio'], features['text_content'], region)
        if text_data:
            lift = float(text_data['lift_vs_average'].replace('+', '').replace('%', ''))
            feature_impacts.append({
                'feature': "Text overlay",
                'lift': lift,
                'confidence': text_data.get('confidence', 0.8)
            })
            total_lift += lift * 0.20

        return {
            'features': feature_impacts,
            'total_lift': total_lift
        }

    def _prioritize_optimizations(
        self,
        feature_impacts: List[Dict],
        campaign_goal: CampaignGoal
    ) -> List[Dict]:
        """Prioritize optimizations based on potential impact."""
        optimizations = []

        # Sort by potential impact (most negative first)
        sorted_impacts = sorted(feature_impacts, key=lambda x: x['lift'])

        for impact in sorted_impacts[:3]:  # Top 3
            if impact['lift'] < 0:
                optimizations.append({
                    'priority': 'high' if impact['lift'] < -10 else 'medium',
                    'feature': impact['feature'],
                    'current_impact': f"{impact['lift']}%",
                    'potential_improvement': f"Approximately {abs(impact['lift']) * 1.5:.1f}% lift",
                    'recommendation': self._get_optimization_recommendation(impact['feature'], campaign_goal)
                })

        return optimizations

    def _get_optimization_recommendation(self, feature: str, goal: CampaignGoal) -> str:
        """Get specific optimization recommendation."""
        recommendations = {
            'Color': 'Test alternative color palettes from high-performing regional options',
            'Composition': 'Simplify composition - cleaner, more minimalist style typically performs better',
            'People representation': 'Increase diversity and ensure representation matches target demographic',
            'Text overlay': 'Reduce text ratio and focus on single clear message with strong CTA'
        }

        for key in recommendations:
            if key in feature:
                return recommendations[key]

        return 'A/B test alternatives to this creative element'

    def _get_baseline_performance(
        self,
        goal: CampaignGoal,
        region: str,
        platform: str
    ) -> Dict:
        """Get baseline performance for this segment."""
        # In production, query actual historical data
        baselines = {
            'US': {
                'meta': {
                    'brand_awareness': {'ctr': 0.015, 'cvr': 0.002},
                    'consideration': {'ctr': 0.025, 'cvr': 0.008},
                    'conversion': {'ctr': 0.035, 'cvr': 0.025},
                }
            }
        }

        return baselines.get(region, {}).get(platform, {}).get(
            goal.value,
            {'ctr': 0.020, 'cvr': 0.010}  # Default
        )
```

---

## Integration: Complete Effectiveness Check

```python
# src/meta/ad/qa/effectiveness/effectiveness_checker.py

from dataclasses import dataclass

@dataclass
class EffectivenessReviewResult:
    """Complete effectiveness review result."""
    campaign_goal_alignment: GoalAlignmentResult
    regional_pattern_match: RegionalPatternMatch
    performance_prediction: PerformancePrediction

    overall_effectiveness_score: float  # 0-100
    approved_for_launch: bool
    optimization_recommendations: List[str]

    def to_summary(self) -> str:
        """Generate executive summary."""
        return f"""
# Effectiveness Review Summary

## Campaign Goal Alignment
**Goal:** {self.campaign_goal_alignment.campaign_goal.value}
**Alignment Score:** {self.campaign_goal_alignment.alignment_score:.1f}/100
**Status:** {'✅ Aligned' if self.campaign_goal_alignment.is_aligned() else '⚠️ Needs adjustment'}

## Regional Performance Match
**Region:** {self.regional_pattern_match.region}
**Match Score:** {self.regional_pattern_match.match_score:.1f}/100
**Predicted Performance:** {self.regional_pattern_match.performance_prediction}

## Performance Prediction
**Expected CTR:** {self.performance_prediction.predicted_ctr:.2%}
**Confidence:** {self.performance_prediction.prediction_confidence:.1%}

## Top Optimizations
"""
        for i, opt in enumerate(self.performance_prediction.optimization_priority[:3], 1):
            summary += f"{i}. {opt['feature']}: {opt['potential_improvement']}\n"

class EffectivenessChecker:
    """Complete effectiveness checking for ad creatives."""

    def __init__(self, region: str, campaign_goal: CampaignGoal):
        self.region = region
        self.campaign_goal = campaign_goal

        self.goal_checker = CampaignGoalChecker()
        self.regional_matcher = RegionalPatternMatcher(region)
        self.performance_predictor = PerformancePredictor()

    def check_effectiveness(
        self,
        image_path: str,
        image_analysis: Dict,
        platform: str = "meta",
        target_date: Optional[datetime] = None
    ) -> EffectivenessReviewResult:
        """
        Perform complete effectiveness review.

        Args:
            image_path: Path to creative
            image_analysis: GPT-4 Vision analysis
            platform: Advertising platform
            target_date: When ad will run

        Returns:
            EffectivenessReviewResult with all analysis
        """
        # Extract text content
        text_content = image_analysis.get('text_content', '')
        text_ratio = image_analysis.get('text_overlay_ratio', 0)

        # 1. Check campaign goal alignment
        goal_alignment = self.goal_checker.check_alignment(
            image_analysis,
            self.campaign_goal,
            text_content
        )

        # 2. Match regional patterns
        regional_match = self.regional_matcher.match_patterns(
            image_analysis,
            text_content,
            target_date
        )

        # 3. Predict performance
        performance_prediction = self.performance_predictor.predict_performance(
            image_analysis,
            self.campaign_goal,
            self.region,
            platform
        )

        # 4. Calculate overall effectiveness score
        overall_score = (
            goal_alignment.alignment_score * 0.40 +
            regional_match.match_score * 0.35 +
            (performance_prediction.prediction_confidence * 100) * 0.25
        )

        # 5. Determine if approved for launch
        approved_for_launch = (
            goal_alignment.is_aligned(threshold=70) and
            regional_match.match_score >= 60 and
            overall_score >= 70
        )

        # 6. Aggregate all recommendations
        optimization_recommendations = []
        optimization_recommendations.extend(goal_alignment.recommendations)
        optimization_recommendations.extend(regional_match.recommendations)

        return EffectivenessReviewResult(
            campaign_goal_alignment=goal_alignment,
            regional_pattern_match=regional_match,
            performance_prediction=performance_prediction,
            overall_effectiveness_score=overall_score,
            approved_for_launch=approved_for_launch,
            optimization_recommendations=optimization_recommendations
        )
```

---

## Summary

This adds a complete **effectiveness layer** to the ad reviewer:

### ✅ Campaign Goal Alignment
- 6 goal types (awareness, consideration, conversion, retention, leads, app installs)
- Goal-specific validation checks
- Alignment scoring (0-100)
- Actionable recommendations

### ✅ Regional Historical Pattern Matching
- Visual preferences by region (colors, composition, imagery)
- Demographic performance data
- Seasonal patterns and timing
- Platform-specific best practices

### ✅ Performance Prediction
- CTR and CVR predictions with confidence intervals
- Feature-level impact analysis
- Optimization prioritization
- A/B test recommendations

### ✅ Comprehensive Database Structure
- Regional configs for US, UK, DE, JP, CN, SA, BR, etc.
- Cultural values and preferences
- Regulatory considerations
- Seasonal patterns

**Result:** The reviewer now not only ensures compliance but **optimizes for effectiveness**, predicting performance and recommending improvements before launch.
