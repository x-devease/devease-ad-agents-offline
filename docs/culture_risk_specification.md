# Culture Risk & Safety Check Specification

## Overview

This specification defines comprehensive culture risk and safety checks for automated ad review across different locales. It addresses:

1. **Religious misfit** - Inappropriate religious imagery, symbols, or references
2. **Fraudulent content** - Misleading claims, fake endorsements, deception
3. **Violence & weapons** - Guns, blood, fighting, aggressive imagery
4. **Adult/sexual content** - Nudity, sexualization, inappropriate content
5. **Hate speech & discrimination** - Racist, sexist, or discriminatory content
6. **Political sensitivity** - Political symbols, controversial figures, elections
7. **Self-harm & dangerous activities** - Self-injury, risky behaviors
8. **Substance abuse** - Drugs, alcohol, tobacco promotion
9. **Child safety** - Child exploitation, inappropriate depiction of minors
10. **Privacy violations** - Personal information, non-consented imagery

**All checks are locale-specific** with market-appropriate severity levels and cultural context.

---

## Risk Category Structure

```yaml
risk_checks:
  # Risk detection enabled/disabled
  enabled: true

  # Overall strictness level
  strictness: enum                    # conservative | moderate | permissive

  # Locale-specific overrides
  locale_overrides:
    - locale: "SA"                    # Saudi Arabia
      strictness: "conservative"      # Stricter checks
      additional_checks:
        - "gender_segregation"
        - "religious_textures"

    - locale: "US"                    # United States
      strictness: "moderate"
      relaxed_checks:
        - "alcohol_promotion"         # Legal with restrictions

  # Risk categories
  categories:
    religious_misfit:
      enabled: true
      severity: "high"                # low | medium | high | critical
      auto_reject: true               # Auto-reject on violation

    fraudulent_content:
      enabled: true
      severity: "critical"
      auto_reject: true

    violence_weapons:
      enabled: true
      severity: "high"
      auto_reject: false              # Flag for review

    # ... (other categories)
```

---

## 1. Religious Misfit & Sensitivity

### Schema

```yaml
risk_checks:
  categories:
    religious_misfit:
      enabled: true
      severity: "high"
      auto_reject: true

      # Visual checks (GPT-4 Vision)
      visual_checks:
        # Forbidden religious symbols by locale
        forbidden_symbols:
          # Global (forbidden in ads generally)
          global:
            - "swastika"
            - "isis_flag"
            - "terrorist_symbols"

          # Locale-specific
          - locale: "SA"              # Saudi Arabia
            symbols:
              - "non_islam_religious_symbols"  # Cross, Star of David, Buddha
              - "prophets_depictions"
              - "religious_figures"
              reason: "Religious sensitivity"
            allowed:
              - "islamic_geometric_patterns"
              - "calligraphy"

          - locale: "IN"              # India
            symbols:
              - "religious_figures_in_commercial"
              - "sacred_texts_used_commercially"
            sensitive_contexts:
              - "religious_humor"
              - "religious_stereotypes"

          - locale: "FR"              # France
            symbols:
              - "religious_symbols_in_public_institutions"
            allowed:
              - "secular_religious_expression"

        # Inappropriate religious imagery
        inappropriate_imagery:
          - "religious_figures_in_commercial_contexts"
          - "sacred_texts_used_as_decorations"
          - "religious_gestures_used_commercially"
          - "religious_attire_misappropriated"

        # Religious holidays and festivals
        holiday_restrictions:
          - locale: "global"
            holidays:
              - name: "ramadan"
                restrictions:
                  - "no_food_or_drink_imagery"  # Fasting hours
                  - "no_revealing_clothing"
                  - "respectful_modest_tone"

              - name: "christmas"
                restrictions:
                  - "secular_only_if_targeting_general_audience"
                  - "avoid_religious_imagery_in_marketing"

              - name: "diwali"
                restrictions:
                  - "no_firecrackers"           # Safety + environmental
                  - "respectful_use_of_lamps"

      # Text overlay checks
      text_checks:
        forbidden_phrases:
          # Global
          global:
            - regex: ".*god.*.*
              severity: "medium"
              reason: "Avoid religious claims"

            - regex: ".*miracle.*"
              severity: "medium"
              reason: "Avoid supernatural claims"

          # Locale-specific
          - locale: "PK"              # Pakistan
            phrases:
              - "blasphemous_terms"
              - "religious_insults"
            allowed:
              - "respectful_religious_greetings"

        # Religious terminology requiring caution
        caution_terms:
          - term: "pray"
            context_check: true
            allowed_contexts: ["idiomatic_use", "non_religious"]
            forbidden_contexts: ["religious_invocation"]

          - term: "blessed"
            context_check: true
            severity: "low"

        # Claim restrictions
        claim_restrictions:
          - "divine_promises"              # "God's promise"
          - "religious_endorsements"       # "Church approved"
          - "faith_based_claims"           # "Have faith in..."
          - "religious_guarantees"         # "Answered prayers"

      # Locale-specific religious contexts
      locale_contexts:
        - locale: "IL"              # Israel
          considerations:
            - "jewish_sabbath"              # No business on Shabbat
            - "kashrut_certifications"      # Kosher symbols
            - "religious_holidays"
          forbidden:
            - "pork_products"
            - "mixing_meat_and_dairy_imagery"

        - locale: "AE"              # United Arab Emirates
          considerations:
            - "islamic_values"
            - "ramadan_timing"
            - "prayer_times"
          forbidden:
            - "revealing_clothing"
            - "alcohol_consumption"
            - "pork_products"

        - locale: "TR"              # Turkey
          considerations:
            - "secular_vs_islamic_balance"
            - "ataturk_imagery"             # Protected symbol
          forbidden:
            - "disrespecting_national_founders"
            - "terrorism_symbols"

        - locale: "ID"              # Indonesia
          considerations:
            - "muslim_majority_sensitivity"
            - "religious_harmony"
          forbidden:
            - "religious_conversion_content"
            - "sectarian_conflict_imagery"
```

### Examples to Flag

```yaml
# Flag examples for religious misfit
flag_examples:
  religious_misfit:
    - description: "Religious figure used in commercial"
      image_elements:
        - "person_depicted_as_religious_figure"
        - "product_positioned_as_religious_offering"
      severity: "critical"
      action: "reject"

    - description: "Sacred text used as background pattern"
      image_elements:
        - "religious_script_decorative_use"
      text_elements:
        - "sacred_text_background_texture"
      severity: "high"
      action: "flag_for_review"

    - description: "Religious gesture commercialized"
      image_elements:
        - "praying_hands_with_product"
        - "namaste_gesture_for_sales"
      severity: "medium"
      action: "flag_for_review"

    - description: "Inappropriate for Ramadan"
      locale: "SA"
      image_elements:
        - "people_eating_or_drinking"
        - "revealing_clothing"
      date_sensitive:
        event: "ramadan"
      severity: "high"
      action: "reject"
```

---

## 2. Fraudulent Content & Deception

### Schema

```yaml
risk_checks:
  categories:
    fraudulent_content:
      enabled: true
      severity: "critical"
      auto_reject: true

      # Visual deception
      visual_checks:
        fake_endorsements:
          - "celebrity_endorsement_without_permission"
          - "government_official_endorsement"
          - "expert_endorsement_fake"
          - "customer_testimonials_fabricated"

        misleading_imagery:
          - "before_after_manipulation"
          - "product_results_exaggerated"
          - "fake_product_shots"
          - "stock_photos misrepresented as real"

        fake_urgency:
          - "fake_countdown_timers"
          - "false_scarcity"                # "Only 3 left!"
          - "fake_limited_time"
          - "manufactured_urgency"

        impersonation:
          - "fake_brand logos"
          - "impersonating_official_brands"
          - "fake_certifications"
          - "fake_badges_or_awards"

      # Text deception
      text_checks:
        misleading_claims:
          - pattern: "100%.*"
            severity: "high"
            requires_proof: true
            examples:
              - "100% natural"
              - "100% effective"
              - "100% satisfaction"

          - pattern: ".*guarantee.*"
            severity: "high"
            requires_disclaimer: true
            examples:
              - "money back guarantee"
              - "lifetime guarantee"
              - "guaranteed results"

          - pattern: ".*(free|risk|no cost).*"
            severity: "medium"
            requires_transparency: true
            check_hidden_costs: true

        fake_social_proof:
          - pattern: ".*review.*"
            check_authenticity: true
            examples:
              - "5000+ 5-star reviews"
              - "#1 rated product"

          - pattern: ".*best.*"
            requires_evidence: true
            examples:
              - "best in class"
              - "best selling"
              - "best price"

        price_deception:
          - "fake_original_price"           # "Was $999, now $99!"
          - "hidden_fees_not_disclosed"
          - "bait_and_switch_pricing"
          - "false_price_comparison"

        health_fraud:
          - pattern: ".*(cure|heal|treat).*"
            requires_medical_disclaimer: true
            severity: "critical"
            examples:
              - "cures insomnia"
              - "treats anxiety"
              - "heals pain"

          - pattern: ".*(lose.*weight|burn.*fat).*"
            requires_evidence: true
            severity: "high"

          - "false_medical_claims"
          - "fake_before_after_pictures"
          - "unproven_scientific_claims"

      # Locale-specific fraud patterns
      locale_patterns:
        - locale: "US"
          regulations:
            - "FTC_guidelines_compliance"
            - "FDA_claims_if_health_related"
          required_disclaimers:
            - "results_not_typical"
            - "medical_disclaimer_if_applicable"

        - locale: "EU"
          regulations:
            - "GDPR_compliance"
            - "consumer_protection_laws"
          required_disclaimers:
            - "terms_and_conditions_link"
            - "price_including_vat"

        - locale: "BR"              # Brazil
          regulations:
            - "PROCON_rules"              # Consumer protection
            - "clear_pricing_required"
```

### Machine Detection Rules

```yaml
fraud_detection:
  # GPT-4 Vision prompts for detection
  vision_prompts:
    fake_endorsement: |
      Detect if this image contains:
      1. Celebrity or public figure faces
      2. Government or official seals/badges
      3. Testimonial quotes or reviews
      4. Expert credentials or certifications
      5. Any indicators of endorsement

      Return: {
        "has_celebrity": boolean,
        "has_official_seal": boolean,
        "has_testimonial": boolean,
        "faces_detected": [names if identifiable],
        "risk_score": 0-1
      }

    misleading_imagery: |
      Analyze if this image contains:
      1. Before/after comparisons
      2. Exaggerated product results
      3. Unrealistic product demonstrations
      4. Manipulated or AI-generated content

      Return: {
        "has_before_after": boolean,
        "results_appear_exaggerated": boolean,
        "appears_ai_generated": boolean,
        "manipulation_indicators": [list],
        "risk_score": 0-1
      }

  # Text analysis rules
  text_patterns:
    absolute_claims:
      - regex: "\\b(100%|always|never|every|all|none|perfect|guarantee)\\b"
        severity: "high"
        requires_proof: true

    superlatives:
      - regex: "\\b(best|worst|#1|top|leading|premier|ultimate)\\b"
        severity: "medium"
        requires_comparative_data: true

    urgency_indicators:
      - regex: "\\b(now|today|limited|only|hurry|last chance|expires|ending)\\b"
        severity: "low"
        requires_genuine_scarcity: true
```

---

## 3. Violence & Weapons

### Schema

```yaml
risk_checks:
  categories:
    violence_weapons:
      enabled: true
      severity: "high"
      auto_reject: false

      visual_checks:
        # Weapons by category
        weapons:
          firearms:
            - "handguns"
            - "rifles"
            - "shotguns"
            - "assault_weapons"
            allowed_contexts: []           # No acceptable contexts
            severity: "critical"

          melee_weapons:
            - "knives"
            - "swords"
            - "axes"
            - "baseball_bats_used_as_weapons"
            allowed_contexts:
              - "kitchen_cutlery_in_cooking_context"
              - "sports_equipment_in_sports_context"
            severity: "high"

          improvised_weapons:
            - "objects_used_as_weapons"
            - "broken_glass"
            - "improvised_club"
            allowed_contexts: []
            severity: "high"

        # Violence indicators
        violence:
          physical_violence:
            - "fighting"
            - "punching"
            - "kicking"
            - "physical_assault"
            severity: "critical"

          blood_gore:
            - "visible_blood"
            - "injuries"
            - "wounds"
            - "gore"
            severity: "critical"

          threatening_behavior:
            - "threatening_postures"
            - "intimidating_gestures"
            - "aggressive_body_language"
            severity: "high"

          self_harm:
            - "cutting"
            - "suicide_attempts"
            - "self_injury"
            severity: "critical"

        # Contextual violence
        contextual:
          - context: "video_games"
            violence: "allowed_if_rating_appropriate"
            max_rating: "T_for_teen"

          - context: "movies_entertainment"
            violence: "allowed_if_appropriately_rated"
            requires_disclaimer: true

          - context: "news_documentary"
            violence: "allowed_if_editorially_necessary"
            requires_context: true

      # Locale-specific rules
      locale_rules:
        - locale: "US"
          gun_culture_sensitivity: "high"
          firearms_in_ads: "generally_prohibited"
          exceptions:
            - "military_recruitment"
            - "licensed_hunting_apparel"

        - locale: "JP"              # Japan
          violence_tolerance: "low"
          weapons: "strictly_prohibited"
          cultural_context: "Low tolerance for violence in media"

        - locale: "DE"              # Germany
          nazi_symbols: "strictly_forbidden"
          violence: "heavily_restricted"
          requires: "jugendschutz" (youth protection) compliance

        - locale: "AU"              # Australia
          weapons: "prohibited"
          violence: "restricted"
          rating_system: "AU_classification_board"

      text_checks:
        violent_language:
          - pattern: ".*(kill|murder|shoot|stab|fight|beat).*"
            severity: "high"
            context_check: true
            allowed_contexts:
              - "sports_metaphors"
              - "gaming_context"

        threat_language:
          - pattern: ".*(threat|hurt|harm|destroy).*"
            severity: "critical"
```

---

## 4. Adult & Sexual Content

### Schema

```yaml
risk_checks:
  categories:
    adult_content:
      enabled: true
      severity: "critical"
      auto_reject: true

      visual_checks:
        nudity:
          - "female_nudity"
          - "male_nudity"
          - "partial_nudity"
          - "implied_nudity"
          allowed_contexts: []

        sexualization:
          - "sexualized_poses"
          - "provocative_clothing"
          - "sexual_gestures"
          - "suggestive_facial_expressions"
          severity: "critical"

        inappropriate_content:
          - "sexual_activity_implied"
          - "fetish_content"
          - "adult_products_unless_context_appropriate"
          severity: "critical"

      text_checks:
        sexual_language:
          - pattern: ".*(explicit terms)*"
            severity: "critical"
            censor: true

        innuendo:
          - pattern: ".*(suggestive phrases)*"
            severity: "high"
            requires_context_review: true

      locale_rules:
        - locale: "US"
          swimwear_allowed: true
          contextual_appropriateness: required

        - locale: "SA"              # Saudi Arabia
          modesty_required: true
          swimwear_imagery: "prohibited"
          gender_mixed_imagery: "restricted"

        - locale: "JP"              # Japan
          sexualization_tolerance: "lower_than_us"
          idol_culture_sensitivity: "be_aware_of_idol_exploitation"
```

---

## 5. Hate Speech & Discrimination

### Schema

```yaml
risk_checks:
  categories:
    hate_speech:
      enabled: true
      severity: "critical"
      auto_reject: true

      protected_characteristics:
        - characteristic: "race"
          slurs: ["forbidden racial terms"]
          stereotypes: ["forbidden racial stereotypes"]
          negative_depictions: ["minstrelsy", "blackface", "racial caricatures"]

        - characteristic: "religion"
          slurs: ["forbidden religious slurs"]
          stereotypes: ["religious caricatures"]
          blasphemy: true             # For some locales

        - characteristic: "gender"
          slurs: ["sexist language"]
          stereotypes: ["gender roles", "gender norms enforced"]
          misogyny: ["objectification", "subordination"]

        - characteristic: "sexual_orientation"
          slurs: ["homophobic slurs"]
          stereotypes: ["harmful LGBTQ+ stereotypes"]
          conversion_therapy: "prohibited"

        - characteristic: "disability"
          slurs: ["ableist language"]
          stereotypes: ["helpless victim", "inspirational_porn"]
          mockery: ["mocking_disability"]

        - characteristic: "age"
          ageism: ["mocking elderly", "age-based discrimination"]

        - characteristic: "national_origin"
          xenophobia: ["anti-immigrant sentiment", "nationality-based hate"]

      visual_indicators:
        - "hate_symbols"
        - "hate_group_flags"
        - "discriminatory_caricatures"
        - "segregation_imagery"
        - "superiority_inferiority_imagery"

      text_indicators:
        - "slurs_and_epithets"
        - "dehumanizing_language"
        - "calls_for_exclusion"
        - "superiority_claims"
        - "stereotype_reinforcement"

      locale_rules:
        - locale: "DE"              # Germany
          nazi_content: "strictly_forbidden"
          holocaust_denial: "criminal_offense"
          hate_speech: "illegal"

        - locale: "FR"              # France
          hate_speech: "illegal"
          denial_of_crimes_against_humanity: "illegal"

        - locale: "US"
          first_amendment: "some_protection"
          platform_rules: "still_apply"
          incitement: "not_protected"
```

---

## 6. Political Sensitivity

### Schema

```yaml
risk_checks:
  categories:
    political_content:
      enabled: true
      severity: "medium"
      auto_reject: false

      visual_checks:
        political_symbols:
          - "national_flags"
          - "political_party_logos"
          - "campaign_materials"
          - "political_figures"

        controversial_figures:
          - "controversial_politicians"
          - "controversial_activists"
          - "polarizing_public_figures"

        election_content:
          - "campaign_advertisements"
          - "voting_instructions"
          - "election_disinformation"

      text_checks:
        political_language:
          - pattern: ".*(vote|election|campaign|candidate).*"
            severity: "medium"
            requires_disclaimer: true

        policy_positions:
          - "partisan_positions"
          - "controversial_policy_advocacy"

      locale_rules:
        - locale: "US"
          election_period: "strict_enforcement_pre_election"
          political_ads: "require_transparency_disclosures"

        - locale: "IN"              # India
          election_commission_rules: "strict"
          political_advertising: "regulated"
          model_code_of_conduct: "during_elections"

        - locale: "BR"              # Brazil
          political_advertising: "heavily_regulated"
          election_periods: "specific_restrictions"
```

---

## 7. Self-Harm & Dangerous Activities

### Schema

```yaml
risk_checks:
  categories:
    self_harm:
      enabled: true
      severity: "critical"
      auto_reject: true

      visual_checks:
        - "self_injury"
        - "cutting"
        - "burning"
        - "suicide_attempts"
        - "hanging_preparation"
        - "overdose_pills_visible"

      text_checks:
        - pattern: ".*(kill myself|want to die|end it all|suicide).*"
          severity: "critical"
          action: "block_and_refer_to_crisis_resources"

        - pattern: ".*(cut|hurt myself|self_harm).*"
          severity: "critical"

    dangerous_activities:
      enabled: true
      severity: "high"
      auto_reject: false

      visual_checks:
        - "dangerous_stunts"
        - "reckless_behavior"
        - "unsafe_product_use"
        - "violation_of_safety_guidelines"

        risky_sports:
          - "extreme_sports_without_safety_gear"
          - "professional_only_stunts"

      text_checks:
        - pattern: ".*(dangerous_stunt_challenges).*"
          severity: "high"

      product_safety:
        - "misuse_of_product"
        - "bypassing_safety_features"
        - "dangerous_modifications"
```

---

## 8. Substance Abuse

### Schema

```yaml
risk_checks:
  categories:
    substance_abuse:
      enabled: true
      severity: "high"
      auto_reject: false

      visual_checks:
        illegal_drugs:
          - "drug_paraphernalia"
          - "drug_use_imagery"
          - "drug_dealing_imagery"
          severity: "critical"
          allowed_contexts: []

        controlled_substances:
          - "prescription_drug_misuse"
          - "pill_bottles_scattered"
          - "excessive_pills_visible"
          severity: "high"

        alcohol:
          - "excessive_consumption"
          - "underage_drinking"
          - "drunk_behavior"
          - "driving_under_influence"
          severity: "high"
          allowed_contexts:
            - "responsible_consumption"
            - "legal_advertising_age_restricted"

        tobacco:
          - "smoking_imagery"
          - "vaping"
          - "chewing_tobacco"
          severity: "high"

      text_checks:
        drug_references:
          - pattern: ".*(high|stoned|wasted|tripping).*"
            severity: "high"

          - pattern: ".*(drug_slang_terms)*"
            severity: "critical"

        alcohol_promotion:
          - pattern: ".*(get_wasted|drink_till_you_drop|party_hard).*"
            severity: "high"

      locale_rules:
        - locale: "US"
          legal_drug_ads: "allowed_with_restrictions"
          alcohol_ads: "legal_with_age_gating"
          tobacco_ads: "prohibited"

        - locale: "SA"              # Saudi Arabia
          alcohol: "strictly_prohibited"
          drugs: "strictly_prohibited"
          penalties: "legal_consequences"

        - locale: "AE"              # UAE
          alcohol: "prohibited_in_ads"
          drugs: "strictly_prohibited"

        - locale: "NL"              # Netherlands
          cannabis: "decriminalized_but_not_in_ads"
          drugs: "regulated"
```

---

## 9. Child Safety

### Schema

```yaml
risk_checks:
  categories:
    child_safety:
      enabled: true
      severity: "critical"
      auto_reject: true

      visual_checks:
        exploitation:
          - "sexualized_children"
          - "children_in_appropriate_contexts"
          - "child_labor_imagery"
          severity: "critical"

        inappropriate_content:
          - "children_with_adult_products"
          - "children_in_dangerous_situations"
          - "children_with_weapons"
          - "children_with_drugs_alcohol"
          severity: "critical"

        privacy:
          - "identifiable_children_without_consent"
          - "school_uniforms_identifiable"
          severity: "high"

      text_checks:
        - pattern: ".*(inappropriate_terms_about_children).*"
          severity: "critical"

      locale_rules:
        - locale: "global"
          COPPA_compliance: "for_US"
          GDPR_kids: "for_EU"
          age_verification: "required_if_applicable"

        - locale: "US"
          COPPA: "Children's Online Privacy Protection Act"
          parental_consent: "required_for_under_13"

        - locale: "EU"
          GDPR_child_consent: "required_for_under_16"
          data_protection: "strict"
```

---

## 10. Privacy Violations

### Schema

```yaml
risk_checks:
  categories:
    privacy_violations:
      enabled: true
      severity: "high"
      auto_reject: false

      visual_checks:
        personal_information:
          - "visible_phone_numbers"
          - "email_addresses"
          - "home_addresses"
          - "license_plates"
          - "social_security_numbers"
          - "credit_card_numbers"

        non_consented_imagery:
          - "private_individuals_without_consent"
          - "cctv_footage"
          - "dashcam_footage"
          - "leaked_private_photos"

      locale_rules:
        - locale: "EU"
          GDPR: "strict_compliance"
          right_to_be_forgotten: "applicable"
          consent: "explicit_required"

        - locale: "CA"              # California
          CCPA: "California Consumer Privacy Act"
          privacy_rights: "protected"

        - locale: "BR"              # Brazil
          LGPD: "Lei Geral de Proteção de Dados"
          data_protection: "strict"
```

---

## Complete Example: Locale-Specific Config

```yaml
# config/ad/reviewer/risks/saudi_arabia.yaml

locale:
  code: "SA"
  name: "Saudi Arabia"
  region: "Middle East"

risk_checks:
  # Overall strictness
  strictness: "conservative"

  # Religious checks (CRITICAL for SA)
  religious_misfit:
    enabled: true
    severity: "critical"
    auto_reject: true

    visual_checks:
      forbidden_symbols:
        - "cross_or_christian_symbols"
        - "star_of_david_or_jewish_symbols"
        - "buddha_or_eastern_religious_figures"
        - "hindu_deities"
        - "prophets_depictions"
        - "religious_figures_in_commercial_context"

      required_cultural_sensitivity:
        - "modest_dress_required"
        - "no_revealing_clothing"
        - "gender_appropriate_interactions"
        - "respectful_use_of_islamic_patterns"

    text_checks:
      forbidden_phrases:
        - ".*god.*.*                    # Unless in Islamic context
        - ".*jesus.*"
        - "religious_conversion_language"

    holiday_restrictions:
      ramadan:
        - "no_food_or_drink_imagery_during_daylight"
        - "modest_clothing_emphasized"
        - "respectful_tone_required"
        - "avoid_party_or_festive_imagery"

      eid:
        - "celebratory_imagery_allowed"
        - "family_focus_appropriate"
        - "religious_observance_respected"

  # Violence/weapons (HIGH concern)
  violence_weapons:
    enabled: true
    severity: "critical"
    auto_reject: true

    visual_checks:
      forbidden:
        - "any_firearms"
        - "weapons_of_any_kind"
        - "violence_or_aggression"
        - "blood_or_gore"

  # Adult content (CRITICAL)
  adult_content:
    enabled: true
    severity: "critical"
    auto_reject: true

    visual_checks:
      forbidden:
        - "any_nudity"
        - "sexualized_content"
        - "revealing_clothing"
        - "gender_mixed_immodesty"

    cultural_rules:
      - "women_in_modest_attire"
      - "no_physical_contact_between_genders"
      - "respectful_depiction_of_women"

  # Substance abuse (CRITICAL)
  substance_abuse:
    enabled: true
    severity: "critical"
    auto_reject: true

    visual_checks:
      forbidden:
        - "alcohol_consumption_or_bottles"
        - "drugs_or_paraphernalia"
        - "smoking_or_vaping"

    text_checks:
      forbidden:
        - "alcohol_promotion"
        - "party_or_club_imagery"

  # Political content (HIGH concern)
  political_content:
    enabled: true
    severity: "high"
    auto_reject: false

    visual_checks:
      caution_required:
        - "royal_family_imagery"         # Special respect required
        - "government_symbols"
        - "national_motto"

  # Cultural values
  cultural_values:
    - "family_oriented"
    - "religious_observance"
    - "modesty_and_decency"
    - "respect_for_tradition"

  # Allowed themes
  allowed_themes:
    - "family_bonding"
    - "technological_innovation"
    - "quality_and_reliability"
    - "adversities_overcome_with_faith"

  # Required disclaimers
  required_disclaimers:
    - "halal_certification_if_food_product"
    - "modesty_warnings_if_applicable"
```

---

## Example: United States Config

```yaml
# config/ad/reviewer/risks/united_states.yaml

locale:
  code: "US"
  name: "United States"
  region: "North America"

risk_checks:
  strictness: "moderate"

  religious_misfit:
    enabled: true
    severity: "medium"
    auto_reject: false

    # US is religiously diverse but generally permissive
    approach: "inclusive_but_respectful"

    visual_checks:
      caution_required:
        - "religious_figures_in_commercial"  # Flag for review
        - "sacred_texts_decorative"

    text_checks:
      caution:
        - "religious_claims_about_products"

  violence_weapons:
    enabled: true
    severity: "high"
    auto_reject: false

    # US has complex relationship with guns
    firearms:
      generally_prohibited: true
      exceptions:
        - "hunting_apparel"
        - "military_recruitment_official"
        - "licensed_shooting_ranges"

    violence:
      tolerance: "low"
      context_matters: true

  adult_content:
    enabled: true
    severity: "high"
    auto_reject: true

    swimwear: "allowed_in_appropriate_context"
    sexualization: "not_allowed"

  substance_abuse:
    enabled: true
    severity: "high"

    alcohol:
      allowed: true
      restrictions:
        - "must_appear_25_or_older"
        - "no_excessive_consumption"
        - "no_drunk_driving_imagery"
        - "responsible_consumption_message_required"

    tobacco:
      allowed: false                  # Tobacco ads heavily restricted

    cannabis:
      federal_illegal: true
      state_legal: "varies"
      approach: "prohibited_in_national_ads"

  fraud_detection:
    enabled: true
    severity: "critical"

    regulatory_compliance:
      - "FTC_guidelines"
      - "FDA_rules_if_health_claims"

    required_for_health_claims:
      - "disclaimer: *These statements have not been evaluated by the FDA"
      - "no_diagnostic_claims"

  political_content:
    enabled: true
    severity: "medium"

    election_periods:
      - "strict_transparency_requirements"
      - "paid_for_by_disclaimer"

  diversity_requirements:
    - "inclusive_representation"
    - "avoid_stereotypes"
    - "cultural_sensitivity"

  cultural_values:
    - "individualism"
    - "freedom"
    - "innovation"
    - "diversity"
```

---

## Implementation: Risk Detection Pipeline

```python
# src/meta/ad/reviewer/culture/risk_detector.py

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskCategory(Enum):
    RELIGIOUS_MISFIT = "religious_misfit"
    FRAUDULENT_CONTENT = "fraudulent_content"
    VIOLENCE_WEAPONS = "violence_weapons"
    ADULT_CONTENT = "adult_content"
    HATE_SPEECH = "hate_speech"
    POLITICAL_CONTENT = "political_content"
    SELF_HARM = "self_harm"
    SUBSTANCE_ABUSE = "substance_abuse"
    CHILD_SAFETY = "child_safety"
    PRIVACY_VIOLATIONS = "privacy_violations"

@dataclass
class RiskViolation:
    """A detected risk violation."""
    category: RiskCategory
    severity: Severity
    description: str
    detected_elements: List[str]
    confidence: float  # 0-1
    locale_specific: bool
    requires_human_review: bool
    auto_reject: bool
    recommendation: str

@dataclass
class RiskCheckResult:
    """Result of risk checking."""
    locale: str
    overall_risk_level: Severity
    violations: List[RiskViolation]
    approved: bool
    confidence: float

    def get_critical_violations(self) -> List[RiskViolation]:
        """Get all critical violations."""
        return [v for v in self.violations if v.severity == Severity.CRITICAL]

    def get_auto_reject_violations(self) -> List[RiskViolation]:
        """Get violations that trigger auto-reject."""
        return [v for v in self.violations if v.auto_reject]

class CultureRiskDetector:
    """Detects culture-based risks in ad creatives."""

    def __init__(self, locale: str):
        self.locale = locale
        self.risk_config = self._load_locale_config(locale)
        self.vision_analyzer = VisionAnalyzer()
        self.text_analyzer = TextAnalyzer()

    def check_risks(
        self,
        image_path: str,
        image_analysis: Dict,
        text_content: Optional[str] = None
    ) -> RiskCheckResult:
        """
        Check for cultural risks in an ad creative.

        Args:
            image_path: Path to the creative image
            image_analysis: Pre-computed image analysis
            text_content: Text overlays in the image

        Returns:
            RiskCheckResult with all detected violations
        """
        violations = []

        # Check each enabled risk category
        for category in self.risk_config['categories']:
            if not category['enabled']:
                continue

            category_violations = self._check_category(
                category,
                image_path,
                image_analysis,
                text_content
            )
            violations.extend(category_violations)

        # Determine overall risk level
        overall_risk = self._calculate_overall_risk(violations)

        # Determine if auto-rejected
        auto_reject_violations = [v for v in violations if v.auto_reject]
        approved = len(auto_reject_violations) == 0

        # Calculate confidence
        confidence = self._calculate_confidence(violations)

        return RiskCheckResult(
            locale=self.locale,
            overall_risk_level=overall_risk,
            violations=violations,
            approved=approved,
            confidence=confidence
        )

    def _check_category(
        self,
        category: Dict,
        image_path: str,
        image_analysis: Dict,
        text_content: Optional[str]
    ) -> List[RiskViolation]:
        """Check a specific risk category."""
        violations = []
        category_name = category['name']

        # Visual checks
        if 'visual_checks' in category:
            visual_violations = self._check_visual_risks(
                category_name,
                category['visual_checks'],
                image_analysis
            )
            violations.extend(visual_violations)

        # Text checks
        if 'text_checks' in category and text_content:
            text_violations = self._check_text_risks(
                category_name,
                category['text_checks'],
                text_content
            )
            violations.extend(text_violations)

        # Locale-specific checks
        if 'locale_rules' in category:
            locale_violations = self._check_locale_rules(
                category_name,
                category['locale_rules'],
                image_analysis,
                text_content
            )
            violations.extend(locale_violations)

        return violations

    def _check_visual_risks(
        self,
        category: str,
        checks: Dict,
        image_analysis: Dict
    ) -> List[RiskViolation]:
        """Check for visual risk indicators."""
        violations = []

        for check_type, items in checks.items():
            if check_type == 'forbidden_symbols':
                detected = self._detect_forbidden_symbols(
                    items,
                    image_analysis
                )
                if detected:
                    violations.append(RiskViolation(
                        category=RiskCategory[category.upper()],
                        severity=Severity.CRITICAL,
                        description=f"Detected forbidden symbol: {', '.join(detected)}",
                        detected_elements=detected,
                        confidence=0.9,
                        locale_specific=False,
                        requires_human_review=False,
                        auto_reject=True,
                        recommendation="Remove forbidden symbols immediately"
                    ))

            elif check_type == 'violence':
                detected = self._detect_violence(image_analysis)
                if detected:
                    violations.append(RiskViolation(
                        category=RiskCategory.VIOLENCE_WEAPONS,
                        severity=Severity.HIGH,
                        description=f"Violent content detected: {detected}",
                        detected_elements=[detected],
                        confidence=0.85,
                        locale_specific=False,
                        requires_human_review=True,
                        auto_reject=False,
                        recommendation="Remove violent imagery or provide justification"
                    ))

        return violations

    def _detect_forbidden_symbols(
        self,
        forbidden_list: List[str],
        image_analysis: Dict
    ) -> List[str]:
        """Detect forbidden symbols in image."""
        detected = []

        # Use GPT-4 Vision for symbol detection
        symbols = image_analysis.get('symbols', [])
        text_content = image_analysis.get('text_content', '')

        for forbidden in forbidden_list:
            # Check if forbidden symbol matches detected symbols
            if any(forbidden.lower() in s.lower() for s in symbols):
                detected.append(forbidden)

        return detected

    def _detect_violence(self, image_analysis: Dict) -> Optional[str]:
        """Detect violent content."""
        # Check for weapons
        weapons = image_analysis.get('weapons_detected', [])
        if weapons:
            return f"Weapons detected: {', '.join(weapons)}"

        # Check for violence indicators
        violence_indicators = image_analysis.get('violence_indicators', [])
        if violence_indicators:
            return f"Violence detected: {', '.join(violence_indicators)}"

        # Check for blood/gore
        if image_analysis.get('blood_detected', False):
            return "Blood or gore detected"

        return None

    def _check_text_risks(
        self,
        category: str,
        checks: Dict,
        text_content: str
    ) -> List[RiskViolation]:
        """Check for text-based risks."""
        violations = []

        for check_type, patterns in checks.items():
            if check_type == 'forbidden_phrases':
                for pattern in patterns:
                    import re
                    if re.search(pattern, text_content, re.IGNORECASE):
                        violations.append(RiskViolation(
                            category=RiskCategory[category.upper()],
                            severity=Severity.HIGH,
                            description=f"Flagged phrase detected matching: {pattern}",
                            detected_elements=[text_content],
                            confidence=0.95,
                            locale_specific=False,
                            requires_human_review=True,
                            auto_reject=False,
                            recommendation="Review flagged phrase for appropriateness"
                        ))

        return violations

    def _calculate_overall_risk(self, violations: List[RiskViolation]) -> Severity:
        """Calculate overall risk level."""
        if not violations:
            return Severity.LOW

        # If any critical violations, overall is critical
        if any(v.severity == Severity.CRITICAL for v in violations):
            return Severity.CRITICAL

        # If any high-severity violations, overall is high
        if any(v.severity == Severity.HIGH for v in violations):
            return Severity.HIGH

        # Count violations
        if len(violations) >= 3:
            return Severity.HIGH
        elif len(violations) >= 2:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _calculate_confidence(self, violations: List[RiskViolation]) -> float:
        """Calculate overall confidence in risk assessment."""
        if not violations:
            return 1.0

        # Average confidence of all violations
        return sum(v.confidence for v in violations) / len(violations)

    def _load_locale_config(self, locale: str) -> Dict:
        """Load locale-specific risk configuration."""
        import yaml

        config_path = f"config/ad/reviewer/risks/{locale.lower()}.yaml"
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Fall back to default config
            return self._get_default_config()
```

---

## Summary

This comprehensive culture risk specification provides:

1. **10 Risk Categories** - Religious, fraud, violence, adult content, hate speech, political, self-harm, substance abuse, child safety, privacy

2. **Locale-Specific Rules** - Different thresholds and requirements for each market (SA, US, EU, JP, etc.)

3. **Visual + Text Checks** - Both image analysis and text overlay scanning

4. **Severity Levels** - Low, medium, high, critical with appropriate actions

5. **Machine-Checkable** - Specific patterns, symbols, and phrases to detect

6. **Auto-Reject Logic** - Clear rules for automatic rejection vs. human review

7. **Cultural Context** - Understanding local norms, values, and taboos

**Key benefits:**
- Prevents costly mistakes and brand damage
- Ensures legal compliance across markets
- Respects cultural sensitivities
- Protects brand reputation
- Scalable automated review with human oversight for edge cases
