# Locale-Specific Cultural & Risk Checks

## Overview

This specification defines comprehensive locale-specific cultural checks and content risk assessment for automated ad review. It covers:

1. **Religious sensitivities** by region/religion
2. **Content risk categories** (fraud, violence, sexual, etc.)
3. **Visual elements** to avoid by culture
4. **Text overlay risks** by language/locale
5. **Symbol/gesture meanings** across cultures
6. **Color associations** by market
7. **Local advertising compliance** requirements
8. **Cultural taboos** by region

---

## Complete Schema

```yaml
# ============================================================================
# LOCALE-SPECIFIC CULTURAL & RISK CHECKS
# ============================================================================

locale_risk_checks:
  # Default/global checks (apply to all locales)
  global_baseline:
    enabled: true

    # Universal content risks (never allowed)
    universal_prohibitions:
      content_types:
        - category: "violence"
          severity: "critical"
          detection_method: "gpt4_vision"
          threshold: "zero_tolerance"
          examples:
            - "weapons, guns, knives"
            - "blood, gore, injuries"
            - "physical altercations, fighting"
            - "threatening gestures, intimidation"

        - category: "sexual_content"
          severity: "critical"
          detection_method: "gpt4_vision"
          threshold: "zero_tolerance"
          examples:
            - "nudity, partial nudity"
            - "sexualized poses, provocative clothing"
            - "suggestive gestures, facial expressions"
            - "intimate physical contact"

        - category: "hate_symbols"
          severity: "critical"
          detection_method: "symbol_database"
          threshold: "zero_tolerance"
          examples:
            - "nazi symbols (swastika, SS, etc.)"
            - "KKK symbols, white supremacist imagery"
            - "terrorist organization symbols"
            - "confederate flag (context-dependent)"

        - category: "illegal_activities"
          severity: "critical"
          detection_method: "gpt4_vision"
          threshold: "zero_tolerance"
          examples:
            - "drug use, drug paraphernalia"
            - "underage drinking, smoking"
            - "criminal activities depiction"
            - "dangerous stunts, self-harm"

        - category: "fraudulent_appearances"
          severity: "high"
          detection_method: "gpt4_vision"
          threshold: "strict"
          examples:
            - "fake urgency indicators (countdowns, 'act now' overlays)"
            - "misleading price presentation (strikethrough abuse)"
            - "fake endorsements, celebrity impersonation"
            - "government impersonation, official badges"
            - "pyramid scheme visuals"
            - "get-rich-quick imagery"

    # Text overlay risks (detected via OCR + GPT-4)
    text_overlay_risks:
      universal_prohibitions:
        - type: "misleading_claims"
          severity: "critical"
          patterns:
            - "\\bguarantee\\b.*\\b100%\\b"
            - "\\brisk\\s+free\\b"
            - "\\bget\\s+rich\\b"
            - "\\b(act|click)\\s+now\\b.*(!{2,}|‼️)"
            - "\\b(limited|only).*\\d+\\s+(left|remaining)\\b"
            - "\\b(expiring|ends|expires)\\s+(today|now|soon)\\b.*!{2,}"

        - type: "hyperbolic_language"
          severity: "medium"
          patterns:
            - "\\b(best|#1|world['']s\\s+(best|leading))\\b"
            - "\\bunbelievable|miracle|magic\\b"
            - "\\b(never|always|every|everyone)\\b"
            - "🔥{2,}|💰{2,}|⚠️{2,}|❗{2,}|‼️{2,}"

        - type: "fear_urgency_tactics"
          severity: "high"
          patterns:
            - "\\b(don['']t|do\\s+not).*\\b(miss|wait)\\b"
            - "\\b(last|final).*\\b(chance|opportunity)\\b"
            - "\\b(will|won['']t).*\\b(regret|sorry)\\b"
            - "\\b(losing|missed|gone|forever)\\b"

        - type: "spam_indicators"
          severity: "medium"
          patterns:
            - "(!!!|‼️|❗️){2,}"
            - "(FREE|WIN|CASH|NOW|HURRY){2,}"
            - "\\b(click|tap|visit).*\\b(link|here|now)\\b.*!{2,}"
            - "ALL CAPS HEADLINE.*!"

  # ---------------------------------------------------------------------------
  # LOCALE-SPECIFIC CHECKS
  # ---------------------------------------------------------------------------

  # United States
  US:
    region_name: "United States"
    cultural_context: "western_individualistic"
    primary_language: "en"
    legal_framework: "FTC Guidelines"

    # Religious sensitivities
    religious_considerations:
      christianity:
        major_holidays:
          - name: "Christmas"
            date_range: "Dec_1_-_Dec_31"
            guidance: "Secular imagery only (snow, winter, gifts)"
            forbidden:
              - "religious figures (Jesus, Mary, saints)"
              - "biblical scenes"
              - "overtly religious messaging"
            allowed:
              - "secular Christmas imagery"
              - "holiday season messaging"

          - name: "Easter"
            date_range: "variable_mar_apr"
            guidance: "Secular spring imagery only"
            forbidden:
              - "crucifixion, crosses"
              - "religious iconography"
            allowed:
              - "eggs, bunnies, spring flowers"

      other_faiths:
        - name: "Jewish holidays"
          sensitivity: "high"
          guidance: "Respectful acknowledgment only, no co-opting"
          forbidden:
            - "religious symbols for commercial gain"
            - "stereotypical imagery"

        - name: "Islamic holidays"
          sensitivity: "high"
          guidance: "Respectful acknowledgment only"
          forbidden:
            - "religious imagery"
            - "cultural stereotypes"

    # Cultural sensitivities
    cultural_sensitivities:
      race_ethnicity:
        sensitivity_level: "high"
        avoid:
          - "racial stereotypes, caricatures"
          - "blackface, brownface, yellowface"
          - "cultural appropriation (native headdresses, etc.)"
          - "historically insensitive imagery (confederate flag, etc.)"
        guidance: "Diverse, authentic representation preferred"

      gender:
        sensitivity_level: "medium"
        avoid:
          - "gender stereotypes (women in kitchen, men with tools)"
          - "sexualized portrayals of any gender"
          - "toxic masculinity imagery"
        guidance: "Balanced, modern gender roles"

      lgbtq_plus:
        sensitivity_level: "medium"
        guidance: "Inclusive representation acceptable and encouraged"
        avoid:
          - "stereotypical portrayals"
          - "tokenism (single LGBTQ+ person for diversity points)"

      political:
        sensitivity_level: "high"
        avoid:
          - "partisan political symbols"
          - "controversial political figures"
          - "election-related imagery"
          - "capitol riot imagery, January 6th references"
        guidance: "Stay apolitical"

    # Visual element restrictions
    visual_restrictions:
      gestures:
        avoid:
          - gesture: "middle_finger"
            meaning: "obscene, offensive"
            severity: "critical"

          - gesture: "OK_sign (thumb+index finger)"
            meaning: "Can be associated with white supremacy in certain contexts"
            severity: "high"
            context_check: true

          - gesture: "thumbs_down"
            meaning: "negative, avoid in advertising"
            severity: "low"

      symbols:
        avoid:
          - "confederate flag"
          - "swastika (in any context)"
          - "noose imagery (racial violence)"
          - "guns pointed at camera"

    # Text overlay restrictions
    text_restrictions:
      language_specific:
        en:
          forbidden_phrases:
            - "chinaman, oriental"
            - "gyp, gypped (derogatory slur)"
            - "illegal alien"
            - "master/slave terminology"

          discouraged_phrases:
            - "crazy, insane (ableist language)"
            - "gyp (slur, use 'cheat' instead)"

      legal_disclaimers_required:
        - trigger: "weight_loss_claims"
          disclaimer: "Results not typical. Individual results vary."

        - trigger: "financial_returns"
          disclaimer: "Not financial advice. Past performance doesn't guarantee future results."

        - trigger: "health_benefits"
          disclaimer: "These statements have not been evaluated by the FDA."

    # Color associations
    color_associations:
      positive:
        - color: "blue"
          meanings: ["trust", "reliability", "corporate"]

        - color: "green"
          meanings: ["money", "environment", "growth"]

      negative:
        - color: "red"
          meanings: ["danger", "debt", "loss"]  # except for sales CTAs

      caution:
        - color: "purple"
          context: "Can imply royalty or luxury, may not fit mass market"

    # Local compliance
    compliance_requirements:
      ftc_guidelines:
        truth_in_advertising:
          enabled: true
          requirements:
            - "Clear disclosure of material connections"
            - "No misleading price claims"
            - "No fake reviews/testimonials"
            - "Clear terms and conditions"

        CAN_spam_act:
          enabled: true
          requirements:
            - "Opt-out mechanism in messages"
            - "Clear sender identification"
            - "No deceptive subject lines"

  # United Kingdom
  GB:
    region_name: "United Kingdom"
    cultural_context: "western_british"
    primary_language: "en"
    legal_framework: "ASA CAP Code"

    religious_considerations:
      christianity:
        major_holidays:
          - name: "Christmas"
            guidance: "More religious acceptance than US, but keep secular for general ads"
            date_range: "Dec_1_-_Dec_31"
            allowed:
              - "modest religious imagery"
              - "traditional Christmas scenes"

          - name: "Easter"
            guidance: "Religious imagery more acceptable"
            allowed:
              - "crosses, religious Easter imagery (modest)"

    cultural_sensitivities:
      class_sensitivity:
        sensitivity_level: "medium"
        avoid:
          - "overt class stereotypes"
          - "looking down on working class"
        guidance: "Class-conscious society, avoid elitist imagery"

      brexit:
        sensitivity_level: "medium"
        avoid:
          - "political EU/UK imagery"
          - "pro/anti-Brexit messaging"
        guidance: "Remain apolitical"

    visual_restrictions:
      gestures:
        avoid:
          - gesture: "V_sign_with_palm_facing_inward"
            meaning: "offensive (similar to middle finger)"
            severity: "critical"

          - gesture: "V_sign_with_palm_outward"
            meaning: "peace/victory sign (acceptable)"
            note: "Palm orientation matters"

      symbols:
        avoid:
          - "IRA imagery, political symbols"
          - "royal family without permission (has rules)"

    text_restrictions:
      language_specific:
        en_GB:
          spelling: "use British spelling (colour, centre, organisation)"
          forbidden_phrases:
            - "soccer (use 'football')"
            - "period (use 'full stop')"

          discouraged_phrases:
            - "awesome, incredible (seen as overly American/hyperbolic)"

    compliance_requirements:
      asa_cap_code:
        truthfulness:
          enabled: true
          requirements:
            - "No misleading claims"
            - "Substantiation for all claims"
            - "No undue pressure (harder sell than US allowed)"

        price_claims:
          enabled: true
          requirements:
            - "Was price must be genuine previous price"
            - "No fake sales"
            - "Clear total cost (inc. VAT)"

        health_claims:
          enabled: true
          requirements:
            - "No miracle cure claims"
            - "Approved health claims only"
            - "No implying medical endorsement"

  # Germany
  DE:
    region_name: "Germany"
    cultural_context: "western_germanic"
    primary_language: "de"
    legal_framework: "UWG, Landesmedienanstalt"

    religious_considerations:
      christianity:
        sensitivity: "medium"
        major_holidays:
          - name: "Christmas"
            guidance: "Religious imagery acceptable"
            date_range: "Dec_1_-_Dec_26"
            allowed:
              - "religious Christmas scenes"
              - "Advent imagery"

    cultural_sensitivities:
      historical_sensitivity:
        sensitivity_level: "critical"
        avoid:
          - "Nazi imagery, symbols, references (ILLEGAL)"
          - "Hitler moustache styling"
          - "military imagery resembling historical"
          - "nationalistic imagery"
        guidance: "Extreme legal restrictions on Nazi/hate symbols"

      data_privacy:
        sensitivity_level: "high"
        avoid:
          - "tracking pixels disclosure without consent"
          - "data collection without clear notice"
        guidance: "GDPR strictly enforced"

    visual_restrictions:
      symbols:
        strictly_forbidden:
          - "swastika (ILLEGAL)"
          - "SS runes, other Nazi symbols (ILLEGAL)"
          - "Hitler imagery (ILLEGAL)"
          - "communist symbols in certain contexts"

        avoid:
          - "excessive nationalism"
          - "military imagery"

    text_restrictions:
      language_specific:
        de:
          formality: "Formal 'Sie' preferred over informal 'Du'"
          forbidden_phrases:
            - "hyperbole like 'weltbest' (world's best) without proof"
            - "kostenlos (free) with hidden costs"

      superlatives:
        regulation: "strict"
        forbidden:
          - "best, fastest, cheapest without objective proof"
          - "number 1 without current market data"
        allowed:
          - "one of the best (with source)"
          - "leading (with evidence)"

    compliance_requirements:
      uwg_unfair_competition_act:
        enabled: true
        requirements:
          - "No misleading advertising"
          - "No aggressive sales tactics (stricter than US)"
          - "Clear pricing including all fees"
          - "No 'bait and switch'"

      gesetze_gesetz_gegen_unlauteren_wettbewerb:
        impressum_required: true
        data_protection: "GDPR compliance required"

    color_associations:
      positive:
        - color: "blue"
          meanings: ["trust", "quality", "engineering"]

        - color: "silver/grey"
          meanings: ["technology", "precision", "german_engineering"]

      caution:
        - color: "brown"
          meanings: ["can remind of historical period, avoid"]

  # Japan
  JP:
    region_name: "Japan"
    cultural_context: "east_asian_high_context"
    primary_language: "ja"
    legal_framework: "JFTC Guidelines"

    religious_considerations:
      shinto_buddhism:
        sensitivity: "high"
        major_holidays:
          - name: "Oshogatsu (New Year)"
            date_range: "Jan_1_-_Jan_3"
            guidance: "Respectful traditional imagery"
            forbidden:
              - "disrespectful treatment of shrines, temples"
            allowed:
              - "traditional Japanese New Year imagery"

          - name: "Obon (Festival of Dead)"
            date_range: "august_13-16"
            guidance: "Respectful acknowledgment, avoid commercial exploitation"
            forbidden:
              - "commercial use of religious imagery"

    cultural_sensitivities:
      honorifics_etiquette:
        sensitivity_level: "high"
        avoid:
          - "disrespectful body language (slouching, pointing)"
          - "showing bottom of shoes"
          - "touching someone's head (if people depicted)"
        guidance: "Respectful, formal presentation preferred"

      yakuza_organized_crime:
        sensitivity_level: "high"
        avoid:
          - "tattoos (associated with yakuza)"
          - "missing fingers (yubitsume)"
          - "yakuza-style imagery"

      historical_sensitivity:
        sensitivity_level: "medium"
        avoid:
          - "WWII imagery"
          - "nationalistic symbolism"
          - "controversial war memorials"

    visual_restrictions:
      gestures:
        avoid:
          - gesture: "beckoning_with_palm_up"
            meaning: "dog calling gesture, rude"
            alternative: "beckon with palm down"

          - gesture: "pointing_with_one_finger"
            meaning: "rude, aggressive"
            alternative: "gesture with entire hand"

          - gesture: "thumbs_up"
            meaning: "can mean 'boyfriend' or 'lover', context-dependent"
            severity: "medium"

          - gesture: "peace_sign (V_sign)"
            meaning: "common in photos, generally acceptable"
            note: "very common, not offensive"

      symbols:
        avoid:
          - "rising sun flag (controversial in Asia)"
          - "yakuza-associated imagery"
          - "tattoos (associated with criminals)"

    text_restrictions:
      language_specific:
        ja:
          text_density: "Minimal text preferred (Japanese aesthetic)"
          max_text_overlay_percent: 10

          formality: "Polite/formal language (keigo) preferred"
          pronouns: "Avoid first-person pronouns (watashi/boku) in ads"

          forbidden_phrases:
            - "death-related words (shinu, shinu - even in idioms)"
            - "four (shi - sounds like death)"
            - "direct refusal language (indirect preferred)"

          number_taboos:
            - number: 4
              reason: "sounds like 'death' (shi)"
              avoid_in: "prices, quantities, phone numbers"

            - number: 9
              reason: "sounds like 'suffering' (ku)"
              avoid_in: "prices, quantities"

            - number: 49
              reason: "sounds like 'suffer until death'"
              avoid_in: "prices, quantities"

    color_associations:
      positive:
        - color: "red"
          meanings: ["luck", "happiness", "celebration"]
          usage: "very common in advertising"

        - color: "white"
          meanings: ["purity", "death (contextual)"]
          note: "White also used in funerals, context matters"

        - color: "gold"
          meanings: ["premium", "quality", "celebration"]

      negative:
        - color: "black"
          meanings: ["death", "misfortune"]
          usage: "use sparingly"

      caution:
        - color: "purple"
          meanings: ["can represent royalty or nobility"]
          note: "Positive association"

    compliance_requirements:
      jftc_guidelines:
        truth_in_advertising:
          enabled: true
          requirements:
            - "No misleading claims"
            - "Clear disclosure of paid endorsements"
            - "No fake reviews"

      premium_display_act:
        enabled: true
        requirements:
          - "Clear indication of premium-rate services"
          - "No hidden fees"

  # Saudi Arabia
  SA:
    region_name: "Saudi Arabia"
    cultural_context: "middle_eastern_islamic"
    primary_language: "ar"
    legal_framework: "General Commission for Audiovisual Media"

    religious_considerations:
      islam:
        sensitivity: "critical"
        guidance: "Islamic values and Sharia law govern all content"

        daily_prayer_times:
          respect_required: true
          guidance: "Avoid scheduling ads during prayer times when possible"

        ramadan:
          date_range: "islamic_month_9"
          guidance: "Respectful, spiritual tone preferred"
          allowed:
            - " Ramadan Kareem messaging"
            - "family, community imagery"
            - "dates, traditional foods"
          forbidden:
            - "eating, drinking imagery during fasting hours"
            - "revealing clothing"
            - "partying, music imagery"

        eid_al_fitr:
          guidance: "Celebratory but modest"
          allowed:
            - "Eid Mubarak messaging"
            - "family celebration imagery"
            - "gifts, new clothes"
          forbidden:
            - "immodest clothing"
            - "mixed-gender party imagery"

        friday:
          guidance: "Friday is holy day (day of worship)"
          forbidden:
            - "working, business imagery on Friday"

        quran_verses:
          usage: "strictly prohibited"
          reason: "religious text cannot be used commercially"

    cultural_sensitivities:
      gender_observance:
        sensitivity_level: "critical"
        guidance: "Gender segregation norms"

        women_in_imagery:
          allowed: true
          requirements:
            - "modest clothing (hijab preferred, abaya for traditional settings)"
            - "no revealing clothing"
            - "no physical contact with men in imagery"
            - "no solo women in suggestive poses"
          forbidden:
            - "women without hijab in traditional contexts"
            - "revealing or tight clothing"
            - "sexualized portrayal"

        men_in_imagery:
          requirements:
            - "modest clothing"
            - "no shorts above knee"
            - "no revealing tops"

      family_social_structure:
        sensitivity_level: "high"
        guidance: "Respect for elders and family honor"
        avoid:
          - "disrespectful depictions of elders"
          - "challenging parental authority"
          - "unmarried couples together"

    visual_restrictions:
      gestures:
        avoid:
          - gesture: "showing_sole_of_foot"
            meaning: "highly offensive, insulting"
            severity: "critical"

          - gesture: "using_left_hand"
            meaning: "considered unclean"
            guidance: "use right hand for eating, greeting, giving items"

          - gesture: "pointing_with_index_finger"
            meaning: "rude, accusatory"
            alternative: "use entire hand or thumb"

      symbols:
        strictly_forbidden:
          - "any depiction of Prophet Muhammad (PBUH) or any prophet"
          - "Allah, Muhammad in Arabic calligraphy for commercial use"
          - "religious imagery for commercial purposes"

        avoid:
          - "alcohol imagery (strictly prohibited)"
          - "pigs, pork imagery"
          - "dogs (considered unclean in some contexts)"
          - "gambling imagery"

    text_restrictions:
      language_specific:
        ar:
          direction: "right-to-left (RTL)"
          text_density: "Arabic script is dense, use minimal text"

          religious_phrases_forbidden:
            - "any verses from Quran"
            - "Hadith for commercial use"
            - "Islamic greetings in inappropriate contexts"

          religious_phrases_allowed:
            - phrase: "Alhamdulillah (Praise be to God)"
              context: "acceptable for thankfulness"

            - phrase: "InshaAllah (God willing)"
              context: "acceptable for future plans"

          forbidden_topics:
            - "alcohol, bars, drinking"
            - "gambling, casinos, betting"
            - "dating, non-marital relationships"
            - "pork products"
            - "anything anti-Islamic"

    color_associations:
      positive:
        - color: "green"
          meanings: ["Islam", "prosperity", "luck"]
          usage: "very common, positive"

        - color: "white"
          meanings: ["purity", "peace"]
          usage: "common"

        - color: "gold"
          meanings: ["premium", "quality", "prosperity"]

      negative:
        - color: "yellow"
          meanings: "can indicate betrayal, cowardice in some contexts"
          usage: "avoid or use sparingly"

      caution:
        - color: "red"
          meanings: ["danger", but also "celebration in some contexts"]
          note: "Context-dependent, use carefully"

    compliance_requirements:
      general_commission_audiovisual_media:
        content_standards:
          enabled: true
          requirements:
            - "Respect Islamic values and Sharia"
            - "No harmful content"
            - "No violation of public morals"
            - "No false or misleading content"

      advertising_regulations:
        products_requiring_special_approval:
          - "health products"
          - "financial services"
          - "educational institutions"
          - "real estate"

        forbidden_categories:
          - "alcohol"
          - "gambling"
          - "dating services"
          - "pork products"

  # India
  IN:
    region_name: "India"
    cultural_context: "south_asian_collectivist"
    primary_language: "en, hi, and 20+ others"
    legal_framework: "ASCI Guidelines, CCC"

    religious_considerations:
      multiple_faiths:
        sensitivity: "high"
        guidance: "Secular approach, respect all faiths"

        hinduism:
          major_holidays:
            - name: "Diwali"
              date_range: "oct_nov_variable"
              guidance: "Celebratory, festival of lights"
              allowed:
                - "diyas (lamps), rangoli, flowers"
                - "family celebration imagery"
                - "new beginnings, prosperity messaging"
              forbidden:
                - "religious figures for commercial gain"
                - "inaccurate religious symbolism"

            - name: "Holi"
              date_range: "march"
              guidance: "Festival of colors"
              allowed:
                - "color, celebration imagery"
              forbidden:
                - "inappropriate use of color on people"

          symbols:
            avoid:
              - "religious figures in commercial contexts"
              - "swastika (sacred symbol, but commercial use controversial)"
              - "om symbol for commercial purposes"

        islam:
          sensitivity: "high"
          guidance: "Respect Muslim minority"
          considerations:
            - "avoid beef imagery (cow sacred to Hindus, meat forbidden for Muslims)"
            - "respect Ramadan, Eid"
            - "modest clothing preferences"

        christianity:
          sensitivity: "medium"
          major_holidays:
            - name: "Christmas"
              guidance: "Celebrated, secular imagery preferred"
            - name: "Easter"
              guidance: "Celebrated by Christian minority"

        sikhism:
          sensitivity: "high"
          guidance: "Respect Sikh community"
          avoid:
            - "disrespectful treatment of turban, kirpan"
            - "misrepresentation of Sikh symbols"

    cultural_sensitivities:
      religious_diversity:
        sensitivity_level: "critical"
        guidance: "India is multi-religious, avoid favoring one faith"
        avoid:
          - "exclusive focus on one religion"
          - "religious controversy or conflict imagery"
          - "sacred animals in inappropriate contexts (cows, peacocks)"

      caste_sensitivity:
        sensitivity_level: "high"
        avoid:
          - "caste-based imagery or references"
          - "depicting caste hierarchies"
          - "discriminatory representations"

      linguistic_diversity:
        sensitivity_level: "medium"
        guidance: "India has 22 official languages"
        avoid:
          - "imposing one language as superior"
          - "making fun of accents or dialects"

    visual_restrictions:
      gestures:
        avoid:
          - gesture: "pointing_with_feet"
            meaning: "highly disrespectful (feet considered unclean)"
            severity: "high"

          - gesture: "showing_sole_of_foot"
            meaning: "offensive, especially toward people or religious objects"
            severity: "high"

          - gesture: "touching_someone_with_feet"
            meaning: "extremely offensive"
            severity: "critical"

          - gesture: "namaste_with_hands_together"
            meaning: "respectful greeting (acceptable and encouraged)"
            note: "positive gesture"

      symbols:
        avoid:
          - "religious symbols in commercial contexts"
          - "national emblem misuse (has legal restrictions)"
          - "map of India with disputed territories (legal issue)"

    text_restrictions:
      language_specific:
        en_IN:
          phrases:
            - "use British English spelling (colour, centre)"
            - "Indian English expressions acceptable"

        hi:
          formality: "Respectful, formal preferred"
          religious_references: "avoid commercial use of religious terms"

        universal:
          forbidden_topics:
            - "religious conversion"
            - "caste discrimination"
            - "regional conflicts"
            - "political controversies"

    color_associations:
      positive:
        - color: "saffron (orange)"
          meanings: ["purity", "renunciation", "Hinduism"]
          usage: "auspicious, common in religious contexts"

        - color: "red"
          meanings: ["prosperity", "fertility", "marriage"]
          usage: "very positive"

        - color: "yellow"
          meanings: ["commerce", "learning", "prosperity"]
          usage: "positive"

        - color: "green"
          meanings: ["Islam", "prosperity"]
          usage: "positive"

      negative:
        - color: "black"
          meanings: ["evil, negativity, inauspicious"]
          usage: "avoid, especially for celebrations"

      caution:
        - color: "white"
          meanings: ["purity but also mourning, widowhood"]
          note: "Context-dependent"

    compliance_requirements:
      asci_advertising_standards_council_of_india:
        code:
          enabled: true
          requirements:
            - "Truthful and honest representation"
            - "No offensive to public decency"
            - "No harmful to Indian values"
            - "Respect for all religions"

      legal_restrictions:
        products_requiring_disclaimer:
          - "health supplements: 'Not evaluated by FDA'"
          - "financial products: risk disclosures"
          - "real estate: RERA registration number"

        prohibited:
          - "surrogacy advertising"
          - "baby food for under 2 years (breast milk promotion)"
          - "tobacco and alcohol"

  # China (PRC)
  CN:
    region_name: "China (People's Republic)"
    cultural_context: "east_asian_collectivist_communist"
    primary_language: "zh"
    legal_framework: "Advertising Law of PRC"

    religious_considerations:
      state_atheism:
        guidance: "Officially secular state, religious content discouraged"
        sensitivity: "medium"

      buddhism_taoism:
        sensitivity: "low"
        guidance: "Traditional practices accepted"
        allowed:
          - "subtle cultural references"
        forbidden:
          - "overt religious messaging"
          - "superstitious content"

    cultural_sensitivities:
      political_sensitivity:
        sensitivity_level: "critical"
        avoid:
          - "Tibetan independence imagery"
          - "Taiwan as separate country"
          - "Tiananmen Square references"
          - "criticism of Communist Party"
          - "Hong Kong protest imagery"
          - "human rights issues"

        guidance: "Strict political censorship, avoid controversy"

      historical_sensitivity:
        sensitivity_level: "high"
        avoid:
          - "Cultural Revolution references"
          - "Japanese war crimes (very sensitive)"
          - "Mao criticism"

      national_identity:
        sensitivity_level: "high"
        guidance: "Patriotic imagery generally positive"
        allowed:
          - "national pride imagery"
          - "Chinese achievement"
        forbidden:
          - "anything seen as anti-China"

    visual_restrictions:
      gestures:
        avoid:
          - gesture: "pointing_with_one_finger"
            meaning: "rude"
            alternative: "use open hand"

          - gesture: "beckoning_with_index_finger"
            meaning: "rude, for dogs"
            alternative: "palm down, all fingers beckoning"

      symbols:
        strictly_forbidden:
          - "Taiwan independence symbols"
          - "Tibetan independence imagery"
          - "Falun Gong symbols (banned organization)"
          - "Tiananmen Square references"

        avoid:
          - "Japanese imperial symbols"
          - "anything questioning territorial integrity"

    text_restrictions:
      language_specific:
        zh:
          text_density: "Chinese characters dense, use minimal text"

          number_superstitions:
            - number: 4
              reason: "sounds like 'death' (si)"
              avoid_in: "prices, phone numbers, addresses"

            - number: 14
              reason: "sounds like 'surely die' (shi si)"
              avoid_in: "prices"

            - number: 8
              reason: "sounds like 'prosperity' (ba)"
              usage: "very positive, use for prices"

          forbidden_phrases:
            - "any criticism of government or Party"
            - "democracy advocacy language"
            - "human rights language"
            - "Taiwan as country (must say 'Taiwan region')"

          forbidden_topics:
            - "political reform"
            - "Tiananmen Square 1989"
            - "Taiwan independence"
            - "Tibet independence"
            - "Falun Gong"
            - "any anti-China sentiment"

    color_associations:
      positive:
        - color: "red"
          meanings: ["luck", "prosperity", "celebration", "Communist Party"]
          usage: "very positive, ubiquitous"

        - color: "gold"
          meanings: ["wealth", "prosperity", "premium"]

        - color: "yellow"
          meanings: ["imperial", "power", "historical royalty"]

      negative:
        - color: "white"
          meanings: ["death, mourning, funerals"]
          usage: "avoid, especially for celebrations"

        - color: "black"
          meanings: ["can be negative, but also modern/tech"]
          note: "Context-dependent"

      caution:
        - color: "green"
          meanings: ["can imply infidelity in relationships (wearing green hat)"]
          usage: "be careful with male imagery + green"

    compliance_requirements:
      advertising_law_prc:
        content_restrictions:
          enabled: true
          requirements:
            - "No harmful to social stability"
            - "No violation of national dignity"
            - "No superstition content"
            - "No pornographic or violent content"

        product_restrictions:
          require_special_approval:
            - "health products"
            - "cosmetics"
            - "infant formula"
            - "financial services"

        legal_disclaimers_required:
          - "medical products: contraindications, side effects"
          - "financial products: risk warnings"
          - "nutritional supplements: 'not medicine'"

      internet_content_regulation:
        great_firewall:
          guidance: "Content must comply with internet regulations"
          requirements:
            - "No blocked website references"
            - "No VPN promotion"
            - "No circumvention tools"

  # Brazil
  BR:
    region_name: "Brazil"
    cultural_context: "latin_american_brazilian"
    primary_language: "pt"
    legal_framework: "CONAR, Código de Defesa do Consumidor"

    religious_considerations:
      christianity_catholicism:
        sensitivity: "medium"
        major_holidays:
          - name: "Christmas"
            date_range: "Dec_1_-_Dec_25"
            guidance: "Religious imagery acceptable"
            allowed:
              - "Nativity scenes"
              - "religious Christmas imagery"

          - name: "Easter"
            date_range: "mar_apr_variable"
            guidance: "Religious imagery acceptable"
            allowed:
              - "crucifix, religious Easter imagery"

      african_religions:
        sensitivity: "high"
        guidance: "Respect Candomblé, Umbanda traditions"
        avoid:
          - "disrespectful portrayal of African religious symbols"
          - "stereotypical imagery"

    cultural_sensitivities:
      race_ethnicity:
        sensitivity_level: "high"
        guidance: "Brazil has racial diversity but also racism"
        avoid:
          - "racial stereotypes, caricatures"
          - "blackface (very offensive)"
          - "portraying all Brazilians as mixed race"
        guidance: "Authentic representation of diversity"

      economic_sensitivity:
        sensitivity_level: "medium"
        avoid:
          - "classist imagery (favelas for poverty porn)"
          - "mocking poverty"
          - "excessive wealth display"

      carnaval:
        guidance: "Cultural celebration"
        allowed:
          - "Carnival imagery, celebration"
        caution:
          - "sexualization during Carnival season"

    visual_restrictions:
      gestures:
        avoid:
          - gesture: "OK_sign (thumb+index finger forming circle)"
            meaning: "highly offensive (equivalent to middle finger in US)"
            severity: "critical"
            note: "THIS IS CRITICAL: OK sign = obscene in Brazil"

          - gesture: "thumbs_up"
            meaning: "generally positive, acceptable"

          - gesture: "middle_finger"
            meaning: "offensive, as in US"
            severity: "critical"

      symbols:
        avoid:
          - "discriminatory symbols"
          - "racist imagery"

    text_restrictions:
      language_specific:
        pt_BR:
          formality: "Brazilian Portuguese informal friendly tone common"

          forbidden_phrases:
            - "racial slurs (very offensive)"

          discouraged:
            - "overly formal language (cold)"

      legal_disclaimers_required:
        - trigger: "weight_loss"
          disclaimer: "Results vary. Exercise and diet required."

        - trigger: "financial_returns"
          disclaimer: "Past performance doesn't guarantee future results."

    color_associations:
      positive:
        - color: "green"
          meanings: ["hope", "Brazil", "nature"]
          usage: "very positive (national color)"

        - color: "yellow"
          meanings: ["wealth", "Brazil", "celebration"]
          usage: "very positive (national color)"

        - color: "blue"
          meanings: ["Brazil", "sky"]
          usage: "positive (national color)"

      note: "Green, yellow, blue are Brazilian flag colors (all positive)"

    compliance_requirements:
      conar_conselho_nacional_autopropaganda:
        code:
          enabled: true
          requirements:
            - "Truthful advertising"
            - "No misleading claims"
            - "No offense to public decency"
            - "Respect for consumer rights"

      consumer_protection_code:
        enabled: true
        requirements:
          - "Clear product information"
            - "No abusive practices"
            - "No unfair contract terms"

  # United Arab Emirates (Dubai/International context)
  AE:
    region_name: "United Arab Emirates"
    cultural_context: "middle_eastern_islamic_international"
    primary_language: "en, ar"
    legal_framework: "National Media Council"

    # Note: UAE is more liberal than KSA but still Islamic
    # Similar to SA rules but with more flexibility for international audience

    religious_considerations:
      islam:
        sensitivity: "high"
        guidance: "Islamic values govern, but more international than KSA"

        ramadan:
          date_range: "islamic_month_9"
          guidance: "Respectful, moderate tone"
          allowed:
            - "Ramadan Kareem messaging"
            - "ifrar imagery (breaking fast)"
            - "modest celebration"
          forbidden:
            - "eating, drinking in daylight hours during Ramadan"

    cultural_sensitivities:
      international_tolerance:
        sensitivity_level: "medium"
        guidance: "UAE is international hub, more tolerant than KSA"
        allowed:
          - "Western clothing in appropriate contexts"
          - "mixed-gender imagery (conservative)"
        forbidden:
          - "revealing clothing"
          - "sexualized content"

    visual_restrictions:
      gestures:
        avoid:
          - gesture: "showing_sole_of_foot"
            meaning: "offensive"
            severity: "high"

          - gesture: "using_left_hand_to_eat_give"
            meaning: "unclean"
            severity: "medium"

    text_restrictions:
      language_specific:
        ar:
          direction: "right-to-left"
          religious_usage: "limited, respectful only"

        en:
          international_style: "Western English acceptable"

      forbidden_topics:
        - "alcohol (except in licensed venues context)"
        - "gambling"
        - "dating services"
        - "pork products"

    color_associations:
      positive:
        - color: "green"
          meanings: ["Islam", "prosperity", "UAE"]

        - color: "white"
          meanings: ["purity", "peace"]

    compliance_requirements:
      national_media_council:
        enabled: true
        requirements:
          - "Respect Islamic values"
          - "No harm to UAE reputation"
          - "No offensive content"
```

---

## Implementation: Risk Detection Engine

```python
# src/meta/ad/reviewer/criteria/locale_risk_checker.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re

class RiskSeverity(Enum):
    CRITICAL = "critical"    # Instant rejection, log as violation
    HIGH = "high"           # Strong warning, requires review
    MEDIUM = "medium"       # Warning, flag for consideration
    LOW = "low"             # Informational, track metrics

@dataclass
class LocaleRiskCheck:
    """Result of a locale-specific risk check."""
    locale: str
    risk_category: str
    severity: RiskSeverity
    detected: bool
    details: str
    recommendation: str
    confidence: float  # 0-1

@dataclass
class LocaleRiskAssessment:
    """Complete locale-specific risk assessment."""
    locale: str
    overall_risk_level: RiskSeverity
    checks: List[LocaleRiskCheck]
    approved: bool
    blockers: List[str]  # Critical issues that block approval

class LocaleRiskChecker:
    """Checks locale-specific cultural and content risks."""

    def __init__(self, locale_risk_config: Dict):
        """Initialize with locale risk configuration."""
        self.config = locale_risk_config
        self.global_prohibitions = locale_risk_config.get('global_baseline', {})

    def check_locale_risks(
        self,
        locale: str,
        image_analysis: Dict,
        text_content: str,
        detected_gestures: List[str],
        detected_symbols: List[str],
        detected_colors: List[str]
    ) -> LocaleRiskAssessment:
        """
        Perform comprehensive locale-specific risk check.

        Args:
            locale: Locale code (e.g., 'US', 'JP', 'SA')
            image_analysis: GPT-4 Vision analysis results
            text_content: OCR-extracted text content
            detected_gestures: List of detected hand gestures
            detected_symbols: List of detected symbols
            detected_colors: List of dominant colors

        Returns:
            LocaleRiskAssessment with all risk checks
        """
        checks = []
        blockers = []

        # 1. Global baseline checks (apply to all locales)
        universal_checks = self._check_universal_prohibitions(
            image_analysis, text_content
        )
        checks.extend(universal_checks)

        # 2. Locale-specific checks if available
        if locale in self.config:
            locale_config = self.config[locale]

            # Religious sensitivities
            religious_checks = self._check_reigious_sensitivities(
                locale, locale_config, image_analysis
            )
            checks.extend(religious_checks)

            # Cultural sensitivities
            cultural_checks = self._check_cultural_sensitivities(
                locale, locale_config, image_analysis
            )
            checks.extend(cultural_checks)

            # Visual/gesture restrictions
            visual_checks = self._check_visual_restrictions(
                locale, locale_config, detected_gestures, detected_symbols
            )
            checks.extend(visual_checks)

            # Text overlay restrictions
            text_checks = self._check_text_restrictions(
                locale, locale_config, text_content
            )
            checks.extend(text_checks)

            # Color associations
            color_checks = self._check_color_associations(
                locale, locale_config, detected_colors
            )
            checks.extend(color_checks)

        # 3. Determine overall risk
        critical_count = sum(1 for c in checks if c.severity == RiskSeverity.CRITICAL and c.detected)
        high_count = sum(1 for c in checks if c.severity == RiskSeverity.HIGH and c.detected)

        if critical_count > 0:
            overall_risk = RiskSeverity.CRITICAL
            blockers = [c.details for c in checks if c.severity == RiskSeverity.CRITICAL and c.detected]
        elif high_count >= 3:
            overall_risk = RiskSeverity.HIGH
        elif high_count > 0:
            overall_risk = RiskSeverity.HIGH
        else:
            overall_risk = RiskSeverity.MEDIUM

        return LocaleRiskAssessment(
            locale=locale,
            overall_risk_level=overall_risk,
            checks=checks,
            approved=critical_count == 0,
            blockers=blockers
        )

    def _check_universal_prohibitions(
        self,
        image_analysis: Dict,
        text_content: str
    ) -> List[LocaleRiskCheck]:
        """Check universal prohibitions (violence, sexual content, etc.)."""
        checks = []
        prohibitions = self.global_prohibitions.get('universal_prohibitions', {})

        content_types = prohibitions.get('content_types', [])

        for content_type in content_types:
            category = content_type['category']
            severity = RiskSeverity(content_type['severity'])
            examples = content_type.get('examples', [])

            # Check image analysis
            detected = False
            details = ""

            if category == "violence":
                visual_elements = image_analysis.get('visual_elements', [])
                detected_elements = [el for el in visual_elements if el in examples]
                if detected_elements:
                    detected = True
                    details = f"Violent content detected: {', '.join(detected_elements)}"

            elif category == "sexual_content":
                visual_elements = image_analysis.get('visual_elements', [])
                detected_elements = [el for el in visual_elements if el in examples]
                if detected_elements:
                    detected = True
                    details = f"Sexual content detected: {', '.join(detected_elements)}"

            elif category == "hate_symbols":
                symbols = image_analysis.get('symbols', [])
                detected_symbols = [s for s in symbols if any(e in s.lower() for e in examples)]
                if detected_symbols:
                    detected = True
                    details = f"Hate symbols detected: {', '.join(detected_symbols)}"

            elif category == "fraudulent_appearances":
                # Check text for fraudulent patterns
                text_risks = self.global_prohibitions.get('text_overlay_risks', {})
                misleading = text_risks.get('universal_prohibitions', [])
                for risk_category in misleading:
                    if risk_category['type'] == 'misleading_claims':
                        patterns = risk_category.get('patterns', [])
                        for pattern in patterns:
                            if re.search(pattern, text_content, re.IGNORECASE):
                                detected = True
                                details = f"Fraudulent appearance: {pattern}"
                                break

            if detected:
                checks.append(LocaleRiskCheck(
                    locale='GLOBAL',
                    risk_category=category,
                    severity=severity,
                    detected=True,
                    details=details,
                    recommendation=f"Remove {category} content immediately",
                    confidence=0.9
                ))

        return checks

    def _check_visual_restrictions(
        self,
        locale: str,
        locale_config: Dict,
        detected_gestures: List[str],
        detected_symbols: List[str]
    ) -> List[LocaleRiskCheck]:
        """Check locale-specific visual restrictions (gestures, symbols)."""
        checks = []
        visual = locale_config.get('visual_restrictions', {})

        # Check gestures
        gestures = visual.get('gestures', {}).get('avoid', [])
        for gesture_spec in gestures:
            gesture = gesture_spec['gesture']
            if gesture in detected_gestures:
                checks.append(LocaleRiskCheck(
                    locale=locale,
                    risk_category="prohibited_gesture",
                    severity=RiskSeverity(gesture_spec.get('severity', 'high')),
                    detected=True,
                    details=gesture_spec['meaning'],
                    recommendation=gesture_spec.get('alternative', 'Remove gesture'),
                    confidence=0.95
                ))

        # Check symbols
        symbols_avoid = visual.get('symbols', {}).get('avoid', [])
        symbols_forbidden = visual.get('symbols', {}).get('strictly_forbidden', [])

        all_forbidden = symbols_avoid + symbols_forbidden

        for symbol in detected_symbols:
            for forbidden in all_forbidden:
                if forbidden.lower() in symbol.lower():
                    severity = RiskSeverity.CRITICAL if forbidden in symbols_forbidden else RiskSeverity.HIGH
                    checks.append(LocaleRiskCheck(
                        locale=locale,
                        risk_category="prohibited_symbol",
                        severity=severity,
                        detected=True,
                        details=f"Prohibited symbol detected: {symbol}",
                        recommendation="Remove symbol immediately",
                        confidence=0.9
                    ))

        return checks

    def _check_text_restrictions(
        self,
        locale: str,
        locale_config: Dict,
        text_content: str
    ) -> List[LocaleRiskCheck]:
        """Check locale-specific text restrictions."""
        checks = []
        text_restrictions = locale_config.get('text_restrictions', {})

        # Check language-specific restrictions
        language_specific = text_restrictions.get('language_specific', {})

        for lang, lang_config in language_specific.items():
            # Check forbidden phrases
            forbidden_phrases = lang_config.get('forbidden_phrases', [])
            for phrase in forbidden_phrases:
                if phrase.lower() in text_content.lower():
                    checks.append(LocaleRiskCheck(
                        locale=locale,
                        risk_category="forbidden_phrase",
                        severity=RiskSeverity.HIGH,
                        detected=True,
                        details=f"Forbidden phrase detected: {phrase}",
                        recommendation="Remove or replace phrase",
                        confidence=0.95
                    ))

            # Check forbidden patterns (superstitions, etc.)
            forbidden_patterns = lang_config.get('forbidden_patterns', [])
            for pattern_info in forbidden_patterns:
                pattern = pattern_info['pattern']
                reason = pattern_info['reason']
                if re.search(pattern, text_content, re.IGNORECASE):
                    checks.append(LocaleRiskCheck(
                        locale=locale,
                        risk_category="cultural_taboo",
                        severity=RiskSeverity.MEDIUM,
                        detected=True,
                        details=f"Taboo detected: {reason}",
                        recommendation=pattern_info.get('alternative', 'Remove'),
                        confidence=0.85
                    ))

        return checks

    def _check_color_associations(
        self,
        locale: str,
        locale_config: Dict,
        detected_colors: List[str]
    ) -> List[LocaleRiskCheck]:
        """Check color associations for locale."""
        checks = []
        color_assoc = locale_config.get('color_associations', {})

        # Check negative color associations
        negative_colors = color_assoc.get('negative', [])
        for color_spec in negative_colors:
            color = color_spec['color']
            if color in detected_colors:
                checks.append(LocaleRiskCheck(
                    locale=locale,
                    risk_category="negative_color_association",
                    severity=RiskSeverity.LOW,
                    detected=True,
                    details=f"Color {color} has negative associations: {color_spec['meanings']}",
                    recommendation=color_spec.get('usage', 'Consider avoiding'),
                    confidence=0.7
                ))

        return checks

    def _check_religious_sensitivities(
        self,
        locale: str,
        locale_config: Dict,
        image_analysis: Dict
    ) -> List[LocaleRiskCheck]:
        """Check religious sensitivities for locale."""
        checks = []
        religious = locale_config.get('religious_considerations', {})

        # This would use GPT-4 Vision to detect religious imagery
        # and cross-reference with locale-specific restrictions

        # Implementation would depend on image analysis capabilities

        return checks
```

---

## Integration with Brand Guidelines

```yaml
# Add to main brand guidelines YAML

locale_risk_checks:
  enabled: true
  config_path: "config/ad/reviewer/locale_risks.yaml"
  default_locale: "US"
  target_locales: ["US", "GB", "DE", "JP", "SA", "IN", "CN", "BR"]

  # Fallback for locales not in config
  unknown_locale_handling: "strict"  # strict | permissive | warn_only
```

---

## Summary

This comprehensive locale-specific risk check specification covers:

1. **Universal Prohibitions** - Content never allowed (violence, sexual content, hate symbols, fraud)
2. **Religious Sensitivities** - Holiday guidance, religious imagery rules, sacred symbols
3. **Cultural Sensitivities** - Race, gender, class, historical sensitivities
4. **Visual Restrictions** - Prohibited gestures, symbols, imagery
5. **Text Restrictions** - Forbidden phrases, language-specific taboos, superstitions
6. **Color Associations** - Positive/negative meanings by culture
7. **Compliance Requirements** - Local advertising regulations

The system is designed to:
- **Prevent cultural offenses** before ads go live
- **Flag religious misfits** automatically
- **Detect fraudulent appearances** (fake urgency, misleading claims)
- **Enforce local laws** (FTC, ASA, UWG, etc.)
- **Provide actionable recommendations** for each violation
