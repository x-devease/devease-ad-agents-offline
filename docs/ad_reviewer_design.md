# Ad Reviewer Design Document

## Overview

The **Ad Reviewer** is a meta component responsible for comprehensively evaluating ad creatives across three critical dimensions:

1. **Brand Guidelines Compliance** - Ensures ads adhere to brand identity, visual style, and logo specifications
2. **Culture Fit & Risk Assessment** - Evaluates cultural appropriateness and identifies potential reputational risks
3. **Miner & Generator Evaluation** - Assesses the quality and effectiveness of recommendations and generated outputs

This component acts as a quality gate in the creative pipeline, validating outputs from the ad-generator and providing feedback to improve both ad-miner recommendations and ad-generator prompts.

---

## Architecture

### Directory Structure

```
src/meta/ad/reviewer/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── reviewer.py                # Main AdReviewer class
│   ├── paths.py                   # Path management
│   └── constants.py               # Review criteria and scoring constants
├── criteria/
│   ├── __init__.py
│   ├── brand_guidelines.py        # Brand guideline checking logic
│   ├── culture_fit.py             # Culture fit assessment logic
│   ├── technical_quality.py       # Technical quality checks (resolution, clarity, etc.)
│   └── compliance.py              # Meta/Facebook ad compliance rules
├── analyzers/
│   ├── __init__.py
│   ├── image_analyzer.py          # Image analysis (GPT-4 Vision integration)
│   ├── text_analyzer.py           # Text overlay detection and analysis
│   ├── color_analyzer.py          # Color palette extraction and verification
│   └── brand_analyzer.py          # Brand identity verification
├── evaluators/
│   ├── __init__.py
│   ├── miner_evaluator.py         # Evaluate ad-miner recommendations
│   ├── generator_evaluator.py    # Evaluate ad-generator outputs
│   └── feedback_aggregator.py    # Aggregate feedback for miner/generator
├── pipeline/
│   ├── __init__.py
│   ├── pipeline.py                # ReviewPipeline orchestrator
│   └── report_generator.py        # Generate review reports (JSON/MD/HTML)
├── models/
│   ├── __init__.py
│   ├── review_result.py           # ReviewResult dataclass
│   ├── criteria_config.py         # CriteriaConfig dataclass
│   └── score_breakdown.py         # ScoreBreakdown dataclass
└── utils/
    ├── __init__.py
    ├── scoring.py                 # Scoring utilities
    ├── vision_api.py              # GPT-4 Vision API wrapper
    └── formatters.py              # Output formatters
```

### Configuration & Paths

```python
@dataclass
class ReviewPipelineConfig:
    """Configuration for the review pipeline."""
    customer: str
    platform: str = "meta"
    product_context: Optional[Dict[str, Any]] = None
    review_criteria: Optional[CriteriaConfig] = None
    output_format: str = "json"  # json, markdown, html
    strict_mode: bool = False    # Fail on critical violations
    enable_miner_eval: bool = True
    enable_generator_eval: bool = True

@dataclass
class CriteriaConfig:
    """Review criteria configuration."""
    brand_weight: float = 0.4        # 40% weight for brand compliance
    culture_weight: float = 0.3      # 30% weight for culture fit
    technical_weight: float = 0.2    # 20% weight for technical quality
    compliance_weight: float = 0.1   # 10% weight for compliance

    # Brand thresholds
    logo_compliance_required: bool = True
    color_tolerance_delta_e: float = 5.0  # Color difference threshold

    # Culture fit thresholds
    min_culture_score: float = 70.0
    risk_categories: List[str] = field(default_factory=lambda: [
        "violence", "hate_speech", "sexual_content",
        "political", "religious", "controversial"
    ])

    # Technical quality thresholds
    min_resolution: Tuple[int, int] = (1080, 1080)
    min_text_contrast_ratio: float = 4.5  # WCAG AA
    max_blur_score: float = 0.3
```

**Path Organization:**

```
config/ad/reviewer/
└── {customer}/
    └── {platform}/
        ├── criteria.yaml                # Review criteria overrides
        ├── brand_guidelines.yaml        # Brand-specific rules
        └── culture_context.yaml         # Cultural context and sensitivities

results/ad/reviewer/
└── {customer}/
    └── {platform}/
        └── {date}/
            ├── reviews/                 # Individual ad reviews
            │   ├── {image_id}.json
            │   └── {image_id}.md
            ├── batch_reports/           # Batch review summaries
            │   └── {batch_id}_summary.md
            └── feedback/                # Feedback for miner/generator
                ├── miner_feedback.json
                └── generator_feedback.json
```

---

## Core Components

### 1. Main Reviewer Class (`core/reviewer.py`)

```python
class AdReviewer:
    """Main ad reviewer orchestrator."""

    def __init__(self, config: ReviewPipelineConfig):
        self.config = config
        self.paths = ReviewerPaths(
            customer=config.customer,
            platform=config.platform
        )

        # Load brand guidelines (reuse from generator)
        self.brand_guidelines = get_brand_guidelines(config.customer)

        # Initialize analyzers
        self.image_analyzer = ImageAnalyzer()
        self.text_analyzer = TextAnalyzer()
        self.color_analyzer = ColorAnalyzer()
        self.brand_analyzer = BrandAnalyzer(self.brand_guidelines)

        # Initialize evaluators
        self.miner_evaluator = MinerEvaluator() if config.enable_miner_eval else None
        self.generator_evaluator = GeneratorEvaluator() if config.enable_generator_eval else None

        # Load criteria configuration
        self.criteria = self._load_criteria()

    def review_creative(
        self,
        image_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        recommendation_data: Optional[Dict] = None,
        generation_data: Optional[Dict] = None
    ) -> ReviewResult:
        """
        Review a single creative.

        Args:
            image_path: Path to the generated creative image
            metadata: Optional metadata (product context, platform, etc.)
            recommendation_data: Optional data from ad-miner recommendations
            generation_data: Optional data from ad-generator (prompts used, etc.)

        Returns:
            ReviewResult with comprehensive analysis
        """
        # 1. Analyze image using GPT-4 Vision
        image_analysis = self.image_analyzer.analyze(image_path)

        # 2. Check brand guidelines compliance
        brand_compliance = self.brand_analyzer.check_compliance(
            image_analysis, image_path
        )

        # 3. Assess culture fit and risks
        culture_assessment = self._assess_culture_fit(image_analysis)

        # 4. Check technical quality
        technical_quality = self._check_technical_quality(
            image_path, image_analysis
        )

        # 5. Check Meta compliance
        compliance_status = self._check_compliance(image_analysis)

        # 6. Evaluate miner recommendations (if data provided)
        miner_evaluation = None
        if self.miner_evaluator and recommendation_data:
            miner_evaluation = self.miner_evaluator.evaluate(
                image_path, image_analysis, recommendation_data
            )

        # 7. Evaluate generator output (if data provided)
        generator_evaluation = None
        if self.generator_evaluator and generation_data:
            generator_evaluation = self.generator_evaluator.evaluate(
                image_path, image_analysis, generation_data
            )

        # 8. Calculate overall score
        overall_score = self._calculate_overall_score({
            'brand': brand_compliance.score,
            'culture': culture_assessment.score,
            'technical': technical_quality.score,
            'compliance': compliance_status.score
        })

        # 9. Generate recommendations
        recommendations = self._generate_recommendations({
            'brand': brand_compliance,
            'culture': culture_assessment,
            'technical': technical_quality,
            'compliance': compliance_status,
            'miner': miner_evaluation,
            'generator': generator_evaluation
        })

        return ReviewResult(
            image_path=image_path,
            overall_score=overall_score,
            brand_compliance=brand_compliance,
            culture_fit=culture_assessment,
            technical_quality=technical_quality,
            compliance_status=compliance_status,
            miner_evaluation=miner_evaluation,
            generator_evaluation=generator_evaluation,
            recommendations=recommendations,
            metadata=metadata or {},
            reviewed_at=datetime.now().isoformat()
        )
```

### 2. Brand Guidelines Review (`criteria/brand_guidelines.py`)

```python
class BrandGuidelinesChecker:
    """Checks compliance with brand guidelines."""

    def __init__(self, brand_guidelines: Dict[str, Any]):
        self.guidelines = brand_guidelines

    def check_compliance(
        self,
        image_analysis: ImageAnalysis,
        image_path: str
    ) -> BrandComplianceResult:
        """
        Check brand guideline compliance.

        Returns:
            BrandComplianceResult with detailed violations and scores
        """
        violations = []
        checks = {}

        # 1. Color compliance
        color_check = self._check_colors(image_analysis)
        checks['colors'] = color_check
        if not color_check['compliant']:
            violations.append({
                'category': 'colors',
                'severity': 'high' if color_check['delta_e'] > 10 else 'medium',
                'issue': f"Color deviation ΔE={color_check['delta_e']:.1f}",
                'detected': color_check['detected_colors'],
                'expected': self.guidelines.get('primary_colors', [])
            })

        # 2. Logo compliance
        logo_check = self._check_logo(image_analysis, image_path)
        checks['logo'] = logo_check
        if not logo_check['compliant']:
            violations.append({
                'category': 'logo',
                'severity': 'critical',
                'issue': logo_check['issue'],
                'details': logo_check
            })

        # 3. Typography compliance
        typography_check = self._check_typography(image_analysis)
        checks['typography'] = typography_check
        if not typography_check['compliant']:
            violations.append({
                'category': 'typography',
                'severity': 'medium',
                'issue': typography_check['issue'],
                'details': typography_check
            })

        # 4. Visual style compliance
        style_check = self._check_visual_style(image_analysis)
        checks['style'] = style_check

        # 5. Tone and messaging
        tone_check = self._check_tone(image_analysis)
        checks['tone'] = tone_check

        # Calculate overall brand score
        score = self._calculate_brand_score(checks, violations)

        return BrandComplianceResult(
            score=score,
            compliant=score >= 80.0,
            checks=checks,
            violations=violations,
            guideline_coverage=self._calculate_coverage(checks)
        )

    def _check_colors(self, image_analysis: ImageAnalysis) -> Dict:
        """Extract colors and verify against brand palette."""
        detected_colors = self.color_analyzer.extract_palette(image_analysis)
        brand_colors = self.guidelines.get('primary_colors', [])

        # Calculate Delta E for each detected color
        max_delta_e = 0
        for detected in detected_colors:
            for brand_color in brand_colors:
                delta_e = self._calculate_delta_e(detected, brand_color)
                max_delta_e = max(max_delta_e, delta_e)

        tolerance = self.criteria.color_tolerance_delta_e
        return {
            'compliant': max_delta_e <= tolerance,
            'delta_e': max_delta_e,
            'detected_colors': detected_colors,
            'brand_colors': brand_colors
        }

    def _check_logo(self, image_analysis: ImageAnalysis, image_path: str) -> Dict:
        """Verify logo presence, placement, size, and quality."""
        logo_specs = self.guidelines.get('logo_specification', {})

        # Use GPT-4 Vision to detect logo
        logo_detection = detect_logo(
            image_path,
            brand_name=self.guidelines.get('name')
        )

        if not logo_detection.get('present'):
            return {
                'compliant': False,
                'issue': 'Logo not detected in image',
                'detected': logo_detection
            }

        issues = []

        # Check placement
        expected_placement = logo_specs.get('placement', 'any')
        detected_placement = logo_detection.get('placement')
        if expected_placement != 'any' and detected_placement != expected_placement:
            issues.append(f"Logo placement: expected {expected_placement}, detected {detected_placement}")

        # Check size
        expected_size_range = logo_specs.get('size_percentage', {})
        detected_size = logo_detection.get('size_percentage')
        if detected_size:
            min_size, max_size = self._parse_size_range(expected_size_range)
            if not (min_size <= detected_size <= max_size):
                issues.append(f"Logo size: {detected_size:.1f}% outside range {min_size}-{max_size}%")

        # Check quality (sharpness, blur)
        if logo_detection.get('blur_score', 0) > 0.1:
            issues.append(f"Logo appears blurred (blur score: {logo_detection['blur_score']:.2f})")

        return {
            'compliant': len(issues) == 0,
            'issue': '; '.join(issues) if issues else None,
            'detected': logo_detection
        }

    def _check_typography(self, image_analysis: ImageAnalysis) -> Dict:
        """Check text readability, contrast, and brand font usage."""
        text_regions = image_analysis.get('text_regions', [])

        issues = []
        for region in text_regions:
            # Check contrast ratio
            contrast_ratio = region.get('contrast_ratio', 0)
            if contrast_ratio < 4.5:
                issues.append(f"Low contrast text (ratio: {contrast_ratio:.1f}, required: 4.5:1)")

            # Check font characteristics
            font_weight = region.get('font_weight')
            if font_weight and font_weight < 400:
                issues.append(f"Text too light (weight: {font_weight}, recommended: 700+)")

            # Check for text cut-off
            if region.get('is_cut_off', False):
                issues.append("Text appears cut off at image edge")

        return {
            'compliant': len(issues) == 0,
            'issue': '; '.join(issues) if issues else None,
            'text_regions': text_regions
        }

    def _calculate_delta_e(self, color1: str, color2: str) -> float:
        """Calculate CIE76 Delta E color difference."""
        # Convert hex to LAB and calculate Delta E
        rgb1 = self._hex_to_rgb(color1)
        rgb2 = self._hex_to_rgb(color2)
        lab1 = rgb_to_lab(rgb1)
        lab2 = rgb_to_lab(rgb2)
        return sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))
```

### 3. Culture Fit & Risk Assessment (`criteria/culture_fit.py`)

```python
class CultureFitAnalyzer:
    """Analyzes cultural fit and identifies potential risks."""

    def __init__(self, criteria: CriteriaConfig):
        self.criteria = criteria
        self.risk_categories = criteria.risk_categories

    def assess(
        self,
        image_analysis: ImageAnalysis,
        target_markets: Optional[List[str]] = None
    ) -> CultureFitResult:
        """
        Assess cultural fit and identify risks.

        Args:
            image_analysis: Image analysis results
            target_markets: List of target markets/countries

        Returns:
            CultureFitResult with score and risk flags
        """
        detected_risks = []
        cultural_contexts = []

        # 1. Analyze visual content for risky elements
        visual_risks = self._analyze_visual_risks(image_analysis)
        detected_risks.extend(visual_risks)

        # 2. Analyze text content for cultural sensitivity
        text_risks = self._analyze_text_risks(image_analysis)
        detected_risks.extend(text_risks)

        # 3. Check for market-specific cultural issues
        if target_markets:
            market_risks = self._check_market_context(
                image_analysis, target_markets
            )
            detected_risks.extend(market_risks)
            cultural_contexts = self._load_cultural_context(target_markets)

        # 4. Analyze symbols and gestures
        symbol_risks = self._analyze_symbols(image_analysis, target_markets)
        detected_risks.extend(symbol_risks)

        # 5. Check for diversity and representation issues
        diversity_analysis = self._analyze_diversity(image_analysis)

        # Calculate culture fit score
        score = self._calculate_culture_score(detected_risks, diversity_analysis)

        return CultureFitResult(
            score=score,
            compliant=score >= self.criteria.min_culture_score,
            detected_risks=detected_risks,
            cultural_contexts=cultural_contexts,
            diversity_analysis=diversity_analysis,
            market_specific_issues=self._group_by_market(detected_risks)
        )

    def _analyze_visual_risks(self, image_analysis: ImageAnalysis) -> List[Dict]:
        """Detect potentially harmful visual content."""
        risks = []
        visual_elements = image_analysis.get('visual_elements', [])

        # Check for violence/weapons
        if any(el in visual_elements for el in ['weapons', 'violence', 'blood']):
            risks.append({
                'category': 'violence',
                'severity': 'high',
                'detected_elements': [el for el in visual_elements if el in ['weapons', 'violence', 'blood']],
                'recommendation': 'Remove violent elements or weapons from creative'
            })

        # Check for sexual content
        if any(el in visual_elements for el in ['sexual', 'suggestive', 'explicit']):
            risks.append({
                'category': 'sexual_content',
                'severity': 'critical',
                'detected_elements': [el for el in visual_elements if el in ['sexual', 'suggestive', 'explicit']],
                'recommendation': 'Remove sexualized content - violates Meta policies'
            })

        # Check for hate symbols
        hate_symbols = self._check_hate_symbols(visual_elements)
        if hate_symbols:
            risks.append({
                'category': 'hate_speech',
                'severity': 'critical',
                'detected_symbols': hate_symbols,
                'recommendation': 'Remove hate symbols - violates Meta policies and legal requirements'
            })

        # Check for controversial political/religious content
        if any(el in visual_elements for el in ['political', 'religious', 'controversial']):
            risks.append({
                'category': 'political',
                'severity': 'medium',
                'detected_elements': [el for el in visual_elements if el in ['political', 'religious', 'controversial']],
                'recommendation': 'Consider depoliticizing creative to avoid alienating audiences'
            })

        return risks

    def _analyze_text_risks(self, image_analysis: ImageAnalysis) -> List[Dict]:
        """Analyze text content for cultural sensitivity issues."""
        risks = []
        text_content = image_analysis.get('text_content', '')

        if not text_content:
            return risks

        # Use GPT-4 to analyze text for cultural issues
        text_analysis = self._analyze_text_sensitivity(text_content)

        if text_analysis.get('offensive_language'):
            risks.append({
                'category': 'offensive_language',
                'severity': 'high',
                'detected_phrases': text_analysis['offensive_phrases'],
                'recommendation': 'Remove or rewrite offensive language'
            })

        if text_analysis.get('cultural_stereotypes'):
            risks.append({
                'category': 'cultural_stereotypes',
                'severity': 'medium',
                'detected_stereotypes': text_analysis['cultural_stereotypes'],
                'recommendation': 'Avoid cultural stereotypes - use inclusive language'
            })

        if text_analysis.get('religious_insensitivity'):
            risks.append({
                'category': 'religious',
                'severity': 'high',
                'issue': text_analysis['religious_insensitivity'],
                'recommendation': 'Review religious references for sensitivity'
            })

        return risks

    def _check_market_context(
        self,
        image_analysis: ImageAnalysis,
        markets: List[str]
    ) -> List[Dict]:
        """Check for market-specific cultural issues."""
        risks = []

        # Load cultural context rules for each market
        for market in markets:
            context = self._load_cultural_context(market)

            # Check for prohibited symbols/gestures
            prohibited = context.get('prohibited_symbols', [])
            detected_symbols = image_analysis.get('symbols', [])
            violations = [s for s in detected_symbols if s in prohibited]

            if violations:
                risks.append({
                    'category': 'cultural_inappropriateness',
                    'severity': 'high',
                    'market': market,
                    'detected_symbols': violations,
                    'cultural_context': context.get('explanation', ''),
                    'recommendation': f"Remove symbols inappropriate for {market} market"
                })

            # Check for color meanings
            color_meanings = context.get('color_associations', {})
            detected_colors = image_analysis.get('dominant_colors', [])
            for color in detected_colors:
                if color in color_meanings and color_meanings[color]['negative']:
                    risks.append({
                        'category': 'color_association',
                        'severity': 'medium',
                        'market': market,
                        'color': color,
                        'meaning': color_meanings[color],
                        'recommendation': f"Consider avoiding {color} in {market} market"
                    })

        return risks

    def _analyze_diversity(self, image_analysis: ImageAnalysis) -> Dict:
        """Analyze diversity and representation in the creative."""
        people = image_analysis.get('people_detected', [])

        if not people:
            return {
                'has_people': False,
                'note': 'No people detected - diversity not applicable'
            }

        # Analyze demographics
        demographics = {
            'gender': {},
            'age_range': {},
            'ethnicity': {},
            'count': len(people)
        }

        for person in people:
            for attr in ['gender', 'age_range', 'ethnicity']:
                value = person.get(attr, 'unknown')
                demographics[attr][value] = demographics[attr].get(value, 0) + 1

        # Check for balanced representation
        has_diverse_representation = (
            len(demographics.get('ethnicity', {})) > 1 or
            len(demographics.get('age_range', {})) > 1
        )

        return {
            'has_people': True,
            'demographics': demographics,
            'diverse_representation': has_diverse_representation,
            'recommendation': (
                'Good diverse representation' if has_diverse_representation
                else 'Consider including more diverse representation'
            )
        }

    def _calculate_culture_score(
        self,
        risks: List[Dict],
        diversity_analysis: Dict
    ) -> float:
        """Calculate overall culture fit score (0-100)."""
        base_score = 100.0

        # Deduct points for each risk based on severity
        severity_weights = {
            'critical': 50,
            'high': 25,
            'medium': 10,
            'low': 5
        }

        for risk in risks:
            severity = risk.get('severity', 'medium')
            base_score -= severity_weights.get(severity, 10)

        # Bonus for good diversity representation
        if diversity_analysis.get('diverse_representation'):
            base_score += 5

        return max(0.0, min(100.0, base_score))
```

### 4. Miner & Generator Evaluation (`evaluators/`)

#### Miner Evaluator (`evaluators/miner_evaluator.py`)

```python
class MinerEvaluator:
    """Evaluates the quality and effectiveness of ad-miner recommendations."""

    def __init__(self):
        self.quality_metrics = [
            'recommendation_relevance',
            'feature_accuracy',
            'pattern_strength',
            'actionability'
        ]

    def evaluate(
        self,
        image_path: str,
        image_analysis: ImageAnalysis,
        recommendation_data: Dict
    ) -> MinerEvaluation:
        """
        Evaluate how well the ad-miner recommendations translated to the final creative.

        Args:
            image_path: Path to the generated creative
            image_analysis: Analysis of the generated creative
            recommendation_data: Original recommendations from ad-miner

        Returns:
            MinerEvaluation with scores and feedback
        """
        # 1. Check if recommended features are present
        feature_adoption = self._check_feature_adoption(
            image_analysis, recommendation_data
        )

        # 2. Verify recommended patterns were followed
        pattern_adherence = self._check_pattern_adherence(
            image_analysis, recommendation_data
        )

        # 3. Assess quality of recommended elements
        quality_assessment = self._assess_recommended_quality(
            image_analysis, recommendation_data
        )

        # 4. Calculate overall miner score
        overall_score = self._calculate_miner_score({
            'feature_adoption': feature_adoption,
            'pattern_adherence': pattern_adherence,
            'quality': quality_assessment
        })

        # 5. Generate feedback for miner improvement
        feedback = self._generate_miner_feedback({
            'feature_adoption': feature_adoption,
            'pattern_adherence': pattern_adherence,
            'quality': quality_assessment,
            'recommendations': recommendation_data
        })

        return MinerEvaluation(
            overall_score=overall_score,
            feature_adoption=feature_adoption,
            pattern_adherence=pattern_adherence,
            quality_assessment=quality_assessment,
            feedback=feedback
        )

    def _check_feature_adoption(
        self,
        image_analysis: ImageAnalysis,
        recommendations: Dict
    ) -> Dict:
        """Check which recommended features are present in the creative."""
        recommended_features = recommendations.get('features', {})
        detected_features = image_analysis.get('features', {})

        adoption_stats = {
            'total_recommended': len(recommended_features),
            'adopted': 0,
            'partial': 0,
            'missing': 0,
            'details': []
        }

        for feature, spec in recommended_features.items():
            detected = detected_features.get(feature)

            if not detected:
                adoption_stats['missing'] += 1
                adoption_stats['details'].append({
                    'feature': feature,
                    'status': 'missing',
                    'recommended': spec
                })
            elif self._is_feature_match(spec, detected):
                adoption_stats['adopted'] += 1
                adoption_stats['details'].append({
                    'feature': feature,
                    'status': 'adopted',
                    'recommended': spec,
                    'detected': detected
                })
            else:
                adoption_stats['partial'] += 1
                adoption_stats['details'].append({
                    'feature': feature,
                    'status': 'partial',
                    'recommended': spec,
                    'detected': detected
                })

        adoption_rate = (
            adoption_stats['adopted'] + 0.5 * adoption_stats['partial']
        ) / adoption_stats['total_recommended'] if adoption_stats['total_recommended'] > 0 else 0

        return {
            'adoption_rate': adoption_rate,
            'stats': adoption_stats,
            'score': adoption_rate * 100
        }

    def _check_pattern_adherence(
        self,
        image_analysis: ImageAnalysis,
        recommendations: Dict
    ) -> Dict:
        """Check if recommended creative patterns were followed."""
        patterns = recommendations.get('patterns', {})
        detected_layout = image_analysis.get('layout', {})
        detected_style = image_analysis.get('style', {})

        adherence_scores = {}

        # Layout pattern adherence
        if 'layout' in patterns:
            recommended_layout = patterns['layout']
            layout_match = self._compare_layouts(recommended_layout, detected_layout)
            adherence_scores['layout'] = layout_match

        # Style pattern adherence
        if 'style' in patterns:
            recommended_style = patterns['style']
            style_match = self._compare_styles(recommended_style, detected_style)
            adherence_scores['style'] = style_match

        # Composition pattern adherence
        if 'composition' in patterns:
            recommended_composition = patterns['composition']
            composition_match = self._compare_compositions(
                recommended_composition,
                image_analysis
            )
            adherence_scores['composition'] = composition_match

        overall_adherence = (
            sum(adherence_scores.values()) / len(adherence_scores)
            if adherence_scores else 0
        )

        return {
            'overall_adherence': overall_adherence,
            'pattern_scores': adherence_scores,
            'score': overall_adherence * 100
        }

    def _generate_miner_feedback(self, evaluation_data: Dict) -> Dict:
        """Generate structured feedback for miner improvement."""
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'feature_adoption_feedback': [],
            'pattern_feedback': [],
            'quality_feedback': [],
            'actionable_insights': []
        }

        # Feature adoption feedback
        feature_details = evaluation_data['feature_adoption']['stats']['details']
        for detail in feature_details:
            if detail['status'] == 'missing':
                feedback['feature_adoption_feedback'].append({
                    'type': 'missing_feature',
                    'feature': detail['feature'],
                    'issue': f"Feature '{detail['feature']}' was recommended but not detected in generated creative",
                    'suggested_action': 'investigate why feature was not adopted - prompt issue or generation failure?'
                })
            elif detail['status'] == 'partial':
                feedback['feature_adoption_feedback'].append({
                    'type': 'partial_feature',
                    'feature': detail['feature'],
                    'issue': f"Feature '{detail['feature']}' partially adopted",
                    'recommended': detail['recommended'],
                    'detected': detail['detected'],
                    'suggested_action': 'refine feature specification for better clarity'
                })

        # Pattern feedback
        pattern_scores = evaluation_data['pattern_adherence']['pattern_scores']
        for pattern, score in pattern_scores.items():
            if score < 0.7:
                feedback['pattern_feedback'].append({
                    'pattern': pattern,
                    'score': score,
                    'issue': f"Low adherence to {pattern} pattern",
                    'suggested_action': f'review {pattern} pattern recommendations and adjust weightings'
                })

        # Aggregate insights
        if feedback['feature_adoption_feedback']:
            feedback['actionable_insights'].append({
                'category': 'feature_recommendation_quality',
                'insight': 'Low feature adoption rate suggests prompt clarity issues',
                'action': 'Improve feature descriptions in recommendations for better generator adherence'
            })

        return feedback
```

#### Generator Evaluator (`evaluators/generator_evaluator.py`)

```python
class GeneratorEvaluator:
    """Evaluates the quality of ad-generator outputs."""

    def __init__(self):
        self.quality_dimensions = [
            'prompt_fidelity',
            'visual_quality',
            'brand_consistency',
            'technical_execution'
        ]

    def evaluate(
        self,
        image_path: str,
        image_analysis: ImageAnalysis,
        generation_data: Dict
    ) -> GeneratorEvaluation:
        """
        Evaluate the quality of the generated creative.

        Args:
            image_path: Path to the generated creative
            image_analysis: Analysis of the generated creative
            generation_data: Generation metadata (prompts used, model, etc.)

        Returns:
            GeneratorEvaluation with scores and feedback
        """
        # 1. Check prompt fidelity (did generator follow the prompt?)
        prompt_fidelity = self._check_prompt_fidelity(
            image_analysis, generation_data
        )

        # 2. Assess visual quality
        visual_quality = self._assess_visual_quality(
            image_path, image_analysis
        )

        # 3. Check brand consistency
        brand_consistency = self._check_brand_consistency(
            image_analysis, generation_data
        )

        # 4. Assess technical execution
        technical_execution = self._assess_technical_execution(
            image_path, image_analysis
        )

        # 5. Calculate overall generator score
        overall_score = self._calculate_generator_score({
            'prompt_fidelity': prompt_fidelity,
            'visual_quality': visual_quality,
            'brand_consistency': brand_consistency,
            'technical_execution': technical_execution
        })

        # 6. Generate feedback for generator improvement
        feedback = self._generate_generator_feedback({
            'prompt_fidelity': prompt_fidelity,
            'visual_quality': visual_quality,
            'brand_consistency': brand_consistency,
            'technical_execution': technical_execution,
            'generation_data': generation_data
        })

        return GeneratorEvaluation(
            overall_score=overall_score,
            prompt_fidelity=prompt_fidelity,
            visual_quality=visual_quality,
            brand_consistency=brand_consistency,
            technical_execution=technical_execution,
            feedback=feedback
        )

    def _check_prompt_fidelity(
        self,
        image_analysis: ImageAnalysis,
        generation_data: Dict
    ) -> Dict:
        """Check how well the generated image matches the prompt."""
        prompt = generation_data.get('prompt', '')
        converted_prompt = generation_data.get('converted_prompt', '')

        # Extract key elements from prompt
        prompt_elements = self._extract_prompt_elements(prompt)

        # Check which elements are present in the generated image
        detected_elements = image_analysis.get('elements', {})

        fidelity_stats = {
            'total_elements': len(prompt_elements),
            'present': 0,
            'partial': 0,
            'missing': 0,
            'element_details': []
        }

        for element, spec in prompt_elements.items():
            detected = detected_elements.get(element)

            if not detected:
                fidelity_stats['missing'] += 1
                fidelity_stats['element_details'].append({
                    'element': element,
                    'status': 'missing',
                    'specification': spec
                })
            elif self._is_element_present(spec, detected):
                fidelity_stats['present'] += 1
            else:
                fidelity_stats['partial'] += 1
                fidelity_stats['element_details'].append({
                    'element': element,
                    'status': 'partial',
                    'specification': spec,
                    'detected': detected
                })

        fidelity_rate = (
            fidelity_stats['present'] + 0.5 * fidelity_stats['partial']
        ) / fidelity_stats['total_elements'] if fidelity_stats['total_elements'] > 0 else 0

        return {
            'fidelity_rate': fidelity_rate,
            'stats': fidelity_stats,
            'score': fidelity_rate * 100
        }

    def _assess_visual_quality(
        self,
        image_path: str,
        image_analysis: ImageAnalysis
    ) -> Dict:
        """Assess visual quality of the generated image."""
        quality_metrics = {}

        # 1. Resolution check
        resolution = self._get_resolution(image_path)
        min_resolution = (1080, 1080)
        resolution_quality = (
            resolution[0] >= min_resolution[0] and
            resolution[1] >= min_resolution[1]
        )
        quality_metrics['resolution'] = {
            'detected': resolution,
            'required': min_resolution,
            'compliant': resolution_quality
        }

        # 2. Blur/sharpness check
        blur_score = image_analysis.get('blur_score', 0)
        quality_metrics['sharpness'] = {
            'blur_score': blur_score,
            'compliant': blur_score < 0.3,
            'rating': 'sharp' if blur_score < 0.1 else 'acceptable' if blur_score < 0.3 else 'blurry'
        }

        # 3. Noise/artifacts check
        artifacts = image_analysis.get('artifacts', [])
        quality_metrics['artifacts'] = {
            'detected': artifacts,
            'compliant': len(artifacts) == 0,
            'count': len(artifacts)
        }

        # 4. Composition quality
        composition = image_analysis.get('composition', {})
        quality_metrics['composition'] = {
            'balance': composition.get('balance', 'unknown'),
            'rule_of_thirds': composition.get('follows_rule_of_thirds', False),
            'focal_point': composition.get('has_clear_focal_point', False)
        }

        # Calculate overall visual quality score
        score_weights = {
            'resolution': 0.2,
            'sharpness': 0.3,
            'artifacts': 0.2,
            'composition': 0.3
        }

        overall_score = sum(
            score_weights[key] * (100 if metric['compliant'] else 50)
            for key, metric in quality_metrics.items()
        )

        return {
            'overall_score': overall_score,
            'metrics': quality_metrics,
            'score': overall_score
        }

    def _generate_generator_feedback(self, evaluation_data: Dict) -> Dict:
        """Generate structured feedback for generator improvement."""
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'prompt_feedback': [],
            'quality_feedback': [],
            'technical_feedback': [],
            'actionable_insights': []
        }

        # Prompt fidelity feedback
        fidelity = evaluation_data['prompt_fidelity']
        if fidelity['fidelity_rate'] < 0.7:
            feedback['prompt_feedback'].append({
                'type': 'low_fidelity',
                'issue': f"Low prompt fidelity rate: {fidelity['fidelity_rate']:.1%}",
                'missing_elements': [
                    e['element'] for e in fidelity['stats']['element_details']
                    if e['status'] == 'missing'
                ],
                'suggested_action': 'Review prompt construction - clarify missing elements'
            })

        # Visual quality feedback
        visual_quality = evaluation_data['visual_quality']['metrics']
        if not visual_quality['sharpness']['compliant']:
            feedback['quality_feedback'].append({
                'type': 'blur_detected',
                'blur_score': visual_quality['sharpness']['blur_score'],
                'suggested_action': 'Increase sharpness parameters or adjust upsampling settings'
            })

        if not visual_quality['artifacts']['compliant']:
            feedback['quality_feedback'].append({
                'type': 'artifacts_detected',
                'artifacts': visual_quality['artifacts']['detected'],
                'suggested_action': 'Review generation parameters - reduce sampling temperature'
            })

        # Technical execution feedback
        technical = evaluation_data['technical_execution']
        if technical.get('generation_time_seconds', 0) > 60:
            feedback['technical_feedback'].append({
                'type': 'slow_generation',
                'generation_time': technical['generation_time_seconds'],
                'suggested_action': 'Optimize generation pipeline or consider faster model'
            })

        return feedback
```

---

## Data Models

### Review Result (`models/review_result.py`)

```python
@dataclass
class ReviewResult:
    """Complete review result for a creative."""
    # Basic info
    image_path: str
    reviewed_at: str
    metadata: Dict[str, Any]

    # Overall assessment
    overall_score: float  # 0-100
    approved: bool

    # Dimension scores
    brand_compliance: BrandComplianceResult
    culture_fit: CultureFitResult
    technical_quality: TechnicalQualityResult
    compliance_status: ComplianceResult

    # Optional miner/generator evaluation
    miner_evaluation: Optional[MinerEvaluation] = None
    generator_evaluation: Optional[GeneratorEvaluation] = None

    # Recommendations
    recommendations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        report = f"""# Ad Review Report

**Image:** `{self.image_path}`
**Reviewed:** {self.reviewed_at}
**Overall Score:** {self.overall_score:.1f}/100
**Status:** {'✅ APPROVED' if self.approved else '❌ REJECTED'}

---

## Brand Compliance ({self.brand_compliance.score:.1f}/100)

**Status:** {'✅ Compliant' if self.brand_compliance.compliant else '❌ Non-compliant'}

### Checks
"""
        # Add brand compliance details
        for check_name, result in self.brand_compliance.checkes.items():
            status = '✅' if result.get('compliant', False) else '❌'
            report += f"\n- {status} **{check_name.title()}**: {result.get('issue', 'Pass')}"

        if self.brand_compliance.violations:
            report += "\n\n### Violations\n"
            for violation in self.brand_compliance.violations:
                report += f"\n- **[{violation['severity'].upper()}]** {violation['category']}: {violation['issue']}"

        # Add culture fit section
        report += f"""

---

## Culture Fit ({self.culture_fit.score:.1f}/100)

**Status:** {'✅ Appropriate' if self.culture_fit.compliant else '⚠️ Concerns detected'}

"""
        if self.culture_fit.detected_risks:
            report += "### Detected Risks\n"
            for risk in self.culture_fit.detected_risks:
                report += f"\n- **[{risk['severity'].upper()}]** {risk['category']}: {risk.get('recommendation', 'Review required')}"

        # Add recommendations
        report += "\n\n---\n\n## Recommendations\n\n"
        for i, rec in enumerate(self.recommendations, 1):
            report += f"{i}. {rec['text']}\n"

        return report

@dataclass
class BrandComplianceResult:
    """Brand guideline compliance results."""
    score: float
    compliant: bool
    checks: Dict[str, Any]
    violations: List[Dict[str, Any]]
    guideline_coverage: float

@dataclass
class CultureFitResult:
    """Culture fit assessment results."""
    score: float
    compliant: bool
    detected_risks: List[Dict[str, Any]]
    cultural_contexts: List[Dict[str, Any]]
    diversity_analysis: Dict[str, Any]
    market_specific_issues: Dict[str, List[Dict]]

@dataclass
class TechnicalQualityResult:
    """Technical quality assessment results."""
    score: float
    compliant: bool
    resolution: Dict[str, Any]
    sharpness: Dict[str, Any]
    artifacts: List[str]
    color_accuracy: Dict[str, Any]

@dataclass
class ComplianceResult:
    """Meta/Facebook ad compliance results."""
    score: float
    compliant: bool
    policy_violations: List[Dict[str, Any]]
    text_ratio_compliance: Dict[str, Any]
    community_standards: Dict[str, Any]

@dataclass
class MinerEvaluation:
    """Ad-miner evaluation results."""
    overall_score: float
    feature_adoption: Dict[str, Any]
    pattern_adherence: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    feedback: Dict[str, Any]

@dataclass
class GeneratorEvaluation:
    """Ad-generator evaluation results."""
    overall_score: float
    prompt_fidelity: Dict[str, Any]
    visual_quality: Dict[str, Any]
    brand_consistency: Dict[str, Any]
    technical_execution: Dict[str, Any]
    feedback: Dict[str, Any]
```

---

## Pipeline Integration

### Review Pipeline (`pipeline/pipeline.py`)

```python
class ReviewPipeline:
    """End-to-end review pipeline for ad creatives."""

    def __init__(self, config: ReviewPipelineConfig):
        self.config = config
        self.reviewer = AdReviewer(config)
        self.report_generator = ReportGenerator(config.output_format)

    def review_single_creative(
        self,
        image_path: str,
        metadata: Optional[Dict] = None,
        recommendation_path: Optional[str] = None,
        generation_metadata_path: Optional[str] = None
    ) -> ReviewResult:
        """
        Review a single creative.

        Args:
            image_path: Path to the creative image
            metadata: Optional metadata
            recommendation_path: Optional path to ad-miner recommendations
            generation_metadata_path: Optional path to generator metadata

        Returns:
            ReviewResult
        """
        # Load recommendation data if provided
        recommendation_data = None
        if recommendation_path:
            recommendation_data = self._load_recommendations(recommendation_path)

        # Load generation metadata if provided
        generation_data = None
        if generation_metadata_path:
            generation_data = self._load_generation_metadata(generation_metadata_path)

        # Perform review
        result = self.reviewer.review_creative(
            image_path=image_path,
            metadata=metadata,
            recommendation_data=recommendation_data,
            generation_data=generation_data
        )

        # Determine approval status
        result.approved = self._determine_approval(result)

        # Save report
        self._save_report(result)

        return result

    def review_batch(
        self,
        image_paths: List[str],
        metadata: Optional[Dict] = None
    ) -> BatchReviewResult:
        """
        Review multiple creatives in batch.

        Returns:
            BatchReviewResult with aggregate statistics
        """
        results = []

        for image_path in image_paths:
            try:
                result = self.review_single_creative(image_path, metadata)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to review {image_path}: {e}")
                results.append(None)

        # Calculate aggregate statistics
        approved_count = sum(1 for r in results if r and r.approved)
        avg_score = sum(r.overall_score for r in results if r) / len(results) if results else 0

        batch_result = BatchReviewResult(
            total_count=len(image_paths),
            approved_count=approved_count,
            rejected_count=len(image_paths) - approved_count,
            average_score=avg_score,
            results=results,
            generated_at=datetime.now().isoformat()
        )

        # Generate batch summary report
        self._generate_batch_summary(batch_result)

        return batch_result

    def review_generation_output(
        self,
        generation_output_dir: str
    ) -> BatchReviewResult:
        """
        Review all creatives from a generation batch.

        Args:
            generation_output_dir: Directory containing generated creatives

        Returns:
            BatchReviewResult
        """
        # Find all generated images
        image_paths = list(Path(generation_output_dir).glob('*.jpg'))

        # Load generation metadata
        metadata_path = Path(generation_output_dir) / 'generation_metadata.json'
        generation_metadata = None
        if metadata_path.exists():
            generation_metadata = self._load_json(metadata_path)

        # Review all images
        return self.review_batch(
            [str(p) for p in image_paths],
            metadata=generation_metadata
        )

    def _determine_approval(self, result: ReviewResult) -> bool:
        """Determine if creative should be approved."""
        # Must meet minimum score threshold
        if result.overall_score < 70:
            return False

        # Must be brand compliant
        if not result.brand_compliance.compliant:
            return False

        # Must be culturally appropriate
        if not result.culture_fit.compliant:
            return False

        # Must meet Meta compliance standards
        if not result.compliance_status.compliant:
            return False

        # In strict mode, no critical violations allowed
        if self.config.strict_mode:
            critical_violations = [
                v for v in result.brand_compliance.violations
                if v['severity'] == 'critical'
            ]
            if critical_violations:
                return False

        return True

    def _save_report(self, result: ReviewResult):
        """Save review report in configured format."""
        output_dir = self.reviewer.paths.reports_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        image_id = Path(result.image_path).stem

        if self.config.output_format == 'json':
            output_path = output_dir / f'{image_id}.json'
            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

        elif self.config.output_format == 'markdown':
            output_path = output_dir / f'{image_id}.md'
            with open(output_path, 'w') as f:
                f.write(result.to_markdown())

        logger.info(f"Review report saved to {output_path}")

    def _generate_batch_summary(self, batch_result: BatchReviewResult):
        """Generate batch review summary report."""
        summary_dir = self.reviewer.paths.batch_reports_dir
        summary_dir.mkdir(parents=True, exist_ok=True)

        batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = summary_dir / f'{batch_id}_summary.md'

        summary = f"""# Batch Review Summary

**Generated:** {batch_result.generated_at}
**Total Reviewed:** {batch_result.total_count}
**Approved:** {batch_result.approved_count} ({batch_result.approved_count/batch_result.total_count:.1%})
**Rejected:** {batch_result.rejected_count} ({batch_result.rejected_count/batch_result.total_count:.1%})
**Average Score:** {batch_result.average_score:.1f}/100

## Score Distribution

"""
        # Add score distribution
        scores = [r.overall_score for r in batch_result.results if r]
        if scores:
            summary += f"- **90-100:** {sum(1 for s in scores if s >= 90)}\n"
            summary += f"- **80-89:** {sum(1 for s in scores if 80 <= s < 90)}\n"
            summary += f"- **70-79:** {sum(1 for s in scores if 70 <= s < 80)}\n"
            summary += f"- **60-69:** {sum(1 for s in scores if 60 <= s < 70)}\n"
            summary += f"- **< 60:** {sum(1 for s in scores if s < 60)}\n"

        summary += "\n## Common Issues\n\n"

        # Aggregate common issues across all reviews
        all_violations = []
        for result in batch_result.results:
            if result:
                all_violations.extend(result.brand_compliance.violations)

        # Count violation types
        violation_counts = {}
        for violation in all_violations:
            category = violation['category']
            violation_counts[category] = violation_counts.get(category, 0) + 1

        for category, count in sorted(violation_counts.items(), key=lambda x: x[1], reverse=True):
            summary += f"- **{category}:** {count} occurrences\n"

        with open(summary_path, 'w') as f:
            f.write(summary)

        logger.info(f"Batch summary saved to {summary_path}")
```

---

## Integration with Existing Components

### 1. Integration with Ad-Generator

The ad-reviewer can automatically review outputs from the ad-generator:

```python
from src.meta.ad.generator.pipeline.pipeline import CreativePipeline
from src.meta.ad.reviewer.pipeline.pipeline import ReviewPipeline

# Generate creatives
generator_config = CreativePipelineConfig(
    product_name="Power Station",
    customer="moprobo",
    platform="meta"
)

generator = CreativePipeline(generator_config)
generation_result = generator.run(
    source_image_path="path/to/source.jpg",
    num_variations=5
)

# Review the generated creatives
reviewer_config = ReviewPipelineConfig(
    customer="moprobo",
    platform="meta"
)

reviewer = ReviewPipeline(reviewer_config)
batch_review = reviewer.review_generation_output(
    generation_result['output_dir']
)

print(f"Generated {generation_result['count']} creatives")
print(f"Approved: {batch_review.approved_count}/{batch_review.total_count}")
```

### 2. Integration with Ad-Miner

The ad-reviewer evaluates how well miner recommendations were followed:

```python
from src.meta.ad.reviewer.pipeline.pipeline import ReviewPipeline

# Review creative with miner recommendations
reviewer = ReviewPipeline(config)

review_result = reviewer.review_single_creative(
    image_path="generated/creative.jpg",
    recommendation_path="config/ad/miner/moprobo/meta/2024-01-15/recommendations.md",
    generation_metadata_path="generated/creative_metadata.json"
)

# Access miner evaluation
if review_result.miner_evaluation:
    print(f"Miner score: {review_result.miner_evaluation.overall_score:.1f}")
    print(f"Feature adoption: {review_result.miner_evaluation.feature_adoption['adoption_rate']:.1%}")
```

### 3. Feedback Loop

Feedback from the reviewer can be used to improve both miner and generator:

```python
# Save feedback for miner
miner_feedback_path = "results/ad/reviewer/moprobo/meta/2024-01-15/feedback/miner_feedback.json"

with open(miner_feedback_path, 'w') as f:
    json.dump(review_result.miner_evaluation.feedback, f, indent=2)

# Save feedback for generator
generator_feedback_path = "results/ad/reviewer/moprobo/meta/2024-01-15/feedback/generator_feedback.json"

with open(generator_feedback_path, 'w') as f:
    json.dump(review_result.generator_evaluation.feedback, f, indent=2)
```

---

## Implementation Phases

### Phase 1: Core Review Functionality (Priority)
1. Implement `AdReviewer` class with basic structure
2. Implement `BrandGuidelinesChecker` with logo, color, and typography checks
3. Implement `CultureFitAnalyzer` with basic risk detection
4. Implement `TechnicalQualityChecker` for resolution and sharpness
5. Implement `ReviewResult` data models
6. Implement basic `ReviewPipeline`

### Phase 2: Miner & Generator Evaluation
1. Implement `MinerEvaluator` with feature adoption checking
2. Implement `GeneratorEvaluator` with prompt fidelity assessment
3. Implement feedback generation for both components
4. Implement feedback aggregation and persistence

### Phase 3: Advanced Features
1. Implement comprehensive Meta compliance checking
2. Implement market-specific cultural context analysis
3. Implement advanced diversity and representation analysis
4. Implement batch review with aggregate statistics
5. Implement HTML report generation

### Phase 4: Integration & Optimization
1. Integrate with ad-generator pipeline
2. Integrate with ad-miner feedback loop
3. Implement automated review triggers
4. Optimize GPT-4 Vision API usage (batching, caching)
5. Implement review history and trend analysis

---

## Configuration Examples

### Example 1: Basic Review

```python
config = ReviewPipelineConfig(
    customer="moprobo",
    platform="meta"
)
```

### Example 2: Strict Review with Custom Criteria

```python
criteria = CriteriaConfig(
    brand_weight=0.5,
    culture_weight=0.3,
    technical_weight=0.2,
    min_culture_score=85.0,
    color_tolerance_delta_e=3.0  # Stricter color matching
)

config = ReviewPipelineConfig(
    customer="moprobo",
    platform="meta",
    review_criteria=criteria,
    strict_mode=True  # Fail on any critical violation
)
```

### Example 3: Multi-Market Review

```python
config = ReviewPipelineConfig(
    customer="ecoflow",
    platform="meta",
    product_context={
        'target_markets': ['US', 'UK', 'DE', 'JP', 'SA']
    }
)
```

---

## Key Design Decisions

### 1. Separation of Concerns
- **Review Logic** (`core/reviewer.py`) - Orchestrates the review process
- **Criteria Checking** (`criteria/`) - Individual evaluation dimensions
- **Analyzers** (`analyzers/`) - Raw analysis (vision API, color extraction, etc.)
- **Evaluators** (`evaluators/`) - Miner/generator-specific evaluation

### 2. Extensibility
- New review criteria can be added by extending `criteria/` modules
- New analyzers can be added to `analyzers/` as needed
- Pluggable scoring system via `CriteriaConfig`

### 3. Integration with Existing Components
- Reuses `brand_guidelines.py` from ad-generator
- Follows same path organization pattern
- Compatible with existing recommendation and metadata formats

### 4. API-First Design
- GPT-4 Vision API for image analysis
- Extensible to other vision APIs (Google Cloud Vision, AWS Rekognition, etc.)

### 5. Feedback Loop
- Structured feedback for both miner and generator
- Enables continuous improvement of the entire pipeline

---

## Future Enhancements

1. **Automated Retraining**: Use reviewer feedback to automatically retrain miner models
2. **A/B Testing Integration**: Review creatives from different prompt strategies
3. **Trend Analysis**: Track review scores over time to identify quality trends
4. **Competitor Analysis**: Review competitor ads for benchmarking
5. **Real-Time Review**: Integrate with generation pipeline for real-time quality checks
6. **Multi-Modal Review**: Analyze video creatives in addition to images
7. **Explainable AI**: Provide detailed explanations for review decisions

---

## Conclusion

The Ad Reviewer component completes the meta pipeline by providing comprehensive quality assessment across brand compliance, culture fit, and technical dimensions. By evaluating both miner recommendations and generator outputs, it creates a feedback loop that continuously improves the entire creative generation system.

The design follows established patterns from ad-generator and ad-miner, ensuring consistency and maintainability across the codebase.
