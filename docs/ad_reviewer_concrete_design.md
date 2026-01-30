# Ad Reviewer - Concrete Implementation Design

## Table of Contents
1. [File Structure](#file-structure)
2. [Core Classes](#core-classes)
3. [Data Models](#data-models)
4. [Configuration](#configuration)
5. [API Interface](#api-interface)
6. [Data Flow](#data-flow)
7. [Error Handling](#error-handling)
8. [Testing Strategy](#testing-strategy)

---

## 1. File Structure

```
src/meta/ad/reviewer/
├── __init__.py
├── main.py                          # Entry point
│
├── core/
│   ├── __init__.py
│   ├── reviewer.py                  # Main AdReviewer class
│   ├── paths.py                     # Path management
│   └── constants.py                 # Constants and enums
│
├── models/
│   ├── __init__.py
│   ├── review_result.py             # ReviewResult dataclass
│   ├── criteria_config.py           # CriteriaConfig dataclass
│   ├── score_breakdown.py           # ScoreBreakdown dataclass
│   ├── violation.py                 # Violation dataclass
│   └── recommendation.py            # Recommendation dataclass
│
├── analyzers/
│   ├── __init__.py
│   ├── base_analyzer.py             # Base analyzer class
│   ├── vision_analyzer.py           # GPT-4 Vision integration
│   ├── color_analyzer.py            # Color analysis
│   ├── text_analyzer.py             # Text overlay detection
│   ├── logo_analyzer.py             # Logo detection and analysis
│   └── composition_analyzer.py      # Composition analysis
│
├── checkers/
│   ├── __init__.py
│   ├── base_checker.py              # Base checker class
│   ├── brand_guidelines_checker.py  # Brand compliance
│   ├── culture_risk_checker.py      # Culture and risk assessment
│   ├── technical_quality_checker.py # Technical quality
│   ├── compliance_checker.py        # Meta/Facebook compliance
│   ├── campaign_goal_checker.py     # Campaign goal alignment
│   └── regional_pattern_checker.py  # Regional pattern matching
│
├── evaluators/
│   ├── __init__.py
│   ├── base_evaluator.py            # Base evaluator class
│   ├── miner_evaluator.py           # Ad-miner evaluation
│   ├── generator_evaluator.py       # Ad-generator evaluation
│   └── feedback_aggregator.py       # Feedback aggregation
│
├── predictors/
│   ├── __init__.py
│   ├── performance_predictor.py     # Performance prediction
│   └── feature_impact_calculator.py # Feature impact analysis
│
├── pipeline/
│   ├── __init__.py
│   ├── review_pipeline.py           # Main pipeline orchestrator
│   ├── batch_processor.py           # Batch processing
│   └── report_generator.py          # Report generation (JSON/MD/HTML)
│
├── utils/
│   ├── __init__.py
│   ├── scoring.py                   # Scoring utilities
│   ├── vision_api.py                # GPT-4 Vision API wrapper
│   ├── color_utils.py               # Color manipulation (Delta E, etc.)
│   ├── image_utils.py               # Image processing utilities
│   ├── text_utils.py                # Text analysis utilities
│   └── formatters.py                # Output formatters
│
├── config/
│   ├── __init__.py
│   ├── defaults.py                  # Default configurations
│   └── loader.py                    # Config loader
│
└── tests/
    ├── __init__.py
    ├── test_reviewer.py
    ├── test_brand_checker.py
    ├── test_culture_checker.py
    └── fixtures/
        ├── sample_images/
        └── sample_configs/

config/ad/reviewer/
├── brand_guidelines/
│   ├── moprobo.yaml
│   ├── ecoflow.yaml
│   └── generic.yaml
│
├── risk_profiles/
│   ├── united_states.yaml
│   ├── saudi_arabia.yaml
│   ├── germany.yaml
│   ├── japan.yaml
│   ├── brazil.yaml
│   └── global.yaml
│
├── regional_patterns/
│   ├── us_performance.yaml
│   ├── uk_performance.yaml
│   ├── de_performance.yaml
│   ├── jp_performance.yaml
│   ├── cn_performance.yaml
│   ├── sa_performance.yaml
│   └── br_performance.yaml
│
├── campaign_goals/
│   ├── brand_awareness.yaml
│   ├── consideration.yaml
│   ├── conversion.yaml
│   ├── retention.yaml
│   ├── lead_generation.yaml
│   └── app_install.yaml
│
└── criteria/
    ├── default.yaml
    ├── strict.yaml
    └── permissive.yaml

results/ad/reviewer/
└── {customer}/
    └── {platform}/
        └── {date}/
            ├── reviews/                 # Individual reviews
            │   ├── {image_id}.json
            │   └── {image_id}.md
            ├── batches/                 # Batch reviews
            │   └── {batch_id}_summary.json
            ├── feedback/                # Miner/Generator feedback
            │   ├── miner_feedback.json
            │   └── generator_feedback.json
            └── metrics/                 # Performance metrics
                └── metrics.json
```

---

## 2. Core Classes

### 2.1 Main Reviewer Class

```python
# src/meta/ad/reviewer/core/reviewer.py

from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .paths import ReviewerPaths
from .constants import Severity, RiskCategory, CampaignGoal
from ..models.review_result import ReviewResult
from ..models.criteria_config import CriteriaConfig
from ..analyzers.vision_analyzer import VisionAnalyzer
from ..analyzers.color_analyzer import ColorAnalyzer
from ..analyzers.text_analyzer import TextAnalyzer
from ..analyzers.logo_analyzer import LogoAnalyzer
from ..checkers.brand_guidelines_checker import BrandGuidelinesChecker
from ..checkers.culture_risk_checker import CultureRiskChecker
from ..checkers.technical_quality_checker import TechnicalQualityChecker
from ..checkers.compliance_checker import ComplianceChecker
from ..checkers.campaign_goal_checker import CampaignGoalChecker
from ..checkers.regional_pattern_checker import RegionalPatternChecker
from ..evaluators.miner_evaluator import MinerEvaluator
from ..evaluators.generator_evaluator import GeneratorEvaluator
from ..predictors.performance_predictor import PerformancePredictor

logger = logging.getLogger(__name__)


class AdReviewer:
    """
    Main ad reviewer orchestrator.

    Coordinates all analyzers, checkers, and evaluators to provide
    comprehensive review of ad creatives.

    Usage:
        reviewer = AdReviewer(
            customer="moprobo",
            platform="meta",
            region="US",
            campaign_goal=CampaignGoal.CONVERSION
        )

        result = reviewer.review_creative(
            image_path="path/to/creative.jpg",
            metadata={"product": "Power Station"}
        )

        if result.approved:
            print(f"✅ Approved - Score: {result.overall_score:.1f}")
        else:
            print(f"❌ Rejected - {len(result.critical_violations)} critical violations")
    """

    def __init__(
        self,
        customer: str,
        platform: str = "meta",
        region: str = "US",
        campaign_goal: Optional[CampaignGoal] = None,
        criteria_config: Optional[CriteriaConfig] = None,
        output_dir: Optional[Path] = None,
        enable_miner_eval: bool = True,
        enable_generator_eval: bool = True,
        enable_performance_prediction: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize the ad reviewer.

        Args:
            customer: Customer/brand name
            platform: Advertising platform (meta, google, tiktok)
            region: Target region (US, UK, DE, JP, etc.)
            campaign_goal: Campaign objective
            criteria_config: Review criteria configuration
            output_dir: Output directory for reports
            enable_miner_eval: Enable miner evaluation
            enable_generator_eval: Enable generator evaluation
            enable_performance_prediction: Enable performance prediction
            strict_mode: Fail on any violation (no warnings)
        """
        self.customer = customer
        self.platform = platform
        self.region = region
        self.campaign_goal = campaign_goal
        self.strict_mode = strict_mode
        self.enable_miner_eval = enable_miner_eval
        self.enable_generator_eval = enable_generator_eval
        self.enable_performance_prediction = enable_performance_prediction

        # Initialize paths
        self.paths = ReviewerPaths(
            customer=customer,
            platform=platform,
            output_dir=output_dir
        )

        # Load configuration
        self.criteria_config = criteria_config or self._load_default_criteria()
        self.brand_guidelines = self._load_brand_guidelines(customer)
        self.risk_profile = self._load_risk_profile(region)

        # Initialize analyzers
        self.vision_analyzer = VisionAnalyzer()
        self.color_analyzer = ColorAnalyzer()
        self.text_analyzer = TextAnalyzer()
        self.logo_analyzer = LogoAnalyzer(self.brand_guidelines)

        # Initialize checkers
        self.brand_checker = BrandGuidelinesChecker(
            self.brand_guidelines,
            self.criteria_config
        )
        self.culture_checker = CultureRiskChecker(
            self.risk_profile,
            region,
            self.criteria_config
        )
        self.technical_checker = TechnicalQualityChecker(
            self.criteria_config
        )
        self.compliance_checker = ComplianceChecker(
            platform,
            self.criteria_config
        )
        self.goal_checker = CampaignGoalChecker(
            campaign_goal,
            self.criteria_config
        ) if campaign_goal else None
        self.regional_checker = RegionalPatternChecker(
            region,
            self.criteria_config
        )

        # Initialize evaluators
        self.miner_evaluator = MinerEvaluator() if enable_miner_eval else None
        self.generator_evaluator = GeneratorEvaluator() if enable_generator_eval else None

        # Initialize predictor
        self.performance_predictor = PerformancePredictor(
            region,
            platform
        ) if enable_performance_prediction else None

        logger.info(
            f"AdReviewer initialized for {customer}/{platform} in {region} "
            f"(strict_mode={strict_mode})"
        )

    def review_creative(
        self,
        image_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        recommendation_data: Optional[Dict] = None,
        generation_data: Optional[Dict] = None,
        save_report: bool = True
    ) -> ReviewResult:
        """
        Review a single creative comprehensively.

        Args:
            image_path: Path to the creative image
            metadata: Optional metadata (product context, etc.)
            recommendation_data: Optional data from ad-miner
            generation_data: Optional data from ad-generator
            save_report: Whether to save review report

        Returns:
            ReviewResult with complete analysis
        """
        logger.info(f"Reviewing creative: {image_path}")
        start_time = datetime.now()

        try:
            # Step 1: Analyze image with GPT-4 Vision
            logger.debug("Step 1: Vision analysis")
            image_analysis = self._analyze_image(image_path)

            # Step 2: Check brand guidelines compliance
            logger.debug("Step 2: Brand guidelines check")
            brand_result = self.brand_checker.check(image_path, image_analysis)

            # Step 3: Check cultural fit and risks
            logger.debug("Step 3: Culture risk check")
            culture_result = self.culture_checker.check(
                image_path,
                image_analysis
            )

            # Step 4: Check technical quality
            logger.debug("Step 4: Technical quality check")
            technical_result = self.technical_checker.check(
                image_path,
                image_analysis
            )

            # Step 5: Check platform compliance
            logger.debug("Step 5: Platform compliance check")
            compliance_result = self.compliance_checker.check(
                image_path,
                image_analysis
            )

            # Step 6: Check campaign goal alignment (if configured)
            goal_result = None
            if self.goal_checker:
                logger.debug("Step 6: Campaign goal check")
                goal_result = self.goal_checker.check(
                    image_analysis,
                    metadata
                )

            # Step 7: Check regional pattern match
            regional_result = None
            if self.regional_checker:
                logger.debug("Step 7: Regional pattern check")
                regional_result = self.regional_checker.check(
                    image_analysis,
                    metadata
                )

            # Step 8: Evaluate miner recommendations (if data provided)
            miner_result = None
            if self.miner_evaluator and recommendation_data:
                logger.debug("Step 8: Miner evaluation")
                miner_result = self.miner_evaluator.evaluate(
                    image_path,
                    image_analysis,
                    recommendation_data
                )

            # Step 9: Evaluate generator output (if data provided)
            generator_result = None
            if self.generator_evaluator and generation_data:
                logger.debug("Step 9: Generator evaluation")
                generator_result = self.generator_evaluator.evaluate(
                    image_path,
                    image_analysis,
                    generation_data
                )

            # Step 10: Predict performance (if enabled)
            performance_prediction = None
            if self.performance_predictor:
                logger.debug("Step 10: Performance prediction")
                performance_prediction = self.performance_predictor.predict(
                    image_analysis,
                    self.campaign_goal,
                    metadata
                )

            # Step 11: Calculate overall score and determine approval
            logger.debug("Step 11: Calculate overall score")
            overall_score = self._calculate_overall_score({
                'brand': brand_result.score,
                'culture': culture_result.score,
                'technical': technical_result.score,
                'compliance': compliance_result.score,
                'goal': goal_result.score if goal_result else None,
                'regional': regional_result.match_score if regional_result else None
            })

            # Step 12: Aggregate all violations
            all_violations = self._aggregate_violations({
                'brand': brand_result.violations,
                'culture': culture_result.violations,
                'technical': technical_result.violations,
                'compliance': compliance_result.violations
            })

            # Step 13: Determine approval status
            approved = self._determine_approval(
                overall_score,
                all_violations
            )

            # Step 14: Generate recommendations
            recommendations = self._generate_recommendations({
                'brand': brand_result,
                'culture': culture_result,
                'technical': technical_result,
                'compliance': compliance_result,
                'goal': goal_result,
                'regional': regional_result,
                'miner': miner_result,
                'generator': generator_result,
                'performance': performance_prediction
            })

            # Step 15: Create result object
            duration = (datetime.now() - start_time).total_seconds()

            result = ReviewResult(
                image_path=image_path,
                customer=self.customer,
                platform=self.platform,
                region=self.region,
                campaign_goal=self.campaign_goal,
                overall_score=overall_score,
                approved=approved,
                brand_compliance=brand_result,
                culture_fit=culture_result,
                technical_quality=technical_result,
                compliance_status=compliance_result,
                goal_alignment=goal_result,
                regional_match=regional_result,
                miner_evaluation=miner_result,
                generator_evaluation=generator_result,
                performance_prediction=performance_prediction,
                violations=all_violations,
                recommendations=recommendations,
                metadata=metadata or {},
                reviewed_at=datetime.now().isoformat(),
                review_duration_seconds=duration
            )

            # Step 16: Save report if requested
            if save_report:
                self._save_report(result)

            logger.info(
                f"Review complete: Score={overall_score:.1f}, "
                f"Approved={approved}, Duration={duration:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Review failed for {image_path}: {e}", exc_info=True)
            raise ReviewError(f"Failed to review creative: {e}")

    def review_batch(
        self,
        image_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        max_workers: int = 4
    ) -> BatchReviewResult:
        """
        Review multiple creatives in parallel.

        Args:
            image_paths: List of image paths to review
            metadata: Shared metadata for all creatives
            max_workers: Maximum parallel workers

        Returns:
            BatchReviewResult with aggregate statistics
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info(f"Starting batch review of {len(image_paths)} creatives")

        results = []
        failed = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all review tasks
            future_to_path = {
                executor.submit(
                    self.review_creative,
                    path,
                    metadata,
                    save_report=False
                ): path
                for path in image_paths
            }

            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to review {path}: {e}")
                    failed.append({
                        'path': path,
                        'error': str(e)
                    })

        # Calculate aggregate statistics
        batch_result = self._create_batch_result(results, failed, metadata)

        # Save batch summary
        self._save_batch_summary(batch_result)

        logger.info(
            f"Batch review complete: {batch_result.approved_count}/{batch_result.total_count} approved"
        )

        return batch_result

    def _analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image using all available analyzers."""
        # Vision analysis (GPT-4 Vision)
        vision_result = self.vision_analyzer.analyze(image_path)

        # Color analysis
        color_result = self.color_analyzer.analyze(image_path, vision_result)

        # Text analysis
        text_result = self.text_analyzer.analyze(image_path, vision_result)

        # Logo analysis
        logo_result = self.logo_analyzer.analyze(image_path, vision_result)

        # Merge all analysis results
        return {
            **vision_result,
            'colors': color_result,
            'text_overlays': text_result,
            'logo': logo_result
        }

    def _calculate_overall_score(self, scores: Dict[str, Optional[float]]) -> float:
        """Calculate overall score from component scores."""
        weights = self.criteria_config.scoring_weights

        total_weight = 0.0
        weighted_sum = 0.0

        for component, score in scores.items():
            if score is not None and component in weights:
                weight = weights[component]
                weighted_sum += score * weight
                total_weight += weight

        return (weighted_sum / total_weight) if total_weight > 0 else 0.0

    def _aggregate_violations(self, violation_groups: Dict[str, List]) -> List['Violation']:
        """Aggregate all violations from all checkers."""
        from ..models.violation import Violation

        all_violations = []

        for checker_name, violations in violation_groups.items():
            for violation_data in violations:
                if isinstance(violation_data, dict):
                    all_violations.append(Violation(**violation_data))
                elif isinstance(violation_data, Violation):
                    all_violations.append(violation_data)

        # Sort by severity (critical first)
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3
        }
        all_violations.sort(key=lambda v: severity_order.get(v.severity, 4))

        return all_violations

    def _determine_approval(
        self,
        overall_score: float,
        violations: List['Violation']
    ) -> bool:
        """Determine if creative should be approved."""
        # Check minimum score threshold
        min_score = self.criteria_config.min_overall_score
        if overall_score < min_score:
            logger.debug(f"Rejected: Score {overall_score:.1f} < {min_score}")
            return False

        # Check for critical violations
        critical_violations = [v for v in violations if v.severity == Severity.CRITICAL]
        if critical_violations:
            logger.debug(f"Rejected: {len(critical_violations)} critical violations")
            return False

        # In strict mode, reject on any violation
        if self.strict_mode and violations:
            logger.debug(f"Rejected (strict mode): {len(violations)} violations")
            return False

        # Check required components
        required_checks = {
            'brand_compliance': self.criteria_config.require_brand_compliance,
            'culture_fit': self.criteria_config.require_culture_fit,
            'technical_quality': self.criteria_config.require_technical_quality,
            'compliance': self.criteria_config.require_compliance
        }

        # In production, check actual results here
        # For now, assume if score passes, components pass

        return True

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations from all results."""
        recommendations = []

        # Brand recommendations
        if 'brand' in results:
            brand = results['brand']
            if hasattr(brand, 'recommendations'):
                recommendations.extend(brand.recommendations)

        # Culture recommendations
        if 'culture' in results:
            culture = results['culture']
            if hasattr(culture, 'recommendations'):
                recommendations.extend(culture.recommendations)

        # Technical recommendations
        if 'technical' in results:
            technical = results['technical']
            if hasattr(technical, 'recommendations'):
                recommendations.extend(technical.recommendations)

        # Goal alignment recommendations
        if 'goal' in results and results['goal']:
            goal = results['goal']
            if hasattr(goal, 'recommendations'):
                recommendations.extend(goal.recommendations)

        # Regional pattern recommendations
        if 'regional' in results and results['regional']:
            regional = results['regional']
            if hasattr(regional, 'recommendations'):
                recommendations.extend(regional.recommendations)

        # Performance optimization recommendations
        if 'performance' in results and results['performance']:
            performance = results['performance']
            if hasattr(performance, 'optimization_recommendations'):
                recommendations.extend(performance.optimization_recommendations)

        return recommendations

    def _save_report(self, result: ReviewResult):
        """Save review report in multiple formats."""
        # Save JSON
        json_path = self.paths.get_review_path(result.image_id, 'json')
        result.save_json(json_path)

        # Save Markdown
        md_path = self.paths.get_review_path(result.image_id, 'md')
        result.save_markdown(md_path)

        logger.debug(f"Reports saved to {json_path} and {md_path}")

    def _load_default_criteria(self) -> CriteriaConfig:
        """Load default criteria configuration."""
        from ..config.loader import ConfigLoader

        loader = ConfigLoader()
        return loader.load_criteria_config('default')

    def _load_brand_guidelines(self, customer: str) -> Dict[str, Any]:
        """Load brand guidelines for customer."""
        from ..config.loader import ConfigLoader

        loader = ConfigLoader()
        return loader.load_brand_guidelines(customer)

    def _load_risk_profile(self, region: str) -> Dict[str, Any]:
        """Load risk profile for region."""
        from ..config.loader import ConfigLoader

        loader = ConfigLoader()
        return loader.load_risk_profile(region)


class ReviewError(Exception):
    """Exception raised during review process."""
    pass
```

---

## 3. Data Models

### 3.1 Review Result

```python
# src/meta/ad/reviewer/models/review_result.py

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import json

from .violation import Violation
from .score_breakdown import ScoreBreakdown
from ..core.constants import CampaignGoal


@dataclass
class BrandComplianceResult:
    """Brand guidelines compliance result."""
    score: float
    compliant: bool
    checks: Dict[str, Any]
    violations: List[Dict[str, Any]]
    guideline_coverage: float
    logo_detected: bool
    logo_compliance: Optional[Dict[str, Any]] = None
    color_compliance: Optional[Dict[str, Any]] = None
    typography_compliance: Optional[Dict[str, Any]] = None


@dataclass
class CultureFitResult:
    """Culture fit assessment result."""
    score: float
    compliant: bool
    detected_risks: List[Dict[str, Any]]
    cultural_contexts: List[Dict[str, Any]]
    diversity_analysis: Dict[str, Any]
    market_specific_issues: Dict[str, List[Dict]]
    risk_categories: List[str]


@dataclass
class TechnicalQualityResult:
    """Technical quality assessment result."""
    score: float
    compliant: bool
    resolution: Dict[str, Any]
    sharpness: Dict[str, Any]
    artifacts: List[str]
    color_accuracy: Dict[str, Any]
    text_quality: Dict[str, Any]
    file_size: Optional[int] = None
    format: Optional[str] = None


@dataclass
class ComplianceResult:
    """Platform compliance result."""
    score: float
    compliant: bool
    policy_violations: List[Dict[str, Any]]
    text_ratio_compliance: Dict[str, Any]
    community_standards: Dict[str, Any]
    platform_specific_rules: Dict[str, Any]


@dataclass
class GoalAlignmentResult:
    """Campaign goal alignment result."""
    campaign_goal: Optional[CampaignGoal]
    alignment_score: float
    passed_checks: List[Dict[str, Any]]
    failed_checks: List[Dict[str, Any]]
    recommendations: List[str]


@dataclass
class RegionalMatchResult:
    """Regional pattern match result."""
    region: str
    match_score: float
    performance_prediction: str
    high_performing_elements: List[Dict[str, Any]]
    low_performing_elements: List[Dict[str, Any]]
    seasonal_alignment: Dict[str, Any]
    recommendations: List[str]


@dataclass
class MinerEvaluation:
    """Ad-miner evaluation result."""
    overall_score: float
    feature_adoption: Dict[str, Any]
    pattern_adherence: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    feedback: Dict[str, Any]


@dataclass
class GeneratorEvaluation:
    """Ad-generator evaluation result."""
    overall_score: float
    prompt_fidelity: Dict[str, Any]
    visual_quality: Dict[str, Any]
    brand_consistency: Dict[str, Any]
    technical_execution: Dict[str, Any]
    feedback: Dict[str, Any]


@dataclass
class PerformancePrediction:
    """Performance prediction result."""
    predicted_ctr: float
    predicted_conversion_rate: float
    confidence_interval: tuple
    prediction_confidence: float
    key_positive_factors: List[str]
    key_negative_factors: List[str]
    optimization_priority: List[Dict[str, Any]]


@dataclass
class ReviewResult:
    """Complete review result for a creative."""
    # Basic info
    image_path: str
    customer: str
    platform: str
    region: str
    campaign_goal: Optional[CampaignGoal]
    image_id: str

    # Overall assessment
    overall_score: float
    approved: bool
    reviewed_at: str
    review_duration_seconds: float

    # Component results
    brand_compliance: BrandComplianceResult
    culture_fit: CultureFitResult
    technical_quality: TechnicalQualityResult
    compliance_status: ComplianceResult

    # Optional results
    goal_alignment: Optional[GoalAlignmentResult] = None
    regional_match: Optional[RegionalMatchResult] = None
    miner_evaluation: Optional[MinerEvaluation] = None
    generator_evaluation: Optional[GeneratorEvaluation] = None
    performance_prediction: Optional[PerformancePrediction] = None

    # Aggregates
    violations: List[Violation] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def critical_violations(self) -> List[Violation]:
        """Get all critical violations."""
        return [v for v in self.violations if v.severity.name == 'CRITICAL']

    @property
    def high_violations(self) -> List[Violation]:
        """Get all high-severity violations."""
        return [v for v in self.violations if v.severity.name == 'HIGH']

    @property
    def has_critical_violations(self) -> bool:
        """Check if any critical violations exist."""
        return len(self.critical_violations) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)

        # Convert enums to strings
        if self.campaign_goal:
            data['campaign_goal'] = self.campaign_goal.value

        # Convert violations
        data['violations'] = [v.to_dict() for v in self.violations]

        return data

    def save_json(self, path: str):
        """Save as JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_markdown(self, path: str):
        """Save as Markdown file."""
        md_content = self._generate_markdown()
        with open(path, 'w') as f:
            f.write(md_content)

    def _generate_markdown(self) -> str:
        """Generate markdown report."""
        md = f"""# Ad Review Report

**Customer:** {self.customer}
**Platform:** {self.platform}
**Region:** {self.region}
**Campaign Goal:** {self.campaign_goal.value if self.campaign_goal else 'N/A'}

**Image:** `{self.image_path}`
**Reviewed:** {self.reviewed_at}
**Duration:** {self.review_duration_seconds:.2f}s

---

## Overall Assessment

**Score:** {self.overall_score:.1f}/100
**Status:** {'✅ **APPROVED**' if self.approved else '❌ **REJECTED**'}

---

## Component Scores

### Brand Compliance ({self.brand_compliance.score:.1f}/100)
**Status:** {'✅ Compliant' if self.brand_compliance.compliant else '❌ Non-compliant'}

"""
        # Add brand compliance details
        md += self._brand_compliance_markdown()

        # Culture fit
        md += f"""

### Culture Fit ({self.culture_fit.score:.1f}/100)
**Status:** {'✅ Appropriate' if self.culture_fit.compliant else '⚠️ Concerns detected'}

"""
        md += self._culture_fit_markdown()

        # Technical quality
        md += f"""

### Technical Quality ({self.technical_quality.score:.1f}/100)
**Status:** {'✅ Good quality' if self.technical_quality.compliant else '⚠️ Quality issues detected'}

"""
        md += self._technical_quality_markdown()

        # Compliance
        md += f"""

### Platform Compliance ({self.compliance_status.score:.1f}/100)
**Status:** {'✅ Compliant' if self.compliance_status.compliant else '❌ Policy violations'}

"""
        md += self._compliance_markdown()

        # Goal alignment
        if self.goal_alignment:
            md += f"""

### Campaign Goal Alignment ({self.goal_alignment.alignment_score:.1f}/100)

"""
            md += self._goal_alignment_markdown()

        # Regional match
        if self.regional_match:
            md += f"""

### Regional Pattern Match ({self.regional_match.match_score:.1f}/100)

"""
            md += self._regional_match_markdown()

        # Performance prediction
        if self.performance_prediction:
            md += f"""

### Performance Prediction

"""
            md += self._performance_prediction_markdown()

        # Violations
        if self.violations:
            md += """

---

## Violations

"""
            md += self._violations_markdown()

        # Recommendations
        if self.recommendations:
            md += """

---

## Recommendations

"""
            for i, rec in enumerate(self.recommendations, 1):
                md += f"{i}. {rec}\n"

        return md

    def _brand_compliance_markdown(self) -> str:
        """Generate brand compliance section."""
        md = ""
        for check_name, result in self.brand_compliance.checks.items():
            status = '✅' if result.get('compliant', False) else '❌'
            md += f"- {status} **{check_name.title()}**: {result.get('issue', 'Pass')}\n"

        if self.brand_compliance.violations:
            md += "\n**Violations:**\n"
            for v in self.brand_compliance.violations:
                md += f"- **[{v['severity'].upper()}]** {v['category']}: {v['issue']}\n"

        return md

    def _culture_fit_markdown(self) -> str:
        """Generate culture fit section."""
        md = ""
        if self.culture_fit.detected_risks:
            md += "**Detected Risks:**\n"
            for risk in self.culture_fit.detected_risks:
                md += f"- **[{risk['severity'].upper()}]** {risk['category']}: {risk.get('recommendation', 'Review required')}\n"
        else:
            md += "✅ No cultural risks detected\n"

        return md

    def _technical_quality_markdown(self) -> str:
        """Generate technical quality section."""
        md = f"""
- **Resolution:** {self.technical_quality.resolution.get('detected', 'N/A')}
- **Sharpness:** {self.technical_quality.sharpness.get('rating', 'N/A')}
- **Artifacts:** {len(self.technical_quality.artifacts)} detected
"""
        return md

    def _compliance_markdown(self) -> str:
        """Generate compliance section."""
        md = ""
        if self.compliance_status.policy_violations:
            md += "**Policy Violations:**\n"
            for v in self.compliance_status.policy_violations:
                md += f"- {v}\n"
        else:
            md += "✅ No policy violations\n"

        return md

    def _violations_markdown(self) -> str:
        """Generate violations section."""
        md = ""

        # Group by severity
        by_severity = {}
        for v in self.violations:
            severity = v.severity.name
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(v)

        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if severity in by_severity:
                md += f"\n### {severity.title()}\n\n"
                for v in by_severity[severity]:
                    md += f"- **{v.category}**: {v.description}\n"
                    if v.recommendation:
                        md += f"  - *Recommendation: {v.recommendation}*\n"

        return md


@dataclass
class BatchReviewResult:
    """Result of batch review."""
    total_count: int
    approved_count: int
    rejected_count: int
    average_score: float
    score_distribution: Dict[str, int]
    common_violations: Dict[str, int]
    results: List[ReviewResult]
    failed: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    generated_at: str
    duration_seconds: float

    def save_summary(self, path: str):
        """Save batch summary as JSON."""
        import json
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    def generate_markdown_summary(self) -> str:
        """Generate markdown summary report."""
        md = f"""# Batch Review Summary

**Generated:** {self.generated_at}
**Duration:** {self.duration_seconds:.2f}s

## Overview

- **Total Reviewed:** {self.total_count}
- **Approved:** {self.approved_count} ({self.approved_count/self.total_count:.1%})
- **Rejected:** {self.rejected_count} ({self.rejected_count/self.total_count:.1%})
- **Average Score:** {self.average_score:.1f}/100

## Score Distribution

"""
        for score_range, count in self.score_distribution.items():
            md += f"- **{score_range}:** {count} creatives\n"

        md += "\n## Common Violations\n\n"
        for violation, count in sorted(
            self.common_violations.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            md += f"- **{violation}:** {count} occurrences\n"

        return md
```

### 3.2 Other Data Models

```python
# src/meta/ad/reviewer/models/violation.py

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional

class Severity(Enum):
    """Violation severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ViolationCategory(Enum):
    """Violation categories."""
    BRAND_COMPLIANCE = "brand_compliance"
    CULTURE_FIT = "culture_fit"
    TECHNICAL_QUALITY = "technical_quality"
    COMPLIANCE = "compliance"
    GOAL_ALIGNMENT = "goal_alignment"
    REGIONAL_PATTERN = "regional_pattern"

@dataclass
class Violation:
    """A single rule violation."""
    category: str
    severity: Severity
    description: str
    detected_elements: list
    confidence: float
    requirement: str
    recommendation: Optional[str] = None
    locale_specific: bool = False
    auto_reject: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'category': self.category,
            'severity': self.severity.value,
            'description': self.description,
            'detected_elements': self.detected_elements,
            'confidence': self.confidence,
            'requirement': self.requirement,
            'recommendation': self.recommendation,
            'locale_specific': self.locale_specific,
            'auto_reject': self.auto_reject
        }


# src/meta/ad/reviewer/models/criteria_config.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class CriteriaConfig:
    """Review criteria configuration."""
    # Scoring weights
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        'brand': 0.30,
        'culture': 0.25,
        'technical': 0.20,
        'compliance': 0.15,
        'goal': 0.05,
        'regional': 0.05
    })

    # Minimum thresholds
    min_overall_score: float = 70.0
    min_brand_score: float = 80.0
    min_culture_score: float = 70.0
    min_technical_score: float = 75.0
    min_compliance_score: float = 90.0

    # Required components
    require_brand_compliance: bool = True
    require_culture_fit: bool = True
    require_technical_quality: bool = True
    require_compliance: bool = True

    # Brand thresholds
    logo_compliance_required: bool = True
    logo_min_area_percent: float = 5.0
    logo_max_area_percent: float = 10.0
    color_tolerance_delta_e: float = 5.0
    min_text_contrast_ratio: float = 4.5

    # Culture thresholds
    min_culture_score: float = 70.0
    risk_categories: List[str] = field(default_factory=lambda: [
        "religious_misfit",
        "fraudulent_content",
        "violence_weapons",
        "adult_content",
        "hate_speech",
        "political_content",
        "self_harm",
        "substance_abuse",
        "child_safety",
        "privacy_violations"
    ])

    # Technical thresholds
    min_resolution: tuple = (1080, 1080)
    max_blur_score: float = 0.3
    max_file_size_mb: float = 50.0

    # Compliance
    strict_mode: bool = False
    auto_reject_critical: bool = True
```

---

## 4. Concrete Checker Implementation

### 4.1 Brand Guidelines Checker

```python
# src/meta/ad/reviewer/checkers/brand_guidelines_checker.py

from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from .base_checker import BaseChecker
from ..models.violation import Violation, Severity
from ..models.criteria_config import CriteriaConfig
from ..analyzers.logo_analyzer import LogoAnalyzer
from ..analyzers.color_analyzer import ColorAnalyzer
from ..utils.color_utils import calculate_delta_e

logger = logging.getLogger(__name__)


class BrandGuidelinesChecker(BaseChecker):
    """
    Checks compliance with brand guidelines.

    Validates:
    - Logo presence, placement, size, and quality
    - Color palette compliance
    - Typography and text overlays
    - Visual style and composition
    """

    def __init__(
        self,
        brand_guidelines: Dict[str, Any],
        config: CriteriaConfig
    ):
        super().__init__(config)
        self.brand_guidelines = brand_guidelines
        self.logo_analyzer = LogoAnalyzer(brand_guidelines)
        self.color_analyzer = ColorAnalyzer()

    def check(
        self,
        image_path: str,
        image_analysis: Dict[str, Any]
    ) -> 'BrandComplianceResult':
        """
        Perform complete brand guidelines check.

        Returns:
            BrandComplianceResult with detailed analysis
        """
        logger.debug("Starting brand guidelines check")

        violations = []
        checks = {}
        scores = []

        # 1. Logo compliance
        logo_result = self._check_logo(image_path, image_analysis)
        checks['logo'] = logo_result
        scores.append(logo_result['score'])
        violations.extend(logo_result.get('violations', []))

        # 2. Color compliance
        color_result = self._check_colors(image_analysis)
        checks['colors'] = color_result
        scores.append(color_result['score'])
        violations.extend(color_result.get('violations', []))

        # 3. Typography compliance
        typo_result = self._check_typography(image_analysis)
        checks['typography'] = typo_result
        scores.append(typo_result['score'])
        violations.extend(typo_result.get('violations', []))

        # 4. Visual style compliance
        style_result = self._check_style(image_analysis)
        checks['style'] = style_result
        scores.append(style_result['score'])
        violations.extend(style_result.get('violations', []))

        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 0
        compliant = overall_score >= self.config.min_brand_score and len(
            [v for v in violations if v.get('auto_reject')]
        ) == 0

        logger.debug(
            f"Brand check complete: Score={overall_score:.1f}, "
            f"Compliant={compliant}, Violations={len(violations)}"
        )

        return BrandComplianceResult(
            score=overall_score,
            compliant=compliant,
            checks=checks,
            violations=[v.to_dict() if isinstance(v, Violation) else v for v in violations],
            guideline_coverage=self._calculate_guideline_coverage(checks),
            logo_detected=checks['logo'].get('detected', False),
            logo_compliance=checks['logo'],
            color_compliance=checks['colors'],
            typography_compliance=checks['typography']
        )

    def _check_logo(
        self,
        image_path: str,
        image_analysis: Dict
    ) -> Dict[str, Any]:
        """Check logo compliance."""
        violations = []

        # Analyze logo
        logo_analysis = self.logo_analyzer.analyze(image_path, image_analysis)

        if not logo_analysis.get('present'):
            violations.append(Violation(
                category='logo',
                severity=Severity.CRITICAL,
                description='Logo not detected in image',
                detected_elements=[],
                confidence=0.95,
                requirement='Logo must be present and clearly visible',
                recommendation='Add brand logo to creative, ideally in top-right or top-left position',
                auto_reject=True
            ))
            return {
                'detected': False,
                'compliant': False,
                'score': 0.0,
                'violations': violations
            }

        # Check logo size
        logo_area = logo_analysis.get('area_percentage', 0)
        min_size = self.config.logo_min_area_percent
        max_size = self.config.logo_max_area_percent

        if logo_area < min_size:
            violations.append(Violation(
                category='logo_size',
                severity=Severity.HIGH,
                description=f'Logo too small: {logo_area:.1f}% (minimum: {min_size}%)',
                detected_elements=['logo'],
                confidence=0.90,
                requirement=f'Logo must occupy {min_size}-{max_size}% of image area',
                recommendation=f'Increase logo size to at least {min_size}% of image width',
                auto_reject=False
            ))

        if logo_area > max_size:
            violations.append(Violation(
                category='logo_size',
                severity=Severity.MEDIUM,
                description=f'Logo too large: {logo_area:.1f}% (maximum: {max_size}%)',
                detected_elements=['logo'],
                confidence=0.90,
                requirement=f'Logo must occupy {min_size}-{max_size}% of image area',
                recommendation=f'Reduce logo size to {max_size}% or less',
                auto_reject=False
            ))

        # Check logo placement
        placement = logo_analysis.get('placement', 'unknown')
        allowed_placements = self.brand_guidelines.get(
            'visual_identity',
            {}
        ).get('logo', {}).get('placement', {}).get('allowed_positions', ['top_right', 'top_left'])

        if placement not in allowed_placements:
            violations.append(Violation(
                category='logo_placement',
                severity=Severity.MEDIUM,
                description=f'Logo placement not optimal: {placement}',
                detected_elements=['logo', f'placement:{placement}'],
                confidence=0.85,
                requirement=f'Logo should be in: {", ".join(allowed_placements)}',
                recommendation=f'Move logo to one of: {", ".join(allowed_placements)}',
                auto_reject=False
            ))

        # Check logo quality (blur, sharpness)
        blur_score = logo_analysis.get('blur_score', 0)
        if blur_score > 0.1:
            violations.append(Violation(
                category='logo_quality',
                severity=Severity.HIGH,
                description=f'Logo appears blurry (blur score: {blur_score:.2f})',
                detected_elements=['logo', 'blur'],
                confidence=blur_score,
                requirement='Logo must be sharp and clear',
                recommendation='Use higher resolution logo or check image export quality',
                auto_reject=False
            ))

        # Calculate score
        base_score = 100.0
        for violation in violations:
            if violation.severity == Severity.CRITICAL:
                base_score -= 50.0
            elif violation.severity == Severity.HIGH:
                base_score -= 25.0
            elif violation.severity == Severity.MEDIUM:
                base_score -= 15.0
            elif violation.severity == Severity.LOW:
                base_score -= 5.0

        return {
            'detected': True,
            'compliant': len([v for v in violations if v.severity in [Severity.CRITICAL, Severity.HIGH]]) == 0,
            'score': max(0.0, base_score),
            'violations': violations,
            'details': logo_analysis
        }

    def _check_colors(self, image_analysis: Dict) -> Dict[str, Any]:
        """Check color palette compliance."""
        violations = []

        # Extract detected colors
        detected_colors = image_analysis.get('colors', {}).get('dominant', [])

        # Get brand colors
        brand_colors = self.brand_guidelines.get(
            'visual_identity', {}
        ).get('colors', {}).get('primary', [])

        if not detected_colors:
            return {
                'compliant': False,
                'score': 0.0,
                'violations': [Violation(
                    category='color_analysis',
                    severity=Severity.MEDIUM,
                    description='Could not extract color palette',
                    detected_elements=[],
                    confidence=0.5,
                    requirement='Brand colors should be dominant',
                    recommendation='Ensure image quality is sufficient for color analysis'
                )],
                'details': {}
            }

        # Check each detected color against brand palette
        max_delta_e = 0.0
        matched_colors = []
        unmatched_colors = []

        for detected in detected_colors:
            detected_hex = detected.get('hex', '')
            closest_match = None
            min_delta = 100.0

            for brand_color_spec in brand_colors:
                brand_hex = brand_color_spec.get('hex', '')
                delta_e = calculate_delta_e(detected_hex, brand_hex)

                if delta_e < min_delta:
                    min_delta = delta_e
                    closest_match = brand_color_spec

            max_delta_e = max(max_delta_e, min_delta)

            tolerance = self.config.color_tolerance_delta_e
            if min_delta <= tolerance:
                matched_colors.append({
                    'detected': detected_hex,
                    'brand': closest_match.get('hex', ''),
                    'delta_e': min_delta
                })
            else:
                unmatched_colors.append({
                    'detected': detected_hex,
                    'closest_brand': closest_match.get('hex', '') if closest_match else None,
                    'delta_e': min_delta
                })

        # Check if brand colors are dominant
        brand_color_dominance = len(matched_colors) / len(detected_colors) if detected_colors else 0
        min_dominance = 0.4  # At least 40% of colors should be brand colors

        if brand_color_dominance < min_dominance:
            violations.append(Violation(
                category='color_dominance',
                severity=Severity.HIGH,
                description=f'Brand color dominance too low: {brand_color_dominance:.1%} (minimum: {min_dominance:.1%})',
                detected_elements=[c['detected'] for c in unmatched_colors],
                confidence=0.85,
                requirement=f'At least {min_dominance:.1%} of colors should match brand palette',
                recommendation='Increase use of primary brand colors',
                auto_reject=False
            ))

        # Check for forbidden colors
        forbidden_colors = self.brand_guidelines.get(
            'visual_identity', {}
        ).get('colors', {}).get('forbidden', [])

        for detected in detected_colors:
            detected_hex = detected.get('hex', '')
            for forbidden in forbidden_colors:
                forbidden_hex = forbidden.get('hex', '')
                if calculate_delta_e(detected_hex, forbidden_hex) < 5.0:
                    violations.append(Violation(
                        category='forbidden_color',
                        severity=Severity.HIGH,
                        description=f'Forbidden color detected: {detected_hex} (similar to {forbidden.get("name", "forbidden")})',
                        detected_elements=[detected_hex],
                        confidence=0.90,
                        requirement=f'Color {forbidden.get("name")} is forbidden',
                        recommendation=f'Remove or replace {detected_hex} with approved brand color',
                        auto_reject=True
                    ))

        # Calculate score
        base_score = 100.0
        base_score -= max_delta_e * 2  # Penalize for color deviation
        if brand_color_dominance < min_dominance:
            base_score -= (min_dominance - brand_color_dominance) * 50

        for violation in violations:
            if violation.severity == Severity.CRITICAL:
                base_score -= 40.0
            elif violation.severity == Severity.HIGH:
                base_score -= 20.0

        return {
            'compliant': len([v for v in violations if v.auto_reject]) == 0,
            'score': max(0.0, min(100.0, base_score)),
            'violations': violations,
            'details': {
                'matched_colors': matched_colors,
                'unmatched_colors': unmatched_colors,
                'brand_dominance': brand_color_dominance,
                'max_delta_e': max_delta_e
            }
        }

    def _check_typography(self, image_analysis: Dict) -> Dict[str, Any]:
        """Check typography compliance."""
        violations = []

        # Get text overlay analysis
        text_overlays = image_analysis.get('text_overlays', {})
        text_regions = text_overlays.get('regions', [])

        if not text_regions:
            # No text - this is OK for some creatives
            return {
                'compliant': True,
                'score': 100.0,
                'violations': [],
                'details': {'note': 'No text overlays detected'}
            }

        # Check each text region
        all_compliant = True
        for region in text_regions:
            # Check contrast ratio
            contrast_ratio = region.get('contrast_ratio', 0)
            min_contrast = self.config.min_text_contrast_ratio

            if contrast_ratio < min_contrast:
                violations.append(Violation(
                    category='text_contrast',
                    severity=Severity.MEDIUM,
                    description=f'Low contrast text detected (ratio: {contrast_ratio:.1f}, minimum: {min_contrast:.1f})',
                    detected_elements=[region.get('text', 'text')],
                    confidence=0.85,
                    requirement=f'Text must have contrast ratio ≥ {min_contrast}:1 (WCAG AA)',
                    recommendation='Increase text contrast by using lighter text on darker background or vice versa',
                    auto_reject=False
                ))
                all_compliant = False

            # Check for cut-off text
            is_cut_off = region.get('is_cut_off', False)
            if is_cut_off:
                violations.append(Violation(
                    category='text_integrity',
                    severity=Severity.HIGH,
                    description='Text appears cut off at image edge',
                    detected_elements=[region.get('text', 'text')],
                    confidence=0.90,
                    requirement='All text must be complete and visible',
                    recommendation='Adjust text placement or image size to ensure all text is visible',
                    auto_reject=False
                ))
                all_compliant = False

            # Check font weight
            font_weight = region.get('font_weight', 400)
            min_weight = self.brand_guidelines.get(
                'visual_identity', {}
            ).get('typography', {}).get('text_overlays', {}).get('min_font_weight', 400)

            if font_weight < min_weight:
                violations.append(Violation(
                    category='text_weight',
                    severity=Severity.LOW,
                    description=f'Text weight too light: {font_weight} (minimum: {min_weight})',
                    detected_elements=[region.get('text', 'text')],
                    confidence=0.80,
                    requirement=f'Font weight should be ≥ {min_weight}',
                    recommendation='Use bolder font weight for better readability',
                    auto_reject=False
                ))
                all_compliant = False

        # Check text ratio
        text_ratio = text_overlays.get('text_area_ratio', 0)
        max_ratio = 0.20  # Max 20% text area

        if text_ratio > max_ratio:
            violations.append(Violation(
                category='text_ratio',
                severity=Severity.MEDIUM,
                description=f'Text overlay ratio too high: {text_ratio:.1%} (maximum: {max_ratio:.1%})',
                detected_elements=['text_overlays'],
                confidence=0.90,
                requirement=f'Text should not exceed {max_ratio:.1%} of image area',
                recommendation='Reduce amount of text or simplify design',
                auto_reject=False
            ))

        # Calculate score
        score = 100.0 - len(violations) * 10

        return {
            'compliant': all_compliant,
            'score': max(0.0, score),
            'violations': violations,
            'details': {
                'text_region_count': len(text_regions),
                'text_ratio': text_ratio
            }
        }

    def _check_style(self, image_analysis: Dict) -> Dict[str, Any]:
        """Check visual style compliance."""
        violations = []

        # Get composition style
        composition = image_analysis.get('composition', {})
        detected_style = composition.get('style', 'unknown')

        # Get brand's preferred style
        preferred_styles = self.brand_guidelines.get(
            'visual_identity', {}
        ).get('style', {}).get('aesthetic', {}).get('primary', None)

        # Check for forbidden image treatments
        detected_treatments = composition.get('treatments', [])
        forbidden_treatments = self.brand_guidelines.get(
            'visual_identity', {}
        ).get('style', {}).get('image_treatments', {}).get('forbidden', [])

        for treatment in detected_treatments:
            if any(ft in treatment for ft in forbidden_treatments):
                violations.append(Violation(
                    category='image_treatment',
                    severity=Severity.MEDIUM,
                    description=f'Forbidden image treatment detected: {treatment}',
                    detected_elements=[treatment],
                    confidence=0.85,
                    requirement=f'Treatment not allowed: {treatment}',
                    recommendation=f'Remove {treatment} effect',
                    auto_reject=False
                ))

        # Calculate score
        score = 100.0 - len(violations) * 15

        return {
            'compliant': len(violations) == 0,
            'score': max(0.0, score),
            'violations': violations,
            'details': {
                'detected_style': detected_style,
                'preferred_style': preferred_styles
            }
        }

    def _calculate_guideline_coverage(self, checks: Dict) -> float:
        """Calculate how many guideline areas were checked."""
        total_areas = len(checks)
        passed_areas = sum(1 for c in checks.values() if c.get('compliant', False))
        return (passed_areas / total_areas * 100) if total_areas > 0 else 0
```

### 4.2 Culture Risk Checker

```python
# src/meta/ad/reviewer/checkers/culture_risk_checker.py

from typing import Dict, List, Any
import logging

from .base_checker import BaseChecker
from ..models.violation import Violation, Severity
from ..models.criteria_config import CriteriaConfig

logger = logging.getLogger(__name__)


class CultureRiskChecker(BaseChecker):
    """
    Checks for cultural risks and sensitivities.

    Validates against:
    - Religious misfit
    - Fraudulent content
    - Violence and weapons
    - Adult content
    - Hate speech
    - Political sensitivity
    - Self-harm
    - Substance abuse
    - Child safety
    - Privacy violations
    """

    def __init__(
        self,
        risk_profile: Dict[str, Any],
        region: str,
        config: CriteriaConfig
    ):
        super().__init__(config)
        self.risk_profile = risk_profile
        self.region = region
        self.enabled_categories = self._get_enabled_categories()

    def check(
        self,
        image_path: str,
        image_analysis: Dict[str, Any]
    ) -> 'CultureFitResult':
        """
        Perform comprehensive culture risk check.

        Returns:
            CultureFitResult with detailed analysis
        """
        logger.debug(f"Starting culture risk check for region: {self.region}")

        all_risks = []
        scores = []

        # Check each enabled risk category
        for category in self.enabled_categories:
            category_result = self._check_category(category, image_analysis)
            all_risks.extend(category_result['risks'])
            scores.append(category_result['score'])

        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 100.0

        # Determine compliance
        critical_risks = [r for r in all_risks if r.get('severity') == 'critical']
        compliant = len(critical_risks) == 0 and overall_score >= self.config.min_culture_score

        # Analyze diversity
        diversity_analysis = self._analyze_diversity(image_analysis)

        # Get market-specific issues
        market_issues = self._group_by_market(all_risks)

        logger.debug(
            f"Culture risk check complete: Score={overall_score:.1f}, "
            f"Risks={len(all_risks)}, Compliant={compliant}"
        )

        return CultureFitResult(
            score=overall_score,
            compliant=compliant,
            detected_risks=[r.to_dict() if isinstance(r, Violation) else r for r in all_risks],
            cultural_contexts=[],
            diversity_analysis=diversity_analysis,
            market_specific_issues=market_issues,
            risk_categories=[c for c in self.enabled_categories]
        )

    def _get_enabled_categories(self) -> List[str]:
        """Get list of enabled risk categories."""
        all_categories = [
            'religious_misfit',
            'fraudulent_content',
            'violence_weapons',
            'adult_content',
            'hate_speech',
            'political_content',
            'self_harm',
            'substance_abuse',
            'child_safety',
            'privacy_violations'
        ]

        enabled = []
        for category in all_categories:
            cat_config = self.risk_profile.get('categories', {}).get(category, {})
            if cat_config.get('enabled', False):
                enabled.append(category)

        return enabled

    def _check_category(self, category: str, image_analysis: Dict) -> Dict:
        """Check a specific risk category."""
        risks = []
        score = 100.0

        cat_config = self.risk_profile.get('categories', {}).get(category, {})
        severity = Severity(cat_config.get('severity', 'medium'))

        # Visual checks
        if 'visual_checks' in cat_config:
            visual_risks = self._check_visual_risks(category, cat_config, image_analysis)
            risks.extend(visual_risks)

        # Text checks
        if 'text_checks' in cat_config:
            text_content = image_analysis.get('text_overlays', {}).get('full_text', '')
            text_risks = self._check_text_risks(category, cat_config, text_content)
            risks.extend(text_risks)

        # Calculate score
        for risk in risks:
            if risk.severity == Severity.CRITICAL:
                score -= 40.0
            elif risk.severity == Severity.HIGH:
                score -= 25.0
            elif risk.severity == Severity.MEDIUM:
                score -= 15.0
            elif risk.severity == Severity.LOW:
                score -= 5.0

        return {
            'category': category,
            'risks': risks,
            'score': max(0.0, score)
        }

    def _check_visual_risks(
        self,
        category: str,
        cat_config: Dict,
        image_analysis: Dict
    ) -> List[Violation]:
        """Check for visual risks."""
        violations = []
        visual_checks = cat_config.get('visual_checks', {})

        # Check forbidden symbols
        if 'forbidden_symbols' in visual_checks:
            forbidden = visual_checks['forbidden_symbols']
            detected_symbols = image_analysis.get('symbols', [])

            for symbol_list in forbidden:
                if isinstance(symbol_list, dict):
                    # Locale-specific
                    symbols = symbol_list.get('symbols', [])
                else:
                    symbols = symbol_list

                for detected in detected_symbols:
                    if any(s.lower() in detected.lower() for s in symbols):
                        violations.append(Violation(
                            category=category,
                            severity=Severity.CRITICAL,
                            description=f'Forbidden symbol detected: {detected}',
                            detected_elements=[detected],
                            confidence=0.90,
                            requirement='This symbol is not permitted',
                            recommendation='Remove forbidden symbol immediately',
                            auto_reject=True
                        ))

        # Check for violence/weapons
        if category == 'violence_weapons':
            weapons = image_analysis.get('weapons_detected', [])
            if weapons:
                violations.append(Violation(
                    category='violence_weapons',
                    severity=Severity.HIGH,
                    description=f'Weapons detected: {", ".join(weapons)}',
                    detected_elements=weapons,
                    confidence=0.85,
                    requirement='Weapons are generally not allowed',
                    recommendation='Remove all weapons from creative',
                    auto_reject=False
                ))

            violence = image_analysis.get('violence_indicators', [])
            if violence:
                violations.append(Violation(
                    category='violence_weapons',
                    severity=Severity.CRITICAL,
                    description=f'Violence detected: {", ".join(violence)}',
                    detected_elements=violence,
                    confidence=0.90,
                    requirement='Violent content is prohibited',
                    recommendation='Remove violent imagery',
                    auto_reject=True
                ))

        # Check for adult content
        if category == 'adult_content':
            nudity = image_analysis.get('nudity_detected', False)
            if nudity:
                violations.append(Violation(
                    category='adult_content',
                    severity=Severity.CRITICAL,
                    description='Nudity or sexual content detected',
                    detected_elements=['nudity'],
                    confidence=0.95,
                    requirement='Adult content is prohibited',
                    recommendation='Remove all nudity or sexualized content',
                    auto_reject=True
                ))

        return violations

    def _check_text_risks(
        self,
        category: str,
        cat_config: Dict,
        text_content: str
    ) -> List[Violation]:
        """Check for text-based risks."""
        import re
        violations = []

        if not text_content:
            return violations

        text_checks = cat_config.get('text_checks', {})

        # Check forbidden phrases
        if 'forbidden_phrases' in text_checks:
            forbidden = text_checks['forbidden_phrases']

            for pattern in forbidden:
                if isinstance(pattern, dict):
                    pattern_str = pattern.get('pattern', pattern)
                else:
                    pattern_str = pattern

                try:
                    if re.search(pattern_str, text_content, re.IGNORECASE):
                        violations.append(Violation(
                            category=category,
                            severity=Severity.MEDIUM,
                            description=f'Flagged phrase detected matching pattern: {pattern_str}',
                            detected_elements=[text_content],
                            confidence=0.85,
                            requirement='Avoid sensitive language patterns',
                            recommendation='Rewrite text to avoid flagged patterns',
                            auto_reject=False
                        ))
                except re.error:
                    logger.warning(f"Invalid regex pattern: {pattern_str}")

        return violations

    def _analyze_diversity(self, image_analysis: Dict) -> Dict[str, Any]:
        """Analyze diversity and representation."""
        people = image_analysis.get('people', [])

        if not people:
            return {
                'has_people': False,
                'note': 'No people detected'
            }

        # Analyze demographics
        demographics = {
            'count': len(people),
            'genders': {},
            'age_ranges': {},
            'ethnicities': {}
        }

        for person in people:
            for attr in ['gender', 'age_range', 'ethnicity']:
                value = person.get(attr, 'unknown')
                if attr == 'genders':
                    demographics['genders'][value] = demographics['genders'].get(value, 0) + 1
                elif attr == 'age_ranges':
                    demographics['age_ranges'][value] = demographics['age_ranges'].get(value, 0) + 1
                elif attr == 'ethnicities':
                    demographics['ethnicities'][value] = demographics['ethnicities'].get(value, 0) + 1

        # Check for diversity
        has_diverse_representation = (
            len(demographics['ethnicities']) > 1 or
            len(demographics['age_ranges']) > 1
        )

        return {
            'has_people': True,
            'demographics': demographics,
            'diverse_representation': has_diverse_representation
        }

    def _group_by_market(self, risks: List) -> Dict[str, List[Dict]]:
        """Group risks by market/region."""
        # For now, just group by locale-specific flag
        market_specific = {
            self.region: [r.to_dict() if isinstance(r, Violation) else r for r in risks]
        }

        return market_specific
```

---

This is a comprehensive, concrete design. Should I continue with more concrete implementations for the other components?
