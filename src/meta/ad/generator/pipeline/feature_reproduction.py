"""
Feature Reproduction & Validation System.
Provides traceability from formula → prompt → image → validation:
- Track which features from visual_recommendation.json are actually in the prompt
- Map each prompt to its generated image(s)
- Validate features are present in generated images
This creates an audit trail for generation and validates feature reproduction.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .feature_validator import FeatureValidator


logger = logging.getLogger(__name__)
class FeatureStatus(str, Enum):
    """Status of a feature in the generation pipeline."""
    REQUESTED = "requested"  # In formula, not yet processed
    INCLUDED = "included"  # Confirmed in prompt
    EXCLUDED = "excluded"  # Not included in prompt
@dataclass
class FeatureIdentity:
    """Feature identity information."""
    feature_name: str
    feature_type: str  # entrance, headroom, synergy, negative
    expected_value: str
@dataclass
class FeatureTracking:
    """Feature tracking through the pipeline."""
    in_prompt: bool = False
    prompt_mention: Optional[str] = None  # Exact text mentioning this feature
    status: FeatureStatus = FeatureStatus.REQUESTED
@dataclass
class FeatureRecord:
    """Record of a single feature through the pipeline."""
    identity: FeatureIdentity
    tracking: Optional[FeatureTracking] = None
    roas_impact: float = 0.0
    notes: Optional[str] = None
    def __post_init__(self):
        """Initialize optional sub-configs with defaults."""
        if self.tracking is None:
            self.tracking = FeatureTracking()
    @property
    def feature_name(self) -> str:
        """Get feature name from identity."""
        return self.identity.feature_name
    @property
    def feature_type(self) -> str:
        """Get feature type from identity."""
        return self.identity.feature_type
    @property
    def expected_value(self) -> str:
        """Get expected value from identity."""
        return self.identity.expected_value
    @property
    def in_prompt(self) -> bool:
        """Get in_prompt from tracking."""
        return self.tracking.in_prompt
    @in_prompt.setter
    def in_prompt(self, value: bool):
        """Set in_prompt in tracking."""
        self.tracking.in_prompt = value
    @property
    def prompt_mention(self) -> Optional[str]:
        """Get prompt_mention from tracking."""
        return self.tracking.prompt_mention
    @prompt_mention.setter
    def prompt_mention(self, value: Optional[str]):
        """Set prompt_mention in tracking."""
        self.tracking.prompt_mention = value
    @property
    def status(self) -> FeatureStatus:
        """Get status from tracking."""
        return self.tracking.status
    @status.setter
    def status(self, value: FeatureStatus):
        """Set status in tracking."""
        self.tracking.status = value
@dataclass
class ImageRecordPaths:
    """Paths for image record."""
    image_path: str
    upscaled_path: Optional[str] = None
    watermarked_path: Optional[str] = None
@dataclass
class ImageRecordConfig:
    """Configuration for creating an image record."""
    prompt_record: "PromptRecord"
    paths: ImageRecordPaths
    variation_index: int
    generation_model: str
    validate_features: bool = True
    @property
    def image_path(self) -> str:
        """Get image path from paths."""
        return self.paths.image_path
    @property
    def upscaled_path(self) -> Optional[str]:
        """Get upscaled path from paths."""
        return self.paths.upscaled_path
    @property
    def watermarked_path(self) -> Optional[str]:
        """Get watermarked path from paths."""
        return self.paths.watermarked_path
@dataclass
class ImageRecord:
    """Record of a single generated image."""
    image_id: str
    prompt_id: str
    paths: ImageRecordPaths
    variation_index: int
    generated_at: str
    generation_model: str
    feature_validation: Optional[List[Dict[str, Any]]] = (
        None  # NEW: Validation results
    )
    @property
    def image_path(self) -> str:
        """Get image path from paths."""
        return self.paths.image_path
    @property
    def upscaled_path(self) -> Optional[str]:
        """Get upscaled path from paths."""
        return self.paths.upscaled_path
    @property
    def watermarked_path(self) -> Optional[str]:
        """Get watermarked path from paths."""
        return self.paths.watermarked_path
@dataclass
class PromptMetadata:
    """Metadata for a prompt record."""
    prompt_id: str
    prompt_type: str  # "llm_enhanced" or "technical"
    generated_at: str
    formula_version: str
@dataclass
class PromptRecord:
    """Record of a generated prompt and its images."""
    metadata: PromptMetadata
    prompt_text: str
    product_context: Dict[str, Any]
    features_requested: List[FeatureRecord] = field(default_factory=list)
    features_in_prompt: List[str] = field(default_factory=list)
    images: List[ImageRecord] = field(default_factory=list)
    @property
    def prompt_id(self) -> str:
        """Get prompt ID from metadata."""
        return self.metadata.prompt_id
    @property
    def prompt_type(self) -> str:
        """Get prompt type from metadata."""
        return self.metadata.prompt_type
    @property
    def generated_at(self) -> str:
        """Get generated at from metadata."""
        return self.metadata.generated_at
    @property
    def formula_version(self) -> str:
        """Get formula version from metadata."""
        return self.metadata.formula_version
@dataclass
class SessionMetadata:
    """Session metadata information."""
    product_name: str = ""
    market: str = "US"
    formula_path: str = ""
    output_dir: str = ""
@dataclass
class SessionStats:
    """Session statistics."""
    total_images: int = 0
    total_cost_estimate: float = 0.0
@dataclass
class GenerationSession:
    """Complete session record for batch generation."""
    session_id: str
    started_at: str
    metadata: Optional[SessionMetadata] = None
    stats: Optional[SessionStats] = None
    completed_at: Optional[str] = None
    prompts: List[PromptRecord] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        """Initialize optional sub-configs with defaults."""
        if self.metadata is None:
            self.metadata = SessionMetadata()
        if self.stats is None:
            self.stats = SessionStats()
    def get_product_name(self) -> str:
        """Get product name from metadata."""
        return self.metadata.product_name
    def set_product_name(self, value: str) -> None:
        """Set product name in metadata."""
        self.metadata.product_name = value
    def get_market(self) -> str:
        """Get market from metadata."""
        return self.metadata.market
    def set_market(self, value: str) -> None:
        """Set market in metadata."""
        self.metadata.market = value
    def get_formula_path(self) -> str:
        """Get formula path from metadata."""
        return self.metadata.formula_path
    def set_formula_path(self, value: str) -> None:
        """Set formula path in metadata."""
        self.metadata.formula_path = value
    def get_output_dir(self) -> str:
        """Get output dir from metadata."""
        return self.metadata.output_dir
    def set_output_dir(self, value: str) -> None:
        """Set output dir in metadata."""
        self.metadata.output_dir = value
    def get_total_images(self) -> int:
        """Get total images from stats."""
        return self.stats.total_images
    def set_total_images(self, value: int) -> None:
        """Set total images in stats."""
        self.stats.total_images = value
    def get_total_cost_estimate(self) -> float:
        """Get total cost estimate from stats."""
        return self.stats.total_cost_estimate
    def set_total_cost_estimate(self, value: float) -> None:
        """Set total cost estimate in stats."""
        self.stats.total_cost_estimate = value
class FeatureReproductionTracker:
    """
    Tracks and validates feature reproduction through the generation pipeline.
    Creates an audit trail: Formula → Prompt → Image → Validation
    """
    def __init__(
        self,
        output_dir: Path,
        validate_images: bool = True,
        vision_api_key: Optional[str] = None,
    ):
        """
        Initialize tracker.
        Args:
            output_dir: Directory to save reproduction data
            validate_images: Whether to validate features in generated images
            vision_api_key: Anthropic API key for vision validation
        """
        self.output_dir = Path(output_dir)
        self.reproduction_dir = self.output_dir / "tracking"
        self.reproduction_dir.mkdir(parents=True, exist_ok=True)
        self.validate_images = validate_images
        # Initialize image validator if enabled
        self.image_validator = None
        if validate_images:
            try:
                self.image_validator = FeatureValidator(api_key=vision_api_key)
                logger.info("Image feature validation enabled")
            except ValueError as e:
                logger.warning("Failed to initialize FeatureValidator: %s", e)
                logger.warning("Image validation disabled")
                self.validate_images = False
        self.session = GenerationSession(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            started_at=datetime.now().isoformat(),
            metadata=SessionMetadata(output_dir=str(self.output_dir)),
        )
        self._prompt_counter = 0
        self._image_counter = 0
    def set_session_info(
        self,
        product_name: str,
        market: str,
        formula_path: str,
        config: Dict[str, Any],
    ) -> None:
        """Set session-level information."""
        self.session.set_product_name(product_name)
        self.session.set_market(market)
        self.session.set_formula_path(formula_path)
        self.session.config = config
    def create_prompt_record(
        self,
        prompt_text: str,
        prompt_type: str,
        product_context: Dict[str, Any],
        visual_recommendation: Dict[str, Any],
    ) -> PromptRecord:
        """
        Create a prompt record and analyze which features are included.
        Args:
            prompt_text: The generated prompt
            prompt_type: "llm_enhanced" or "technical"
            product_context: Product context dict
            visual_recommendation: The visual formula used
        Returns:
            PromptRecord with feature analysis
        """
        self._prompt_counter += 1
        prompt_id = f"prompt_{self._prompt_counter:03d}"
        # Extract features from formula
        feature_records = self._extract_features_from_formula(
            visual_recommendation
        )
        # Analyze which features appear in the prompt
        features_in_prompt = []
        for record in feature_records:
            in_prompt, mention = self._check_feature_in_prompt(
                record.feature_name,
                record.expected_value,
                prompt_text,
            )
            record.in_prompt = in_prompt
            record.prompt_mention = mention
            if in_prompt:
                record.status = FeatureStatus.INCLUDED
                features_in_prompt.append(record.feature_name)
            else:
                record.status = FeatureStatus.EXCLUDED
        prompt_record = PromptRecord(
            metadata=PromptMetadata(
                prompt_id=prompt_id,
                prompt_type=prompt_type,
                generated_at=datetime.now().isoformat(),
                formula_version=visual_recommendation.get(
                    "generated_date", "unknown"
                ),
            ),
            prompt_text=prompt_text,
            product_context=product_context,
            features_requested=feature_records,
            features_in_prompt=features_in_prompt,
        )
        self.session.prompts.append(prompt_record)
        # Log feature inclusion stats
        total = len(feature_records)
        included = len(features_in_prompt)
        logger.info(
            "Prompt %s: %d/%d features included (%.0f%%)",
            prompt_id,
            included,
            total,
            (included / total * 100) if total > 0 else 0,
        )
        return prompt_record
    def add_image_record(
        self,
        config: ImageRecordConfig,
    ) -> ImageRecord:
        """
        Add an image record to a prompt and optionally validate features.
        Args:
            config: ImageRecordConfig object with all image record parameters
        Returns:
            ImageRecord
        """
        self._image_counter += 1
        image_id = f"img_{self._image_counter:03d}"
        # Validate features if enabled
        validation_results = self._validate_image_features(
            image_id, config.image_path, config.prompt_record,
            config.validate_features
        )
        # Create image record
        paths = ImageRecordPaths(
            image_path=config.image_path,
            upscaled_path=config.upscaled_path,
            watermarked_path=config.watermarked_path,
        )
        image_record = ImageRecord(
            image_id=image_id,
            prompt_id=config.prompt_record.prompt_id,
            paths=paths,
            variation_index=config.variation_index,
            generated_at=datetime.now().isoformat(),
            generation_model=config.generation_model,
            feature_validation=validation_results,
        )
        config.prompt_record.images.append(image_record)
        self.session.set_total_images(self.session.get_total_images() + 1)
        return image_record
    def _validate_image_features(
        self,
        image_id: str,
        image_path: str,
        prompt_record: PromptRecord,
        validate_features: bool,
    ) -> Optional[List[Dict[str, Any]]]:
        """Validate features in image and return results."""
        if not validate_features or not self.image_validator:
            return None
        # Prepare expected features
        expected_features = [
            {
                "feature_name": f.feature_name,
                "expected_value": f.expected_value,
            }
            for f in prompt_record.features_requested
        ]
        if not expected_features:
            return None
        # Perform validation
        return self._perform_feature_validation(
            image_id, image_path, expected_features, prompt_record.prompt_text
        )
    def _perform_feature_validation(
        self,
        image_id: str,
        image_path: str,
        expected_features: List[Dict],
        prompt_text: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Perform feature validation and log results."""
        try:
            logger.info("Validating features in image %s...", image_id)
            validation_results_list = (
                self.image_validator.validate_features_in_image(
                    image_path=image_path,
                    expected_features=expected_features,
                    prompt_text=prompt_text,
                )
            )
            # Convert to dict for JSON serialization
            validation_results = [
                {
                    "feature_name": r.feature_name,
                    "expected_value": r.expected_value,
                    "detected_value": r.detected_value,
                    "is_present": r.is_present,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "image_region": r.image_region,
                }
                for r in validation_results_list
            ]
            # Log and warn if needed
            self._log_validation_results(image_id, validation_results)
            return validation_results
        except (ValueError, RuntimeError, OSError, IOError) as exc:
            logger.exception(
                "Failed to validate features in image %s: %s", image_id, exc
            )
            return None
    def _log_validation_results(
        self, image_id: str, validation_results: List[Dict[str, Any]]
    ) -> None:
        """Log validation results and warn if recovery rate is low."""
        present_count = sum(1 for r in validation_results if r["is_present"])
        total_count = len(validation_results)
        logger.info(
            "Image %s: %d/%d features visually confirmed (%.0f%%)",
            image_id,
            present_count,
            total_count,
            (present_count / total_count * 100) if total_count > 0 else 0,
        )
        # Warn if low recovery rate
        recovery_rate = present_count / total_count if total_count > 0 else 0
        if recovery_rate < 0.80:
            logger.warning(
                "WARNING: Low feature recovery rate (%.0f%%) in image %s. "
                "Generated image may not match recommendations.",
                recovery_rate * 100,
                image_id,
            )
    def _extract_features_from_formula(
        self,
        visual_recommendation: Dict[str, Any],
    ) -> List[FeatureRecord]:
        """Extract all features from formula into records."""
        records = []
        # Entrance features
        for feat in visual_recommendation.get("entrance_features", []):
            records.append(
                FeatureRecord(
                    identity=FeatureIdentity(
                        feature_name=feat.get("feature_name", ""),
                        feature_type="entrance",
                        expected_value=feat.get("feature_value", ""),
                    ),
                    roas_impact=feat.get("avg_roas", 0),
                )
            )
        # Headroom features
        for feat in visual_recommendation.get("headroom_features", []):
            records.append(
                FeatureRecord(
                    identity=FeatureIdentity(
                        feature_name=feat.get("feature_name", ""),
                        feature_type="headroom",
                        expected_value=feat.get("feature_value", ""),
                    ),
                    roas_impact=feat.get("avg_roas", 0),
                )
            )
        # Synergy pairs (record both features)
        for pair in visual_recommendation.get("synergy_pairs", []):
            records.append(
                FeatureRecord(
                    identity=FeatureIdentity(
                        feature_name=pair.get("feature1_name", ""),
                        feature_type="synergy",
                        expected_value=pair.get("feature1_value", ""),
                    ),
                    roas_impact=pair.get("predicted_roas", 0),
                    notes=f"Synergy with {pair.get('feature2_name')}",
                )
            )
            records.append(
                FeatureRecord(
                    identity=FeatureIdentity(
                        feature_name=pair.get("feature2_name", ""),
                        feature_type="synergy",
                        expected_value=pair.get("feature2_value", ""),
                    ),
                    roas_impact=pair.get("predicted_roas", 0),
                    notes=f"Synergy with {pair.get('feature1_name')}",
                )
            )
        # Negative guidance
        for neg_guidance in visual_recommendation.get("negative_guidance", []):
            records.append(
                FeatureRecord(
                    identity=FeatureIdentity(
                        feature_name=neg_guidance.get("feature_name", ""),
                        feature_type="negative",
                        expected_value=f"AVOID: {neg_guidance.get('feature_value', '')}",
                    ),
                    notes=neg_guidance.get("alternative_suggestion"),
                )
            )
        return records
    def validate_prompt_vs_formula(
        self,
        prompt: str,
        formula: Dict[str, Any],
        customer: str,
        min_coverage: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Validate that prompt covers features from formula.

        Args:
            prompt: Generated prompt text
            formula: Visual formula with entrance_features, headroom_features
            customer: Customer name for logging
            min_coverage: Minimum feature coverage threshold (default 95%)

        Returns:
            Dict with validation results:
            {
                "passed": bool,
                "coverage": float,
                "total_features": int,
                "covered_features": int,
                "missing_features": List[str],
                "critical_missing": List[str],
                "warnings": List[str]
            }
        """
        # Extract all features from formula
        feature_records = self._extract_features_from_formula(formula)
        total_features = len(feature_records)

        # Check coverage
        covered_features = []
        missing_features = []
        for record in feature_records:
            in_prompt, _ = self._check_feature_in_prompt(
                record.feature_name,
                record.expected_value,
                prompt
            )
            if in_prompt:
                covered_features.append(record.feature_name)
            else:
                missing_features.append(record.feature_name)

        coverage = len(covered_features) / total_features if total_features > 0 else 0.0
        passed = coverage >= min_coverage

        # Identify critical missing features (entrance > headroom)
        critical_missing = []
        for record in feature_records:
            if record.feature_type == "entrance" and record.feature_name in missing_features:
                critical_missing.append(record.feature_name)

        # Generate warnings
        warnings = []
        if not passed:
            warnings.append(
                f"Feature coverage {coverage*100:.0f}% below threshold {min_coverage*100:.0f}%"
            )
        if critical_missing:
            warnings.append(
                f"{len(critical_missing)} critical entrance features missing: {critical_missing}"
            )

        result = {
            "passed": passed,
            "coverage": coverage,
            "total_features": total_features,
            "covered_features": len(covered_features),
            "missing_features": missing_features,
            "critical_missing": critical_missing,
            "warnings": warnings,
            "customer": customer,
        }

        # Log results
        if passed:
            logger.info(
                "Prompt validation PASSED: %d/%d features covered (%.0f%%)",
                len(covered_features),
                total_features,
                coverage * 100,
            )
        else:
            logger.warning(
                "Prompt validation FAILED: %d/%d features covered (%.0f%%). Missing: %s",
                len(covered_features),
                total_features,
                coverage * 100,
                missing_features[:5],  # Log first 5
            )

        return result

    def _check_feature_in_prompt(
        self,
        feature_name: str,
        expected_value: str,
        prompt_text: str,
    ) -> tuple:
        """
        Check if a feature appears in the prompt.
        Returns (is_present, mention_text)
        """
        prompt_lower = prompt_text.lower()
        feature_lower = feature_name.lower().replace("_", " ")
        value_lower = expected_value.lower()
        # Check for feature name
        if feature_lower in prompt_lower:
            # Find the sentence containing the feature
            sentences = prompt_text.split(".")
            for sent in sentences:
                if feature_lower in sent.lower():
                    return True, sent.strip()
        # Check for value
        if value_lower in prompt_lower:
            sentences = prompt_text.split(".")
            for sent in sentences:
                if value_lower in sent.lower():
                    return True, sent.strip()
        # Check for semantic equivalents
        equivalents = {
            "warm-dominant": ["warm", "warm tones", "warm glow", "inviting"],
            "partial": ["partially visible", "partial view", "partially"],
            "dominant": ["prominently", "prominent", "dominant", "stands out"],
            "generous": ["generous space", "ample space", "breathing room"],
            "balanced": ["balanced", "harmonious", "even distribution"],
            "lifestyle context": ["lifestyle", "person using", "in use"],
        }
        for equiv_key, equiv_values in equivalents.items():
            if equiv_key not in value_lower:
                continue
            for equiv in equiv_values:
                if equiv not in prompt_lower:
                    continue
                sentences = prompt_text.split(".")
                for sent in sentences:
                    if equiv in sent.lower():
                        return True, sent.strip()
        return False, None
    def finalize_session(self) -> None:
        """Finalize the session and save all reproduction data."""
        self.session.completed_at = datetime.now().isoformat()
        # Estimate costs
        self.session.set_total_cost_estimate(self._estimate_costs())
        # Save main session file
        session_path = self.reproduction_dir / "session.json"
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(self._to_dict(self.session), f, indent=2, default=str)
        # Save individual prompt files for easy browsing
        for prompt in self.session.prompts:
            prompt_path = self.reproduction_dir / f"{prompt.prompt_id}.json"
            with open(prompt_path, "w", encoding="utf-8") as f:
                json.dump(self._to_dict(prompt), f, indent=2, default=str)
        # Generate summary report
        self._generate_summary_report()
        # Generate HTML viewer
        self._generate_html_viewer()
        logger.info("Reproduction data saved to: %s", self.reproduction_dir)
    def _estimate_costs(self) -> float:
        """Estimate total costs for the session."""
        costs = {
            "llm_prompt": 0.01,
            "image_generation": 0.02,
            "upscaling": 0.04,
        }
        total = 0.0
        total += len(self.session.prompts) * costs["llm_prompt"]
        total += self.session.get_total_images() * costs["image_generation"]
        # Count upscaled images
        upscaled = sum(
            1
            for p in self.session.prompts
            for img in p.images
            if img.upscaled_path
        )
        total += upscaled * costs["upscaling"]
        return total
    def _to_dict(self, obj: Any) -> Any:
        """Convert dataclass to dict recursively."""
        if hasattr(obj, "__dataclass_fields__"):
            return {k: self._to_dict(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [self._to_dict(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self._to_dict(v) for k, v in obj.items()}
        if isinstance(obj, Enum):
            return obj.value
        return obj
    def _generate_summary_report(self) -> None:
        """Generate a markdown summary report in results directory."""
        report_lines = self._build_report_header()
        # Add feature recovery section
        feature_stats = self._aggregate_feature_stats()
        self._add_feature_recovery_section(report_lines, feature_stats)
        # Add images section
        self._add_images_section(report_lines)
        # Save report
        self._save_report(report_lines)
    def _build_report_header(self) -> List[str]:
        """Build the header section of the report."""
        return [
            "# Generation Session Report",
            "",
            f"**Session ID:** {self.session.session_id}",
            f"**Product:** {self.session.get_product_name()}",
            f"**Market:** {self.session.get_market()}",
            f"**Started:** {self.session.started_at}",
            f"**Completed:** {self.session.completed_at}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Prompts | {len(self.session.prompts)} |",
            f"| Total Images | {self.session.get_total_images()} |",
            f"| Est. Cost | ${self.session.get_total_cost_estimate():.2f} |",
            "",
            "## Feature Recovery Summary",
            "",
        ]
    def _add_feature_recovery_section(
        self, report_lines: List[str], feature_stats: Dict
    ) -> None:
        """Add feature recovery table and summary to report."""
        if not feature_stats:
            return
        report_lines.extend(
            [
                "| Feature | Expected | In Prompt | In Image | Status |",
                "|---------|----------|-----------|----------|--------|",
            ]
        )
        for name, stats in feature_stats.items():
            report_lines.append(
                self._format_feature_row(name, stats)
            )
        # Add recovery rate summary
        self._add_recovery_rate_summary(report_lines, feature_stats)
    def _format_feature_row(self, name: str, stats: Dict) -> str:
        """Format a single feature table row."""
        in_prompt = "[OK]" if stats["in_prompt"] else "[NO]"
        in_image = "[OK]" if stats.get("in_image", False) else "[NO]"
        expected_val = self._truncate_expected_value(stats.get("expected_value", ""))
        status = self._determine_feature_status(stats)
        return f"| {name} | {expected_val} | {in_prompt} | {in_image} | {status} |"
    def _truncate_expected_value(self, value: Any) -> str:
        """Truncate expected value for table display."""
        if not value:
            return "—"
        value_str = str(value)
        if len(value_str) > 25:
            return value_str[:22] + "..."
        return value_str
    def _determine_feature_status(self, stats: Dict) -> str:
        """Determine feature reproduction status."""
        in_prompt = stats["in_prompt"]
        in_image = stats.get("in_image", False)
        if in_prompt and in_image:
            return "[OK] Reproduced"
        if in_prompt and not in_image:
            return "[WARN] Lost in generation"
        if not in_prompt and in_image:
            return "[INFO] Added (not requested)"
        return "[FAIL] Missing"
    def _add_recovery_rate_summary(
        self, report_lines: List[str], feature_stats: Dict
    ) -> None:
        """Add overall recovery rate summary to report."""
        total_features = len(feature_stats)
        if total_features == 0:
            return
        prompt_recovery = sum(1 for s in feature_stats.values() if s["in_prompt"])
        image_recovery = sum(1 for s in feature_stats.values() if s.get("in_image", False))
        prompt_rate = (prompt_recovery / total_features) * 100
        image_rate = (image_recovery / total_features) * 100
        report_lines.extend(
            [
                "",
                "**Overall Recovery:**",
                f"- Prompt: {prompt_recovery}/{total_features} features ({prompt_rate:.0f}%)",
                f"- Image: {image_recovery}/{total_features} features ({image_rate:.0f}%)",
                "",
            ]
        )
    def _add_images_section(self, report_lines: List[str]) -> None:
        """Add images section to report."""
        report_lines.extend(["## Images", ""])
        for prompt in self.session.prompts:
            report_lines.append(f"### {prompt.prompt_id}")
            report_lines.append("")
            for img in prompt.images:
                report_lines.append(f"- `{img.image_id}`: {img.image_path}")
                if img.feature_validation:
                    report_lines.append(
                        self._format_validation_summary(img.feature_validation)
                    )
            report_lines.append("")
    def _format_validation_summary(self, validation: List[Dict]) -> str:
        """Format feature validation summary for an image."""
        present_count = sum(1 for v in validation if v["is_present"])
        total_count = len(validation)
        rate = (present_count / total_count * 100) if total_count > 0 else 0
        return f"  - Feature validation: {present_count}/{total_count} ({rate:.0f}%)"
    def _save_report(self, report_lines: List[str]) -> None:
        """Save report to results directory."""
        results_dir = self.output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        report_path = results_dir / "report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        logger.info("Report saved to: %s", report_path)
    def _aggregate_feature_stats(self) -> Dict[str, Dict]:
        """Aggregate feature statistics across all prompts and images."""
        stats = {}
        for prompt in self.session.prompts:
            for feat in prompt.features_requested:
                if feat.feature_name not in stats:
                    stats[feat.feature_name] = {
                        "type": feat.feature_type,
                        "in_prompt": feat.in_prompt,
                        "status": feat.status.value,
                        "expected_value": feat.expected_value,
                        "in_image": False,  # NEW: Track if present in images
                    }
                # Check if feature is present in any image
                if self._is_feature_in_images(feat.feature_name, prompt.images):
                    stats[feat.feature_name]["in_image"] = True
        return stats
    def _is_feature_in_images(self, feature_name: str, images: List) -> bool:
        """Check if feature is present in any of the given images."""
        for img in images:
            if not img.feature_validation:
                continue
            for validation in img.feature_validation:
                if (
                    validation["feature_name"] == feature_name
                    and validation["is_present"]
                ):
                    return True
        return False
    def _generate_html_viewer(self) -> None:
        """Generate an interactive HTML viewer for the session."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generation Session: {self.session.session_id}</title>
    <style>
        :root {{
            --bg: #1a1a2e;
            --card: #16213e;
            --accent: #0f3460;
            --text: #e4e4e4;
            --success: #4ade80;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
        }}
        .header {{
            text-align: center;
            margin-bottom: 2rem;
        }}
        .header h1 {{ color: #fff; margin-bottom: 0.5rem; }}
        .header p {{ opacity: 0.7; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: var(--card);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
        }}
        .stat-card .value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--success);
        }}
        .stat-card .label {{ opacity: 0.7; font-size: 0.9rem; }}
        .section {{ margin-bottom: 2rem; }}
        .section h2 {{
            border-bottom: 2px solid var(--accent);
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }}
        .prompt-card {{
            background: var(--card);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }}
        .prompt-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 1rem;
        }}
        .prompt-text {{
            background: var(--accent);
            padding: 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            max-height: 150px;
            overflow-y: auto;
            margin-bottom: 1rem;
        }}
        .images-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
        }}
        .image-card {{
            background: var(--accent);
            border-radius: 8px;
            overflow: hidden;
        }}
        .image-card img {{
            width: 100%;
            height: 200px;
            object-fit: cover;
        }}
        .image-info {{
            padding: 0.75rem;
        }}
        .features-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        .features-table th, .features-table td {{
            padding: 0.5rem;
            text-align: left;
            border-bottom: 1px solid var(--accent);
        }}
        .features-table th {{ opacity: 0.7; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Generation Session</h1>
        <p>{self.session.get_product_name()} | {self.session.get_market()} Market</p>
        <p style="font-size: 0.8rem; opacity: 0.5;">
            {self.session.session_id}
        </p>
    </div>
    <div class="stats">
        <div class="stat-card">
            <div class="value">{len(self.session.prompts)}</div>
            <div class="label">Prompts</div>
        </div>
        <div class="stat-card">
            <div class="value">{self.session.get_total_images()}</div>
            <div class="label">Images</div>
        </div>
        <div class="stat-card">
            <div class="value">${self.session.get_total_cost_estimate():.2f}</div>
            <div class="label">Est. Cost</div>
        </div>
    </div>
    <div class="section">
        <h2>Prompts & Images</h2>
        {self._generate_prompt_cards_html()}
    </div>
    <div class="section">
        <h2>Feature Reproduction</h2>
        {self._generate_features_table_html()}
    </div>
</body>
</html>"""
        viewer_path = self.reproduction_dir / "viewer.html"
        viewer_path.write_text(html, encoding="utf-8")
    def _generate_prompt_cards_html(self) -> str:
        """Generate HTML for prompt cards."""
        cards = []
        for prompt in self.session.prompts:
            images_html = ""
            for img in prompt.images:
                # Use relative path from reproduction dir
                try:
                    img_rel = Path(img.image_path).relative_to(self.output_dir)
                    img_src = f"../{img_rel}"
                except ValueError:
                    img_src = img.image_path
                images_html += f"""
                <div class="image-card">
                    <img src="{img_src}" alt="{img.image_id}">
                    <div class="image-info">
                        <span style="font-size:0.8rem;opacity:0.7;">
                            {img.image_id}
                        </span>
                    </div>
                </div>
                """
            # Truncate prompt for display
            prompt_preview = prompt.prompt_text[:500]
            if len(prompt.prompt_text) > 500:
                prompt_preview += "..."
            feat_count = len(prompt.features_in_prompt)
            total_feat = len(prompt.features_requested)
            cards.append(
                f"""
            <div class="prompt-card">
                <div class="prompt-header">
                    <strong>{prompt.prompt_id}</strong>
                    <span>{feat_count}/{total_feat} features</span>
                </div>
                <div class="prompt-text">{prompt_preview}</div>
                <div class="images-grid">{images_html}</div>
            </div>
            """
            )
        return "\n".join(cards)
    def _generate_features_table_html(self) -> str:
        """Generate HTML for features table."""
        stats = self._aggregate_feature_stats()
        if not stats:
            return "<p>No feature data available.</p>"
        rows = []
        for name, data in stats.items():
            in_prompt = "[OK]" if data["in_prompt"] else "[NO]"
            rows.append(
                f"""
            <tr>
                <td>{name}</td>
                <td>{data['type']}</td>
                <td>{in_prompt}</td>
            </tr>
            """
            )
        return f"""
        <table class="features-table">
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Type</th>
                    <th>In Prompt</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        """
