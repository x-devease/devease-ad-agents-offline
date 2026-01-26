"""
Creative Pipeline: End-to-End Orchestrator for Recommendation-Based Generation.

This module orchestrates the complete pipeline:
1. Load Visual Recommendation from scorer repository
2. Generate prompts using template-based Mask Orchestrator
3. Generate images using enhanced prompts
4. Return generated images with metadata

This is the main integration point that ties together all components.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ad.generator.orchestrator.prompt_builder import PromptBuilder, PromptBuilderConfig
from src.ad.generator.pipeline.feature_reproduction import FeatureReproductionTracker
from src.ad.generator.pipeline.product_context import (
    ProductContextConfig,
    ProductIdentity,
    create_product_context,
)
from src.ad.generator.pipeline.recommendation_loader import RecommendationLoader
from src.ad.generator.pipeline.ad_recommender_adapter import (
    load_recommendations_as_visual_formula,
)


logger = logging.getLogger(__name__)


@dataclass
class RecommendationPaths:
    """Paths for recommendation loading."""

    recommendation_repo_path: Optional[Path] = None
    recommendation_path: Optional[Path] = None


@dataclass
class PromptBuilderModes:
    """PromptBuilder mode configuration (DEPRECATED - all modes are now professional quality)."""

    lean_mode: bool = False  # DEPRECATED
    v2_mode: bool = True  # DEPRECATED
    branch_name: Optional[str] = None
    step2_mode: bool = False

    # Feature flags (all default to True for professional quality)
    anti_hallucination_enhanced: bool = True
    camera_specs: bool = True
    material_textures: bool = True
    three_point_lighting: bool = True
    depth_of_field: bool = True
    post_processing: bool = True
    shadow_specification: bool = True
    frame_occupancy: bool = True
    visual_flow: bool = True
    color_accuracy_tolerance: bool = True


@dataclass
class CreativePipelineConfig:
    """Configuration for CreativePipeline initialization."""

    product_name: str
    product_context: Optional[Dict[str, Any]] = None
    output_dir: Optional[Path] = None
    recommendation_paths: Optional[RecommendationPaths] = None
    modes: Optional[PromptBuilderModes] = None

    def __post_init__(self):
        """Initialize optional sub-configs with defaults."""
        if self.recommendation_paths is None:
            self.recommendation_paths = RecommendationPaths()
        if self.modes is None:
            self.modes = PromptBuilderModes()

    @property
    def recommendation_repo_path(self) -> Optional[Path]:
        """Get recommendation repo path from paths."""
        return self.recommendation_paths.recommendation_repo_path

    @property
    def recommendation_path(self) -> Optional[Path]:
        """Get recommendation path from paths."""
        return self.recommendation_paths.recommendation_path

    @property
    def lean_mode(self) -> bool:
        """Get lean mode from modes."""
        return self.modes.lean_mode

    @property
    def v2_mode(self) -> bool:
        """Get v2 mode from modes."""
        return self.modes.v2_mode

    @property
    def branch_name(self) -> Optional[str]:
        """Get branch name from modes."""
        return self.modes.branch_name

    @property
    def step2_mode(self) -> bool:
        """Get step2 mode from modes."""
        return self.modes.step2_mode


class CreativePipeline:
    """
    End-to-end pipeline for recommendation-based creative generation.

    This orchestrates:
    - Visual recommendation loading from scorer repository
    - Template-based prompt generation using PromptBuilder
    - Image generation

    Usage:
        pipeline = CreativePipeline(CreativePipelineConfig(product_name="Power Station"))
        results = pipeline.run(
            source_image_path="product.jpg",
            num_variations=3,
        )
    """

    def __init__(self, config: CreativePipelineConfig):
        """
        Initialize CreativePipeline.

        Args:
            config: CreativePipelineConfig object with all pipeline parameters
        """
        # Store config reference (1 attribute)
        self._config = config

        # Output setup (computed property, not stored as separate attribute)
        self._setup_output_directory()

        # Product context (1 attribute)
        self.product_context = self._build_product_context(config)

        # Components (3 attributes)
        self.recommendation_loader = self._create_recommendation_loader(config)
        self.prompt_builder = self._create_prompt_builder(config, self.product_context)
        self.tracker = FeatureReproductionTracker(output_dir=self.output_dir)

        # Session state (1 attribute)
        self.visual_recommendation = None

        # Set up tracker session (formula_path may not exist yet)
        formula_path = (
            str(self.recommendation_loader.recommendation_path)
            if hasattr(self.recommendation_loader, "recommendation_path")
            else "unknown"
        )
        self.tracker.set_session_info(
            product_name=config.product_name,
            market=self.product_context.get("market", "US"),
            formula_path=formula_path,
            config={},
        )

        logger.info(
            "CreativePipeline initialized: product=%s, mode=%s, branch=%s",
            config.product_name,
            "v2" if config.v2_mode else "lean" if config.lean_mode else "standard",
            config.branch_name,
        )

    @property
    def product_name(self) -> str:
        """Get product name from config."""
        return self._config.product_name

    def _setup_output_directory(self) -> None:
        """Set up and create output directory with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        product_slug = (
            self._config.product_name.lower().replace(" ", "_").replace("-", "_")[:30]
        )

        if self._config.output_dir:
            output_path = Path(self._config.output_dir)
            if not any(c.isdigit() for c in output_path.name):
                self._output_dir = output_path.parent / f"{output_path.name}_{timestamp}"
            else:
                self._output_dir = output_path
        else:
            self._output_dir = Path(f"./output/{product_slug}_{timestamp}")

        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", self._output_dir)

    @property
    def output_dir(self) -> Path:
        """Get output directory."""
        return self._output_dir

    @property
    def _recommendation_repo(self) -> Path:
        """Get recommendation repository path."""
        return (
            Path(self._config.recommendation_repo_path)
            if self._config.recommendation_repo_path
            else Path(__file__).resolve().parents[4] / "devease-creative-scorer-offline"
        )

    def _build_product_context(
        self, config: CreativePipelineConfig
    ) -> Dict[str, Any]:
        """Build product context from config."""
        if config.product_context:
            return {
                "product_name": config.product_name,
                **config.product_context,
            }
        return create_product_context(
            ProductContextConfig(
                identity=ProductIdentity(product_name=config.product_name)
            )
        )

    def _create_recommendation_loader(
        self, config: CreativePipelineConfig
    ) -> RecommendationLoader:
        """Create recommendation loader."""
        return RecommendationLoader(
            scorer_repo_path=self._recommendation_repo,
            recommendation_path=config.recommendation_path,
        )

    def _create_prompt_builder(
        self, config: CreativePipelineConfig, product_context: Dict[str, Any]
    ) -> PromptBuilder:
        """Create prompt builder."""
        return PromptBuilder(
            PromptBuilderConfig(
                branch_name=config.branch_name,
                step2_mode=config.step2_mode,
                product_context=product_context,
                anti_hallucination_enhanced=config.modes.anti_hallucination_enhanced,
                camera_specs=config.modes.camera_specs,
                material_textures=config.modes.material_textures,
                three_point_lighting=config.modes.three_point_lighting,
                depth_of_field=config.modes.depth_of_field,
                post_processing=config.modes.post_processing,
                shadow_specification=config.modes.shadow_specification,
                frame_occupancy=config.modes.frame_occupancy,
                visual_flow=config.modes.visual_flow,
                color_accuracy_tolerance=config.modes.color_accuracy_tolerance,
            )
        )

    def load_recommendation(self) -> Dict[str, Any]:
        """
        Load visual recommendation from scorer repository or ad/recommender.
        
        Supports two formats:
        1. Creative scorer format (entrance_features/headroom_features)
        2. Ad recommender format (recommendations array) - auto-converts
        """
        rec_path = self.recommendation_loader.recommendation_path
        
        # Check if this is an ad/recommender format file
        if rec_path and rec_path.exists():
            # Try to detect format by checking file extension and content
            if rec_path.suffix in [".md", ".json"]:
                try:
                    # Try loading as ad/recommender format first
                    visual_formula = load_recommendations_as_visual_formula(
                        rec_path,
                        min_confidence="medium",
                        min_high_performer_pct=0.25,
                    )
                    logger.info(
                        "Loaded recommendations from ad/recommender format: %s",
                        rec_path
                    )
                    return visual_formula
                except (ValueError, KeyError, FileNotFoundError) as e:
                    logger.debug(
                        "Not ad/recommender format (will try scorer format): %s", e
                    )
                    # Fall through to scorer format
        
        # Default: try scorer format
        try:
            return self.recommendation_loader.load()
        except FileNotFoundError:
            # If scorer format not found, try ad/recommender format from default location
            # Try MD first (primary format), then JSON as fallback
            base_dir = Path("config/ad/recommender")
            # Try to infer customer/platform from product_name or use defaults
            customer = self._config.product_name.lower().replace(" ", "_")
            platform = "meta"
            
            md_path = base_dir / customer / platform / "recommendations.md"
            json_path = base_dir / customer / platform / "recommendations.json"
            
            if md_path.exists():
                logger.info(
                    "Scorer format not found, using ad/recommender MD: %s",
                    md_path
                )
                return load_recommendations_as_visual_formula(
                    md_path,
                    min_confidence="medium",
                    min_high_performer_pct=0.25,
                )
            elif json_path.exists():
                logger.info(
                    "Scorer format not found, using ad/recommender JSON (fallback): %s",
                    json_path
                )
                return load_recommendations_as_visual_formula(
                    json_path,
                    min_confidence="medium",
                    min_high_performer_pct=0.25,
                )
            raise FileNotFoundError(
                f"Neither scorer format nor ad/recommender format found. "
                f"Tried: {self.recommendation_loader.recommendation_path}, "
                f"{md_path}, {json_path}"
            )

    def generate_prompt(
        self,
        visual_recommendation: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate prompt from recommendation using PromptBuilder.

        Args:
            visual_recommendation: Visual recommendation dict (loads if not provided)

        Returns:
            Visual prompt for image generation
        """
        if visual_recommendation is None:
            visual_recommendation = self.load_recommendation()
        # Store for reference
        self.visual_recommendation = visual_recommendation
        # Generate prompt using PromptBuilder (deterministic, template-based)
        prompt = self.prompt_builder.generate_p0_prompt(
            visual_formula=visual_recommendation,
            product_context=self.product_context,
        )

        return prompt

    def run(
        self,
        source_image_path: str,
        num_variations: int = 1,
        save_prompts: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run the complete creative generation pipeline.

        Args:
            source_image_path: Path to source product image
            num_variations: Number of creative variations to generate
            save_prompts: If True, save generated prompts to output directory

        Returns:
            List of generation results with metadata
        """
        logger.info("Running creative pipeline for %s", self.product_name)
        logger.info("Source image: %s", source_image_path)
        # Load recommendation
        visual_recommendation = self.load_recommendation()
        logger.info(
            "Loaded recommendation with %d entrance_features, %d headroom_features",
            len(visual_recommendation.get("entrance_features", {})),
            len(visual_recommendation.get("headroom_features", {})),
        )
        # Generate prompts
        results = []
        customer = self._config.product_name.lower().replace(" ", "_")
        for i in range(num_variations):
            logger.info("Generating variation %d/%d", i + 1, num_variations)
            prompt = self.generate_prompt(visual_recommendation)
            logger.info("Generated prompt: %s...", prompt[:100])

            # Validate prompt coverage
            validation_result = self.tracker.validate_prompt_vs_formula(
                prompt=prompt,
                formula=visual_recommendation,
                customer=customer,
                min_coverage=0.95
            )
            logger.info(
                "Prompt validation: %s (%.0f%% coverage, %d/%d features)",
                "PASSED" if validation_result["passed"] else "FAILED",
                validation_result["coverage"] * 100,
                validation_result["covered_features"],
                validation_result["total_features"]
            )

            # Track prompt
            if save_prompts:
                # Save to config/ad/recommender/{customer}/{platform}/prompts.md
                platform = "meta"
                prompts_dir = Path("config/ad/recommender") / customer / platform
                prompts_dir.mkdir(parents=True, exist_ok=True)
                prompt_file = prompts_dir / "prompts.md"
                prompt_file.write_text(prompt, encoding="utf-8")
                logger.debug("Saved prompt to %s", prompt_file)

            results.append(
                {
                    "variation": i + 1,
                    "prompt": prompt,
                    "source_image_path": source_image_path,
                    "recommendation": visual_recommendation,
                    "validation": validation_result,
                }
            )

        logger.info(
            "Pipeline complete: %d variations generated", num_variations
        )
        return results
