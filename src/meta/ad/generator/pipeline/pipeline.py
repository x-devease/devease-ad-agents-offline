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
import re
from typing import Any, Dict, List, Optional

from src.meta.ad.generator.core.paths import Paths
from src.meta.ad.generator.orchestrator.prompt_builder import PromptBuilder, PromptBuilderConfig
from src.meta.ad.generator.pipeline.feature_reproduction import FeatureReproductionTracker
from src.meta.ad.generator.pipeline.product_context import (
    ProductContextConfig,
    ProductIdentity,
    create_product_context,
)
from src.meta.ad.generator.pipeline.recommendation_loader import RecommendationLoader
from src.meta.ad.generator.pipeline.ad_recommender_adapter import (
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

        # Initialize path manager with customer/platform/date
        customer = config.product_name.lower().replace(" ", "_")
        platform = "meta"  # Default platform, can be overridden in config
        date = datetime.now().strftime("%Y-%m-%d")
        self.paths = Paths(customer=customer, platform=platform, date=date)

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
        """Set up output directory using organized path structure."""
        # Use the organized results path from Paths class
        self._output_dir = self.paths.generated_output()
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
            from datetime import datetime
            base_dir = Path("config/ad/recommender")
            # Try to infer customer/platform from product_name or use defaults
            customer = self._config.product_name.lower().replace(" ", "_")
            platform = "meta"
            date = datetime.now().strftime("%Y-%m-%d")

            md_path = base_dir / customer / platform / date / "recommendations.md"
            json_path = base_dir / customer / platform / date / "recommendations.json"

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

    def _extract_prompt_features(self, prompt: str) -> Dict[str, str]:
        """
        Extract key features from generated prompt for descriptive naming.

        Args:
            prompt: Generated prompt string

        Returns:
            Dict with extracted features: position, color, style, atmosphere
        """
        features = {
            "position": "center",
            "color": "neutral",
            "style": "professional",
            "atmosphere": "studio"
        }

        # Extract product position
        position_patterns = [
            r"(bottom-right|bottom left|top-right|top left|bottom-right|bottom_right)",
            r"(center|middle)",
            r"(left|right)side",
        ]
        for pattern in position_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                pos = match.group(1).replace("-", "_").replace(" ", "_")
                features["position"] = pos
                break

        # Extract color balance/atmosphere
        color_patterns = [
            r"(cool-dominant|cool-dominant|warm-dominant|warm_dominant)",
            r"(cool|warm|neutral)",
            r"(high-key|low-key)",
        ]
        for pattern in color_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                color = match.group(1).replace("-", "_")
                features["color"] = color
                break

        # Extract lighting style
        if "three-point" in prompt.lower() or "professional studio" in prompt.lower():
            features["style"] = "studio"
        elif "natural light" in prompt.lower() or "daylight" in prompt.lower():
            features["style"] = "natural"
        elif "dramatic" in prompt.lower():
            features["style"] = "dramatic"

        # Extract atmosphere/environment
        if "home environment" in prompt.lower():
            features["atmosphere"] = "home"
        elif "isolated" in prompt.lower() and "clean background" in prompt.lower():
            features["atmosphere"] = "isolated"
        elif "lifestyle" in prompt.lower():
            features["atmosphere"] = "lifestyle"

        return features

    def _generate_prompt_filename(
        self,
        variation_index: int,
        prompt: str
    ) -> str:
        """
        Generate descriptive filename for prompt based on extracted features.

        Args:
            variation_index: Variation number (0-indexed)
            prompt: Generated prompt string

        Returns:
            Descriptive filename like: bottom_right_cool_studio_001.md
        """
        features = self._extract_prompt_features(prompt)

        # Build filename from key features
        parts = [
            features["position"],
            features["color"],
            features["style"]
        ]

        # Add atmosphere if different from style
        if features["atmosphere"] != features["style"]:
            parts.append(features["atmosphere"])

        # Join and add variation number
        base_name = "_".join(parts)
        filename = f"{base_name}_{variation_index+1:03d}.md"

        # Sanitize filename (remove special chars, but keep .md extension)
        # First sanitize the base name only
        base_name_sanitized = re.sub(r'[^\w\-_]', '_', base_name)
        # Then add extension
        filename = f"{base_name_sanitized}_{variation_index+1:03d}.md"

        if len(filename) > 80:
            # Truncate if too long
            parts = parts[:2]  # Keep position and color only
            base_name = "_".join(parts)
            base_name_sanitized = re.sub(r'[^\w\-_]', '_', base_name)
            filename = f"{base_name_sanitized}_{variation_index+1:03d}.md"

        return filename


    def _sample_dimension_combinations(
        self,
        visual_formula: Dict[str, Any],
        num_combinations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate strategic dimension combinations from recommendations.

        Strategy:
        1. Extract dimensions with multiple High/Medium confidence options
        2. Create baseline (all top ROAS)
        3. Sample combinations by varying 1-2 dimensions at a time

        Args:
            visual_formula: Original visual formula with recommendations
            num_combinations: Number of combinations to generate

        Returns:
            List of modified visual formulas for each combination
        """
        import copy

        # Extract all features from entrance and headroom (they are lists)
        entrance = visual_formula.get("entrance_features", [])
        headroom = visual_formula.get("headroom_features", [])

        # Group by dimension (original_feature name)
        # Track multiple options per dimension
        dimension_options = {}  # {dimension: [feature_dict, ...]}

        # Process entrance features
        for feature in entrance:
            if isinstance(feature, dict):
                orig_feature = feature.get("_original_feature", feature.get("feature_name", ""))
                value = feature.get("feature_value", "")
                roas = feature.get("avg_roas", 0)
                
                if orig_feature and value:
                    if orig_feature not in dimension_options:
                        dimension_options[orig_feature] = []
                    dimension_options[orig_feature].append({
                        "feature_dict": feature,
                        "value": value,
                        "roas": roas,
                        "source": "entrance"
                    })

        # Process headroom features
        for feature in headroom:
            if isinstance(feature, dict):
                orig_feature = feature.get("_original_feature", feature.get("feature_name", ""))
                value = feature.get("feature_value", "")
                roas = feature.get("avg_roas", 0)
                
                if orig_feature and value:
                    if orig_feature not in dimension_options:
                        dimension_options[orig_feature] = []
                    dimension_options[orig_feature].append({
                        "feature_dict": feature,
                        "value": value,
                        "roas": roas,
                        "source": "headroom"
                    })

        # Filter to dimensions with multiple options
        multi_option_dimensions = {
            dim: options
            for dim, options in dimension_options.items()
            if len(options) > 1
        }

        if not multi_option_dimensions:
            logger.info("No dimensions with multiple options found")
            return [visual_formula]

        logger.info(
            "Found %d dimensions with multiple options: %s",
            len(multi_option_dimensions),
            list(multi_option_dimensions.keys())
        )

        # Sort dimensions by ROAS range (prioritize high-impact dimensions)
        dimension_impact = []
        for dim, options in multi_option_dimensions.items():
            roas_values = [opt["roas"] for opt in options]
            impact = max(roas_values) - min(roas_values)
            dimension_impact.append((dim, impact, options))

        # Sort by impact (highest first)
        dimension_impact.sort(key=lambda x: x[1], reverse=True)

        # Generate combinations
        combinations = []

        # Combination 0: Baseline (original)
        baseline = copy.deepcopy(visual_formula)
        combinations.append(baseline)

        # Remaining combinations: vary 1-2 dimensions at a time
        for i in range(1, num_combinations):
            combo = copy.deepcopy(visual_formula)

            # Select which dimensions to vary
            # Vary 1-2 dimensions per combination
            num_to_vary = 1 if i % 2 == 1 else 2

            # Select dimensions with highest impact
            for j in range(min(num_to_vary, len(dimension_impact))):
                dim_idx = (i + j) % len(dimension_impact)
                dim_name, _, options = dimension_impact[dim_idx]

                # Pick a non-top option (alternative)
                # Sort by ROAS descending
                sorted_options = sorted(options, key=lambda x: x["roas"], reverse=True)
                
                if len(sorted_options) > 1:
                    # Pick index i (cycles through options)
                    option_idx = (i + j) % len(sorted_options)
                    selected_option = sorted_options[option_idx]

                    # Find and replace in combo
                    combo_list = combo.get(f"{selected_option['source']}_features", [])
                    
                    for k, feat in enumerate(combo_list):
                        if isinstance(feat, dict):
                            orig_feat = feat.get("_original_feature", feat.get("feature_name", ""))
                            if orig_feat == dim_name:
                                # Replace with selected option
                                combo_list[k] = copy.deepcopy(selected_option["feature_dict"])
                                logger.info(
                                    "Combination %d: Varying %s to %s (ROAS: %.2f)",
                                    i, dim_name, selected_option["value"], selected_option["roas"]
                                )
                                break

            combinations.append(combo)

        logger.info("Generated %d dimension combinations", len(combinations))
        return combinations

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
                # Save to config/ad/generator/prompts/{customer}/{platform}/{date}/bottom_right_cool_studio_001.md
                prompts_dir = self.paths.prompts_output()
                prompts_dir.mkdir(parents=True, exist_ok=True)
                prompt_filename = self._generate_prompt_filename(i, prompt)
                prompt_file = prompts_dir / prompt_filename
                prompt_file.write_text(prompt, encoding="utf-8")
                logger.info("Saved prompt to %s", prompt_file)

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
