"""
Template-Driven Pipeline Orchestrator: End-to-End Ad Generation.

Integrates all template system stages:
1. Product Preprocessor (trim transparency, perspective detection)
2. Template Selector (psychology-driven auto-selection)
3. Background Generator (perspective-aware NanoBanana Pro)
4. Physics Compositor (dual-layer shadows, light matching, light wrap)
5. Smart Typer (psychology-adaptive text overlay)

Input:
- Master blueprint from Ad Miner (config/ad/{customer}/master_blueprint.yaml)
- Campaign content (config/ad/{customer}/campaign_content.yaml)
- Product image (config/ad/{customer}/products/{product}/product_raw.png)

Output:
- Generated ad candidates (results/{customer}/{platform}/ad_generator/generated/)

Author: Ad System
Date: 2026-01-30
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

import yaml
from PIL import Image

from .paths import GeneratorPaths
from .product_preprocessor import (
    ProductPreprocessor,
    PreprocessorResult,
    PerspectiveType,
)
from .template_selector import (
    PsychologyTemplateSelector,
    TemplateSpec,
)
from .smart_typer import (
    CampaignContent,
    SmartTyper,
    TextRenderResult,
)
from .background_generator import (
    NanoBackgroundGenerator,
    GeneratedBackground,
)
from .physics_compositor import (
    PhysicsCompositor,
    CompositingConfig,
    CompositingResult,
)


logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """Pipeline execution stages."""

    PREPROCESSING = "preprocessing"
    TEMPLATE_SELECTION = "template_selection"
    BACKGROUND_GENERATION = "background_generation"
    COMPOSITING = "compositing"
    TEXT_OVERLAY = "text_overlay"
    COMPLETE = "complete"


@dataclass
class PipelineConfig:
    """
    Template-Driven Pipeline configuration.

    Attributes:
        customer: Customer name
        platform: Platform name (facebook, tiktok, instagram)
        product: Product name
        num_variants: Number of ad variants to generate
        generate_backgrounds: Whether to generate new backgrounds
        save_intermediates: Whether to save intermediate outputs
        config_dir: Optional custom config directory (for testing)
        output_dir: Optional custom output directory (for testing)
    """
    customer: str
    platform: str
    product: str
    num_variants: int = 1
    generate_backgrounds: bool = True
    save_intermediates: bool = True
    config_dir: Optional[Path] = None
    output_dir: Optional[Path] = None


@dataclass
class PipelineResult:
    """
    Result from template-driven pipeline execution.

    Attributes:
        generated_images: List of generated ad images
        metadata: Pipeline execution metadata
        intermediates: Intermediate outputs (if save_intermediates=True)
    """
    generated_images: List[Tuple[Image.Image, Dict[str, Any]]]  # (image, metadata)
    metadata: Dict[str, Any]
    intermediates: Dict[str, Any] = field(default_factory=dict)


class TemplatePipeline:
    """
    Template-Driven Ad Generator Pipeline: Psychology-Based Compositing.

    Usage:
        pipeline = TemplatePipeline(
            customer="moprobo",
            platform="meta",
            product="Power Station"
        )
        results = pipeline.run(
            product_image_path="product.png",
            num_variants=3
        )
        # Save outputs
        for i, (image, metadata) in enumerate(results.generated_images):
            image.save(f"ad_candidate_{i+1}.png")
    """

    def __init__(
        self,
        config: PipelineConfig,
    ):
        """
        Initialize template-driven pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Initialize path manager with custom directories if provided
        self.paths = GeneratorPaths(
            customer=config.customer,
            platform=config.platform,
            config_dir=config.config_dir or Path("config"),
            output_dir=config.output_dir or Path("results"),
        )

        # Ensure directories exist
        self.paths.ensure_directories(product=config.product)

        # Initialize components (lazy initialization)
        self._preprocessor: Optional[ProductPreprocessor] = None
        self._template_selector: Optional[PsychologyTemplateSelector] = None
        self._background_generator: Optional[NanoBackgroundGenerator] = None
        self._compositor: Optional[PhysicsCompositor] = None
        self._smart_typer: Optional[SmartTyper] = None

        # Cache loaded data
        self._master_blueprint: Optional[Dict[str, Any]] = None
        self._campaign_content: Optional[CampaignContent] = None
        self._preprocessor_result: Optional[PreprocessorResult] = None
        self._selected_template: Optional[TemplateSpec] = None

        logger.info(
            f"Initialized Template Pipeline: {config.customer}/{config.platform}/{config.product}"
        )

    @property
    def preprocessor(self) -> ProductPreprocessor:
        """Lazy initialization of product preprocessor."""
        if self._preprocessor is None:
            self._preprocessor = ProductPreprocessor()
        return self._preprocessor

    @property
    def template_selector(self) -> PsychologyTemplateSelector:
        """Lazy initialization of template selector."""
        if self._template_selector is None:
            # Create loader with correct config path
            from .template_selector import TemplateLoader

            # Construct full config path using config_dir
            config_path = (
                self.paths.config_dir /
                self.config.customer /
                self.config.platform /
                "config.yaml"
            )

            loader = TemplateLoader(
                customer=self.config.customer,
                platform=self.config.platform,
                config_path=config_path
            )
            self._template_selector = PsychologyTemplateSelector(loader=loader)
        return self._template_selector

    @property
    def background_generator(self) -> NanoBackgroundGenerator:
        """Lazy initialization of background generator."""
        if self._background_generator is None:
            self._background_generator = NanoBackgroundGenerator()
        return self._background_generator

    @property
    def compositor(self) -> PhysicsCompositor:
        """Lazy initialization of physics compositor."""
        if self._compositor is None:
            self._compositor = PhysicsCompositor()
        return self._compositor

    @property
    def smart_typer(self) -> SmartTyper:
        """Lazy initialization of Smart Typer."""
        if self._smart_typer is None:
            self._smart_typer = SmartTyper()
        return self._smart_typer

    def run(
        self,
        product_image_path: Optional[Path] = None,
        background_image_path: Optional[Path] = None,
        num_variants: Optional[int] = None,
    ) -> PipelineResult:
        """
        Run complete template-driven pipeline.

        Args:
            product_image_path: Path to product image (uses default if None)
            background_image_path: Optional path to existing background (skips generation)
            num_variants: Number of variants (uses config if None)

        Returns:
            PipelineResult with generated images
        """
        num_variants = num_variants or self.config.num_variants

        logger.info(f"Starting Template Pipeline: {num_variants} variants")

        # Load configuration
        self._load_configs()

        # Stage 1: Preprocess product
        self._stage_preprocessing(product_image_path)

        # Stage 2: Select template based on psychology
        self._stage_template_selection()

        # Stage 3: Generate or load backgrounds
        backgrounds = self._stage_background_generation(
            background_image_path=background_image_path,
            num_variants=num_variants,
        )

        # Stage 4 & 5: Composite and render text for each variant
        generated_images = []
        for i, background in enumerate(backgrounds):
            logger.info(f"Generating variant {i + 1}/{num_variants}")

            # Stage 4: Physics-aware compositing
            composited = self._stage_compositing(background)

            # Stage 5: Smart text overlay
            final_image = self._stage_text_overlay(composited)

            # Collect result
            metadata = {
                "variant": i + 1,
                "template_id": self._selected_template.template_id,
                "psychology_driver": self._selected_template.psychology_driver,
                "perspective": self._preprocessor_result.perspective.value,
            }

            generated_images.append((final_image, metadata))

            # Save output
            self._save_output(final_image, i + 1, metadata)

        logger.info(f"Template Pipeline complete: {len(generated_images)} variants generated")

        return PipelineResult(
            generated_images=generated_images,
            metadata={
                "customer": self.config.customer,
                "platform": self.config.platform,
                "product": self.config.product,
                "num_variants": len(generated_images),
                "template_id": self._selected_template.template_id if self._selected_template else None,
                "psychology_driver": self._selected_template.psychology_driver if self._selected_template else None,
                "timestamp": datetime.now().isoformat(),
            },
            intermediates={},
        )

    def _load_configs(self):
        """Load master blueprint and campaign content."""
        # Load master blueprint
        blueprint_path = self.paths.get_blueprint_path()
        if not blueprint_path.exists():
            raise FileNotFoundError(f"Master blueprint not found: {blueprint_path}")

        with open(blueprint_path, "r", encoding="utf-8") as f:
            self._master_blueprint = yaml.safe_load(f)

        logger.info(f"Loaded master blueprint: {blueprint_path}")

        # Load campaign content
        campaign_path = self.paths.get_campaign_content_path()
        if not campaign_path.exists():
            logger.warning(f"Campaign content not found: {campaign_path}, using defaults")
            self._campaign_content = CampaignContent(
                headline="Your Headline Here",
                sub_text="Sub Text",
                cta_text="Shop Now",
                brand_color="#000000",
            )
        else:
            with open(campaign_path, "r", encoding="utf-8") as f:
                campaign_data = yaml.safe_load(f)
            self._campaign_content = CampaignContent.from_dict(campaign_data)

        logger.info(f"Loaded campaign content: {campaign_path}")

    def _stage_preprocessing(self, product_image_path: Optional[Path]):
        """Stage 1: Preprocess product (trim, perspective, mask)."""
        logger.info("Stage 1: Product Preprocessing (trim, perspective, mask)")

        # Use default product path if not provided
        if product_image_path is None:
            product_image_path = self.paths.get_product_input_path(self.config.product)

        self._preprocessor_result = self.preprocessor.process(product_image_path)

        logger.info(
            f"  Trimmed: {self._preprocessor_result.trimmed_image.size}, "
            f"Perspective: {self._preprocessor_result.perspective.value}"
        )

    def _stage_template_selection(self):
        """Stage 2: Select template based on psychology_driver."""
        logger.info("Stage 2: Psychology-Driven Template Selection")

        self._selected_template = self.template_selector.select_from_blueprint(
            self._master_blueprint
        )

        logger.info(
            f"  Selected: {self._selected_template.template_id} "
            f"({self._selected_template.display_name})"
        )

    def _stage_background_generation(
        self,
        background_image_path: Optional[Path],
        num_variants: int,
    ) -> List[Image.Image]:
        """Stage 3: Generate or load backgrounds."""
        logger.info("Stage 3: Background Generation")

        backgrounds = []

        if background_image_path:
            # Load existing background
            logger.info(f"  Loading existing background: {background_image_path}")
            bg = Image.open(background_image_path).convert("RGB")
            backgrounds = [bg] * num_variants

        elif self.config.generate_backgrounds:
            # Generate new backgrounds
            logger.info("  Generating new backgrounds with NanoBanana Pro")

            generated = self.background_generator.generate_from_blueprint(
                blueprint=self._master_blueprint,
                perspective=self._preprocessor_result.perspective,
                output_dir=self.paths.get_backgrounds_output_path(self.config.product),
            )

            if generated:
                backgrounds = [bg.image for bg in generated[:num_variants]]
            else:
                # Fallback: use blank backgrounds
                logger.warning("  Background generation failed, using blank backgrounds")
                backgrounds = [
                    Image.new("RGB", (1080, 1080), (240, 240, 240))
                    for _ in range(num_variants)
                ]
        else:
            # Use blank backgrounds
            logger.info("  Using blank backgrounds")
            backgrounds = [
                Image.new("RGB", (1080, 1080), (240, 240, 240))
                for _ in range(num_variants)
            ]

        logger.info(f"  Prepared {len(backgrounds)} backgrounds")

        return backgrounds

    def _stage_compositing(self, background: Image.Image) -> Image.Image:
        """Stage 4: Physics-aware compositing."""
        logger.info("Stage 4: Physics-Aware Compositing")

        # Extract compositing config from blueprint
        compositing_config = CompositingConfig.from_dict(
            self._master_blueprint.get("compositing", {})
        )

        # Composite product onto background
        result = self.compositor.composite(
            product_image=self._preprocessor_result.trimmed_image,
            background_image=background,
            product_mask=self._preprocessor_result.mask,
            product_position=(100, 100),  # Center-ish
            config=compositing_config,
        )

        logger.info("  Compositing complete")

        return result.composited_image

    def _stage_text_overlay(self, composited: Image.Image) -> Image.Image:
        """Stage 5: Smart text overlay."""
        logger.info("Stage 5: Smart Text Overlay")

        result = self.smart_typer.render_text(
            image=composited,
            product_mask=self._preprocessor_result.mask,
            campaign_content=self._campaign_content,
            template_spec=self._selected_template,
            product_position=(100, 100),
            product_size=self._preprocessor_result.trimmed_image.size,
        )

        logger.info("  Text overlay complete")

        return result.image

    def _save_output(self, image: Image.Image, variant: int, metadata: Dict[str, Any]):
        """Save generated output."""
        output_dir = self.paths.get_generated_output_path(self.config.product)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        filename = f"ad_candidate_{variant:03d}.png"
        output_path = output_dir / filename
        image.save(output_path)
        logger.info(f"  Saved: {output_path}")

        # Save metadata
        metadata_path = output_dir / f"{filename}.yaml"
        with open(metadata_path, "w", encoding="utf-8") as f:
            yaml.dump(metadata, f, default_flow_style=False)


# Convenience function
def generate_ads(
    customer: str,
    platform: str,
    product: str,
    product_image_path: Optional[Path] = None,
    background_image_path: Optional[Path] = None,
    num_variants: int = 1,
) -> PipelineResult:
    """
    Convenience function for quick ad generation.

    Args:
        customer: Customer name
        platform: Platform name (facebook, tiktok, instagram)
        product: Product name
        product_image_path: Optional path to product image
        background_image_path: Optional path to background image
        num_variants: Number of variants to generate

    Returns:
        PipelineResult with generated images

    Example:
        results = generate_ads(
            customer="moprobo",
            platform="meta",
            product="Power Station",
            num_variants=3
        )
        for i, (image, metadata) in enumerate(results.generated_images):
            image.save(f"output_{i+1}.png")
    """
    config = PipelineConfig(
        customer=customer,
        platform=platform,
        product=product,
        num_variants=num_variants,
    )

    pipeline = TemplatePipeline(config)
    return pipeline.run(
        product_image_path=product_image_path,
        background_image_path=background_image_path,
        num_variants=num_variants,
    )


# Main execution
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python pipeline.py <customer> <platform> <product> [num_variants]")
        sys.exit(1)

    customer = sys.argv[1]
    platform = sys.argv[2]
    product = sys.argv[3]
    num_variants = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    results = generate_ads(
        customer=customer,
        platform=platform,
        product=product,
        num_variants=num_variants,
    )

    print(f"\nGenerated {len(results.generated_images)} ad candidates")
    print(f"Output directory: results/{customer}/{platform}/ad_generator/")
