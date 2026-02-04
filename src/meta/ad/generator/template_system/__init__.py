"""
Ad Generator: Template-Driven Creative Engine.

Template-driven, psychology-aware compositing system.

Core Components:
- GeneratorPaths: Single config per customer, platform-specific outputs
- ProductPreprocessor: Trim transparency, perspective detection
- PsychologyTemplateSelector: Auto-select templates based on psychology_driver
- NanoBackgroundGenerator: Perspective-aware background generation
- PhysicsCompositor: Dual-layer shadows, light matching, light wrap
- SmartTyper: Psychology-adaptive text overlay with collision detection
- TemplatePipeline: End-to-end orchestrator

Author: Ad System
Date: 2026-01-30
"""

# Path Management
from src.meta.ad.generator.template_system.paths import (
    GeneratorPaths,
    create_customer_structure,
)

# Product Preprocessing
from src.meta.ad.generator.template_system.product_preprocessor import (
    ProductPreprocessor,
    PreprocessorResult,
    PerspectiveType,
    preprocess_product,
)

# Template System
from src.meta.ad.generator.template_system.template_selector import (
    TemplateLoader,
    TemplateSpec,
    PsychologySpec,
    PsychologyTemplateSelector,
    select_template_from_blueprint,
    load_template_by_id,
)

# Background Generation
from src.meta.ad.generator.template_system.background_generator import (
    GenerationConfig,
    BackgroundPrompt,
    GeneratedBackground,
    NanoBackgroundGenerator,
    generate_backgrounds_from_blueprint,
)

# Physics Compositing
from src.meta.ad.generator.template_system.physics_compositor import (
    ShadowDirection,
    CompositingConfig,
    CompositingResult,
    PhysicsCompositor,
    composite_physics_aware,
)

# Smart Text Overlay
from src.meta.ad.generator.template_system.smart_typer import (
    CampaignContent,
    SmartColorCalculator,
    CollisionDetector,
    SmartTyper,
    render_text_overlay,
)

# Pipeline Orchestrator
from src.meta.ad.generator.template_system.pipeline import (
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    TemplatePipeline,
    generate_ads,
)

__all__ = [
    # Path Management
    "GeneratorPaths",
    "create_customer_structure",

    # Product Preprocessing
    "ProductPreprocessor",
    "PreprocessorResult",
    "PerspectiveType",
    "preprocess_product",

    # Template System
    "TemplateLoader",
    "TemplateSpec",
    "PsychologySpec",
    "PsychologyTemplateSelector",
    "select_template_from_blueprint",
    "load_template_by_id",

    # Background Generation
    "GenerationConfig",
    "BackgroundPrompt",
    "GeneratedBackground",
    "NanoBackgroundGenerator",
    "generate_backgrounds_from_blueprint",

    # Physics Compositing
    "ShadowDirection",
    "CompositingConfig",
    "CompositingResult",
    "PhysicsCompositor",
    "composite_physics_aware",

    # Smart Text Overlay
    "CampaignContent",
    "SmartColorCalculator",
    "CollisionDetector",
    "SmartTyper",
    "render_text_overlay",

    # Pipeline Orchestrator
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
    "TemplatePipeline",
    "generate_ads",
]
