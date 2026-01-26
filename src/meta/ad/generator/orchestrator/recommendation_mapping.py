"""
Recommendation Feature to Template Placeholder Mapping.

Maps recommendation feature names (from ad/recommender) to template placeholders
and provides value transformation functions.
"""

from typing import Callable, Dict, Optional, Tuple


# Mapping: recommendation_feature -> (placeholder_name, value_transform_function)
# value_transform: Optional function to transform recommendation value to placeholder value
RECOMMENDATION_TO_PLACEHOLDER: Dict[str, Tuple[str, Optional[Callable[[str], str]]]] = {
    # Camera and view features
    "direction": ("global_view_definition", lambda v: {
        "overhead": "top-down bird's eye view",
        "front": "front-facing view",
        "side": "side profile view",
        "45-degree": "45-degree angled view",
    }.get(v.lower(), f"{v.lower()} view")),
    
    # Lighting features
    "lighting_style": ("lighting_detail", lambda v: {
        "studio": "professional studio lighting setup",
        "natural": "natural window or outdoor lighting",
        "artificial": "artificial studio lighting",
    }.get(v.lower(), f"{v.lower()} lighting")),
    
    "lighting_type": ("lighting_detail", lambda v: {
        "artificial": "artificial studio lighting",
        "natural": "natural daylight lighting",
    }.get(v.lower(), f"{v.lower()} lighting")),
    
    "mood_lighting": ("lighting_detail", lambda v: {
        "clinical": "clinical, clean lighting",
        "energetic": "energetic, vibrant lighting",
        "natural": "natural, soft lighting",
    }.get(v.lower(), f"{v.lower()} mood lighting")),
    
    # Color features
    "primary_colors": ("color_constraint", lambda v: f"Color palette: {v}"),
    
    "temperature": ("atmosphere_description", lambda v: {
        "cool": "Cool",
        "warm": "Warm",
        "neutral": "Neutral",
    }.get(v.lower(), v.capitalize() if v else "Neutral")),
    
    "color_balance": ("color_balance", None),  # Already mapped, pass through
    
    # Composition and layout
    "product_position": ("product_position", None),  # Already mapped, pass through
    
    "visual_prominence": ("visual_prominence", None),  # Already mapped, pass through
    
    "product_visibility": ("product_visibility", None),  # Already mapped, pass through
    
    "background_content_type": ("static_context", lambda v: {
        "solid-color": "solid-color background",
        "textured": "textured background",
        "environment": "natural home environment",
    }.get(v.lower(), f"{v} background")),
    
    "context_richness": ("static_context", lambda v: {
        "moderate": "moderately detailed environment",
        "rich": "rich, detailed environment",
        "minimal": "minimal, clean environment",
    }.get(v.lower(), f"{v} context")),
    
    # Relationship and interaction
    "relationship_depiction": ("relationship_depiction", None),  # Already mapped, pass through
    
    "human_elements": ("human_elements", None),  # Already mapped, pass through
    
    "product_context": ("static_context", lambda v: {
        "isolated": "isolated product on clean background",
        "in-use": "product in use context",
        "lifestyle": "lifestyle context",
    }.get(v.lower(), f"{v} context")),
    
    # Visual flow and composition
    "visual_flow": ("composition_style", lambda v: {
        "forced": "forced visual flow directing attention to product",
        "natural": "natural visual flow",
        "z-pattern": "z-pattern eye tracking",
    }.get(v.lower(), f"{v} visual flow")),
    
    "composition_style": ("composition_style", None),  # Already mapped, pass through
    
    # Depth and layers
    "depth_layers": ("static_context", lambda v: {
        "shallow": "shallow depth of field with minimal background layers",
        "deep": "deep depth with multiple background layers",
        "moderate": "moderate depth with layered composition",
    }.get(v.lower(), f"{v} depth layers")),
    
    # Contrast and visual impact
    "contrast_level": ("color_balance", lambda v: {
        "high": "high contrast lighting",
        "medium": "medium contrast lighting",
        "low": "low contrast, soft lighting",
    }.get(v.lower(), f"{v} contrast")),
    
    "background_tone_contrast": ("color_balance", lambda v: {
        "high": "high background contrast",
        "medium": "medium background contrast",
        "low": "low background contrast",
    }.get(v.lower(), f"{v} background contrast")),
    
    # Product presentation
    "product_angle": ("global_view_definition", lambda v: {
        "45-degree": "45-degree angled view",
        "90-degree": "straight-on view",
        "overhead": "top-down bird's eye view",
    }.get(v.lower(), f"{v} angle")),
    
    "product_placement": ("product_position", None),  # Alias for product_position
    
    "horizontal_alignment": ("product_position", lambda v: {
        "right": "right-aligned",
        "left": "left-aligned",
        "center": "center-aligned",
    }.get(v.lower(), f"{v} alignment")),
    
    # Shadow and lighting direction
    "shadow_direction": ("lighting_detail", lambda v: {
        "overhead": "overhead lighting creating downward shadows",
        "side": "side lighting creating directional shadows",
        "front": "front lighting with minimal shadows",
    }.get(v.lower(), f"{v} shadow direction")),
    
    # Image style
    "image_style": ("static_context", lambda v: {
        "professional": "professional photography style",
        "lifestyle": "lifestyle photography style",
        "studio": "studio photography style",
    }.get(v.lower(), f"{v} style")),
    
    "visual_complexity": ("static_context", lambda v: {
        "simple": "simple, clean composition",
        "moderate": "moderately complex composition",
        "complex": "complex, detailed composition",
    }.get(v.lower(), f"{v} complexity")),
    
    # Text and CTA (may not map directly to visual placeholders)
    "text_elements": (None, None),  # Text is handled separately, not in visual formula
    "cta_visuals": (None, None),  # CTA is handled separately
    
    # Person-related (negative guidance)
    "person_count": (None, None),  # Handled via negative_guidance
    "person_age_group": (None, None),  # Handled via negative_guidance
    "person_gender": (None, None),  # Handled via negative_guidance
    "person_activity": (None, None),  # Handled via negative_guidance
    "person_relationship_type": (None, None),  # Handled via negative_guidance
    "primary_focal_point": (None, None),  # Handled via negative_guidance if "person"
    
    # Architectural and other features
    "architectural_elements_presence": ("static_context", lambda v: {
        "no": "no architectural elements",
        "yes": "with architectural elements",
    }.get(v.lower(), f"{v} architectural elements")),
    
    "problem_solution_narrative": (None, None),  # Narrative feature, not visual
}


def get_placeholder_for_feature(feature_name: str) -> Optional[str]:
    """
    Get template placeholder name for a recommendation feature.
    
    Args:
        feature_name: Recommendation feature name (e.g., "direction", "lighting_style")
    
    Returns:
        Placeholder name (e.g., "global_view_definition") or None if not mapped
    """
    mapping = RECOMMENDATION_TO_PLACEHOLDER.get(feature_name)
    if mapping:
        return mapping[0]
    return None


def transform_feature_value(feature_name: str, value: str) -> str:
    """
    Transform recommendation feature value to placeholder-appropriate value.
    
    Args:
        feature_name: Recommendation feature name
        value: Raw feature value from recommendation
    
    Returns:
        Transformed value for use in template placeholder
    """
    mapping = RECOMMENDATION_TO_PLACEHOLDER.get(feature_name)
    if mapping and mapping[1]:  # Has transform function
        try:
            return mapping[1](value)
        except Exception as e:
            logger.warning(
                "Transform failed for %s=%s: %s, using original value",
                feature_name, value, e
            )
            return value
    # No transform or None transform -> pass through
    return value


def is_mapped_feature(feature_name: str) -> bool:
    """
    Check if a recommendation feature has a placeholder mapping.
    
    Args:
        feature_name: Recommendation feature name
    
    Returns:
        True if feature maps to a placeholder, False otherwise
    """
    mapping = RECOMMENDATION_TO_PLACEHOLDER.get(feature_name)
    return mapping is not None and mapping[0] is not None


# Import logger
import logging
logger = logging.getLogger(__name__)
