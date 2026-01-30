"""
Stage 3: The Upscaler V2.0 - Psychology-Aware CoT Expansion

Enhances V1.8 upscaler with psychology-driven token expansion.
Generates prompts that align with target psychological triggers.
"""
import logging
import pandas as pd
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CoTUpscalerV2:
    """
    V2.0 Psychology-Aware Chain-of-Thought Upscaler.

    Extends V1.8 with psychology-driven prompt engineering:
    - Psychology-specific CoT prompts
    - Psychology-aligned quality headers
    - Psychology-aware negative prompts
    - Psychology-driven compositing rules
    """

    # Psychology-specific quality headers
    PSYCHOLOGY_QUALITY_HEADERS = {
        "Trust_Authority": (
            "Raw photo, 8k uhd, masterpiece, professional clinical quality, "
            "sterile clean environment, medical-grade precision"
        ),
        "Luxury_Aspiration": (
            "Raw photo, 8k uhd, luxury masterpiece, premium high-end quality, "
            "cinematic dramatic sophistication"
        ),
        "FOMO": (
            "Raw photo, 8k uhd, urgent masterpiece, high-contrast attention-grabbing, "
            "immediate impact quality"
        ),
        "Social_Proof": (
            "Raw photo, 8k uhd, lifestyle masterpiece, authentic natural quality, "
            "relatable candid feeling"
        )
    }

    # Psychology-specific prompt instructions
    PSYCHOLOGY_INSTRUCTIONS = {
        "Trust_Authority": """
IMPORTANT: This expansion must convey TRUST and AUTHORITY.
Emphasize:
- Professionalism, expertise, credibility
- Clean, clinical, sterile qualities
- Rational, objective, transparent
- Precise, accurate, reliable

Avoid:
- Emotional, playful, mysterious elements
- Warm/cozy tones (use cool/neutral)
- Cluttered or chaotic compositions
- Shadows or obscuring elements
""",
        "Luxury_Aspiration": """
IMPORTANT: This expansion must convey LUXURY and ASPIRATION.
Emphasize:
- Premium, exclusive, expensive qualities
- Status, achievement, exclusivity
- Sophisticated, refined, elegant
- Dramatic, impactful, memorable

Avoid:
- Cheap, ordinary, casual elements
- Flat or boring lighting
- Basic or simple descriptions
""",
        "FOMO": """
IMPORTANT: This expansion must convey URGENCY and SCARCITY.
Emphasize:
- Immediate, time-sensitive, scarce
- Attention-grabbing, urgent
- Limited, exclusive opportunity
- Action-oriented, compelling

Avoid:
- Calm, relaxed, abundant elements
- Slow, peaceful descriptions
- Overly generous or spacious compositions
""",
        "Social_Proof": """
IMPORTANT: This expansion must convey BELONGING and VALIDATION.
Emphasize:
- Popular, validated, relatable
- Authentic, human, connection
- Community, belonging, shared experience
- Natural, candid, lifestyle

Avoid:
- Sterile, isolated, artificial elements
- Overly polished or perfect descriptions
- Cold or unemotional tones
"""
    }

    # Psychology-specific feature expansions
    PSYCHOLOGY_FEATURE_EXPANSIONS = {
        "Trust_Authority": {
            "surface_material": {
                "default": "White matte ceramic texture, smooth non-reflective finish, medical-grade surface",
                "Marble": "Italian White Carrara Marble, polished sterile surface, clinical precision",
                "Wood": "Light oak professional finish, clean grain pattern, medical-grade quality",
                "Metal": "Brushed aluminum clinical surface, professional sterile finish"
            },
            "lighting_style": {
                "default": "Soft cool daylight, shadowless clinical lighting, high key illumination",
                "Window Light": "North-facing window light, soft cool diffuse, clinical shadowless",
                "Studio Light": "Professional studio lighting, shadowless clinical setup, neutral 5000K"
            },
            "direction": {
                "default": "Overhead aerial view 90 degrees, objective clinical framing, centered composition",
                "overhead": "Top-down aerial view, objective clinical, balanced symmetry",
                "front": "Frontal view, professional direct, transparent presentation"
            }
        },
        "Luxury_Aspiration": {
            "surface_material": {
                "default": "Polished black marble with gold veining, luxury glossy finish, premium texture",
                "Marble": "Italian Nero Marquina marble, gold veining, dramatic polished finish",
                "Wood": "Macassar ebony, rich grain pattern, premium glossy finish",
                "Metal": "Brushed gold or brass, premium luxury finish, sophisticated sheen"
            },
            "lighting_style": {
                "default": "Dramatic warm spotlight, moody chiaroscuro, premium ambient glow",
                "Window Light": "Golden hour window light, rich warm tones, luxury ambiance",
                "Studio Light": "High-end studio lighting, dramatic rim light, cinematic quality"
            },
            "direction": {
                "default": "Low angle heroic shot, dynamic 45-degree perspective, cinematic depth of field",
                "45-degree": "Three-quarter heroic angle, dynamic perspective, powerful presentation",
                "low_angle": "Low angle heroic view, dramatic powerful, aspirational framing"
            }
        },
        "FOMO": {
            "surface_material": {
                "default": "High contrast surface, immediate visual impact, urgent presentation",
                "Marble": "High-contrast marble, bold dramatic presentation, urgent energy"
            },
            "lighting_style": {
                "default": "Bright direct lighting, high contrast, urgent attention-grabbing",
                "Studio Light": "Bright studio lighting, high contrast, immediate impact"
            },
            "direction": {
                "default": "Close-up direct framing, immediate attention, urgent presentation",
                "close_up": "Extreme close-up, urgent proximity, immediate connection"
            }
        },
        "Social_Proof": {
            "surface_material": {
                "default": "Natural wood texture, authentic warm surface, relatable quality",
                "Wood": "Natural oak wood, authentic grain, warm relatable finish",
                "Fabric": "Soft natural fabric, comfortable texture, authentic warmth"
            },
            "lighting_style": {
                "default": "Soft natural warm light, authentic atmosphere, relatable ambiance",
                "Window Light": "Natural window light, soft warm diffuse, authentic candid feel",
                "Natural Light": "Golden hour natural light, warm authentic, lifestyle quality"
            },
            "direction": {
                "default": "Eye level relatable view, natural perspective, authentic framing",
                "eye_level": "Eye-level view, relatable perspective, natural presentation",
                "45-degree": "Natural three-quarter view, candid authentic, lifestyle angle"
            }
        }
    }

    # Psychology-specific negative prompts
    PSYCHOLOGY_NEGATIVE_PROMPTS = {
        "Trust_Authority": [
            "dark", "messy", "warm yellow light", "neon", "cluttered",
            "shadows", "vibrant colors", "vintage", "mysterious",
            "emotional", "playful", "ornate", "decorative"
        ],
        "Luxury_Aspiration": [
            "plastic", "cluttered", "flat lighting", "bright white",
            "cartoonish", "cheap", "basic", "simple",
            "ordinary", "casual", "messy", "worn"
        ],
        "FOMO": [
            "calm", "serene", "minimalist clean", "cool tones",
            "spacious", "slow", "relaxed", "peaceful",
            "abundant", "unlimited", "permanent"
        ],
        "Social_Proof": [
            "sterile", "clinical", "minimalist isolated",
            "dramatic", "cool unemotional", "artificial",
            "staged", "polished", "perfect", "synthetic"
        ]
    }

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        workflow_templates: Optional[dict] = None,
        fidelity_config: Optional[dict] = None,
        brand_guidelines: Optional[dict] = None
    ):
        """
        Initialize V2 upscaler with psychology awareness.

        Args:
            llm_client: LLM client for CoT expansion
            workflow_templates: Workflow templates from config
            fidelity_config: Fidelity configuration
            brand_guidelines: Optional brand guidelines
        """
        self.llm_client = llm_client
        self.workflow_templates = workflow_templates or {}
        self.fidelity_config = fidelity_config or {}
        self.brand_guidelines = brand_guidelines or {}

    def expand_token_psychology_aware(
        self,
        raw_tag: str,
        feature_name: str,
        target_psychology: str,
        context: dict
    ) -> str:
        """
        Expand token with psychology-aware prompting.

        Args:
            raw_tag: Raw tag value
            feature_name: Feature name
            target_psychology: Target psychology (Trust_Authority, etc.)
            context: Additional context

        Returns:
            Psychology-aligned expanded token
        """
        # Try psychology-specific expansion first
        psych_key = target_psychology.replace("-", "_").replace(" ", "_")

        if psych_key in self.PSYCHOLOGY_FEATURE_EXPANSIONS:
            feature_expansions = self.PSYCHOLOGY_FEATURE_EXPANSIONS[psych_key]

            if feature_name in feature_expansions:
                # Use psychology-specific expansion
                expansions = feature_expansions[feature_name]

                # Try specific value first, then default
                if raw_tag in expansions:
                    return expansions[raw_tag]
                else:
                    # Check if raw_tag contains any key
                    for key, expansion in expansions.items():
                        if key != "default" and key.lower() in raw_tag.lower():
                            return expansion
                    else:
                        # Use default expansion
                        return expansions.get("default", raw_tag)

        # Fallback: Build psychology-aware prompt for LLM
        if self.llm_client:
            return self._expand_with_llm_psychology_aware(
                raw_tag,
                feature_name,
                target_psychology,
                context
            )

        # Final fallback: Return basic expansion
        return f"Premium {raw_tag}"

    def _expand_with_llm_psychology_aware(
        self,
        raw_tag: str,
        feature_name: str,
        target_psychology: str,
        context: dict
    ) -> str:
        """Expand using LLM with psychology-aware prompt."""
        # Build psychology-aware prompt
        instruction = self.PSYCHOLOGY_INSTRUCTIONS.get(target_psychology, "")

        prompt = f"""
Expand this visual feature for a {target_psychity} advertisement:

Feature: {raw_tag}
Category: {feature_name}
Product: {context.get('product_type', 'unknown')}

{instruction}

Provide a clear, descriptive expansion (20-50 words) that aligns with {target_psychology} psychology.
"""

        try:
            # Call LLM
            response = self.llm_client.generate(
                prompt,
                temperature=self.fidelity_config.get("temperature", 0.3),
                max_tokens=self.fidelity_config.get("max_tokens", 200)
            )

            return response.strip().strip('"')

        except Exception as e:
            logger.error(f"LLM expansion failed: {e}")
            return f"Premium {raw_tag}"

    def build_psychology_aware_blueprint(
        self,
        locked_combinations: Dict[str, Dict],
        losers_df: pd.DataFrame,
        raw_tags: Dict[str, List],
        context: Dict[str, Any],
        target_psychology: str
    ) -> Dict[str, Any]:
        """
        Build psychology-aware master blueprint.

        Args:
            locked_combinations: Psychologically validated combinations
            losers_df: Loser creatives
            raw_tags: Raw tags from miner
            context: Additional context
            target_psychology: Target psychology

        Returns:
            Psychology-aware master blueprint
        """
        logger.info(f"Building psychology-aware blueprint for {target_psychology}")

        blueprint = {
            "meta_info": {
                "recipe_id": "recipe_psych_v2.0",
                "mining_strictness": context.get("mining_strictness", "Top_10_Percent"),
                "target_psychology": target_psychology,
                "psychology_mode": context.get("psychology_mode", "unknown"),
                "fidelity_expanded": True
            },
            "strategy_rationale": {
                "psychology_driver": target_psychology,
                "combinations": {}
            },
            "nano_generation_rules": {
                "prompt_slots": {},
                "negative_prompt": self.PSYCHOLOGY_NEGATIVE_PROMPTS.get(
                    target_psychology,
                    self.PSYCHOLOGY_NEGATIVE_PROMPTS["Trust_Authority"]
                ),
                "inference_config": {
                    "steps": 8,
                    "cfg_scale": 1.5,
                    "width": 1024,
                    "height": 1024
                }
            },
            "python_compositing_rules": {
                "shadow_logic": {},
                "color_grading": {},
                "composition_style": {}
            }
        }

        # Add psychology-specific quality headers
        blueprint["nano_generation_rules"]["prompt_slots"]["quality_headers"] = \
            self.PSYCHOLOGY_QUALITY_HEADERS.get(
                target_psychology,
                self.PSYCHOLOGY_QUALITY_HEADERS["Trust_Authority"]
            )

        # Process each locked combination
        for combo_key, combo_data in locked_combinations.items():
            # Extract features
            primary_value = combo_data.get("primary_value", "")
            secondary_value = combo_data.get("secondary_value", "")
            primary_feature = combo_data.get("primary_feature", "")
            secondary_feature = combo_data.get("secondary_feature", "")

            # Expand with psychology awareness
            primary_expanded = self.expand_token_psychology_aware(
                primary_value,
                primary_feature,
                target_psychology,
                context
            )

            secondary_expanded = self.expand_token_psychology_aware(
                secondary_value,
                secondary_feature,
                target_psychology,
                context
            )

            # Map to prompt slots
            if primary_feature == "surface_material":
                blueprint["nano_generation_rules"]["prompt_slots"]["surface_material_expanded"] = \
                    primary_expanded
            elif primary_feature in ["product_context", "scene_subject"]:
                blueprint["nano_generation_rules"]["prompt_slots"]["scene_subject_expanded"] = \
                    primary_expanded

            if secondary_feature in ["lighting_style", "lighting_type"]:
                blueprint["nano_generation_rules"]["prompt_slots"]["lighting_atmosphere_expanded"] = \
                    secondary_expanded
            elif secondary_feature in ["direction", "camera_angle"]:
                blueprint["nano_generation_rules"]["prompt_slots"]["camera_technical"] = \
                    secondary_expanded

            # Add to strategy rationale
            blueprint["strategy_rationale"]["combinations"][combo_key] = {
                "locked_combination": {
                    "primary_feature": primary_feature,
                    "primary_value": primary_value,
                    "secondary_feature": secondary_feature,
                    "secondary_value": secondary_value
                },
                "expanded_combination": {
                    "primary_expanded": primary_expanded,
                    "secondary_expanded": secondary_expanded
                },
                "psychological_alignment": combo_data.get("psychological_alignment", {}),
                "confidence_score": combo_data.get("confidence_score", 0.0)
            }

        # Add psychology-specific compositing rules
        self._add_psychology_compositing_rules(blueprint, target_psychology)

        # Add negative prompts from losers
        negative_tags = self._extract_loser_tags(losers_df)
        blueprint["nano_generation_rules"]["negative_prompt"].extend(negative_tags)

        # Deduplicate negative prompts
        blueprint["nano_generation_rules"]["negative_prompt"] = \
            list(set(blueprint["nano_generation_rules"]["negative_prompt"]))

        return blueprint

    def _add_psychology_compositing_rules(
        self,
        blueprint: Dict[str, Any],
        target_psychology: str
    ):
        """Add psychology-specific compositing rules."""
        if target_psychology == "Trust_Authority":
            blueprint["python_compositing_rules"].update({
                "shadow_logic": {
                    "type": "Soft_Diffused",
                    "reasoning": "Authority requires clean, shadowless presentation"
                },
                "color_grading": {
                    "temperature": "cool_neutral",
                    "saturation": "muted",
                    "contrast": "medium_high",
                    "reasoning": "Cool tones convey rationality and professionalism"
                },
                "composition_style": {
                    "style": "minimalist_centered",
                    "negative_space_ratio": 0.6,
                    "reasoning": "Generous negative space = confidence and transparency"
                }
            })

        elif target_psychology == "Luxury_Aspiration":
            blueprint["python_compositing_rules"].update({
                "shadow_logic": {
                    "type": "Dramatic_Hard",
                    "reasoning": "Luxury benefits from dramatic shadows and depth"
                },
                "color_grading": {
                    "temperature": "warm_rich",
                    "saturation": "high",
                    "contrast": "high",
                    "reasoning": "Rich colors and high contrast convey premium quality"
                },
                "composition_style": {
                    "style": "minimalist_luxury",
                    "negative_space_ratio": 0.7,
                    "reasoning": "Generous space = exclusivity and expense"
                }
            })

        elif target_psychology == "FOMO":
            blueprint["python_compositing_rules"].update({
                "shadow_logic": {
                    "type": "High_Contrast",
                    "reasoning": "Urgency requires immediate visual impact"
                },
                "color_grading": {
                    "temperature": "warm_urgent",
                    "saturation": "high",
                    "contrast": "very_high",
                    "reasoning": "High saturation and contrast grab attention"
                },
                "composition_style": {
                    "style": "center_focus",
                    "negative_space_ratio": 0.2,
                    "reasoning": "Tight framing = scarcity and urgency"
                }
            })

        elif target_psychology == "Social_Proof":
            blueprint["python_compositing_rules"].update({
                "shadow_logic": {
                    "type": "Natural_Soft",
                    "reasoning": "Authenticity requires natural-looking lighting"
                },
                "color_grading": {
                    "temperature": "warm_natural",
                    "saturation": "medium",
                    "contrast": "medium",
                    "reasoning": "Natural colors feel authentic and relatable"
                },
                "composition_style": {
                    "style": "lifestyle_candid",
                    "negative_space_ratio": 0.4,
                    "reasoning": "Balanced composition feels natural and relatable"
                }
            })

    def _extract_loser_tags(self, losers_df: pd.DataFrame) -> List[str]:
        """Extract common tags from losers for negative prompts."""
        negative_tags = []

        # Find features overrepresented in losers
        for feature in ["lighting_style", "background_content_type", "color_balance"]:
            if feature in losers_df.columns:
                mode_value = losers_df[feature].mode()
                if len(mode_value) > 0:
                    negative_tags.append(str(mode_value[0]))

        return negative_tags
