"""
Prompt Builder - V1.0

Dynamically generates NanoBanana Pro prompts from mined patterns.

Separation of concerns:
- Ad Miner: Discovers WHAT works (patterns, features, psychology)
- Prompt Builder: Generates HOW to use it (prompts, generation instructions)
"""
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds generation prompts from mined patterns.

    Reads patterns.yaml and dynamically generates prompts for NanoBanana Pro
    based on combinatorial patterns, individual features, and psychology patterns.
    """

    def __init__(self, patterns_path: Path, config: Optional[Dict] = None):
        """
        Initialize prompt builder with mined patterns.

        Args:
            patterns_path: Path to patterns.yaml from ad miner
            config: Optional generation config (aspect_ratio, cfg_scale, steps, etc.)
        """
        import yaml

        self.patterns_path = patterns_path
        with open(patterns_path, 'r') as f:
            self.patterns = yaml.safe_load(f)

        self.config = config or self._default_config()

        logger.info(f"Loaded patterns from {patterns_path}")
        logger.info(f"  Combinatorial patterns: {len(self.patterns.get('combinatorial_patterns', []))}")
        logger.info(f"  Individual features: {len(self.patterns.get('individual_features', []))}")
        logger.info(f"  Psychology patterns: {len(self.patterns.get('psychology_patterns', []))}")

    def _default_config(self) -> Dict:
        """Default generation config from moprobo config."""
        return {
            "model": "nanobanana_pro",
            "aspect_ratio": "3:4",
            "cfg_scale": 3.5,
            "steps": 8,
            "batch_size": 20,
            "negative_prompt": "cartoon, illustration, text, watermark, distorted"
        }

    def build_top_combination_prompt(self) -> Dict[str, Any]:
        """
        Build prompt from top combination (highest confidence combinatorial pattern).

        Returns:
            Dict with prompt text, features used, psychology overlay, and generation config
        """
        comb_patterns = self.patterns.get("combinatorial_patterns", [])
        if not comb_patterns:
            logger.warning("No combinatorial patterns found")
            return {}

        # Use top pattern (highest confidence)
        top_pattern = comb_patterns[0]
        features_used = top_pattern.get("features", {}).copy()

        # Add optional fields if not present
        if "camera_angle" not in features_used:
            features_used["camera_angle"] = "45-degree"
        if "color_temperature" not in features_used:
            features_used["color_temperature"] = "Warm"

        # Get ROAS lift
        roas_lift = top_pattern.get("roas_lift_multiple", 3.5)

        # Build natural language prompt
        prompt = self._build_natural_prompt(features_used)

        # Add psychology overlay if available
        psych_patterns = self.patterns.get("psychology_patterns", [])
        psychology = self._extract_psychology_overlay(psych_patterns)

        # Build combination name from features
        feature_names = " + ".join([f"{k}: {v}" for k, v in top_pattern.get("features", {}).items()])

        return {
            "prompt_id": "top_combination_primary",
            "prompt_name": f"{top_pattern.get('combination', feature_names)} (Top)",
            "strategy": "top_combination",
            "confidence": top_pattern.get("confidence", 0.92),
            "roas_lift": roas_lift,
            "nano_prompt": prompt,
            "features_used": features_used,
            "psychology_overlay": psychology,
            "generation_config": self.config
        }

    def build_supporting_combination_prompts(self, max_prompts: int = 3) -> List[Dict[str, Any]]:
        """
        Build prompts from supporting combinations.

        Args:
            max_prompts: Maximum number of supporting prompts to generate

        Returns:
            List of prompt dicts
        """
        comb_patterns = self.patterns.get("combinatorial_patterns", [])
        supporting = comb_patterns[1:max_prompts+1]  # Skip first (locked)

        prompts = []
        for pattern in supporting:
            features = pattern.get("features", {})
            prompt_text = self._build_natural_prompt(features)

            psychology = self._extract_psychology_overlay(
                self.patterns.get("psychology_patterns", [])
            )

            prompts.append({
                "prompt_id": f"supporting_combo_{len(prompts)+1}",
                "prompt_name": pattern.get("combination", "Unknown").title(),
                "strategy": "supporting_combination",
                "confidence": pattern.get("confidence", 0.0),
                "roas_lift": pattern.get("roas_lift_multiple", 0.0),
                "nano_prompt": prompt_text,
                "features_used": features,
                "psychology_overlay": psychology,
                "generation_config": self.config
            })

        return prompts

    def build_individual_feature_prompts(self, max_prompts: int = 5) -> List[Dict[str, Any]]:
        """
        Build prompts from top individual features.

        Args:
            max_prompts: Maximum number of individual feature prompts

        Returns:
            List of prompt dicts
        """
        ind_features = self.patterns.get("individual_features", [])[:max_prompts]

        prompts = []
        for feature in ind_features:
            feature_name = feature.get("feature", "")
            feature_value = feature.get("value", "")

            # Build features dict for this individual feature
            features_used = {feature_name: feature_value}

            # Get best combination if available
            best_combo = feature.get("best_combination", "")
            if best_combo:
                # Parse best combination string like "surface_material: Marble + lighting_style: Window Light"
                parts = best_combo.split(" + ")
                for part in parts:
                    if ":" in part:
                        k, v = part.split(":", 1)
                        features_used[k.strip()] = v.strip()

            prompt_text = self._build_natural_prompt(features_used)

            prompts.append({
                "prompt_id": f"individual_{feature_name}_{len(prompts)+1}",
                "prompt_name": f"{feature_value} ({feature_name})",
                "strategy": "individual_feature",
                "confidence": feature.get("confidence", 0.0),
                "roas_lift": feature.get("individual_roas_lift", 0.0),
                "nano_prompt": prompt_text,
                "features_used": features_used,
                "psychology_overlay": {},
                "generation_config": self.config,
                "note": feature.get("reason", "")
            })

        return prompts

    def build_psychology_prompts(self, max_prompts: int = 2) -> List[Dict[str, Any]]:
        """
        Build prompts emphasizing psychology patterns.

        Reads psychology_patterns from patterns.yaml and creates prompts that
        combine the psychology pattern with the top visual combination.

        Args:
            max_prompts: Maximum number of psychology prompts

        Returns:
            List of prompt dicts
        """
        psych_patterns = self.patterns.get("psychology_patterns", [])
        if not psych_patterns:
            logger.warning("No psychology_patterns found in patterns.yaml")
            return []

        psych_patterns = psych_patterns[:max_prompts]

        prompts = []
        for psych in psych_patterns:
            # Get base visual features from top combination
            comb_patterns = self.patterns.get("combinatorial_patterns", [])
            features_used = {}

            if comb_patterns:
                # Use features from top pattern
                top_pattern = comb_patterns[0]
                features_used = top_pattern.get("features", {}).copy()

                # Add defaults if missing
                if "surface_material" not in features_used:
                    features_used["surface_material"] = "Marble"
                if "lighting_style" not in features_used:
                    features_used["lighting_style"] = "Window Light"
            else:
                # Fallback defaults
                features_used = {
                    "surface_material": "Marble",
                    "lighting_style": "Window Light"
                }

            # Build prompt
            prompt_text = self._build_natural_prompt(features_used)

            # Extract full psychology components from mined pattern
            psych_components = psych.get("components", {})
            psychology = {
                "template_id": psych_components.get("template_id", "trust_authority"),
                "headline_font": psych_components.get("headline_font", "Serif_Bold"),
                "primary_color": psych_components.get("primary_color", "#003366"),
                "copy_pattern": psych_components.get("copy_pattern", "Expert recommended"),
                "layout": psych_components.get("layout", "centered"),
                "position": psych_components.get("position", "Bottom_Center")
            }

            prompts.append({
                "prompt_id": f"psychology_{psych.get('pattern', 'unknown')}",
                "prompt_name": psych.get("display_name", "Psychology Pattern"),
                "strategy": "psychology_pattern",
                "confidence": psych.get("confidence", 0.0),
                "roas_lift": psych.get("individual_roas_lift", 0.0),
                "nano_prompt": prompt_text,
                "features_used": features_used,
                "psychology_overlay": psychology,
                "generation_config": self.config,
                "note": f"Psychology overlay enhances base creative by {psych.get('individual_roas_pct', 0)}%"
            })

        return prompts

    def build_all_prompts(self) -> Dict[str, List[Dict]]:
        """
        Build all prompts from patterns.

        Returns:
            Dict with prompt categories:
            - top_combination: [primary top prompt]
            - supporting_combinations: [supporting prompts]
            - individual_features: [individual feature prompts]
            - psychology: [psychology-focused prompts]
        """
        logger.info("Building all prompts from patterns...")

        all_prompts = {
            "top_combination": [self.build_top_combination_prompt()],
            "supporting_combinations": self.build_supporting_combination_prompts(max_prompts=3),
            "individual_features": self.build_individual_feature_prompts(max_prompts=5),
            "psychology": self.build_psychology_prompts(max_prompts=2)
        }

        total_prompts = sum(len(v) for v in all_prompts.values())
        logger.info(f"Generated {total_prompts} total prompts")

        return all_prompts

    def _build_natural_prompt(self, features: Dict[str, str]) -> str:
        """
        Build natural language prompt from feature dict.

        Args:
            features: Dict of feature_name → value

        Returns:
            Natural language prompt string
        """
        prompt_parts = []

        # Product context - read from patterns metadata
        metadata = self.patterns.get("metadata", {})
        product_name = metadata.get("product", "product")
        customer = metadata.get("customer", "")

        # Format product name (capitalize first letter)
        product = product_name.capitalize() if product_name else "Product"

        # Surface material
        if "surface_material" in features:
            material = features["surface_material"]
            prompt_parts.append(f"on luxury {material} table surface")

        # Lighting
        if "lighting_style" in features:
            lighting = features["lighting_style"]
            if lighting == "Window Light":
                prompt_parts.append("Window Light natural illumination")
            elif lighting == "Studio Lighting":
                prompt_parts.append("Studio Lighting")

        # Camera angle
        if "camera_angle" in features:
            angle = features["camera_angle"]
            prompt_parts.append(f"{angle} camera angle")

        # Color temperature
        if "color_temperature" in features:
            temp = features["color_temperature"]
            prompt_parts.append(f"{temp} color temperature")

        # Product position
        if "product_position" in features:
            position = features["product_position"]
            if position == "Centered":
                prompt_parts.append("centered composition")
            elif position == "Rule-of-Thirds":
                prompt_parts.append("rule-of-thirds composition")

        # Environment/mood
        prompt_parts.append("premium home environment")
        prompt_parts.append("high-quality product photography")
        prompt_parts.append("depth of field")

        # Build final prompt
        prompt = f"{product} " + ", ".join(prompt_parts)

        return prompt

    def _extract_psychology_overlay(self, psych_patterns: List[Dict]) -> Dict:
        """
        Extract psychology overlay components.

        Args:
            psych_patterns: List of psychology patterns

        Returns:
            Dict with psychology components
        """
        if not psych_patterns:
            return {}

        # Use first (highest confidence) psychology pattern
        psych = psych_patterns[0]
        components = psych.get("components", {})

        return {
            "template_id": components.get("template_id", "trust_authority"),
            "headline_font": components.get("headline_font", "Serif_Bold"),
            "primary_color": components.get("primary_color", "#003366"),
            "copy_pattern": components.get("copy_pattern", "Expert recommended"),
            "cta_text": "Shop Now"
        }
