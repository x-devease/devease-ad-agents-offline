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

# Try to import nano agent for prompt enhancement
try:
    from src.agents.nano import enhance_prompt
    NANO_AGENT_AVAILABLE = True
except ImportError:
    NANO_AGENT_AVAILABLE = False
    enhance_prompt = None

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds generation prompts from mined patterns.

    Reads patterns.yaml and dynamically generates prompts for NanoBanana Pro
    based on combinatorial patterns, individual features, and psychology patterns.
    """

    def __init__(self, patterns_path: Path, config: Optional[Dict] = None, use_enhancement: Optional[bool] = None):
        """
        Initialize prompt builder with mined patterns.

        Args:
            patterns_path: Path to patterns.yaml from ad miner
            config: Optional generation config (aspect_ratio, cfg_scale, steps, etc.) and prompt_building limits
            use_enhancement: Deprecated - use config['prompt_building']['enable_nano_enhancement'] instead
        """
        import yaml

        self.patterns_path = patterns_path
        with open(patterns_path, 'r') as f:
            self.patterns = yaml.safe_load(f)

        self.config = config or self._default_config()

        # Extract prompt building limits from config
        prompt_building_config = self.config.get('prompt_building', {})
        self.max_prompts_config = prompt_building_config.get('max_prompts', {})

        # Determine enhancement setting (prefer config, fallback to parameter, default True)
        if use_enhancement is not None:
            logger.warning("use_enhancement parameter is deprecated, use config['prompt_building']['enable_nano_enhancement'] instead")
        enhancement_enabled = prompt_building_config.get('enable_nano_enhancement', True)
        self.use_enhancement = enhancement_enabled and NANO_AGENT_AVAILABLE

        if self.use_enhancement:
            logger.info("Nano agent prompt enhancement: ENABLED")
        elif not NANO_AGENT_AVAILABLE:
            logger.warning("Nano agent not available, using basic prompts")
        else:
            logger.info("Nano agent prompt enhancement: DISABLED")

        logger.info(f"Loaded patterns from {patterns_path}")
        logger.info(f"  Combinatorial patterns: {len(self.patterns.get('combinatorial_patterns', []))}")
        logger.info(f"  Individual features: {len(self.patterns.get('individual_features', []))}")
        logger.info(f"  Psychology patterns: {len(self.patterns.get('psychology_patterns', []))}")

        # Log prompt building limits
        logger.info(f"Prompt building limits from config:")
        logger.info(f"  Top combination: {self._get_max_prompts('top_combination', default=1)}")
        logger.info(f"  Supporting combinations: {self._get_max_prompts('supporting_combinations', default=3)}")
        logger.info(f"  Individual features: {self._get_max_prompts('individual_features', default=5)}")
        logger.info(f"  Psychology patterns: {self._get_max_prompts('psychology_patterns', default=2)}")

    def _get_max_prompts(self, category: str, default: int = None) -> int:
        """
        Get max prompts for a category from config.

        Args:
            category: Category name (top_combination, supporting_combinations, etc.)
            default: Default value if not in config

        Returns:
            Max prompts value from config or default
        """
        return self.max_prompts_config.get(category, default if default is not None else 5)

    def _check_anti_patterns(self, features: Dict[str, str]) -> List[str]:
        """
        Check if features match any anti-patterns.

        Args:
            features: Dict of feature_name → value

        Returns:
            List of anti-pattern violations (empty if none)
        """
        anti_patterns = self.patterns.get("anti_patterns", [])
        violations = []

        for anti_pattern in anti_patterns:
            avoid_features = anti_pattern.get("avoid_features", {})
            # Check if all avoid features are present in the given features
            matches = all(
                features.get(k) == v
                for k, v in avoid_features.items()
            )

            if matches:
                violation = f"Anti-pattern: {avoid_features}"
                violations.append(violation)
                logger.warning(f"{violation} - {anti_pattern.get('reason', 'Reason not provided')}")

        return violations

    def _get_confidence_threshold(self, category: str) -> float:
        """
        Get confidence threshold for a category from config.

        Args:
            category: Category name (individual_features, etc.)

        Returns:
            Confidence threshold (0.0 to 1.0)
        """
        thresholds = self.config.get('prompt_building', {}).get('confidence_thresholds', {})
        return thresholds.get(category, 0.0)  # Default: no threshold (include all)

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

        # Check for anti-pattern violations
        anti_pattern_violations = self._check_anti_patterns(features_used)

        # Add psychology overlay if available
        psych_patterns = self.patterns.get("psychology_patterns", [])
        psychology = self._extract_psychology_overlay(psych_patterns)

        # Build combination name from features
        feature_names = " + ".join([f"{k}: {v}" for k, v in top_pattern.get("features", {}).items()])

        return {
            "prompt_id": "top_combination_primary",
            "prompt_name": f"{top_pattern.get('combination', feature_names)} (Top)",
            "strategy": "top_combination",
            "category": "top_combination",  # Category metadata for A/B testing
            "confidence": top_pattern.get("confidence", 0.92),
            "roas_lift": roas_lift,
            "nano_prompt": prompt,
            "features_used": features_used,
            "psychology_overlay": psychology,
            "generation_config": self.config,
            "anti_pattern_violations": anti_pattern_violations,  # Anti-pattern validation
            "passed_anti_pattern_check": len(anti_pattern_violations) == 0
        }

    def build_supporting_combination_prompts(self, max_prompts: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Build prompts from supporting combinations.

        Args:
            max_prompts: Maximum number of supporting prompts to generate (overrides config)

        Returns:
            List of prompt dicts
        """
        # Use config value if not overridden
        if max_prompts is None:
            max_prompts = self._get_max_prompts('supporting_combinations', default=3)

        comb_patterns = self.patterns.get("combinatorial_patterns", [])
        supporting = comb_patterns[1:max_prompts+1]  # Skip first (locked)

        prompts = []
        for pattern in supporting:
            features = pattern.get("features", {})
            prompt_text = self._build_natural_prompt(features)

            # Check for anti-pattern violations
            anti_pattern_violations = self._check_anti_patterns(features)

            psychology = self._extract_psychology_overlay(
                self.patterns.get("psychology_patterns", [])
            )

            prompts.append({
                "prompt_id": f"supporting_combo_{len(prompts)+1}",
                "prompt_name": pattern.get("combination", "Unknown").title(),
                "strategy": "supporting_combination",
                "category": "supporting_combinations",  # Category metadata for A/B testing
                "confidence": pattern.get("confidence", 0.0),
                "roas_lift": pattern.get("roas_lift_multiple", 0.0),
                "nano_prompt": prompt_text,
                "features_used": features,
                "psychology_overlay": psychology,
                "generation_config": self.config,
                "anti_pattern_violations": anti_pattern_violations,  # Anti-pattern validation
                "passed_anti_pattern_check": len(anti_pattern_violations) == 0
            })

        return prompts

    def build_individual_feature_prompts(self, max_prompts: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Build prompts from top individual features.

        Applies anti-pattern validation and confidence threshold filtering.

        Args:
            max_prompts: Maximum number of individual feature prompts (overrides config)

        Returns:
            List of prompt dicts
        """
        # Use config value if not overridden
        if max_prompts is None:
            max_prompts = self._get_max_prompts('individual_features', default=5)

        # Get confidence threshold for filtering
        confidence_threshold = self._get_confidence_threshold('individual_features')

        ind_features = self.patterns.get("individual_features", [])

        # Filter by confidence threshold if configured
        if confidence_threshold > 0:
            ind_features = [
                f for f in ind_features
                if f.get('confidence', 0.0) >= confidence_threshold
            ]
            logger.info(f"Filtered individual features by confidence >= {confidence_threshold}: {len(ind_features)} remaining")

        # Apply max_prompts limit
        ind_features = ind_features[:max_prompts]

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

            # Check for anti-pattern violations
            anti_pattern_violations = self._check_anti_patterns(features_used)

            prompts.append({
                "prompt_id": f"individual_{feature_name}_{len(prompts)+1}",
                "prompt_name": f"{feature_value} ({feature_name})",
                "strategy": "individual_feature",
                "category": "individual_features",  # Category metadata for A/B testing
                "confidence": feature.get("confidence", 0.0),
                "roas_lift": feature.get("individual_roas_lift", 0.0),
                "nano_prompt": prompt_text,
                "features_used": features_used,
                "psychology_overlay": {},
                "generation_config": self.config,
                "note": feature.get("reason", ""),
                "anti_pattern_violations": anti_pattern_violations,  # Anti-pattern validation
                "passed_anti_pattern_check": len(anti_pattern_violations) == 0
            })

        return prompts

    def build_psychology_prompts(self, max_prompts: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Build prompts emphasizing psychology patterns.

        Reads psychology_patterns from patterns.yaml and creates prompts that
        combine the psychology pattern with the top visual combination.

        Args:
            max_prompts: Maximum number of psychology prompts (overrides config)

        Returns:
            List of prompt dicts
        """
        # Use config value if not overridden
        if max_prompts is None:
            max_prompts = self._get_max_prompts('psychology_patterns', default=2)

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

            # Check for anti-pattern violations
            anti_pattern_violations = self._check_anti_patterns(features_used)

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
                "category": "psychology_patterns",  # Category metadata for A/B testing
                "confidence": psych.get("confidence", 0.0),
                "roas_lift": psych.get("individual_roas_lift", 0.0),
                "nano_prompt": prompt_text,
                "features_used": features_used,
                "psychology_overlay": psychology,
                "generation_config": self.config,
                "note": f"Psychology overlay enhances base creative by {psych.get('individual_roas_pct', 0)}%",
                "anti_pattern_violations": anti_pattern_violations,  # Anti-pattern validation
                "passed_anti_pattern_check": len(anti_pattern_violations) == 0
            })

        return prompts

    def build_all_prompts(self) -> Dict[str, List[Dict]]:
        """
        Build all prompts from patterns.

        Uses max_prompts values from config to determine how many prompts
        to generate for each category.

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
            "supporting_combinations": self.build_supporting_combination_prompts(),  # Uses config
            "individual_features": self.build_individual_feature_prompts(),  # Uses config
            "psychology": self.build_psychology_prompts()  # Uses config
        }

        total_prompts = sum(len(v) for v in all_prompts.values())
        logger.info(f"Generated {total_prompts} total prompts")

        return all_prompts

    def _build_natural_prompt(self, features: Dict[str, str]) -> str:
        """
        Build natural language prompt from feature dict.

        Uses explicit product photography keywords to trigger high-fidelity
        enhancement in nano agent (SPECIFIC_REQUEST category, PRODUCT_PHOTOGRAPHY intent,
        3-5 techniques, 0.90+ confidence).

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

        # START WITH KEYWORDS THAT TRIGGER SPECIFIC_REQUEST CATEGORY
        # Must include: "\d+K" OR "white background" OR "studio lighting" OR "\d+.*degrees"
        prompt_parts.append("Professional product photograph")
        prompt_parts.append("4K")  # No space - matches \d+K pattern

        # Check lighting to add "studio lighting" if needed
        if "lighting_style" in features:
            lighting = features["lighting_style"]
            if lighting == "Studio Lighting":
                prompt_parts.append("studio lighting")  # Triggers SPECIFIC_REQUEST
            elif lighting == "Window Light":
                # Add both natural and studio for quality
                prompt_parts.append("studio lighting")
                prompt_parts.append("with natural window light enhancement")

        # Camera angle - use "degrees" to match pattern
        if "camera_angle" in features:
            angle = features["camera_angle"]
            # Convert "45-degree" to "45 degrees" to match pattern
            angle_str = angle.replace("-degree", " degrees") if "-degree" in angle else angle
            prompt_parts.append(f"{angle_str} camera angle")

        # Technical specs for high fidelity
        prompt_parts.append("high detail")
        prompt_parts.append("material focus")
        prompt_parts.append("commercial product photography")
        prompt_parts.append("white background")  # Also triggers SPECIFIC_REQUEST

        # Product description - AVOID "show" to prevent BASIC_DIRECTION
        prompt_parts.append(f"featuring {product.lower()}")
        prompt_parts.append("cleaning product in action")

        # Add visual details from features
        if "surface_material" in features:
            material = features["surface_material"]
            prompt_parts.append(f"on {material.lower()} surface")
            prompt_parts.append(f"luxury {material.lower()} texture")

        # Add color temperature
        if "color_temperature" in features:
            temp = features["color_temperature"]
            prompt_parts.append(f"{temp.lower()} tone")
            prompt_parts.append(f"{temp.lower()} color palette")

        # Add product position with context
        if "product_position" in features:
            position = features["product_position"]
            if position == "Centered":
                prompt_parts.append("centered framing")
                prompt_parts.append("strong focal point")
            elif position == "Rule-of-Thirds":
                prompt_parts.append("rule-of-thirds composition")
                prompt_parts.append("dynamic visual balance")

        # Add mood and context elements (helps quality score)
        prompt_parts.append("premium quality aesthetic")
        prompt_parts.append("professional studio photography")
        prompt_parts.append("accurate color reproduction")
        prompt_parts.append("sharp focus")
        prompt_parts.append("fine details visible")

        # BACKGROUND DESCRIPTION - Use surface_material for rich background details
        if "surface_material" in features:
            material = features["surface_material"]
            if material == "Marble":
                prompt_parts.append("on polished white marble surface with subtle grey veining")
                prompt_parts.append("luxurious marble backdrop with natural texture")
                prompt_parts.append("soft reflections on marble surface")
                prompt_parts.append("seamless marble background extending to edges")
            elif material == "Wood":
                prompt_parts.append("on natural wood surface with visible grain")
                prompt_parts.append("warm wood backdrop with organic texture")
                prompt_parts.append("home environment background with wood accents")
            elif material == "Glass":
                prompt_parts.append("on glossy glass surface with subtle reflections")
                prompt_parts.append("transparent background element with light refraction")
                prompt_parts.append("sleek modern glass backdrop")
            else:
                prompt_parts.append(f"on premium {material.lower()} surface")
                prompt_parts.append(f"{material.lower()} texture background with detail")
        else:
            # Default studio background
            prompt_parts.append("on pure white seamless background")
            prompt_parts.append("studio infinity curve backdrop")
            prompt_parts.append("clean white background with soft grey vignette at edges")

        # Add shadow/reflection details
        prompt_parts.append("soft contact shadow grounding product")
        prompt_parts.append("natural product shadow cast by lighting")

        # Add negative space for text overlay
        prompt_parts.append("generous negative space in upper regions for text overlay")
        prompt_parts.append("balanced composition with breathing room")

        # Add purpose/context for quality
        prompt_parts.append("for commercial advertising")
        prompt_parts.append("premium brand presentation")

        # ANTI-HALLUCINATION CONSTRAINTS - Preserve logo and brand identity
        # These constraints prevent the model from modifying critical brand elements
        metadata = self.patterns.get("metadata", {})
        product_name = metadata.get("product", "product")

        prompt_parts.append(f"Brand identity preservation: {product} logo must remain exactly as shown")
        prompt_parts.append("Do NOT modify product appearance, logo text, or brand colors")
        prompt_parts.append("Preserve ALL text elements exactly - no text changes or hallucinations")
        prompt_parts.append("Do NOT create variations or redesigns of the product or packaging")
        prompt_parts.append("Logo and brand markings must be accurate and legible")
        prompt_parts.append("Follow brand guidelines exactly - maintain brand integrity")

        # Build final prompt - use period separation for clarity
        base_prompt = ". ".join(prompt_parts) + "."

        # DEBUG: Print raw base prompt to see what we're sending to nano agent
        import sys
        if '--debug-prompt' in sys.argv:
            print(f"\n{'='*80}\nRAW BASE PROMPT ({len(base_prompt)} chars):\n{'='*80}\n{base_prompt}\n{'='*80}\n", file=sys.stderr)

        # Apply nano agent enhancement if enabled
        if self.use_enhancement:
            try:
                logger.debug(f"Enhancing prompt with nano agent...")
                enhanced_prompt = enhance_prompt(base_prompt)
                logger.debug(f"Enhancement complete")
                return enhanced_prompt
            except Exception as e:
                logger.warning(f"Nano agent enhancement failed: {e}")
                logger.warning(f"Falling back to basic prompt")
                return base_prompt

        return base_prompt

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
