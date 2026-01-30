"""
Stage 3: The Upscaler - V1.8 Multi-Workflow CoT Upscaler

Transforms raw tags into high-fidelity descriptions using intelligent workflow selection.
Supports 6 workflow templates with quality validation and fallback strategies.
"""
import logging
import yaml
import hashlib
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowMode(Enum):
    """Workflow selection modes."""
    AUTO = "auto"           # Automatic selection based on triggers
    MANUAL = "manual"       # User-specified workflow
    MULTI = "multi"         # Combine multiple workflows


@dataclass
class QualityMetrics:
    """Quality metrics for expanded tokens."""
    token_count: int
    technical_term_count: int
    keyword_match_count: int
    clarity_score: float
    brand_alignment: float
    overall_score: float


class CoTUpscaler:
    """
    V1.8 Multi-Workflow Chain-of-Thought Upscaler.

    Transforms raw tags into high-fidelity descriptions using intelligent workflow selection.

    Core Philosophy: Information Density = Image Fidelity
    """

    # Technical terms for quality validation
    TECHNICAL_TERMS = [
        "texture", "finish", "grain", "vein", "polished", "matte", "gloss",
        "reflection", "refraction", "spectral", "volumetric", "ambient",
        "microscopic", "crystalline", "anisotropic", "porosity"
    ]

    def __init__(
        self,
        llm_client,
        workflow_templates: dict,
        fidelity_config: dict,
        brand_guidelines: Optional[dict] = None
    ):
        """
        Initialize upscaler with multi-workflow support.

        Args:
            llm_client: LLM client for CoT expansion
            workflow_templates: Workflow templates dict from customer config (Part J)
            fidelity_config: Fidelity configuration from customer config
            brand_guidelines: Optional brand guidelines for brand_consistent workflow
        """
        self.llm_client = llm_client
        self.fidelity_config = fidelity_config
        self.brand_guidelines = brand_guidelines or {}

        # Workflow templates from customer config
        self.workflow_templates = workflow_templates

        logger.info(f"Loaded {len(self.workflow_templates)} workflow templates")

        # Initialize cache
        self.expansion_cache: Dict[str, str] = {}
        if fidelity_config.get("enable_cache", True):
            self._load_cache()

    def select_workflow(
        self,
        product_type: str,
        campaign_goal: str,
        daily_budget_cents: int,
        product_margin: str,
        brand_maturity: str,
        manual_workflow: Optional[str] = None
    ) -> str:
        """
        Select appropriate workflow based on context.

        Args:
            product_type: Product category
            campaign_goal: Campaign objective
            daily_budget_cents: Daily budget
            product_margin: Product margin level
            brand_maturity: Brand maturity stage
            manual_workflow: Manually specified workflow

        Returns:
            Selected workflow name
        """
        # Priority 1: Manual selection
        if manual_workflow:
            if manual_workflow in self.workflow_templates:
                logger.info(f"Using manually selected workflow: {manual_workflow}")
                return manual_workflow
            else:
                logger.warning(f"Manual workflow '{manual_workflow}' not found, using auto-selection")

        # Priority 2: Automatic selection based on triggers
        workflow_mode = self.fidelity_config.get("workflow_mode", "auto")

        if workflow_mode == WorkflowMode.MANUAL.value:
            # No manual workflow specified, fallback to first available
            return next(iter(self.workflow_templates.keys()))

        # Score each workflow based on trigger conditions
        workflow_scores = []

        for workflow_name, workflow_config in self.workflow_templates.items():
            score = 0.0
            triggers = workflow_config.get("trigger_conditions", {})

            # Check each trigger condition
            if "campaign_goal" in triggers:
                if campaign_goal in triggers["campaign_goal"]:
                    score += 3.0  # High weight for goal match

            if "daily_budget_cents" in triggers:
                if daily_budget_cents >= triggers["daily_budget_cents"]:
                    score += 2.0

            if "product_margin" in triggers:
                if triggers["product_margin"] == "any" or triggers["product_margin"] == product_margin:
                    score += 1.5

            if "brand_maturity" in triggers:
                if triggers["brand_maturity"] == "any" or triggers["brand_maturity"] == brand_maturity:
                    score += 1.0

            # Add priority bonus
            priority = workflow_config.get("priority", 999)
            score -= (priority * 0.1)  # Lower priority number = higher priority

            workflow_scores.append((workflow_name, score))

        # Sort by score (descending) and select top
        workflow_scores.sort(key=lambda x: x[1], reverse=True)
        selected_workflow = workflow_scores[0][0]

        logger.info(
            f"Auto-selected workflow: {selected_workflow} "
            f"(score: {workflow_scores[0][1]:.2f})"
        )

        return selected_workflow

    def expand_token(
        self,
        raw_tag: str,
        feature_name: str,
        workflow_name: str,
        context: dict
    ) -> str:
        """
        Expand token using specified workflow.

        Args:
            raw_tag: Raw tag value
            feature_name: Feature name
            workflow_name: Selected workflow
            context: Additional context (product, goal, etc.)

        Returns:
            Expanded token
        """
        # Check cache first
        cache_key = self._generate_cache_key(raw_tag, feature_name, workflow_name)
        if cache_key in self.expansion_cache:
            logger.debug(f"Cache hit for {raw_tag} ({feature_name})")
            return self.expansion_cache[cache_key]

        # Get workflow configuration
        workflow = self.workflow_templates.get(workflow_name)
        if not workflow:
            logger.warning(f"Workflow '{workflow_name}' not found, using fallback")
            workflow = self.workflow_templates.get("fallback_standard", {})

        # Select appropriate CoT prompt for this feature
        cot_prompts = workflow.get("cot_prompts", {})

        # Map feature names to CoT prompt types
        prompt_type = self._map_feature_to_prompt_type(feature_name)

        if prompt_type not in cot_prompts:
            logger.warning(f"No CoT prompt for {prompt_type}, using standard")
            prompt_type = "standard_expansion"

        cot_config = cot_prompts[prompt_type]
        prompt_template = cot_config.get("prompt_template", "")

        # Build prompt with context
        prompt = self._build_prompt(
            prompt_template,
            raw_tag=raw_tag,
            feature_name=feature_name,
            context=context,
            examples=cot_config.get("examples", [])
        )

        # Call LLM with retry logic
        max_retries = self.fidelity_config.get("max_retries", 2)
        expanded = None

        for attempt in range(max_retries + 1):
            try:
                response = self._call_llm(prompt)
                expanded = response.strip().strip('"')

                # Validate quality
                if self.fidelity_config.get("enable_quality_validation", True):
                    quality = self._validate_quality(expanded, workflow)

                    if quality.overall_score < self.fidelity_config.get("min_quality_score", 0.7):
                        logger.warning(
                            f"Quality validation failed (score: {quality.overall_score:.2f}), "
                            f"retry {attempt + 1}/{max_retries}"
                        )
                        if attempt < max_retries:
                            continue

                break

            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    continue
                else:
                    # Use fallback
                    expanded = self._apply_fallback_expansion(raw_tag, feature_name)

        # Cache result
        self.expansion_cache[cache_key] = expanded

        return expanded

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API (to be implemented by specific client).

        Args:
            prompt: The prompt to send

        Returns:
            LLM response text
        """
        # This is a placeholder - actual implementation depends on LLM client
        # For now, return a simple expansion
        if hasattr(self.llm_client, 'generate'):
            return self.llm_client.generate(
                prompt,
                temperature=self.fidelity_config.get("temperature", 0.3),
                max_tokens=self.fidelity_config.get("max_tokens", 200),
                timeout=self.fidelity_config.get("timeout_seconds", 30)
            )
        else:
            # Fallback for testing
            logger.warning("LLM client has no 'generate' method, using mock response")
            return f"Premium {prompt.split(':')[0].strip() if ':' in prompt else prompt} with refined finish"

    def _map_feature_to_prompt_type(self, feature_name: str) -> str:
        """Map feature name to CoT prompt type."""
        mapping = {
            "surface_material": "texture_analysis",
            "lighting_style": "lighting_physics",
            "lighting_type": "lighting_physics",
            "product_context": "material_science",
            "product_position": "product_clarity",
            "direction": "product_clarity",
            "primary_colors": "color_system",
        }
        return mapping.get(feature_name, "standard_expansion")

    def _build_prompt(
        self,
        template: str,
        raw_tag: str,
        feature_name: str,
        context: dict,
        examples: list
    ) -> str:
        """Build complete CoT prompt from template."""
        # Add examples if available
        examples_section = ""
        if examples:
            examples_section = "\nExamples:\n"
            for example in examples[:3]:  # Max 3 examples
                examples_section += f"- {example}\n"

        # Build prompt
        prompt = template.format(
            raw_value=raw_tag,
            feature_name=feature_name,
            product_type=context.get("product_type", "unknown"),
            brand_name=context.get("brand_name", ""),
            brand_guidelines=str(context.get("brand_guidelines", {})),
            color_system=str(context.get("color_system", {})),
            primary_colors=str(context.get("primary_colors", [])),
            secondary_colors=str(context.get("secondary_colors", [])),
        )

        # Add examples section
        if examples_section:
            prompt += "\n" + examples_section

        # Add output instruction
        prompt += "\nOutput: A clear, detailed description (30-100 words)."

        return prompt

    def _validate_quality(self, expanded: str, workflow: dict) -> QualityMetrics:
        """
        Validate quality of expanded token.

        Args:
            expanded: Expanded token text
            workflow: Workflow configuration

        Returns:
            Quality metrics
        """
        # Get quality thresholds
        thresholds = workflow.get("quality_thresholds", {})

        # Calculate metrics
        token_count = len(expanded.split())

        # Count technical terms
        technical_term_count = sum(
            1 for term in self.TECHNICAL_TERMS
            if term.lower() in expanded.lower()
        )

        # Check required keywords
        required_keywords = thresholds.get("required_keywords", [])
        keyword_match_count = sum(
            1 for kw in required_keywords
            if kw.lower() in expanded.lower()
        )

        # Calculate scores
        token_score = 1.0 if token_count >= thresholds.get("min_token_count", 20) else 0.5
        technical_score = min(
            1.0,
            technical_term_count / max(1, thresholds.get("min_technical_terms", 3))
        )
        keyword_score = min(
            1.0,
            keyword_match_count / max(1, len(required_keywords))
        )

        # Overall score (weighted average)
        overall_score = (token_score * 0.3 + technical_score * 0.4 + keyword_score * 0.3)

        return QualityMetrics(
            token_count=token_count,
            technical_term_count=technical_term_count,
            keyword_match_count=keyword_match_count,
            clarity_score=token_score,
            brand_alignment=0.8,  # Placeholder (could use VLM for real validation)
            overall_score=overall_score
        )

    def _apply_fallback_expansion(self, raw_tag: str, feature_name: str) -> str:
        """Apply fallback expansion strategy."""
        fallback_strategy = self.fidelity_config.get("fallback_strategy", "standard")

        if fallback_strategy == "cached":
            # Try to find cached expansion from any workflow
            for workflow_name in self.workflow_templates:
                cache_key = self._generate_cache_key(raw_tag, feature_name, workflow_name)
                if cache_key in self.expansion_cache:
                    logger.info(f"Using cached expansion from {workflow_name}")
                    return self.expansion_cache[cache_key]

        # Use standard fallback expansion
        fallback_map = {
            "surface_material": f"Premium {raw_tag} texture with refined finish",
            "lighting_style": f"Professional {raw_tag} lighting setup",
            "lighting_type": f"Soft {raw_tag} illumination",
            "product_context": f"Clean {raw_tag} composition",
            "product_position": f"Product positioned at {raw_tag}",
            "direction": f"{raw_tag} camera angle",
        }

        fallback = fallback_map.get(feature_name, raw_tag)
        logger.warning(f"Using fallback expansion: {fallback}")

        return fallback

    def _generate_cache_key(self, raw_tag: str, feature_name: str, workflow: str) -> str:
        """Generate cache key for expansion."""
        key_string = f"{raw_tag}|{feature_name}|{workflow}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _load_cache(self):
        """Load expansion cache from disk."""
        cache_path = Path("config/ad/miner/cache/expansion_cache.json")
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    self.expansion_cache = json.load(f)
                logger.info(f"Loaded {len(self.expansion_cache)} cached expansions")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")

    def _save_cache(self):
        """Save expansion cache to disk."""
        cache_path = Path("config/ad/miner/cache/expansion_cache.json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(cache_path, "w") as f:
                json.dump(self.expansion_cache, f, indent=2)
            logger.info(f"Saved {len(self.expansion_cache)} cached expansions")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def expand_locked_combination(
        self,
        locked_combination: dict,
        context: dict
    ) -> dict:
        """
        Expand all tokens in a locked combination using selected workflow.

        Args:
            locked_combination: Locked combination from synthesizer
            context: Context dict for workflow selection

        Returns:
            Dict with expanded tokens and quality metrics
        """
        # Select workflow
        workflow_name = self.select_workflow(
            product_type=context.get("product_type", "unknown"),
            campaign_goal=context.get("campaign_goal", "conversion"),
            daily_budget_cents=context.get("daily_budget_cents", 10000),
            product_margin=context.get("product_margin", "medium"),
            brand_maturity=context.get("brand_maturity", "growing"),
            manual_workflow=self.fidelity_config.get("selected_workflow")
        )

        logger.info(f"Selected workflow: {workflow_name}")

        # Expand tokens
        expanded = {
            "workflow_used": workflow_name,
            "workflow_priority": self.workflow_templates[workflow_name].get("priority", 999)
        }

        # Expand primary value
        primary_feature = locked_combination.get("primary_feature")
        primary_value = locked_combination.get("primary_value")

        if primary_feature and primary_value:
            expanded_primary = self.expand_token(
                primary_value,
                primary_feature,
                workflow_name,
                context
            )
            expanded["primary_value_expanded"] = expanded_primary
            expanded["primary_feature"] = primary_feature

        # Expand secondary value
        secondary_feature = locked_combination.get("secondary_feature")
        secondary_value = locked_combination.get("secondary_value")

        if secondary_feature and secondary_value:
            expanded_secondary = self.expand_token(
                secondary_value,
                secondary_feature,
                workflow_name,
                context
            )
            expanded["secondary_value_expanded"] = expanded_secondary
            expanded["secondary_feature"] = secondary_feature

        # Copy other fields
        for key, value in locked_combination.items():
            if key not in expanded:
                expanded[key] = value

        return expanded

    def fill_prompt_template(
        self,
        expanded_combination: dict,
        workflow_name: str,
        additional_features: dict
    ) -> dict:
        """
        Fill expanded tokens into workflow-specific template structure.

        Args:
            expanded_combination: Expanded locked combination
            workflow_name: Workflow used for expansion
            additional_features: Additional raw tags to expand

        Returns:
            Complete prompt_slots dict
        """
        workflow = self.workflow_templates.get(workflow_name, {})
        template_structure = workflow.get("template_structure", "")

        prompt_slots = {}

        # Fill standard slots
        prompt_slots["quality_headers"] = self._get_quality_headers(workflow_name)

        # Fill from expanded combination
        if "primary_value_expanded" in expanded_combination:
            primary_feature = expanded_combination["primary_feature"]
            expanded_value = expanded_combination["primary_value_expanded"]

            if primary_feature == "surface_material":
                prompt_slots["surface_material_expanded"] = expanded_value
            elif primary_feature == "product_context":
                prompt_slots["scene_subject_expanded"] = expanded_value

        if "secondary_value_expanded" in expanded_combination:
            secondary_feature = expanded_combination["secondary_feature"]
            expanded_value = expanded_combination["secondary_value_expanded"]

            if secondary_feature in ["lighting_style", "lighting_type"]:
                prompt_slots["lighting_atmosphere_expanded"] = expanded_value

        # Expand additional features
        for feature, value in additional_features.items():
            if feature == "product_position":
                prompt_slots["camera_technical"] = self._expand_position_to_camera(value)
            elif feature == "direction":
                prompt_slots["camera_technical"] = self._expand_direction_to_camera(value)

        # Build final prompt from template
        final_prompt = self._build_final_prompt(template_structure, prompt_slots)
        prompt_slots["final_prompt"] = final_prompt

        return prompt_slots

    def _get_quality_headers(self, workflow_name: str) -> str:
        """Get quality headers for workflow."""
        quality_map = {
            "ultra_realistic": "Raw photo, 8k uhd, masterpiece, hyperrealistic, sharp focus, high detail, microscopic clarity",
            "brand_consistent": "Professional photo, on-brand, high quality, consistent style, brand-aligned",
            "performance_optimized": "Conversion-optimized photo, clear product, professional quality, trust-building",
            "lifestyle_aspirational": "Emotional lifestyle photo, authentic feeling, aspirational mood, cinematic quality",
            "minimalist_clean": "Clean minimalist photo, sharp detail, balanced composition, modern aesthetic",
            "fallback_standard": "Raw photo, 8k uhd, masterpiece, hyperrealistic, sharp focus",
        }
        return quality_map.get(workflow_name, quality_map["fallback_standard"])

    def _build_final_prompt(self, template: str, slots: dict) -> str:
        """Build final prompt by filling slots into template."""
        # Replace slot placeholders with actual values
        prompt = template

        # Map slot types to slot keys
        slot_mapping = {
            "quality": "quality_headers",
            "subject": "scene_subject_expanded",
            "scene": "scene_subject_expanded",
            "lighting": "lighting_atmosphere_expanded",
            "light": "lighting_atmosphere_expanded",
            "material": "surface_material_expanded",
            "camera": "camera_technical",
        }

        for slot_key, slot_value in slots.items():
            if slot_key == "final_prompt":
                continue

            # Find and replace in template
            for template_type, slot_key_name in slot_mapping.items():
                if slot_key_name == slot_key and slot_value:
                    # Replace [TemplateType: Description] with actual value
                    pattern = rf'\[{template_type}:[^\]]+\]'
                    prompt = re.sub(pattern, slot_value, prompt)

        return prompt

    def _expand_position_to_camera(self, position: str) -> str:
        """Expand product position to camera technical."""
        camera_map = {
            "bottom-right": "Shot on Phase One XF IQ4, 80mm lens, f/2.8, shallow depth of field, rule of thirds composition, product in lower-right power position",
            "center": "Shot on Phase One XF IQ4, 50mm lens, f/5.6, balanced composition, centered framing, equal weight distribution",
            "top-left": "Shot on Phase One XF IQ4, 35mm lens, f/8, deep depth of field, dynamic top-left placement",
        }
        return camera_map.get(position, "Shot on Phase One XF IQ4, high resolution, professional quality")

    def _expand_direction_to_camera(self, direction: str) -> str:
        """Expand direction to camera technical."""
        direction_map = {
            "overhead": "Top-down aerial view, 90-degree angle",
            "45-degree": "Three-quarter angle, dynamic perspective",
            "front": "Frontal view, eye-level angle",
            "side": "Side profile, 90-degree angle",
        }
        return direction_map.get(direction, "Dynamic angle")

    def run(
        self,
        locked_combinations: dict,
        losers_df: pd.DataFrame,
        raw_tags: dict,
        context: dict
    ) -> dict:
        """
        Run complete upscaler pipeline.

        Args:
            locked_combinations: Locked combinations from synthesizer
            losers_df: Loser creatives DataFrame
            raw_tags: Raw tags from miner
            context: Context for workflow selection

        Returns:
            Complete master_blueprint dict
        """
        logger.info("Starting Stage 3: The Upscaler")

        blueprint = {
            "meta_info": {
                "recipe_id": "recipe_hifi_v1.8",
                "mining_strictness": context.get("mining_strictness", "Top_10_Percent"),
                "fidelity_expanded": True,
            },
            "strategy_rationale": {},
            "nano_generation_rules": {
                "prompt_template_structure": "",
                "prompt_slots": {},
                "negative_prompt": [],
                "inference_config": {
                    "steps": 8,
                    "cfg_scale": 1.5,
                    "width": 1024,
                    "height": 1024
                }
            },
            "python_compositing_rules": {
                "shadow_logic": {}
            }
        }

        # Process each locked combination
        for combo_key, combo_data in locked_combinations.items():
            # Expand locked combination
            expanded = self.expand_locked_combination(
                combo_data["locked_combination"],
                context
            )

            # Add to strategy rationale
            blueprint["strategy_rationale"][combo_key] = {
                "locked_combination": combo_data["locked_combination"],
                "expanded_combination": expanded,
                "confidence_score": combo_data["confidence_score"],
                "reasoning": combo_data.get("reasoning", "")
            }

            # Fill prompt template
            additional_features = {k: v for k, v in raw_tags.items()
                                   if k not in ["surface_material", "lighting_style",
                                               "lighting_type", "product_context"]}
            prompt_slots = self.fill_prompt_template(expanded, expanded.get("workflow_used", "fallback_standard"), additional_features)

            blueprint["nano_generation_rules"]["prompt_slots"].update(prompt_slots)

            # Add negative prompts from losers
            negative_tags = self._extract_loser_tags(losers_df)
            blueprint["nano_generation_rules"]["negative_prompt"].extend(negative_tags)

            # Add compositing rules
            if "lighting" in combo_key.lower():
                blueprint["python_compositing_rules"]["shadow_logic"] = {
                    "type": "Hard_Contact_Shadow",
                    "direction": self._infer_shadow_direction(expanded)
                }

        logger.info("Stage 3: The Upscaler completed")

        return blueprint

    def _extract_loser_tags(self, losers_df: pd.DataFrame) -> list:
        """Extract common tags from losers for negative prompts."""
        negative_tags = []

        # Find features overrepresented in losers
        for feature in ["lighting_style", "background_content_type", "color_balance"]:
            if feature in losers_df.columns:
                mode_value = losers_df[feature].mode()
                if len(mode_value) > 0:
                    negative_tags.append(str(mode_value[0]))

        # Add standard negatives
        negative_tags.extend(["cgi", "3d render", "cartoon", "blurry", "pixelated", "jpeg artifacts"])

        return negative_tags

    def _infer_shadow_direction(self, expanded: dict) -> str:
        """Infer shadow direction from lighting."""
        lighting = expanded.get("secondary_value_expanded", "").lower()

        if "left" in lighting:
            return "Left_to_Right"
        elif "right" in lighting:
            return "Right_to_Left"
        else:
            return "Omni_directional"
