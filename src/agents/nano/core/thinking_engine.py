"""
Thinking Engine (Strategy) for Nano Banana Pro Agent.

Generates thinking blocks and plans which NB techniques to apply.
Enhanced with feature extraction and comprehensive analysis.
"""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Optional

from src.agents.nano.core.types import (
    ThinkingBlock,
    PromptCategory,
    PromptIntent,
    AgentInput,
)


logger = logging.getLogger(__name__)


class ThinkingEngine:
    """
    Generate strategic thinking and plan technique application.

    Analyzes the request and decides:
    - Which NB techniques to apply
    - What risks exist
    - How to mitigate risks
    - What the output should accomplish

    Enhanced with:
    - Feature extraction from prompts
    - Emotion and setting detection
    - More sophisticated risk assessment
    - Comprehensive mitigation strategies
    """

    # Risk patterns to check for
    RISK_PATTERNS = {
        "hallucination": [
            "creative", "artistic", "interpretive", "stylized",
        ],
        "inconsistency": [
            "sequence", "story", "series", "multiple",
        ],
        "loss_brand_integrity": [
            "creative", "unique", "different",
        ],
        "unrealistic_physics": [
            "liquid", "water", "pouring", "falling", "floating",
        ],
        "text_illegibility": [
            "text", "title", "label", "chart", "infographic",
        ],
    }

    # Feature patterns for better analysis
    FEATURE_PATTERNS = {
        "product_features": [
            r"\d+°?\s*(?:degree|rotation)?",
            r"spray",
            r"microfiber",
            r"adjustable",
            r"extendable",
            r"cordless",
            r"lightweight",
            r"ergonomic",
            r"rechargeable",
        ],
        "emotional_keywords": [
            r"happy",
            r"satisfied",
            r"relieved",
            r"proud",
            r"excited",
            r"confident",
            r"relaxed",
            r"joyful",
        ],
        "setting_keywords": [
            r"kitchen",
            r"bathroom",
            r"living room",
            r"bedroom",
            r"outdoor",
            r"office",
            r"garage",
        ],
    }

    # Example thinking blocks for different scenarios (few-shot learning)
    EXAMPLE_BLOCKS = {
        "product_photo": """Analysis: Request: product photography. Input Category: ultra simple. Product: 360° Mop. Detected Features: 360°, spray. Complexity: simple. Enhancement: Needs comprehensive visual detail, lighting, and composition.

Techniques to apply:
  - Natural language prompting (conversational, descriptive)
  - Character consistency (product)
  - High-resolution output (4K)
  - Physics understanding (material properties)

Risks:
  - vague_input - prompt lacks specific details
  - generic - may produce generic results

Mitigation:
  - Add specific descriptive details: lighting, camera angle, composition
  - Add comprehensive visual descriptions and context

Context: Target audience: E-commerce customers. Emotional goal: Confidence. Purpose: Show product clearly.""",

        "lifestyle": """Analysis: Request: lifestyle advertisement. Input Category: basic direction. Product: 360° Mop. Brand: CleanHome. Complexity: moderate. Enhancement: Needs specific style, atmosphere, and technical specs.

Techniques to apply:
  - Natural language prompting (conversational, descriptive)
  - Natural environment setting
  - Human emotion rendering
  - Authenticity over perfection

Risks:
  - stock_photo_appearance - may look inauthentic
  - style_inconsistent - style may vary

Mitigation:
  - Add authenticity imperfections: natural shadows
  - Define style explicitly: modern, clean
  - Include lived-in details

Context: Target audience: Homeowners, families. Emotional goal: Trust, satisfaction. Purpose: Demonstrate benefits.""",
    }

    def __init__(self):
        """Initialize the thinking engine."""
        self.technique_priority = {
            "natural_language": 1,
            "text_rendering": 2,
            "character_consistency": 2,
            "physics_understanding": 3,
            "dimensional_translation": 3,
            "search_grounding": 4,
            "storyboarding": 4,
            "structural_control": 5,
        }

    def generate_thinking(
        self,
        agent_input: AgentInput,
        category: PromptCategory,
        intent: PromptIntent,
        missing_elements: List[str],
        analysis: Dict[str, Any],
    ) -> ThinkingBlock:
        """
        Generate a thinking block for the prompt.

        Args:
            agent_input: The enriched agent input
            category: Detected prompt category
            intent: Detected intent
            missing_elements: List of what's missing from input
            analysis: Deep analysis results

        Returns:
            ThinkingBlock with analysis, techniques, risks, mitigation
        """

        # What is this request about?
        analysis_text = self._generate_analysis(
            agent_input, category, intent, analysis
        )

        # Which techniques should we apply?
        techniques = self._select_techniques(
            category, intent, missing_elements, analysis
        )

        # What could go wrong?
        risks = self._assess_risks(
            agent_input.generic_prompt, category, intent
        )

        # How do we prevent issues?
        mitigation = self._plan_mitigation(risks, category)

        # For whom and why?
        context = self._generate_context(agent_input, intent)

        return ThinkingBlock(
            analysis=analysis_text,
            techniques=techniques,
            risks=risks,
            mitigation=mitigation,
            context=context,
        )

    def _generate_analysis(
        self,
        agent_input: AgentInput,
        category: PromptCategory,
        intent: PromptIntent,
        analysis: Dict[str, Any],
    ) -> str:
        """Generate the analysis section of the thinking block."""

        parts = []

        # Request type
        parts.append(f"Request: {intent.value.replace('_', ' ')}")

        # Category
        parts.append(f"Input Category: {category.value} ({self._get_category_description(category)})")

        # Product context
        if agent_input.product_context:
            parts.append(
                f"Product: {agent_input.product_context.name} "
                f"({agent_input.product_context.category})"
            )
            if agent_input.product_context.key_features:
                parts.append(
                    f"Key Features: {', '.join(agent_input.product_context.key_features[:3])}"
                )
            else:
                # Try to extract features from prompt
                extracted_features = self._extract_features_from_prompt(
                    agent_input.generic_prompt
                )
                if extracted_features:
                    parts.append(f"Detected Features: {', '.join(extracted_features)}")

        # Brand context
        if agent_input.brand_guidelines:
            parts.append(
                f"Brand: {agent_input.brand_guidelines.brand_name} "
                f"- {agent_input.brand_guidelines.visual_language}"
            )

        # Extract emotional cues
        emotions = self._extract_emotions(agent_input.generic_prompt)
        if emotions:
            parts.append(f"Emotional Cues: {', '.join(emotions)}")

        # Extract setting
        setting = self._extract_setting(agent_input.generic_prompt)
        if setting:
            parts.append(f"Setting: {setting}")

        # Complexity assessment
        complexity = analysis.get("complexity", "simple")
        parts.append(f"Complexity: {complexity}")

        # Add enhancement suggestions based on category
        if category == PromptCategory.ULTRA_SIMPLE:
            parts.append("Enhancement: Needs comprehensive visual detail, lighting, and composition")
        elif category == PromptCategory.BASIC_DIRECTION:
            parts.append("Enhancement: Needs specific style, atmosphere, and technical specs")
        elif category == PromptCategory.SEQUENTIAL:
            parts.append("Enhancement: Needs consistency management and emotional arc")

        return ". ".join(parts) + "."

    def _extract_features_from_prompt(self, prompt: str) -> List[str]:
        """Extract product features from prompt using regex patterns."""
        features = []

        for pattern in self.FEATURE_PATTERNS["product_features"]:
            matches = re.finditer(pattern, prompt, re.IGNORECASE)
            for match in matches:
                features.append(match.group())

        return list(set(features))  # Remove duplicates

    def _extract_emotions(self, prompt: str) -> List[str]:
        """Extract emotional keywords from prompt."""
        emotions = []

        for pattern in self.FEATURE_PATTERNS["emotional_keywords"]:
            if re.search(pattern, prompt, re.IGNORECASE):
                emotions.append(pattern)

        return emotions

    def _extract_setting(self, prompt: str) -> Optional[str]:
        """Extract setting/environment from prompt."""
        for pattern in self.FEATURE_PATTERNS["setting_keywords"]:
            if re.search(pattern, prompt, re.IGNORECASE):
                return pattern

        return None

    def _select_techniques(
        self,
        category: PromptCategory,
        intent: PromptIntent,
        missing_elements: List[str],
        analysis: Dict[str, Any],
    ) -> List[str]:
        """Select which NB techniques to apply."""

        techniques = []

        # Always use natural language
        techniques.append("Natural language prompting (conversational, descriptive)")

        # Category-based technique selection
        if category == PromptCategory.COMPARATIVE:
            techniques.extend([
                "Text rendering & infographics",
                "Data visualization",
                "Structural control (layout precision)",
            ])

        elif category == PromptCategory.SEQUENTIAL:
            techniques.extend([
                "One-shot storyboarding",
                "Character consistency (identity locking)",
                "Emotional progression",
            ])

        elif category == PromptCategory.TECHNICAL:
            techniques.extend([
                "Text rendering (technical fonts)",
                "Dimensional translation (2D → 3D)",
                "Structural control (blueprints)",
            ])

        # Intent-based technique selection
        if intent == PromptIntent.PRODUCT_PHOTOGRAPHY:
            techniques.extend([
                "Character consistency (product)",
                "High-resolution output (4K)",
                "Physics understanding (material properties)",
            ])

        elif intent == PromptIntent.LIFESTYLE_ADVERTISEMENT:
            techniques.extend([
                "Natural environment setting",
                "Human emotion rendering",
                "Authenticity over perfection",
            ])

        # Analysis-based technique selection
        if analysis.get("needs_physics"):
            techniques.append("Physics understanding (liquids, gravity, cloth)")

        if analysis.get("needs_text_overlay"):
            techniques.append("Text rendering optimization")

        if analysis.get("needs_human_element"):
            techniques.append("Character consistency (person)")

        if analysis.get("is_sequential"):
            techniques.append("Storyboard consistency management")

        # Sort by priority
        techniques = self._sort_by_priority(techniques)

        return techniques

    def _sort_by_priority(self, techniques: List[str]) -> List[str]:
        """Sort techniques by priority (lower number = higher priority)."""

        def get_priority(technique: str) -> int:
            for key, priority in self.technique_priority.items():
                if key in technique.lower():
                    return priority
            return 999  # Unknown techniques get lowest priority

        return sorted(techniques, key=get_priority)

    def _assess_risks(
        self,
        prompt: str,
        category: PromptCategory,
        intent: PromptIntent,
    ) -> List[str]:
        """Assess potential risks for this request."""

        risks = []
        prompt_lower = prompt.lower()

        # Check for risk patterns
        for risk_type, patterns in self.RISK_PATTERNS.items():
            for pattern in patterns:
                if pattern in prompt_lower:
                    risks.append(f"{risk_type.replace('_', ' ')} - '{pattern}' detected")
                    break

        # Category-specific risks with more detail
        if category == PromptCategory.ULTRA_SIMPLE:
            risks.append("vague_input - prompt lacks specific details, may produce unpredictable results")
            risks.append("lighting_undefined - no lighting specified, may use default studio lighting")

        elif category == PromptCategory.BASIC_DIRECTION:
            risks.append("style_inconsistent - style direction may be interpreted differently")
            risks.append("composition_generic - may lack strong focal point or visual hierarchy")

        elif category == PromptCategory.SEQUENTIAL:
            risks.append("consistency - maintaining consistency across multiple images is challenging")
            risks.append("emotional_discontinuity - emotional arc may not flow naturally between frames")

        elif category == PromptCategory.COMPARATIVE:
            risks.append("data_clarity - comparative data may not be visually clear")
            risks.append("bias_visual - visual comparison may appear biased or unfair")

        elif category == PromptCategory.TECHNICAL:
            risks.append("precision_loss - technical details may be approximated or simplified")
            risks.append("text_rendering - technical labels and text may be rendered incorrectly")

        # Intent-specific risks
        if intent == PromptIntent.LIFESTYLE_ADVERTISEMENT:
            risks.append("stock_photo_appearance - may look inauthentic or staged")
            risks.append("demographic_mismatch - models may not match target audience")

        elif intent == PromptIntent.PRODUCT_PHOTOGRAPHY:
            risks.append("material_accuracy - materials may not look realistic")
            risks.append("scale_distortion - product proportions may be distorted")

        elif intent == PromptIntent.COMPARATIVE_INFOGRAPHIC:
            risks.append("information_overload - too much data may reduce clarity")
            risks.append("visual_hierarchy - important points may not stand out")

        # Check for common anti-patterns in prompt
        if len(prompt.split()) < 10:
            risks.append("under_specified - prompt is very short, will need significant enhancement")

        if "?" in prompt and "!" not in prompt:
            risks.append("passive_tone - prompt may lack energy and emotion")

        # Default risk if nothing detected
        if not risks:
            risks.append("generic - prompt may produce generic results without specific direction")

        return list(set(risks))  # Remove duplicates

    def _plan_mitigation(
        self,
        risks: List[str],
        category: PromptCategory,
    ) -> List[str]:
        """Plan how to mitigate identified risks."""

        mitigations = []

        for risk in risks:
            risk_lower = risk.lower()

            if "hallucination" in risk_lower:
                mitigations.append("Add anti-hallucination guards (Do NOT add elements not explicitly requested)")
                mitigations.append("Use 'DO NOT invent' constraint for product features")

            elif "consistency" in risk_lower:
                mitigations.append("Use character consistency with reference images or seed locking")
                mitigations.append("Specify consistency requirements explicitly across all frames")

            elif "brand_integrity" in risk_lower:
                mitigations.append("Strict brand guideline enforcement with color and typography")
                mitigations.append("Add brand identity preservation constraint")

            elif "physics" in risk_lower:
                mitigations.append("Specify realistic physics parameters (gravity, material properties)")
                mitigations.append("Add 'must follow real-world physics' constraint")

            elif "text" in risk_lower:
                mitigations.append("Use explicit text rendering specifications with font and size")
                mitigations.append("Add 'text must be legible' constraint with high contrast")

            elif "vague" in risk_lower or "under_specified" in risk_lower:
                mitigations.append("Add specific descriptive details: lighting, camera angle, composition")
                mitigations.append("Include atmosphere, mood, and emotional tone")

            elif "stock_photo" in risk_lower:
                mitigations.append("Add authenticity imperfections: natural wrinkles, subtle shadows")
                mitigations.append("Include 'lived-in' details: slight mess, natural lighting variation")
                mitigations.append("Specify candid, non-posed human expressions and body language")

            elif "data_clarity" in risk_lower or "information_overload" in risk_lower:
                mitigations.append("Use clear data visualization with proper hierarchy and scale")
                mitigations.append("Apply 'less is more' principle - focus on 2-3 key points")
                mitigations.append("Add visual flow guides (arrows, numbered steps) if needed")

            elif "lighting_undefined" in risk_lower:
                mitigations.append("Specify lighting: time of day, light source quality, direction")
                mitigations.append("Add mood lighting appropriate to product category")

            elif "style_inconsistent" in risk_lower:
                mitigations.append("Define style explicitly: modern, traditional, minimalist, dramatic")
                mitigations.append("Add style reference with specific visual descriptors")

            elif "material_accuracy" in risk_lower:
                mitigations.append("Specify material properties: glossy, matte, metallic, fabric texture")
                mitigations.append("Add 'realistic material rendering' constraint with specific surface details")

            elif "emotional_discontinuity" in risk_lower:
                mitigations.append("Define emotional arc for each frame clearly")
                mitigations.append("Specify emotional progression: curiosity → interest → satisfaction")

        # Remove duplicates while preserving order
        seen = set()
        unique_mitigations = []
        for m in mitigations:
            if m not in seen:
                seen.add(m)
                unique_mitigations.append(m)

        # Add default mitigations based on category
        if category == PromptCategory.ULTRA_SIMPLE:
            if not any("specific descriptive" in m.lower() for m in unique_mitigations):
                unique_mitigations.append("Add comprehensive visual descriptions: lighting, composition, atmosphere")

        elif category == PromptCategory.SEQUENTIAL:
            if not any("consistency" in m.lower() for m in unique_mitigations):
                unique_mitigations.append("Add frame-by-frame consistency specifications")

        elif category == PromptCategory.COMPARATIVE:
            if not any("hierarchy" in m.lower() for m in unique_mitigations):
                unique_mitigations.append("Establish clear visual hierarchy with size and position")

        return unique_mitigations

    def _generate_context(self, agent_input: AgentInput, intent: PromptIntent) -> str:
        """Generate the 'for whom, why' context."""

        context_parts = []

        # Who is this for?
        if agent_input.target_audience:
            context_parts.append(f"Target audience: {agent_input.target_audience}")
        else:
            # Default audience based on intent
            if intent == PromptIntent.PRODUCT_PHOTOGRAPHY:
                context_parts.append("Target audience: E-commerce customers, potential buyers researching purchase")
            elif intent == PromptIntent.LIFESTYLE_ADVERTISEMENT:
                context_parts.append("Target audience: Homeowners, families seeking practical solutions")
            elif intent == PromptIntent.COMPARATIVE_INFOGRAPHIC:
                context_parts.append("Target audience: Decision-makers comparing options")
            elif intent == PromptIntent.STORYBOARD_SEQUENCE:
                context_parts.append("Target audience: Potential customers following a narrative")
            elif intent == PromptIntent.TECHNICAL_DIAGRAM:
                context_parts.append("Target audience: Technical users, support staff, installers")
            else:
                context_parts.append("Target audience: General consumers")

        # What emotion should it evoke?
        if agent_input.emotion_goal:
            context_parts.append(f"Emotional goal: {agent_input.emotion_goal}")
        else:
            # Default emotion based on intent with more detail
            if intent == PromptIntent.LIFESTYLE_ADVERTISEMENT:
                context_parts.append("Emotional goal: Trust, satisfaction, aspiration, relief from cleaning burden")
            elif intent == PromptIntent.PRODUCT_PHOTOGRAPHY:
                context_parts.append("Emotional goal: Confidence in product quality, professional trust")
            elif intent == PromptIntent.COMPARATIVE_INFOGRAPHIC:
                context_parts.append("Emotional goal: Confidence in decision, clarity, informed choice")
            elif intent == PromptIntent.STORYBOARD_SEQUENCE:
                context_parts.append("Emotional goal: Engagement, narrative connection, problem-solution satisfaction")
            elif intent == PromptIntent.BRAND_ASSET_GENERATION:
                context_parts.append("Emotional goal: Brand recognition, trust, professionalism")
            else:
                context_parts.append("Emotional goal: Professional, trustworthy, clear")

        # What action should they take?
        if intent == PromptIntent.PRODUCT_PHOTOGRAPHY:
            context_parts.append("Call to action: Examine product details, imagine ownership")
        elif intent == PromptIntent.LIFESTYLE_ADVERTISEMENT:
            context_parts.append("Call to action: See themselves using product, feel the benefit")
        elif intent == PromptIntent.COMPARATIVE_INFOGRAPHIC:
            context_parts.append("Call to action: Understand advantages, make informed choice")
        elif intent == PromptIntent.STORYBOARD_SEQUENCE:
            context_parts.append("Call to action: Follow the journey, experience transformation")

        # Why create this?
        context_parts.append(f"Purpose: {self._get_purpose(intent)}")

        # Add value proposition based on product/brand context
        if agent_input.product_context:
            if agent_input.product_context.key_features:
                context_parts.append(
                    f"Value proposition: {', '.join(agent_input.product_context.key_features[:2])} "
                    f"make life easier"
                )

        if agent_input.brand_guidelines:
            context_parts.append(
                f"Brand personality: {agent_input.brand_guidelines.visual_language} "
                f"builds trust and recognition"
            )

        return ". ".join(context_parts) + "."

    def _get_purpose(self, intent: PromptIntent) -> str:
        """Get the purpose based on intent."""

        purposes = {
            PromptIntent.PRODUCT_PHOTOGRAPHY: "Show product clearly for e-commerce or catalog",
            PromptIntent.LIFESTYLE_ADVERTISEMENT: "Demonstrate product benefits in authentic lifestyle setting",
            PromptIntent.COMPARATIVE_INFOGRAPHIC: "Compare and highlight competitive advantages visually",
            PromptIntent.STORYBOARD_SEQUENCE: "Tell a narrative story with emotional arc",
            PromptIntent.TECHNICAL_DIAGRAM: "Explain technical details with precision",
            PromptIntent.EDIT_REFINEMENT: "Improve existing image quality",
            PromptIntent.BRAND_ASSET_GENERATION: "Create consistent brand assets",
        }

        return purposes.get(intent, "Generate professional visual content")

    def _get_category_description(self, category: PromptCategory) -> str:
        """Get human-readable description of category."""

        descriptions = {
            PromptCategory.ULTRA_SIMPLE: "Very basic input, needs complete enhancement",
            PromptCategory.BASIC_DIRECTION: "Has some direction, needs detail and style",
            PromptCategory.SPECIFIC_REQUEST: "Specific requirements given, needs execution",
            PromptCategory.COMPARATIVE: "Comparison requested, needs data visualization",
            PromptCategory.SEQUENTIAL: "Story/sequence requested, needs consistency management",
            PromptCategory.TECHNICAL: "Technical/diagram requested, needs precision",
        }

        return descriptions.get(category, "Standard request")
