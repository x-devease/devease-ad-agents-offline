"""
GPT-4o Prompt Converter

Converts structured feature-based instructions to natural language prompts
for Nano Banana models using GPT-4o, incorporating ROAS-optimized feature recommendations.
"""

# flake8: noqa
# pylint: disable=line-too-long
import base64
import json
import logging
import mimetypes
import os
from typing import Dict, Optional

try:
    import dotenv
except ImportError:
    dotenv = None  # type: ignore
from openai import OpenAI

from .constants import ENV_FILE, REPO_ENV_FILE


# Import feature descriptions for better GPT-4o understanding
try:
    from ..prompts.feature_descriptions import (
        build_feature_descriptions_context,
    )

    HAS_FEATURE_DESCRIPTIONS = True
except ImportError:
    HAS_FEATURE_DESCRIPTIONS = False

logger = logging.getLogger(__name__)

NANO_BANANA_PROMPT_GENERATION_SYSTEM_PROMPT = """You are a Nano Banana prompt generator for product marketing images.

## THE NANO BANANA FORMULA (MUST FOLLOW)
[Action] + [Subject] + [Setting] + [Style] + [Technical Details]

**CRITICAL: GEOMETRIC CONSTRAINT PRESERVATION**
If the user's prompt includes "Strictly maintain the exact geometric structure and proportions of Image 1", you MUST include this EXACT phrase at the very beginning of your generated prompt, before any other content.

Example with geometric constraint: "Strictly maintain the exact geometric structure and proportions of Image 1. Place the moprobo mop in a modern living room, where warm sunlight streams through large windows, creating soft shadows on hardwood floors nearby. Shot on 85mm f/2.8, shallow depth of field, product occupying 25% of frame."

Example without geometric constraint: "Place the moprobo mop in a modern living room, where warm sunlight streams through large windows, creating soft shadows on hardwood floors nearby. Shot on 85mm f/2.8, shallow depth of field, product occupying 25% of frame."

## CRITICAL CONSTRAINTS

1. **PRODUCT IDENTITY & CONSISTENCY (ABSOLUTE PRIORITY)**: 
   - Preserve EXACT product from source image. No redesigns, no variations.
   - The product MUST look IDENTICAL to the source image: same design, same shape, same features.
   - Maintain visual consistency - the product should look like the SAME product, not a variation.
   - Do NOT alter product design, shape, or appearance.
   - **CRITICAL**: When source image is provided, ONLY change background/scene - product must remain UNCHANGED
   - **CRITICAL**: Product text (like "moprobo") must remain SHARP and CLEAR - preserve exactly as in source
   - **CRITICAL**: All product details must remain EXACTLY as in source image - no modifications whatsoever
2. **LOGO FIDELITY**: Brand text/logos must remain 100% sharp and accurate.
3. **PRODUCT DETAIL & COLOR FIDELITY (CRITICAL)**:
   - Preserve ALL product colors exactly as they appear in the source image (no color shifts, alterations, or tinting)
   - ALL product text, labels, brand names, specifications, and markings must be SHARP and 100% READABLE
   - Product details (buttons, displays, labels, logos, text) must be in perfect focus with deep depth of field
   - Brand name and all product text must be legible with no blur, distortion, missing letters, or illegible characters
   - Use deep focus on the product itself (f/8 or deeper) to ensure all text and details are sharp
   - Product colors must match source image exactly - maintain original color accuracy
   - All product markings, labels, specifications, and text elements must be clearly visible and readable
   - If source has white background, maintain clean, professional product presentation with excellent detail visibility
4. **SINGLE PRODUCT**: Show ONLY ONE product instance (not multiple copies).
5. **NO HALLUCINATIONS**: Do not add extra products or accessories not in source.
6. **PRODUCT COMPLETENESS**: Show the product as a COMPLETE, ASSEMBLED UNIT with all components connected and functional. DO NOT show only individual parts (e.g., just a mop head) separately from the main product body. All components should appear as one integrated, realistic product unit.

## SPATIAL RELATIONSHIP KEYWORDS (USE THESE!)

Use conjunctions to establish clear spatial relationships:
- "where" - connects action to location ("product on counter, WHERE sunlight falls")
- "nearby" - establishes proximity ("product in foreground, NEARBY a window")
- "at the same time" - connects simultaneous elements
- "positioned at" - explicit placement ("positioned at left third of frame")
- "occupying" - size specification ("occupying 25% of frame")

## PHOTOGRAPHIC REALISM (CRITICAL FOR AVOIDING AI LOOK)

**Realistic Lighting:**
- Use physically accurate lighting with believable direction, intensity, and falloff
- Shadows must be consistent with light position (realistic hardness/softness)
- Lighting must have realistic contrast ratio and shadow behavior
- If golden hour: warm light, long soft shadows from one direction
- If studio: three-point lighting with defined key, fill, rim

**Real Camera Parameters:**
- Shot with full-frame camera, 50mm or 85mm lens
- Depth of field: For product details, use f/8 or deeper to ensure ALL product text, labels, and details are sharp
- Background: f/2.8-f/4 with natural bokeh (polygonal aperture shapes) - but product itself must be in deep focus
- Lens-compressed background perspective
- Specify: "Shot on 85mm, product in deep focus (f/8) for sharp details, background softly blurred"
- CRITICAL: Product text and labels require deep depth of field (f/8+) to remain sharp and readable

**Physical Interactions (Contact Shadows):**
- Objects touching surfaces MUST show contact shadows
- Contact shadows darker and more defined than cast shadows
- Physical indentations visible where weight is applied
- Grass/carpet compresses realistically under products

**Material Micro-Imperfections (Avoid Plastic Look):**
- Include: "visible surface texture, subtle dust particles in light"
- Include: "micro surface variations, realistic reflections"
- Include: "slight film grain, natural imperfections"
- Product surfaces show subtle wear consistent with real-world use

**Human Realism (if applicable):**
- Natural micro-expressions, imperfect skin texture
- Realistic hair lighting, natural wrinkles, no AI-smoothing
- Authentic poses and clothing textures

## PRODUCT SIZE GUIDANCE (BASED ON VISUAL PROMINENCE)

When visual_prominence is specified:
- "dominant": Product should occupy 40-50% of frame (hero product, clear focal point)
- "balanced": Product should occupy 25-30% of frame (balanced composition)
- "subtle": Product should occupy 15-20% of frame (contextual, lifestyle focus)

If no specific size is mentioned, use 30% as default (balanced).

## LIFESTYLE CONTEXT & PERSON USAGE

When product_visibility is "partial" OR human_elements includes "lifestyle context":
- Include a person using the product naturally
- Show hands interacting with the product
- Person should complement, not overshadow the product
- Natural, realistic interaction (avoid posed/staged look)
- Product remains the primary focus

## PROMPT STRUCTURE (BE SPECIFIC, NOT PROSE)

Instead of: "A beautiful product photo with nice lighting"
Use: "Product photography of [product name]. Product positioned at foreground-center, occupying 45% of frame (for dominant prominence), where three-point lighting illuminates the product 1.5 stops brighter than background. Contact shadows visible on surface. Shot on 85mm f/2.8, shallow depth of field, subtle dust particles in light, slight film grain."

## WHAT TO AVOID

- AVOID: Long prose descriptions (dilutes attention)
- AVOID: "NEVER" or "DO NOT" (activates forbidden concepts)
- AVOID: Vague terms like "beautiful", "nice", "professional"
- AVOID: Overly smooth/plastic surfaces (add micro-imperfections)
- AVOID: Flat/uniform lighting (add direction and falloff)
- AVOID: Missing contact shadows (always include where objects touch)

## OUTPUT FORMAT

Respond with JSON: {"flux_prompt": "...", "filename": "..."}
- flux_prompt: Optimized prompt following the formula above
- filename: Short descriptive filename (lowercase, underscores, max 50 chars)

Return ONLY the JSON, no additional text."""  # noqa: E501


class PromptConverter:
    """
    Converts structured feature-based instructions to natural language prompts
    for Nano Banana models using GPT-4o, incorporating ROAS-optimized feature recommendations.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize prompt converter.

        Args:
            openai_api_key: OpenAI API key (if None, uses environment variable)
        """
        # Load environment variables
        # Prefer repo .env first, then fallback keys file (do not override repo vars)
        if dotenv is not None:
            if REPO_ENV_FILE.exists():
                dotenv.load_dotenv(REPO_ENV_FILE, override=False)
            if ENV_FILE.exists():
                dotenv.load_dotenv(ENV_FILE, override=False)

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass openai_api_key parameter."
            )

        self.client = OpenAI(api_key=api_key)
        logger.info("PromptConverter initialized")

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 data URI."""
        mime, _ = mimetypes.guess_type(image_path)
        if not mime or not mime.startswith("image/"):
            # Safe fallback
            mime = "image/png"
        with open(image_path, "rb") as f:
            return f"data:{mime};base64,{base64.b64encode(f.read()).decode('utf-8')}"

    def convert_to_flux_prompt(
        self,
        base_prompt: str,
        feature_instructions: Optional[str] = None,
        source_image_path: Optional[str] = None,
        temperature: float = 0.7,
        feature_values: Optional[dict] = None,
        product_context: Optional[str] = None,
        target_endpoint: Optional[
            str
        ] = None,  # noqa: ARG001 - Unused but kept for API compatibility
    ) -> Dict[str, str]:
        """
        Convert base prompt and feature instructions to natural language prompt for Nano Banana.

        Args:
            base_prompt: Base product description or requirement
            feature_instructions: Optional structured feature optimization guidelines
            source_image_path: Optional source image for reference
            temperature: GPT-4o temperature (0.0-1.0)
            feature_values: Optional dict mapping feature names to values
                (used to provide detailed descriptions to GPT-4o)
            product_context: Optional product context text
            target_endpoint: Optional target endpoint (for compatibility, defaults to nano-banana)

        Returns:
            Dict with 'flux_prompt' and 'filename'
        """
        # Build user prompt
        user_prompt_parts = [base_prompt]

        if product_context:
            user_prompt_parts.append(
                "\n\n=== PRODUCT FACTS (OFFICIAL CONTEXT) ==="
            )
            user_prompt_parts.append(
                "Use the following product facts to improve accuracy. "
                "Do not invent specs that are not listed here:"
            )
            user_prompt_parts.append(product_context.strip())

        if feature_instructions:
            user_prompt_parts.append(
                "\n\n=== FEATURE OPTIMIZATION GUIDELINES ==="
            )
            user_prompt_parts.append(
                "The following guidelines are based on ROAS analysis of "
                "high-performing creative images. Incorporate these naturally "
                "into your prompt:"
            )
            user_prompt_parts.append(feature_instructions)
        # Add detailed feature descriptions if available (similar to analyzer.py)
        if HAS_FEATURE_DESCRIPTIONS and feature_values:
            descriptions_context = build_feature_descriptions_context(
                feature_values
            )
            if descriptions_context:
                user_prompt_parts.append("\n\n=== FEATURE DESCRIPTIONS ===")
                user_prompt_parts.append(
                    "The following descriptions explain what each feature value means. "
                    "Use these to create accurate, detailed prompts:"
                )
                user_prompt_parts.append(descriptions_context)

        user_prompt_text = "\n".join(user_prompt_parts)
        # Use Nano Banana system prompt (default for all models)
        system_prompt = NANO_BANANA_PROMPT_GENERATION_SYSTEM_PROMPT
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]
        # Add user message with optional image
        if source_image_path:
            img64 = self._encode_image_to_base64(source_image_path)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": img64},
                        },
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": user_prompt_text})

        logger.info("🔄 Converting to Nano Banana prompt via GPT-4o...")
        logger.debug("Base prompt: %s", base_prompt[:100])
        if feature_instructions:
            logger.debug(
                "Feature instructions length: %s", len(feature_instructions)
            )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=temperature,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError(
                    "GPT-4o returned null message.content (expected JSON string)."
                )
            if not isinstance(content, str):
                raise ValueError(
                    f"GPT-4o returned non-string message.content: {type(content)}"
                )
            data = json.loads(content)

            flux_prompt = data.get("flux_prompt", "").strip()
            filename = (
                data.get("filename", "generated")
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")[:100]
            )

            if not flux_prompt:
                raise ValueError("GPT-4o returned empty flux_prompt")

            logger.info("Nano Banana prompt generated successfully")
            logger.debug("Generated prompt: %s", flux_prompt[:200])

            return {
                "flux_prompt": flux_prompt,
                "filename": filename or "generated",
            }

        except json.JSONDecodeError as e:
            logger.error("Failed to parse GPT-4o JSON response: %s", e)
            raise ValueError(f"Invalid JSON response from GPT-4o: {e}") from e
        except Exception as e:
            logger.error("Error converting prompt via GPT-4o: %s", e)
            raise

    def convert_to_nano_banana(
        self,
        base_prompt: str,
        source_image_path: Optional[str] = None,
        recommendations: Optional[dict] = None,
        temperature: float = 0.7,
        product_context: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build feature instructions from ad/recommender output and convert to Nano Banana prompt.

        Args:
            base_prompt: Base product description.
            source_image_path: Optional source image for reference.
            recommendations: Dict with "recommendations" list (from JSON or MD).
            temperature: GPT-4o temperature.
            product_context: Optional product context.

        Returns:
            Dict with 'flux_prompt' and 'filename'.
        """
        feature_instructions = None
        feature_values = None
        recs = (recommendations or {}).get("recommendations") or []
        if recs:
            dos = []
            donts = []
            fv = {}
            for r in recs:
                feat = r.get("feature", "").strip()
                rec_val = r.get("recommended", "").strip()
                typ = r.get("type", "improvement")
                conf = r.get("reason", r.get("confidence", ""))
                if not feat:
                    continue
                line = f"- **{feat}**: {rec_val}"
                if conf:
                    line += f" ({conf})"
                if typ == "anti_pattern":
                    donts.append(line)
                else:
                    dos.append(line)
                    if rec_val and not rec_val.upper().startswith("NOT "):
                        fv[feat] = rec_val
            parts = []
            if dos:
                parts.append("**DOs (incorporate these):**")
                parts.extend(dos)
            if donts:
                parts.append("**DON'Ts (avoid these):**")
                parts.extend(donts)
            if parts:
                feature_instructions = "\n".join(parts)
            if fv:
                feature_values = fv
        return self.convert_to_flux_prompt(
            base_prompt=base_prompt,
            feature_instructions=feature_instructions,
            source_image_path=source_image_path,
            temperature=temperature,
            feature_values=feature_values,
            product_context=product_context,
        )
