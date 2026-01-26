"""
Simplified image generator using FAL API.
Works with enhanced prompts from the prompt enhancement system.
"""

# flake8: noqa
# pylint: disable=line-too-long
# pylint: disable=logging-fstring-interpolation,broad-exception-caught
# pylint: disable=missing-timeout,implicit-str-concat
import base64
from datetime import datetime
import logging
import mimetypes
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Optional

import dotenv
import requests


try:
    import fal_client
except ImportError:
    fal_client = None  # type: ignore

from .constants import (
    ENV_FILE,
    POLLING_INITIAL_DELAY_SECONDS,
    POLLING_INTERVAL_SECONDS,
    POLLING_MAX_WAIT_SECONDS,
    REPO_ENV_FILE,
)
from ..paths import Paths
from .prompt_converter import PromptConverter
from .text_overlay import (
    TextElementConfig,
    TextOverlay,
    TextOverlayConfig,
    TextPosition,
)
from .watermark import PremiumWatermark


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ALLOWED_AR = {
    "original",
    "1:1",
    "3:2",
    "2:3",
    "4:3",
    "3:4",
    "16:9",
    "9:16",
    "21:9",
    "9:21",
}


def _resolve_aspect_ratio(choice: str) -> str:
    """Resolve aspect ratio choice to supported value."""
    if choice in ALLOWED_AR:
        return choice
    # A4 ≈ 1:√2 (≈ 1:1.414). Use a close, supported ratio:
    if choice == "A4 Portrait":
        return "3:4"  # 0.75 (closest supported to 1/√2 ≈ 0.707)
    if choice == "A4 Landscape":
        return "4:3"  # 1.333 (closest supported to √2 ≈ 1.414)
    return "original"


class ImageGenerator:
    """
    Simplified image generator that works with enhanced prompts.

    This generator takes prompts directly (from prompt enhancement system)
    and generates images using FAL API (Nano Banana models).
    """

    def __init__(
        self,
        model: str = "nano-banana-pro",
        aspect_ratio: str = "original",
        resolution: str = "2K",
        enable_upscaling: bool = True,
        enable_watermark: bool = True,
        enable_text_overlay: bool = False,
        text_overlay_config: Optional[TextOverlayConfig] = None,
        output_dir: Optional[str] = None,
        use_gpt4o_conversion: bool = True,
        openai_api_key: Optional[str] = None,
        # Path configuration for organized output
        customer: Optional[str] = None,
        platform: Optional[str] = None,
        date: Optional[str] = None,
        # Generation control parameters (for attention map optimization)
        strength: float = 0.85,
        guidance_scale: float = 8.0,
        num_inference_steps: int = 30,
    ):
        """
        Initialize image generator.

        Raises:
            ImportError: If fal_client is not installed

        Args:
            model: Model to use ("nano-banana-pro")
            aspect_ratio: Aspect ratio for generated images
            resolution: Output resolution for models that support it.
                Allowed values: "1K", "2K", "4K".
            enable_upscaling: Whether to enable upscaling
            enable_watermark: Whether to apply watermark
            output_dir: Output directory (overrides path-based organization)
                If not provided, uses customer/platform/date structure
            use_gpt4o_conversion: Whether to use GPT-4o to convert
                structured instructions to natural language prompts
            openai_api_key: OpenAI API key (if None, uses env variable)
            customer: Customer name for path organization (optional)
            platform: Platform name for path organization (optional, default: "meta")
            date: Date string for path organization (optional, default: today YYYY-MM-DD)
            strength: How much to preserve source image (0.0-1.0).
                Higher = more preservation (logos, product details).
                Default 0.85 for logo/product preservation.
            guidance_scale: How strictly to follow the prompt (1.0-20.0).
                Higher = more prompt adherence. Default 8.0.
            num_inference_steps: Quality vs speed tradeoff (10-50).
                More steps = better quality but slower. Default 30.

        Path Organization:
            If customer/platform/date provided:
                results/ad/generator/generated/{customer}/{platform}/{date}/
            Otherwise:
                ./generated_images
        """
        self.model = model.lower()
        self.aspect_ratio = aspect_ratio
        self.resolution = str(resolution or "1K").strip()
        self.enable_upscaling = enable_upscaling
        self.enable_watermark = enable_watermark
        self.use_gpt4o_conversion = use_gpt4o_conversion
        # Generation control parameters
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        # Load environment variables
        # Prefer repo .env first, then fallback keys file
        # (do not override repo vars)
        if REPO_ENV_FILE.exists():
            dotenv.load_dotenv(REPO_ENV_FILE, override=False)
        if ENV_FILE.exists():
            dotenv.load_dotenv(ENV_FILE, override=False)

        # Preflight: normalize fal credentials (common failure is whitespace)
        def _clean_env(name: str) -> Optional[str]:
            val = os.getenv(name)
            if val is None:
                return None
            cleaned = val.strip().strip('"').strip("'")
            if cleaned != val:
                os.environ[name] = cleaned
            return cleaned

        fal_key = _clean_env("FAL_KEY")
        # Support both conventional and legacy/custom env var names
        # (some local setups may use FAL_KEY_prod).
        fal_key_id = _clean_env("FAL_KEY_ID") or _clean_env("FAL_KEY_prod")
        fal_key_secret = _clean_env("FAL_KEY_SECRET")
        # If both forms are present, fal_client may prefer ID/SECRET.
        if fal_key and (fal_key_id or fal_key_secret):
            logger.warning(
                "Both FAL_KEY and FAL_KEY_ID/FAL_KEY_SECRET are set. "
                "fal_client may prefer the ID/SECRET pair."
            )
        # Set up API endpoint for nano-banana-pro
        # Ref: fal-ai/nano-banana-pro/edit
        self.api_endpoint = "fal-ai/nano-banana-pro/edit"
        # Set up output directory with organized path structure
        if output_dir:
            # Use explicit output_dir if provided
            self.output_dir = Path(output_dir)
        elif customer:
            # Use organized path structure: results/ad/generator/generated/{customer}/{platform}/{date}/
            platform = platform or "meta"
            date_str = date or datetime.now().strftime("%Y-%m-%d")
            paths = Paths(customer=customer, platform=platform, date=date_str)
            self.output_dir = paths.generated_output()
            logger.info(
                "Using organized output path: %s",
                self.output_dir
            )
        else:
            # Fallback to default location
            self.output_dir = Path("./generated_images")
            logger.warning(
                "No customer/platform provided, using default output path: %s",
                self.output_dir
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Initialize GPT-4o prompt converter if enabled
        if self.use_gpt4o_conversion:
            try:
                self.prompt_converter = PromptConverter(
                    openai_api_key=openai_api_key
                )
            except ValueError as e:
                logger.warning(
                    "GPT-4o conversion requested but OpenAI API key not found: %s. "
                    "Falling back to direct prompt usage.",
                    e,
                )
                self.use_gpt4o_conversion = False
                self.prompt_converter = None
        else:
            self.prompt_converter = None
        # Initialize watermark if enabled
        self.watermark = PremiumWatermark() if self.enable_watermark else None
        # Initialize text overlay if enabled
        if enable_text_overlay and text_overlay_config:
            self.text_overlay = TextOverlay(text_overlay_config)
        else:
            self.text_overlay = None

        logger.info(
            "ImageGenerator initialized: model=%s, endpoint=%s, "
            "gpt4o_conversion=%s",
            model,
            self.api_endpoint,
            self.use_gpt4o_conversion,
        )

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 data URI."""
        mime, _ = mimetypes.guess_type(image_path)
        if not mime or not mime.startswith("image/"):
            mime = "image/png"
        with open(image_path, "rb") as f:
            return f"data:{mime};base64,{base64.b64encode(f.read()).decode('utf-8')}"

    def _poll_fal_completion(
        self,
        handler,
        task_name: str = "task",
        max_wait_time: int = POLLING_MAX_WAIT_SECONDS,
    ) -> Optional[Dict[str, Any]]:
        """Poll fal.ai for task completion with custom intervals."""
        start_time = time.time()
        poll_count = 0
        # Use configured polling intervals
        initial_delay = POLLING_INITIAL_DELAY_SECONDS
        poll_interval = POLLING_INTERVAL_SECONDS
        max_attempts = int(max_wait_time / poll_interval)

        logger.info(f"Polling {task_name} completion (max {max_wait_time}s)")
        # Initial delay before first poll
        logger.info(f"Initial wait {initial_delay}s before polling...")
        time.sleep(initial_delay)

        for attempt in range(max_attempts):
            poll_count += 1
            elapsed = time.time() - start_time

            if elapsed > max_wait_time:
                logger.info(
                    f"Timeout waiting for {task_name} after {elapsed:.1f}s"
                )
                return None

            try:
                # Check status using handler.status()
                status = handler.status(with_logs=False)
                # Check if completed using isinstance
                if isinstance(status, fal_client.Completed):
                    logger.info(
                        "%s completed after %d polls (%.1fs)",
                        task_name,
                        attempt + 1,
                        elapsed,
                    )
                    # Get the final result
                    result = handler.get()
                    if isinstance(result, dict):
                        return result
                    return None
                if hasattr(status, "status") and status.status == "FAILED":
                    logger.info("%s failed after %.1fs", task_name, elapsed)
                    return None
                logger.debug(
                    f"Polling {task_name}: attempt {attempt + 1}/{max_attempts} ({elapsed:.1f}s)..."
                )

            except Exception as e:
                logger.warning(f"Poll error: {str(e)}")

            time.sleep(poll_interval)

        logger.info(f"Max polls reached for {task_name}")
        return None

    def generate(
        self,
        prompt: str,
        source_image_path: Optional[str] = None,
        output_filename: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        feature_instructions: Optional[str] = None,
        feature_values: Optional[Dict[str, str]] = None,
        product_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate image from prompt.

        Args:
            prompt: Base prompt text (or enhanced prompt if GPT-4o conversion disabled)
            source_image_path: Optional source image for image-to-image generation
            output_filename: Optional output filename (auto-generated if not provided)
            aspect_ratio: Optional aspect ratio override
            resolution: Optional output resolution override for supported endpoints
                (e.g., "1K", "2K", "4K" for nano-banana-pro/edit).
            feature_instructions: Optional structured feature optimization guidelines
                (used with GPT-4o conversion)
            feature_values: Optional dict mapping feature names to values
                (used to provide detailed descriptions to GPT-4o)

        Returns:
            Dict with 'success', 'image_path', 'upscaled_path' (if upscaled),
            'flux_prompt' (if GPT-4o conversion used)
        """
        # Determine prompt_type and model tag from endpoint
        ep_lower = self.api_endpoint.lower()
        prompt_type = "nano_banana_edit"
        model_tag = "__nano_banana"
        model_name = (
            "nano-banana-pro"
            if "nano-banana-pro" in ep_lower
            else "nano-banana"
        )
        # Convert prompt via GPT-4o if enabled and feature instructions provided
        # OR if source image + product context provided (image + GPT flow)
        flux_prompt = prompt
        generated_filename = output_filename

        if self.use_gpt4o_conversion and self.prompt_converter:
            should_convert = (
                feature_instructions
                or self._is_structured_prompt(prompt)
                or (source_image_path and product_context)
            )
            if should_convert:
                logger.info(
                    "🔄 Converting structured instructions to prompt..."
                )
                try:
                    result = self.prompt_converter.convert_to_flux_prompt(
                        base_prompt=prompt,
                        feature_instructions=feature_instructions,
                        source_image_path=source_image_path,
                        feature_values=feature_values,
                        product_context=product_context,
                        target_endpoint=self.api_endpoint,
                    )
                    flux_prompt = result["flux_prompt"]
                    if not generated_filename:
                        # GPT-generated filename - add model tag if auto-naming
                        base_name = result["filename"]
                        generated_filename = f"{base_name}{model_tag}.jpg"
                    logger.info("Converted to prompt via GPT-4o")
                except Exception as e:
                    logger.warning(
                        "Failed to convert prompt via GPT-4o: %s. Using original prompt.",
                        e,
                    )
                    # Fall back to original prompt
                    flux_prompt = prompt
        # Generate output filename if not provided
        if not generated_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Add model tag when auto-generating filename
            generated_filename = f"generated_{timestamp}{model_tag}.jpg"
        # Use provided aspect ratio or default
        aspect_ratio_value = aspect_ratio or self.aspect_ratio
        resolved_ar = _resolve_aspect_ratio(aspect_ratio_value)
        # Prepare API arguments (endpoint-specific schema)
        args: Dict[str, Any] = {"prompt": flux_prompt}

        if not source_image_path:
            return {
                "success": False,
                "error": (
                    "This fal.ai endpoint requires a source image "
                    f"(missing image_urls). You are using: {self.api_endpoint}\n\n"
                    "Fix:\n"
                    "- Provide --source-image for image-to-image."
                ),
            }

        # source_image_path is guaranteed to exist here (validated above)
        img_data_uri = self._encode_image_to_base64(source_image_path)
        # Nano Banana schema (expects a list)
        args["image_urls"] = [img_data_uri]
        if resolved_ar != "original":
            args["aspect_ratio"] = resolved_ar
        # Nano Banana Pro supports explicit resolution (1K/2K/4K). Only send this
        # param to endpoints that advertise it to avoid schema errors.
        if "nano-banana-pro" in ep_lower:
            res = str(resolution or self.resolution or "").strip()
            if res:
                args["resolution"] = res
        # Add generation control parameters for attention map optimization
        # These help the model preserve source image features (logos, product)
        # while following prompt instructions more precisely.
        # Nano Banana models support strength, guidance_scale, steps
        args["strength"] = self.strength
        args["guidance_scale"] = self.guidance_scale
        args["steps"] = self.num_inference_steps
        logger.info(
            "   Generation params: strength=%.2f, guidance=%.1f, steps=%d",
            self.strength,
            self.guidance_scale,
            self.num_inference_steps,
        )

        logger.info("Generating image with prompt: %s...", flux_prompt[:100])
        logger.info("   Model: %s, Aspect Ratio: %s", self.model, resolved_ar)
        # Submit generation request
        try:
            handler = fal_client.submit(self.api_endpoint, arguments=args)
            result = self._poll_fal_completion(
                handler, "image generation", max_wait_time=300
            )

            if not result or "images" not in result or not result["images"]:
                logger.error("No image generated")
                return {"success": False, "error": "No image generated"}
            # Download generated image
            image_url = result["images"][0]["url"]
            response = requests.get(image_url)
            if response.status_code != 200:
                logger.error(
                    f"Failed to download image: HTTP {response.status_code}"
                )
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                }
            # Save image
            # NOTE: output_filename may be None; generated_filename is always set.
            output_path = self.output_dir / generated_filename
            with open(output_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Image saved: {output_path}")

            result_dict = {
                "success": True,
                "image_path": str(output_path),
                "image_path_original": str(
                    output_path
                ),  # Always the unmodified version
                "prompt": prompt,
                "converted_prompt": (
                    flux_prompt if self.use_gpt4o_conversion else None
                ),
                # New metadata fields for status management
                "prompt_type": prompt_type,
                "target_endpoint": self.api_endpoint,
                "model": model_name,
                "final_prompt": flux_prompt,  # The actual prompt used (canonical)
            }
            # Apply text overlay if enabled - BEFORE watermark
            # Text overlay is part of the image content, watermark is branding
            watermark_input = str(output_path)
            if self.text_overlay:
                logger.info("Applying text overlay...")
                text_overlay_dir = self.output_dir / "with_text_overlay"
                text_overlay_dir.mkdir(parents=True, exist_ok=True)
                text_overlay_path = text_overlay_dir / generated_filename
                # Copy original to text overlay location
                shutil.copy2(output_path, text_overlay_path)
                # Apply text overlay
                result = self.text_overlay.apply_text_overlay(str(text_overlay_path))
                if result:
                    result_dict["image_path_with_text"] = str(text_overlay_path)
                    watermark_input = str(text_overlay_path)
                    logger.info("Text overlay applied: %s", text_overlay_path)
                else:
                    logger.warning("Failed to apply text overlay")
            # Apply watermark if enabled - AFTER text overlay
            if self.enable_watermark and self.watermark:
                logger.info("Applying watermark...")
                # Create watermarked directory
                watermarked_dir = self.output_dir / "watermarked"
                watermarked_dir.mkdir(parents=True, exist_ok=True)
                watermarked_path = watermarked_dir / generated_filename
                # Copy (possibly text-overlayed) image to watermarked location
                shutil.copy2(watermark_input, watermarked_path)
                watermarked = self.watermark.apply_watermark(str(watermarked_path))
                if watermarked:
                    result_dict["image_path_watermarked"] = str(
                        watermarked_path
                    )
                    logger.info("Watermarked image saved: %s", watermarked_path)
                else:
                    logger.warning("Failed to apply watermark")
            # Upscale if enabled
            if self.enable_upscaling:
                upscaled_path = self.upscale_image(
                    str(output_path), output_filename
                )
                if upscaled_path:
                    result_dict["upscaled_path"] = upscaled_path

            return result_dict

        except Exception as e:
            msg = str(e)
            # Make auth failures actionable (most common first-run issue)
            if "401" in msg or "Unauthorized" in msg or "No user found" in msg:
                msg = (
                    f"{msg}\n\n"
                    "Auth check:\n"
                    "- Ensure FAL_KEY is a valid fal.ai API key for your account.\n"
                    "- If you use ID/SECRET style keys, set BOTH FAL_KEY_ID and "
                    "FAL_KEY_SECRET.\n"
                    "- Remove duplicates: don't set both FAL_KEY and "
                    "FAL_KEY_ID/FAL_KEY_SECRET unless you know which one is used.\n"
                    "- Make sure there is no whitespace/newlines around the key.\n"
                )
            logger.error("Error generating image: %s", msg)
            return {"success": False, "error": msg}

    def _is_structured_prompt(self, prompt: str) -> bool:
        """
        Check if prompt appears to be structured instructions rather than
        natural language.

        Args:
            prompt: Prompt text to check

        Returns:
            True if prompt appears structured (has markers like [OK], [FAIL], ===)
        """
        structured_markers = [
            "[OK]",
            "[FAIL]",
            "===",
            "MUST INCLUDE",
            "NEVER INCLUDE",
        ]
        return any(marker in prompt for marker in structured_markers)

    def upscale_image(
        self, image_path: str, output_filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Upscale image using Clarity Upscaler.

        Args:
            image_path: Path to image to upscale
            output_filename: Optional output filename

        Returns:
            Path to upscaled image or None if failed
        """
        logger.info(f"Upscaling image: {image_path}")

        try:
            # Read the image and convert to URL for fal.ai
            image_data_uri = self._encode_image_to_base64(image_path)
            # Submit enhancement request using fal.ai client
            logger.info("Submitting upscaling request...")
            handler = fal_client.submit(
                "fal-ai/clarity-upscaler",
                arguments={"image_url": image_data_uri},
            )
            # Poll for completion
            logger.info("Polling for upscaling completion...")
            upscale_result = self._poll_fal_completion(
                handler, "image upscaling", max_wait_time=300
            )

            if not upscale_result or "image" not in upscale_result:
                logger.error("No upscaled image generated")
                return None
            # Download the upscaled image
            upscaled_image_url = upscale_result["image"]["url"]
            logger.info(f"Upscaling complete: {upscaled_image_url}")

            response = requests.get(upscaled_image_url)
            if response.status_code != 200:
                logger.error(
                    f"Failed to download upscaled image: HTTP {response.status_code}"
                )
                return None
            # Create upscaled directory
            upscaled_dir = self.output_dir / "upscaled"
            upscaled_dir.mkdir(exist_ok=True)
            # Generate output filename if not provided
            if not output_filename:
                output_filename = Path(image_path).name

            output_path = upscaled_dir / output_filename
            with open(output_path, "wb") as f:
                f.write(response.content)

            file_size = os.path.getsize(output_path)
            logger.info(
                f"Upscaled image saved: {output_path} ({file_size} bytes)"
            )

            return str(output_path)

        except Exception as e:
            logger.error(f"Error upscaling image: {e}")
            return None
