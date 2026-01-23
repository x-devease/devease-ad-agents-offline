"""GPT-4 Vision Feature Extractor for image feature extraction.

This module provides the GPT4FeatureExtractor class for extracting visual
features from images using OpenAI's GPT-4 Vision API. It supports batch
processing, progress tracking, and feature extraction for both training
data and new images.
"""

# pylint: disable=too-many-lines

import base64
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI
from openai.types.chat import ChatCompletionContentPartParam

from src.utils.config_manager import ConfigManager
from src.utils.constants import FeaturesConstants as Constants

logger = logging.getLogger(__name__)


class GPT4FeatureExtractor:
    """GPT-4 Vision Feature Extractor for extracting features from ad images.

    This class provides functionality to extract visual features from images
    using OpenAI's GPT-4 Vision API. It supports both batch processing of
    classified images (top/bottom performers) and direct folder-based
    processing.

    Attributes:
        client: OpenAI API client instance.
        images_folder: Path to the folder containing images to process.
        prompt_template: Prompt template loaded from configuration file.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize GPT-4 Vision feature extractor.

        Args:
            api_key: OpenAI API key. If None, will use environment variable or
                ~/.devease/keys file.
        """
        self.client = OpenAI(
            api_key=api_key, timeout=Constants.DEFAULT_API_TIMEOUT
        )
        self.images_folder = Constants.DEFAULT_IMAGES_FOLDER

        # Load prompt template from configuration file
        prompts_config = ConfigManager.get_config(None, "ad/recommender/gpt4/prompts.yaml")
        self.prompt_template = prompts_config.get("prompt_template", "")

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string for GPT-4 Vision API.

        Args:
            image_path: Path to the image file to encode.

        Returns:
            Base64-encoded string of the image data.

        Raises:
            FileNotFoundError: If the image file does not exist.
            IOError: If the image file cannot be read.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_prompt(
        self, image_filenames: List[str], performance_type: str
    ) -> str:
        """Get structured prompt for GPT-4 Vision feature extraction.

        Args:
            image_filenames: List of image filenames to extract features from.
            performance_type: Performance type label
                (e.g., "high", "low", "unknown").
        Returns:
            Formatted prompt string for GPT-4 Vision API.
        """
        # Format the prompt template with the provided parameters
        prompt = self.prompt_template.format(
            len_image_filenames=len(image_filenames),
            performance_type=performance_type,
            image_filenames_str=", ".join(image_filenames),
        )
        return prompt

    # pylint: disable=too-many-locals,too-many-return-statements,too-many-branches,too-many-statements,broad-exception-caught
    def extract_batch_features(
        self, image_filenames, performance_type, batch_number
    ):
        """Extract features from a batch of images using GPT-4 Vision

        Returns:
            tuple: (results, error_info) where:
                - results: List of extraction results (None if failed)
                - error_info: Dict with error details if failed,
                    None if successful
        """

        image_count = len(image_filenames)
        logger.info(
            " extracting %s-performing batch %d (%d images)...",
            performance_type,
            batch_number,
            image_count,
        )

        # Prepare images for GPT-4.1 using local files
        image_contents: List[ChatCompletionContentPartParam] = []
        valid_filenames = []
        invalid_filenames = []

        for filename in image_filenames:
            image_path = os.path.join(self.images_folder, filename)
            if os.path.exists(image_path):
                try:
                    # Try to encode the image to check if it's valid
                    encoded = self.encode_image(image_path)
                    image_contents.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded}"
                            },
                        }
                    )
                    valid_filenames.append(filename)
                except Exception as error:
                    logger.info(
                        "WARNING: Warning: Could not encode image %s: %s",
                        filename,
                        error,
                    )
                    invalid_filenames.append(filename)
            else:
                logger.info(
                    "WARNING: Warning: Image %s not found at %s",
                    filename,
                    image_path,
                )
                invalid_filenames.append(filename)

        if not image_contents:
            error_info = {
                "error_type": "no_valid_images",
                "error_message": (
                    f"No valid images found in batch {batch_number}"
                ),
                "invalid_images": invalid_filenames,
                "batch_number": batch_number,
            }
            logger.error("No valid images found in batch %d", batch_number)
            return None, error_info

        # Create prompt
        prompt = self.get_prompt(valid_filenames, performance_type)

        # Build message content parts (prompt text + images)
        content_parts: List[ChatCompletionContentPartParam] = [
            {"type": "text", "text": prompt}
        ]
        content_parts.extend(image_contents)

        # Call GPT-4.1 with structured JSON output (with improved retry logic)
        max_retries = 10  # Increased retry count
        base_delay = 2.0  # Base delay in seconds
        max_delay = 60.0  # Maximum delay in seconds

        last_error = None
        error_type = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in visual "
                            "marketing extraction. Your task is to extract "
                            "ad images and extract structured "
                            "characteristics that affect "
                            "click-through rates.\n\n"
                            "CRITICAL REQUIREMENTS:\n"
                            "1. CONSISTENCY: Your extraction must be "
                            "consistent and reproducible. The same image "
                            "extracted multiple times should yield "
                            "identical results.\n"
                            "2. OBJECTIVITY: Focus on objective, "
                            "measurable visual characteristics. Avoid "
                            "subjective interpretations or personal "
                            "preferences.\n"
                            "3. PRECISION: When uncertain, choose the "
                            "option that most closely matches the image. "
                            "Do not use ambiguous classifications.\n"
                            "4. VERIFICATION: After classification, "
                            "verify that your selections are internally "
                            "consistent and match the image "
                            "characteristics.\n\n"
                            "Return your response in valid JSON format.",
                        },
                        {
                            "role": "user",
                            "content": content_parts,
                        },
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=6000,
                    temperature=0.0,
                )
                # Success, exit retry loop
                break

            except Exception as error:
                last_error = error
                error_str = str(error)
                is_timeout = (
                    "timeout" in error_str.lower()
                    or "timed out" in error_str.lower()
                )
                is_rate_limit = (
                    "rate limit" in error_str.lower() or "429" in error_str
                )
                is_image_error = (
                    "image" in error_str.lower()
                    or "invalid" in error_str.lower()
                    or "format" in error_str.lower()
                )

                error_type = (
                    "timeout"
                    if is_timeout
                    else (
                        "rate_limit"
                        if is_rate_limit
                        else ("image_error" if is_image_error else "api_error")
                    )
                )

                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = min(base_delay * (2**attempt), max_delay)
                    jitter = random.uniform(0, base_delay)
                    wait_time = delay + jitter

                    error_preview = error_str[:100]
                    logger.info(
                        "WARNING: Attempt %d/%d failed for batch %d (%s): %s",
                        attempt + 1,
                        max_retries,
                        batch_number,
                        error_type,
                        error_preview,
                    )
                    logger.info("Retrying in %.1f seconds...", wait_time)
                    time.sleep(wait_time)
                else:
                    # Last attempt failed, return error info
                    logger.info(
                        "ERROR: All %d attempts failed for batch %d",
                        max_retries,
                        batch_number,
                    )
                    logger.info("   Last error: %s", last_error)
                    error_info = {
                        "error_type": error_type,
                        "error_message": str(last_error),
                        "batch_number": batch_number,
                        "image_filenames": valid_filenames,
                        "invalid_images": invalid_filenames,
                        "attempts": max_retries,
                    }
                    return None, error_info

        # Parse the JSON response
        try:
            response_text = response.choices[0].message.content

            if response_text:
                try:
                    # Parse JSON response
                    analysis_data = json.loads(response_text)

                    # Ensure performance_label is set for each extraction
                    results = None
                    if (
                        isinstance(analysis_data, dict)
                        and "analyses" in analysis_data
                    ):
                        results = analysis_data["analyses"]
                    elif isinstance(analysis_data, list):
                        results = analysis_data
                    else:
                        logger.info(
                            "WARNING: Unexpected response format for batch %d",
                            batch_number,
                        )
                        error_info = {
                            "error_type": "parse_error",
                            "error_message": "Unexpected response format",
                            "batch_number": batch_number,
                            "image_filenames": valid_filenames,
                        }
                        return None, error_info

                    # Set performance label and verify all images are present
                    result_filenames = set()
                    for extraction_result in results:
                        extraction_result["performance_label"] = (
                            performance_type
                        )
                        filename = extraction_result.get("filename", "")
                        if filename:
                            result_filenames.add(filename)

                    # Check if all images were extracted
                    missing_images = set(valid_filenames) - result_filenames
                    if missing_images:
                        logger.info(
                            "WARNING: Warning: Some images missing from "
                            "response: %s",
                            missing_images,
                        )
                        results_count = len(results)
                        valid_count = len(valid_filenames)
                        logger.info(
                            "   Successfully extracted: %d/%d images",
                            results_count,
                            valid_count,
                        )
                        # Return partial success with error info for missing
                        # images. This allows successful images to be saved
                        # while marking missing ones for retry
                        partial_error_info = {
                            "error_type": "partial_success",
                            "error_message": (
                                f"Only {results_count}/{valid_count} "
                                "images extracted"
                            ),
                            "batch_number": batch_number,
                            "missing_images": list(missing_images),
                            "successful_images": list(result_filenames),
                            "image_filenames": valid_filenames,
                        }
                        # Return results (partial success) along with error
                        # info for missing images
                        return results, partial_error_info

                    return results, None

                except json.JSONDecodeError as error:
                    logger.info(
                        "ERROR: JSON parsing error for batch %d: %s",
                        batch_number,
                        error,
                    )
                    preview = response_text[:200]
                    logger.info("Response preview: %s...", preview)
                    error_info = {
                        "error_type": "parse_error",
                        "error_message": f"JSON decode error: {error}",
                        "batch_number": batch_number,
                        "image_filenames": valid_filenames,
                        "response_preview": response_text[:200],
                    }
                    return None, error_info
            else:
                logger.error("Empty response for batch %d", batch_number)
                error_info = {
                    "error_type": "empty_response",
                    "error_message": "API returned empty response",
                    "batch_number": batch_number,
                    "image_filenames": valid_filenames,
                }
                return None, error_info

        except Exception as error:
            logger.info(
                "ERROR: Error processing response for batch %d: %s",
                batch_number,
                error,
            )
            error_info = {
                "error_type": "response_error",
                "error_message": str(error),
                "batch_number": batch_number,
                "image_filenames": valid_filenames,
            }
            return None, error_info

    # pylint: disable=too-many-locals,broad-exception-caught
    def process_all_batches(self, top_150_file, bottom_150_file):
        """Process all batches of top and bottom performers"""

        logger.info(" GPT-4.1 VISUAL extraction - PHASE 2")
        logger.info("=" * 50)

        # Load data
        top_150_df = pd.read_csv(top_150_file)
        bottom_150_df = pd.read_csv(bottom_150_file)

        all_analyses = []

        # Process top 150 in batches of 5
        for i in range(0, 150, 5):
            batch_num = (i // 5) + 1
            batch_filenames = (
                top_150_df["Image Filename"].iloc[i : i + 5].tolist()
            )

            extraction, error_info = self.extract_batch_features(
                batch_filenames, "high", batch_num
            )
            if extraction:
                all_analyses.extend(extraction)
            elif error_info:
                error_type = error_info.get("error_type", "unknown")
                logger.info(
                    "WARNING: Batch %d failed (%s), continuing...",
                    batch_num,
                    error_type,
                )

            # Rate limiting - increased for stability
            time.sleep(5)

            # Progress update
            if extraction:
                logger.info(" Completed high-performing batch %d/30", batch_num)

        # Process bottom 150 in batches of 5
        for i in range(0, 150, 5):
            batch_num = (i // 5) + 1
            batch_filenames = (
                bottom_150_df["Image Filename"].iloc[i : i + 5].tolist()
            )

            extraction, error_info = self.extract_batch_features(
                batch_filenames, "low", batch_num
            )
            if extraction:
                all_analyses.extend(extraction)
            elif error_info:
                error_type = error_info.get("error_type", "unknown")
                logger.info(
                    "WARNING: Batch %d failed (%s), continuing...",
                    batch_num,
                    error_type,
                )

            # Rate limiting - increased for stability
            time.sleep(5)

            # Progress update
            if extraction:
                logger.info(" Completed low-performing batch %d/30", batch_num)

        # Save results
        if all_analyses:
            # Validate performance labels
            high_count = sum(
                1 for a in all_analyses if a.get("performance_label") == "high"
            )
            low_count = sum(
                1 for a in all_analyses if a.get("performance_label") == "low"
            )

            logger.info("\n extraction breakdown:")
            logger.info("  - High performing images: %d", high_count)
            logger.info("  - Low performing images: %d", low_count)

            with open(
                "data/gpt4_analysis_results.json", "w", encoding="utf-8"
            ) as json_file:
                json.dump(all_analyses, json_file, indent=2)

            # Convert to DataFrame for easier extraction
            df_analysis = self.flatten_analysis_data(all_analyses)
            df_analysis.to_csv("data/gpt4_analysis_results.csv", index=False)

            analyses_count = len(all_analyses)
            logger.info(
                "\n extraction complete! Processed %d images", analyses_count
            )
            logger.info(" Results saved:")
            logger.info("  - data/gpt4_analysis_results.json (raw extraction)")
            logger.info("  - data/gpt4_analysis_results.csv (flattened data)")
        else:
            logger.error("No extraction results generated")

    def flatten_analysis_data(self, analyses):
        """Flatten the nested extraction data into a DataFrame"""

        flattened_data = []

        for extraction in analyses:
            flat_row = {
                "filename": extraction.get("filename", ""),
                "performance_label": extraction.get(
                    "performance_label", ""
                ),  # Include performance label
                "primary_colors": ", ".join(
                    extraction.get("visual_elements", {}).get(
                        "primary_colors", []
                    )
                ),
                "color_harmony": extraction.get("visual_elements", {}).get(
                    "color_harmony", ""
                ),
                "contrast_level": extraction.get("visual_elements", {}).get(
                    "contrast_level", ""
                ),
                "brightness": extraction.get("visual_elements", {}).get(
                    "brightness", ""
                ),
                "color_saturation": extraction.get("visual_elements", {}).get(
                    "color_saturation", ""
                ),
                "color_vibrancy": extraction.get("visual_elements", {}).get(
                    "color_vibrancy", ""
                ),
                "color_diversity": extraction.get("visual_elements", {}).get(
                    "color_diversity", ""
                ),
                "color_balance": extraction.get("visual_elements", {}).get(
                    "color_balance", ""
                ),
                "background_tone_contrast": extraction.get(
                    "visual_elements", {}
                ).get("background_tone_contrast", ""),
                "element_separation": extraction.get("visual_elements", {}).get(
                    "element_separation", ""
                ),
                "local_contrast": extraction.get("visual_elements", {}).get(
                    "local_contrast", ""
                ),
                "highlight_ratio": extraction.get("visual_elements", {}).get(
                    "highlight_ratio", ""
                ),
                "product_placement": extraction.get("composition", {}).get(
                    "product_placement", ""
                ),
                "background_type": extraction.get("composition", {}).get(
                    "background_type", ""
                ),
                "image_style": extraction.get("composition", {}).get(
                    "image_style", ""
                ),
                "framing": extraction.get("composition", {}).get("framing", ""),
                "visual_complexity": extraction.get("composition", {}).get(
                    "visual_complexity", ""
                ),
                "primary_focal_point": extraction.get("composition", {}).get(
                    "primary_focal_point", ""
                ),
                "visual_flow": extraction.get("composition", {}).get(
                    "visual_flow", ""
                ),
                "balance_type": extraction.get("composition", {}).get(
                    "balance_type", ""
                ),
                "negative_space_usage": extraction.get("composition", {}).get(
                    "negative_space_usage", ""
                ),
                "rule_of_thirds": extraction.get("composition", {}).get(
                    "rule_of_thirds", ""
                ),
                "golden_ratio": extraction.get("composition", {}).get(
                    "golden_ratio", ""
                ),
                "symmetry_type": extraction.get("composition", {}).get(
                    "symmetry_type", ""
                ),
                "depth_layers": extraction.get("composition", {}).get(
                    "depth_layers", ""
                ),
                "leading_lines": extraction.get("composition", {}).get(
                    "leading_lines", ""
                ),
                "horizon_position": extraction.get("composition", {}).get(
                    "horizon_position", ""
                ),
                "vertical_alignment": extraction.get("composition", {}).get(
                    "vertical_alignment", ""
                ),
                "horizontal_alignment": extraction.get("composition", {}).get(
                    "horizontal_alignment", ""
                ),
                "perspective_type": extraction.get("composition", {}).get(
                    "perspective_type", ""
                ),
                "focal_point_count": extraction.get("composition", {}).get(
                    "focal_point_count", ""
                ),
                "eye_tracking_path": extraction.get("composition", {}).get(
                    "eye_tracking_path", ""
                ),
                "lighting_type": extraction.get("lighting", {}).get(
                    "lighting_type", ""
                ),
                "temperature": extraction.get("lighting", {}).get(
                    "temperature", ""
                ),
                "direction": extraction.get("lighting", {}).get(
                    "direction", ""
                ),
                "lighting_style": extraction.get("lighting", {}).get(
                    "lighting_style", ""
                ),
                "brightness_distribution": extraction.get("lighting", {}).get(
                    "brightness_distribution", ""
                ),
                "shadow_quality": extraction.get("lighting", {}).get(
                    "shadow_quality", ""
                ),
                "highlight_intensity": extraction.get("lighting", {}).get(
                    "highlight_intensity", ""
                ),
                "lighting_consistency": extraction.get("lighting", {}).get(
                    "lighting_consistency", ""
                ),
                "shadow_direction": extraction.get("lighting", {}).get(
                    "shadow_direction", ""
                ),
                "mood_lighting": extraction.get("lighting", {}).get(
                    "mood_lighting", ""
                ),
                "time_of_day": extraction.get("lighting", {}).get(
                    "time_of_day", ""
                ),
                "lighting_color": extraction.get("lighting", {}).get(
                    "lighting_color", ""
                ),
                "product_presentation": extraction.get(
                    "content_elements", {}
                ).get("product_presentation", ""),
                "product_angle": extraction.get("content_elements", {}).get(
                    "product_angle", ""
                ),
                "text_elements": ", ".join(
                    extraction.get("content_elements", {}).get(
                        "text_elements", []
                    )
                ),
                "human_elements": extraction.get("content_elements", {}).get(
                    "human_elements", ""
                ),
                "props_context": extraction.get("content_elements", {}).get(
                    "props_context", ""
                ),
                "product_to_frame_ratio": extraction.get(
                    "content_elements", {}
                ).get("product_to_frame_ratio", 50),
                "product_position": extraction.get("content_elements", {}).get(
                    "product_position", "center"
                ),
                "camera_perspective": extraction.get(
                    "content_elements", {}
                ).get("camera_perspective", "eye-level"),
                "product_visibility": extraction.get(
                    "content_elements", {}
                ).get("product_visibility", "full"),
                "visual_prominence": extraction.get("content_elements", {}).get(
                    "visual_prominence", "balanced"
                ),
                "product_context": extraction.get("content_elements", {}).get(
                    "product_context", "isolated"
                ),
                "scene_type": extraction.get("content_elements", {}).get(
                    "scene_type", "indoor"
                ),
                "environmental_context": extraction.get(
                    "content_elements", {}
                ).get("environmental_context", "home"),
                "specific_location": extraction.get("content_elements", {}).get(
                    "specific_location", "other"
                ),
                "usage_scenario": extraction.get("content_elements", {}).get(
                    "usage_scenario", "display"
                ),
                "human_presence": extraction.get("content_elements", {}).get(
                    "human_presence", False
                ),
                "lifestyle_appeal": extraction.get("content_elements", {}).get(
                    "lifestyle_appeal", "neutral"
                ),
                "product_count": extraction.get("content_elements", {}).get(
                    "product_count", ""
                ),
                "product_interaction": extraction.get(
                    "content_elements", {}
                ).get("product_interaction", ""),
                "product_state": extraction.get("content_elements", {}).get(
                    "product_state", ""
                ),
                "product_scale": extraction.get("content_elements", {}).get(
                    "product_scale", ""
                ),
                "product_orientation": extraction.get(
                    "content_elements", {}
                ).get("product_orientation", ""),
                "weather_condition": extraction.get("content_elements", {}).get(
                    "weather_condition", ""
                ),
                "time_period": extraction.get("content_elements", {}).get(
                    "time_period", ""
                ),
                "season": extraction.get("content_elements", {}).get(
                    "season", ""
                ),
                "atmosphere": extraction.get("content_elements", {}).get(
                    "atmosphere", ""
                ),
                "person_count": extraction.get("content_elements", {}).get(
                    "person_count", ""
                ),
                "person_age_group": extraction.get("content_elements", {}).get(
                    "person_age_group", ""
                ),
                "person_gender": extraction.get("content_elements", {}).get(
                    "person_gender", ""
                ),
                "person_ethnicity": extraction.get("content_elements", {}).get(
                    "person_ethnicity", ""
                ),
                "person_relationship_type": extraction.get(
                    "content_elements", {}
                ).get("person_relationship_type", ""),
                "person_activity": extraction.get("content_elements", {}).get(
                    "person_activity", ""
                ),
                "object_diversity": extraction.get("content_elements", {}).get(
                    "object_diversity", ""
                ),
                "temporal_context": extraction.get("content_elements", {}).get(
                    "temporal_context", ""
                ),
                "relationship_depiction": extraction.get(
                    "content_elements", {}
                ).get("relationship_depiction", ""),
                "use_case_clarity": extraction.get("content_elements", {}).get(
                    "use_case_clarity", ""
                ),
                "problem_solution_narrative": extraction.get(
                    "content_elements", {}
                ).get("problem_solution_narrative", ""),
                "furniture_presence": extraction.get(
                    "content_elements", {}
                ).get("furniture_presence", ""),
                "appliance_presence": extraction.get(
                    "content_elements", {}
                ).get("appliance_presence", ""),
                "vehicle_presence": extraction.get("content_elements", {}).get(
                    "vehicle_presence", ""
                ),
                "outdoor_equipment_presence": extraction.get(
                    "content_elements", {}
                ).get("outdoor_equipment_presence", ""),
                "electronic_devices_presence": extraction.get(
                    "content_elements", {}
                ).get("electronic_devices_presence", ""),
                "natural_elements_presence": extraction.get(
                    "content_elements", {}
                ).get("natural_elements_presence", ""),
                "architectural_elements_presence": extraction.get(
                    "content_elements", {}
                ).get("architectural_elements_presence", ""),
                "background_content_type": extraction.get(
                    "content_elements", {}
                ).get("background_content_type", ""),
                "background_complexity": extraction.get(
                    "content_elements", {}
                ).get("background_complexity", ""),
                "background_relevance": extraction.get(
                    "content_elements", {}
                ).get("background_relevance", ""),
                "foreground_background_separation": extraction.get(
                    "content_elements", {}
                ).get("foreground_background_separation", ""),
                "content_storytelling": extraction.get(
                    "content_elements", {}
                ).get("content_storytelling", ""),
                "image_quality": extraction.get("technical_quality", {}).get(
                    "image_quality", ""
                ),
                "brand_consistency": extraction.get(
                    "technical_quality", {}
                ).get("brand_consistency", ""),
                "cta_visuals": ", ".join(
                    extraction.get("technical_quality", {}).get(
                        "cta_visuals", []
                    )
                ),
                "content_clarity": extraction.get("technical_quality", {}).get(
                    "content_clarity", ""
                ),
                "visual_impact": extraction.get("technical_quality", {}).get(
                    "visual_impact", ""
                ),
                "information_density": extraction.get(
                    "technical_quality", {}
                ).get("information_density", ""),
                "content_authenticity": extraction.get(
                    "technical_quality", {}
                ).get("content_authenticity", ""),
                "sharpness": extraction.get("technical_quality", {}).get(
                    "sharpness", ""
                ),
                "exposure_quality": extraction.get("technical_quality", {}).get(
                    "exposure_quality", ""
                ),
                "noise_level": extraction.get("technical_quality", {}).get(
                    "noise_level", ""
                ),
                "primary_appeal": extraction.get("overall_assessment", {}).get(
                    "primary_appeal", ""
                ),
                "target_audience": extraction.get("overall_assessment", {}).get(
                    "target_audience", ""
                ),
                "emotional_tone": extraction.get("overall_assessment", {}).get(
                    "emotional_tone", ""
                ),
                "urgency_level": extraction.get("overall_assessment", {}).get(
                    "urgency_level", ""
                ),
                "value_proposition": extraction.get(
                    "overall_assessment", {}
                ).get("value_proposition", ""),
                "testimonial_presence": extraction.get("social_proof", {}).get(
                    "testimonial_presence", ""
                ),
                "rating_display": extraction.get("social_proof", {}).get(
                    "rating_display", ""
                ),
                "user_count": extraction.get("social_proof", {}).get(
                    "user_count", ""
                ),
                "social_media_elements": extraction.get("social_proof", {}).get(
                    "social_media_elements", ""
                ),
                "scene_mood": extraction.get("scene_context", {}).get(
                    "scene_mood", ""
                ),
                "activity_level": extraction.get("scene_context", {}).get(
                    "activity_level", ""
                ),
                "spatial_relationship": extraction.get("scene_context", {}).get(
                    "spatial_relationship", ""
                ),
                "context_richness": extraction.get("scene_context", {}).get(
                    "context_richness", ""
                ),
            }
            flattened_data.append(flat_row)

        return pd.DataFrame(flattened_data)

    def _load_progress(self, progress_file: Path) -> Dict[str, Any]:
        """Load progress from checkpoint file.

        Args:
            progress_file: Path to the progress JSON file.

        Returns:
            Dictionary containing progress data (completed_images,
            completed_batches, failed_images, failed_batches).
            Returns empty dict if file doesn't exist.
        """
        if progress_file.exists():
            try:
                with open(progress_file, "r", encoding="utf-8") as file_handle:
                    return json.load(file_handle)
            except Exception as error:
                logger.warning("Could not load progress file: %s", error)
        return {
            "completed_images": [],
            "completed_batches": [],
            "failed_images": [],
            "failed_batches": {},
        }

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def _save_progress(
        self,
        progress_file: Path,
        completed_images: List[str],
        completed_batches: List[int],
        failed_images: Optional[List[str]] = None,
        failed_batches: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save progress to checkpoint file.

        Args:
            progress_file: Path to the progress JSON file.
            completed_images: List of completed image filenames.
            completed_batches: List of completed batch numbers.
            failed_images: Optional list of failed image filenames.
            failed_batches: Optional dictionary of failed batch information.
        """
        try:
            progress_data = {
                "completed_images": completed_images,
                "completed_batches": completed_batches,
                "failed_images": failed_images or [],
                "failed_batches": failed_batches or {},
                "last_updated": time.time(),
            }
            with open(progress_file, "w", encoding="utf-8") as file_handle:
                json.dump(
                    progress_data, file_handle, indent=2, ensure_ascii=False
                )
        except Exception as error:
            logger.warning("Could not save progress: %s", error)

    def _save_checkpoint(
        self,
        checkpoint_file: Path,
        all_analyses: List[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Save checkpoint with current extraction results.

        Args:
            checkpoint_file: Path to the checkpoint JSON file.
            all_analyses: List of all extraction results to save.
            output_path: Path to output directory for CSV checkpoint.
        """
        try:
            # Save JSON checkpoint
            with open(checkpoint_file, "w", encoding="utf-8") as file_handle:
                json.dump(
                    all_analyses, file_handle, indent=2, ensure_ascii=False
                )

            # Also update CSV if we have analyses
            if all_analyses:
                df = self.flatten_analysis_data(all_analyses)
                csv_path = output_path / "image_features_checkpoint.csv"
                df.to_csv(csv_path, index=False)
        except Exception as error:
            logger.warning("Could not save checkpoint: %s", error)

    # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments,broad-exception-caught
    def extract_features_from_folder(
        self,
        images_folder: str = None,
        output_dir: str = "data",
        batch_size: int = 5,
        rate_limit_delay: float = 3.0,
        performance_label: str = "unknown",
        resume: bool = True,
    ):
        """
        Extract features from all images in a folder directly without
        pre-classification.

        This is an alternative to process_all_batches() that doesn't require
        pre-prepared top_150_images.csv and bottom_150_images.csv files.

        Args:
            images_folder: Directory containing images
                (defaults to self.images_folder)
            output_dir: Directory to save results
            batch_size: Number of images per batch
            rate_limit_delay: Delay between batches in seconds
            performance_label: Label to assign to all images
                (default: "unknown")
            resume: Whether to resume from previous checkpoint
                (default: True)

        Returns:
            List of extraction results
        """
        # Use provided folder or default
        folder = (
            Path(images_folder) if images_folder else Path(self.images_folder)
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Progress tracking files
        progress_file = output_path / "analysis_progress.json"
        checkpoint_file = output_path / "image_features_checkpoint.json"

        # Get all image files
        image_extensions = Constants.SUPPORTED_IMAGE_EXTENSIONS
        image_files = sorted(
            [
                f.name
                for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
        )

        if not image_files:
            logger.error("No image files found in %s", folder)
            return []

        # Load previous progress if resuming
        completed_images = set()
        completed_batches = set()
        failed_images = set()
        failed_batches = {}
        all_analyses = []

        if resume and checkpoint_file.exists():
            try:
                logger.info("Loading checkpoint...")
                with open(
                    checkpoint_file, "r", encoding="utf-8"
                ) as file_handle:
                    all_analyses = json.load(file_handle)
                # Extract completed image filenames from checkpoint
                completed_images = {
                    extraction.get("filename", "")
                    for extraction in all_analyses
                }
                completed_images.discard("")  # Remove empty strings

                # Load batch progress
                progress_data = self._load_progress(progress_file)
                completed_batches = set(
                    progress_data.get("completed_batches", [])
                )
                failed_images = set(progress_data.get("failed_images", []))
                failed_batches = progress_data.get("failed_batches", {})

                completed_count = len(completed_images)
                logger.info(
                    " Resumed: %d images already extracted",
                    completed_count,
                )
                batches_count = len(completed_batches)
                logger.info("   Completed batches: %d", batches_count)
                if failed_images:
                    failed_count = len(failed_images)
                    logger.info(
                        "   Failed images: %d (will retry)", failed_count
                    )
                if failed_batches:
                    failed_batches_count = len(failed_batches)
                    logger.info(
                        "   Failed batches: %d (will retry)",
                        failed_batches_count,
                    )
            except Exception as error:
                logger.warning("Could not load checkpoint: %s", error)
                logger.info("   Starting fresh...")
                all_analyses = []
                completed_images = set()
                completed_batches = set()
                failed_images = set()
                failed_batches = {}

        # Update images_folder for extract_batch_features
        original_folder = self.images_folder
        self.images_folder = str(folder)

        logger.info("=" * 60)
        logger.info("GPT-4 Vision Image Feature Extractor")
        logger.info("=" * 60)
        folder_resolved = folder.resolve()
        logger.info("Images folder: %s", folder_resolved)
        total_images = len(image_files)
        logger.info("Total images: %d", total_images)
        completed_count = len(completed_images)
        logger.info("Already completed: %d", completed_count)
        remaining_count = total_images - completed_count
        logger.info("Remaining: %d", remaining_count)
        logger.info("Batch size: %d", batch_size)
        total_batches = (len(image_files) + batch_size - 1) // batch_size
        logger.info("Total batches: %d", total_batches)
        logger.info("=" * 60)

        # Process batches
        for i in range(0, len(image_files), batch_size):
            batch_num = (i // batch_size) + 1
            batch_files = image_files[i : i + batch_size]

            # Skip if batch already completed
            if resume and batch_num in completed_batches:
                logger.info(
                    "\n[Batch %d/%d]   Skipping (already completed)",
                    batch_num,
                    total_batches,
                )
                continue

            # Filter out already completed images
            remaining_files = [
                f for f in batch_files if f not in completed_images
            ]

            if not remaining_files:
                logger.info(
                    "\n[Batch %d/%d]   Skipping (all images already "
                    "extracted)",
                    batch_num,
                    total_batches,
                )
                completed_batches.add(batch_num)
                # Remove from failed batches if it was previously failed
                failed_batches.pop(str(batch_num), None)
                self._save_progress(
                    progress_file,
                    list(completed_images),
                    list(completed_batches),
                    list(failed_images),
                    failed_batches,
                )
                continue

            # If batch was previously partially successful, only retry
            # missing images
            if resume and str(batch_num) in failed_batches:
                batch_failure_info = failed_batches[str(batch_num)]
                if batch_failure_info.get("error_type") == "partial_success":
                    missing_images = set(
                        batch_failure_info.get("missing_images", [])
                    )
                    # Only retry images that were missing
                    remaining_files = [
                        f for f in remaining_files if f in missing_images
                    ]
                    if remaining_files:
                        retry_count = len(remaining_files)
                        logger.info(
                            "   Retrying %d missing images from previous "
                            "partial success",
                            retry_count,
                        )
                    else:
                        # All missing images are now completed, mark batch
                        # as complete
                        completed_batches.add(batch_num)
                        failed_batches.pop(str(batch_num), None)
                        self._save_progress(
                            progress_file,
                            list(completed_images),
                            list(completed_batches),
                            list(failed_images),
                            failed_batches,
                        )
                        continue

            remaining_count = len(remaining_files)
            logger.info(
                "\n[Batch %d/%d] extracting %d images...",
                batch_num,
                total_batches,
                remaining_count,
            )
            if len(remaining_files) < len(batch_files):
                already_done = len(batch_files) - len(remaining_files)
                logger.info("   (%d already completed)", already_done)

            extraction, error_info = self.extract_batch_features(
                remaining_files, performance_label, batch_num
            )

            if extraction:
                # Success (full or partial) - add results
                all_analyses.extend(extraction)
                # Track completed images
                for result in extraction:
                    filename = result.get("filename", "")
                    if filename:
                        completed_images.add(filename)
                        # Remove from failed images if it was previously failed
                        failed_images.discard(filename)

                # Handle partial success (some images missing from response)
                if (
                    error_info
                    and error_info.get("error_type") == "partial_success"
                ):
                    missing_images = set(error_info.get("missing_images", []))
                    # Mark missing images as failed for retry
                    for img in missing_images:
                        failed_images.add(img)

                    extracted_count = len(extraction)
                    remaining_count = len(remaining_files)
                    missing_count = len(missing_images)
                    logger.info(
                        " Batch %d partial success: %d/%d images extracted",
                        batch_num,
                        extracted_count,
                        remaining_count,
                    )
                    logger.info(
                        "   Missing images (%d) will be retried: %s",
                        missing_count,
                        missing_images,
                    )

                    # Don't mark batch as completed if there are missing images
                    # This ensures the batch will be retried for missing images
                    if missing_images:
                        # Record partial failure info
                        failed_batches[str(batch_num)] = {
                            "error_type": "partial_success",
                            "error_message": error_info.get(
                                "error_message", ""
                            ),
                            "missing_images": list(missing_images),
                            "successful_images": list(
                                error_info.get("successful_images", [])
                            ),
                            "timestamp": time.time(),
                        }
                else:
                    # Full success - mark batch as completed
                    completed_batches.add(batch_num)
                    # Remove from failed batches if it was previously failed
                    failed_batches.pop(str(batch_num), None)
                    extraction_count = len(extraction)
                    logger.info(
                        " Batch %d complete: %d images extracted",
                        batch_num,
                        extraction_count,
                    )

                # Save checkpoint after each successful batch (even if partial)
                self._save_checkpoint(
                    checkpoint_file, all_analyses, output_path
                )
                self._save_progress(
                    progress_file,
                    list(completed_images),
                    list(completed_batches),
                    list(failed_images),
                    failed_batches,
                )
            else:
                # Failed - record error info
                error_type = (
                    error_info.get("error_type", "unknown")
                    if error_info
                    else "unknown"
                )
                error_msg = (
                    error_info.get("error_message", "Unknown error")
                    if error_info
                    else "Unknown error"
                )

                # Record failed images
                failed_image_list = (
                    error_info.get("image_filenames", remaining_files)
                    if error_info
                    else remaining_files
                )
                for img in failed_image_list:
                    failed_images.add(img)

                # Record failed batch info
                failed_batches[str(batch_num)] = {
                    "error_type": error_type,
                    "error_message": error_msg,
                    "image_filenames": failed_image_list,
                    "timestamp": time.time(),
                }

                error_preview = error_msg[:100]
                logger.info(
                    "WARNING: Batch %d failed (%s): %s",
                    batch_num,
                    error_type,
                    error_preview,
                )
                logger.info("   Will retry on next run")

                # Still save progress to record the failure
                self._save_progress(
                    progress_file,
                    list(completed_images),
                    list(completed_batches),
                    list(failed_images),
                    failed_batches,
                )

                # Save checkpoint even if batch failed
                # (to preserve completed images)
                if all_analyses:
                    self._save_checkpoint(
                        checkpoint_file, all_analyses, output_path
                    )

            # Rate limiting
            if batch_num < total_batches:
                logger.info(
                    "   Waiting %ds before next batch...", rate_limit_delay
                )
                time.sleep(rate_limit_delay)

        # Restore original folder
        self.images_folder = original_folder

        logger.info("\n%s", "=" * 60)
        logger.info("extraction Complete!")
        analyses_count = len(all_analyses)
        logger.info("Total images extracted: %d", analyses_count)
        logger.info("%s", "=" * 60)

        # Save final results
        if all_analyses:
            # Save JSON (final version)
            json_path = output_path / "image_features.json"
            with open(json_path, "w", encoding="utf-8") as file_handle:
                json.dump(
                    all_analyses, file_handle, indent=2, ensure_ascii=False
                )
            logger.info("\n Saved JSON: %s", json_path)

            # Save CSV (final version)
            df = self.flatten_analysis_data(all_analyses)
            csv_path = output_path / "image_features.csv"
            df.to_csv(csv_path, index=False)
            logger.info(" Saved CSV: %s", csv_path)
            row_count = df.shape[0]
            col_count = df.shape[1]
            logger.info(
                "   Shape: %s (%d rows, %d columns)",
                df.shape,
                row_count,
                col_count,
            )

            # Clean up checkpoint files if everything completed successfully
            if len(completed_images) >= len(image_files):
                logger.info(
                    "\n All images completed! Cleaning up checkpoint files..."
                )
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                if progress_file.exists():
                    progress_file.unlink()
                checkpoint_csv = output_path / "image_features_checkpoint.csv"
                if checkpoint_csv.exists():
                    checkpoint_csv.unlink()
            else:
                remaining_count = len(image_files) - len(completed_images)
                failed_count = len(failed_images)
                logger.info(
                    "\nProgress saved. %d images remaining.", remaining_count
                )
                if failed_count > 0:
                    logger.info(
                        "   Failed images: %d (will be retried)", failed_count
                    )
                logger.info(
                    "   Run again with --resume to continue from where you "
                    "left off."
                )

                # Save failed images list for reference
                if failed_images:
                    failed_list_file = output_path / "failed_images.json"
                    failed_data = {
                        "failed_images": list(failed_images),
                        "failed_batches": failed_batches,
                        "total_failed": len(failed_images),
                        "last_updated": time.time(),
                    }
                    with open(
                        failed_list_file, "w", encoding="utf-8"
                    ) as file_handle:
                        json.dump(
                            failed_data,
                            file_handle,
                            indent=2,
                            ensure_ascii=False,
                        )
                    logger.info(
                        "   Failed images list saved to: %s", failed_list_file
                    )

        return all_analyses
