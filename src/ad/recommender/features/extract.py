"""Feature extraction and ROAS integration functions.

This module provides functions for:
- Extracting features from single images or batches of images
  using the GPT-4 Vision API
- Adding ROAS (Return on Ad Spend) data to extracted image features
  for model training and analysis
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.api_keys import get_openai_api_key

from .extractors.gpt4_feature_extractor import GPT4FeatureExtractor
from .transformers.gpt4_feature_transformer import convert_to_features

logger = logging.getLogger(__name__)


# pylint: disable=too-many-locals
def extract_single_image_features(
    extractor: GPT4FeatureExtractor,
    image_path: str,
    cpc_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Extract features from a single image and return feature values.

    Args:
        extractor: GPT4FeatureExtractor instance for API calls.
        image_path: Image path (absolute path or relative to images_folder).
        cpc_value: Optional value for CPC_transformed feature.

    Returns:
        Dictionary containing image filename as 'id' and feature values.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the GPT API response cannot be parsed.
        Exception: For other API or processing errors.
    """
    # Get image filename as id
    image_filename = os.path.basename(image_path)

    # Determine actual image path
    if os.path.isabs(image_path) or os.path.exists(image_path):
        # Absolute path or path in current directory
        actual_image_path = image_path
    else:
        # Try to find in images_folder
        actual_image_path = os.path.join(
            extractor.images_folder, image_filename
        )

    # Check if image exists
    if not os.path.exists(actual_image_path):
        raise FileNotFoundError(f"Image not found: {actual_image_path}")

    # Use extractor to analyze image (single image)
    try:
        # Directly call GPT API to analyze single image
        encoded_image = extractor.encode_image(actual_image_path)
        image_contents = [
            {
                "type": "image_url",
                "image_url": {
                    "url": (f"data:image/jpeg;base64,{encoded_image}")
                },
            }
        ]

        # Create feature extraction prompt (using existing prompt template)
        prompt = extractor.get_prompt([image_filename], "prediction")

        # Build message content
        content_parts = [{"type": "text", "text": prompt}]
        content_parts.extend(image_contents)

        # Call GPT API (using same model as extract_batch_features)
        # Try gpt-4.1 first, fallback to gpt-4o if it fails
        try:
            response = extractor.client.chat.completions.create(
                model="gpt-4.1",  # Use same model as extract_batch_features
                messages=[{"role": "user", "content": content_parts}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=4000,
            )
        # pylint: disable=broad-exception-caught
        except Exception as error:
            # If gpt-4.1 is not available, try gpt-4o
            logger.warning("WARNING: gpt-4.1 failed, trying gpt-4o: %s", error)
            try:
                response = extractor.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": content_parts}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=4000,
                )
            # pylint: disable=broad-exception-caught
            except Exception as error2:
                # Finally try gpt-4-vision-preview
                logger.warning(
                    "WARNING: gpt-4o failed, trying gpt-4-vision-preview: %s",
                    error2,
                )
                response = extractor.client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[{"role": "user", "content": content_parts}],
                    temperature=0.1,
                    max_tokens=4000,
                )

        # Parse response
        response_text = response.choices[0].message.content
        if not response_text:
            raise ValueError("Empty response from GPT API")

        # Parse JSON
        analysis_data = json.loads(response_text)

        # Handle different response formats
        if isinstance(analysis_data, dict) and "analyses" in analysis_data:
            gpt_result = analysis_data["analyses"][0]
        elif isinstance(analysis_data, list):
            gpt_result = analysis_data[0]
        elif isinstance(analysis_data, dict):
            gpt_result = analysis_data
        else:
            raise ValueError(
                f"Unexpected response format: {type(analysis_data)}"
            )

        # Map to 29 features (new model)
        features = convert_to_features(gpt_result)

        # Apply feature weights (no weights_file for single image,
        # just CPC if provided)
        if cpc_value is not None:
            cpc_transformed = 1.0 / (1.0 + cpc_value)
            features["CPC_transformed"] = cpc_transformed

        # Add id
        result = {"id": image_filename}
        result.update(features)

        return result

    # pylint: disable=broad-exception-caught
    except Exception as error:
        logger.error("ERROR: Error analyzing image %s: %s", image_path, error)
        raise


# pylint: disable=too-many-locals
def extract_batch_features(
    image_paths: List[str],
    output_csv: str = "new_images_features.csv",
    cpc_values: Optional[List[float]] = None,
    weights_file: str = (  # pylint: disable=unused-argument
        "data/echo/results_taboola_new/feature_weights.json"
    ),
    api_key: Optional[str] = None,
) -> str:
    """
    Extract features from images in batch and generate CSV file

    Args:
        image_paths: List of image paths (can be file paths or filenames,
            if filename then search in images_folder)
        output_csv: Output CSV file path
        cpc_values: List of CPC values (optional, corresponds to image_paths)
        weights_file: Path to weights file
        api_key: OpenAI API key (optional, will be read from environment)

    Returns:
        str: Output CSV file path
    """
    # Check API key
    if api_key is None:
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY "
                "environment variable or create ~/.devease/keys file"
            )

    # Initialize extractor
    extractor = GPT4FeatureExtractor(api_key)

    # Load selected features list (ensure correct order)
    selected_features_file = (
        "data/echo/results_taboola_new/selected_features.json"
    )
    selected_features = None
    if os.path.exists(selected_features_file):
        with open(selected_features_file, "r", encoding="utf-8") as file:
            selected_features = json.load(file).get("final_features")

    # Process each image
    results = []
    total = len(image_paths)

    logger.info("Extracting features from %d images...", total)
    logger.info("=" * 50)

    for i, image_path in enumerate(image_paths, 1):
        logger.info(
            "Extracting features from image %d/%d: %s",
            i,
            total,
            os.path.basename(image_path),
        )

        try:
            # Get corresponding CPC value
            cpc_value = None
            if cpc_values and i <= len(cpc_values):
                cpc_value = cpc_values[i - 1]

            # Extract features from image
            result = extract_single_image_features(
                extractor, image_path, cpc_value
            )
            results.append(result)

            logger.info("  Completed")

            # Rate limiting
            if i < total:
                time.sleep(2)

        # pylint: disable=broad-exception-caught
        except Exception as error:
            logger.error("  ERROR: %s", error)
            # Continue processing next image
            continue

    if not results:
        raise ValueError("No images were successfully processed")

    # Create DataFrame, ensure correct column order
    df = pd.DataFrame(results)

    # If selected_features is available, use it for column ordering
    if selected_features:
        # Column order: id + features (in selected_features order)
        columns = ["id"] + selected_features
        # Ensure all columns exist, fill missing columns with 0.0
        for col in columns:
            if col not in df.columns:
                df[col] = 0.0
        # Reorder columns
        df = df[columns]
    else:
        # If no selected_features file, use all columns from results
        # Ensure 'id' is first
        if "id" in df.columns:
            columns = ["id"] + [col for col in df.columns if col != "id"]
            df = df[columns]

    # Save CSV
    output_path = output_csv
    df.to_csv(output_path, index=False)

    logger.info("\nSuccessfully processed %d/%d images", len(results), total)
    logger.info("Results saved to: %s", output_path)

    return output_path


# pylint: disable=too-many-locals
def add_roas_to_features(
    features_csv: str = "data/image_features.csv",
    ad_data_csv: Optional[str] = None,
    output_csv: str = "data/features_with_roas.csv",
    synthetic: bool = False,
    synthetic_method: str = "feature_based",
) -> pd.DataFrame:  # pylint: disable=too-many-locals
    """
    Add ROAS data to image features.

    This function adds ad performance data (ROAS) to image features
    extracted by GPT-4 Vision, creating a merged dataset ready for
    model training and analysis.

    The function can:
    - Load and merge real ROAS data from ad performance CSV files
    - Generate synthetic ROAS values for testing when real data is unavailable
    - Automatically fall back to synthetic data if no matches are found

    Args:
        features_csv: Path to image features CSV file
        ad_data_csv: Path to ad performance data CSV file
            (optional, will use synthetic if not provided)
        output_csv: Path to output CSV file with features and ROAS data
        synthetic: If True, create synthetic ROAS values for testing
            instead of loading real data
        synthetic_method: Method for generating synthetic ROAS
            ('random' or 'feature_based')

    Returns:
        DataFrame containing image features with added ROAS data

    Raises:
        FileNotFoundError: If required files are not found
        ValueError: If data cannot be processed
    """
    # Import here to avoid circular dependency
    # pylint: disable=import-outside-toplevel
    from pathlib import Path

    from .lib import (
        create_synthetic_roas,
        load_ad_data,
        load_feature_data,
        merge_features_with_roas,
        parse_creative_info_from_filenames,
    )

    # Load image features data
    logger.info("Loading image features from: %s", features_csv)
    features_df = load_feature_data(features_csv)

    if synthetic:
        # Create synthetic ROAS values for testing
        logger.info("Creating synthetic ROAS values for testing...")
        merged_df = create_synthetic_roas(features_df, method=synthetic_method)
    else:
        # Try to load real ROAS data and add it to features
        if ad_data_csv and os.path.exists(ad_data_csv):
            logger.info("Loading ad performance data from: %s", ad_data_csv)
            ad_df = load_ad_data(ad_data_csv)

            # Extract creative info from filenames to enable matching
            logger.info("Extracting creative info from filenames...")
            info_df = parse_creative_info_from_filenames(features_df)

            # Merge ROAS data into features
            logger.info("Adding ROAS data to features...")
            merged_df = merge_features_with_roas(features_df, ad_df, info_df)

            # If no ROAS matches found, fall back to synthetic data
            if merged_df["roas_parsed"].notna().sum() == 0:
                logger.warning(
                    "No ROAS matches found. "
                    "Falling back to synthetic ROAS data."
                )
                merged_df = create_synthetic_roas(
                    features_df, method="feature_based"
                )
        else:
            # No ad data file provided or file doesn't exist, use synthetic
            if ad_data_csv:
                logger.warning("Ad data file not found: %s", ad_data_csv)
            else:
                logger.warning("No ad data file provided.")
            logger.warning("Creating synthetic ROAS data for testing.")
            merged_df = create_synthetic_roas(
                features_df, method="feature_based"
            )

    # Save output
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    # Log summary statistics
    # pylint: disable=logging-not-lazy
    logger.info("\n" + "=" * 60)
    logger.info("Features with ROAS data saved to: %s", output_path)
    logger.info("Dataset shape: %s", merged_df.shape)
    logger.info("ROAS statistics:")
    logger.info(
        "  - Records with ROAS: %d",
        merged_df["roas_parsed"].notna().sum(),
    )
    logger.info("  - Mean ROAS: %.4f", merged_df["roas_parsed"].mean())
    logger.info("  - Std ROAS: %.4f", merged_df["roas_parsed"].std())
    logger.info("  - Min ROAS: %.4f", merged_df["roas_parsed"].min())
    logger.info("  - Max ROAS: %.4f", merged_df["roas_parsed"].max())
    # pylint: disable=logging-not-lazy
    logger.info("=" * 60)

    return merged_df
