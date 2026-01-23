"""Interaction feature creation for creative performance prediction.

This module creates domain-specific interaction features that capture
combinations of creative elements that may have synergistic effects
on performance (e.g., "Red CTA button" performs differently than
"Red" + "CTA button" considered separately).
"""

import logging
from typing import List, Optional

import pandas as pd
from scipy.stats import chi2_contingency


logger = logging.getLogger(__name__)


# Common column names in the creative dataset
CREATIVE_COLUMNS = {
    # Platform and format
    "platform": ["platform", "platform_name"],
    "format": ["format", "creative_format", "ad_format"],
    # CTA elements
    "cta_button": ["cta_button", "has_cta", "button_present"],
    "cta_color": ["cta_color", "button_color"],
    "cta_text": ["cta_text", "button_text"],
    "cta_position": ["cta_position", "button_position"],
    # Text elements
    "headline_text": ["headline", "headline_text", "title"],
    "headline_size": ["headline_size", "title_size", "font_size"],
    "headline_position": ["headline_position", "title_position"],
    "body_text": ["body_text", "description"],
    # Visual elements
    "dominant_color": ["dominant_color", "primary_color", "main_color"],
    "color_vibrancy": ["color_vibrancy", "vibrance"],
    "brightness": ["brightness", "lightness"],
    # Human and product
    "human_presence": ["human_presence", "has_human", "person_present"],
    "human_count": ["human_count", "num_people", "n_humans"],
    "product_interaction": ["product_interaction", "product_type"],
    # Layout
    "layout": ["layout", "layout_type", "composition"],
    "text_overlay": ["text_overlay", "has_text_overlay"],
}


def _find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """Find the actual column name from a list of possible names.

    Args:
        df: Input dataframe.
        possible_names: List of possible column names to search for.

    Returns:
        The first matching column name, or None if not found.
    """
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def _create_platform_format(df: pd.DataFrame) -> int:
    """Create platform × format interaction feature.

    Returns:
        Count of created features (1 or 0).
    """
    platform = _find_column(df, CREATIVE_COLUMNS["platform"])
    format_col = _find_column(df, CREATIVE_COLUMNS["format"])

    if platform and format_col:
        df["platform_format"] = (
            df[platform].astype(str) + "_" + df[format_col].astype(str)
        )
        logger.debug("Created platform_format interaction")
        return 1
    return 0


def _create_cta_color_combo(df: pd.DataFrame) -> int:
    """Create CTA button × color interaction feature.

    Returns:
        Count of created features (1 or 0).
    """
    cta_button = _find_column(df, CREATIVE_COLUMNS["cta_button"])
    cta_color = _find_column(df, CREATIVE_COLUMNS["cta_color"])

    if cta_button and cta_color:
        if df[cta_button].dtype == bool:
            df["cta_color_combo"] = df.apply(
                lambda row: (
                    f"has_cta_{row[cta_color]}" if row[cta_button] else "no_cta"
                ),
                axis=1,
            )
        else:
            df["cta_color_combo"] = (
                df[cta_button].astype(str) + "_" + df[cta_color].astype(str)
            )
        logger.debug("Created cta_color_combo interaction")
        return 1
    return 0


def _create_cta_position_combo(df: pd.DataFrame) -> int:
    """Create CTA button × position interaction feature.

    Returns:
        Count of created features (1 or 0).
    """
    cta_button = _find_column(df, CREATIVE_COLUMNS["cta_button"])
    cta_position = _find_column(df, CREATIVE_COLUMNS["cta_position"])

    if cta_button and cta_position:
        if df[cta_button].dtype == bool:
            df["cta_position_combo"] = df.apply(
                lambda row: (
                    f"cta_at_{row[cta_position]}"
                    if row[cta_button]
                    else "no_cta"
                ),
                axis=1,
            )
        else:
            df["cta_position_combo"] = (
                df[cta_button].astype(str) + "_" + df[cta_position].astype(str)
            )
        logger.debug("Created cta_position_combo interaction")
        return 1
    return 0


def _create_headline_layout(df: pd.DataFrame) -> int:
    """Create headline size × position interaction feature.

    Returns:
        Count of created features (1 or 0).
    """
    headline_size = _find_column(df, CREATIVE_COLUMNS["headline_size"])
    headline_position = _find_column(df, CREATIVE_COLUMNS["headline_position"])

    if headline_size and headline_position:
        df["headline_layout"] = (
            df[headline_size].astype(str)
            + "_"
            + df[headline_position].astype(str)
        )
        logger.debug("Created headline_layout interaction")
        return 1
    return 0


def _create_human_product_combo(df: pd.DataFrame) -> int:
    """Create human presence × product interaction feature.

    Returns:
        Count of created features (1 or 0).
    """
    human_presence = _find_column(df, CREATIVE_COLUMNS["human_presence"])
    product_interaction = _find_column(
        df, CREATIVE_COLUMNS["product_interaction"]
    )

    if human_presence and product_interaction:
        df["human_product_combo"] = (
            df[human_presence].astype(str)
            + "_"
            + df[product_interaction].astype(str)
        )
        logger.debug("Created human_product_combo interaction")
        return 1
    return 0


def _create_visual_intensity(df: pd.DataFrame) -> int:
    """Create color vibrancy × brightness interaction feature.

    Returns:
        Count of created features (1 or 0).
    """
    color_vibrancy = _find_column(df, CREATIVE_COLUMNS["color_vibrancy"])
    brightness = _find_column(df, CREATIVE_COLUMNS["brightness"])

    if color_vibrancy and brightness:
        vib_bins = pd.cut(
            df[color_vibrancy],
            bins=[0, 0.33, 0.67, 1.0],
            labels=["low_vib", "med_vib", "high_vib"],
        )
        bright_bins = pd.cut(
            df[brightness],
            bins=[0, 0.33, 0.67, 1.0],
            labels=["dark", "medium", "bright"],
        )
        df["visual_intensity"] = (
            vib_bins.astype(str) + "_" + bright_bins.astype(str)
        )
        logger.debug("Created visual_intensity interaction")
        return 1
    return 0


def _create_layout_text_combo(df: pd.DataFrame) -> int:
    """Create layout × text overlay interaction feature.

    Returns:
        Count of created features (1 or 0).
    """
    layout = _find_column(df, CREATIVE_COLUMNS["layout"])
    text_overlay = _find_column(df, CREATIVE_COLUMNS["text_overlay"])

    if layout and text_overlay:
        df["layout_text_combo"] = (
            df[layout].astype(str) + "_" + df[text_overlay].astype(str)
        )
        logger.debug("Created layout_text_combo interaction")
        return 1
    return 0


def _create_human_product_type(df: pd.DataFrame) -> int:
    """Create human count × product type interaction feature.

    Returns:
        Count of created features (1 or 0).
    """
    human_count = _find_column(df, CREATIVE_COLUMNS["human_count"])
    product_interaction = _find_column(
        df, CREATIVE_COLUMNS["product_interaction"]
    )

    if human_count and product_interaction:
        if df[human_count].dtype in [int, float]:
            human_bins = pd.cut(
                df[human_count],
                bins=[-1, 0, 1, 5, float("inf")],
                labels=["no_human", "single_person", "few_people", "crowd"],
            )
        else:
            human_bins = df[human_count]

        df["human_product_type"] = (
            human_bins.astype(str) + "_" + df[product_interaction].astype(str)
        )
        logger.debug("Created human_product_type interaction")
        return 1
    return 0


def _create_platform_cta(df: pd.DataFrame) -> int:
    """Create platform × CTA presence interaction feature.

    Returns:
        Count of created features (1 or 0).
    """
    platform = _find_column(df, CREATIVE_COLUMNS["platform"])
    cta_button = _find_column(df, CREATIVE_COLUMNS["cta_button"])

    if platform and cta_button:
        if df[cta_button].dtype == bool:
            df["platform_cta"] = (
                df[platform].astype(str)
                + "_"
                + df[cta_button].apply(lambda x: "has_cta" if x else "no_cta")
            )
        else:
            df["platform_cta"] = (
                df[platform].astype(str) + "_" + df[cta_button].astype(str)
            )
        logger.debug("Created platform_cta interaction")
        return 1
    return 0


def _create_format_human(df: pd.DataFrame) -> int:
    """Create format × human presence interaction feature.

    Returns:
        Count of created features (1 or 0).
    """
    format_col = _find_column(df, CREATIVE_COLUMNS["format"])
    human_presence = _find_column(df, CREATIVE_COLUMNS["human_presence"])

    if format_col and human_presence:
        df["format_human"] = (
            df[format_col].astype(str) + "_" + df[human_presence].astype(str)
        )
        logger.debug("Created format_human interaction")
        return 1
    return 0


def create_interaction_features(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Create domain-specific interaction features for creative analysis.

    This function creates interaction features that capture meaningful
    combinations of creative elements that may have synergistic effects.

    Args:
        df: Input dataframe with raw creative features.
        verbose: Whether to log which interactions were created.

    Returns:
        Dataframe with additional interaction feature columns.

    Examples:
        >>> df = create_interaction_features(df_raw)
        >>> # New columns like 'platform_format', 'cta_color_combo' added
    """
    df = df.copy()
    created_count = 0

    # Use helper functions to create each interaction type
    created_count += _create_platform_format(df)
    created_count += _create_cta_color_combo(df)
    created_count += _create_cta_position_combo(df)
    created_count += _create_headline_layout(df)
    created_count += _create_human_product_combo(df)
    created_count += _create_visual_intensity(df)
    created_count += _create_layout_text_combo(df)
    created_count += _create_human_product_type(df)
    created_count += _create_platform_cta(df)
    created_count += _create_format_human(df)

    if verbose and created_count > 0:
        logger.info("Created %d interaction features", created_count)

    return df


def get_interaction_features(
    df: pd.DataFrame,
) -> List[str]:
    """Get list of interaction feature column names that would be created.

    Useful for:
    - Preparing feature lists before feature creation
    - Documentation
    - Config file generation

    Args:
        df: Sample dataframe to check which columns exist.

    Returns:
        List of interaction feature names that would be created.
    """
    interactions = []

    platform = _find_column(df, CREATIVE_COLUMNS["platform"])
    format_col = _find_column(df, CREATIVE_COLUMNS["format"])
    cta_button = _find_column(df, CREATIVE_COLUMNS["cta_button"])
    cta_color = _find_column(df, CREATIVE_COLUMNS["cta_color"])
    cta_position = _find_column(df, CREATIVE_COLUMNS["cta_position"])
    headline_size = _find_column(df, CREATIVE_COLUMNS["headline_size"])
    headline_position = _find_column(df, CREATIVE_COLUMNS["headline_position"])
    human_presence = _find_column(df, CREATIVE_COLUMNS["human_presence"])
    product_interaction = _find_column(
        df, CREATIVE_COLUMNS["product_interaction"]
    )
    color_vibrancy = _find_column(df, CREATIVE_COLUMNS["color_vibrancy"])
    brightness = _find_column(df, CREATIVE_COLUMNS["brightness"])
    layout = _find_column(df, CREATIVE_COLUMNS["layout"])
    text_overlay = _find_column(df, CREATIVE_COLUMNS["text_overlay"])
    human_count = _find_column(df, CREATIVE_COLUMNS["human_count"])

    if platform and format_col:
        interactions.append("platform_format")

    if cta_button and cta_color:
        interactions.append("cta_color_combo")

    if cta_button and cta_position:
        interactions.append("cta_position_combo")

    if headline_size and headline_position:
        interactions.append("headline_layout")

    if human_presence and product_interaction:
        interactions.append("human_product_combo")

    if color_vibrancy and brightness:
        interactions.append("visual_intensity")

    if layout and text_overlay:
        interactions.append("layout_text_combo")

    if human_count and product_interaction:
        interactions.append("human_product_type")

    if platform and cta_button:
        interactions.append("platform_cta")

    if format_col and human_presence:
        interactions.append("format_human")

    return interactions


def discover_interactions(
    df: pd.DataFrame,
    target_col: str,
    max_interactions: int = 20,
    min_p_value: float = 0.05,
) -> List[str]:
    """Automatically discover high-value interaction features.

    Uses statistical testing to find feature combinations that have
    significant relationships with performance. KISS approach:
    1. Get top categorical features by lift (from pattern analysis)
    2. Create 2-way combinations
    3. Test with chi-square
    4. Keep only significant interactions

    Args:
        df: DataFrame with features and target.
        target_col: Target column (e.g., 'mean_roas').
        max_interactions: Maximum number of interactions to create.
        min_p_value: Minimum significance threshold (default: 0.05).

    Returns:
        List of discovered interaction feature names.
    """
    logger.info("Discovering high-value interaction features...")

    # Step 1: Get categorical columns
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != target_col]

    if len(categorical_cols) < 2:
        logger.warning("Need at least 2 categorical features for interactions")
        return []

    # Step 2: Create binary target (top 25% vs others)
    high_threshold = df[target_col].quantile(0.75)
    df["_temp_target"] = (df[target_col] >= high_threshold).astype(
        int
    )  # 1 for high, 0 for others

    # Step 3: Test all pairs
    discovered = []
    tested = 0

    for i, col1 in enumerate(categorical_cols):
        for col2 in categorical_cols[i + 1 :]:
            # Create interaction: col1_col2
            interaction_name = f"{col1}_{col2}"

            # Skip if too many interactions
            if len(discovered) >= max_interactions:
                break

            # Create interaction feature
            df["_temp_interaction"] = (
                df[col1].astype(str) + "_" + df[col2].astype(str)
            )

            # Build contingency table
            try:
                contingency = pd.crosstab(
                    df["_temp_interaction"], df["_temp_target"]
                )

                # Chi-square test
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue

                _, p_value, _, _ = chi2_contingency(contingency)

                tested += 1

                # Keep if significant
                if p_value < min_p_value:
                    # Make it permanent
                    df[interaction_name] = df["_temp_interaction"]
                    discovered.append(interaction_name)
                    logger.info(
                        "Discovered: %s (p=%.4f, categories=%d)",
                        interaction_name,
                        p_value,
                        contingency.shape[0],
                    )
            except RuntimeError:
                continue  # Skip problematic combinations

    # Cleanup temp column
    df.drop(
        columns=["_temp_target", "_temp_interaction"],
        errors="ignore",
        inplace=True,
    )

    logger.info(
        "Discovered %d significant interactions (tested %d combinations)",
        len(discovered),
        tested,
    )

    return discovered
