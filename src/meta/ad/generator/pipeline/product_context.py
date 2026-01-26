"""
Product Context Utilities

Helper functions to create and load product context for creative generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


@dataclass
class ProductIdentity:
    """Core product identity information."""

    product_name: str
    category: Optional[str] = None
    brand: Optional[str] = None
    brand_display_style: Optional[str] = None


@dataclass
class ProductMarketing:
    """Product marketing information."""

    target_audience: Optional[str] = None
    key_message: Optional[str] = None
    market: Optional[str] = None


@dataclass
class ProductContextFiles:
    """Paths to product context files."""

    product_context_file: Optional[Path] = None
    product_context_json: Optional[Path] = None


@dataclass
class ProductContextConfig:
    """Configuration for product context creation."""

    identity: ProductIdentity
    marketing: Optional[ProductMarketing] = None
    additional_context: Optional[str] = None
    files: Optional[ProductContextFiles] = None

    def __post_init__(self):
        """Initialize optional sub-configs with defaults."""
        if self.marketing is None:
            self.marketing = ProductMarketing()
        if self.files is None:
            self.files = ProductContextFiles()


def create_product_context(
    config: ProductContextConfig,
) -> Dict[str, Any]:
    """
    Helper function to create a product context dict.

    Priority:
    1. Load from JSON config file if provided (product_context_json)
    2. Load from JSON config file via auto-discovery
    3. Use provided parameters and load detailed context from text file

    Args:
        config: ProductContextConfig object with all product context parameters

    Returns:
        Product context dict for use with CreativePipeline
    """
    # Extract values from config
    product_name = config.identity.product_name
    brand = config.identity.brand
    category = config.identity.category
    brand_display_style = config.identity.brand_display_style
    target_audience = config.marketing.target_audience if config.marketing else None
    key_message = config.marketing.key_message if config.marketing else None
    market = config.marketing.market if config.marketing else None
    additional_context = config.additional_context
    product_context_file = config.files.product_context_file if config.files else None
    product_context_json = config.files.product_context_json if config.files else None

    # Priority 1: Try to load from JSON config file
    json_config = _load_json_config(
        product_context_json, product_name, brand
    )

    if json_config:
        return _finalize_json_config(
            json_config, product_name
        )

    # Priority 2: Use provided parameters and load detailed context from text file
    detailed_context = _load_detailed_context(
        product_context_file, product_name, brand
    )

    # Merge additional_context with detailed_context if both exist
    merged_context = _merge_contexts(additional_context, detailed_context)

    return {
        "product_name": product_name,
        "category": category or "General",
        "brand": brand or "N/A",
        "brand_display_style": brand_display_style or "exact",
        "target_audience": target_audience or "General consumers",
        "key_message": key_message or "Quality product",
        "market": market or "US",
        "additional_context": merged_context,
        "detailed_product_context": detailed_context,
        # Keep separate for prompt generation
    }


def _load_json_config(
    product_context_json: Optional[str],
    product_name: str,
    brand: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Load JSON config from explicit path or auto-discovery."""
    if product_context_json:
        return load_product_context_from_json(
            json_file_path=product_context_json
        )
    if brand or product_name:
        return load_product_context_from_json(
            product_name=product_name, brand=brand
        )
    return None


def _finalize_json_config(
    json_config: Dict[str, Any],
    product_name: str,
) -> Dict[str, Any]:
    """Finalize JSON config with detailed context and product name."""
    logger.info("Loaded product context from JSON config file")
    # Load detailed context from text file if specified in JSON
    if "product_context_file" in json_config:
        context_file_path = Path(json_config["product_context_file"])
        if not context_file_path.is_absolute():
            # Relative to project root
            project_root = Path(__file__).resolve().parents[3]
            context_file_path = project_root / context_file_path

        if context_file_path.exists():
            detailed_context = context_file_path.read_text(encoding="utf-8")
            json_config["detailed_product_context"] = detailed_context
            logger.info(
                "Loaded detailed context from %s",
                json_config["product_context_file"],
            )
    # Ensure product_name is set
    if not json_config.get("product_name"):
        json_config["product_name"] = product_name

    return json_config


def _load_detailed_context(
    product_context_file: Optional[str],
    product_name: str,
    brand: Optional[str],
) -> Optional[str]:
    """Load detailed product context from file."""
    if not (product_context_file or (brand and product_name)):
        return None

    detailed_context = load_product_context_from_file(
        context_file_path=product_context_file,
        product_name=product_name,
        brand=brand,
    )
    if detailed_context:
        logger.info(
            "Loaded detailed product context from file (%d chars)",
            len(detailed_context),
        )
    return detailed_context


def _merge_contexts(
    additional_context: Optional[str],
    detailed_context: Optional[str],
) -> Optional[str]:
    """Merge additional and detailed context."""
    if additional_context and detailed_context:
        return f"{detailed_context}\n\n{additional_context}"
    if detailed_context:
        return detailed_context
    return additional_context


def load_product_context_from_json(
    json_file_path: Optional[Path] = None,
    product_name: Optional[str] = None,
    brand: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load product context from JSON configuration file.

    Looks for JSON files in:
    1. Explicit path if provided
    2. data/product_context/{brand}_{product_name}.json
    3. data/product_context/{brand}_*.json
    4. data/product_context/*.json (first match)

    Args:
        json_file_path: Explicit path to JSON file
        product_name: Product name for auto-discovery
        brand: Brand name for auto-discovery

    Returns:
        Product context dict or None if not found
    """
    result = None
    # Try explicit path first
    if json_file_path:
        path = Path(json_file_path)
        if path.exists():
            result = _load_json_file(path)
        return result
    # Auto-discover JSON file
    data_dir = Path(__file__).resolve().parents[3] / "datasets" / "product_context"
    if not data_dir.exists():
        return None
    # Build search patterns
    patterns = []
    if brand and product_name:
        patterns.append(
            f"{brand.lower()}_{product_name.lower().replace(' ', '_')}.json"
        )
    if brand:
        patterns.append(f"{brand.lower()}_*.json")
    patterns.append("*.json")
    # Try each pattern
    for pattern in patterns:
        matches = list(data_dir.glob(pattern))
        if not matches:
            continue
        # Prefer files with product name in filename
        if product_name:
            preferred = [
                m
                for m in matches
                if product_name.lower().replace(" ", "_") in m.stem.lower()
            ]
            if preferred and (result := _load_json_file(preferred[0])) is not None:
                return result
        # Try first match
        if (result := _load_json_file(matches[0])) is not None:
            return result

    return None


def _load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Helper to load JSON file with error handling."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(
            "Failed to load JSON config from %s: %s", file_path, e
        )
        return None


def load_product_context_from_file(
    context_file_path: Optional[Path] = None,
    product_name: Optional[str] = None,
    brand: Optional[str] = None,
) -> Optional[str]:
    """
    Load detailed product context from a text file.

    Looks for product context files in:
    1. Explicit path if provided
    2. data/product_context/{brand}_{product_name}_*.txt
    3. data/product_context/{brand}_*.txt
    4. data/product_context/*.txt (first match)

    Args:
        context_file_path: Explicit path to context file
        product_name: Product name for auto-discovery
        brand: Brand name for auto-discovery

    Returns:
        Product context text or None if not found
    """
    if context_file_path:
        path = Path(context_file_path)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None
    # Auto-discover context file
    data_dir = Path(__file__).resolve().parents[3] / "datasets" / "product_context"
    if not data_dir.exists():
        return None
    # Try specific patterns
    patterns = []
    if brand and product_name:
        # Try: {brand}_{product_name}_*.txt
        patterns.append(
            f"{brand.lower()}_{product_name.lower().replace(' ', '_')}_*.txt"
        )
    if brand:
        # Try: {brand}_*.txt
        patterns.append(f"{brand.lower()}_*.txt")
    # Try: any .txt file
    patterns.append("*.txt")

    for pattern in patterns:
        matches = list(data_dir.glob(pattern))
        if matches:
            # Prefer files with "official" or "summary" in name
            preferred = [
                m
                for m in matches
                if "official" in m.stem.lower() or "summary" in m.stem.lower()
            ]
            if preferred:
                return preferred[0].read_text(encoding="utf-8")
            return matches[0].read_text(encoding="utf-8")

    return None
