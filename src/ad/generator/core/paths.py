"""
Path utilities for creative generation.

Provides centralized path management following the daily budget allocation
repo structure with customer/platform organization:

config/
  ad/
    recommender/
      {customer}/
        {platform}/
          {date}/
            recommendations.md
      gpt4/
        features.yaml
        prompts.yaml
    generator/
      prompts/
        {customer}/
          {platform}/
            {date}/
              {prompt_type}/
      generated/
        {customer}/
          {platform}/
            {date}/
              {model}/
      templates/
        {customer}/
          {platform}/
            generation_config.yaml
      {customer}/
        {platform}/
          generation_config.yaml
          prompt_templates.yaml
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)
# Default directories
DEFAULT_DATA_DIR = Path("datasets")
DEFAULT_CONFIG_DIR = Path("config")
DEFAULT_OUTPUT_DIR = Path("results")


class Paths:
    """
    Centralized path management with customer/platform structure.

    Example:
        paths = Paths(
            customer="moprobo",
            platform="taboola",
            date="2026-01-21"
        )
        recommendations_path = paths.recommendations()
        prompts_output_dir = paths.prompts_output("structured")
    """

    def __init__(
        self,
        customer: str,
        platform: str,
        date: Optional[str] = None,
        data_dir: Optional[Path] = None,
        config_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize path manager.

        Args:
            customer: Customer/account name
            platform: Platform name (taboola, facebook, etc.)
            date: Date string (YYYY-MM-DD), defaults to today
            data_dir: Base data directory (default: data/)
            config_dir: Base config directory (default: config/)
            output_dir: Base output directory (default: output/)
        """
        self.customer = customer.lower().replace("-", "_")
        self.platform = platform.lower().replace("-", "_")
        self.date = date or datetime.now().strftime("%Y-%m-%d")
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR

    def recommendations(
        self,
        customer: Optional[str] = None,
        platform: Optional[str] = None,
        date: Optional[str] = None,
    ) -> Path:
        """
        Get path to recommendations.md from creative scorer.

        Args:
            customer: Override customer
            platform: Override platform
            date: Override date

        Returns:
            Path to recommendations.md

        Example:
            paths.recommendations()
            # → config/ad/recommender/moprobo/taboola/2026-01-26/recommendations.md
        """
        cust = customer.lower().replace("-", "_") if customer else self.customer
        plat = platform.lower().replace("-", "_") if platform else self.platform
        d = date or self.date

        return (
            self.config_dir
            / "ad"
            / "recommender"
            / cust
            / plat
            / d
            / "recommendations.md"
        )

    def recommendations_dir(
        self,
        customer: Optional[str] = None,
        platform: Optional[str] = None,
        date: Optional[str] = None,
    ) -> Path:
        """
        Get directory containing recommendations.

        Returns:
            Path to recommendations directory
        """
        return self.recommendations(customer, platform, date).parent

    def prompts_output(
        self,
        prompt_type: str,
        customer: Optional[str] = None,
        platform: Optional[str] = None,
        date: Optional[str] = None,
    ) -> Path:
        """
        Get path for prompt output directory.

        Args:
            prompt_type: Type of prompt (structured, nano, variants)
            customer: Override customer
            platform: Override platform
            date: Override date

        Returns:
            Path to prompt output directory

        Example:
            paths.prompts_output("structured")
            # → config/ad/generator/prompts/moprobo/taboola/2026-01-21/structured/
        """
        cust = customer.lower().replace("-", "_") if customer else self.customer
        plat = platform.lower().replace("-", "_") if platform else self.platform
        d = date or self.date

        return self.config_dir / "ad" / "generator" / "prompts" / cust / plat / d / prompt_type

    def prompts_file(
        self,
        prompt_type: str,
        filename: str = "prompts.json",
        customer: Optional[str] = None,
        platform: Optional[str] = None,
        date: Optional[str] = None,
    ) -> Path:
        """
        Get path for prompt output file.

        Args:
            prompt_type: Type of prompt (structured, nano, variants)
            filename: Output filename (default: prompts.json)
            customer: Override customer
            platform: Override platform
            date: Override date

        Returns:
            Path to prompt output file
        """
        return (
            self.prompts_output(prompt_type, customer, platform, date)
            / filename
        )

    def generated_output(
        self,
        customer: Optional[str] = None,
        platform: Optional[str] = None,
        date: Optional[str] = None,
    ) -> Path:
        """
        Get path for generated images output directory.

        Args:
            customer: Override customer
            platform: Override platform
            date: Override date

        Returns:
            Path to generated images directory

        Example:
            paths.generated_output()
            # → results/ad/generator/generated/moprobo/taboola/2026-01-21/
        """
        cust = customer.lower().replace("-", "_") if customer else self.customer
        plat = platform.lower().replace("-", "_") if platform else self.platform
        d = date or self.date

        return self.output_dir / "ad" / "generator" / "generated" / cust / plat / d

    def config_file(self, config_name: str = "generation_config.yaml") -> Path:
        """
        Get path to config file.

        Args:
            config_name: Config filename

        Returns:
            Path to config file

        Example:
            paths.config_file()
            # → config/moprobo/taboola/generation_config.yaml
        """
        return self.config_dir / "ad" / "generator" / self.customer / self.platform / config_name

    def templates_dir(self) -> Path:
        """
        Get path to templates directory for customer/platform.

        Returns:
            Path to templates directory
        """
        return self.config_dir / "ad" / "generator" / "templates" / self.customer / self.platform

    def template_file(self, template_name: str) -> Path:
        """
        Get path to specific template file.

        Args:
            template_name: Template filename

        Returns:
            Path to template file
        """
        return self.templates_dir() / template_name

    def ensure_directories(self) -> list[Path]:
        """
        Ensure all necessary directories exist.

        Creates directories if they don't exist.

        Returns:
            List of created/verified directories
        """
        directories = [
            self.recommendations_dir(),
            self.prompts_output("structured"),
            self.prompts_output("nano"),
            self.prompts_output("variants"),
            self.generated_output(),
            self.templates_dir(),
        ]

        created = []
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
            logger.debug("Ensured directory exists: %s", directory)

        logger.info("Ensured %d directories exist", len(created))
        return created

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Paths(customer={self.customer}, platform={self.platform}, "
            f"date={self.date})"
        )


def detect_from_path(file_path: Path) -> dict[str, str]:
    """
    Detect customer, platform, date from a file path.

    Args:
        file_path: Path to parse

    Returns:
        Dict with customer, platform, date keys

    Example:
        path = Path("config/ad/recommender/recommendations/moprobo/taboola/2026-01-21/recommendations.json")
        info = detect_from_path(path)
        # → {"customer": "moprobo", "platform": "taboola", "date": "2026-01-21"}
    """
    parts = file_path.parts

    result = {}

    try:
        # Look for standard patterns
        if "recommendations" in parts:
            idx = parts.index("recommendations")
            if idx + 3 < len(parts):
                result["customer"] = parts[idx + 1]
                result["platform"] = parts[idx + 2]
                result["date"] = parts[idx + 3]
        elif "prompts" in parts:
            idx = parts.index("prompts")
            if idx + 3 < len(parts):
                result["customer"] = parts[idx + 1]
                result["platform"] = parts[idx + 2]
                result["date"] = parts[idx + 3]
        elif "generated" in parts:
            idx = parts.index("generated")
            if idx + 3 < len(parts):
                result["customer"] = parts[idx + 1]
                result["platform"] = parts[idx + 2]
                result["date"] = parts[idx + 3]
    except (IndexError, ValueError):
        logger.warning(
            "Could not detect customer/platform/date from path: %s", file_path
        )

    return result


def create_customer_platform_dirs(
    customer: str,
    platform: str,
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> dict[str, Path]:
    """
    Create standard directory structure for a customer/platform.

    Args:
        customer: Customer name
        platform: Platform name
        config_dir: Base config directory

    Returns:
        Dict with created directory paths

    Example:
        dirs = create_customer_platform_dirs("moprobo", "taboola")
        # Creates:
        # - config/ad/recommender/recommendations/moprobo/taboola/
        # - config/ad/generator/prompts/moprobo/taboola/
        # - config/ad/generator/generated/moprobo/taboola/
        # - config/ad/generator/moprobo/taboola/
    """
    customer_clean = customer.lower().replace("-", "_")
    platform_clean = platform.lower().replace("-", "_")

    directories = {
        "recommendations": config_dir
        / "ad"
        / "recommender"
        / "recommendations"
        / customer_clean
        / platform_clean,
        "prompts": config_dir / "ad" / "generator" / "prompts" / customer_clean / platform_clean,
        "generated": config_dir / "ad" / "generator" / "generated" / customer_clean / platform_clean,
        "config": config_dir / "ad" / "generator" / customer_clean / platform_clean,
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Created directory: %s", directory)

    return directories
