"""JSON I/O and Markdown generator for mined patterns."""

from pathlib import Path
from typing import Any, Dict
import json
import logging
from datetime import datetime

from ..validation import OutputSchemaValidator

logger = logging.getLogger(__name__)


class PatternsIO:
    """
    Handle reading/writing patterns JSON and generating Markdown.

    This class manages:
    - Saving patterns to JSON with validation
    - Loading patterns from JSON with validation
    - Generating human-readable Markdown from JSON
    """

    def __init__(self, schema_version: str = "2.0"):
        """
        Initialize PatternsIO.

        Args:
            schema_version: Expected schema version (default: "2.0")
        """
        self.schema_version = schema_version

    def save_patterns(
        self,
        patterns_data: Dict[str, Any],
        json_path: str | Path,
        validate: bool = True,
    ) -> bool:
        """
        Save patterns to JSON file with optional validation.

        Args:
            patterns_data: Dictionary with patterns data
            json_path: Path to save JSON file
            validate: Whether to validate before saving (default: True)

        Returns:
            True if successful, False otherwise
        """
        json_path = Path(json_path)

        # Validate if requested
        if validate:
            logger.info("Validating patterns data before saving...")
            # Save to temp file for validation
            temp_path = json_path.with_suffix('.tmp.json')

            try:
                with open(temp_path, 'w') as f:
                    json.dump(patterns_data, f, indent=2)

                validator = OutputSchemaValidator(temp_path)
                is_valid = validator.validate()

                if not is_valid:
                    logger.error("Validation failed, not saving patterns")
                    logger.error(f"Errors: {validator.errors}")
                    temp_path.unlink()
                    return False

                logger.info("Validation passed, saving patterns")

            except Exception as e:
                logger.error(f"Validation error: {e}")
                if temp_path.exists():
                    temp_path.unlink()
                return False

        # Ensure parent directory exists
        json_path.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON
        try:
            with open(json_path, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            logger.info(f"Saved patterns to {json_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")
            return False

    def load_patterns(
        self,
        json_path: str | Path,
        validate: bool = True,
    ) -> Dict[str, Any] | None:
        """
        Load patterns from JSON file with optional validation.

        Args:
            json_path: Path to JSON file
            validate: Whether to validate after loading (default: True)

        Returns:
            Dictionary with patterns data, or None if loading/validation fails
        """
        json_path = Path(json_path)

        if not json_path.exists():
            logger.error(f"JSON file not found: {json_path}")
            return None

        # Load JSON
        try:
            with open(json_path) as f:
                patterns_data = json.load(f)
            logger.info(f"Loaded patterns from {json_path}")
        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            return None

        # Validate if requested
        if validate:
            logger.info("Validating loaded patterns...")
            validator = OutputSchemaValidator(json_path)
            is_valid = validator.validate()

            if not is_valid:
                logger.warning("Validation produced warnings/errors")
                if validator.errors:
                    logger.error(f"Errors: {validator.errors}")
                if validator.warnings:
                    logger.warning(f"Warnings: {validator.warnings}")

            # Check schema version
            loaded_version = patterns_data.get('metadata', {}).get('schema_version')
            if loaded_version != self.schema_version:
                logger.warning(
                    f"Schema version mismatch: expected {self.schema_version}, "
                    f"got {loaded_version}"
                )

        return patterns_data

    def generate_markdown(
        self,
        patterns_data: Dict[str, Any],
        md_path: str | Path | None = None,
    ) -> str:
        """
        Generate human-readable Markdown from patterns JSON.

        Args:
            patterns_data: Dictionary with patterns data
            md_path: Optional path to save markdown file

        Returns:
            Markdown string
        """
        md_lines = []

        # Header
        metadata = patterns_data.get('metadata', {})
        customer = metadata.get('customer', 'Unknown')
        product = metadata.get('product', 'all')
        branch = metadata.get('branch', 'all')
        goal = metadata.get('campaign_goal', 'all')
        analysis_date = metadata.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))

        md_lines.append(f"# Mined Creative Patterns")
        md_lines.append("")
        md_lines.append(f"**Customer:** {customer}")
        md_lines.append(f"**Product:** {product}")
        md_lines.append(f"**Branch:** {branch}")
        md_lines.append(f"**Campaign Goal:** {goal}")
        md_lines.append(f"**Analysis Date:** {analysis_date}")
        md_lines.append("")

        # Data Quality Section
        data_quality = metadata.get('data_quality', {})
        if data_quality:
            md_lines.append("## Data Quality")
            md_lines.append("")
            md_lines.append(f"- **Sample Size:** {metadata.get('sample_size', 'N/A')}")
            md_lines.append(f"- **Completeness Score:** {data_quality.get('completeness_score', 'N/A')}")
            md_lines.append(f"- **Average ROAS:** {data_quality.get('avg_roas', 'N/A')}")
            md_lines.append(f"- **Top Quartile ROAS:** {data_quality.get('top_quartile_roas', 'N/A')}")
            md_lines.append(f"- **Bottom Quartile ROAS:** {data_quality.get('bottom_quartile_roas', 'N/A')}")
            md_lines.append("")

        # Positive Patterns Section
        patterns = patterns_data.get('patterns', [])
        if patterns:
            md_lines.append("## Positive Patterns (DOs)")
            md_lines.append("")
            md_lines.append(f"Found {len(patterns)} positive patterns ranked by priority:")
            md_lines.append("")

            for i, pattern in enumerate(patterns, 1):
                feature = pattern.get('feature', 'Unknown')
                value = pattern.get('value', 'Unknown')
                confidence = pattern.get('confidence', 'Unknown')
                roas_lift = pattern.get('roas_lift_multiple', 0)
                roas_lift_pct = pattern.get('roas_lift_pct', 0)
                top_prev = pattern.get('top_quartile_prevalence', 0)
                priority = pattern.get('priority_score', 0)
                reason = pattern.get('reason', '')

                md_lines.append(f"### {i}. {feature} = {value}")
                md_lines.append("")
                md_lines.append(f"- **Confidence:** {confidence}")
                md_lines.append(f"- **ROAS Lift:** {roas_lift}x ({roas_lift_pct:.1f}% increase)")
                md_lines.append(f"- **Top Quartile Prevalence:** {top_prev:.1%}")
                md_lines.append(f"- **Priority Score:** {priority:.1f}/10")
                md_lines.append(f"- **Reason:** {reason}")
                md_lines.append("")

        # Anti-Patterns Section
        anti_patterns = patterns_data.get('anti_patterns', [])
        if anti_patterns:
            md_lines.append("## Negative Patterns (DON'Ts)")
            md_lines.append("")
            md_lines.append(f"Found {len(anti_patterns)} negative patterns to avoid:")
            md_lines.append("")

            for i, pattern in enumerate(anti_patterns, 1):
                feature = pattern.get('feature', 'Unknown')
                avoid_value = pattern.get('avoid_value', 'Unknown')
                confidence = pattern.get('confidence', 'Unknown')
                roas_penalty = pattern.get('roas_penalty_multiple', 0)
                roas_penalty_pct = pattern.get('roas_penalty_pct', 0)
                bottom_prev = pattern.get('bottom_quartile_prevalence', 0)
                reason = pattern.get('reason', '')

                md_lines.append(f"### {i}. Avoid: {feature} = {avoid_value}")
                md_lines.append("")
                md_lines.append(f"- **Confidence:** {confidence}")
                md_lines.append(f"- **ROAS Penalty:** {roas_penalty}x ({roas_penalty_pct:.1f}% decrease)")
                md_lines.append(f"- **Bottom Quartile Prevalence:** {bottom_prev:.1%}")
                md_lines.append(f"- **Reason:** {reason}")
                md_lines.append("")

        # Low-Priority Insights Section
        low_priority = patterns_data.get('low_priority_insights', [])
        if low_priority:
            md_lines.append("## Low-Priority Insights")
            md_lines.append("")
            md_lines.append(f"Found {len(low_priority)} minor trends worth watching:")
            md_lines.append("")

            for i, insight in enumerate(low_priority, 1):
                feature = insight.get('feature', 'Unknown')
                value = insight.get('value', 'Unknown')
                roas_lift = insight.get('roas_lift_multiple', 0)
                reason = insight.get('reason', '')

                md_lines.append(f"### {i}. {feature} = {value}")
                md_lines.append("")
                md_lines.append(f"- **ROAS Lift:** {roas_lift}x")
                md_lines.append(f"- **Reason:** {reason}")
                md_lines.append("")

        # Generation Instructions Section
        gen_instructions = patterns_data.get('generation_instructions')
        if gen_instructions:
            md_lines.append("## Generation Instructions")
            md_lines.append("")

            must_include = gen_instructions.get('must_include')
            if must_include:
                md_lines.append(f"**Must Include:** {', '.join(must_include)}")
                md_lines.append("")

            prioritize = gen_instructions.get('prioritize')
            if prioritize:
                md_lines.append(f"**Prioritize:** {', '.join(prioritize)}")
                md_lines.append("")

            avoid = gen_instructions.get('avoid')
            if avoid:
                md_lines.append(f"**Avoid:** {', '.join(avoid)}")
                md_lines.append("")

        # Footer
        md_lines.append("---")
        md_lines.append("")
        md_lines.append(f"*Generated by Ad Miner v{self.schema_version}*")
        md_lines.append(f"*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        # Join lines
        markdown = '\n'.join(md_lines)

        # Save to file if path provided
        if md_path:
            md_path = Path(md_path)
            md_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with open(md_path, 'w') as f:
                    f.write(markdown)
                logger.info(f"Saved markdown to {md_path}")
            except Exception as e:
                logger.error(f"Failed to save markdown: {e}")

        return markdown

    def generate_and_save_markdown(
        self,
        patterns_data: Dict[str, Any],
        md_path: str | Path,
    ) -> bool:
        """
        Generate and save markdown in one step.

        Args:
            patterns_data: Dictionary with patterns data
            md_path: Path to save markdown file

        Returns:
            True if successful, False otherwise
        """
        try:
            self.generate_markdown(patterns_data, md_path)
            return True
        except Exception as e:
            logger.error(f"Failed to generate markdown: {e}")
            return False


def load_patterns_json(
    json_path: str | Path,
    validate: bool = True,
) -> Dict[str, Any] | None:
    """
    Convenience function to load patterns JSON.

    Args:
        json_path: Path to JSON file
        validate: Whether to validate

    Returns:
        Dictionary with patterns data or None
    """
    io = PatternsIO()
    return io.load_patterns(json_path, validate=validate)


def save_patterns_json(
    patterns_data: Dict[str, Any],
    json_path: str | Path,
    validate: bool = True,
) -> bool:
    """
    Convenience function to save patterns JSON.

    Args:
        patterns_data: Dictionary with patterns data
        json_path: Path to save JSON file
        validate: Whether to validate before saving

    Returns:
        True if successful, False otherwise
    """
    io = PatternsIO()
    return io.save_patterns(patterns_data, json_path, validate=validate)


def generate_patterns_markdown(
    patterns_data: Dict[str, Any],
    md_path: str | Path | None = None,
) -> str:
    """
    Convenience function to generate markdown.

    Args:
        patterns_data: Dictionary with patterns data
        md_path: Optional path to save markdown

    Returns:
        Markdown string
    """
    io = PatternsIO()
    return io.generate_markdown(patterns_data, md_path)
