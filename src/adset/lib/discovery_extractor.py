"""Rule extraction and YAML generation.

This module converts discovered rules into formats compatible with
the existing decision rules system.
"""

from typing import Any, Dict, List, Optional

import yaml

from .discovery_models import DiscoveredRule


class RuleExtractor:
    """Extract and format rules for deployment.

    Converts discovered rules into formats compatible with the
    DecisionRules system and generates YAML config files.
    """

    def __init__(self, config_schema: Optional[Dict[str, Any]] = None):
        """Initialize rule extractor.

        Args:
            config_schema: Schema for target config format
        """
        self.config_schema = config_schema or {}

    def convert_to_decision_rule(
        self,
        discovered_rule: DiscoveredRule,
    ) -> Dict[str, Any]:
        """Convert DiscoveredRule to DecisionRules format.

        Args:
            discovered_rule: Rule to convert

        Returns:
            Dictionary compatible with DecisionRules structure

        Example:
            {
                "rule_name": "high_efficiency_rising_roas",
                "conditions": {
                    "purchase_roas_7d": {"min": 3.0},
                    "roas_trend": {"min": 0.05},
                    "efficiency": {"min": 0.10},
                },
                "action": {
                    "adjustment_factor": 1.15,
                    "reason": "high_roas_rising_healthy"
                }
            }
        """
        # Build rule name from conditions
        rule_name = self._generate_rule_name(discovered_rule)

        # Format conditions
        conditions = self._format_conditions(discovered_rule.conditions)

        # Determine action
        action = {
            "adjustment_factor": discovered_rule.adjustment_factor,
            "reason": self._generate_reason(discovered_rule),
        }

        # Add metadata
        rule_dict = {
            "rule_id": discovered_rule.rule_id,
            "rule_name": rule_name,
            "conditions": conditions,
            "action": action,
            "metadata": {
                "support": discovered_rule.support,
                "confidence": discovered_rule.confidence,
                "lift": discovered_rule.lift,
                "discovery_method": discovered_rule.discovery_method,
                "discovery_date": discovered_rule.discovery_date.isoformat(),
            },
        }

        return rule_dict

    def _generate_rule_name(self, rule: DiscoveredRule) -> str:
        """Generate human-readable rule name.

        Args:
            rule: Discovered rule

        Returns:
            Rule name string
        """
        # Extract key features
        key_features = list(rule.conditions.keys())[:3]

        # Generate name based on outcome and features
        if rule.outcome == "increase":
            base = "high_performer"
        else:
            base = "low_performer"

        if key_features:
            features_str = "_".join(key_features[:2])
            return f"{base}_{features_str}"
        else:
            return f"{base}_discovered_rule"

    def _format_conditions(
        self,
        conditions: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """Format conditions for DecisionRules.

        Args:
            conditions: Raw conditions from DiscoveredRule

        Returns:
            Formatted conditions dict

        Example:
            {
                "purchase_roas_7d": {"min": 3.0},
                "roas_trend": {"min": 0.05},
            }
        """
        formatted = {}

        for feature, condition in conditions.items():
            if isinstance(condition, dict):
                formatted[feature] = {}
                if "max" in condition:
                    formatted[feature]["max"] = condition["max"]
                if "min" in condition:
                    formatted[feature]["min"] = condition["min"]
            else:
                # Direct value comparison
                formatted[feature] = {"equals": condition}

        return formatted

    def _generate_reason(self, rule: DiscoveredRule) -> str:
        """Generate human-readable reason for rule.

        Args:
            rule: Discovered rule

        Returns:
            Reason string
        """
        if rule.outcome == "increase":
            return (
                f"Discovered high-performing pattern "
                f"(confidence: {rule.confidence:.2f}, "
                f"lift: {rule.lift:.2f})"
            )
        else:
            return (
                f"Discovered low-performing pattern "
                f"(confidence: {rule.confidence:.2f}, "
                f"lift: {rule.lift:.2f})"
            )

    def generate_yaml_config(
        self,
        rules: List[DiscoveredRule],
        output_path: str,
        config_type: str = "decision_rules",
    ) -> None:
        """Generate YAML config file from discovered rules.

        Args:
            rules: List of discovered rules
            output_path: Path to write YAML file
            config_type: Type of config (decision_rules, safety_rules, etc.)
        """
        # Convert rules to dict format
        rules_dict = {}

        for rule in rules:
            converted = self.convert_to_decision_rule(rule)
            rule_id = converted.pop("rule_id")
            rules_dict[rule_id] = converted

        # Create YAML structure
        if config_type == "decision_rules":
            yaml_content = {"discovered_decision_rules": rules_dict}
        elif config_type == "safety_rules":
            yaml_content = {"discovered_safety_rules": rules_dict}
        else:
            yaml_content = {f"discovered_{config_type}": rules_dict}

        # Write to file
        with open(output_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    def prioritize_rules(
        self,
        rules: List[DiscoveredRule],
    ) -> List[DiscoveredRule]:
        """Prioritize rules by tier.

        Assigns tiers similar to DecisionRules priority system:
        - Tier 2: Excellent performers (multi-factor confirmation)
        - Tier 3: High performers (strong signals)
        - Tier 4: Efficiency-based
        - Tier 5: Volume-based
        - Tier 15: Default/maintenance

        Args:
            rules: List of discovered rules

        Returns:
            Rules with assigned tiers
        """
        for rule in rules:
            # Determine tier based on characteristics
            if rule.outcome == "increase":
                if rule.confidence > 0.9 and rule.lift > 1.5:
                    rule.tier = 2  # Excellent
                elif rule.confidence > 0.8 and rule.lift > 1.3:
                    rule.tier = 3  # High
                elif rule.support > 100 and rule.confidence > 0.7:
                    rule.tier = 4  # Strong signal
                else:
                    rule.tier = 15  # Default
            else:  # decrease outcome
                if rule.confidence > 0.85:
                    rule.tier = 6  # Declining
                else:
                    rule.tier = 15  # Default

        # Sort by tier
        return sorted(rules, key=lambda r: r.tier if r.tier else 99)

    def merge_with_existing_config(
        self,
        new_rules: List[DiscoveredRule],
        existing_config_path: str,
        output_path: str,
    ) -> None:
        """Merge discovered rules with existing config.

        Args:
            new_rules: Newly discovered rules
            existing_config_path: Path to existing YAML config
            output_path: Path to write merged config
        """
        # Load existing config
        with open(existing_config_path, "r") as f:
            existing_config = yaml.safe_load(f) or {}

        # Convert new rules
        new_rules_dict = {}
        for rule in new_rules:
            converted = self.convert_to_decision_rule(rule)
            rule_id = converted.pop("rule_id")
            new_rules_dict[rule_id] = converted

        # Merge (discovered rules go in separate section)
        existing_config["discovered_decision_rules"] = new_rules_dict

        # Write merged config
        with open(output_path, "w") as f:
            yaml.dump(existing_config, f, default_flow_style=False, sort_keys=False)
