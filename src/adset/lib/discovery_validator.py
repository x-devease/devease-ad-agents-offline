"""Rule validation for discovered patterns.

This module validates discovered rules using forward-looking ROAS,
cross-validation, and safety constraint checking.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from .discovery_models import DiscoveredRule, ValidationResult


class RuleValidator:
    """Validate discovered rules using multiple methods.

    Ensures discovered patterns are safe, effective, and generalize
    to new data before deployment.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        safety_rules: Optional[dict] = None,
    ):
        """Initialize rule validator.

        Args:
            df_train: Training data for rule discovery
            df_test: Hold-out test data for validation
            safety_rules: Dictionary of safety rule parameters
        """
        self.df_train = df_train
        self.df_test = df_test
        self.safety_rules = safety_rules or {
            "max_daily_increase_pct": 0.19,
            "max_daily_decrease_pct": 0.10,
            "min_budget": 1.0,
        }

    def validate_rule(
        self,
        rule: DiscoveredRule,
        roas_col: str = "purchase_roas_7d",
        method: str = "forward_looking",
    ) -> ValidationResult:
        """Validate rule using forward-looking ROAS.

        Args:
            rule: Discovered rule to validate
            roas_col: Column name for ROAS metric
            method: Validation method ("forward_looking", "cross_validation")

        Returns:
            ValidationResult with metrics and recommendation
        """
        # Apply rule to test data
        matched_rows = self._apply_rule(self.df_test, rule)

        if len(matched_rows) == 0:
            # Rule doesn't match any test data
            return ValidationResult(
                rule_id=rule.rule_id,
                mean_roas=0.0,
                std_roas=0.0,
                budget_utilization=0.0,
                safety_violations=0,
                recommendation="reject",
                validation_samples=0,
                metadata={"reason": "No matching samples in test data"},
            )

        # Calculate ROAS for matched rows
        roas_values = matched_rows[roas_col].values
        mean_roas = float(np.mean(roas_values))
        std_roas = float(np.std(roas_values))

        # Calculate budget utilization (proportion of adsets matching rule)
        budget_utilization = len(matched_rows) / len(self.df_test)

        # Check safety violations
        safety_violations = self._check_safety_compatibility(rule, matched_rows)

        # Calculate forward-looking ROAS (next period performance)
        forward_roas = self._calculate_forward_looking_roas(matched_rows, roas_col)

        # Calculate backtest ROAS (historical performance)
        backtest_roas = self._calculate_backtest_roas(self.df_train, rule, roas_col)

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(roas_values)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            rule,
            mean_roas,
            budget_utilization,
            safety_violations,
            len(matched_rows),
        )

        return ValidationResult(
            rule_id=rule.rule_id,
            mean_roas=mean_roas,
            std_roas=std_roas,
            budget_utilization=budget_utilization,
            safety_violations=safety_violations,
            recommendation=recommendation,
            validation_samples=len(matched_rows),
            forward_roas=forward_roas,
            backtest_roas=backtest_roas,
            confidence_interval=confidence_interval,
            metadata={
                "method": method,
                "matched_rows": len(matched_rows),
                "total_rows": len(self.df_test),
            },
        )

    def _apply_rule(
        self,
        df: pd.DataFrame,
        rule: DiscoveredRule,
    ) -> pd.DataFrame:
        """Apply rule conditions to dataframe.

        Args:
            df: Dataframe to filter
            rule: Rule with conditions

        Returns:
            Filtered dataframe matching rule conditions
        """
        mask = pd.Series([True] * len(df), index=df.index)

        for feature, condition in rule.conditions.items():
            if feature not in df.columns:
                continue

            if isinstance(condition, dict):
                if "max" in condition and "min" in condition:
                    mask &= (df[feature] >= condition["min"]) & (
                        df[feature] <= condition["max"]
                    )
                elif "max" in condition:
                    mask &= df[feature] <= condition["max"]
                elif "min" in condition:
                    mask &= df[feature] >= condition["min"]
            elif isinstance(condition, (int, float)):
                mask &= df[feature] == condition

        return df[mask]

    def _check_safety_compatibility(
        self,
        rule: DiscoveredRule,
        matched_rows: pd.DataFrame,
    ) -> int:
        """Check if rule violates safety constraints.

        Args:
            rule: Rule to check
            matched_rows: Dataframe of matched adsets

        Returns:
            Number of safety violations
        """
        violations = 0

        # Check if adjustment exceeds safety limits
        if rule.adjustment_factor > 1.0:
            increase_pct = rule.adjustment_factor - 1.0
            if increase_pct > self.safety_rules["max_daily_increase_pct"]:
                violations += 1
        elif rule.adjustment_factor < 1.0:
            decrease_pct = 1.0 - rule.adjustment_factor
            if decrease_pct > self.safety_rules["max_daily_decrease_pct"]:
                violations += 1

        # Check for minimum budget violations
        if "spend" in matched_rows.columns:
            if (
                matched_rows["spend"] * rule.adjustment_factor
                < self.safety_rules["min_budget"]
            ).any():
                violations += 1

        return violations

    def _calculate_forward_looking_roas(
        self,
        matched_rows: pd.DataFrame,
        roas_col: str,
    ) -> Optional[float]:
        """Calculate forward-looking ROAS (next period).

        For now, returns current ROAS as placeholder.
        In production, would fetch next period data.

        Args:
            matched_rows: Matched dataframe
            roas_col: ROAS column name

        Returns:
            Forward ROAS value
        """
        if roas_col in matched_rows.columns:
            return float(matched_rows[roas_col].mean())
        return None

    def _calculate_backtest_roas(
        self,
        df_train: pd.DataFrame,
        rule: DiscoveredRule,
        roas_col: str,
    ) -> Optional[float]:
        """Calculate historical backtest ROAS.

        Args:
            df_train: Training dataframe
            rule: Rule to backtest
            roas_col: ROAS column name

        Returns:
            Backtest ROAS value
        """
        matched_rows = self._apply_rule(df_train, rule)

        if len(matched_rows) == 0 or roas_col not in matched_rows.columns:
            return None

        return float(matched_rows[roas_col].mean())

    def _calculate_confidence_interval(
        self,
        values: np.ndarray,
        confidence: float = 0.95,
    ) -> Optional[tuple[float, float]]:
        """Calculate confidence interval for ROAS.

        Args:
            values: ROAS values
            confidence: Confidence level (default 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(values) < 2:
            return None

        mean = np.mean(values)
        std_err = np.std(values) / np.sqrt(len(values))

        import scipy.stats as stats

        t_value = stats.t.ppf((1 + confidence) / 2, len(values) - 1)

        margin = t_value * std_err
        return (float(mean - margin), float(mean + margin))

    def _generate_recommendation(
        self,
        rule: DiscoveredRule,
        mean_roas: float,
        budget_utilization: float,
        safety_violations: int,
        validation_samples: int,
    ) -> str:
        """Generate deployment recommendation.

        Args:
            rule: Rule being validated
            mean_roas: Mean ROAS from validation
            budget_utilization: Budget utilization rate
            safety_violations: Number of safety violations
            validation_samples: Number of validation samples

        Returns:
            Recommendation ("deploy", "test", "reject")
        """
        # Reject if safety violations
        if safety_violations > 0:
            return "reject"

        # Reject if too few samples
        if validation_samples < 50:
            return "reject"

        # Deploy if high ROAS and good utilization
        if mean_roas > 2.5 and budget_utilization > 0.05:
            return "deploy"

        # Test if moderate performance
        if mean_roas > 1.5 and budget_utilization > 0.03:
            return "test"

        # Reject otherwise
        return "reject"

    def detect_overfitting(
        self,
        rule: DiscoveredRule,
        roas_col: str = "purchase_roas_7d",
    ) -> bool:
        """Detect if rule is overfit.

        Checks:
        - Low support (< 50 samples)
        - High train-test gap
        - Too specific conditions

        Args:
            rule: Rule to check
            roas_col: ROAS column name

        Returns:
            True if rule appears overfit
        """
        # Check support
        if rule.support < 50:
            return True

        # Check train-test performance gap
        train_matched = self._apply_rule(self.df_train, rule)
        test_matched = self._apply_rule(self.df_test, rule)

        if len(train_matched) == 0 or len(test_matched) == 0:
            return True

        train_roas = (
            train_matched[roas_col].mean() if roas_col in train_matched.columns else 0
        )
        test_roas = (
            test_matched[roas_col].mean() if roas_col in test_matched.columns else 0
        )

        # If test ROAS is much lower than train ROAS, likely overfit
        if train_roas > 0 and test_roas / train_roas < 0.7:
            return True

        return False

    def rank_rules_by_validation_score(
        self,
        rules: List[DiscoveredRule],
        roas_col: str = "purchase_roas_7d",
    ) -> List[tuple[float, DiscoveredRule]]:
        """Rank rules by validation performance.

        Args:
            rules: List of rules to validate
            roas_col: ROAS column name

        Returns:
            List of (score, rule) tuples sorted by score (descending)
        """
        scored_rules = []

        for rule in rules:
            validation = self.validate_rule(rule, roas_col)

            # Calculate composite score
            if validation.recommendation == "reject":
                score = 0.0
            else:
                # Weight ROAS and utilization
                score = (
                    validation.mean_roas * 0.7
                    + validation.budget_utilization * 100 * 0.3
                )

            scored_rules.append((score, rule))

        # Sort by score (descending)
        scored_rules.sort(key=lambda x: x[0], reverse=True)

        return scored_rules
