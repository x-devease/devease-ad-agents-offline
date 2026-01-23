"""Decision tree-based rule mining for pattern discovery.

This module uses decision trees to automatically extract human-readable
decision rules from data.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

from .discovery_models import DiscoveredRule


class DecisionTreeMiner:
    """Mine decision rules from decision trees.

    This class trains decision trees on historical data and extracts
    human-readable rules that can be used for budget allocation.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: int = 20,
        criterion: str = "entropy",
        min_samples_split: int = 40,
        random_state: int = 42,
    ):
        """Initialize the decision tree miner.

        Args:
            max_depth: Maximum depth of the decision tree
            min_samples_leaf: Minimum samples required at leaf node
            criterion: Splitting criterion ("entropy" or "gini")
            min_samples_split: Minimum samples required to split node
            random_state: Random seed for reproducibility
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.tree: Optional[DecisionTreeClassifier] = None
        self.feature_names: Optional[List[str]] = None

    def mine_rules(
        self,
        df_features: pd.DataFrame,
        target_col: str = "purchase_roas_7d",
        target_threshold: float = 2.0,
        positive_outcome: str = "increase",
        negative_outcome: str = "decrease",
    ) -> List[DiscoveredRule]:
        """Train decision tree and extract rules.

        Args:
            df_features: Feature dataframe with target column
            target_col: Name of target column (ROAS metric)
            target_threshold: Threshold for binary classification
            positive_outcome: Outcome label for high ROAS
            negative_outcome: Outcome label for low ROAS

        Returns:
            List of discovered rules sorted by quality
        """
        # Prepare data
        X, y, feature_names = self._prepare_data(
            df_features, target_col, target_threshold
        )

        # Train decision tree
        self.tree = self._train_tree(X, y)
        self.feature_names = feature_names

        # Extract rules from tree
        rules = self.extract_rules_from_tree(
            self.tree,
            feature_names,
            positive_outcome,
            negative_outcome,
            target_threshold,
        )

        # Rank rules by quality
        ranked_rules = self.rank_rules_by_quality(rules)

        return ranked_rules

    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for decision tree training.

        Args:
            df: Feature dataframe
            target_col: Target column name
            threshold: Threshold for binary classification

        Returns:
            Tuple of (X, y, feature_names)
        """
        # Separate features and target (only numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col]

        if not feature_cols:
            raise ValueError(
                f"No numeric feature columns found. "
                f"Target column: {target_col}, "
                f"Available numeric columns: {numeric_cols}"
            )

        X = df[feature_cols].values
        y = (df[target_col] >= threshold).astype(int)

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        return X, y, feature_cols

    def _train_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> DecisionTreeClassifier:
        """Train decision tree classifier.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Trained decision tree
        """
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            criterion=self.criterion,
            random_state=self.random_state,
        )

        tree.fit(X, y)
        return tree

    def extract_rules_from_tree(
        self,
        tree: DecisionTreeClassifier,
        feature_names: List[str],
        positive_outcome: str = "increase",
        negative_outcome: str = "decrease",
        target_threshold: float = 2.0,
    ) -> List[DiscoveredRule]:
        """Extract human-readable rules from tree structure.

        Args:
            tree: Trained decision tree
            feature_names: Names of features
            positive_outcome: Label for positive class
            negative_outcome: Label for negative class
            target_threshold: ROAS threshold used for classification

        Returns:
            List of discovered rules
        """
        rules = []

        # Traverse tree recursively
        def traverse(node_id: int, current_conditions: Dict[str, Any]):
            """Recursively traverse tree to extract rules."""

            # Check if leaf node
            if tree.tree_.feature[node_id] != _tree.TREE_UNDEFINED:
                # Internal node - continue traversal
                feature_idx = tree.tree_.feature[node_id]
                feature_name = feature_names[feature_idx]
                threshold = tree.tree_.threshold[node_id]

                # Left child (feature <= threshold)
                left_conditions = current_conditions.copy()
                left_conditions[feature_name] = {
                    "max": threshold,
                    "operator": "<=",
                }
                traverse(tree.tree_.children_left[node_id], left_conditions)

                # Right child (feature > threshold)
                right_conditions = current_conditions.copy()
                right_conditions[feature_name] = {
                    "min": threshold,
                    "operator": ">",
                }
                traverse(tree.tree_.children_right[node_id], right_conditions)
            else:
                # Leaf node - create rule
                value = tree.tree_.value[node_id]
                class_samples = value[0]
                total_samples = class_samples.sum()

                if total_samples < self.min_samples_leaf:
                    return

                # Determine class and confidence
                predicted_class = np.argmax(class_samples)
                class_count = class_samples[predicted_class]
                confidence = class_count / total_samples

                # Skip low confidence rules
                if confidence < 0.7:
                    return

                # Calculate lift (improvement over baseline)
                baseline_distribution = tree.tree_.value[0][0]
                baseline_prob = baseline_distribution[1] / baseline_distribution.sum()
                lift = confidence / (baseline_prob + 1e-10)

                # Determine outcome and adjustment
                if predicted_class == 1:
                    outcome = positive_outcome
                    # Higher confidence = higher adjustment
                    adjustment_factor = 1.0 + (confidence - 0.7) * 0.5  # 1.0 to 1.15
                else:
                    outcome = negative_outcome
                    adjustment_factor = 1.0 - (confidence - 0.7) * 0.3  # 1.0 to 0.85

                # Ensure adjustment factor is reasonable
                adjustment_factor = np.clip(adjustment_factor, 0.85, 1.15)

                # Create rule
                rule = DiscoveredRule(
                    rule_id=f"dt_rule_{len(rules)}",
                    conditions=current_conditions,
                    outcome=outcome,
                    adjustment_factor=adjustment_factor,
                    support=int(total_samples),
                    confidence=float(confidence),
                    lift=float(lift),
                    discovery_method="decision_tree",
                    metadata={
                        "node_id": node_id,
                        "class_distribution": class_samples.tolist(),
                        "predicted_class": int(predicted_class),
                    },
                )

                rules.append(rule)

        # Start traversal from root
        traverse(0, {})

        return rules

    def rank_rules_by_quality(
        self,
        rules: List[DiscoveredRule],
        metric: str = "support_confidence",
    ) -> List[DiscoveredRule]:
        """Rank rules by quality metrics.

        Args:
            rules: List of discovered rules
            metric: Ranking metric
                - "support_confidence": support * confidence
                - "lift": lift value
                - "confidence": confidence only
                - "support": support only

        Returns:
            Sorted list of rules (highest quality first)
        """
        if metric == "support_confidence":
            scores = [r.support * r.confidence for r in rules]
        elif metric == "lift":
            scores = [r.lift for r in rules]
        elif metric == "confidence":
            scores = [r.confidence for r in rules]
        elif metric == "support":
            scores = [r.support for r in rules]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Sort by score (descending)
        sorted_rules = [r for _, r in sorted(zip(scores, rules), key=lambda x: -x[0])]

        return sorted_rules

    def get_feature_importance(
        self,
    ) -> Optional[Dict[str, float]]:
        """Get feature importance from trained tree.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.tree is None or self.feature_names is None:
            return None

        importances = self.tree.feature_importances_

        return {name: float(imp) for name, imp in zip(self.feature_names, importances)}
