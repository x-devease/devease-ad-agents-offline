"""
Target transformation for handling skewed distributions.

Supports multiple transformations: log1p, sqrt, boxcox, yeo-johnson.
"""

import numpy as np
from scipy import stats, special
from sklearn.preprocessing import PowerTransformer
from typing import Tuple, Optional
import joblib
from pathlib import Path


class TargetTransformer:
    """
    Transform target variable to handle skewness.

    Supports multiple transformations with automatic selection.
    """

    def __init__(self, method: str = "auto"):
        """
        Initialize target transformer.

        Args:
            method: Transformation method ('log1p', 'sqrt', 'boxcox', 'yeo-johnson', 'auto')
        """
        self.method = method
        self.fitted_transformer = None
        self.lambda_ = None
        self.is_fitted = False

    def fit(self, y: np.ndarray) -> "TargetTransformer":
        """
        Fit transformer to target.

        Args:
            y: Target vector

        Returns:
            self
        """
        # Determine best method if auto
        if self.method == "auto":
            self.method = self._select_best_method(y)

        # Apply selected transformation
        if self.method == "log1p":
            # No fitting needed for log1p
            pass

        elif self.method == "sqrt":
            # No fitting needed for sqrt
            pass

        elif self.method == "boxcox":
            # Box-Cox requires positive values
            if (y <= 0).any():
                # Shift to make positive
                shift = abs(y.min()) + 1
                y_shifted = y + shift
            else:
                y_shifted = y
                shift = 0

            transformed, self.lambda_ = stats.boxcox(y_shifted)
            self.shift = shift

        elif self.method == "yeo-johnson":
            self.fitted_transformer = PowerTransformer(
                method="yeo-johnson", standardize=False
            )
            self.fitted_transformer.fit(y.reshape(-1, 1))

        self.is_fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform target.

        Args:
            y: Target vector

        Returns:
            Transformed target
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")

        if self.method == "log1p":
            return np.log1p(y)

        elif self.method == "sqrt":
            return np.sqrt(y)

        elif self.method == "boxcox":
            if hasattr(self, "shift"):
                y_shifted = y + self.shift
            else:
                y_shifted = y
            return stats.boxcox(y_shifted, lmbda=self.lambda_)

        elif self.method == "yeo-johnson":
            return self.fitted_transformer.transform(y.reshape(-1, 1)).flatten()

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse transform target.

        Args:
            y_transformed: Transformed target

        Returns:
            Original scale target
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before inverse transform")

        if self.method == "log1p":
            return np.expm1(y_transformed)

        elif self.method == "sqrt":
            return np.square(y_transformed)

        elif self.method == "boxcox":
            # pylint: disable=no-member
            y_original = special.inv_boxcox(y_transformed, self.lambda_)
            if hasattr(self, "shift"):
                return y_original - self.shift
            return y_original

        elif self.method == "yeo-johnson":
            return self.fitted_transformer.inverse_transform(
                y_transformed.reshape(-1, 1)
            ).flatten()

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _select_best_method(self, y: np.ndarray) -> str:
        """
        Select best transformation method based on normality test.

        Args:
            y: Target vector

        Returns:
            Best method name
        """
        methods = ["log1p", "sqrt", "yeo-johnson"]
        best_method = "yeo-johnson"
        best_pvalue = 0

        for method in methods:
            try:
                transformer = TargetTransformer(method=method)
                transformer.fit(y)
                y_transformed = transformer.transform(y)

                # Test for normality
                _, pvalue = stats.normaltest(y_transformed)

                if pvalue > best_pvalue:
                    best_pvalue = pvalue
                    best_method = method

            except Exception:
                continue

        return best_method

    def save(self, path: Path) -> None:
        """Save transformer."""
        joblib.dump(
            {
                "method": self.method,
                "fitted_transformer": self.fitted_transformer,
                "lambda_": self.lambda_,
                "is_fitted": self.is_fitted,
                "shift": getattr(self, "shift", None),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TargetTransformer":
        """Load transformer."""
        data = joblib.load(path)
        transformer = cls(method=data["method"])
        transformer.fitted_transformer = data["fitted_transformer"]
        transformer.lambda_ = data["lambda_"]
        transformer.is_fitted = data["is_fitted"]
        if data["shift"] is not None:
            transformer.shift = data["shift"]
        return transformer
