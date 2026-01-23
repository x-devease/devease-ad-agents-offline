"""Prediction module for making predictions on new creatives.

This module provides functionality to load trained models and make predictions
on new creative image features.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    from catboost import CatBoost

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    CatBoost = None  # type: ignore


logger = logging.getLogger(__name__)


class CreativePredictor:
    """Predictor for creative performance using trained models.

    This class loads a trained model and makes predictions on new creative
    features.

    Attributes:
        model: The trained model (CatBoost, Random Forest, or MLP)
        feature_columns: List of feature columns the model expects
        label_encoders: Dictionary of label encoders for categorical features
        model_type: Type of model ('catboost', 'random_forest', 'mlp')
        is_classification: Whether the model is a classification model
    """

    def __init__(
        self,
        model_path: str,
        metadata_path: Optional[str] = None,
    ):
        """Initialize the predictor with a trained model.

        Args:
            model_path: Path to the saved model file (.pkl or .cbm)
            metadata_path: Optional path to model metadata JSON file
        """
        self.model = None
        self.feature_columns = []
        self.label_encoders = {}
        self.model_type = None
        self.is_classification = False

        self._load_model(model_path, metadata_path)

    def _load_model(self, model_path: str, metadata_path: Optional[str] = None):
        """Load model and metadata from disk.

        Args:
            model_path: Path to model file
            metadata_path: Optional path to metadata file

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model format is unsupported
        """
        model_path_obj = Path(model_path)

        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load metadata if provided
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.feature_columns = metadata.get("feature_columns", [])
            self.label_encoders = dict(
                metadata.get("label_encoders", {}).items()
            )
            self.is_classification = metadata.get("is_classification", False)
            self.model_type = metadata.get("model_type", "unknown")
            logger.info("Loaded metadata for %s model", self.model_type)
        else:
            logger.warning(
                "No metadata provided, attempting to infer from model"
            )

        # Load model based on file extension
        suffix = model_path_obj.suffix.lower()

        if suffix == ".pkl":
            with open(model_path_obj, "rb") as f:
                self.model = pickle.load(f)
            self.model_type = "unknown"
            logger.info("Loaded model from %s", model_path)
        elif suffix == ".cbm":
            if not HAS_CATBOOST:
                raise ImportError("CatBoost is not installed")
            self.model = CatBoost()
            self.model.load_model(model_path_obj)
            self.model_type = "catboost"
            logger.info("Loaded CatBoost model from %s", model_path)
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

    def predict(
        self,
        features: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> np.ndarray:
        """Make predictions on new features.

        Args:
            features: Input features can be:
                - pandas DataFrame
                - Single dictionary of feature values
                - List of dictionaries for multiple samples

        Returns:
            numpy array of predictions

        Raises:
            ValueError: If features format is invalid
        """
        # Convert to DataFrame if needed
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, list):
            df = pd.DataFrame(features)
        elif isinstance(features, pd.DataFrame):
            df = features.copy()
        else:
            raise ValueError(
                "features must be a DataFrame, dict, or list of dicts"
            )

        # Encode categorical features using saved encoders
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                # Handle unseen categories by mapping to 'unknown'
                df[col] = df[col].apply(
                    lambda x, enc=encoder: x if x in enc.classes_ else "unknown"
                )
                # Extend encoder for 'unknown' if not present
                if "unknown" not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, "unknown")
                df[col] = encoder.transform(df[col].astype(str))

        # Fill missing values
        df = df.fillna(0)

        # Ensure feature columns match
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(df.columns)
            if missing_cols:
                logger.warning(
                    "Missing columns: %s. Filling with 0.", missing_cols
                )
                for col in missing_cols:
                    df[col] = 0

            # Reorder columns to match training
            df = df[self.feature_columns]

        # Make prediction
        if self.model_type == "catboost":
            predictions = self.model.predict(df)
        else:
            predictions = self.model.predict(df.values)

        return predictions

    def predict_proba(
        self,
        features: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> np.ndarray:
        """Get prediction probabilities (classification only).

        Args:
            features: Input features

        Returns:
            numpy array of prediction probabilities

        Raises:
            ValueError: If model is not a classification model
        """
        if not self.is_classification:
            raise ValueError(
                "predict_proba only available for classification models"
            )

        # Convert to DataFrame
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, list):
            df = pd.DataFrame(features)
        else:
            df = features.copy()

        # Encode and fill
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x, enc=encoder: x if x in enc.classes_ else "unknown"
                )
                if "unknown" not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, "unknown")
                df[col] = encoder.transform(df[col].astype(str))

        df = df.fillna(0)

        if self.feature_columns:
            df = df[self.feature_columns]

        # Get probabilities
        if self.model_type == "catboost":
            proba = self.model.predict_proba(df)
        else:
            proba = self.model.predict_proba(df.values)

        return proba


def load_predictor(
    model_path: str, metadata_path: Optional[str] = None
) -> CreativePredictor:
    """Convenience function to load a predictor.

    Args:
        model_path: Path to model file
        metadata_path: Optional path to metadata file

    Returns:
        CreativePredictor instance
    """
    return CreativePredictor(model_path, metadata_path)
