"""Model persistence utilities for saving and loading trained models."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict


def save_models(
    model_results: Dict[str, Any], output_path: str, model_files: Dict[str, str]
) -> Dict[str, str]:
    """Persist trained models to disk.

    Args:
        model_results: Model training results dict. Expected keys include
            model types (e.g., "regression", "classification", "mlp", etc.).
        output_path: Output directory for models. Models will be stored under
            `<output_path>/models/`.
        model_files: Mapping of model type to filename.

    Returns:
        Mapping of saved model types to file paths.
    """
    models_dir = Path(output_path) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    for model_type, filename in model_files.items():
        if model_type in model_results:
            model_path = models_dir / filename
            with open(model_path, "wb") as f:
                pickle.dump(model_results[model_type], f)
            saved_paths[model_type] = str(model_path)
            logging.info("Saved %s model to: %s", model_type, model_path)

    return saved_paths


def load_model(models_dir: str, filename: str) -> Any:
    """Load a persisted model from disk.

    Args:
        models_dir: Directory containing saved models.
        filename: Name of the model file to load.

    Returns:
        The loaded model object if the file exists; otherwise None.
    """
    model_path = Path(models_dir) / filename

    if not model_path.exists():
        logging.warning("Model file not found: %s", model_path)
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logging.info("Loaded model from: %s", model_path)
    return model
