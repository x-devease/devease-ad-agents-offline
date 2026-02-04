"""
Detector Factory for creating detector instances.

This module provides a factory pattern for creating detector instances,
making it easy to instantiate detectors by name and register new detectors.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Type, Optional

from src.meta.diagnoser.core.issue_detector import BaseDetector
from src.meta.diagnoser.detectors import (
    FatigueDetector,
    LatencyDetector,
    DarkHoursDetector,
    PerformanceDetector,
    ConfigurationDetector,
)

logger = logging.getLogger(__name__)


class DetectorFactory:
    """
    Factory for creating detector instances.

    Provides centralized detector creation with configuration support.
    Maintains a registry of available detector classes.

    Usage:
        factory = DetectorFactory()

        # Create single detector
        detector = factory.create("FatigueDetector")

        # Create with custom config
        detector = factory.create(
            "FatigueDetector",
            config={"thresholds": {"fatigue_freq_threshold": 2.5}}
        )

        # Create all detectors
        all_detectors = factory.create_all()
    """

    # Registry of available detector classes
    _detector_classes: Dict[str, Type[BaseDetector]] = {
        "FatigueDetector": FatigueDetector,
        "LatencyDetector": LatencyDetector,
        "DarkHoursDetector": DarkHoursDetector,
        "PerformanceDetector": PerformanceDetector,
        "ConfigurationDetector": ConfigurationDetector,
    }

    @classmethod
    def create(
        cls,
        detector_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseDetector:
        """
        Create a detector instance.

        Args:
            detector_type: Name of the detector class (e.g., "FatigueDetector")
            config: Optional configuration dictionary for the detector

        Returns:
            Configured detector instance

        Raises:
            ValueError: If detector_type is not registered

        Example:
            >>> factory = DetectorFactory()
            >>> detector = factory.create("FatigueDetector")
            >>> type(detector).__name__
            'FatigueDetector'
        """
        if detector_type not in cls._detector_classes:
            available = list(cls._detector_classes.keys())
            raise ValueError(
                f"Unknown detector type: {detector_type}. "
                f"Available detectors: {available}"
            )

        detector_class = cls._detector_classes[detector_type]
        detector = detector_class(config=config)

        logger.debug(f"Created {detector_type} instance")
        return detector

    @classmethod
    def create_all(
        cls,
        configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, BaseDetector]:
        """
        Create all available detector instances.

        Args:
            configs: Optional dict mapping detector names to their configs
                    Example: {"FatigueDetector": {"thresholds": {...}}}

        Returns:
            Dict mapping detector names to detector instances

        Example:
            >>> factory = DetectorFactory()
            >>> detectors = factory.create_all()
            >>> list(detectors.keys())
            ['FatigueDetector', 'LatencyDetector', 'DarkHoursDetector', ...]
        """
        configs = configs or {}
        detectors = {}

        for detector_name in cls._detector_classes.keys():
            detector_config = configs.get(detector_name)
            detectors[detector_name] = cls.create(detector_name, config=detector_config)

        logger.debug(f"Created {len(detectors)} detector instances")
        return detectors

    @classmethod
    def register_detector(
        cls,
        name: str,
        detector_class: Type[BaseDetector]
    ) -> None:
        """
        Register a new detector type.

        Allows adding custom detectors to the factory registry.

        Args:
            name: Name for the detector (e.g., "CustomDetector")
            detector_class: Detector class to register

        Raises:
            TypeError: If detector_class is not a subclass of BaseDetector

        Example:
            >>> class CustomDetector(BaseDetector):
            ...     def detect(self, data, entity_id):
            ...         return []
            >>> DetectorFactory.register_detector("CustomDetector", CustomDetector)
        """
        if not issubclass(detector_class, BaseDetector):
            raise TypeError(
                f"detector_class must be a subclass of BaseDetector, "
                f"got {type(detector_class)}"
            )

        cls._detector_classes[name] = detector_class
        logger.info(f"Registered detector: {name}")

    @classmethod
    def list_detectors(cls) -> list[str]:
        """
        List all registered detector types.

        Returns:
            List of detector names

        Example:
            >>> DetectorFactory.list_detectors()
            ['FatigueDetector', 'LatencyDetector', 'DarkHoursDetector', ...]
        """
        return list(cls._detector_classes.keys())

    @classmethod
    def is_registered(cls, detector_type: str) -> bool:
        """
        Check if a detector type is registered.

        Args:
            detector_type: Name of the detector to check

        Returns:
            True if detector is registered, False otherwise

        Example:
            >>> DetectorFactory.is_registered("FatigueDetector")
            True
            >>> DetectorFactory.is_registered("UnknownDetector")
            False
        """
        return detector_type in cls._detector_classes
