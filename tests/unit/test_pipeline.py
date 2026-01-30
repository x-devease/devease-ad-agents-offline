"""
Unit tests for VisualQAMatrix pipeline.

Tests the main orchestrator and integration with generator sessions.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock

from src.meta.ad.reviewer.pipeline import VisualQAMatrix
from src.meta.ad.reviewer.schemas.audit_report import GuardStatus


class TestVisualQAMatrix:
    """Test suite for VisualQAMatrix pipeline."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a mock config file."""
        config_data = {
            'qa_risk_matrix': {
                'enabled': True,
                'geometry': {
                    'enabled': True,
                    'tolerance': 0.02,
                    'min_features': 10,
                    'fallback_to_contour': True,
                    'contour_tolerance': 0.05
                },
                'aesthetics': {
                    'enabled': False,  # Disable for tests
                    'min_score': 7.0,
                    'model': 'gpt-4o-mini'
                },
                'cultural': {
                    'enabled': False,  # Disable for tests
                    'target_region': 'USA_Europe',
                    'risk_threshold': 'HIGH',
                    'custom_rules': []
                },
                'performance': {
                    'enabled': False,  # Disable for tests
                    'psychology_weight': 0.40,
                    'saliency_weight': 0.30,
                    'consistency_weight': 0.30
                },
                'pipeline': {
                    'stop_on_first_fail': True,
                    'log_all_checks': True,
                    'save_reports': True,
                    'report_dir': 'results/ad/reviewer/'
                }
            }
        }

        config_file = tmp_path / "config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        return config_file

    @pytest.fixture
    def mock_vlm_client(self):
        """Create a mock VLM client."""
        client = MagicMock()
        client.check_aesthetics.return_value = {
            'score': 8.0,
            'issues': [],
            'has_negative_feature': False,
            'detected_features': [],
            'reasoning': 'Good quality'
        }
        client.check_culture.return_value = {
            'risk_level': 'LOW',
            'detected_issues': [],
            'confidence': 0.95,
            'reasoning': 'No issues'
        }
        client.score_performance.return_value = {
            'psychology_alignment': 75,
            'saliency_clarity': 80,
            'consistency_realism': 70,
            'overall_score': 75,
            'reasoning': 'Good alignment',
            'suggestions': []
        }
        return client

    def test_initialization(self, mock_config):
        """Test pipeline initialization."""
        pipeline = VisualQAMatrix(config_path=mock_config)

        assert pipeline.config_path == mock_config
        assert pipeline.geometric_guard is not None
        assert pipeline.aesthetic_guard is not None
        assert pipeline.cultural_guard is not None
        assert pipeline.performance_guard is not None

    def test_initialization_with_vlm_client(self, mock_config, mock_vlm_client):
        """Test pipeline initialization with custom VLM client."""
        pipeline = VisualQAMatrix(
            config_path=mock_config,
            vlm_client=mock_vlm_client
        )

        assert pipeline.vlm_client == mock_vlm_client

    def test_audit_creates_report(self, mock_config, tmp_path):
        """Test that audit creates a report with expected structure."""
        # Create test images
        import numpy as np
        import cv2

        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        product_path = tmp_path / "product.jpg"
        candidate_path = tmp_path / "candidate.jpg"
        cv2.imwrite(str(product_path), test_image)
        cv2.imwrite(str(candidate_path), test_image)

        # Run audit (only geometric guard enabled)
        pipeline = VisualQAMatrix(config_path=mock_config)
        blueprint = {
            'id': 'test_blueprint',
            'strategy_rationale': {
                'psychology_driver': 'trust'
            },
            'nano_generation_rules': {
                'negative_prompt': []
            }
        }

        report = pipeline.audit(
            candidate_image_path=str(candidate_path),
            product_image_path=str(product_path),
            blueprint=blueprint
        )

        assert report is not None
        assert report.status in [GuardStatus.PASS, GuardStatus.FAIL]
        assert report.image_path == str(candidate_path)
        assert report.product_image_path == str(product_path)
        assert report.blueprint_id == 'test_blueprint'

    @pytest.mark.skipif(
        True,  # Skip by default - requires real session.json
        reason="Requires actual generator session file"
    )
    def test_audit_session(self, mock_config):
        """Test auditing a complete generator session."""
        pipeline = VisualQAMatrix(config_path=mock_config)

        session_path = "path/to/session.json"
        reports = pipeline.audit_session(session_path=session_path)

        assert isinstance(reports, list)
        # Additional assertions depend on session content


class TestAuditReport:
    """Test suite for AuditReport dataclass."""

    def test_report_creation(self):
        """Test creating an audit report."""
        from src.meta.ad.reviewer.schemas.audit_report import AuditReport

        report = AuditReport(
            session_id="test_session",
            prompt_id="prompt_001",
            image_id="img_001",
            image_path="/path/to/image.jpg",
            product_image_path="/path/to/product.jpg",
            generation_model="nano-banana-pro",
            blueprint_id="blueprint_001",
            psychology_driver="trust"
        )

        assert report.session_id == "test_session"
        assert report.status == GuardStatus.PENDING
        assert report.performance_score is None

    def test_report_to_dict(self):
        """Test converting report to dict."""
        from src.meta.ad.reviewer.schemas.audit_report import AuditReport

        report = AuditReport(
            session_id="test",
            prompt_id="p1",
            image_id="i1",
            image_path="img.jpg",
            product_image_path="prod.jpg",
            generation_model="nano-banana",
            blueprint_id="b1"
        )

        data = report.to_dict()

        assert isinstance(data, dict)
        assert data['session_id'] == "test"
        assert 'status' in data

    def test_report_passed_property(self):
        """Test passed property."""
        from src.meta.ad.reviewer.schemas.audit_report import AuditReport

        report = AuditReport(
            session_id="test",
            prompt_id="p1",
            image_id="i1",
            image_path="img.jpg",
            product_image_path="prod.jpg",
            generation_model="nano-banana",
            blueprint_id="b1",
            status=GuardStatus.PASS
        )

        assert report.passed is True
        assert report.failed is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
