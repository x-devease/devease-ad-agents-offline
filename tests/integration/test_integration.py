"""
Integration tests for Ad Reviewer with Generator output.

Tests the complete flow from generator session.json to audit reports.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from src.meta.ad.qa.pipeline import VisualQAMatrix
from src.meta.ad.qa.schemas.audit_report import GuardStatus


class TestGeneratorIntegration:
    """Test integration with Ad Generator session output."""

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
                    'enabled': False,  # Disable for tests without VLM
                    'min_score': 7.0,
                    'model': 'gpt-4o-mini'
                },
                'cultural': {
                    'enabled': False,  # Disable for tests without VLM
                    'target_region': 'USA_Europe',
                    'risk_threshold': 'HIGH',
                    'custom_rules': []
                },
                'performance': {
                    'enabled': False,  # Disable for tests without VLM
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
    def mock_generator_session(self, tmp_path):
        """Create a mock generator session.json file."""
        session_data = {
            "session_id": "20260130_test_integration",
            "started_at": "2026-01-30T14:30:22",
            "completed_at": "2026-01-30T14:35:45",
            "metadata": {
                "product_name": "Power Station",
                "market": "US",
                "formula_path": "/path/to/recommendations.md"
            },
            "stats": {
                "total_images": 2,
                "total_cost_estimate": 0.15
            },
            "prompts": [
                {
                    "prompt_id": "prompt_001",
                    "prompt_type": "llm_enhanced",
                    "generated_at": "2026-01-30T14:30:25",
                    "prompt_text": "Professional studio lighting with warm tones, product positioned bottom-right...",
                    "features_requested": [],
                    "features_in_prompt": ["product_position", "lighting_style"],
                    "product_context": {
                        "source_image_path": str(tmp_path / "product.jpg")
                    },
                    "images": [
                        {
                            "image_id": "img_001",
                            "image_path": str(tmp_path / "generated_001.jpg"),
                            "generation_model": "nano-banana-pro",
                            "variation_index": 0,
                            "generated_at": "2026-01-30T14:30:30"
                        },
                        {
                            "image_id": "img_002",
                            "image_path": str(tmp_path / "generated_002.jpg"),
                            "generation_model": "nano-banana-pro",
                            "variation_index": 1,
                            "generated_at": "2026-01-30T14:30:35"
                        }
                    ]
                }
            ]
        }

        # Create session file
        session_file = tmp_path / "session.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        # Create test images
        import cv2
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(tmp_path / "product.jpg"), test_image)
        cv2.imwrite(str(tmp_path / "generated_001.jpg"), test_image)
        cv2.imwrite(str(tmp_path / "generated_002.jpg"), test_image)

        return session_file

    def test_audit_session_with_mock_generator_output(
        self,
        mock_config,
        mock_generator_session
    ):
        """Test auditing a complete generator session."""
        # Initialize reviewer
        pipeline = VisualQAMatrix(config_path=mock_config)

        # Audit the session
        reports = pipeline.audit_session(session_path=mock_generator_session)

        # Verify results
        assert isinstance(reports, list)
        assert len(reports) == 2  # Two images in session

        # Check first report
        report1 = reports[0]
        assert report1.session_id == "20260130_test_integration"
        assert report1.prompt_id == "prompt_001"
        assert report1.image_id in ["img_001", "img_002"]
        assert report1.generation_model == "nano-banana-pro"

        # Check that geometric guard ran
        assert report1.geometric is not None
        assert report1.geometric.guard_name == "geometric"
        assert report1.geometric.status in [GuardStatus.PASS, GuardStatus.FAIL]

    def test_audit_session_metadata_extraction(
        self,
        mock_config,
        mock_generator_session
    ):
        """Test that metadata is correctly extracted from generator session."""
        pipeline = VisualQAMatrix(config_path=mock_config)
        reports = pipeline.audit_session(session_path=mock_generator_session)

        report = reports[0]

        # Verify generator metadata
        assert report.session_id == "20260130_test_integration"
        assert report.prompt_id == "prompt_001"
        assert report.generation_model == "nano-banana-pro"

        # Verify image paths are preserved
        assert report.image_path is not None
        assert Path(report.image_path).exists()
        assert report.product_image_path is not None
        assert Path(report.product_image_path).exists()

    def test_audit_with_all_guards_enabled(
        self,
        mock_config,
        mock_generator_session,
        tmp_path
    ):
        """Test audit with all guards enabled (requires VLM mocking)."""
        # Create mock VLM client
        mock_vlm = MagicMock()
        mock_vlm.check_aesthetics.return_value = {
            'score': 8.0,
            'issues': [],
            'has_negative_feature': False,
            'detected_features': [],
            'reasoning': 'Good quality'
        }
        mock_vlm.check_culture.return_value = {
            'risk_level': 'LOW',
            'detected_issues': [],
            'confidence': 0.95,
            'reasoning': 'No issues'
        }
        mock_vlm.score_performance.return_value = {
            'psychology_alignment': 75,
            'saliency_clarity': 80,
            'consistency_realism': 70,
            'overall_score': 75,
            'reasoning': 'Good alignment',
            'suggestions': []
        }

        # Update config to enable all guards
        import yaml
        with open(mock_config, 'r') as f:
            config_data = yaml.safe_load(f)

        config_data['qa_risk_matrix']['aesthetics']['enabled'] = True
        config_data['qa_risk_matrix']['cultural']['enabled'] = True
        config_data['qa_risk_matrix']['performance']['enabled'] = True

        with open(mock_config, 'w') as f:
            yaml.dump(config_data, f)

        # Initialize with mocked VLM
        pipeline = VisualQAMatrix(config_path=mock_config, vlm_client=mock_vlm)
        reports = pipeline.audit_session(session_path=mock_generator_session)

        report = reports[0]

        # Verify all guards ran
        assert report.geometric is not None
        assert report.aesthetic is not None
        assert report.cultural is not None
        assert report.performance is not None

        # Verify performance score was calculated
        assert report.performance_score is not None
        assert report.performance_score > 0

    def test_save_reports_to_disk(
        self,
        mock_config,
        mock_generator_session,
        tmp_path
    ):
        """Test saving audit reports to disk."""
        pipeline = VisualQAMatrix(config_path=mock_config)
        reports = pipeline.audit_session(session_path=mock_generator_session)

        # Save reports
        output_dir = tmp_path / "reviewer_reports"
        for report in reports:
            report_path = output_dir / f"{report.image_id}_audit.json"
            report.to_json(str(report_path))

            # Verify file was created
            assert report_path.exists()

            # Verify content can be loaded back
            with open(report_path, 'r') as f:
                loaded_data = json.load(f)

            assert loaded_data['session_id'] == report.session_id
            assert loaded_data['image_id'] == report.image_id

    def test_report_serialization_roundtrip(
        self,
        mock_config,
        mock_generator_session
    ):
        """Test that reports can be serialized and deserialized correctly."""
        pipeline = VisualQAMatrix(config_path=mock_config)
        reports = pipeline.audit_session(session_path=mock_generator_session)

        report = reports[0]

        # Convert to dict
        report_dict = report.to_dict()

        # Verify enum values are converted to strings
        assert report_dict['status'] in ['pending', 'pass', 'fail']
        if report_dict.get('geometric'):
            assert report_dict['geometric']['status'] in ['pending', 'pass', 'fail']

        # Convert back from dict would require from_dict implementation
        # For now just verify the structure is correct
        assert 'session_id' in report_dict
        assert 'image_path' in report_dict
        assert 'total_execution_time_ms' in report_dict


class TestReportAggregation:
    """Test report aggregation and summary statistics."""

    @pytest.fixture
    def sample_reports(self, tmp_path):
        """Create sample audit reports."""
        from src.meta.ad.qa.schemas.audit_report import (
            AuditReport, GuardStatus, GeometricResult, PerformanceScore
        )

        reports = []

        # Create a passing report
        report1 = AuditReport(
            session_id="test_session",
            prompt_id="p1",
            image_id="img_001",
            image_path=str(tmp_path / "img1.jpg"),
            product_image_path=str(tmp_path / "prod.jpg"),
            generation_model="nano-banana",
            blueprint_id="b1"
        )
        report1.status = GuardStatus.PASS
        report1.performance_score = 85
        report1.geometric = GeometricResult(
            guard_name="geometric",
            status=GuardStatus.PASS,
            reasoning="Valid",
            aspect_ratio_delta=0.01,
            num_matched_features=50,
            method_used="homography"
        )
        reports.append(report1)

        # Create a failing report
        report2 = AuditReport(
            session_id="test_session",
            prompt_id="p1",
            image_id="img_002",
            image_path=str(tmp_path / "img2.jpg"),
            product_image_path=str(tmp_path / "prod.jpg"),
            generation_model="nano-banana",
            blueprint_id="b1"
        )
        report2.status = GuardStatus.FAIL
        report2.fail_guard = "geometric"
        report2.fail_reason = "Product distortion detected"
        report2.geometric = GeometricResult(
            guard_name="geometric",
            status=GuardStatus.FAIL,
            reasoning="Distortion detected",
            aspect_ratio_delta=0.10,
            num_matched_features=30,
            method_used="homography"
        )
        reports.append(report2)

        # Create another passing report
        report3 = AuditReport(
            session_id="test_session",
            prompt_id="p1",
            image_id="img_003",
            image_path=str(tmp_path / "img3.jpg"),
            product_image_path=str(tmp_path / "prod.jpg"),
            generation_model="nano-banana",
            blueprint_id="b1"
        )
        report3.status = GuardStatus.PASS
        report3.performance_score = 92
        report3.geometric = GeometricResult(
            guard_name="geometric",
            status=GuardStatus.PASS,
            reasoning="Valid",
            aspect_ratio_delta=0.005,
            num_matched_features=65,
            method_used="homography"
        )
        reports.append(report3)

        return reports

    def test_filter_passed_failed(self, sample_reports):
        """Test filtering reports by status."""
        passed = [r for r in sample_reports if r.passed]
        failed = [r for r in sample_reports if r.failed]

        assert len(passed) == 2
        assert len(failed) == 1

    def test_sort_by_performance_score(self, sample_reports):
        """Test sorting reports by performance score."""
        passed = [r for r in sample_reports if r.passed]
        passed.sort(key=lambda r: r.performance_score or 0, reverse=True)

        # Verify descending order
        assert passed[0].performance_score >= passed[1].performance_score
        assert passed[0].image_id == "img_003"  # Score 92
        assert passed[1].image_id == "img_001"  # Score 85

    def test_get_top_performers(self, sample_reports):
        """Test getting top N performers."""
        passed = [r for r in sample_reports if r.passed]
        passed.sort(key=lambda r: r.performance_score or 0, reverse=True)

        top_3 = passed[:3]

        assert len(top_3) == 2  # Only 2 passed
        assert top_3[0].performance_score == 92
        assert top_3[0].image_id == "img_003"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
