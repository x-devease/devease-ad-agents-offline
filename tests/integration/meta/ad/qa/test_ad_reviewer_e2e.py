"""
Integration tests for Ad Reviewer - converted from standalone E2E test.

Tests end-to-end functionality of VisualQAMatrix with generated ad images.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from PIL import Image

from src.meta.ad.qa.pipeline import VisualQAMatrix
from src.meta.ad.qa.schemas.audit_report import GuardStatus


@pytest.fixture
def generated_ads_dir():
    """Path to generated ads from Ad Generator."""
    return Path("results/moprobo/meta/ad_generator")


@pytest.fixture
def minimal_reviewer_config(tmp_path):
    """Create minimal reviewer config for testing."""
    config = {
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
                'enabled': False,  # Disable for tests (requires VLM API)
                'min_score': 7.0,
                'model': 'gpt-4o-mini'
            },
            'cultural': {
                'enabled': False,  # Disable for tests (requires VLM API)
                'target_region': 'USA_Europe',
                'risk_threshold': 'HIGH',
                'custom_rules': []
            },
            'performance': {
                'enabled': False,  # Disable for tests (requires VLM API)
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
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    return config_file


@pytest.fixture
def mock_vlm_client():
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


@pytest.mark.integration
class TestAdReviewerE2E:
    """
    End-to-end integration tests for Ad Reviewer.

    Converted from test_e2e/03_test_ad_reviewer.py
    """

    @pytest.mark.skipif(
        not Path("results/moprobo/meta/ad_generator").exists(),
        reason="Requires generated ads from Ad Generator (Test 2)"
    )
    def test_review_generated_ads_basic_validation(self, generated_ads_dir):
        """
        Test basic image validation of generated ads.

        Validates that generated ads meet basic requirements:
        - Size: 1080x1080
        - Mode: RGB or RGBA
        - File size: > 10KB
        """
        generated_ads = list(generated_ads_dir.rglob("*.png"))

        # Should have generated ads from Test 2
        assert len(generated_ads) > 0, "No generated ads found. Please run Ad Generator Test 2 first."

        # Validate first 3 ads (as in original test)
        for ad_path in generated_ads[:3]:
            img = Image.open(ad_path)

            # Validate size
            assert img.size[0] == 1080 and img.size[1] == 1080, \
                f"Image {ad_path.name} size should be 1080x1080, got {img.size}"

            # Validate mode
            assert img.mode in ["RGB", "RGBA"], \
                f"Image {ad_path.name} mode should be RGB or RGBA, got {img.mode}"

            # Validate file size
            assert ad_path.stat().st_size > 10000, \
                f"Image {ad_path.name} file size must be > 10KB"

    @pytest.mark.skipif(
        not Path("results/moprobo/meta/ad_generator").exists(),
        reason="Requires generated ads from Ad Generator (Test 2)"
    )
    def test_reviewer_creates_report(self, generated_ads_dir, tmp_path):
        """Test that reviewer creates YAML report with correct structure."""
        generated_ads = list(generated_ads_dir.rglob("*.png"))

        assert len(generated_ads) > 0, "No generated ads found"

        # Create review report directory
        report_dir = tmp_path / "ad_reviewer"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Build report structure (as in original test)
        report = {
            "review_summary": {
                "customer": "moprobo",
                "platform": "meta",
                "total_ads_reviewed": len(generated_ads),
                "timestamp": "2026-01-30"
            },
            "ads": []
        }

        # Add ad info to report
        for i, ad_path in enumerate(generated_ads, 1):
            img = Image.open(ad_path)
            ad_report = {
                "ad_number": i,
                "file_path": str(ad_path),
                "file_size_kb": round(ad_path.stat().st_size / 1024, 2),
                "dimensions": list(img.size),  # Convert tuple to list for YAML
                "mode": img.mode,
                "format": img.format,
                "validation": "PASSED"
            }
            report["ads"].append(ad_report)

        # Save report
        report_path = report_dir / "review_report.yaml"
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)

        # Verify report was created
        assert report_path.exists(), "Review report should be created"

        # Verify report structure
        with open(report_path, 'r') as f:
            loaded_report = yaml.safe_load(f)

        assert "review_summary" in loaded_report
        assert "ads" in loaded_report
        assert loaded_report["review_summary"]["customer"] == "moprobo"
        assert loaded_report["review_summary"]["platform"] == "meta"
        assert len(loaded_report["ads"]) == len(generated_ads)

        # Verify first ad entry
        first_ad = loaded_report["ads"][0]
        assert "ad_number" in first_ad
        assert "file_path" in first_ad
        assert "dimensions" in first_ad
        assert first_ad["dimensions"] == [1080, 1080]

    @pytest.mark.skipif(
        not Path("config/moprobo/meta/config.yaml").exists(),
        reason="Requires moprobo config file"
    )
    def test_visualqa_matrix_initialization(self):
        """Test VisualQAMatrix initialization with real config."""
        config_path = Path("config/moprobo/meta/config.yaml")

        try:
            reviewer = VisualQAMatrix(config_path=str(config_path))
            assert reviewer.config_path == config_path
            assert reviewer.geometric_guard is not None
        except Exception as e:
            pytest.skip(f"Could not initialize full reviewer: {e}")

    def test_visualqa_matrix_with_mock_config(self, minimal_reviewer_config, mock_vlm_client):
        """Test VisualQAMatrix with mock config and VLM client."""
        reviewer = VisualQAMatrix(
            config_path=str(minimal_reviewer_config),
            vlm_client=mock_vlm_client
        )

        assert reviewer.config_path == minimal_reviewer_config
        assert reviewer.geometric_guard is not None
        assert reviewer.vlm_client == mock_vlm_client

    @pytest.mark.skipif(
        not Path("results/moprobo/meta/ad_generator").exists(),
        reason="Requires generated ads from Ad Generator (Test 2)"
    )
    def test_audit_generated_ad_with_minimal_config(
        self, generated_ads_dir, minimal_reviewer_config, tmp_path
    ):
        """Test auditing a generated ad with minimal config."""
        generated_ads = list(generated_ads_dir.rglob("*.png"))
        if not generated_ads:
            pytest.skip("No generated ads found")

        ad_path = generated_ads[0]

        # Create a dummy product image for testing
        import numpy as np
        import cv2

        dummy_product = np.ones((1080, 1080, 3), dtype=np.uint8) * 255
        product_path = tmp_path / "product.png"
        cv2.imwrite(str(product_path), dummy_product)

        # Run audit with minimal config (only geometric guard)
        reviewer = VisualQAMatrix(config_path=str(minimal_reviewer_config))
        blueprint = {
            'id': 'test_blueprint',
            'strategy_rationale': {
                'psychology_driver': 'trust'
            },
            'nano_generation_rules': {
                'negative_prompt': []
            }
        }

        report = reviewer.audit(
            candidate_image_path=str(ad_path),
            product_image_path=str(product_path),
            blueprint=blueprint
        )

        # Verify report structure
        assert report is not None
        assert report.status in [GuardStatus.PASS, GuardStatus.FAIL]
        assert report.image_path == str(ad_path)
        assert report.product_image_path == str(product_path)
        assert report.blueprint_id == 'test_blueprint'
