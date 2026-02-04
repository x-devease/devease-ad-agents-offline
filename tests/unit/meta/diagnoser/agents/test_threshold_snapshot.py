"""
Unit tests for threshold snapshot generation.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.generate_threshold_snapshot import (
    generate_snapshot,
    extract_fatigue_thresholds,
    extract_latency_thresholds,
    extract_dark_hours_thresholds,
    get_git_commit,
)


@pytest.fixture
def mock_fatigue_detector():
    """Mock FatigueDetector."""
    detector = MagicMock()
    detector.DEFAULT_THRESHOLDS = {
        "window_size_days": 23,
        "cpa_increase_threshold": 1.10,
        "min_golden_days": 2,
        "consecutive_days": 2,
        "min_frequency_threshold": 3.0,
    }
    return detector


@pytest.fixture
def mock_latency_detector():
    """Mock LatencyDetector."""
    detector = MagicMock()
    detector.DEFAULT_THRESHOLDS = {
        "roas_threshold": 1.0,
        "rolling_window_days": 3,
        "min_daily_spend": 10.0,
        "min_drop_ratio": 0.3,
    }
    return detector


@pytest.fixture
def mock_dark_hours_detector():
    """Mock DarkHoursDetector."""
    detector = MagicMock()
    detector.DEFAULT_THRESHOLDS = {
        "target_roas": 1.0,
        "cvr_threshold_ratio": 0.5,
        "min_spend_ratio_hourly": 0.3,
        "min_spend_ratio_daily": 0.3,
        "min_days": 3,
    }
    return detector


class TestExtractFatigueThresholds:
    """Test FatigueDetector threshold extraction."""

    @patch('scripts.generate_threshold_snapshot.FatigueDetector')
    def test_extract_fatigue_thresholds_success(self, mock_detector_class, mock_fatigue_detector):
        """Test successful extraction of FatigueDetector thresholds."""
        mock_detector_class.return_value = mock_fatigue_detector

        thresholds = extract_fatigue_thresholds()

        assert thresholds is not None
        assert "window_size_days" in thresholds
        assert thresholds["cpa_increase_threshold"] == 1.10
        assert thresholds["min_golden_days"] == 2

    @patch('scripts.generate_threshold_snapshot.FatigueDetector')
    def test_extract_fatigue_thresholds_all_params(self, mock_detector_class, mock_fatigue_detector):
        """Test extraction of all FatigueDetector parameters."""
        mock_detector_class.return_value = mock_fatigue_detector

        thresholds = extract_fatigue_thresholds()

        expected_params = [
            "window_size_days",
            "cpa_increase_threshold",
            "min_golden_days",
            "consecutive_days",
            "min_frequency_threshold",
        ]
        for param in expected_params:
            assert param in thresholds


class TestExtractLatencyThresholds:
    """Test LatencyDetector threshold extraction."""

    @patch('scripts.generate_threshold_snapshot.LatencyDetector')
    def test_extract_latency_thresholds_success(self, mock_detector_class, mock_latency_detector):
        """Test successful extraction of LatencyDetector thresholds."""
        mock_detector_class.return_value = mock_latency_detector

        thresholds = extract_latency_thresholds()

        assert thresholds is not None
        assert "roas_threshold" in thresholds
        assert thresholds["rolling_window_days"] == 3
        assert thresholds["min_daily_spend"] == 10.0


class TestExtractDarkHoursThresholds:
    """Test DarkHoursDetector threshold extraction."""

    @patch('scripts.generate_threshold_snapshot.DarkHoursDetector')
    def test_extract_dark_hours_thresholds_success(self, mock_detector_class, mock_dark_hours_detector):
        """Test successful extraction of DarkHoursDetector thresholds."""
        mock_detector_class.return_value = mock_dark_hours_detector

        thresholds = extract_dark_hours_thresholds()

        assert thresholds is not None
        assert "target_roas" in thresholds
        assert thresholds["cvr_threshold_ratio"] == 0.5
        assert thresholds["min_days"] == 3


class TestGetGitCommit:
    """Test git commit hash extraction."""

    @patch('subprocess.run')
    def test_get_git_commit_success(self, mock_run):
        """Test successful git commit retrieval."""
        mock_run.return_value = MagicMock(stdout="abc123\n", returncode=0)

        commit = get_git_commit()

        assert commit == "abc123"
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_get_git_commit_failure(self, mock_run):
        """Test git commit retrieval failure."""
        mock_run.return_value = MagicMock(stdout="", returncode=1)

        commit = get_git_commit()

        assert commit == "unknown"


class TestGenerateSnapshot:
    """Test full snapshot generation."""

    @patch('scripts.generate_threshold_snapshot.extract_dark_hours_thresholds')
    @patch('scripts.generate_threshold_snapshot.extract_latency_thresholds')
    @patch('scripts.generate_threshold_snapshot.extract_fatigue_thresholds')
    @patch('scripts.generate_threshold_snapshot.get_git_commit')
    def test_generate_snapshot_structure(
        self, mock_git, mock_fatigue, mock_latency, mock_dark_hours
    ):
        """Test snapshot generation produces correct structure."""
        mock_git.return_value = "test_commit"
        mock_fatigue.return_value = {"cpa_increase_threshold": 1.10}
        mock_latency.return_value = {"roas_threshold": 1.0}
        mock_dark_hours.return_value = {"target_roas": 1.0}

        snapshot = generate_snapshot()

        assert "timestamp" in snapshot
        assert "git_commit" in snapshot
        assert "detectors" in snapshot
        assert snapshot["git_commit"] == "test_commit"

    @patch('scripts.generate_threshold_snapshot.extract_dark_hours_thresholds')
    @patch('scripts.generate_threshold_snapshot.extract_latency_thresholds')
    @patch('scripts.generate_threshold_snapshot.extract_fatigue_thresholds')
    @patch('scripts.generate_threshold_snapshot.get_git_commit')
    def test_generate_snapshot_all_detectors(
        self, mock_git, mock_fatigue, mock_latency, mock_dark_hours
    ):
        """Test snapshot includes all detectors."""
        mock_git.return_value = "test_commit"
        mock_fatigue.return_value = {"cpa_increase_threshold": 1.10}
        mock_latency.return_value = {"roas_threshold": 1.0}
        mock_dark_hours.return_value = {"target_roas": 1.0}

        snapshot = generate_snapshot()

        assert "FatigueDetector" in snapshot["detectors"]
        assert "LatencyDetector" in snapshot["detectors"]
        assert "DarkHoursDetector" in snapshot["detectors"]

    @patch('scripts.generate_threshold_snapshot.extract_dark_hours_thresholds')
    @patch('scripts.generate_threshold_snapshot.extract_latency_thresholds')
    @patch('scripts.generate_threshold_snapshot.extract_fatigue_thresholds')
    @patch('scripts.generate_threshold_snapshot.get_git_commit')
    def test_generate_snapshot_timestamp_format(
        self, mock_git, mock_fatigue, mock_latency, mock_dark_hours
    ):
        """Test snapshot timestamp is in ISO format."""
        mock_git.return_value = "test_commit"
        mock_fatigue.return_value = {"cpa_increase_threshold": 1.10}
        mock_latency.return_value = {"roas_threshold": 1.0}
        mock_dark_hours.return_value = {"target_roas": 1.0}

        snapshot = generate_snapshot()

        # Verify ISO format
        datetime.fromisoformat(snapshot["timestamp"])

    @patch('scripts.generate_threshold_snapshot.extract_dark_hours_thresholds')
    @patch('scripts.generate_threshold_snapshot.extract_latency_thresholds')
    @patch('scripts.generate_threshold_snapshot.extract_fatigue_thresholds')
    @patch('scripts.generate_threshold_snapshot.get_git_commit')
    def test_generate_snapshot_detector_file_paths(
        self, mock_git, mock_fatigue, mock_latency, mock_dark_hours
    ):
        """Test snapshot includes correct file paths."""
        mock_git.return_value = "test_commit"
        mock_fatigue.return_value = {"cpa_increase_threshold": 1.10}
        mock_latency.return_value = {"roas_threshold": 1.0}
        mock_dark_hours.return_value = {"target_roas": 1.0}

        snapshot = generate_snapshot()

        fatigue_path = snapshot["detectors"]["FatigueDetector"]["file"]
        latency_path = snapshot["detectors"]["LatencyDetector"]["file"]
        dark_hours_path = snapshot["detectors"]["DarkHoursDetector"]["file"]

        assert "fatigue_detector.py" in fatigue_path
        assert "latency_detector.py" in latency_path
        assert "dark_hours_detector.py" in dark_hours_path


class TestSnapshotSerialization:
    """Test snapshot JSON serialization."""

    def test_snapshot_to_json(self, tmp_path):
        """Test snapshot can be serialized to JSON."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "git_commit": "test123",
            "detectors": {
                "FatigueDetector": {
                    "file": "test.py",
                    "thresholds": {"cpa_increase_threshold": 1.10}
                }
            }
        }

        output_file = tmp_path / "test_snapshot.json"
        with open(output_file, 'w') as f:
            json.dump(snapshot, f, indent=2)

        assert output_file.exists()

        # Verify it can be loaded back
        with open(output_file) as f:
            loaded = json.load(f)

        assert loaded == snapshot
