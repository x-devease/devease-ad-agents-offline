"""
Unit tests for threshold snapshot generation.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.generate_threshold_snapshot import (
    generate_snapshot,
    extract_fatigue_thresholds,
    extract_latency_thresholds,
    extract_dark_hours_thresholds,
    get_git_commit,
)


class TestExtractFatigueThresholds:
    """Test FatigueDetector threshold extraction."""

    def test_extract_fatigue_thresholds_success(self):
        """Test successful extraction of FatigueDetector thresholds."""
        thresholds = extract_fatigue_thresholds()

        assert thresholds is not None
        assert isinstance(thresholds, dict)
        assert "cpa_increase_threshold" in thresholds

    def test_extract_fatigue_thresholds_all_params(self):
        """Test extraction of all FatigueDetector parameters."""
        thresholds = extract_fatigue_thresholds()

        # Check for expected threshold parameters
        expected_params = ["window_size_days", "cpa_increase_threshold"]
        for param in expected_params:
            assert param in thresholds


class TestExtractLatencyThresholds:
    """Test LatencyDetector threshold extraction."""

    def test_extract_latency_thresholds_success(self):
        """Test successful extraction of LatencyDetector thresholds."""
        thresholds = extract_latency_thresholds()

        assert thresholds is not None
        assert isinstance(thresholds, dict)

    def test_extract_latency_has_expected_keys(self):
        """Test latency thresholds have expected keys."""
        thresholds = extract_latency_thresholds()

        # Should have roas_threshold at minimum
        assert "roas_threshold" in thresholds


class TestExtractDarkHoursThresholds:
    """Test DarkHoursDetector threshold extraction."""

    def test_extract_dark_hours_thresholds_success(self):
        """Test successful extraction of DarkHoursDetector thresholds."""
        thresholds = extract_dark_hours_thresholds()

        assert thresholds is not None
        assert isinstance(thresholds, dict)

    def test_extract_dark_hours_has_target_roas(self):
        """Test dark hours thresholds include target_roas."""
        thresholds = extract_dark_hours_thresholds()

        assert "target_roas" in thresholds


class TestGetGitCommit:
    """Test git commit hash extraction."""

    @patch('subprocess.run')
    def test_get_git_commit_success(self, mock_run):
        """Test successful git commit retrieval."""
        mock_run.return_value.stdout = "abc123\n"
        mock_run.return_value.returncode = 0

        commit = get_git_commit()

        assert commit == "abc123"
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_get_git_commit_failure(self, mock_run):
        """Test git commit retrieval failure returns 'unknown'."""
        import subprocess
        mock_run.side_effect = subprocess.CalledProcessError(1, 'git')

        commit = get_git_commit()

        # Should return "unknown" on failure
        assert commit == "unknown"


class TestGenerateSnapshot:
    """Test full snapshot generation."""

    def test_generate_snapshot_structure(self):
        """Test snapshot generation produces correct structure."""
        snapshot = generate_snapshot()

        assert "timestamp" in snapshot
        assert "git_commit" in snapshot
        assert "detectors" in snapshot

    def test_generate_snapshot_all_detectors(self):
        """Test snapshot includes all detectors."""
        snapshot = generate_snapshot()

        assert "FatigueDetector" in snapshot["detectors"]
        assert "LatencyDetector" in snapshot["detectors"]
        assert "DarkHoursDetector" in snapshot["detectors"]

    def test_generate_snapshot_timestamp_format(self):
        """Test snapshot timestamp is in ISO format."""
        snapshot = generate_snapshot()

        # Verify ISO format
        datetime.fromisoformat(snapshot["timestamp"])

    def test_generate_snapshot_detector_file_paths(self):
        """Test snapshot includes correct file paths."""
        snapshot = generate_snapshot()

        fatigue_path = snapshot["detectors"]["FatigueDetector"]["file"]
        latency_path = snapshot["detectors"]["LatencyDetector"]["file"]
        dark_hours_path = snapshot["detectors"]["DarkHoursDetector"]["file"]

        assert "fatigue_detector.py" in fatigue_path
        assert "latency_detector.py" in latency_path
        assert "dark_hours_detector.py" in dark_hours_path

    def test_generate_snapshot_thresholds_are_dicts(self):
        """Test all thresholds are dictionaries."""
        snapshot = generate_snapshot()

        for detector_name, detector_data in snapshot["detectors"].items():
            assert isinstance(detector_data["thresholds"], dict)


class TestSnapshotSerialization:
    """Test snapshot JSON serialization."""

    def test_snapshot_to_json(self, tmp_path):
        """Test snapshot can be serialized to JSON."""
        snapshot = generate_snapshot()

        output_file = tmp_path / "test_snapshot.json"
        with open(output_file, 'w') as f:
            json.dump(snapshot, f, indent=2)

        assert output_file.exists()

        # Verify it can be loaded back
        with open(output_file) as f:
            loaded = json.load(f)

        assert loaded == snapshot

    def test_snapshot_json_has_required_keys(self):
        """Test snapshot JSON has all required keys."""
        snapshot = generate_snapshot()

        required_keys = ["timestamp", "git_commit", "detectors"]
        for key in required_keys:
            assert key in snapshot
