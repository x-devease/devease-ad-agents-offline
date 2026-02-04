#!/usr/bin/env python3
"""
Generate Threshold Snapshot for Agent Prompts.

Extracts DEFAULT_THRESHOLDS from all detector classes and generates
a JSON snapshot file that can be injected into agent prompts at runtime.

This ensures prompts always use the correct, up-to-date threshold values.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def extract_fatigue_thresholds() -> Dict[str, Any]:
    """Extract DEFAULT_THRESHOLDS from FatigueDetector."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.meta.diagnoser.detectors.fatigue_detector import FatigueDetector

    return FatigueDetector.DEFAULT_THRESHOLDS.copy()


def extract_latency_thresholds() -> Dict[str, Any]:
    """Extract DEFAULT_THRESHOLDS from LatencyDetector."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.meta.diagnoser.detectors.latency_detector import LatencyDetector

    return LatencyDetector.DEFAULT_THRESHOLDS.copy()


def extract_dark_hours_thresholds() -> Dict[str, Any]:
    """Extract DEFAULT_THRESHOLDS from DarkHoursDetector."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.meta.diagnoser.detectors.dark_hours_detector import DarkHoursDetector

    return DarkHoursDetector.DEFAULT_THRESHOLDS.copy()


def generate_snapshot() -> Dict[str, Any]:
    """Generate complete threshold snapshot."""
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "detectors": {
            "FatigueDetector": {
                "file": "src/meta/diagnoser/detectors/fatigue_detector.py",
                "thresholds": extract_fatigue_thresholds(),
            },
            "LatencyDetector": {
                "file": "src/meta/diagnoser/detectors/latency_detector.py",
                "thresholds": extract_latency_thresholds(),
            },
            "DarkHoursDetector": {
                "file": "src/meta/diagnoser/detectors/dark_hours_detector.py",
                "thresholds": extract_dark_hours_thresholds(),
            },
        },
        "metadata": {
            "version": "1.0.0",
            "description": "Runtime threshold snapshot for agent prompts",
            "warning": "DO NOT manually edit this file. It is auto-generated.",
        },
    }

    return snapshot


def save_snapshot(snapshot: Dict[str, Any], output_path: str):
    """Save snapshot to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"✅ Threshold snapshot saved to: {output_file}")
    print(f"   Git commit: {snapshot['git_commit']}")
    print(f"   Timestamp: {snapshot['timestamp']}")

    # Print summary
    print("\n📊 Detector Thresholds Summary:")
    for detector_name, detector_data in snapshot["detectors"].items():
        print(f"\n   {detector_name}:")
        for key, value in detector_data["thresholds"].items():
            print(f"      {key}: {value}")


def main():
    """Main entry point."""
    print("🔍 Generating threshold snapshot from detector code...")

    # Generate snapshot
    snapshot = generate_snapshot()

    # Save to prompts directory
    output_path = "src/meta/diagnoser/agents/prompts/threshold_snapshot.json"
    save_snapshot(snapshot, output_path)

    print("\n✅ Done! Run this script whenever detector thresholds change.")


if __name__ == "__main__":
    main()
