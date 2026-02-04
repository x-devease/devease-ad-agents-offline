#!/usr/bin/env python3
"""
Run all diagnoser agents until satisfaction.

This script runs the diagnoser orchestrator to optimize all detectors
until they meet their target metrics or reach max iterations.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.meta.diagnoser.agents.orchestrator import Orchestrator


def run_all_diagnosers(max_iterations: int = 10):
    """
    Run all diagnoser agents until satisfied.

    Args:
        max_iterations: Maximum optimization iterations per detector
    """
    print("\n" + "="*80)
    print("DIAGNOSER AGENT OPTIMIZATION")
    print("="*80 + "\n")

    # All available detectors
    detectors = [
        "FatigueDetector",
        "LatencyDetector",
        "DarkHoursDetector",
        "PerformanceDetector",
        "ConfigurationDetector"
    ]

    print(f"📋 Detected {len(detectors)} agents to optimize:")
    for i, detector in enumerate(detectors, 1):
        print(f"   {i}. {detector}")
    print()

    # Initialize orchestrator
    orchestrator = Orchestrator(
        max_iterations=max_iterations,
        use_real_llm=False  # Use mock mode for now
    )

    # Target metrics for each detector
    target_metrics = {
        "FatigueDetector": {
            "f1": 0.75,
            "precision": 0.70,
            "recall": 0.80
        },
        "LatencyDetector": {
            "f1": 0.75,
            "precision": 0.70,
            "recall": 0.80
        },
        "DarkHoursDetector": {
            "f1": 0.75,
            "precision": 0.70,
            "recall": 0.80
        },
        "PerformanceDetector": {
            "f1": 0.70,
            "precision": 0.65,
            "recall": 0.75
        },
        "ConfigurationDetector": {
            "f1": 0.70,
            "precision": 0.65,
            "recall": 0.75
        }
    }

    print(f"🎯 Target Metrics:")
    for detector, metrics in target_metrics.items():
        if detector in detectors:
            print(f"   {detector}:")
            for metric, value in metrics.items():
                print(f"      - {metric}: {value:.2f}")
    print()

    print(f"⚙️  Configuration:")
    print(f"   - Max iterations: {max_iterations}")
    print(f"   - LLM mode: Mock (safe for testing)")
    print(f"   - Parallel execution: Yes")
    print()

    print("="*80)
    print("STARTING OPTIMIZATION CYCLES")
    print("="*80 + "\n")

    # Run parallel optimization for all detectors
    results = orchestrator.run_parallel_optimization(
        detectors=detectors,
        target_metrics=target_metrics
    )

    # Print results summary
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*80 + "\n")

    for detector, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            metrics = result.get("final_metrics", {})
            print(f"✅ {detector}")
            print(f"   Status: {status}")
            if metrics:
                print(f"   Final Metrics:")
                for metric, value in metrics.items():
                    target = target_metrics.get(detector, {}).get(metric, 0.0)
                    satisfied = "✓" if value >= target else "✗"
                    print(f"      - {metric}: {value:.4f} (target: {target:.2f}) {satisfied}")
        else:
            print(f"❌ {detector}")
            print(f"   Status: {status}")
            print(f"   Message: {result.get('message', 'No message')}")
        print()

    # Check overall satisfaction
    print("="*80)
    satisfied_count = sum(
        1 for r in results.values()
        if r.get("status") == "success"
    )

    print(f"\n📊 OVERALL: {satisfied_count}/{len(detectors)} agents satisfied")

    if satisfied_count == len(detectors):
        print("\n🎉 ALL AGENTS SATISFIED!")
        return 0
    else:
        print(f"\n⚠️  {len(detectors) - satisfied_count} agents not satisfied")
        print("   Consider increasing max_iterations or adjusting target metrics")
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all diagnoser agents until satisfaction"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum optimization iterations per detector (default: 10)"
    )

    args = parser.parse_args()

    try:
        exit_code = run_all_diagnosers(max_iterations=args.max_iterations)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
