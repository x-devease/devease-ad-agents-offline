#!/usr/bin/env python3
"""
Demo script for the AI Agent Orchestrator.

Runs a complete optimization cycle on the FatigueDetector.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.meta.diagnoser.agents import Orchestrator


def main():
    """Run demo optimization cycle."""
    print("\n" + "="*80)
    print("AI Agent Team - Demo Optimization Cycle")
    print("="*80)

    # Initialize orchestrator
    orchestrator = Orchestrator(max_iterations=1)

    # Run optimization on FatigueDetector
    print("\nTarget Detector: FatigueDetector")
    print("Goal: Improve recall while maintaining high precision")

    target_metrics = {
        "f1_score": 0.75,
        "recall": 0.65
    }

    # Run the cycle
    result = orchestrator.run_optimization_cycle(
        detector="FatigueDetector",
        target_metrics=target_metrics
    )

    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    if result["status"] in ["success", "failed"]:
        print(f"\nStatus: {result['status'].upper()}")
        print(f"Experiment ID: {result.get('experiment_id', 'N/A')}")
        print(f"Outcome: {result.get('outcome', 'N/A')}")

        if result.get("evaluation"):
            eval_result = result["evaluation"]["evaluation_result"]
            print(f"\nDecision: {eval_result['decision']}")
            print(f"Overall Score: {eval_result['overall_score']}/100")

            metrics = eval_result.get("metrics", {})
            if "lift" in metrics:
                print(f"\nMetrics Lift:")
                for metric, lift in metrics["lift"].items():
                    print(f"  {metric}: {lift}")

        print("\n✓ Demo completed successfully!")
        print("\nExperiment archived to Memory Agent:")
        print(f"  Location: src/meta/diagnoser/agents/memory/experiments/")

    elif result["status"] == "rejected":
        print(f"\n❌ Experiment was rejected during review phase")
        if "review" in result:
            concerns = result["review"]["feedback"].get("concerns", [])
            print(f"  Concerns: {concerns}")

    else:
        print(f"\n❌ Error: {result.get('message', 'Unknown error')}")

    print("\n" + "="*80)

    return 0 if result["status"] in ["success", "failed"] else 1


if __name__ == "__main__":
    exit(main())
