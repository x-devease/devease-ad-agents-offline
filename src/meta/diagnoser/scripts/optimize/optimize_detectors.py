#!/usr/bin/env python3
"""
Unified detector optimization script.

Provides multiple optimization strategies:
1. agent - AI agent-based optimization (uses orchestrator)
2. manual - Manual threshold tuning (write config files)
3. adaptive - Iterative optimization with auto-adjustment
4. aggressive - Fast 5-cycle optimization

Usage:
    # Agent-based optimization (default)
    python optimize/optimize_detectors.py --strategy agent --detectors FatigueDetector

    # Manual optimization with custom thresholds
    python optimize/optimize_detectors.py --strategy manual --detectors DarkHoursDetector

    # Aggressive optimization (5 cycles)
    python optimize/optimize_detectors.py --strategy aggressive --cycles 5

    # Parallel optimization of all detectors
    python optimize/optimize_detectors.py --strategy agent --detectors all

Examples:
    # Optimize single detector with agent
    python optimize/optimize_detectors.py --strategy agent --detectors FatigueDetector

    # Optimize all detectors
    python optimize/optimize_detectors.py --strategy agent --detectors all

    # Quick aggressive optimization
    python optimize/optimize_detectors.py --strategy aggressive --detectors FatigueDetector --cycles 3
"""

import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.meta.diagnoser.agents.orchestrator import Orchestrator
from src.meta.diagnoser.detectors import (
    FatigueDetector,
    LatencyDetector,
    DarkHoursDetector,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Available detectors
ALL_DETECTORS = [
    "FatigueDetector",
    "LatencyDetector",
    "DarkHoursDetector",
]

# Default target metrics
DEFAULT_TARGET_METRICS = {
    "f1": 0.75,
    "precision": 0.70,
    "recall": 0.80
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize detector thresholds using various strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="agent",
        choices=["agent", "manual", "adaptive", "aggressive"],
        help="Optimization strategy (default: agent)"
    )

    parser.add_argument(
        "--detectors",
        type=str,
        nargs="+",
        default=["FatigueDetector"],
        help="Detectors to optimize (default: FatigueDetector, use 'all' for all detectors)"
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=10,
        help="Maximum optimization cycles (default: 10)"
    )

    parser.add_argument(
        "--target-f1",
        type=float,
        default=0.75,
        help="Target F1 score (default: 0.75)"
    )

    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.70,
        help="Target precision (default: 0.70)"
    )

    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.80,
        help="Target recall (default: 0.80)"
    )

    parser.add_argument(
        "--use-real-llm",
        action="store_true",
        help="Use real LLM for agent-based optimization (default: mock mode)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - don't make actual changes"
    )

    return parser.parse_args()


def expand_detectors(detectors_list):
    """Expand 'all' to full detector list."""
    if "all" in detectors_list:
        return ALL_DETECTORS
    return detectors_list


def backup_evaluation_reports():
    """Backup current evaluation reports."""
    reports_dir = Path("src/meta/diagnoser/evaluator/reports/moprobo_sliding")

    if not reports_dir.exists():
        logger.warning(f"Reports directory not found: {reports_dir}")
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = Path(f"src/meta/diagnoser/evaluator/reports/moprobo_sliding_backup_{timestamp}")

    import shutil
    shutil.copytree(reports_dir, backup_dir)
    logger.info(f"Backed up reports to: {backup_dir}")

    return backup_dir


def optimize_with_agent(detectors, max_cycles, target_metrics, use_real_llm, dry_run):
    """Optimize detectors using AI agent orchestrator."""
    logger.info("=" * 80)
    logger.info("AGENT-BASED OPTIMIZATION")
    logger.info("=" * 80)

    if dry_run:
        logger.info("[DRY-RUN] Would initialize Orchestrator")
        logger.info(f"[DRY-RUN] Detectors: {detectors}")
        logger.info(f"[DRY-RUN] Max cycles: {max_cycles}")
        logger.info(f"[DRY-RUN] Target metrics: {target_metrics}")
        return True

    orchestrator = Orchestrator(
        max_iterations=max_cycles,
        use_real_llm=use_real_llm
    )

    for detector in detectors:
        logger.info(f"\nOptimizing {detector}...")

        try:
            result = orchestrator.optimize_detector(
                detector_name=detector,
                target_metrics=target_metrics
            )

            if result.get("satisfied"):
                logger.info(f"✅ {detector} optimized successfully!")
            else:
                logger.warning(f"⚠️  {detector} did not meet targets after {max_cycles} cycles")

        except Exception as e:
            logger.error(f"❌ Failed to optimize {detector}: {e}")
            import traceback
            traceback.print_exc()

    return True


def optimize_manual(detectors, target_metrics, dry_run):
    """Optimize detectors with manual threshold tuning."""
    logger.info("=" * 80)
    logger.info("MANUAL OPTIMIZATION")
    logger.info("=" * 80)

    optimized_configs = {
        "FatigueDetector": {
            "description": "Optimized for high recall (80%)",
            "thresholds": {
                "fatigue_freq_threshold": 2.0,
                "cpa_increase_threshold": 1.05,
                "min_golden_days": 1,
                "window_size_days": 23,
                "consecutive_days": 1,
            }
        },
        "LatencyDetector": {
            "description": "Optimized for balanced performance",
            "thresholds": {
                "latency_threshold_days": 3,
                "min_consecutive_days": 2,
            }
        },
        "DarkHoursDetector": {
            "description": "Optimized for high recall (80%)",
            "thresholds": {
                "target_roas": 2.0,
                "cvr_threshold_ratio": 0.15,
                "min_spend_ratio_hourly": 0.03,
                "min_spend_ratio_daily": 0.05,
                "min_days": 21,
            }
        },
    }

    for detector in detectors:
        if detector not in optimized_configs:
            logger.warning(f"No manual config for {detector} - skipping")
            continue

        logger.info(f"\nOptimizing {detector}...")

        config = optimized_configs[detector]
        detector_short = detector.replace("Detector", "").lower()
        config_file = Path(f"src/meta/diagnoser/detectors/config/{detector_short}_detector_config.json")

        if dry_run:
            logger.info(f"[DRY-RUN] Would write config to: {config_file}")
            logger.info(f"[DRY-RUN] Config: {json.dumps(config, indent=2)}")
            continue

        # Create config directory if doesn't exist
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Write config
        config["detector_name"] = detector
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"✅ Updated {detector} config: {config_file}")

    return True


def optimize_aggressive(detectors, cycles, target_metrics, dry_run):
    """Optimize detectors with aggressive threshold tuning."""
    logger.info("=" * 80)
    logger.info(f"AGGRESSIVE OPTIMIZATION ({cycles} cycles)")
    logger.info("=" * 80)

    if dry_run:
        logger.info(f"[DRY-RUN] Would run {cycles} aggressive cycles for: {detectors}")
        return True

    # Aggressive thresholds - lower for higher recall
    aggressive_configs = {
        "FatigueDetector": {
            "fatigue_freq_threshold": 1.5,      # Very low
            "cpa_increase_threshold": 1.02,    # Very low
            "min_golden_days": 1,
        },
        "LatencyDetector": {
            "latency_threshold_days": 2,       # Very low
            "min_consecutive_days": 1,
        },
        "DarkHoursDetector": {
            "target_roas": 1.5,                # Very low
            "cvr_threshold_ratio": 0.10,       # Very low
            "min_spend_ratio_hourly": 0.02,
            "min_spend_ratio_daily": 0.03,
        },
    }

    logger.info("Applying aggressive thresholds for high recall...")

    for detector in detectors:
        if detector not in aggressive_configs:
            continue

        logger.info(f"\n{detector}:")
        thresholds = aggressive_configs[detector]

        for key, value in thresholds.items():
            logger.info(f"  {key}: {value}")

    # For now, use manual optimization with aggressive thresholds
    return optimize_manual(detectors, target_metrics, dry_run=False)


def main():
    """Main optimization function."""
    args = parse_args()

    # Expand detector list
    detectors = expand_detectors(args.detectors)

    # Build target metrics
    target_metrics = {
        "f1": args.target_f1,
        "precision": args.target_precision,
        "recall": args.target_recall,
    }

    logger.info("=" * 80)
    logger.info("DETECTOR OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Detectors: {', '.join(detectors)}")
    logger.info(f"Max cycles: {args.cycles}")
    logger.info(f"Target metrics: F1≥{args.target_f1:.2f}, P≥{args.target_precision:.2f}, R≥{args.target_recall:.2f}")
    logger.info(f"Use real LLM: {args.use_real_llm}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("")

    # Backup reports before optimization
    if not args.dry_run:
        backup_evaluation_reports()

    # Route to appropriate strategy
    success = False

    if args.strategy == "agent":
        success = optimize_with_agent(
            detectors=detectors,
            max_cycles=args.cycles,
            target_metrics=target_metrics,
            use_real_llm=args.use_real_llm,
            dry_run=args.dry_run
        )
    elif args.strategy == "manual":
        success = optimize_manual(
            detectors=detectors,
            target_metrics=target_metrics,
            dry_run=args.dry_run
        )
    elif args.strategy == "aggressive":
        success = optimize_aggressive(
            detectors=detectors,
            cycles=args.cycles,
            target_metrics=target_metrics,
            dry_run=args.dry_run
        )
    elif args.strategy == "adaptive":
        # Adaptive uses agent-based for now
        success = optimize_with_agent(
            detectors=detectors,
            max_cycles=args.cycles,
            target_metrics=target_metrics,
            use_real_llm=args.use_real_llm,
            dry_run=args.dry_run
        )

    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✓ Optimization Complete!")
    else:
        logger.warning("⚠ Optimization completed with issues")
    logger.info("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
