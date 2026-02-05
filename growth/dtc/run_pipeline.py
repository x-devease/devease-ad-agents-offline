#!/usr/bin/env python3
"""
Main Pipeline Runner - Shopify Store Discovery & Profiling System

This script orchestrates the entire pipeline:
1. Discover new Shopify domains via CT logs
2. Profile stores (SKU count, pricing, velocity)
3. Scrape Meta Ad Library for ad intelligence
4. Enrich with decision maker contact info
5. Generate prioritized outreach tasks

Usage:
    python run_pipeline.py [--full | --step STEP] [--options]
"""

import argparse
import asyncio
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class PipelineRunner:
    """Orchestrate the entire lead discovery pipeline."""

    def __init__(self, base_dir: Path = Path(__file__).parent):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.config_dir = base_dir / "config"

        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)

    def print_banner(self):
        """Print pipeline banner."""
        print(f"\n{'='*70}")
        print(f"  Shopify DTC Store Discovery & Profiling Pipeline")
        print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

    def print_section(self, title: str):
        """Print section header."""
        print(f"\n{'─'*70}")
        print(f"  {title}")
        print(f"{'─'*70}\n")

    def step1_discover_domains(self, lookback_days: int = 1) -> bool:
        """
        Step 1: Discover new Shopify domains via Certificate Transparency logs.

        Args:
            lookback_days: Days to look back for new certificates

        Returns:
            True if successful
        """
        self.print_section("Step 1: Discovering New Shopify Domains")

        cmd = [
            sys.executable,
            str(self.base_dir / "discover_domains.py"),
            "--lookback-days", str(lookback_days),
            "--output", str(self.data_dir / "new_domains.csv")
        ]

        return self._run_command(cmd)

    def step2_profile_stores(self, min_sku: int = 10) -> bool:
        """
        Step 2: Profile Shopify stores for SKU count and metrics.

        Args:
            min_sku: Minimum SKU count threshold

        Returns:
            True if successful
        """
        self.print_section("Step 2: Profiling Stores")

        cmd = [
            sys.executable,
            str(self.base_dir / "profile_shopify.py"),
            "--input", str(self.data_dir / "new_domains.csv"),
            "--output", str(self.data_dir / "store_profiles.json"),
            "--min-sku", str(min_sku)
        ]

        return self._run_command(cmd)

    def step3_fetch_meta_intel(self, headful: bool = False) -> bool:
        """
        Step 3: Scrape Meta Ad Library for ad intelligence.

        Args:
            headful: Run with visible browser

        Returns:
            True if successful
        """
        self.print_section("Step 3: Fetching Meta Ad Intelligence")

        cmd = [
            sys.executable,
            str(self.base_dir / "fetch_meta_intel.py"),
            "--input", str(self.data_dir / "store_profiles.json"),
            "--output", str(self.data_dir / "ad_intent.json")
        ]

        if headful:
            cmd.append("--headful")

        # This is an async module, run it directly
        return self._run_async_command(cmd)

    def step4_enrich_contacts(self) -> bool:
        """
        Step 4: Enrich targets with decision maker contact info.

        Returns:
            True if successful
        """
        self.print_section("Step 4: Enriching Decision Maker Contacts")

        cmd = [
            sys.executable,
            str(self.base_dir / "enrich_decision_makers.py"),
            "--input", str(self.data_dir / "ad_intent.json"),
            "--store-profiles", str(self.data_dir / "store_profiles.json"),
            "--output", str(self.data_dir / "final_targets.json")
        ]

        return self._run_command(cmd)

    def step5_orchestrate_tasks(self, top_count: int = 20) -> bool:
        """
        Step 5: Generate prioritized outreach tasks.

        Args:
            top_count: Number of top leads to generate tasks for

        Returns:
            True if successful
        """
        self.print_section("Step 5: Orchestration Tasks")

        cmd = [
            sys.executable,
            str(self.base_dir / "orchestrate_tasks.py"),
            "--input", str(self.data_dir / "final_targets.json"),
            "--output", str(self.data_dir / "tasks.yaml"),
            "--config", str(self.config_dir / "settings.yaml"),
            "--top", str(top_count)
        ]

        return self._run_command(cmd)

    def _run_command(self, cmd: list) -> bool:
        """
        Run a command synchronously.

        Args:
            cmd: Command list

        Returns:
            True if successful
        """
        try:
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                check=True,
                capture_output=False
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Command failed with exit code {e.returncode}")
            return False
        except FileNotFoundError:
            print(f"[ERROR] Could not find {cmd[0]}")
            return False

    def _run_async_command(self, cmd: list) -> bool:
        """
        Run an async module by importing and calling it.

        Args:
            cmd: Command list (first element is the script name)

        Returns:
            True if successful
        """
        script_name = Path(cmd[1]).stem

        # Import the module
        if script_name == "fetch_meta_intel":
            from fetch_meta_intel import main_async

            # Build args object
            class Args:
                def __init__(self, cmd_list):
                    self.input = cmd_list[cmd_list.index("--input") + 1]
                    self.output = cmd_list[cmd_list.index("--output") + 1]
                    self.headful = "--headful" in cmd_list
                    self.timeout = 30000

            args = Args(cmd)

            try:
                asyncio.run(main_async(args))
                return True
            except Exception as e:
                print(f"[ERROR] Async execution failed: {e}")
                return False

        return False

    def run_full_pipeline(
        self,
        lookback_days: int = 1,
        min_sku: int = 10,
        top_count: int = 20,
        headful: bool = False
    ) -> bool:
        """
        Run the complete pipeline.

        Args:
            lookback_days: Days to look back for new certificates
            min_sku: Minimum SKU count threshold
            top_count: Number of top leads for tasks
            headful: Run Meta scraper with visible browser

        Returns:
            True if all steps successful
        """
        self.print_banner()

        results = []

        # Step 1: Discover domains
        results.append(self.step1_discover_domains(lookback_days))

        # Check if we found any domains
        if not (self.data_dir / "new_domains.csv").exists():
            print("\n[WARNING] No new domains found. Stopping pipeline.")
            return False

        # Step 2: Profile stores
        results.append(self.step2_profile_stores(min_sku))

        # Step 3: Fetch Meta intel
        results.append(self.step3_fetch_meta_intel(headful))

        # Step 4: Enrich contacts
        results.append(self.step4_enrich_contacts())

        # Step 5: Orchestrate tasks
        results.append(self.step5_orchestrate_tasks(top_count))

        # Print final summary
        self.print_summary(results)

        return all(results)

    def print_summary(self, results: list):
        """Print pipeline execution summary."""
        print(f"\n{'='*70}")
        print(f"  Pipeline Complete - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        steps = [
            "Discover Domains",
            "Profile Stores",
            "Fetch Meta Intel",
            "Enrich Contacts",
            "Orchestrate Tasks"
        ]

        for step, result in zip(steps, results):
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {status}: {step}")

        print(f"\n{'='*70}\n")

        if all(results):
            print("  🎉 All steps completed successfully!")
            print(f"  Check {self.data_dir / 'tasks.yaml'} for your outreach tasks.\n")
        else:
            print("  ⚠️  Some steps failed. Check logs above for details.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run the Shopify DTC Store Discovery Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_pipeline.py --full

  # Run specific step
  python run_pipeline.py --step discover

  # Run with custom options
  python run_pipeline.py --full --lookback-days 2 --min-sku 20 --top 50

  # Run Meta scraper with visible browser (for debugging)
  python run_pipeline.py --step meta --headful
        """
    )

    parser.add_argument(
        '--full',
        action='store_true',
        help='Run the complete pipeline'
    )
    parser.add_argument(
        '--step',
        type=str,
        choices=['discover', 'profile', 'meta', 'enrich', 'orchestrate'],
        help='Run a specific pipeline step'
    )
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=1,
        help='Days to look back for new certificates (default: 1)'
    )
    parser.add_argument(
        '--min-sku',
        type=int,
        default=10,
        help='Minimum SKU count threshold (default: 10)'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=20,
        help='Number of top leads to generate tasks for (default: 20)'
    )
    parser.add_argument(
        '--headful',
        action='store_true',
        help='Run Meta scraper with visible browser'
    )

    args = parser.parse_args()

    runner = PipelineRunner()

    if args.full:
        # Run complete pipeline
        success = runner.run_full_pipeline(
            lookback_days=args.lookback_days,
            min_sku=args.min_sku,
            top_count=args.top,
            headful=args.headful
        )
        sys.exit(0 if success else 1)

    elif args.step:
        # Run specific step
        step_map = {
            'discover': lambda: runner.step1_discover_domains(args.lookback_days),
            'profile': lambda: runner.step2_profile_stores(args.min_sku),
            'meta': lambda: runner.step3_fetch_meta_intel(args.headful),
            'enrich': lambda: runner.step4_enrich_contacts(),
            'orchestrate': lambda: runner.step5_orchestrate_tasks(args.top)
        }

        success = step_map[args.step]()
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
