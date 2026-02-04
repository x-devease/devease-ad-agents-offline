"""
Diagnoser CLI commands for run.py integration.
"""

import click
import logging
from pathlib import Path

from src.meta.diagnoser import Diagnoser
from src.meta.diagnoser.detectors import (
    FatigueDetector,
    LatencyDetector,
    DarkHoursDetector,
    infer_status_changes,
)
from src.config.path_manager import PathManager


logger = logging.getLogger(__name__)


@click.group()
def diagnose():
    """Diagnose ad performance issues."""
    pass


@diagnose.command()
@click.option("--customer", required=True, help="Customer name")
@click.option("--platform", default="meta", help="Platform name (default: meta)")
@click.option("--entity", default="account", type=click.Choice(["account", "campaign", "adset"]))
@click.option("--entity-id", default=None, help="Specific entity ID to diagnose")
@click.option("--detector", default=None, type=click.Choice(["fatigue", "latency", "dark_hours", "all"]))
@click.option("--output", default=None, help="Output directory for reports")
def run(customer, platform, entity, entity_id, detector, output):
    """
    Run diagnostics on ad performance data.

    Examples:

        # Diagnose entire account
        python run.py diagnose --customer moprobo --entity account

        # Diagnose specific campaign
        python run.py diagnose --customer moprobo --entity campaign --entity-id 12345

        # Run only fatigue detector
        python run.py diagnose --customer moprobo --detector fatigue

        # Run all detectors on adset
        python run.py diagnose --customer moprobo --entity adset --detector all
    """
    click.echo(f"🔍 Running diagnostics for {customer}/{platform}...")
    click.echo(f"   Entity: {entity}" + (f" ({entity_id})" if entity_id else ""))
    click.echo(f"   Detector: {detector or 'all'}")

    # Setup paths
    path_manager = PathManager(customer, platform)

    # Determine input data paths
    if detector in ["fatigue", "all", None]:
        # Check for processed data
        processed_path = path_manager.get_dataset_path("processed")
        if processed_path and (processed_path / "ad_daily_processed.csv").exists():
            ad_daily_path = processed_path / "ad_daily_processed.csv"
            click.echo(f"✅ Using preprocessed data: {ad_daily_path}")
        else:
            # Check for raw data
            ad_daily_path = path_manager.get_raw_data_path("ad_daily_insights")
            if not ad_daily_path or not ad_daily_path.exists():
                click.echo(f"❌ Data not found. Run preprocessing first:")
                click.echo(f"   python scripts/parse_moprobo_data.py --customer {customer}")
                return

    # Initialize diagnoser
    diagnoser = Diagnoser()

    # Run detection
    if detector == "fatigue" or detector is None:
        click.echo("\n📊 Running Creative Fatigue Detection...")
        _run_fatigue_detection(path_manager, entity, entity_id, output)

    if detector == "latency" or detector is None:
        click.echo("\n⏱️  Running Human Latency Detection...")
        _run_latency_detection(path_manager, entity, entity_id, output)

    if detector == "dark_hours" or detector is None:
        click.echo("\n🌙 Running Dark Hours Analysis...")
        _run_dark_hours_detection(path_manager, entity, entity_id, output)

    if detector == "all":
        click.echo("\n⏱️  Running Human Latency Detection...")
        _run_latency_detection(path_manager, entity, entity_id, output)
        click.echo("\n🌙 Running Dark Hours Analysis...")
        _run_dark_hours_detection(path_manager, entity, entity_id, output)

    click.echo("\n✅ Diagnostics complete!")


def _run_fatigue_detection(path_manager, entity, entity_id, output):
    """Run creative fatigue detection."""
    import pandas as pd

    # Load data
    processed_path = path_manager.get_dataset_path("processed")
    if processed_path and (processed_path / "ad_daily_processed.csv").exists():
        df = pd.read_csv(processed_path / "ad_daily_processed.csv")
    else:
        raw_path = path_manager.get_raw_data_path("ad_daily_insights")
        if not raw_path:
            click.echo("❌ Ad daily data not found")
            return
        df = pd.read_csv(raw_path)

    # Initialize detector
    detector = FatigueDetector()

    # Detect fatigue per ad
    fatigued_ads = []
    for ad_id, ad_data in df.groupby("ad_id"):
        issues = detector.detect(ad_data, ad_id)
        if issues:
            fatigued_ads.extend(issues)

    # Report results
    click.echo(f"   Found {len(fatigued_ads)} fatigued ads")

    if fatigued_ads:
        total_loss = sum(issue.metrics.get("premium_loss", 0) for issue in fatigued_ads)
        click.echo(f"   Total premium loss: ${total_loss:.2f}")

        for issue in fatigued_ads[:5]:  # Show top 5
            click.echo(f"   - {issue.affected_entities[0]}: ${issue.metrics.get('premium_loss', 0):.2f}")


def _run_latency_detection(path_manager, entity, entity_id, output):
    """Run human latency detection."""
    import pandas as pd

    # Load hourly data
    processed_path = path_manager.get_dataset_path("processed")
    if processed_path and (processed_path / "adset_hourly_processed.csv").exists():
        df = pd.read_csv(processed_path / "adset_hourly_processed.csv")
    else:
        raw_path = path_manager.get_raw_data_path("adset_hourly_insights")
        if not raw_path:
            click.echo("❌ Adset hourly data not found")
            return
        df = pd.read_csv(raw_path)

    # Load status changes if available
    status_changes = None
    if processed_path and (processed_path / "status_changes.csv").exists():
        status_changes = pd.read_csv(processed_path / "status_changes.csv")

    # Initialize detector
    detector = LatencyDetector()

    # Detect latency per adset
    latency_issues = []
    for adset_id, adset_data in df.groupby("adset_id"):
        issues = detector.detect(adset_data, adset_id, status_changes)
        latency_issues.extend(issues)

    # Report results
    click.echo(f"   Found {len(latency_issues)} adsets with latency issues")

    if latency_issues:
        total_loss = sum(issue.metrics.get("total_loss", 0) for issue in latency_issues)
        avg_delay = sum(issue.metrics.get("avg_delay_hours", 0) for issue in latency_issues) / len(latency_issues)
        click.echo(f"   Total loss: ${total_loss:.2f}")
        click.echo(f"   Average response delay: {avg_delay:.1f} hours")


def _run_dark_hours_detection(path_manager, entity, entity_id, output):
    """Run dark hours analysis."""
    import pandas as pd

    # Load hourly data
    processed_path = path_manager.get_dataset_path("processed")
    if processed_path and (processed_path / "adset_hourly_processed.csv").exists():
        df = pd.read_csv(processed_path / "adset_hourly_processed.csv")
    else:
        raw_path = path_manager.get_raw_data_path("adset_hourly_insights")
        if not raw_path:
            click.echo("❌ Adset hourly data not found")
            return
        df = pd.read_csv(raw_path)

    # Initialize detector
    detector = DarkHoursDetector()

    # Analyze dark hours per adset
    dark_hours_issues = []
    for adset_id, adset_data in df.groupby("adset_id"):
        issues = detector.detect(adset_data, adset_id)
        dark_hours_issues.extend(issues)

    # Report results
    click.echo(f"   Found {len(dark_hours_issues)} adsets with dark hours")

    if dark_hours_issues:
        total_waste = sum(issue.metrics.get("monthly_waste_usd", 0) for issue in dark_hours_issues)
        click.echo(f"   Total monthly waste: ${total_waste:.2f}")

        for issue in dark_hours_issues[:3]:
            dead_zones = issue.metrics.get("dead_zones", [])
            click.echo(f"   - {issue.affected_entities[0]}: {', '.join(dead_zones)}")


# Register commands
def register_diagnoser_commands(cli):
    """Register diagnoser commands with CLI."""
    cli.add_command(diagnose, name="diagnose")
