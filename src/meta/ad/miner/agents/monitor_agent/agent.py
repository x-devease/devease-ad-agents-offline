"""
Monitor Agent - Continuous Metrics Collection & Anomaly Detection

Objective: Monitor ad miner performance in real-time, detect anomalies,
and trigger evolution cycles when issues are detected.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time."""
    timestamp: str
    metrics: Dict[str, float]


@dataclass
class AnomalyAlert:
    """Alert for detected anomaly."""
    alert_id: str
    severity: str  # "low", "medium", "high", "critical"
    metric_name: str
    current_value: float
    expected_range: tuple[float, float]
    deviation_percent: float
    description: str
    timestamp: str
    resolved: bool = False


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    checks: Dict[str, bool]
    last_check: str
    details: Dict[str, Any] = field(default_factory=dict)


class MonitorAgent:
    """
    Continuous Metrics Collection & Anomaly Detection Agent.

    Responsibilities:
    - Collect metrics from ad miner pipeline
    - Detect anomalies using statistical analysis
    - Generate alerts for issues
    - Perform health checks
    - Trigger evolution cycles when needed
    """

    def __init__(
        self,
        alert_callback: Optional[Callable] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Monitor Agent.

        Args:
            alert_callback: Function to call when anomaly detected
            config: Agent configuration
        """
        self.config = config or {}
        self.alert_callback = alert_callback

        # Metric history
        self.metric_history: List[MetricSnapshot] = []
        self.max_history_size = 1000  # Keep last 1000 snapshots

        # Alert history
        self.alerts: List[AnomalyAlert] = []

        # Health check cache
        self.health_checks: Dict[str, HealthCheck] = {}

        # Baseline metrics (established from historical data)
        self.baselines: Dict[str, Dict[str, float]] = {
            "psychology_accuracy": {
                "min": 0.60,
                "max": 0.85,
                "target": 0.75,
                "stddev": 0.05,
            },
            "pattern_lift": {
                "min": 1.0,
                "max": 3.0,
                "target": 2.0,
                "stddev": 0.5,
            },
            "processing_time": {
                "min": 10.0,
                "max": 120.0,
                "target": 45.0,
                "stddev": 15.0,
            },
            "winner_precision": {
                "min": 0.65,
                "max": 0.90,
                "target": 0.80,
                "stddev": 0.05,
            },
        }

        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            "deviation_percent": 20.0,  # Alert if 20% deviation from baseline
            "consecutive_violations": 3,  # Alert if 3 consecutive violations
            "min_samples": 5,  # Need at least 5 samples before detecting anomalies
        }

        logger.info("Monitor Agent: Initialized")
        logger.info(f"  Monitoring {len(self.baselines)} metrics")
        logger.info(f"  Anomaly threshold: {self.anomaly_thresholds['deviation_percent']}% deviation")

    def collect_metrics(
        self,
        metrics: Dict[str, float],
    ) -> MetricSnapshot:
        """
        Collect metrics snapshot.

        Args:
            metrics: Dictionary of metric names to values

        Returns:
            MetricSnapshot: Collected snapshot
        """
        snapshot = MetricSnapshot(
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
        )

        self.metric_history.append(snapshot)

        # Trim history if needed
        if len(self.metric_history) > self.max_history_size:
            self.metric_history = self.metric_history[-self.max_history_size:]

        logger.debug(f"Collected metrics: {len(metrics)} metrics")

        # Check for anomalies
        self._check_anomalies(metrics)

        return snapshot

    def detect_anomaly(self) -> Optional[AnomalyAlert]:
        """
        Check for anomalies in recent metrics.

        Returns:
            AnomalyAlert if anomaly detected, None otherwise
        """
        if len(self.metric_history) < self.anomaly_thresholds["min_samples"]:
            logger.debug("Not enough data for anomaly detection")
            return None

        # Get latest snapshot
        latest = self.metric_history[-1]

        # Check each metric
        for metric_name, current_value in latest.metrics.items():
            if metric_name not in self.baselines:
                continue

            baseline = self.baselines[metric_name]
            target = baseline["target"]

            # Calculate deviation
            deviation_percent = abs((current_value - target) / target) * 100

            # Check if exceeds threshold
            if deviation_percent > self.anomaly_thresholds["deviation_percent"]:
                alert = AnomalyAlert(
                    alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    severity=self._assess_severity(deviation_percent, metric_name),
                    metric_name=metric_name,
                    current_value=current_value,
                    expected_range=(baseline["min"], baseline["max"]),
                    deviation_percent=deviation_percent,
                    description=f"{metric_name} is {deviation_percent:.1f}% away from target (current: {current_value:.3f}, target: {target:.3f})",
                    timestamp=datetime.now().isoformat(),
                )

                self.alerts.append(alert)

                # Trigger callback if provided
                if self.alert_callback:
                    self.alert_callback(alert)

                logger.warning(f"Anomaly detected: {alert.description}")
                return alert

        return None

    def _check_anomalies(self, metrics: Dict[str, float]):
        """Internal method to check for anomalies after collecting metrics."""

        # Check for critical values
        for metric_name, value in metrics.items():
            if metric_name in self.baselines:
                baseline = self.baselines[metric_name]

                # Check if outside acceptable range
                if value < baseline["min"] or value > baseline["max"]:
                    logger.warning(f"Metric {metric_name} outside acceptable range: {value:.3f}")

    def _assess_severity(self, deviation_percent: float, metric_name: str) -> str:
        """Assess severity of anomaly based on deviation."""

        if deviation_percent > 50:
            return "critical"
        elif deviation_percent > 30:
            return "high"
        elif deviation_percent > 20:
            return "medium"
        else:
            return "low"

    def health_check(self, component: str) -> HealthCheck:
        """
        Perform health check on a component.

        Args:
            component: Component name to check

        Returns:
            HealthCheck: Health check result
        """
        checks = {}
        details = {}

        if component == "pattern_mining":
            # Check pattern mining health
            checks["metrics_available"] = len(self.metric_history) > 0
            checks["recent_anomalies"] = len([a for a in self.alerts if not a.resolved]) < 3

            if self.metric_history:
                latest = self.metric_history[-1].metrics
                details["latest_metrics"] = latest
                checks["accuracy_acceptable"] = latest.get("psychology_accuracy", 0) > 0.60
                checks["lift_acceptable"] = latest.get("pattern_lift", 0) > 1.0

        elif component == "memory_agent":
            # Check memory agent health
            checks["memory_available"] = True  # Placeholder
            checks["search_functional"] = True  # Placeholder

        elif component == "pipeline":
            # Check pipeline health
            checks["processing_time_acceptable"] = True
            checks["no_errors"] = True

        # Determine overall status
        if all(checks.values()):
            status = "healthy"
        elif any(checks.values()):
            status = "degraded"
        else:
            status = "unhealthy"

        health_check = HealthCheck(
            component=component,
            status=status,
            checks=checks,
            last_check=datetime.now().isoformat(),
            details=details,
        )

        self.health_checks[component] = health_check

        return health_check

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status.

        Returns:
            Dict with system status summary
        """
        # Perform health checks on all components
        components = ["pattern_mining", "memory_agent", "pipeline"]
        health_results = {c: self.health_check(c) for c in components}

        # Count alerts by severity
        unresolved_alerts = [a for a in self.alerts if not a.resolved]
        alert_counts = {
            "critical": len([a for a in unresolved_alerts if a.severity == "critical"]),
            "high": len([a for a in unresolved_alerts if a.severity == "high"]),
            "medium": len([a for a in unresolved_alerts if a.severity == "medium"]),
            "low": len([a for a in unresolved_alerts if a.severity == "low"]),
        }

        # Overall status
        if alert_counts["critical"] > 0:
            overall_status = "critical"
        elif alert_counts["high"] > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        return {
            "status": overall_status,
            "components": health_results,
            "alerts": {
                "total_unresolved": len(unresolved_alerts),
                "by_severity": alert_counts,
            },
            "metrics": {
                "history_size": len(self.metric_history),
                "latest": self.metric_history[-1].metrics if self.metric_history else {},
            },
        }

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Mark an alert as resolved.

        Args:
            alert_id: Alert ID to resolve

        Returns:
            True if alert found and resolved
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info(f"Resolved alert: {alert_id}")
                return True

        logger.warning(f"Alert not found: {alert_id}")
        return False

    def update_baseline(
        self,
        metric_name: str,
        min_value: float,
        max_value: float,
        target: float,
        stddev: float = 0.0,
    ):
        """
        Update baseline for a metric.

        Args:
            metric_name: Name of metric
            min_value: Minimum acceptable value
            max_value: Maximum acceptable value
            target: Target value
            stddev: Standard deviation (optional)
        """
        self.baselines[metric_name] = {
            "min": min_value,
            "max": max_value,
            "target": target,
            "stddev": stddev,
        }

        logger.info(f"Updated baseline for {metric_name}: target={target:.3f}, range=[{min_value:.3f}, {max_value:.3f}]")

    def get_metric_history(
        self,
        metric_name: str,
        last_n: Optional[int] = None,
    ) -> List[float]:
        """
        Get history for a specific metric.

        Args:
            metric_name: Name of metric
            last_n: Only return last N values (optional)

        Returns:
            List of metric values
        """
        values = []

        for snapshot in self.metric_history:
            if metric_name in snapshot.metrics:
                values.append(snapshot.metrics[metric_name])

        if last_n:
            values = values[-last_n:]

        return values

    def get_metric_statistics(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.

        Args:
            metric_name: Name of metric

        Returns:
            Dict with statistics (mean, median, stddev, min, max)
        """
        values = self.get_metric_history(metric_name)

        if not values:
            return {}

        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary."""
        unresolved_alerts = [a for a in self.alerts if not a.resolved]

        return {
            "metric_history_size": len(self.metric_history),
            "total_alerts": len(self.alerts),
            "unresolved_alerts": len(unresolved_alerts),
            "monitored_metrics": list(self.baselines.keys()),
            "config": self.config,
        }
