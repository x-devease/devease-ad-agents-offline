"""Task orchestration with loss aversion and instant value delivery."""

from typing import List, Optional
from datetime import datetime
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from models import Lead, OutreachTask
from signals import SignalDetector, NegativeSignal, format_signal_for_outreach
from diagnostic import DiagnosticReport


class LossAversionTask(BaseModel):
    """Task optimized for loss aversion psychology."""
    task_id: str
    lead_domain: str
    signal_type: str
    severity: str
    estimated_loss: float
    hook: str  # Opening line that triggers loss aversion
    value_payload: str  # Instant value (diagnostic/insight)
    cta_type: str = Field(default="permission")  # "permission" or "wallet"
    cta_text: str
    talking_points: List[str]
    diagnostic_path: Optional[str] = None
    approved: bool = False
    executed: bool = False


class TaskOrchestrator:
    """Orchestrate tasks with empathetic expert tone."""

    def __init__(self, output_dir: str = "data/reports"):
        self.signal_detector = SignalDetector()
        self.diagnostic = DiagnosticReport(output_dir=output_dir)

    def generate_task_for_lead(
        self,
        lead: Lead,
        rank: int
    ) -> Optional[LossAversionTask]:
        """Generate loss aversion task for a lead."""

        if not lead.store or not lead.meta_ads:
            logger.warning(f"Missing data for {lead.domain}, skipping")
            return None

        # Detect negative signals
        signals = self.signal_detector.detect_all(lead.store, lead.meta_ads)

        if not signals:
            logger.info(f"No negative signals for {lead.domain}")
            return None

        # Use the highest severity signal
        primary_signal = signals[0]
        total_loss = sum(s.estimated_loss for s in signals)

        # Generate diagnostic report
        report_path = self.diagnostic.generate_report(lead, signals)

        # Build hook (loss aversion trigger)
        hook = self._build_loss_aversion_hook(lead, primary_signal)

        # Build value payload (instant value)
        value_payload = format_signal_for_outreach(primary_signal)

        # Build CTA
        cta_type, cta_text = self._build_cta(lead)

        # Generate talking points
        talking_points = self._build_talking_points(lead, signals)

        task = LossAversionTask(
            task_id=f"{datetime.now().strftime('%Y%m%d')}_{rank:03d}",
            lead_domain=lead.domain,
            signal_type=primary_signal.signal_type,
            severity=primary_signal.severity,
            estimated_loss=total_loss,
            hook=hook,
            value_payload=value_payload,
            cta_type=cta_type,
            cta_text=cta_text,
            talking_points=talking_points,
            diagnostic_path=report_path
        )

        logger.success(
            f"Generated task for {lead.domain}: "
            f"{primary_signal.signal_type} (${total_loss:.0f}/mo loss)"
        )

        return task

    def _build_loss_aversion_hook(
        self,
        lead: Lead,
        signal: NegativeSignal
    ) -> str:
        """Build opening hook that triggers loss aversion."""

        store_name = lead.domain.replace(".myshopify.com", "").replace("-", " ").title()

        hooks = {
            "critical": (
                f"Not trying to alarm you, but you're leaking "
                f"${signal.estimated_loss * 30:,.0f}/month on {store_name} "
                f"and the fix takes 20 minutes."
            ),
            "high": (
                f"Found {signal.signal_type.replace('_', ' ')} in your setup. "
                f"You're overpaying about ${signal.estimated_loss:,.0f}/day "
                f"for something that's fixable today."
            ),
            "medium": (
                f"Ran a diagnostic on {store_name}. "
                f"Found {signal.signal_type.replace('_', ' ')} "
                f"that's costing you ~${signal.estimated_loss:,.0f}/day."
            )
        }

        return hooks.get(signal.severity, hooks["medium"])

    def _build_cta(self, lead: Lead) -> tuple[str, str]:
        """Build call-to-action (no meeting requests)."""

        # Prefer Twitter if they have a handle
        if lead.contact and lead.contact.twitter_handle:
            return (
                "permission",
                f"Reply 'AUDIT' and I'll DM you the full diagnostic report. "
                f"No signup, no call, just the data."
            )

        # Email CTA
        return (
            "permission",
            "Reply 'send it' and I'll email you the 20-page PDF. "
            "Free. No catch. Just fix the bleeding."
        )

    def _build_talking_points(
        self,
        lead: Lead,
        signals: List[NegativeSignal]
    ) -> List[str]:
        """Build talking points for follow-up."""

        points = []

        # Add impact framing
        total_loss = sum(s.estimated_loss for s in signals)
        points.append(
            f"That's ${total_loss * 365:,.0f}/year in missed revenue. "
            f"Enough to hire 2-3 growth people."
        )

        # Add urgency for critical issues
        critical_signals = [s for s in signals if s.severity == "critical"]
        if critical_signals:
            points.append(
                "One of these is time-sensitive. "
                "Each day you wait = money left on table."
            )

        # Add social proof angle
        points.append(
            "I've seen this exact pattern 50+ times. "
            "The brands that fix it first usually win the niche."
        )

        # Add "easy win" framing
        if any(s.severity == "medium" for s in signals):
            points.append(
                "The medium-priority items? "
                "Those are 15-minute fixes. ROI is stupid."
            )

        return points

    def generate_tasks(
        self,
        leads: List[Lead],
        top_n: int = 20
    ) -> List[LossAversionTask]:
        """Generate tasks from top leads."""

        logger.info(f"Generating loss aversion tasks for top {top_n} leads...")

        tasks = []
        for i, lead in enumerate(leads[:top_n], 1):
            task = self.generate_task_for_lead(lead, i)
            if task:
                tasks.append(task)

        logger.success(f"Generated {len(tasks)} tasks with negative signals")
        return tasks

    def save_tasks_yaml(self, tasks: List[LossAversionTask], filepath: str = "data/tasks.yaml"):
        """Save tasks to YAML for dashboard."""

        import yaml

        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True)

        data = {
            "generated_at": datetime.now().isoformat(),
            "total_tasks": len(tasks),
            "total_estimated_loss": sum(t.estimated_loss for t in tasks),
            "tasks": [t.dict() for t in tasks]
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info(f"Saved {len(tasks)} tasks to {filepath}")


# Twitter thread generator
class TwitterOrchestrator:
    """Generate public proofing Twitter threads."""

    def __init__(self):
        self.signal_detector = SignalDetector()
        self.diagnostic = DiagnosticReport()

    def generate_thread(self, lead: Lead) -> Optional[List[str]]:
        """Generate Twitter thread for public proofing."""

        if not lead.store or not lead.meta_ads:
            return None

        signals = self.signal_detector.detect_all(lead.store, lead.meta_ads)

        if not signals:
            return None

        thread = self.diagnostic.generate_twitter_thread(lead, signals)

        logger.success(
            f"Generated {len(thread)} tweet thread for {lead.domain}"
        )

        return thread

    def generate_public_audit(
        self,
        lead: Lead,
        founder_handle: Optional[str] = None
    ) -> dict:
        """Generate public audit package."""

        signals = self.signal_detector.detect_all(lead.store, lead.meta_ads)

        if not signals:
            return None

        # Generate report
        report_path = self.diagnostic.generate_report(lead, signals)

        # Generate thread
        thread = self.diagnostic.generate_twitter_thread(lead, signals)

        # Build founder handle
        if not founder_handle:
            if lead.contact and lead.contact.twitter_handle:
                founder_handle = lead.contact.twitter_handle
            else:
                founder_handle = lead.domain.replace(".myshopify.com", "")

        total_loss = sum(s.estimated_loss for s in signals)

        return {
            "domain": lead.domain,
            "founder_handle": founder_handle,
            "total_loss": total_loss,
            "thread": thread,
            "report_path": report_path,
            "hook_tweet": thread[0] if thread else None,
            "cta_tweet": thread[-1] if thread else None
        }
