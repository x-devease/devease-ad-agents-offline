"""Human-in-the-loop dashboard with loss aversion messaging."""

from pathlib import Path
from datetime import datetime
from typing import List, Dict

import streamlit as st
import yaml
from loguru import logger

from orchestrate_tasks import LossAversionTask


class TaskManager:
    """Manage loss aversion tasks with YAML persistence."""

    def __init__(self, tasks_file: str = "data/tasks.yaml"):
        self.tasks_file = Path(tasks_file)
        self.tasks_file.parent.mkdir(exist_ok=True)
        self.tasks: List[LossAversionTask] = []
        self._load_tasks()

    def _load_tasks(self):
        """Load tasks from YAML."""
        if not self.tasks_file.exists():
            logger.info(f"No tasks file found at {self.tasks_file}")
            return

        with open(self.tasks_file) as f:
            data = yaml.safe_load(f)

        if data and "tasks" in data:
            self.tasks = [
                LossAversionTask(**task) for task in data["tasks"]
                if not task.get("executed")
            ]
            logger.info(f"Loaded {len(self.tasks)} pending tasks")

    def save_tasks(self):
        """Save tasks to YAML."""
        data = {
            "generated_at": datetime.now().isoformat(),
            "total_tasks": len(self.tasks),
            "total_estimated_loss": sum(t.estimated_loss for t in self.tasks),
            "tasks": [task.dict() for task in self.tasks]
        }

        with open(self.tasks_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info(f"Saved {len(self.tasks)} tasks")

    def approve_task(self, task_id: str):
        """Mark task as approved."""
        for task in self.tasks:
            if task.task_id == task_id:
                task.approved = True
                logger.info(f"Approved task {task_id}")
        self.save_tasks()

    def execute_task(self, task_id: str):
        """Mark task as executed."""
        for task in self.tasks:
            if task.task_id == task_id:
                task.executed = True
                logger.info(f"Executed task {task_id}")

        # Remove executed tasks from list
        self.tasks = [t for t in self.tasks if not t.executed]
        self.save_tasks()

    def get_pending_tasks(self) -> List[LossAversionTask]:
        """Get unapproved tasks."""
        return [t for t in self.tasks if not t.approved]

    def get_approved_tasks(self) -> List[LossAversionTask]:
        """Get approved but unexecuted tasks."""
        return [t for t in self.tasks if t.approved and not t.executed]


# Streamlit UI
def run_dashboard():
    """Run Streamlit dashboard."""

    st.set_page_config(
        page_title="Loss Aversion Lead Gen",
        layout="wide"
    )

    st.title("💸 Loss Aversion Outreach Dashboard")
    st.markdown("### Public Proofing + Instant Value Delivery")

    st.markdown("---")

    # Initialize task manager
    if "task_manager" not in st.session_state:
        st.session_state.task_manager = TaskManager()

    manager = st.session_state.task_manager

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Pending Review", "✅ Approved Queue", "📊 Analytics"])

    with tab1:
        st.header("Pending Tasks (Awaiting Approval)")

        pending = manager.get_pending_tasks()

        if not pending:
            st.info("No pending tasks. Run pipeline first.")
        else:
            # Sort by estimated loss (highest first)
            pending = sorted(pending, key=lambda t: t.estimated_loss, reverse=True)

            for task in pending:
                # Color code by severity
                severity_colors = {
                    "critical": "🚨",
                    "high": "⚠️",
                    "medium": "💡"
                }
                emoji = severity_colors.get(task.severity, "")

                with st.expander(
                    f"{emoji} {task.lead_domain} | "
                    f"${task.estimated_loss:,.0f}/mo loss | "
                    f"{task.signal_type.replace('_', ' ').title()}"
                ):
                    # Hook (loss aversion trigger)
                    st.subheader("🎣 Hook (Opening Line)")
                    st.info(task.hook)

                    # Value payload
                    st.subheader("💎 Instant Value")
                    st.success(task.value_payload)

                    # CTA
                    st.subheader("📢 Call-to-Action")
                    if task.cta_type == "permission":
                        st.info(f"**Ask for permission:** {task.cta_text}")
                    else:
                        st.warning(f"**Wallet address:** {task.cta_text}")

                    # Talking points
                    if task.talking_points:
                        st.subheader("💬 Talking Points")
                        for i, point in enumerate(task.talking_points, 1):
                            st.write(f"{i}. {point}")

                    # Diagnostic report link
                    if task.diagnostic_path:
                        st.subheader("📄 Diagnostic Report")
                        report_path = Path(task.diagnostic_path)
                        if report_path.exists():
                            st.success(f"✅ Report ready: `{report_path.name}`")
                            with open(report_path, 'r') as f:
                                report_content = f.read()
                            if st.button(f"Preview Report", key=f"preview_{task.task_id}"):
                                st.markdown("---")
                                st.markdown(report_content)
                        else:
                            st.warning(f"Report not found: {task.diagnostic_path}")

                    # Severity badge
                    st.markdown(f"**Severity:** `{task.severity.upper()}`")

                    # Action buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(
                            f"✅ Approve & Send",
                            key=f"approve_{task.task_id}",
                            type="primary"
                        ):
                            manager.approve_task(task.task_id)
                            st.success(f"Approved! Task moved to execution queue.")
                            st.rerun()
                    with col2:
                        if st.button(f"❌ Skip", key=f"skip_{task.task_id}"):
                            task.executed = True
                            manager.save_tasks()
                            st.rerun()

    with tab2:
        st.header("Approved Queue (Ready for Execution)")

        approved = manager.get_approved_tasks()

        if not approved:
            st.info("No approved tasks waiting.")
        else:
            # Group by CTA type
            permission_tasks = [t for t in approved if t.cta_type == "permission"]
            wallet_tasks = [t for t in approved if t.cta_type == "wallet"]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📧 Permission-based (Email/DM)")
                for task in permission_tasks:
                    st.markdown(f"- `{task.task_id}`: {task.lead_domain}")
                    st.caption(f"Hook: {task.hook[:80]}...")

            with col2:
                st.subheader("💰 Wallet-based (Crypto)")
                for task in wallet_tasks:
                    st.markdown(f"- `{task.task_id}`: {task.lead_domain}")
                    st.caption(f"Hook: {task.hook[:80]}...")

            st.markdown("---")

            if st.button("🚀 Execute All Approved", type="primary"):
                for task in approved:
                    manager.execute_task(task.task_id)
                    st.success(f"Executed {task.task_id}")

                st.rerun()

    with tab3:
        st.header("Analytics")

        all_tasks = manager.tasks

        if not all_tasks:
            st.info("No task data yet.")
        else:
            # Key metrics
            total_loss = sum(t.estimated_loss for t in all_tasks)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Tasks", len(all_tasks))
            with col2:
                st.metric("Pending", len(manager.get_pending_tasks()))
            with col3:
                st.metric("Approved", len(manager.get_approved_tasks()))
            with col4:
                st.metric("Total Addressable Loss", f"${total_loss:,.0f}/mo")

            # By severity
            st.subheader("Tasks by Severity")
            severity_counts = {}
            for task in all_tasks:
                severity_counts[task.severity] = severity_counts.get(task.severity, 0) + 1

            if severity_counts:
                st.bar_chart(severity_counts)

            # By signal type
            st.subheader("Tasks by Signal Type")
            signal_counts = {}
            for task in all_tasks:
                signal_counts[task.signal_type] = signal_counts.get(task.signal_type, 0) + 1

            if signal_counts:
                # Sort by count
                signal_counts = dict(sorted(signal_counts.items(), key=lambda x: x[1], reverse=True))
                st.bar_chart(signal_counts)

            # Top opportunities
            st.subheader("🎯 Top 10 Opportunities (Highest Loss)")
            top_tasks = sorted(all_tasks, key=lambda t: t.estimated_loss, reverse=True)[:10]

            for i, task in enumerate(top_tasks, 1):
                st.markdown(
                    f"**{i}. {task.lead_domain}** - "
                    f"${task.estimated_loss:,.0f}/mo | "
                    f"{task.severity.upper()} | "
                    f"{task.signal_type}"
                )
                st.caption(task.hook)


if __name__ == "__main__":
    run_dashboard()
