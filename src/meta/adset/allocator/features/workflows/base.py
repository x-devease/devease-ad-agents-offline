"""
Base workflow class for all budget allocation operations.

Provides common functionality for:
- Configuration loading
- Customer discovery
- Progress tracking
- Error handling
- Result reporting
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.utils.customer_paths import get_all_customers
from src.utils.logger_config import setup_logging

logger = setup_logging()


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""

    success: bool
    customer: str
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "customer": self.customer,
            "message": self.message,
            "data": self.data,
            "error": str(self.error) if self.error else None,
        }


@dataclass
class WorkflowMetrics:
    """Metrics tracked during workflow execution."""

    total_customers: int = 0
    successful_customers: int = 0
    failed_customers: int = 0
    skipped_customers: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get workflow duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def success_rate(self) -> float:
        """Get success rate (0-1)."""
        if self.total_customers == 0:
            return 0.0
        return self.successful_customers / self.total_customers


class Workflow(ABC):
    """
    Base class for all workflows.

    Provides common functionality for processing customers,
    tracking progress, and reporting results.
    """

    def __init__(
        self,
        config_path: str = "config/adset/allocator/rules.yaml",
        verbose: bool = True,
    ):
        """Initialize workflow.

        Args:
            config_path: Path to configuration file.
            verbose: Whether to print progress messages.
        """
        self.config_path = config_path
        self.verbose = verbose
        self.metrics = WorkflowMetrics()

    def run(
        self,
        customer: Optional[str] = None,
        platform: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, WorkflowResult]:
        """
        Run workflow for one or all customers.

        Args:
            customer: Customer name. If None, processes all customers.
            platform: Platform name (e.g., 'meta', 'google').
            **kwargs: Additional workflow-specific arguments.

        Returns:
            Dictionary mapping customer names to workflow results.
        """
        import time

        self.metrics.start_time = time.time()

        if customer:
            results = self._run_single_customer(customer, platform, **kwargs)
        else:
            results = self._run_all_customers(platform, **kwargs)

        self.metrics.end_time = time.time()
        self._print_summary(results)

        return results

    def _run_single_customer(
        self,
        customer: str,
        platform: Optional[str],
        **kwargs,
    ) -> Dict[str, WorkflowResult]:
        """Run workflow for a single customer."""
        if self.verbose:
            self._print_header(f"Processing Customer: {customer}")

        result = self._process_customer(customer, platform, **kwargs)

        self.metrics.total_customers = 1
        if result.success:
            self.metrics.successful_customers = 1
        else:
            self.metrics.failed_customers = 1

        return {customer: result}

    def _run_all_customers(
        self,
        platform: Optional[str],
        **kwargs,
    ) -> Dict[str, WorkflowResult]:
        """Run workflow for all customers in config."""
        customers = get_all_customers(self.config_path)

        if not customers:
            logger.warning("No customers found in configuration")
            return {}

        if self.verbose:
            self._print_header(f"Processing {len(customers)} Customer(s)")
            logger.info(f"Customers: {', '.join(customers)}")

        results = {}
        for customer in customers:
            if self.verbose:
                logger.info("")
                logger.info("-" * 70)
                logger.info(f"Processing: {customer}")
                logger.info("-" * 70)

            result = self._process_customer(customer, platform, **kwargs)
            results[customer] = result

            self.metrics.total_customers += 1
            if result.success:
                self.metrics.successful_customers += 1
            else:
                self.metrics.failed_customers += 1

        return results

    @abstractmethod
    def _process_customer(
        self,
        customer: str,
        platform: Optional[str],
        **kwargs,
    ) -> WorkflowResult:
        """
        Process a single customer.

        Must be implemented by subclasses.

        Args:
            customer: Customer name.
            platform: Platform name.
            **kwargs: Additional workflow-specific arguments.

        Returns:
            WorkflowResult containing execution status and data.
        """
        raise NotImplementedError("Subclasses must implement _process_customer")

    def _print_header(self, title: str):
        """Print workflow header."""
        logger.info("=" * 70)
        logger.info(title)
        logger.info("=" * 70)

    def _print_summary(self, results: Dict[str, WorkflowResult]):
        """Print workflow execution summary."""
        if not self.verbose:
            return

        logger.info("")
        logger.info("=" * 70)
        logger.info("WORKFLOW SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total customers: {self.metrics.total_customers}")
        logger.info(
            f"Successful: {self.metrics.successful_customers} "
            f"({self.metrics.success_rate:.1%})"
        )
        logger.info(f"Failed: {self.metrics.failed_customers}")

        if self.metrics.duration_seconds:
            logger.info(f"Duration: {self.metrics.duration_seconds:.2f}s")

        # List failed customers
        failed = [
            customer for customer, result in results.items() if not result.success
        ]
        if failed:
            logger.warning(f"Failed customers: {', '.join(failed)}")

        logger.info("=" * 70)

    def get_customer_config(self, customer: str):
        """
        Load configuration for a specific customer.

        Args:
            customer: Customer name.

        Returns:
            Configuration object.
        """
        from src.meta.adset.allocator.utils.parser import Parser

        try:
            config = Parser(
                config_path=self.config_path, customer_name=customer, platform="meta"
            )
            return config
        except FileNotFoundError as err:
            logger.error(f"Config not found: {err}")
            raise
        except ValueError as err:
            logger.error(f"Config error: {err}")
            raise
