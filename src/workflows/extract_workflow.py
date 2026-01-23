"""
Feature extraction workflow.

Extracts and preprocesses features from raw Meta ads data.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.features import Aggregator, Joiner, Loader
from src.utils.customer_paths import (
    ensure_customer_dirs,
    get_customer_ad_features_path,
    get_customer_adset_features_path,
    get_customer_data_dir,
)
from src.workflows.base import Workflow, WorkflowResult

logger = logging.getLogger(__name__)


class ExtractWorkflow(Workflow):
    """
    Workflow for extracting features from raw Meta ads data.

    Performs:
    1. Load raw data (ad, adset, campaign, account levels)
    2. Join multi-level data
    3. Aggregate features
    4. Save enriched features
    5. Aggregate to adset-level (optional)
    """

    def __init__(
        self,
        config_path: str = "config/default/rules.yaml",
        preprocess: bool = True,
        normalize: bool = True,
        bucket: bool = True,
        aggregate_to_adset: bool = True,
        **kwargs,
    ):
        """Initialize extract workflow.

        Args:
            config_path: Path to configuration file.
            preprocess: Whether to apply preprocessing.
            normalize: Whether to normalize features.
            bucket: Whether to bucket features.
            aggregate_to_adset: Whether to aggregate to adset-level.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(config_path, **kwargs)
        self.preprocess = preprocess
        self.normalize = normalize
        self.bucket = bucket
        self.aggregate_to_adset = aggregate_to_adset

    def _process_customer(
        self,
        customer: str,
        platform: Optional[str],
        ad_file: Optional[str] = None,
        adset_file: Optional[str] = None,
        campaign_file: Optional[str] = None,
        account_file: Optional[str] = None,
        **kwargs,
    ) -> WorkflowResult:
        """
        Extract features for a single customer.

        Args:
            customer: Customer name.
            platform: Platform name.
            ad_file: Explicit path to ad-level CSV.
            adset_file: Explicit path to adset-level CSV.
            campaign_file: Explicit path to campaign-level CSV.
            account_file: Explicit path to account-level CSV.
            **kwargs: Additional arguments.

        Returns:
            WorkflowResult with extraction status.
        """
        try:
            # Ensure directories exist
            ensure_customer_dirs(customer, platform)

            # Get data directory
            data_dir = get_customer_data_dir(customer, platform)
            logger.info(f"Data directory: {data_dir}")

            # Load data
            data = self._load_data(
                data_dir, ad_file, adset_file, campaign_file, account_file
            )

            if "ad" not in data or data["ad"] is None:
                return WorkflowResult(
                    success=False,
                    customer=customer,
                    message="Ad-level data is required",
                )

            # Join multi-level data
            enriched_df = Joiner.join_all_levels(
                ad_df=data["ad"],
                account_df=data.get("account"),
                campaign_df=data.get("campaign"),
                adset_df=data.get("adset"),
            )

            # Aggregate features
            enriched_df = Aggregator.create_aggregated_features(
                enriched_df,
                preprocess=self.preprocess,
                normalize=self.normalize,
                bucket=self.bucket,
            )

            # Save ad-level features
            output_file = get_customer_ad_features_path(customer, platform)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            enriched_df.to_csv(output_file, index=False)
            logger.info(f"Saved ad-level features to: {output_file}")

            # Aggregate to adset-level
            if self.aggregate_to_adset:
                logger.info("Aggregating to adset-level...")
                adset_df = self._aggregate_to_adset(enriched_df)

                adset_output_file = get_customer_adset_features_path(customer, platform)
                adset_output_file.parent.mkdir(parents=True, exist_ok=True)
                adset_df.to_csv(adset_output_file, index=False)
                logger.info(f"Saved adset-level features to: {adset_output_file}")

                return WorkflowResult(
                    success=True,
                    customer=customer,
                    message="Feature extraction and aggregation complete",
                    data={
                        "ad_rows": len(enriched_df),
                        "adset_rows": len(adset_df),
                        "ad_features": len(enriched_df.columns),
                        "adset_features": len(adset_df.columns),
                    },
                )

            return WorkflowResult(
                success=True,
                customer=customer,
                message="Feature extraction complete",
                data={
                    "ad_rows": len(enriched_df),
                    "ad_features": len(enriched_df.columns),
                },
            )

        except FileNotFoundError as err:
            logger.error(f"File not found: {err}")
            return WorkflowResult(
                success=False,
                customer=customer,
                message=f"File not found: {err}",
                error=err,
            )
        except (ValueError, KeyError) as err:
            logger.error(f"Data error: {err}")
            return WorkflowResult(
                success=False,
                customer=customer,
                message=f"Data error: {err}",
                error=err,
            )
        except Exception as err:
            logger.exception(f"Unexpected error: {err}")
            return WorkflowResult(
                success=False,
                customer=customer,
                message=f"Unexpected error: {err}",
                error=err,
            )

    def _load_data(
        self,
        data_dir: str,
        ad_file: Optional[str] = None,
        adset_file: Optional[str] = None,
        campaign_file: Optional[str] = None,
        account_file: Optional[str] = None,
    ) -> dict:
        """Load data from files."""
        if ad_file or adset_file or campaign_file or account_file:
            return Loader.load_all_data(
                data_dir=str(data_dir),
                account_file=account_file,
                campaign_file=campaign_file,
                adset_file=adset_file,
                ad_file=ad_file,
            )
        else:
            return Loader.load_all_data(data_dir=str(data_dir))

    def _aggregate_to_adset(self, enriched_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate ad-level features to adset-level.

        This is a simplified version - the full implementation is in extract.py
        with all the recalculation logic.
        """
        # Import the full implementation from extract.py
        # For now, use a simple aggregation
        from src.cli.commands.extract import _aggregate_ad_to_adset

        return _aggregate_ad_to_adset(enriched_df)
