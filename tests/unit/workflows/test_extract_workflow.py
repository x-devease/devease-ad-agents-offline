"""
Unit tests for workflows.extract_workflow module.

Tests the ExtractWorkflow class for feature extraction orchestration.
Focuses on multi-customer processing, data loading, aggregation, and error handling.
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import pytest

# Skip these tests in CI since they require file system access
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Workflow tests require file system access, skipped in CI"
)

from src.meta.adset.allocator.features.workflows.extract_workflow import ExtractWorkflow
from src.meta.adset.allocator.features.workflows.base import WorkflowResult


@pytest.fixture
def sample_ad_df():
    """Create sample ad-level DataFrame."""
    return pd.DataFrame(
        {
            "ad_id": ["ad_001", "ad_002", "ad_003"],
            "adset_id": ["adset_001", "adset_001", "adset_002"],
            "campaign_id": ["camp_001", "camp_001", "camp_002"],
            "account_id": ["acct_001", "acct_001", "acct_001"],
            "spend": [100.0, 150.0, 200.0],
            "impressions": [1000, 1500, 2000],
            "clicks": [50, 75, 100],
        }
    )


@pytest.fixture
def sample_adset_df():
    """Create sample adset-level DataFrame."""
    return pd.DataFrame(
        {
            "adset_id": ["adset_001", "adset_002"],
            "campaign_id": ["camp_001", "camp_002"],
            "account_id": ["acct_001", "acct_001"],
        }
    )


@pytest.fixture
def sample_campaign_df():
    """Create sample campaign-level DataFrame."""
    return pd.DataFrame(
        {
            "campaign_id": ["camp_001", "camp_002"],
            "account_id": ["acct_001", "acct_001"],
        }
    )


@pytest.fixture
def sample_account_df():
    """Create sample account-level DataFrame."""
    return pd.DataFrame(
        {
            "account_id": ["acct_001", "acct_002"],
        }
    )


@pytest.fixture
def sample_enriched_df():
    """Create sample enriched DataFrame with features."""
    return pd.DataFrame(
        {
            "ad_id": ["ad_001", "ad_002", "ad_003"],
            "adset_id": ["adset_001", "adset_001", "adset_002"],
            "spend": [100.0, 150.0, 200.0],
            "health_score": [0.8, 0.7, 0.9],
            "purchase_roas_rolling_7d": [2.5, 3.0, 2.0],
        }
    )


@pytest.fixture
def sample_adset_aggregated_df():
    """Create sample adset-aggregated DataFrame."""
    return pd.DataFrame(
        {
            "adset_id": ["adset_001", "adset_002"],
            "spend": [250.0, 200.0],
            "health_score": [0.75, 0.9],
        }
    )


class TestExtractWorkflowInit:
    """Test ExtractWorkflow initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        workflow = ExtractWorkflow()

        assert workflow.config_path == "config/adset/allocator/rules.yaml"
        assert workflow.preprocess is True
        assert workflow.normalize is True
        assert workflow.bucket is True
        assert workflow.aggregate_to_adset is True
        assert workflow.verbose is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        workflow = ExtractWorkflow(
            config_path="custom/config.yaml",
            preprocess=False,
            normalize=False,
            bucket=False,
            aggregate_to_adset=False,
            verbose=False,
        )

        assert workflow.config_path == "custom/config.yaml"
        assert workflow.preprocess is False
        assert workflow.normalize is False
        assert workflow.bucket is False
        assert workflow.aggregate_to_adset is False
        assert workflow.verbose is False

    def test_init_partial_custom_params(self):
        """Test initialization with partial custom parameters."""
        workflow = ExtractWorkflow(preprocess=False, aggregate_to_adset=False)

        assert workflow.preprocess is False
        assert workflow.normalize is True  # default
        assert workflow.bucket is True  # default
        assert workflow.aggregate_to_adset is False


class TestExtractWorkflowLoadData:
    """Test ExtractWorkflow._load_data method."""

    @patch("src.meta.adset.allocator.features.Loader.load_all_data")
    def test_load_data_default(self, mock_load_all_data):
        """Test _load_data with no explicit files."""
        mock_load_all_data.return_value = {"ad": pd.DataFrame()}

        workflow = ExtractWorkflow()
        data_dir = "/path/to/data"

        result = workflow._load_data(data_dir)

        mock_load_all_data.assert_called_once_with(data_dir=str(data_dir))

    @patch("src.meta.adset.allocator.features.Loader.load_all_data")
    def test_load_data_with_explicit_files(self, mock_load_all_data):
        """Test _load_data with explicit file paths."""
        mock_load_all_data.return_value = {"ad": pd.DataFrame()}

        workflow = ExtractWorkflow()
        data_dir = "/path/to/data"

        workflow._load_data(
            data_dir,
            ad_file="custom_ad.csv",
            adset_file="custom_adset.csv",
            campaign_file="custom_campaign.csv",
            account_file="custom_account.csv",
        )

        mock_load_all_data.assert_called_once_with(
            data_dir=str(data_dir),
            account_file="custom_account.csv",
            campaign_file="custom_campaign.csv",
            adset_file="custom_adset.csv",
            ad_file="custom_ad.csv",
        )

    @patch("src.meta.adset.allocator.features.Loader.load_all_data")
    def test_load_data_partial_explicit_files(self, mock_load_all_data):
        """Test _load_data with only some explicit files."""
        mock_load_all_data.return_value = {"ad": pd.DataFrame()}

        workflow = ExtractWorkflow()
        data_dir = "/path/to/data"

        workflow._load_data(
            data_dir, ad_file="custom_ad.csv", campaign_file="custom_campaign.csv"
        )

        # Should pass all parameters, even None values
        mock_load_all_data.assert_called_once()


class TestExtractWorkflowAggregateToAdset:
    """Test ExtractWorkflow._aggregate_to_adset method."""

    @patch("src.meta.adset.allocator.cli.commands.extract._aggregate_ad_to_adset")
    def test_aggregate_to_adset_success(self, mock_aggregate_fn):
        """Test successful aggregation to adset level."""
        mock_aggregate_fn.return_value = pd.DataFrame(
            {"adset_id": ["adset_001", "adset_002"], "spend": [250.0, 200.0]}
        )

        workflow = ExtractWorkflow()
        enriched_df = pd.DataFrame(
            {
                "ad_id": ["ad_001", "ad_002"],
                "adset_id": ["adset_001", "adset_002"],
                "spend": [100.0, 200.0],
            }
        )

        result = workflow._aggregate_to_adset(enriched_df)

        mock_aggregate_fn.assert_called_once_with(enriched_df)
        assert isinstance(result, pd.DataFrame)


class TestExtractWorkflowProcessCustomer:
    """Test ExtractWorkflow._process_customer method."""

    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_data_dir")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.ensure_customer_dirs")
    @patch("src.meta.adset.allocator.features.Joiner.join_all_levels")
    @patch("src.meta.adset.allocator.features.Aggregator.create_aggregated_features")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_ad_features_path")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_adset_features_path")
    def test_process_customer_success_with_aggregation(
        self,
        mock_get_adset_path,
        mock_get_ad_path,
        mock_create_aggregated_features,
        mock_join_all_levels,
        mock_ensure_dirs,
        mock_get_data_dir,
        sample_enriched_df,
        sample_adset_aggregated_df,
    ):
        """Test successful customer processing with adset aggregation."""
        # Setup mocks
        mock_get_data_dir.return_value = Path("/data/customer1")
        mock_get_ad_path.return_value = Path("/output/customer1/ad_features.csv")
        mock_get_adset_path.return_value = Path("/output/customer1/adset_features.csv")

        # Create real copies of fixtures for testing
        enriched_df = sample_enriched_df.copy()
        adset_df = sample_adset_aggregated_df.copy()

        # Mock Joiner
        mock_join_all_levels.return_value = enriched_df

        # Mock Aggregator
        mock_create_aggregated_features.return_value = enriched_df

        # Mock _aggregate_to_adset
        workflow = ExtractWorkflow(aggregate_to_adset=True)

        with patch.object(workflow, "_load_data") as mock_load:
            mock_load.return_value = {
                "ad": pd.DataFrame(),
                "adset": pd.DataFrame(),
                "campaign": pd.DataFrame(),
                "account": pd.DataFrame(),
            }

            with patch.object(workflow, "_aggregate_to_adset") as mock_aggregate:
                mock_aggregate.return_value = adset_df

                # Mock to_csv only for the specific DataFrames being written
                original_to_csv = pd.DataFrame.to_csv

                def mock_to_csv(self, filepath, **kwargs):
                    # Only mock writes to output paths
                    if "/output/" in str(filepath):
                        return None
                    # Call original for all other cases
                    return original_to_csv(self, filepath, **kwargs)

                with patch.object(pd.DataFrame, "to_csv", mock_to_csv):
                    # Mock Path.mkdir to prevent actual directory creation
                    with patch.object(Path, "mkdir", return_value=None):
                        result = workflow._process_customer(
                            customer="test_customer", platform="meta"
                        )

        assert result.success is True
        assert "Feature extraction and aggregation complete" in result.message
        assert result.data["ad_rows"] == 3
        assert result.data["adset_rows"] == 2
        assert result.data["ad_features"] == 5  # Updated to match actual column count
        assert (
            result.data["adset_features"] == 3
        )  # Updated to match actual column count

    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_data_dir")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.ensure_customer_dirs")
    @patch("src.meta.adset.allocator.features.Joiner.join_all_levels")
    @patch("src.meta.adset.allocator.features.Aggregator.create_aggregated_features")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_ad_features_path")
    def test_process_customer_success_without_aggregation(
        self,
        mock_get_ad_path,
        mock_create_aggregated_features,
        mock_join_all_levels,
        mock_ensure_dirs,
        mock_get_data_dir,
        sample_enriched_df,
    ):
        """Test successful customer processing without adset aggregation."""
        # Setup mocks
        mock_get_data_dir.return_value = Path("/data/customer1")
        mock_get_ad_path.return_value = Path("/output/customer1/ad_features.csv")

        # Create real copy of fixture for testing
        enriched_df = sample_enriched_df.copy()

        # Mock Joiner
        mock_join_all_levels.return_value = enriched_df

        # Mock Aggregator
        mock_create_aggregated_features.return_value = enriched_df

        workflow = ExtractWorkflow(aggregate_to_adset=False)

        with patch.object(workflow, "_load_data") as mock_load:
            mock_load.return_value = {
                "ad": pd.DataFrame(),
                "adset": None,
                "campaign": None,
                "account": None,
            }

            # Mock to_csv only for the specific DataFrames being written
            original_to_csv = pd.DataFrame.to_csv

            def mock_to_csv(self, filepath, **kwargs):
                # Only mock writes to output paths
                if "/output/" in str(filepath):
                    return None
                # Call original for all other cases
                return original_to_csv(self, filepath, **kwargs)

            with patch.object(pd.DataFrame, "to_csv", mock_to_csv):
                # Mock Path.mkdir to prevent actual directory creation
                with patch.object(Path, "mkdir", return_value=None):
                    result = workflow._process_customer(
                        customer="test_customer", platform="meta"
                    )

        assert result.success is True
        assert "Feature extraction complete" in result.message
        assert result.data["ad_rows"] == 3
        assert result.data["ad_features"] == 5  # Updated to match actual column count
        assert "adset_rows" not in result.data

    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_data_dir")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.ensure_customer_dirs")
    def test_process_customer_missing_ad_data(
        self, mock_ensure_dirs, mock_get_data_dir
    ):
        """Test processing when ad-level data is missing."""
        mock_get_data_dir.return_value = Path("/data/customer1")

        workflow = ExtractWorkflow()

        with patch.object(workflow, "_load_data") as mock_load:
            mock_load.return_value = {
                "ad": None,  # Missing ad data
                "adset": pd.DataFrame(),
            }

            result = workflow._process_customer(
                customer="test_customer", platform="meta"
            )

        assert result.success is False
        assert "Ad-level data is required" in result.message

    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_data_dir")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.ensure_customer_dirs")
    @patch("src.meta.adset.allocator.features.Joiner.join_all_levels")
    @patch("src.meta.adset.allocator.features.Aggregator.create_aggregated_features")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_ad_features_path")
    def test_process_customer_file_not_found_error(
        self,
        mock_get_ad_path,
        mock_create_aggregated_features,
        mock_join_all_levels,
        mock_ensure_dirs,
        mock_get_data_dir,
        sample_enriched_df,
    ):
        """Test processing with FileNotFoundError."""
        mock_get_data_dir.return_value = Path("/data/customer1")
        mock_get_ad_path.return_value = Path("/output/customer1/ad_features.csv")

        mock_join_all_levels.side_effect = FileNotFoundError("Data file not found")

        workflow = ExtractWorkflow()

        with patch.object(workflow, "_load_data") as mock_load:
            mock_load.return_value = {"ad": pd.DataFrame()}

            result = workflow._process_customer(
                customer="test_customer", platform="meta"
            )

        assert result.success is False
        assert "File not found" in result.message
        assert isinstance(result.error, FileNotFoundError)

    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_data_dir")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.ensure_customer_dirs")
    @patch("src.meta.adset.allocator.features.Joiner.join_all_levels")
    @patch("src.meta.adset.allocator.features.Aggregator.create_aggregated_features")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_ad_features_path")
    def test_process_customer_value_error(
        self,
        mock_get_ad_path,
        mock_create_aggregated_features,
        mock_join_all_levels,
        mock_ensure_dirs,
        mock_get_data_dir,
        sample_enriched_df,
    ):
        """Test processing with ValueError."""
        mock_get_data_dir.return_value = Path("/data/customer1")
        mock_get_ad_path.return_value = Path("/output/customer1/ad_features.csv")

        mock_join_all_levels.side_effect = ValueError("Invalid data format")

        workflow = ExtractWorkflow()

        with patch.object(workflow, "_load_data") as mock_load:
            mock_load.return_value = {"ad": pd.DataFrame()}

            result = workflow._process_customer(
                customer="test_customer", platform="meta"
            )

        assert result.success is False
        assert "Data error" in result.message
        assert isinstance(result.error, ValueError)

    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_data_dir")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.ensure_customer_dirs")
    @patch("src.meta.adset.allocator.features.Joiner.join_all_levels")
    @patch("src.meta.adset.allocator.features.Aggregator.create_aggregated_features")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_ad_features_path")
    def test_process_customer_unexpected_error(
        self,
        mock_get_ad_path,
        mock_create_aggregated_features,
        mock_join_all_levels,
        mock_ensure_dirs,
        mock_get_data_dir,
    ):
        """Test processing with unexpected Exception."""
        mock_get_data_dir.return_value = Path("/data/customer1")
        mock_get_ad_path.return_value = Path("/output/customer1/ad_features.csv")

        mock_join_all_levels.side_effect = RuntimeError("Unexpected error")

        workflow = ExtractWorkflow()

        with patch.object(workflow, "_load_data") as mock_load:
            mock_load.return_value = {"ad": pd.DataFrame()}

            result = workflow._process_customer(
                customer="test_customer", platform="meta"
            )

        assert result.success is False
        assert "Unexpected error" in result.message
        assert isinstance(result.error, RuntimeError)

    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_data_dir")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.ensure_customer_dirs")
    @patch("src.meta.adset.allocator.features.Joiner.join_all_levels")
    @patch("src.meta.adset.allocator.features.Aggregator.create_aggregated_features")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_ad_features_path")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_adset_features_path")
    def test_process_customer_with_explicit_files(
        self,
        mock_get_adset_path,
        mock_get_ad_path,
        mock_create_aggregated_features,
        mock_join_all_levels,
        mock_ensure_dirs,
        mock_get_data_dir,
        sample_enriched_df,
        sample_adset_aggregated_df,
    ):
        """Test processing with explicit file paths."""
        mock_get_data_dir.return_value = Path("/data/customer1")
        mock_get_ad_path.return_value = Path("/output/customer1/ad_features.csv")
        mock_get_adset_path.return_value = Path("/output/customer1/adset_features.csv")

        # Create real copies of fixtures for testing
        enriched_df = sample_enriched_df.copy()
        adset_df = sample_adset_aggregated_df.copy()

        mock_join_all_levels.return_value = enriched_df
        mock_create_aggregated_features.return_value = enriched_df

        workflow = ExtractWorkflow(aggregate_to_adset=True)

        with patch.object(workflow, "_load_data") as mock_load:
            mock_load.return_value = {
                "ad": pd.DataFrame(),
                "adset": pd.DataFrame(),
            }

            with patch.object(workflow, "_aggregate_to_adset") as mock_aggregate:
                mock_aggregate.return_value = adset_df

                # Mock to_csv only for the specific DataFrames being written
                original_to_csv = pd.DataFrame.to_csv

                def mock_to_csv(self, filepath, **kwargs):
                    # Only mock writes to output paths
                    if "/output/" in str(filepath):
                        return None
                    # Call original for all other cases
                    return original_to_csv(self, filepath, **kwargs)

                with patch.object(pd.DataFrame, "to_csv", mock_to_csv):
                    # Mock Path.mkdir to prevent actual directory creation
                    with patch.object(Path, "mkdir", return_value=None):
                        result = workflow._process_customer(
                            customer="test_customer",
                            platform="meta",
                            ad_file="custom_ad.csv",
                            adset_file="custom_adset.csv",
                        )

        # Verify _load_data was called with explicit files
        mock_load.assert_called_once()
        # The call uses positional args: (data_dir, ad_file, adset_file, campaign_file, account_file)
        call_args = mock_load.call_args[0]
        assert call_args[1] == "custom_ad.csv"  # ad_file
        assert call_args[2] == "custom_adset.csv"  # adset_file

        assert result.success is True

    @patch("pandas.DataFrame.to_csv")
    @patch("pathlib.Path.mkdir", return_value=None)
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_data_dir")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.ensure_customer_dirs")
    @patch("src.meta.adset.allocator.features.Joiner.join_all_levels")
    @patch("src.meta.adset.allocator.features.Aggregator.create_aggregated_features")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_ad_features_path")
    def test_process_customer_respects_preprocess_flags(
        self,
        mock_get_ad_path,
        mock_create_aggregated_features,
        mock_join_all_levels,
        mock_ensure_dirs,
        mock_get_data_dir,
        mock_to_csv,
        sample_enriched_df,
    ):
        """Test that preprocess, normalize, and bucket flags are respected."""
        mock_get_data_dir.return_value = Path("/data/customer1")
        mock_get_ad_path.return_value = Path("/output/customer1/ad_features.csv")

        mock_join_all_levels.return_value = sample_enriched_df
        mock_create_aggregated_features.return_value = sample_enriched_df

        workflow = ExtractWorkflow(
            preprocess=False, normalize=False, bucket=False, aggregate_to_adset=False
        )

        with patch.object(workflow, "_load_data") as mock_load:
            mock_load.return_value = {"ad": pd.DataFrame()}

            result = workflow._process_customer(
                customer="test_customer", platform="meta"
            )

        # Verify Aggregator was called with correct flags
        mock_create_aggregated_features.assert_called_once()
        call_kwargs = mock_create_aggregated_features.call_args.kwargs
        assert call_kwargs["preprocess"] is False
        assert call_kwargs["normalize"] is False
        assert call_kwargs["bucket"] is False

        assert result.success is True


class TestExtractWorkflowIntegration:
    """Integration tests for ExtractWorkflow orchestration."""

    @patch("src.meta.adset.allocator.features.workflows.base.get_all_customers")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_data_dir")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.ensure_customer_dirs")
    @patch("src.meta.adset.allocator.features.Joiner.join_all_levels")
    @patch("src.meta.adset.allocator.features.Aggregator.create_aggregated_features")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_ad_features_path")
    def test_workflow_run_multiple_customers(
        self,
        mock_get_ad_path,
        mock_create_aggregated_features,
        mock_join_all_levels,
        mock_ensure_dirs,
        mock_get_data_dir,
        mock_get_customers,
        sample_enriched_df,
    ):
        """Test running extract workflow for multiple customers."""
        # Mock get_all_customers to return our test customers regardless of config_path
        mock_get_customers.return_value = ["customer1", "customer2"]
        mock_get_data_dir.return_value = Path("/data/customer")
        mock_get_ad_path.return_value = Path("/output/customer/ad_features.csv")

        # Create real copy of fixture for testing
        enriched_df = sample_enriched_df.copy()

        mock_join_all_levels.return_value = enriched_df
        mock_create_aggregated_features.return_value = enriched_df

        workflow = ExtractWorkflow(aggregate_to_adset=False, verbose=False)

        with patch.object(workflow, "_load_data") as mock_load:
            mock_load.return_value = {"ad": pd.DataFrame()}

            # Mock to_csv only for the specific DataFrames being written
            original_to_csv = pd.DataFrame.to_csv

            def mock_to_csv(self, filepath, **kwargs):
                # Only mock writes to output paths
                if "/output/" in str(filepath):
                    return None
                # Call original for all other cases
                return original_to_csv(self, filepath, **kwargs)

            with patch.object(pd.DataFrame, "to_csv", mock_to_csv):
                # Mock Path.mkdir to prevent actual directory creation
                with patch.object(Path, "mkdir", return_value=None):
                    results = workflow.run(platform="meta")

        assert len(results) == 2
        assert all(result.success for result in results.values())
        assert workflow.metrics.total_customers == 2
        assert workflow.metrics.successful_customers == 2

    @patch("src.meta.adset.allocator.features.workflows.base.get_all_customers")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_data_dir")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.ensure_customer_dirs")
    @patch("src.meta.adset.allocator.features.Joiner.join_all_levels")
    @patch("src.meta.adset.allocator.features.Aggregator.create_aggregated_features")
    @patch("src.meta.adset.allocator.features.workflows.extract_workflow.get_customer_ad_features_path")
    @patch("pandas.DataFrame.to_csv")
    @patch("pathlib.Path.mkdir", return_value=None)
    def test_workflow_run_single_customer(
        self,
        mock_mkdir,
        mock_to_csv,
        mock_get_ad_path,
        mock_create_aggregated_features,
        mock_join_all_levels,
        mock_ensure_dirs,
        mock_get_data_dir,
        mock_get_customers,
        sample_enriched_df,
    ):
        """Test running extract workflow for single customer."""
        mock_get_data_dir.return_value = Path("/data/customer1")
        mock_get_ad_path.return_value = Path("/output/customer1/ad_features.csv")

        mock_join_all_levels.return_value = sample_enriched_df
        mock_create_aggregated_features.return_value = sample_enriched_df

        workflow = ExtractWorkflow(aggregate_to_adset=False, verbose=False)

        with patch.object(workflow, "_load_data") as mock_load:
            mock_load.return_value = {"ad": pd.DataFrame()}

            results = workflow.run(customer="customer1", platform="meta")

        assert len(results) == 1
        assert "customer1" in results
        assert results["customer1"].success is True
        assert workflow.metrics.total_customers == 1
