"""
Helper utilities for testing scripts.
"""

import subprocess
from pathlib import Path

import pandas as pd


def run_script(script_path, args):
    """
    Run a script and return the result.

    Args:
        script_path: Path to the script to run
        args: List of command-line arguments

    Returns:
        subprocess.CompletedProcess: The result of running the script
    """
    return subprocess.run(
        [script_path] + args,
        capture_output=True,
        text=True,
        check=False,
        cwd=Path.cwd(),
    )


def run_script_and_verify_output(script_path, args, output_file, expected_columns=None):
    """
    Run a script and verify its output file exists and has content.

    Args:
        script_path: Path to the script to run
        args: List of command-line arguments
        output_file: Path to expected output file
        expected_columns: Optional list of expected column names

    Returns:
        pandas.DataFrame: The output DataFrame

    Raises:
        AssertionError: If script fails or output is invalid
    """
    result = run_script(script_path, args)

    # Check that script executed successfully
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Check that output file was created
    assert output_file.exists()

    # Check that output file has content
    output_df = pd.read_csv(output_file)
    assert len(output_df) > 0

    # Check expected columns if provided
    if expected_columns:
        for col in expected_columns:
            assert col in output_df.columns, f"Missing column: {col}"

    return output_df
