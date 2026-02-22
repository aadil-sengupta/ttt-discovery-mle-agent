"""
Pipeline variant executor for TTT-Discover agent.

Runs LLM-generated Python pipeline code in a subprocess with timeout,
captures the cross-validation score from stdout, and checks whether
a valid submission.csv was produced.

In TTT-Discover terms, this implements the transition function T(a) -> s'
and reward evaluation R(s').
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Pattern to extract CV score from pipeline stdout
CV_SCORE_PATTERN = re.compile(r"CV_SCORE:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def run_pipeline(
    code: str,
    variant_id: str,
    code_dir: Path,
    submission_dir: Path,
    timeout_seconds: int = 600,
) -> tuple[float, Optional[str]]:
    """Execute a pipeline variant and return (score, error_or_none).

    The generated code must:
      1. Perform cross-validation and print 'CV_SCORE: <float>' to stdout
      2. Train on full data and write predictions to submission_dir/submission.csv

    Args:
        code: Full Python script to execute
        variant_id: Unique ID for this variant (used for file naming)
        code_dir: Directory to save the script file
        submission_dir: Directory where submission.csv should be written
        timeout_seconds: Max execution time before killing the process

    Returns:
        Tuple of (cv_score, error_message).
        cv_score is 0.0 if execution failed.
        error_message is None on success.
    """
    # Write the code to a file
    script_path = code_dir / f"variant_{variant_id}.py"
    script_path.write_text(code)
    logger.info(f"Running variant {variant_id}: {script_path}")

    # Clear any previous submission so we can detect if a new one is produced
    submission_path = submission_dir / "submission.csv"
    had_previous = submission_path.exists()

    # We use a temp marker to track if THIS variant produced a submission
    marker = submission_dir / f".marker_{variant_id}"

    try:
        # Run in subprocess with timeout
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(code_dir),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        stdout = result.stdout
        stderr = result.stderr

        # Log output
        if stdout:
            # Truncate for logging but keep full for parsing
            log_stdout = stdout[:2000] + ("..." if len(stdout) > 2000 else "")
            logger.info(f"Variant {variant_id} stdout:\n{log_stdout}")
        if stderr:
            log_stderr = stderr[:2000] + ("..." if len(stderr) > 2000 else "")
            logger.warning(f"Variant {variant_id} stderr:\n{log_stderr}")

        if result.returncode != 0:
            error_msg = (
                f"Exit code {result.returncode}: {stderr[-500:]}"
                if stderr
                else f"Exit code {result.returncode}"
            )
            logger.warning(f"Variant {variant_id} failed: {error_msg}")
            return 0.0, error_msg

        # Parse CV score from stdout
        score = _extract_cv_score(stdout)
        if score is None:
            logger.warning(f"Variant {variant_id}: No CV_SCORE found in output")
            return 0.0, "No CV_SCORE printed to stdout"

        # Check if submission was produced
        if not submission_path.exists():
            logger.warning(f"Variant {variant_id}: No submission.csv produced")
            return score, "No submission.csv produced"

        logger.info(f"Variant {variant_id} completed: CV_SCORE={score:.6f}")
        return score, None

    except subprocess.TimeoutExpired:
        error_msg = f"Timed out after {timeout_seconds}s"
        logger.warning(f"Variant {variant_id}: {error_msg}")
        return 0.0, error_msg

    except Exception as e:
        error_msg = f"Execution error: {str(e)}"
        logger.error(f"Variant {variant_id}: {error_msg}")
        return 0.0, error_msg


def _extract_cv_score(stdout: str) -> Optional[float]:
    """Extract the CV score from pipeline stdout.

    Looks for the pattern 'CV_SCORE: <number>' in the output.
    If multiple matches, returns the last one (final score).
    """
    matches = CV_SCORE_PATTERN.findall(stdout)
    if not matches:
        return None
    try:
        score = float(matches[-1])  # Take the last CV_SCORE printed
        if not (0.0 <= score <= 1e10):  # Sanity check
            logger.warning(f"CV_SCORE {score} outside expected range, using 0.0")
            return 0.0
        return score
    except (ValueError, IndexError):
        return None


def validate_submission(submission_dir: Path) -> bool:
    """Validate submission via the MLE-bench grading server.

    Posts the submission.csv to http://localhost:5000/validate which
    returns valid/invalid (no score).
    """
    submission_path = submission_dir / "submission.csv"
    if not submission_path.exists():
        return False

    try:
        result = subprocess.run(
            [
                "curl",
                "-s",
                "-X",
                "POST",
                "-F",
                f"file=@{submission_path}",
                "http://localhost:5000/validate",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        response = result.stdout.lower()
        is_valid = "valid" in response and "invalid" not in response
        logger.info(
            f"Submission validation: {result.stdout.strip()} -> valid={is_valid}"
        )
        return is_valid
    except Exception as e:
        logger.warning(f"Submission validation failed: {e}")
        return False


def generate_data_summary(data_dir: Path) -> str:
    """Generate a summary of the competition data for use in LLM prompts.

    Scans CSV files in the data directory and produces a structured summary
    of columns, types, shapes, and sample rows.
    """
    import pandas as pd

    summary_parts = []
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        # Check for nested directories
        csv_files = sorted(data_dir.rglob("*.csv"))

    for csv_path in csv_files[:5]:  # Limit to 5 files to keep prompt manageable
        try:
            # Read just a sample to avoid memory issues with large files
            df = pd.read_csv(csv_path, nrows=5)
            full_shape = _get_csv_shape(csv_path)

            part = f"\n### {csv_path.name}\n"
            part += f"- Shape: {full_shape[0]} rows x {full_shape[1]} columns\n"
            part += f"- Columns: {list(df.columns)}\n"
            part += f"- Dtypes:\n"
            for col in df.columns:
                part += f"  - {col}: {df[col].dtype}\n"
            part += f"- First 3 rows:\n{df.head(3).to_string()}\n"

            # Check for missing values in a larger sample
            df_sample = pd.read_csv(csv_path, nrows=1000)
            missing = df_sample.isnull().sum()
            if missing.any():
                part += f"- Missing values (in first 1000 rows):\n"
                for col in missing[missing > 0].index:
                    part += f"  - {col}: {missing[col]}\n"

            summary_parts.append(part)
        except Exception as e:
            summary_parts.append(f"\n### {csv_path.name}\n- Error reading: {e}\n")

    # Also list non-CSV files
    other_files = [
        f.name
        for f in data_dir.iterdir()
        if f.is_file() and f.suffix != ".csv" and not f.name.startswith(".")
    ]
    if other_files:
        summary_parts.append(f"\n### Other files in data directory:\n{other_files}\n")

    return "\n".join(summary_parts) if summary_parts else "No data files found."


def _get_csv_shape(path: Path) -> tuple[int, int]:
    """Get the full shape of a CSV without loading it all into memory."""
    import pandas as pd

    try:
        # Count rows efficiently
        df_header = pd.read_csv(path, nrows=0)
        ncols = len(df_header.columns)

        # Count lines (subtract 1 for header)
        with open(path, "r") as f:
            nrows = sum(1 for _ in f) - 1

        return (nrows, ncols)
    except Exception:
        return (0, 0)
