"""
LLM code generation for TTT-Discover agent.

Generates ML pipeline code via OpenAI API in two modes:
  1. Initial generation: Create a pipeline from scratch given competition description
  2. Mutation: Improve an existing pipeline given its code and CV score

In TTT-Discover terms, this implements the policy pi_theta that generates
actions a ~ pi_theta(. | d, s, c) where:
  - d is the problem description
  - s is the seed state (existing pipeline code, or empty for initial)
  - c is the context (buffer history summary)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from openai import OpenAI

from buffer import PipelineVariant

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert machine learning engineer competing in a Kaggle-style competition.
Your task is to write a complete, self-contained Python script that:

1. Reads the training data from the DATA_DIR specified below.
2. Performs feature engineering and preprocessing.
3. Trains a model using stratified K-fold cross-validation (K=5) and prints the \
mean CV score as: CV_SCORE: <score>
4. Trains on the full training data and generates predictions on the test set.
5. Writes a valid submission CSV to the SUBMISSION_DIR specified below.

CRITICAL REQUIREMENTS:
- The script MUST print exactly one line containing "CV_SCORE: <number>" where <number> \
is the mean cross-validation score (e.g. "CV_SCORE: 0.8542"). This is how we track \
your performance.
- The script MUST write a file called "submission.csv" to the SUBMISSION_DIR.
- The submission.csv MUST match the format shown in sample_submission.csv.
- The script must be completely self-contained. No imports from custom modules.
- Use only standard ML libraries: pandas, numpy, scikit-learn, xgboost, lightgbm, \
catboost, scipy, torch (if needed).
- Handle missing values, categorical features, and edge cases robustly.
- Do NOT use multiprocessing or threading.
- Do NOT read from or write to any directory other than DATA_DIR and SUBMISSION_DIR.
- Use appropriate error handling so the script does not crash on unexpected data.

DATA_DIR: {data_dir}
SUBMISSION_DIR: {submission_dir}
"""

INITIAL_GENERATION_PROMPT = """\
## Competition Instructions

{instructions}

## Data Summary

{data_summary}

## Task

Write a complete Python script to solve this competition. Start with a solid baseline \
approach (e.g., gradient boosted trees with basic feature engineering). Focus on:
- Correctly identifying the target variable and metric
- Proper train/test splitting that matches the sample_submission format
- Robust feature preprocessing (handle missing values, encode categoricals)
- A reasonable model choice for the data type and size

Remember: Print "CV_SCORE: <score>" and write "submission.csv" to the submission directory.

Write ONLY the Python code, no explanations. Start with imports.
"""

MUTATION_PROMPT = """\
## Competition Instructions

{instructions}

## Data Summary

{data_summary}

## Current Best Pipeline (CV_SCORE: {seed_score:.6f})

```python
{seed_code}
```

## Buffer History (top variants)

{buffer_summary}

## Task

Improve the pipeline above. Its current cross-validation score is {seed_score:.6f}.

Try ONE of these improvement strategies:
1. **Feature engineering**: Add interaction features, polynomial features, target encoding, \
or domain-specific transformations.
2. **Model tuning**: Adjust hyperparameters (learning rate, max_depth, n_estimators, \
regularization, etc.).
3. **Model swap**: Try a different algorithm (LightGBM, CatBoost, XGBoost, RandomForest, \
ExtraTrees, or an ensemble).
4. **Preprocessing**: Try different imputation strategies, scaling methods, or outlier handling.
5. **Ensemble**: Blend predictions from multiple models (averaging or stacking).
6. **Feature selection**: Remove noisy features or use importance-based selection.

Pick the strategy most likely to improve the score given the current approach and history.

{improvement_hint}

Write the COMPLETE improved Python script (not just the changes). Include a brief \
comment at the top describing what you changed and why.

Remember: Print "CV_SCORE: <score>" and write "submission.csv" to the submission directory.

Write ONLY the Python code, no explanations. Start with a comment describing your change, \
then imports.
"""


def _format_buffer_summary(top_variants: list[PipelineVariant]) -> str:
    """Format the top buffer variants as context for the mutation prompt."""
    if not top_variants:
        return "No previous variants."

    lines = []
    for i, v in enumerate(top_variants):
        desc = v.description or "No description"
        status = "SUCCESS" if v.is_successful else f"FAILED: {v.error or 'unknown'}"
        lines.append(
            f"{i + 1}. [{status}] CV_SCORE={v.score:.6f} (gen {v.generation}) - {desc}"
        )
    return "\n".join(lines)


def _generate_improvement_hint(
    seed: PipelineVariant, top_variants: list[PipelineVariant]
) -> str:
    """Generate a targeted improvement hint based on buffer history.

    This serves a similar role to the 'context c' in TTT-Discover's
    state-action reuse, providing the LLM with relevant history to
    inform its next generation.
    """
    hints = []

    # Detect if many variants have similar scores (plateau)
    if top_variants and len(top_variants) >= 3:
        scores = [v.score for v in top_variants[:3]]
        if max(scores) - min(scores) < 0.001:
            hints.append(
                "NOTE: Recent variants have very similar scores. Consider a more "
                "aggressive change like swapping the model type entirely or adding "
                "a fundamentally different feature engineering approach."
            )

    # Detect if the seed has been visited many times
    if seed.visit_count >= 3:
        hints.append(
            f"NOTE: This seed has been selected {seed.visit_count} times already. "
            "Try a significantly different approach to avoid generating similar variants."
        )

    return "\n".join(hints) if hints else ""


class PipelineGenerator:
    """Generates ML pipeline code via OpenAI API.

    Supports two modes:
      - generate_initial(): Create a pipeline from scratch
      - generate_mutation(): Improve an existing pipeline
    """

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        data_dir: str = "/home/data",
        submission_dir: str = "/home/submission",
        instructions: str = "",
        data_summary: str = "",
    ) -> None:
        self.client = OpenAI()  # Uses OPENAI_API_KEY from env
        self.model = model
        self.data_dir = data_dir
        self.submission_dir = submission_dir
        self.instructions = instructions
        self.data_summary = data_summary

    def generate_initial(self) -> tuple[str, str]:
        """Generate an initial pipeline from scratch.

        Returns:
            Tuple of (code, description) where description is a brief
            summary of the approach.
        """
        system = SYSTEM_PROMPT.format(
            data_dir=self.data_dir,
            submission_dir=self.submission_dir,
        )
        user = INITIAL_GENERATION_PROMPT.format(
            instructions=self.instructions,
            data_summary=self.data_summary,
        )

        logger.info(f"Generating initial pipeline with {self.model}")
        code = self._call_llm(system, user)
        code = _extract_code(code)
        description = _extract_description(code)
        return code, description

    def generate_mutation(
        self,
        seed: PipelineVariant,
        top_variants: list[PipelineVariant],
    ) -> tuple[str, str]:
        """Generate an improved variant by mutating a seed pipeline.

        Args:
            seed: The pipeline variant to improve upon
            top_variants: Top-k variants from the buffer for context

        Returns:
            Tuple of (code, description)
        """
        system = SYSTEM_PROMPT.format(
            data_dir=self.data_dir,
            submission_dir=self.submission_dir,
        )
        user = MUTATION_PROMPT.format(
            instructions=self.instructions,
            data_summary=self.data_summary,
            seed_score=seed.score,
            seed_code=seed.code,
            buffer_summary=_format_buffer_summary(top_variants),
            improvement_hint=_generate_improvement_hint(seed, top_variants),
        )

        logger.info(
            f"Generating mutation from seed {seed.id} "
            f"(score={seed.score:.6f}) with {self.model}"
        )
        code = self._call_llm(system, user)
        code = _extract_code(code)
        description = _extract_description(code)
        return code, description

    def _call_llm(self, system: str, user: str) -> str:
        """Make an OpenAI API call and return the response text."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.8,
                max_tokens=16384,
            )
            content = response.choices[0].message.content or ""
            logger.info(
                f"LLM response: {len(content)} chars, "
                f"usage={response.usage.total_tokens if response.usage else 'n/a'} tokens"
            )
            return content
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise


def _extract_code(response: str) -> str:
    """Extract Python code from LLM response.

    Handles responses that may include markdown code blocks or raw code.
    """
    # Try to extract from markdown code block
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        # Return the longest code block (likely the main script)
        return max(matches, key=len).strip()

    # If no code block, assume the entire response is code
    # Strip any leading/trailing markdown or explanation
    lines = response.strip().split("\n")

    # Find first line that looks like Python code
    start = 0
    for i, line in enumerate(lines):
        if line.startswith(("import ", "from ", "#", '"""', "'''", "import\t")):
            start = i
            break

    return "\n".join(lines[start:]).strip()


def _extract_description(code: str) -> str:
    """Extract a brief description from the first comment in the code."""
    lines = code.strip().split("\n")
    desc_lines = []
    for line in lines[:5]:
        line = line.strip()
        if line.startswith("#"):
            desc_lines.append(line.lstrip("# ").strip())
        elif not line:
            continue
        else:
            break
    return " ".join(desc_lines)[:200] if desc_lines else "No description"
