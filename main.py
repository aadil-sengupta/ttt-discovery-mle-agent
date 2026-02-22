"""
TTT-Discover agent orchestrator for MLE-Bench.

Implements the core TTT-Discover loop (arXiv:2601.16175) adapted for
prompt-based evolution with a frozen LLM instead of RL weight updates:

    1. Generate data summary for LLM context
    2. Seed the buffer with K initial pipelines (generate from scratch)
    3. While time budget remains:
        a. Select seed variant via PUCT
        b. Generate mutation via LLM
        c. Execute the variant and record CV score
        d. Add to buffer with lineage tracking
    4. Select best-scoring variant and re-run to produce final submission.csv

Usage (called from start.sh):
    python main.py \\
        --data-dir /home/data \\
        --submission-dir /home/submission \\
        --code-dir /home/code \\
        --logs-dir /home/logs \\
        --instructions full_instructions.txt \\
        --model gpt-4o-2024-08-06 \\
        --max-iterations 50 \\
        --initial-variants 5 \\
        --puct-c 1.4 \\
        --per-variant-timeout 600
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

from buffer import SolutionBuffer, PipelineVariant, make_variant_id
from sampler import PUCTSampler
from generator import PipelineGenerator
from runner import run_pipeline, validate_submission, generate_data_summary

logger = logging.getLogger("tttDiscovery")

# Reserve time at the end for final submission re-run + validation
FINAL_RESERVE_SECONDS = 120
# Minimum time required to attempt another iteration
MIN_ITERATION_SECONDS = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTT-Discover agent for MLE-Bench")

    # Paths (set by start.sh)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)

    # Model and hyperparameters (set by config.yaml kwargs)
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--initial-variants", type=int, default=5)
    parser.add_argument("--puct-c", type=float, default=1.4)
    parser.add_argument("--per-variant-timeout", type=int, default=600)

    return parser.parse_args()


def setup_logging(logs_dir: Path) -> None:
    """Configure logging to both file and stderr."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "tttDiscovery.log"

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


def read_instructions(path: Path) -> str:
    """Read the assembled instructions file."""
    if path.exists():
        text = path.read_text()
        logger.info(f"Read instructions: {len(text)} chars from {path}")
        return text
    logger.warning(f"Instructions file not found: {path}")
    return ""


def time_remaining(start_time: float, time_limit: float) -> float:
    """Seconds remaining in the time budget."""
    return time_limit - (time.time() - start_time)


def seed_buffer(
    generator: PipelineGenerator,
    buffer: SolutionBuffer,
    args: argparse.Namespace,
    start_time: float,
    time_limit: float,
) -> None:
    """Seed the buffer with K initial pipelines generated from scratch.

    Each pipeline is generated independently via the LLM, executed, scored,
    and added to the buffer. This creates the initial population for PUCT
    seed selection.
    """
    k = args.initial_variants
    logger.info(f"=== Seeding buffer with {k} initial variants ===")

    for i in range(k):
        remaining = time_remaining(start_time, time_limit)
        if remaining < FINAL_RESERVE_SECONDS + MIN_ITERATION_SECONDS:
            logger.warning(
                f"Time budget low ({remaining:.0f}s), stopping seeding after {i} variants"
            )
            break

        logger.info(f"--- Generating initial variant {i + 1}/{k} ---")
        variant_id = make_variant_id()

        try:
            code, description = generator.generate_initial()
        except Exception as e:
            logger.error(f"LLM generation failed for initial variant {i + 1}: {e}")
            # Record the failure so we know we tried
            variant = PipelineVariant(
                id=variant_id,
                code="",
                score=0.0,
                parent_id=None,
                generation=0,
                error=f"LLM generation failed: {e}",
                description="Failed generation",
            )
            buffer.add(variant)
            continue

        # Execute the pipeline
        score, error = run_pipeline(
            code=code,
            variant_id=variant_id,
            code_dir=args.code_dir,
            submission_dir=args.submission_dir,
            timeout_seconds=args.per_variant_timeout,
        )

        # Check submission validity if execution succeeded
        submission_valid = False
        if error is None:
            submission_valid = validate_submission(args.submission_dir)

        variant = PipelineVariant(
            id=variant_id,
            code=code,
            score=score,
            parent_id=None,
            generation=0,
            error=error,
            submission_valid=submission_valid,
            description=description,
        )
        buffer.add(variant)

        stats = buffer.get_score_stats()
        logger.info(
            f"After seeding {i + 1}/{k}: "
            f"buffer={stats['total']} total, {stats['valid']} valid, "
            f"best={stats['best_score']:.6f}"
        )

    # Save buffer checkpoint after seeding
    buffer.save(args.logs_dir / "buffer_after_seeding.json")


def mutation_loop(
    generator: PipelineGenerator,
    sampler: PUCTSampler,
    buffer: SolutionBuffer,
    args: argparse.Namespace,
    start_time: float,
    time_limit: float,
) -> None:
    """Main TTT-Discover mutation loop.

    Repeatedly: select seed via PUCT -> mutate via LLM -> execute -> record.
    Continues until time budget or max iterations are exhausted.
    """
    max_iter = args.max_iterations
    generation = 1  # Seeding was generation 0

    logger.info(f"=== Starting mutation loop (max {max_iter} iterations) ===")

    for iteration in range(max_iter):
        remaining = time_remaining(start_time, time_limit)
        if remaining < FINAL_RESERVE_SECONDS + MIN_ITERATION_SECONDS:
            logger.info(
                f"Time budget exhausted ({remaining:.0f}s remaining), "
                f"stopping after {iteration} mutations"
            )
            break

        logger.info(
            f"--- Mutation iteration {iteration + 1}/{max_iter} "
            f"({remaining:.0f}s remaining) ---"
        )

        # Select seed via PUCT
        seed = sampler.select(buffer)
        variant_id = make_variant_id()

        if seed is None:
            # No valid variants in buffer - try generating from scratch again
            logger.warning("No valid seeds available, generating from scratch")
            try:
                code, description = generator.generate_initial()
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                variant = PipelineVariant(
                    id=variant_id,
                    code="",
                    score=0.0,
                    parent_id=None,
                    generation=generation,
                    error=f"LLM generation failed: {e}",
                    description="Failed generation (fallback)",
                )
                buffer.add(variant)
                generation += 1
                continue
            parent_id = None
        else:
            # Generate mutation from seed
            top_variants = buffer.get_top_k(5)
            try:
                code, description = generator.generate_mutation(seed, top_variants)
            except Exception as e:
                logger.error(f"LLM mutation failed from seed {seed.id}: {e}")
                variant = PipelineVariant(
                    id=variant_id,
                    code="",
                    score=0.0,
                    parent_id=seed.id,
                    generation=generation,
                    error=f"LLM mutation failed: {e}",
                    description="Failed mutation",
                )
                buffer.add(variant)
                generation += 1
                continue
            parent_id = seed.id

        # Execute the pipeline
        score, error = run_pipeline(
            code=code,
            variant_id=variant_id,
            code_dir=args.code_dir,
            submission_dir=args.submission_dir,
            timeout_seconds=args.per_variant_timeout,
        )

        # Check submission validity
        submission_valid = False
        if error is None:
            submission_valid = validate_submission(args.submission_dir)

        variant = PipelineVariant(
            id=variant_id,
            code=code,
            score=score,
            parent_id=parent_id,
            generation=generation,
            error=error,
            submission_valid=submission_valid,
            description=description,
        )
        buffer.add(variant)
        generation += 1

        # Log progress
        stats = buffer.get_score_stats()
        logger.info(
            f"Iteration {iteration + 1}: score={score:.6f} "
            f"{'OK' if error is None else 'FAIL'} | "
            f"Buffer: {stats['valid']}/{stats['total']} valid, "
            f"best={stats['best_score']:.6f}, "
            f"mean={stats['mean_score']:.6f}"
        )

        # Periodic buffer checkpoint
        if (iteration + 1) % 5 == 0:
            buffer.save(args.logs_dir / "buffer_checkpoint.json")


def finalize_submission(
    buffer: SolutionBuffer,
    args: argparse.Namespace,
) -> bool:
    """Select the best variant and re-run it to produce the final submission.csv.

    Re-running ensures we get a clean submission file from the best pipeline,
    not a leftover from a later (possibly worse) variant.

    Returns True if a valid submission was produced.
    """
    logger.info("=== Finalizing submission ===")

    best = buffer.get_best()
    if best is None:
        logger.error("No successful variants in buffer! No submission to produce.")
        return False

    logger.info(
        f"Best variant: id={best.id} score={best.score:.6f} "
        f"gen={best.generation} desc='{best.description}'"
    )

    # Re-run the best variant to produce a fresh submission.csv
    logger.info(f"Re-running best variant {best.id} for final submission...")
    score, error = run_pipeline(
        code=best.code,
        variant_id=f"{best.id}_final",
        code_dir=args.code_dir,
        submission_dir=args.submission_dir,
        timeout_seconds=args.per_variant_timeout,
    )

    if error is not None:
        logger.error(
            f"Final re-run failed: {error}. Falling back to submission from buffer."
        )
        # Try to use a previously saved submission from any successful variant
        return _fallback_submission(buffer, args)

    submission_path = args.submission_dir / "submission.csv"
    if not submission_path.exists():
        logger.error("Final re-run did not produce submission.csv!")
        return _fallback_submission(buffer, args)

    # Validate
    is_valid = validate_submission(args.submission_dir)
    logger.info(
        f"Final submission: score={score:.6f}, valid={is_valid}, path={submission_path}"
    )
    return submission_path.exists()


def _fallback_submission(
    buffer: SolutionBuffer,
    args: argparse.Namespace,
) -> bool:
    """Try to produce a submission from any successful variant in the buffer.

    Iterates through variants by score (descending) and re-runs until one
    produces a valid submission.csv.
    """
    logger.warning("Attempting fallback submission from buffer...")
    top_variants = buffer.get_top_k(10)

    for i, variant in enumerate(top_variants):
        if not variant.submission_valid:
            continue

        logger.info(
            f"Fallback attempt {i + 1}: variant {variant.id} (score={variant.score:.6f})"
        )
        score, error = run_pipeline(
            code=variant.code,
            variant_id=f"{variant.id}_fallback",
            code_dir=args.code_dir,
            submission_dir=args.submission_dir,
            timeout_seconds=args.per_variant_timeout,
        )

        if error is None and (args.submission_dir / "submission.csv").exists():
            logger.info(f"Fallback succeeded with variant {variant.id}")
            return True

    logger.error("All fallback attempts failed. No submission produced.")
    return False


def main() -> None:
    args = parse_args()

    # Setup
    setup_logging(args.logs_dir)
    args.code_dir.mkdir(parents=True, exist_ok=True)
    args.submission_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TTT-Discover Agent Starting")
    logger.info("=" * 60)
    logger.info(f"  model:              {args.model}")
    logger.info(f"  max_iterations:     {args.max_iterations}")
    logger.info(f"  initial_variants:   {args.initial_variants}")
    logger.info(f"  puct_c:             {args.puct_c}")
    logger.info(f"  per_variant_timeout:{args.per_variant_timeout}s")
    logger.info(f"  data_dir:           {args.data_dir}")
    logger.info(f"  submission_dir:     {args.submission_dir}")
    logger.info(f"  code_dir:           {args.code_dir}")
    logger.info(f"  logs_dir:           {args.logs_dir}")

    # Time budget: use TIME_LIMIT_SECS env var if available, else default to 24h
    import os

    time_limit = int(os.environ.get("TIME_LIMIT_SECS", 86400))
    start_time = time.time()
    logger.info(f"  time_limit:         {time_limit}s ({time_limit / 3600:.1f}h)")

    # Read competition instructions
    instructions = read_instructions(args.instructions)

    # Generate data summary for LLM context
    logger.info("Generating data summary...")
    data_summary = generate_data_summary(args.data_dir)
    logger.info(f"Data summary: {len(data_summary)} chars")

    # Initialize components
    buffer = SolutionBuffer()
    sampler = PUCTSampler(exploration_c=args.puct_c)
    generator = PipelineGenerator(
        model=args.model,
        data_dir=str(args.data_dir),
        submission_dir=str(args.submission_dir),
        instructions=instructions,
        data_summary=data_summary,
    )

    # Phase 1: Seed the buffer with initial variants
    seed_buffer(generator, buffer, args, start_time, time_limit)

    # Phase 2: Iterative mutation loop
    mutation_loop(generator, sampler, buffer, args, start_time, time_limit)

    # Phase 3: Finalize - select best and produce submission
    success = finalize_submission(buffer, args)

    # Save final buffer state
    buffer.save(args.logs_dir / "buffer_final.json")

    # Log final summary
    stats = buffer.get_score_stats()
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("TTT-Discover Agent Complete")
    logger.info("=" * 60)
    logger.info(f"  Elapsed:    {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
    logger.info(f"  Variants:   {stats['total']} total, {stats['valid']} valid")
    logger.info(f"  Best score: {stats['best_score']:.6f}")
    logger.info(f"  Mean score: {stats['mean_score']:.6f}")
    logger.info(f"  Submission: {'PRODUCED' if success else 'MISSING'}")

    if not success:
        logger.error("FATAL: No valid submission.csv produced!")
        sys.exit(1)


if __name__ == "__main__":
    main()
