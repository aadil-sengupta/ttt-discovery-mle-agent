"""
PUCT-inspired seed selection for TTT-Discover agent.

Implements the state reuse heuristic from TTT-Discover (Section 3.2):

    Score(s) = Q(s) + c * P(s) * sqrt(1 + T) / (1 + n(s))

Where:
    Q(s) = max reward among children of s (or R(s) if no children)
    P(s) = rank-based prior proportional to reward rank in the buffer
    n(s) = visit count (times s was selected as seed)
    T    = total number of expansions
    c    = exploration coefficient

Key design choice from the paper (Section 3.2):
    'Rather than the mean (as in prior work), we use the maximum reward
     of children in Q(s): we care about the best outcome starting from
     a state, not the average.'
"""

from __future__ import annotations

import logging
import math
import random
from typing import Optional

from buffer import SolutionBuffer, PipelineVariant

logger = logging.getLogger(__name__)


class PUCTSampler:
    """PUCT-based seed selector for the solution buffer.

    Selects which existing pipeline variant to use as the seed/parent for
    the next LLM mutation. Balances exploitation of high-scoring variants
    with exploration of under-visited ones.
    """

    def __init__(self, exploration_c: float = 1.4) -> None:
        """
        Args:
            exploration_c: Exploration coefficient c in the PUCT formula.
                Higher values favor exploring under-visited states.
                The paper uses this to 'prevent over-exploitation by keeping
                under-visited states as candidates.'
        """
        self.exploration_c = exploration_c

    def select(self, buffer: SolutionBuffer) -> Optional[PipelineVariant]:
        """Select the best seed variant from the buffer using PUCT scores.

        Returns None if the buffer has no valid variants (falls back to
        generating from scratch in the main loop).
        """
        valid = buffer.get_all_valid()
        if not valid:
            logger.info("PUCT: No valid variants in buffer, returning None")
            return None

        T = buffer.total_visits  # Total expansions

        # Compute rank-based prior P(s): sort by score, assign rank-proportional prior
        sorted_by_score = sorted(valid, key=lambda v: v.score, reverse=True)
        rank_priors = self._compute_rank_priors(sorted_by_score)

        # Score each variant
        best_variant = None
        best_puct_score = float("-inf")

        for i, variant in enumerate(sorted_by_score):
            q_value = self._compute_q_value(variant)
            p_value = rank_priors[i]
            n_visits = variant.visit_count

            # PUCT formula: Q(s) + c * P(s) * sqrt(1 + T) / (1 + n(s))
            exploration_bonus = (
                self.exploration_c * p_value * math.sqrt(1 + T) / (1 + n_visits)
            )
            puct_score = q_value + exploration_bonus

            logger.debug(
                f"PUCT: id={variant.id} Q={q_value:.4f} P={p_value:.4f} "
                f"n={n_visits} T={T} bonus={exploration_bonus:.4f} "
                f"total={puct_score:.4f}"
            )

            if puct_score > best_puct_score:
                best_puct_score = puct_score
                best_variant = variant

        if best_variant is not None:
            buffer.increment_visit(best_variant.id)
            logger.info(
                f"PUCT selected: id={best_variant.id} score={best_variant.score:.6f} "
                f"puct={best_puct_score:.4f} visits={best_variant.visit_count}"
            )

        return best_variant

    def _compute_q_value(self, variant: PipelineVariant) -> float:
        """Compute Q(s): max reward among children, or own score if no children.

        From the paper: 'Q(s) is the maximum reward among states generated
        when the initial state was s (or R(s) if s has not yet been selected).'
        """
        if variant.child_best_score > 0:
            return max(variant.score, variant.child_best_score)
        return variant.score

    def _compute_rank_priors(
        self, sorted_variants: list[PipelineVariant]
    ) -> list[float]:
        """Compute rank-based prior P(s) for each variant.

        From the paper: 'P(s) is proportional to s's rank in the buffer
        sorted by reward.' Higher-reward states get higher prior probability.

        We use a simple linear rank-based weighting:
            P(rank_i) = (N - rank_i) / sum(N - rank_j for all j)
        where rank 0 is the best.
        """
        n = len(sorted_variants)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        # Linear rank weights: best gets weight N, worst gets weight 1
        weights = [n - i for i in range(n)]
        total = sum(weights)
        return [w / total for w in weights]

    def select_random(self, buffer: SolutionBuffer) -> Optional[PipelineVariant]:
        """Fallback: select a random valid variant (for diversity)."""
        valid = buffer.get_all_valid()
        if not valid:
            return None
        return random.choice(valid)
