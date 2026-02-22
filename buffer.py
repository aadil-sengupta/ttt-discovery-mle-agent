"""
Solution buffer for TTT-Discover agent.

Stores pipeline variants with their cross-validation scores, parent lineage,
and visit counts for PUCT-based seed selection.

Adapted from the TTT-Discover paper (arXiv:2601.16175) which maintains a buffer H
of (state, action, reward) tuples for state reuse.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineVariant:
    """A single ML pipeline variant stored in the buffer.

    In TTT-Discover terms:
      - code is the 'state' s (a candidate solution)
      - score is the 'reward' R(s)
      - parent_id tracks the lineage for PUCT tree structure
    """

    id: str
    code: str
    score: float  # CV validation score; higher is better. 0.0 = failed/invalid
    parent_id: Optional[str]  # ID of the seed variant this was mutated from
    generation: int  # Which iteration created this variant
    visit_count: int = 0  # Times selected as a PUCT seed (n(s) in the paper)
    child_best_score: float = 0.0  # Max reward among descendants (Q(s) in the paper)
    error: Optional[str] = None  # Execution error message if any
    submission_valid: bool = False  # Whether submission.csv passed /validate
    description: str = ""  # Brief human-readable description of the approach
    metadata: dict = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """Whether this variant executed successfully and produced a positive score."""
        return self.error is None and self.score > 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PipelineVariant:
        return cls(**d)


class SolutionBuffer:
    """Buffer of pipeline variants with scores.

    Corresponds to H_i in Algorithm 1 of TTT-Discover:
      H_{i+1} = H_i U {(s_i, a_i, s'_i, r_i)}

    The buffer stores all attempted variants (including failures) and provides
    methods for querying by score, lineage, and visit counts used by the
    PUCT sampler.
    """

    def __init__(self) -> None:
        self._variants: dict[str, PipelineVariant] = {}  # id -> variant
        self._order: list[str] = []  # insertion order

    def add(self, variant: PipelineVariant) -> None:
        """Add a variant to the buffer and update parent's child_best_score."""
        self._variants[variant.id] = variant
        self._order.append(variant.id)

        # Update ancestor chain's child_best_score (Q-value propagation)
        if variant.score > 0:
            self._propagate_score(variant)

        logger.info(
            f"Buffer add: id={variant.id} score={variant.score:.6f} "
            f"parent={variant.parent_id} gen={variant.generation} "
            f"valid={variant.submission_valid} size={self.size}"
        )

    def _propagate_score(self, variant: PipelineVariant) -> None:
        """Propagate a variant's score up the parent chain.

        In TTT-Discover, Q(s) = max reward among states generated when s was
        the initial state. We use the max (not mean) following the paper:
        'we care about the best outcome starting from a state, not the average.'
        """
        current_id = variant.parent_id
        while current_id is not None and current_id in self._variants:
            parent = self._variants[current_id]
            if variant.score > parent.child_best_score:
                parent.child_best_score = variant.score
            else:
                break  # No further propagation needed if score isn't higher
            current_id = parent.parent_id

    def get(self, variant_id: str) -> Optional[PipelineVariant]:
        """Get a variant by ID."""
        return self._variants.get(variant_id)

    def get_best(self) -> Optional[PipelineVariant]:
        """Return the variant with the highest CV score."""
        valid = self.get_all_valid()
        if not valid:
            return None
        return max(valid, key=lambda v: v.score)

    def get_top_k(self, k: int) -> list[PipelineVariant]:
        """Return the top-k variants by score."""
        valid = self.get_all_valid()
        return sorted(valid, key=lambda v: v.score, reverse=True)[:k]

    def get_all_valid(self) -> list[PipelineVariant]:
        """Return all variants that executed successfully with score > 0."""
        return [v for v in self._variants.values() if v.is_successful]

    def get_all(self) -> list[PipelineVariant]:
        """Return all variants in insertion order."""
        return [self._variants[vid] for vid in self._order]

    @property
    def size(self) -> int:
        """Total number of variants in the buffer."""
        return len(self._variants)

    @property
    def valid_count(self) -> int:
        """Number of successful variants."""
        return len(self.get_all_valid())

    def increment_visit(self, variant_id: str) -> None:
        """Increment the visit count when a variant is selected as seed."""
        if variant_id in self._variants:
            self._variants[variant_id].visit_count += 1

    @property
    def total_visits(self) -> int:
        """Total number of expansions T (used in PUCT formula)."""
        return sum(v.visit_count for v in self._variants.values())

    def get_score_stats(self) -> dict:
        """Return summary statistics about the buffer."""
        valid = self.get_all_valid()
        if not valid:
            return {
                "total": self.size,
                "valid": 0,
                "best_score": 0.0,
                "mean_score": 0.0,
            }
        scores = [v.score for v in valid]
        return {
            "total": self.size,
            "valid": len(valid),
            "best_score": max(scores),
            "mean_score": sum(scores) / len(scores),
            "worst_score": min(scores),
        }

    def save(self, path: Path) -> None:
        """Persist the buffer to a JSON file."""
        data = [v.to_dict() for v in self.get_all()]
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Buffer saved to {path} ({self.size} variants)")

    def load(self, path: Path) -> None:
        """Load a previously saved buffer."""
        data = json.loads(path.read_text())
        for d in data:
            variant = PipelineVariant.from_dict(d)
            self._variants[variant.id] = variant
            self._order.append(variant.id)
        logger.info(f"Buffer loaded from {path} ({self.size} variants)")


def make_variant_id() -> str:
    """Generate a short unique ID for a pipeline variant."""
    return uuid.uuid4().hex[:8]
