from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence


Box = Any


@dataclass(frozen=True)
class MatchInfo:
    """One matched pair and its matcher-specific diagnostics."""
    left_index: int
    right_index: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class MatchResult:
    matches: list[MatchInfo]
    unmatched_left: list[int]
    unmatched_right: list[int]


class Matcher(Protocol):
    """Interchangeable one-to-one matcher interface."""

    def match(self, left: Sequence[Box], right: Sequence[Box]) -> MatchResult:
        ...


@dataclass
class ErrorCandidate:
    box: Box
    final: bool = True
    sources: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FramePrediction:
    fp: list[ErrorCandidate]
    fn: list[ErrorCandidate]


DominantType = Literal["FP", "FN", "balanced"]
