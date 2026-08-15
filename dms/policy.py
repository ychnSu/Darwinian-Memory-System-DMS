"""Execution choice and mutation policy for DMS."""
from __future__ import annotations

import random
from dataclasses import dataclass

from dms.memory import MemoryUnit


@dataclass(frozen=True)
class DMSPolicy:
  epsilon: float = 0.10
  risk_threshold: float = 0.35
  threshold_sensitivity: float = 0.30
  seed: int = 30

  def effective_risk_threshold(self, global_failure_rate: float) -> float:
    """Paper §3.2.4: τ = τ_base * (1 - λ * T_global)."""
    if not 0.0 <= global_failure_rate <= 1.0:
      raise ValueError("global_failure_rate must be between zero and one.")
    if not 0.0 <= self.threshold_sensitivity <= 1.0:
      raise ValueError("threshold_sensitivity must be between zero and one.")
    return self.risk_threshold * (1.0 - self.threshold_sensitivity * global_failure_rate)

  def should_replay(
      self,
      memory: MemoryUnit | None,
      *,
      risk_threshold: float | None = None,
      global_failure_rate: float = 0.5,
  ) -> bool:
    threshold = self.risk_threshold if risk_threshold is None else risk_threshold
    if (
        memory is None
        or memory.risk_lower_bound(
            global_failure_rate=global_failure_rate
        ) >= threshold
    ):
      return False
    return random.Random(self.seed + memory.reuse_count).random() >= self.epsilon
