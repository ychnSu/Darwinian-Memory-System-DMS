"""No-op memory boundary for the pure Zero-shot baseline."""
from __future__ import annotations

from dms.memory import PlanUnit


class ZeroShotMemoryStore:
  """Implements the small store surface used by the Agent without retaining data."""

  memories: dict[str, object]

  def __init__(self):
    self.memories = {}

  def retrieve(self, plan: PlanUnit, *, risk_threshold: float = 0.35) -> None:
    del plan, risk_threshold
    return None

  def record_global_feedback(self, memory_ids: set[str], *, task_succeeded: bool) -> None:
    del memory_ids, task_succeeded

  def prune(self) -> list[str]:
    return []
