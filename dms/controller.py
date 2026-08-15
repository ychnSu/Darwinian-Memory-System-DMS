"""Model-agnostic orchestration contract for one DMS sub-plan."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from dms.memory import DMSMemoryStore, MemoryUnit, PlanUnit, TrajectoryStep
from dms.policy import DMSPolicy


@dataclass(frozen=True)
class ExecutionResult:
  verified: bool
  reason: str = ""


class DMSController:
  """Coordinates retrieval, epsilon mutation, feedback, and replacement.

  The AndroidWorld adapter supplies `generate`, `execute`, and `verify`; this
  separation prevents benchmark plumbing from changing the memory algorithm.
  """

  def __init__(self, store: DMSMemoryStore, policy: DMSPolicy):
    self.store, self.policy = store, policy

  def execute_subplan(
      self,
      plan: PlanUnit,
      generate: Callable[[PlanUnit], list[TrajectoryStep]],
      execute: Callable[[list[TrajectoryStep]], ExecutionResult],
  ) -> ExecutionResult:
    hit = self.store.retrieve(plan, risk_threshold=self.policy.risk_threshold)
    replay = self.policy.should_replay(hit)
    trajectory = hit.trajectory if replay and hit else generate(plan)
    result = execute(trajectory)
    if hit and replay:
      self.store.record_verification(hit.id, verified=result.verified)
    elif hit:  # epsilon mutation: retain only a verified, shorter improvement.
      self.store.replace_if_better(hit.id, trajectory, verified=result.verified)
    elif result.verified and len(trajectory) >= 2:
      self.store.add(MemoryUnit(plan=plan, trajectory=trajectory))
    return result
