"""Task-level lifecycle for DMS global feedback and scheduled pruning."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from dms.androidworld_agent import DMSAndroidWorldAgent


@dataclass(frozen=True)
class TaskRunResult:
  task_succeeded: bool
  agent_finished: bool
  steps: int
  active_memory_ids: set[str]
  pruned_memory_ids: list[str]


class DMSExperimentRunner:
  """Separates local verifier feedback from global AndroidWorld task feedback."""

  def __init__(self, agent: DMSAndroidWorldAgent, *, prune_interval_tasks: int = 5):
    if prune_interval_tasks < 1:
      raise ValueError("prune_interval_tasks must be positive.")
    self.agent = agent
    self.prune_interval_tasks = prune_interval_tasks
    self._completed_tasks = 0

  def run_task(self, goal: str, *, max_steps: int, task_success_evaluator: Callable[[], bool]) -> TaskRunResult:
    """Use AndroidWorld ground truth rather than model self-report as success."""
    self.agent.reset()
    agent_finished = False
    steps = 0
    for steps in range(1, max_steps + 1):
      result = self.agent.step(goal)
      if result.done:
        agent_finished = True
        break
    task_succeeded = bool(task_success_evaluator())
    active = self.agent.finalize_task(task_succeeded=task_succeeded)
    self._completed_tasks += 1
    pruned = []
    if self._completed_tasks % self.prune_interval_tasks == 0:
      pruned = self.agent.store.prune()
    return TaskRunResult(task_succeeded, agent_finished, steps, active, pruned)
