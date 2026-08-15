"""Persistent, self-regulating memory primitives for DMS.

The module is independent of a particular VLM or AndroidWorld action format so
that the same implementation is used by zero-shot, static-memory, and DMS runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from math import exp, log, sqrt
from pathlib import Path
import re
from uuid import uuid4

from dms.embeddings import EmbeddingProvider


_STATIC_CONTEXT_TEXT_LIMIT = 220
_STATIC_CONTEXT_ACTION_LIMIT = 160
_STATIC_CONTEXT_MAX_TRAJECTORY_STEPS = 8
_NON_EXECUTABLE_ACTIONS = frozenset({"complete", "answer", "dms_recovery"})
_CLICK_ACTIONS = frozenset({"click", "tap", "double_tap", "long_press"})
_INVALID_FEEDBACK_PHASES = frozenset({"execution", "anti_loop"})


def _now() -> str:
  return datetime.now(timezone.utc).isoformat()


def _compact_text(value: str, limit: int = _STATIC_CONTEXT_TEXT_LIMIT) -> str:
  """Return a single-line prompt snippet without changing persisted memory."""
  text = " ".join(str(value).split())
  if len(text) <= limit:
    return text
  return text[: limit - 3].rstrip() + "..."


def _compact_action(action: dict) -> str:
  """Summarize an AndroidWorld action for Static Memory prompt context."""
  action_type = str(action.get("action_type", "unknown"))
  details: dict[str, object] = {"action_type": action_type}
  for key in (
      "app_name", "package", "index", "text", "x", "y", "start_x", "start_y",
      "end_x", "end_y", "press_key", "success", "reason",
  ):
    if key in action and action[key] not in (None, ""):
      details[key] = action[key]
  return _compact_text(
      json.dumps(details, ensure_ascii=False, sort_keys=True),
      _STATIC_CONTEXT_ACTION_LIMIT,
  )


def _compact_trajectory(trajectory: list["TrajectoryStep"]) -> str:
  """Return a bounded step-by-step trajectory excerpt for Planner context."""
  if not trajectory:
    return "  No recorded trajectory steps."
  lines: list[str] = []
  for index, step in enumerate(
      trajectory[:_STATIC_CONTEXT_MAX_TRAJECTORY_STEPS], start=1
  ):
    lines.append(
        f"  Step {index}: Observation={_compact_text(step.observation)}; "
        f"Action={_compact_action(step.action)}"
    )
  if len(trajectory) > _STATIC_CONTEXT_MAX_TRAJECTORY_STEPS:
    lines.append(
        f"  ... {len(trajectory) - _STATIC_CONTEXT_MAX_TRAJECTORY_STEPS} more steps omitted"
    )
  return "\n".join(lines)


def _normalize_key(value: str) -> str:
  return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _memory_task_name(memory: "MemoryUnit") -> str:
  """Extract the benchmark task template stored in Static Memory metadata."""
  for text in (memory.description, memory.plan.precondition):
    match = re.search(r"task_template=([^;]+)", text)
    if match:
      return match.group(1).strip()
    match = re.search(r"Task template:\s*([^;]+)", text)
    if match:
      return match.group(1).strip()
  return ""


def _memory_app_text(memory: "MemoryUnit") -> str:
  fields = [memory.description, memory.plan.precondition, memory.plan.goal]
  for step in memory.trajectory:
    action = step.action
    for key in ("app_name", "package"):
      if action.get(key):
        fields.append(str(action[key]))
  return _normalize_key(" ".join(fields))


def _matches_task(memory: "MemoryUnit", task_name: str | None) -> bool:
  return bool(task_name) and _memory_task_name(memory) == task_name


def _matches_app(memory: "MemoryUnit", task_app_names: tuple[str, ...]) -> bool:
  if not task_app_names:
    return False
  haystack = _memory_app_text(memory)
  return any(_normalize_key(app) and _normalize_key(app) in haystack
             for app in task_app_names)


@dataclass(frozen=True)
class PlanUnit:
  """Intent-layer key; both components are required for DMS retrieval."""
  precondition: str
  goal: str


@dataclass(frozen=True)
class TrajectoryStep:
  """Action-layer experience. `action` is an AndroidWorld JSONAction dict."""
  observation: str
  action: dict


@dataclass
class MemoryUnit:
  """Reusable DMS memory with decoupled intent/action layers.

  ``plan`` is the intent layer p=(precondition, goal). ``trajectory`` is the
  action layer tau=[(o_t, a_t)]. Feedback counters are environment-level utility
  signals used by survival-value selection and dynamic pruning.
  """
  plan: PlanUnit
  trajectory: list[TrajectoryStep]
  success: bool | None = None
  description: str = ""
  id: str = field(default_factory=lambda: str(uuid4()))
  reuse_count: int = 0
  verification_failures: int = 0
  task_successes: int = 0
  task_failures: int = 0
  total_action_count: int = 0
  invalid_action_count: int = 0
  invalid_click_count: int = 0
  no_progress_action_count: int = 0
  created_tick: int = 0
  last_used_tick: int = 0
  created_at: str = field(default_factory=_now)
  precondition_embedding: list[float] | None = None
  goal_embedding: list[float] | None = None
  trajectory_path: str = ""
  embedding_status: str = "ready"
  embedding_error: str = ""

  def survival_value(
      self, tick: int, *, new_bonus: float = 1.0, base_half_life: float = 30.0,
      longevity: float = 15.0, decay_steepness: float = 0.5,
      penalty: float = 1.0, invalid_penalty: float = 2.0,
      no_progress_penalty: float = 1.0,
  ) -> float:
    """Paper survival value with environment-feedback reward/penalty terms."""
    return self.survival_components(
        tick,
        new_bonus=new_bonus,
        base_half_life=base_half_life,
        longevity=longevity,
        decay_steepness=decay_steepness,
        penalty=penalty,
        invalid_penalty=invalid_penalty,
        no_progress_penalty=no_progress_penalty,
    )["survival_value"]

  def survival_components(
      self, tick: int, *, new_bonus: float = 1.0, base_half_life: float = 30.0,
      longevity: float = 15.0, decay_steepness: float = 0.5,
      penalty: float = 1.0, invalid_penalty: float = 2.0,
      no_progress_penalty: float = 1.0,
  ) -> dict[str, float]:
    """Return the decomposed value used for diagnostics and plots."""
    utility = log(1 + self.reuse_count) + new_bonus
    half_life = base_half_life + longevity * log(1 + self.reuse_count)
    elapsed = max(0, tick - self.last_used_tick)
    decay = 1 / (1 + exp(decay_steepness * (elapsed - half_life)))
    reliability = 1 / (1 + penalty * self.verification_failures)
    task_completion = self.task_completion_rate()
    invalid_click_rate = self.invalid_click_rate()
    no_progress_rate = self.no_progress_rate()
    invalid_click_factor = 1 / (1 + invalid_penalty * invalid_click_rate)
    no_progress_factor = 1 / (1 + no_progress_penalty * no_progress_rate)
    environment_feedback = (
        task_completion * invalid_click_factor * no_progress_factor
    )
    value = utility * decay * reliability * environment_feedback
    return {
        "survival_value": value,
        "utility": utility,
        "adaptive_decay": decay,
        "reliability": reliability,
        "task_completion": task_completion,
        "invalid_click_rate": invalid_click_rate,
        "no_progress_rate": no_progress_rate,
        "environment_feedback": environment_feedback,
    }

  def risk_lower_bound(
      self, *, global_failure_rate: float = 0.5, prior_strength: float = 2.0
  ) -> float:
    """Paper §3.2.4 Beta-Binomial LCB risk score.

    The prior mean is the ecosystem failure rate T_global and the prior
    strength M controls cold-start smoothing:
    mu_i = (F_i + M * T_global) / (F_i + S_i + M).
    """
    if not 0.0 <= global_failure_rate <= 1.0:
      raise ValueError("global_failure_rate must be between zero and one.")
    if prior_strength <= 0:
      raise ValueError("prior_strength must be positive.")
    denominator = self.task_failures + self.task_successes + prior_strength
    mean = (
        self.task_failures + prior_strength * global_failure_rate
    ) / denominator
    variance = mean * (1 - mean) / (denominator + 1)
    return max(0.0, mean - variance**0.5)

  def task_completion_rate(self, alpha: float = 1.0, beta: float = 1.0) -> float:
    """Smoothed task-completion reward from AndroidWorld evaluator feedback."""
    return (self.task_successes + alpha) / (
        self.task_successes + self.task_failures + alpha + beta
    )

  def invalid_click_rate(self) -> float:
    if self.total_action_count <= 0:
      return 0.0
    return self.invalid_click_count / self.total_action_count

  def no_progress_rate(self) -> float:
    if self.total_action_count <= 0:
      return 0.0
    return self.no_progress_action_count / self.total_action_count


class DMSMemoryStore:
  """JSON-persisted DMS memory bank with deterministic, dependency-free search."""

  def __init__(
      self, path: str | Path, *, embedder: EmbeddingProvider, static: bool = False,
      k_limit: int = 3, retrieval_strategy: str | None = None,
      capacity_min: int = 50, capacity_max: int = 200,
      capacity_step: int = 25,
  ):
    if k_limit < 1:
      raise ValueError("k_limit must be positive.")
    if capacity_min < 2:
      raise ValueError("capacity_min must be at least 2.")
    if capacity_max < capacity_min:
      raise ValueError("capacity_max must be greater than or equal to capacity_min.")
    if capacity_step < 1:
      raise ValueError("capacity_step must be positive.")
    self.path = Path(path)
    self.trajectory_dir = self.path.parent / "memory_trajectories"
    self.embedder = embedder
    self.static = static
    self.k_limit = k_limit
    self.capacity_min = capacity_min
    self.capacity_max = capacity_max
    self.capacity_step = capacity_step
    self.current_capacity = capacity_min
    retrieval_strategy = retrieval_strategy or (
        "chronological_context" if static else "dual_factor"
    )
    if retrieval_strategy not in ("chronological_context", "dual_factor"):
      raise ValueError("retrieval_strategy must be chronological_context or dual_factor.")
    if static != (retrieval_strategy == "chronological_context"):
      raise ValueError("Static stores require chronological context; DMS requires dual-factor retrieval.")
    self.retrieval_strategy = retrieval_strategy
    self.tick = 0
    # Task outcomes are ecosystem-level feedback, distinct from the per-memory
    # outcome counters used by the Beta reputation estimate.  They provide the
    # paper's T_global signal for dynamic risk thresholding.
    self.global_task_successes = 0
    self.global_task_failures = 0
    self.last_pruned_ids: list[str] = []
    self.last_prune_report: dict[str, object] = {}
    self.last_retrieval_report: dict[str, object] = {}
    self.total_created = 0
    self.total_retrievals = 0
    self.total_retrieval_hits = 0
    self.total_retrieval_misses = 0
    self.total_suppressed = 0
    self.total_replays = 0
    self.total_mutations = 0
    self.total_replacements = 0
    self.total_pruned = 0
    self.total_deleted_by_risk = 0
    self.total_embedding_failures = 0
    self.memories: dict[str, MemoryUnit] = {}
    if self.path.exists():
      self._load()

  def add(self, memory: MemoryUnit) -> MemoryUnit:
    if len(memory.trajectory) < 2:
      raise ValueError("DMS memory construction retains only non-trivial trajectories (length >= 2).")
    memory.created_tick = memory.last_used_tick = self.tick
    self._refresh_environment_feedback(memory)
    if self.retrieval_strategy == "dual_factor":
      try:
        embeddings = self.embedder.encode([memory.plan.precondition, memory.plan.goal])
      except Exception as error:  # Keep successful task memory; retry embeddings later.
        memory.precondition_embedding = None
        memory.goal_embedding = None
        memory.embedding_status = "pending"
        memory.embedding_error = f"{type(error).__name__}: {error}"
        self.total_embedding_failures += 1
      else:
        memory.precondition_embedding, memory.goal_embedding = embeddings
        memory.embedding_status = "ready"
        memory.embedding_error = ""
    self.memories[memory.id] = memory
    self.total_created += 1
    self._save()
    return memory

  @staticmethod
  def _refresh_environment_feedback(memory: MemoryUnit) -> None:
    """Derive invalid-action/click statistics from the action trajectory."""
    total_actions = 0
    invalid_actions = 0
    invalid_clicks = 0
    no_progress_actions = 0
    for step in memory.trajectory:
      action = step.action
      action_type = str(action.get("action_type", "")).casefold()
      if action_type not in _NON_EXECUTABLE_ACTIONS:
        total_actions += 1
        continue
      if action_type != "dms_recovery":
        continue
      phase = str(action.get("phase", "")).casefold()
      attempted = action.get("attempted_action")
      if phase in _INVALID_FEEDBACK_PHASES and isinstance(attempted, dict):
        invalid_actions += 1
        attempted_type = str(attempted.get("action_type", "")).casefold()
        if attempted_type in _CLICK_ACTIONS:
          invalid_clicks += 1
      if phase == "anti_loop":
        no_progress_actions += 1
    memory.total_action_count = total_actions
    memory.invalid_action_count = invalid_actions
    memory.invalid_click_count = invalid_clicks
    memory.no_progress_action_count = no_progress_actions

  def retrieve(
      self,
      plan: PlanUnit,
      *,
      risk_threshold: float = 0.35,
      task_app_names: tuple[str, ...] = (),
  ) -> MemoryUnit | None:
    self.tick += 1
    if self.retrieval_strategy == "chronological_context":
      # Static Memory is never an action policy.  Its append-only trajectory
      # timeline is supplied to Planner/Actor as textual context instead.
      return None
    global_failure_rate = self.global_failure_rate()
    self.total_retrievals += 1
    try:
      query_pre, query_goal = self.embedder.encode([plan.precondition, plan.goal])
    except Exception as error:
      self.total_retrieval_misses += 1
      self.total_embedding_failures += 1
      self.last_retrieval_report = {
          "mode": "miss",
          "reason": (
              "Embedding service unavailable during retrieval: "
              f"{type(error).__name__}: {error}"
          ),
          "plan": {
              "precondition": plan.precondition,
              "goal": plan.goal,
          },
          "risk_threshold": risk_threshold,
          "global_failure_rate": global_failure_rate,
          "candidates": [],
      }
      self._save()
      return None
    task_apps = self._normalized_app_names(task_app_names)
    scored: list[tuple[float, float, float, float, bool, MemoryUnit]] = []
    for memory in self.memories.values():
      memory_apps = self._memory_app_names(memory)
      if task_apps and memory_apps and task_apps.isdisjoint(memory_apps):
        continue
      precondition_similarity = (
          self._cosine(query_pre, memory.precondition_embedding)
          if memory.precondition_embedding is not None else 0.0
      )
      goal_similarity = (
          self._cosine(query_goal, memory.goal_embedding)
          if memory.goal_embedding is not None else 0.0
      )
      score = precondition_similarity * goal_similarity
      risk = memory.risk_lower_bound(global_failure_rate=global_failure_rate)
      suppressed = risk >= risk_threshold
      scored.append((
          score, precondition_similarity, goal_similarity, risk, suppressed,
          memory,
      ))
    scored.sort(key=lambda item: (item[0], item[5].survival_value(self.tick)),
                reverse=True)
    self.last_retrieval_report = {
        "plan": {
            "precondition": plan.precondition,
            "goal": plan.goal,
        },
        "risk_threshold": risk_threshold,
        "global_failure_rate": global_failure_rate,
        "candidates": [
            {
                "memory_id": memory.id,
                "score": score,
                "precondition_similarity": precondition_similarity,
                "goal_similarity": goal_similarity,
                "risk_score": risk,
                "survival_value": memory.survival_value(self.tick),
                "suppressed": suppressed,
                "steps": len(memory.trajectory),
            }
            for (
                score, precondition_similarity, goal_similarity, risk,
                suppressed, memory,
            ) in scored[:5]
        ],
    }
    candidates = [item for item in scored if not item[4]]
    if not candidates:
      if scored:
        self.total_suppressed += 1
        self.last_retrieval_report["mode"] = "suppressed"
        self.last_retrieval_report["reason"] = "All candidates exceeded risk threshold."
      else:
        self.total_retrieval_misses += 1
        self.last_retrieval_report["mode"] = "miss"
        self.last_retrieval_report["reason"] = (
            "Memory bank is empty."
            if not self.memories else "No memory matched the task app scope."
        )
      self._save()
      return None
    score, _, _, _, _, hit = candidates[0]
    if score <= 0:
      self.total_retrieval_misses += 1
      self.last_retrieval_report["mode"] = "miss"
      self.last_retrieval_report["reason"] = "Top dual-factor score was non-positive."
      self._save()
      return None
    hit.reuse_count += 1
    hit.last_used_tick = self.tick
    self.total_retrieval_hits += 1
    self.last_retrieval_report["mode"] = "hit"
    self.last_retrieval_report["selected_memory_id"] = hit.id
    self._save()
    return hit

  @staticmethod
  def _normalized_app_names(app_names: tuple[str, ...] | list[str]) -> set[str]:
    return {
        re.sub(r"[^a-z0-9]+", "", str(app_name).casefold())
        for app_name in app_names
        if str(app_name).strip()
    }

  @classmethod
  def _memory_app_names(cls, memory: MemoryUnit) -> set[str]:
    description = memory.description or ""
    match = re.search(r"task_apps=([^;]+)", description)
    if match:
      return cls._normalized_app_names([
          item.strip() for item in match.group(1).split(",")
      ])
    match = re.search(r"task_apps=([^;]+)", memory.plan.precondition)
    if match:
      return cls._normalized_app_names([
          item.strip() for item in match.group(1).split(",")
      ])
    return set()

  def chronological_context(
      self, *, recent_limit: int = 10, task_name: str | None = None,
      task_app_names: tuple[str, ...] = (),
  ) -> str:
    """Return compact relevant Static Memory context without pruning the store."""
    if not self.static:
      return ""
    if recent_limit < 1:
      raise ValueError("recent_limit must be positive.")
    entries: list[str] = []
    ordered = sorted(
        self.memories.values(),
        key=lambda item: (item.created_tick, item.created_at, item.id),
    )
    if task_name or task_app_names:
      task_matches = [
          memory for memory in ordered if _matches_task(memory, task_name)
      ]
      app_matches = (
          [] if task_matches else [
              memory for memory in ordered if _matches_app(memory, task_app_names)
          ]
      )
      relevant = task_matches or app_matches
      if not relevant:
        return "No relevant historical interaction trajectories for this task or app."
    else:
      relevant = ordered

    recent = relevant[-recent_limit:]
    success_entries = [memory for memory in recent if memory.success is True]
    failed_entries = [memory for memory in recent if memory.success is not True]

    def render(memory: MemoryUnit, sequence: int) -> str:
      return (
          f"Candidate workflow {sequence}: Success={memory.success}; "
          f"Precondition={_compact_text(memory.plan.precondition)}; "
          f"Goal={_compact_text(memory.plan.goal)}; "
          f"Recorded steps={len(memory.trajectory)}\n"
          f"  Interpretation: "
          f"{'positive example' if memory.success else 'failed diagnostic; do not imitate blindly'}\n"
          f"  Trajectory excerpt:\n{_compact_trajectory(memory.trajectory)}"
      )

    if success_entries:
      entries.append(
          "Successful candidate workflows to adapt. These are procedural "
          "templates only, not current-state facts:"
      )
      entries.extend(
          render(memory, index)
          for index, memory in enumerate(success_entries, start=1)
      )
    if failed_entries:
      entries.append(
          "Failed candidate workflows to avoid. Use these only as negative "
          "diagnostics:"
      )
      entries.extend(
          render(memory, index)
          for index, memory in enumerate(failed_entries, start=1)
      )
    return "\n\n".join(entries)

  @staticmethod
  def _score(
      query_precondition: list[float], query_goal: list[float], memory: MemoryUnit
  ) -> float:
    """Paper Eq. Dual-Factor Similarity: cos(phi(pre)) * cos(phi(goal))."""
    if memory.precondition_embedding is None or memory.goal_embedding is None:
      return 0.0
    return (
        DMSMemoryStore._cosine(query_precondition, memory.precondition_embedding)
        * DMSMemoryStore._cosine(query_goal, memory.goal_embedding)
    )

  @staticmethod
  def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
      raise ValueError("Embedding vectors must be non-empty and have equal dimensions.")
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

  def record_verification(self, memory_id: str, *, verified: bool) -> bool:
    """Apply local verification feedback and enforce K-limit deletion."""
    memory = self.memories[memory_id]
    if not verified:
      memory.verification_failures += 1
    else:
      memory.verification_failures = 0
    if memory.verification_failures >= self.k_limit:
      del self.memories[memory_id]
      self.total_deleted_by_risk += 1
      self._save()
      return True
    self._save()
    return False

  def record_global_feedback(self, memory_ids: set[str], *, task_succeeded: bool) -> None:
    """Apply global task feedback only after benchmark success is known."""
    if task_succeeded:
      self.global_task_successes += 1
    else:
      self.global_task_failures += 1
    for memory_id in memory_ids:
      memory = self.memories.get(memory_id)
      if memory is None:
        continue
      if task_succeeded:
        memory.task_successes += 1
      else:
        memory.task_failures += 1
      memory.success = bool(task_succeeded)
    self._save()

  def global_failure_rate(self, *, alpha: float = 1.0, beta: float = 1.0) -> float:
    """Smoothed ecosystem failure rate used by dynamic risk thresholding."""
    if alpha <= 0 or beta <= 0:
      raise ValueError("Risk smoothing priors must be positive.")
    return (
        self.global_task_failures + alpha
    ) / (
        self.global_task_failures + self.global_task_successes + alpha + beta
    )

  def replace_if_better(self, memory_id: str, candidate: list[TrajectoryStep], *, verified: bool) -> bool:
    old = self.memories[memory_id]
    if verified and len(candidate) >= 2 and len(candidate) < len(old.trajectory):
      old.trajectory = candidate
      self._refresh_environment_feedback(old)
      self.total_replacements += 1
      self._save()
      return True
    return False

  def record_replay(self, memory_id: str) -> None:
    """Record that a retrieved memory was replayed rather than mutated."""
    if memory_id in self.memories:
      self.total_replays += 1
      self._save()

  def record_mutation(self, memory_id: str) -> None:
    """Record epsilon mutation after a safe retrieval hit."""
    if memory_id in self.memories:
      self.total_mutations += 1
      self._save()

  def prune(
      self, *, threshold: float | None = None, min_keep: int = 1,
      max_remove_fraction: float = 0.5,
  ) -> list[str]:
    """Paper §3.2.3 capacity-triggered pruning with adaptive expansion."""
    memory_size = len(self.memories)
    if self.static or memory_size < 2:
      self.last_pruned_ids = []
      self.last_prune_report = {
          "reason": "static_or_too_small",
          "memory_size": memory_size,
          "current_capacity": self.current_capacity,
      }
      return []
    if memory_size < self.current_capacity:
      self.last_pruned_ids = []
      self.last_prune_report = {
          "reason": "below_capacity",
          "memory_size": memory_size,
          "current_capacity": self.current_capacity,
          "removed_count": 0,
      }
      return []
    if min_keep < 1:
      raise ValueError("min_keep must be positive.")
    if not 0 < max_remove_fraction <= 1:
      raise ValueError("max_remove_fraction must be in (0, 1].")
    scored = sorted(
        (
            (key, memory.survival_value(self.tick), memory)
            for key, memory in self.memories.items()
        ),
        key=lambda item: item[1],
    )
    values = [value for _, value, _ in scored]
    cutoff = threshold if threshold is not None else self.elbow_threshold(values)
    population_mean = sum(values) / len(values)
    if cutoff >= population_mean:
      previous_capacity = self.current_capacity
      if self.current_capacity < self.capacity_max:
        self.current_capacity = min(
            self.current_capacity + self.capacity_step, self.capacity_max
        )
        reason = "expanded_capacity"
      else:
        reason = "high_quality_at_capacity_max"
      self.last_pruned_ids = []
      self.last_prune_report = {
          "reason": reason,
          "cutoff": cutoff,
          "population_mean": population_mean,
          "memory_size": memory_size,
          "previous_capacity": previous_capacity,
          "current_capacity": self.current_capacity,
          "capacity_max": self.capacity_max,
          "removed_count": 0,
      }
      self._save()
      return []
    candidates = [key for key, value, _ in scored if value <= cutoff]
    max_remove = min(
        len(self.memories) - min_keep,
        max(1, int(len(self.memories) * max_remove_fraction)),
    )
    removed = candidates[:max_remove]
    if not removed:
      self.last_pruned_ids = []
      self.last_prune_report = {
          "reason": "no_low_value_tail",
          "cutoff": cutoff,
          "population_mean": population_mean,
          "memory_size_before": len(self.memories),
          "current_capacity": self.current_capacity,
          "removed_count": 0,
      }
      return []
    for key in removed:
      del self.memories[key]
    self.last_pruned_ids = removed
    self.total_pruned += len(removed)
    self.last_prune_report = {
        "reason": "pruned_low_value_tail",
        "cutoff": cutoff,
        "population_mean": population_mean,
        "memory_size_before": len(scored),
        "memory_size_after": len(self.memories),
        "current_capacity": self.current_capacity,
        "removed_count": len(removed),
        "removed_ids": removed,
    }
    self._save()
    return removed

  def memory_stats(self) -> dict[str, float | int | str]:
    """Return aggregate memory diagnostics for per-trial experiment logs."""
    base_stats = {
        "memory_current_capacity": self.current_capacity,
        "memory_total_created": self.total_created,
        "memory_total_retrievals": self.total_retrievals,
        "memory_total_retrieval_hits": self.total_retrieval_hits,
        "memory_total_retrieval_misses": self.total_retrieval_misses,
        "memory_total_suppressed": self.total_suppressed,
        "memory_total_replays": self.total_replays,
        "memory_total_mutations": self.total_mutations,
        "memory_total_replacements": self.total_replacements,
        "memory_total_pruned": self.total_pruned,
        "memory_total_deleted_by_risk": self.total_deleted_by_risk,
        "memory_total_embedding_failures": self.total_embedding_failures,
        "memory_retrieval_hit_rate": (
            self.total_retrieval_hits / self.total_retrievals
            if self.total_retrievals else ""
        ),
        "memory_reuse_rate": (
            self.total_replays / self.total_retrievals
            if self.total_retrievals else ""
        ),
        "memory_mutation_rate": (
            self.total_mutations / self.total_retrievals
            if self.total_retrievals else ""
        ),
    }
    if not self.memories:
      return {
          "memory_size": 0,
          "memory_survival_min": "",
          "memory_survival_mean": "",
          "memory_survival_max": "",
          "memory_invalid_click_rate_mean": "",
          "memory_task_completion_mean": "",
          "memory_last_pruned_count": len(self.last_pruned_ids),
          **base_stats,
      }
    components = [
        memory.survival_components(self.tick) for memory in self.memories.values()
    ]
    survival_values = [item["survival_value"] for item in components]
    invalid_rates = [item["invalid_click_rate"] for item in components]
    completion_rates = [item["task_completion"] for item in components]
    return {
        "memory_size": len(self.memories),
        "memory_survival_min": min(survival_values),
        "memory_survival_mean": sum(survival_values) / len(survival_values),
        "memory_survival_max": max(survival_values),
        "memory_invalid_click_rate_mean": (
            sum(invalid_rates) / len(invalid_rates)
        ),
        "memory_task_completion_mean": (
            sum(completion_rates) / len(completion_rates)
        ),
        "memory_last_pruned_count": len(self.last_pruned_ids),
        **base_stats,
    }

  @staticmethod
  def elbow_threshold(sorted_values: list[float]) -> float:
    """Knee of the ascending survival-value curve by maximum line distance."""
    if len(sorted_values) < 3:
      return float("-inf")
    x1, y1 = 0.0, sorted_values[0]
    x2, y2 = float(len(sorted_values) - 1), sorted_values[-1]
    denominator = sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    if denominator == 0:
      return sorted_values[-1]
    distances = [
        abs((y2 - y1) * index - (x2 - x1) * value + x2 * y1 - y2 * x1) / denominator
        for index, value in enumerate(sorted_values)
    ]
    elbow_index = max(range(1, len(sorted_values) - 1), key=distances.__getitem__)
    return sorted_values[elbow_index]

  def _save(self) -> None:
    self.path.parent.mkdir(parents=True, exist_ok=True)
    if self.static:
      data = {
          "format": "dms_memory_construction_v1",
          "description": (
              "Static baseline memory: append-only DMS 3.2.1 construction "
              "format m=(p,tau,s_meta), without DMS retrieval/replacement/"
              "risk/pruning dynamic updates."
          ),
          "memories": [
              self._serialize_static_memory(memory)
              for memory in self.memories.values()
          ],
      }
      self.path.write_text(
          json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
      )
      return
    data = {
        "format": "dms_hierarchical_memory_v1",
        "description": (
            "Dynamic DMS memory with intent-layer p, action-layer tau, "
            "environment feedback, survival value, pruning diagnostics, and "
            "trajectory files stored separately from metadata."
        ),
        "tick": self.tick,
        "global_task_successes": self.global_task_successes,
        "global_task_failures": self.global_task_failures,
        "capacity_min": self.capacity_min,
        "capacity_max": self.capacity_max,
        "capacity_step": self.capacity_step,
        "current_capacity": self.current_capacity,
        "last_prune_report": self.last_prune_report,
        "last_retrieval_report": self.last_retrieval_report,
        "total_created": self.total_created,
        "total_retrievals": self.total_retrievals,
        "total_retrieval_hits": self.total_retrieval_hits,
        "total_retrieval_misses": self.total_retrieval_misses,
        "total_suppressed": self.total_suppressed,
        "total_replays": self.total_replays,
        "total_mutations": self.total_mutations,
        "total_replacements": self.total_replacements,
        "total_pruned": self.total_pruned,
        "total_deleted_by_risk": self.total_deleted_by_risk,
        "total_embedding_failures": self.total_embedding_failures,
        "memories": [
            self._serialize_dynamic_memory(memory)
            for memory in self.memories.values()
        ],
    }
    self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

  def _serialize_dynamic_memory(self, memory: MemoryUnit) -> dict[str, object]:
    """Persist a DMS memory as explicit intent/action/utility/meta layers."""
    trajectory_path = self._write_dynamic_trajectory(memory)
    return {
        "id": memory.id,
        "intent": {
            "precondition": memory.plan.precondition,
            "goal": memory.plan.goal,
        },
        "action_experience": {
            "trajectory_path": trajectory_path,
            "step_count": len(memory.trajectory),
        },
        "feedback": {
            "success": memory.success,
            "task_successes": memory.task_successes,
            "task_failures": memory.task_failures,
            "total_action_count": memory.total_action_count,
            "invalid_action_count": memory.invalid_action_count,
            "invalid_click_count": memory.invalid_click_count,
            "no_progress_action_count": memory.no_progress_action_count,
        },
        "meta": {
            "description": memory.description,
            "created_tick": memory.created_tick,
            "last_used_tick": memory.last_used_tick,
            "created_at": memory.created_at,
            "reuse_count": memory.reuse_count,
            "verification_failures": memory.verification_failures,
        },
        "embeddings": {
            "precondition_embedding": memory.precondition_embedding,
            "goal_embedding": memory.goal_embedding,
            "status": memory.embedding_status,
            "error": memory.embedding_error,
        },
    }

  def _write_dynamic_trajectory(self, memory: MemoryUnit) -> str:
    """Write a dynamic memory trajectory outside the metadata bank."""
    self.trajectory_dir.mkdir(parents=True, exist_ok=True)
    if memory.trajectory_path:
      path = self._resolve_trajectory_path(memory.trajectory_path)
    else:
      path = self.trajectory_dir / f"{memory.id}.json"
    payload = [
        {"observation": step.observation, "action": step.action}
        for step in memory.trajectory
    ]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
      memory.trajectory_path = str(path.relative_to(self.path.parent))
    except ValueError:
      memory.trajectory_path = str(path)
    return memory.trajectory_path

  def _resolve_trajectory_path(self, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
      return path
    return self.path.parent / path

  def hierarchical_records(self) -> list[dict[str, object]]:
    """Return explicit intent/action/meta records for reports or inspection."""
    return [
        {
            "id": memory.id,
            "intent": {
                "precondition": memory.plan.precondition,
                "goal": memory.plan.goal,
            },
            "action_experience": {
                "trajectory": [
                    {"observation": step.observation, "action": step.action}
                    for step in memory.trajectory
                ],
                "step_count": len(memory.trajectory),
            },
            "utility": memory.survival_components(self.tick),
            "environment_feedback": {
                "task_successes": memory.task_successes,
                "task_failures": memory.task_failures,
                "total_action_count": memory.total_action_count,
                "invalid_action_count": memory.invalid_action_count,
                "invalid_click_count": memory.invalid_click_count,
                "no_progress_action_count": memory.no_progress_action_count,
            },
            "meta": {
                "success": memory.success,
                "description": memory.description,
                "created_tick": memory.created_tick,
                "last_used_tick": memory.last_used_tick,
                "created_at": memory.created_at,
                "reuse_count": memory.reuse_count,
                "verification_failures": memory.verification_failures,
            },
        }
        for memory in self.memories.values()
    ]

  @staticmethod
  def _serialize_static_memory(memory: MemoryUnit) -> dict[str, object]:
    """Persist Static Memory exactly as DMS 3.2.1 memory construction."""
    return {
        "p": {
            "precondition": memory.plan.precondition,
            "goal": memory.plan.goal,
        },
        "tau": [
            {
                "o": step.observation,
                "a": step.action,
            }
            for step in memory.trajectory
        ],
        "s_meta": {
            "memory_id": memory.id,
            "success": memory.success,
            "description": memory.description,
            "created_at": memory.created_at,
        },
    }

  def _load(self) -> None:
    raw = json.loads(self.path.read_text(encoding="utf-8"))
    self.tick = raw.get("tick", 0)
    self.global_task_successes = raw.get("global_task_successes", 0)
    self.global_task_failures = raw.get("global_task_failures", 0)
    self.capacity_min = raw.get("capacity_min", self.capacity_min)
    self.capacity_max = raw.get("capacity_max", self.capacity_max)
    self.capacity_step = raw.get("capacity_step", self.capacity_step)
    self.current_capacity = raw.get("current_capacity", self.capacity_min)
    self.last_prune_report = raw.get("last_prune_report", {}) or {}
    self.last_retrieval_report = raw.get("last_retrieval_report", {}) or {}
    self.total_created = raw.get("total_created", self.total_created)
    self.total_retrievals = raw.get("total_retrievals", self.total_retrievals)
    self.total_retrieval_hits = raw.get(
        "total_retrieval_hits", self.total_retrieval_hits
    )
    self.total_retrieval_misses = raw.get(
        "total_retrieval_misses", self.total_retrieval_misses
    )
    self.total_suppressed = raw.get("total_suppressed", self.total_suppressed)
    self.total_replays = raw.get("total_replays", self.total_replays)
    self.total_mutations = raw.get("total_mutations", self.total_mutations)
    self.total_replacements = raw.get(
        "total_replacements", self.total_replacements
    )
    self.total_pruned = raw.get("total_pruned", self.total_pruned)
    self.total_deleted_by_risk = raw.get(
        "total_deleted_by_risk", self.total_deleted_by_risk
    )
    self.total_embedding_failures = raw.get(
        "total_embedding_failures", self.total_embedding_failures
    )
    for item in raw.get("memories", []):
      if self.static and "p" in item and "tau" in item:
        meta = item.get("s_meta", {})
        memory = MemoryUnit(
            plan=PlanUnit(**item["p"]),
            trajectory=[
                TrajectoryStep(observation=step["o"], action=step["a"])
                for step in item["tau"]
            ],
            success=meta.get("success"),
            description=meta.get("description", ""),
            id=meta.get("memory_id", str(uuid4())),
            created_at=meta.get("created_at", _now()),
        )
        self.memories[memory.id] = memory
        continue
      if "intent" in item and "action_experience" in item:
        feedback = item.get("feedback", {}) or {}
        meta = item.get("meta", {}) or {}
        embeddings = item.get("embeddings", {}) or {}
        action_experience = item.get("action_experience", {})
        trajectory_path = str(action_experience.get("trajectory_path", "") or "")
        trajectory = action_experience.get("trajectory", [])
        if trajectory_path:
          resolved_path = self._resolve_trajectory_path(trajectory_path)
          if resolved_path.exists():
            trajectory = json.loads(resolved_path.read_text(encoding="utf-8"))
        memory = MemoryUnit(
            plan=PlanUnit(**item["intent"]),
            trajectory=[
                TrajectoryStep(
                    observation=step.get("observation", step.get("o", "")),
                    action=step.get("action", step.get("a", {})),
                )
                for step in trajectory
            ],
            success=feedback.get("success"),
            description=meta.get("description", ""),
            id=item.get("id", str(uuid4())),
            reuse_count=meta.get("reuse_count", 0),
            verification_failures=meta.get("verification_failures", 0),
            task_successes=feedback.get("task_successes", 0),
            task_failures=feedback.get("task_failures", 0),
            total_action_count=feedback.get("total_action_count", 0),
            invalid_action_count=feedback.get("invalid_action_count", 0),
            invalid_click_count=feedback.get("invalid_click_count", 0),
            no_progress_action_count=feedback.get("no_progress_action_count", 0),
            created_tick=meta.get("created_tick", 0),
            last_used_tick=meta.get("last_used_tick", 0),
            created_at=meta.get("created_at", _now()),
            precondition_embedding=embeddings.get("precondition_embedding"),
            goal_embedding=embeddings.get("goal_embedding"),
            trajectory_path=trajectory_path,
            embedding_status=embeddings.get("status", "ready"),
            embedding_error=embeddings.get("error", ""),
        )
        self._refresh_environment_feedback(memory)
        self.memories[memory.id] = memory
        continue
      # Drop fields written by pre-paper-alignment builds. They were an
      # engineering extension, not part of the DMS memory definition.
      item.pop("total_clicks", None)
      item.pop("invalid_clicks", None)
      item["plan"] = PlanUnit(**item["plan"])
      item["trajectory"] = [TrajectoryStep(**step) for step in item["trajectory"]]
      memory = MemoryUnit(**item)
      self._refresh_environment_feedback(memory)
      self.memories[memory.id] = memory
