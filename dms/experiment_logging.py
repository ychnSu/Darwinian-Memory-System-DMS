"""Crash-resilient per-trial experiment ledger for DMS evaluation modes."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from typing import Any

from android_world import constants


CSV_FIELDS = (
    "timestamp_utc", "agent_name", "task_template", "k", "seed", "success",
    "SA", "cumulative_SA", "episode_length", "run_time_s", "status",
    "token_total", "token_prompt", "token_completion", "exception_info", "goal",
    "memory_size", "memory_survival_min", "memory_survival_mean",
    "memory_survival_max", "memory_invalid_click_rate_mean",
    "memory_task_completion_mean", "memory_last_pruned_count",
    "memory_current_capacity", "memory_total_created",
    "memory_total_retrievals", "memory_total_retrieval_hits",
    "memory_total_retrieval_misses", "memory_total_suppressed",
    "memory_total_replays", "memory_total_mutations",
    "memory_total_replacements", "memory_total_pruned",
    "memory_total_deleted_by_risk", "memory_total_embedding_failures",
    "memory_retrieval_hit_rate", "memory_reuse_rate", "memory_mutation_rate",
)
_LEGACY_CSV_FIELDS = (
    "timestamp_utc", "agent_name", "task_template", "k", "seed", "success",
    "SA", "cumulative_SA", "episode_length", "run_time_s", "status",
    "token_total", "token_prompt", "token_completion", "exception_info", "goal",
)
_PRE_CAPACITY_CSV_FIELDS = tuple(
    field for field in CSV_FIELDS if field != "memory_current_capacity"
)
_REQUIRED_EXISTING_FIELDS = (
    "timestamp_utc", "agent_name", "task_template", "k", "SA",
    "cumulative_SA", "episode_length", "status", "goal",
)


class IncrementalCsvLedger:
  """Append one fsync'ed row immediately after every completed task instance."""

  def __init__(self, path: str | Path, *, agent_name: str):
    self.path = Path(path)
    self.agent_name = agent_name
    self._fieldnames = CSV_FIELDS
    self._successful_trials = 0
    self._valid_trials = 0
    self._restore_totals()

  def _restore_totals(self) -> None:
    if not self.path.exists() or self.path.stat().st_size == 0:
      return
    with self.path.open("r", newline="", encoding="utf-8") as handle:
      reader = csv.DictReader(handle)
      existing_fields = tuple(reader.fieldnames or ())
      if not all(field in existing_fields for field in _REQUIRED_EXISTING_FIELDS):
        raise ValueError(f"Existing CSV has an incompatible header: {self.path}")
      self._fieldnames = existing_fields
      for row in reader:
        if row.get("SA") in ("0", "1"):
          self._valid_trials += 1
          self._successful_trials += int(row["SA"])

  def append_episode(self, episode: dict[str, Any]) -> None:
    """Persist a trial result before the next task begins.

    ``SA`` is the per-trial Success Accuracy (1/0); ``cumulative_SA`` is the
    running mean over valid trials. Exceptions are retained but excluded from
    the SA denominator, matching AndroidWorld's aggregate accounting.
    """
    outcome = episode.get(constants.EpisodeConstants.IS_SUCCESSFUL)
    valid = isinstance(outcome, (int, float)) and not isinstance(outcome, bool) and not math.isnan(float(outcome))
    # bool is still a legitimate external result representation.
    valid = valid or isinstance(outcome, bool)
    sa = int(float(outcome) > 0.5) if valid else None
    if sa is not None:
      self._valid_trials += 1
      self._successful_trials += sa
    cumulative_sa = (
        self._successful_trials / self._valid_trials if self._valid_trials else ""
    )
    exception = episode.get(constants.EpisodeConstants.EXCEPTION_INFO)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "agent_name": self.agent_name,
        "task_template": episode.get(constants.EpisodeConstants.TASK_TEMPLATE, ""),
        "k": int(episode.get(constants.EpisodeConstants.INSTANCE_ID, 0)) + 1,
        "seed": episode.get(constants.EpisodeConstants.SEED, ""),
        "success": "" if sa is None else bool(sa),
        "SA": "" if sa is None else sa,
        "cumulative_SA": cumulative_sa,
        "episode_length": episode.get(constants.EpisodeConstants.EPISODE_LENGTH, ""),
        "run_time_s": episode.get(constants.EpisodeConstants.RUN_TIME, ""),
        "status": "exception" if exception else ("success" if sa else "failure"),
        "token_total": episode.get("token_total", ""),
        "token_prompt": episode.get("token_prompt", ""),
        "token_completion": episode.get("token_completion", ""),
        "exception_info": exception or "",
        "goal": episode.get(constants.EpisodeConstants.GOAL, ""),
        "memory_size": episode.get("memory_size", ""),
        "memory_survival_min": episode.get("memory_survival_min", ""),
        "memory_survival_mean": episode.get("memory_survival_mean", ""),
        "memory_survival_max": episode.get("memory_survival_max", ""),
        "memory_invalid_click_rate_mean": episode.get(
            "memory_invalid_click_rate_mean", ""
        ),
        "memory_task_completion_mean": episode.get(
            "memory_task_completion_mean", ""
        ),
        "memory_last_pruned_count": episode.get("memory_last_pruned_count", ""),
        "memory_current_capacity": episode.get("memory_current_capacity", ""),
        "memory_total_created": episode.get("memory_total_created", ""),
        "memory_total_retrievals": episode.get("memory_total_retrievals", ""),
        "memory_total_retrieval_hits": episode.get(
            "memory_total_retrieval_hits", ""
        ),
        "memory_total_retrieval_misses": episode.get(
            "memory_total_retrieval_misses", ""
        ),
        "memory_total_suppressed": episode.get("memory_total_suppressed", ""),
        "memory_total_replays": episode.get("memory_total_replays", ""),
        "memory_total_mutations": episode.get("memory_total_mutations", ""),
        "memory_total_replacements": episode.get(
            "memory_total_replacements", ""
        ),
        "memory_total_pruned": episode.get("memory_total_pruned", ""),
        "memory_total_deleted_by_risk": episode.get(
            "memory_total_deleted_by_risk", ""
        ),
        "memory_total_embedding_failures": episode.get(
            "memory_total_embedding_failures", ""
        ),
        "memory_retrieval_hit_rate": episode.get(
            "memory_retrieval_hit_rate", ""
        ),
        "memory_reuse_rate": episode.get("memory_reuse_rate", ""),
        "memory_mutation_rate": episode.get("memory_mutation_rate", ""),
    }
    row = {field: row.get(field, "") for field in self._fieldnames}
    self.path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not self.path.exists() or self.path.stat().st_size == 0
    with self.path.open("a", newline="", encoding="utf-8") as handle:
      writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
      if write_header:
        writer.writeheader()
      writer.writerow(row)
      handle.flush()
      os.fsync(handle.fileno())
