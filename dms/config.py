"""Experiment-mode configuration for the paper's three comparable baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ExperimentModeConfig:
  """Behavioral contract for one experiment arm.

  The VLM, Planner, Actor, Verifier, task seed, and AndroidWorld suite remain
  identical across modes.  Only memory behavior changes.
  """

  name: str
  memory_mode: Literal["zero_shot", "static", "dms"]
  memory_filename: str | None
  retrieval_strategy: Literal["none", "chronological_context", "dual_factor"]
  use_embeddings: bool
  enable_replay: bool
  enable_risk_gate: bool
  enable_mutation: bool
  enable_replacement: bool
  enable_global_feedback: bool
  enable_pruning: bool


ZERO_SHOT_CONFIG = ExperimentModeConfig(
    name="zero_shot",
    memory_mode="zero_shot",
    memory_filename=None,
    retrieval_strategy="none",
    use_embeddings=False,
    enable_replay=False,
    enable_risk_gate=False,
    enable_mutation=False,
    enable_replacement=False,
    enable_global_feedback=False,
    enable_pruning=False,
)

STATIC_MEMORY_CONFIG = ExperimentModeConfig(
    name="static_memory",
    memory_mode="static",
    memory_filename="static_memory.json",
    # Static Memory is prompt context only: it never selects or replays an
    # executable trajectory.
    retrieval_strategy="chronological_context",
    use_embeddings=False,
    enable_replay=False,
    enable_risk_gate=False,
    enable_mutation=False,
    enable_replacement=False,
    enable_global_feedback=False,
    enable_pruning=False,
)

DMS_CONFIG = ExperimentModeConfig(
    name="dms",
    memory_mode="dms",
    memory_filename="dms_memory.json",
    retrieval_strategy="dual_factor",
    use_embeddings=True,
    enable_replay=True,
    enable_risk_gate=True,
    enable_mutation=True,
    enable_replacement=True,
    enable_global_feedback=True,
    enable_pruning=True,
)


def get_experiment_mode(agent_name: str) -> ExperimentModeConfig:
  """Return the explicit baseline/DMS contract selected by ``--agent_name``."""
  configs = {
      "zero_shot": ZERO_SHOT_CONFIG,
      "static_memory": STATIC_MEMORY_CONFIG,
      "dms": DMS_CONFIG,
  }
  try:
    return configs[agent_name]
  except KeyError as error:
    raise ValueError(f"Unknown DMS experiment mode: {agent_name}") from error
