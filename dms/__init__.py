"""Training-free Darwinian Memory System components.

Core memory utilities remain importable without AndroidWorld's optional runtime
dependencies; Agent/VLM adapters are loaded only when explicitly requested.
"""
from importlib import import_module

from dms.controller import DMSController, ExecutionResult
from dms.embeddings import (
    EmbeddingProvider,
    LocalEmbeddingProvider,
    RemoteEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from dms.memory import DMSMemoryStore, MemoryUnit, PlanUnit, TrajectoryStep
from dms.policy import DMSPolicy

__all__ = [
    "AndroidWorldVerifierAdapter", "DMSAndroidWorldAgent", "DMSController",
    "DMSMemoryStore", "DMSPolicy", "EmbeddingProvider", "ExecutionResult",
    "HierarchicalDMSPlanner", "MemoryUnit", "Observation", "PlanUnit",
    "LocalEmbeddingProvider", "QwenVLActor", "QwenVLVerifier",
    "RemoteEmbeddingProvider", "SentenceTransformerEmbeddingProvider",
    "DMSExperimentRunner", "TaskRunResult", "TrajectoryStep", "Verification", "parse_codeact_action", "parse_codeact_actions", "parse_json_action", "parse_plan_batch",
    "parse_verification",
]

_LAZY_EXPORTS = {
    "DMSAndroidWorldAgent": ("dms.androidworld_agent", "DMSAndroidWorldAgent"),
    "Observation": ("dms.androidworld_agent", "Observation"),
    "Verification": ("dms.androidworld_agent", "Verification"),
    "HierarchicalDMSPlanner": ("dms.planner", "HierarchicalDMSPlanner"),
    "parse_plan_batch": ("dms.planner", "parse_plan_batch"),
    "QwenVLActor": ("dms.actor", "QwenVLActor"),
    "parse_codeact_action": ("dms.actor", "parse_codeact_action"),
    "parse_codeact_actions": ("dms.actor", "parse_codeact_actions"),
    "parse_json_action": ("dms.actor", "parse_json_action"),
    "QwenVLVerifier": ("dms.verifier", "QwenVLVerifier"),
    "AndroidWorldVerifierAdapter": ("dms.verifier", "AndroidWorldVerifierAdapter"),
    "parse_verification": ("dms.verifier", "parse_verification"),
    "DMSExperimentRunner": ("dms.task_runner", "DMSExperimentRunner"),
    "TaskRunResult": ("dms.task_runner", "TaskRunResult"),
}


def __getattr__(name: str):
  if name not in _LAZY_EXPORTS:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  module_name, attribute = _LAZY_EXPORTS[name]
  value = getattr(import_module(module_name), attribute)
  globals()[name] = value
  return value
