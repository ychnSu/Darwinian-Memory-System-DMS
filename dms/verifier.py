"""Paper-aligned History-First verifier for DMS."""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Protocol

from dms.controller import ExecutionResult
from dms.memory import PlanUnit, TrajectoryStep

if TYPE_CHECKING:
  from dms.androidworld_agent import Observation, Verification
  from android_world.env import json_action


class VerifierModel(Protocol):
  def complete(
      self, prompt: str, screenshot_png: bytes | None = None, *, temperature: float = 0.0,
      json_schema: dict[str, object] | None = None,
      structured_output: bool | None = None,
  ) -> str: ...


VERIFIER_PROMPT = """Role: You are an expert Android Task Verifier.
Determine whether the execution history achieved the stated sub-plan.

Input information:
1. Current sub-plan: Precondition: {precondition}; Goal: {subgoal}
2. Execution history (PRIMARY source of truth):
{history}
3. Final screenshot and UI state (SECONDARY contradiction check):
Foreground activity: {activity}
Visible UI elements:
{ui_text}

Verification logic (History-First):
1. First check whether the action history logically performs the Goal.
2. If it does, default to verified success.
3. Reject only when the final visual/UI evidence explicitly contradicts success,
such as a visible error, wrong app, or an expected dialog still present.
Return JSON only: {{"verified success": <bool>, "reason": "<string>"}}.
"""

def parse_verification(response: str) -> ExecutionResult:
  """Parse the paper's verifier JSON response into the DMS result type."""
  match = re.search(r"\{.*?\}", response, flags=re.DOTALL)
  if not match:
    raise ValueError("Verifier response does not contain JSON.")
  payload = json.loads(match.group())
  outcome = payload.get("verified success")
  if not isinstance(outcome, bool):
    raise ValueError("Verifier JSON requires boolean 'verified success'.")
  reason = payload.get("reason")
  if not isinstance(reason, str):
    raise ValueError("Verifier JSON requires string 'reason'.")
  return ExecutionResult(verified=outcome, reason=reason)


class QwenVLVerifier:
  """Produces `ExecutionResult` using the DMS paper's verifier procedure."""

  def __init__(self, model: VerifierModel):
    self._model = model
    self._last_audit: dict[str, Any] = {}

  @property
  def last_audit(self) -> dict[str, Any]:
    """Structured evidence from the latest verifier decision."""
    return dict(self._last_audit)

  def evaluate(
      self,
      task_goal: str,
      plan: PlanUnit,
      history: list[TrajectoryStep],
      final_observation: Observation,
  ) -> ExecutionResult:
    history_text = "\n".join(
        f"Observation: {step.observation}\nAction: {step.action}" for step in history
    ) or "No actions were executed."
    prompt = VERIFIER_PROMPT.format(
        precondition=plan.precondition,
        subgoal=plan.goal,
        history=history_text,
        activity=final_observation.foreground_activity,
        ui_text=final_observation.ui_text or "Not available",
    )
    # Import lazily so tests and non-VLM scaffolding need not load Pillow.
    from dms.actor import screenshot_to_png
    answer = self._model.complete(
        prompt, screenshot_to_png(final_observation.screenshot), temperature=0.0,
        structured_output=False,
    )
    try:
      result = parse_verification(answer)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
      self._last_audit = {"response": answer, "accepted": False, "error": str(error)}
      raise
    self._last_audit = {
        "response": answer,
        "accepted": True,
        "verified": result.verified,
        "reason": result.reason,
    }
    return result


class AndroidWorldVerifierAdapter:
  """Adapts QwenVLVerifier results to the Agent's sub-plan control flow."""

  def __init__(self, verifier: QwenVLVerifier):
    self._verifier = verifier

  def verify(
      self,
      goal: str,
      plan: PlanUnit,
      before: Observation,
      after: Observation,
      action: json_action.JSONAction,
      history: list[TrajectoryStep],
  ) -> Verification:
    from dms.androidworld_agent import Verification
    result = self._verifier.evaluate(goal, plan, history, after)
    explicit_infeasible = action.action_type == "complete" and action.success is False
    subplan_complete = result.verified and not explicit_infeasible
    return Verification(
        subplan_complete=subplan_complete,
        task_complete=False,
        verified=result.verified,
        subplan_failed=explicit_infeasible or not result.verified,
    )
