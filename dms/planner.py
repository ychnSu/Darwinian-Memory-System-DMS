"""Paper-aligned hierarchical Planner for Darwinian Memory System.

Section 3.1 defines P(T, o_t, q) = {p_1, ..., p_k}, k <= 5, with
p_i = <Precondition, Goal>.  This module implements that exact planning
boundary; model inference is injected through a small protocol and is not
performed here.
"""
from __future__ import annotations

from collections import deque
import json
import re
from typing import TYPE_CHECKING, Any, Protocol

from dms.actor import screenshot_to_png
from dms.memory import PlanUnit

if TYPE_CHECKING:
  from dms.androidworld_agent import Observation


class PlannerModel(Protocol):
  """A VLM/LLM backend used later by the planner, not tied to vLLM yet."""

  def complete(
      self, prompt: str, screenshot_png: bytes | None = None, *, temperature: float = 0.0,
      max_tokens: int = 512, json_schema: dict[str, object] | None = None,
      structured_output: bool | None = None,
  ) -> str: ...


PLANNER_PROMPT = """You are an Android Task Planner. Your role is to break an
overall Android task into a sequence of 1 to 5 functional plans, assigning
each plan to a specialized agent.

You are given the user's Overall Goal, the current device screenshot, visible
UI elements in JSON form, the current visible Android activity, and the
complete task history from all prior planning cycles. Use the history: do not
repeat failed work, and account for completed work.

AndroidWorld task metadata is supplied to prevent app-scope hallucinations.
Unless the Overall Goal explicitly requires another app, keep plans within the
task-scoped app(s). When a plan needs an app to be opened, name a task-scoped
app rather than an unrelated application.

Create functional goals, not low-level tap instructions. Prefer a single next
functional plan that can be advanced from the current screen; use multiple
assignments only when the next stages are certain and do not require guessing a
future screen. Each assignment's
`task` should preferably be written as:
Precondition: <expected starting state>. Goal: <functional objective>.
The first task may use `Precondition: None`, and a task may omit a precondition
when none is useful. Generate no more than five assignments.

Before using any historical memory, anchor the current state from the current
screenshot, Visible UI elements, Accessibility UI tree, and foreground
activity. The Precondition you write must describe that current visible state,
not a historical state from memory. Use memory only after this anchoring step
to choose the next procedural move.

Available specialized agents:
{agents}

You must call exactly one planning tool. To create plans, respond only with:
<tool_call>
{{"name":"set_tasks_with_agents","arguments":{{"task_assignments":[{{"task":"Precondition: ... Goal: ...","agent":"CodeActAgent"}}]}}}}
</tool_call>
Use one to five task assignments and choose an agent from the available agents.
When, and only when, the Overall Goal has already been achieved, respond only
with:
<tool_call>
{{"name":"complete_goal","arguments":{{"message":"The task is complete."}}}}
</tool_call>

Overall goal:
{goal}

AndroidWorld task template:
{task_name}

AndroidWorld task family:
{task_family}

Task-scoped app(s):
{task_app_names}

Task parameters:
{task_params}

Current foreground activity:
{activity}

Visible UI elements:
{ui_text}

Accessibility UI tree:
{ui_tree_text}

Task history:
{history}

Chronological static-memory candidate workflows:
Use this section only as prior procedural experience for similar tasks. Treat
each historical trajectory as a candidate workflow to adapt, not as a current
fact. It is not evidence of the current device state. Never infer that the
current screen, app, tab, dialog, timer, stopwatch, note, file, or any other UI
state already matches a historical observation.
Current state must be determined only from the current screenshot, Visible UI elements, Accessibility UI tree, and current foreground activity above.
Historical actions and indices may be stale; do not copy them as current low-level instructions.
If the current UI does not visibly show achieved goal evidence, do not call complete_goal.
Do not create a plan whose only purpose is to "ensure" or assert an already-completed state.
{static_memory_history}
"""

PLANNER_RETRY_SUFFIX = """

Your previous candidate was rejected by the planner validator for this reason:
{reason}
Generate a fresh replacement now, using exactly one <tool_call> envelope and
one valid JSON object. Do not include Markdown, placeholders, or commentary.
Every task assignment must be a short, plain-language Android functional goal.
"""

_PLACEHOLDER_PLAN_VALUES = {"...", "n/a", "na", "none", "unknown", "todo"}
_SERIALIZATION_TOKENS = {
    "json", "plan", "plans", "precondition", "goal", "subgoal", "null",
    "true", "false",
}


def _clean_plan_text(value: str, field_name: str) -> str:
  """Recover plain-language plan text from common small-model JSON artifacts.

  The outer response and field aliases are handled separately.  Here we only
  remove serialization residue that surrounds otherwise usable text; a field
  made solely of JSON syntax is still rejected below.
  """
  normalized = re.sub(r"```(?:json)?", " ", value, flags=re.IGNORECASE)
  normalized = re.sub(r"[{}\[\]]", " ", normalized)
  normalized = re.sub(r"\b(?:json|plans?|precondition|goal|subgoal)\s*:\s*", " ", normalized, flags=re.IGNORECASE)
  normalized = " ".join(normalized.strip().split("` \t\r\n,:;"))
  if len(normalized) < 3 or len(normalized) > 360:
    raise ValueError(f"{field_name} must be between 3 and 360 characters.")
  if normalized.casefold().rstrip(".") == "none" and field_name == "Precondition":
    return "None"
  if normalized.casefold() in _PLACEHOLDER_PLAN_VALUES:
    raise ValueError(f"{field_name} cannot be a placeholder.")
  words = re.findall(r"[a-zA-Z]+", normalized.casefold())
  if not any(word not in _SERIALIZATION_TOKENS for word in words):
    raise ValueError(f"{field_name} contains only serialization residue.")
  return normalized


def _extract_json_object(response: str) -> dict[str, Any]:
  """Extract a usable plans object from prose, fences, or failed prior attempts.

  Qwen can emit a malformed JSON attempt followed by a valid replacement in the
  same completion.  Do not fail on the first brace: scan balanced candidates
  and keep the first object that actually contains a plan batch.
  """
  saw_open_brace = False
  for start, opening in enumerate(response):
    if opening != "{":
      continue
    saw_open_brace = True
    depth, in_string, escaped = 0, False, False
    for offset in range(start, len(response)):
      char = response[offset]
      if in_string:
        if escaped:
          escaped = False
        elif char == "\\":
          escaped = True
        elif char == '"':
          in_string = False
        continue
      if char == '"':
        in_string = True
      elif char == "{":
        depth += 1
      elif char == "}":
        depth -= 1
        if depth == 0:
          try:
            payload = json.loads(response[start : offset + 1])
          except json.JSONDecodeError:
            break
          if isinstance(payload, dict) and _value_by_alias(
              payload, ("plans", "plan", "task_assignments", "tasks", "sub_tasks")
          ) is not None:
            return payload
          break
  if not saw_open_brace:
    raise ValueError("Planner response does not contain a JSON object.")
  raise ValueError("Planner response contains no usable plans JSON object.")


def _parse_tool_call(response: str) -> dict[str, Any] | None:
  """Read one prompt-level tool envelope without imposing an API schema."""
  match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.DOTALL | re.IGNORECASE)
  if match is None:
    return None
  try:
    call = json.loads(match.group(1))
  except json.JSONDecodeError:
    return None
  if not isinstance(call, dict):
    return None
  return call


def _is_complete_goal_call(response: str) -> bool:
  call = _parse_tool_call(response)
  return bool(call and str(call.get("name", "")).casefold() == "complete_goal")


def _tool_call_to_payload(response: str) -> dict[str, Any] | None:
  """Convert a set_tasks tool envelope to the native DMS plan shape."""
  call = _parse_tool_call(response)
  if call is None:
    return None
  name = str(call.get("name", "")).casefold()
  arguments = call.get("arguments", {})
  if not isinstance(arguments, dict):
    return None
  if name != "set_tasks_with_agents":
    return None
  assignments = arguments.get("task_assignments", [])
  if not isinstance(assignments, list):
    return None
  plans: list[dict[str, str]] = []
  for assignment in assignments[:5]:
    if not isinstance(assignment, dict) or not isinstance(assignment.get("task"), str):
      continue
    task = assignment["task"]
    parts = re.search(r"precondition\s*:\s*(.*?)\s*goal\s*:\s*(.+)", task, re.IGNORECASE | re.DOTALL)
    if parts:
      plans.append({"precondition": parts.group(1), "goal": parts.group(2)})
    elif task.strip():
      # P18 allows a functional goal without an explicit precondition.
      plans.append({"precondition": "None", "goal": task})
  return {"plans": plans} if plans else None


def _value_by_alias(item: dict[str, Any], aliases: tuple[str, ...]) -> Any:
  normalized = {
      re.sub(r"[ _-]", "", str(key).casefold()): value for key, value in item.items()
  }
  for alias in aliases:
    normalized_alias = re.sub(r"[ _-]", "", alias.casefold())
    if normalized_alias in normalized:
      return normalized[normalized_alias]
  return None


def parse_plan_batch(response: str) -> list[PlanUnit]:
  """Recover a bounded paper-format plan batch from flexible model output."""
  payload = _tool_call_to_payload(response) or _extract_json_object(response)
  plans = _value_by_alias(payload, ("plans", "plan", "task_assignments", "tasks", "sub_tasks"))
  if isinstance(plans, dict):
    plans = [plans]
  if not isinstance(plans, list) or not 1 <= len(plans) <= 5:
    raise ValueError("Planner must return between 1 and 5 sub-plans.")
  parsed: list[PlanUnit] = []
  for item in plans:
    if not isinstance(item, dict):
      raise ValueError("Every sub-plan must be a JSON object.")
    precondition = _value_by_alias(item, ("precondition", "condition"))
    goal = _value_by_alias(item, ("goal", "subgoal"))
    task = _value_by_alias(item, ("task",))
    if goal is None and isinstance(task, str):
      match = re.search(
          r"precondition\s*:\s*(.*?)\s*goal\s*:\s*(.+)",
          task, re.IGNORECASE | re.DOTALL,
      )
      if match:
        precondition = match.group(1)
        goal = match.group(2)
      else:
        goal = task
    # The Appendix describes a precondition as highly recommended rather than
    # mandatory. Missing preconditions are normalized to its explicit
    # first-plan form instead of rejecting an otherwise valid assignment.
    if precondition is None:
      precondition = "None"
    if not isinstance(precondition, str) or not precondition.strip():
      raise ValueError("Every sub-plan precondition must be text when present.")
    if not isinstance(goal, str) or not goal.strip():
      raise ValueError("Every sub-plan requires a non-empty goal.")
    parsed.append(PlanUnit(
        _clean_plan_text(precondition, "Precondition"),
        _clean_plan_text(goal, "Goal"),
    ))
  return parsed


class HierarchicalDMSPlanner:
  """Planner that queues a 1-5-plan cycle and replans after local failure."""

  def __init__(self, model: PlannerModel, *, agents: tuple[str, ...] = ("CodeActAgent",)):
    self._model = model
    self._agents = agents
    self._pending: deque[PlanUnit] = deque()
    self._history: list[str] = []
    self._goal_completed = False
    self._task_name = ""
    self._task_family = "android_world"
    self._task_params: dict[str, Any] = {}
    self._task_app_names: tuple[str, ...] = ()
    self._last_audit: dict[str, Any] = {}
    self._static_memory_history = "No historical interaction trajectories."

  def set_task_context(
      self,
      *,
      task_name: str,
      task_params: dict[str, Any],
      task_app_names: tuple[str, ...],
      task_family: str = "android_world",
  ) -> None:
    """Set per-instance AndroidWorld metadata for subsequent planning."""
    self._task_name = task_name
    self._task_family = task_family
    self._task_params = dict(task_params)
    self._task_app_names = tuple(task_app_names)

  def set_static_memory_context(self, history: str) -> None:
    """Set Static baseline's chronological interaction context for planning."""
    self._static_memory_history = history or "No historical interaction trajectories."

  @property
  def goal_completed(self) -> bool:
    """Whether the latest Planner decision was paper-tool ``complete_goal``."""
    return self._goal_completed

  @property
  def last_audit(self) -> dict[str, Any]:
    """Structured evidence from the most recent planning cycle."""
    return dict(self._last_audit)

  def plan(self, goal: str, observation: Observation) -> PlanUnit:
    """Return the next queued p_i, creating P(T, o_t, q) when necessary."""
    if not self._pending:
      self._goal_completed = False
      prompt = PLANNER_PROMPT.format(
          agents=", ".join(self._agents),
          goal=goal,
          task_name=self._task_name or "Not specified",
          task_family=self._task_family or "android_world",
          task_app_names=", ".join(self._task_app_names) or "Not specified",
          task_params=json.dumps(self._task_params, ensure_ascii=False, sort_keys=True, default=str),
          activity=observation.foreground_activity,
          ui_text=observation.ui_text or "Not available",
          ui_tree_text=observation.ui_tree_text or "Not available",
          history="\n".join(self._history) or "No prior sub-plans in this task.",
          static_memory_history=self._static_memory_history,
      )
      # Paper planning remains deterministic.  Decoder/semantic validation is
      # separate from planning itself: a malformed candidate gets one explicit
      # repair attempt, then the error is surfaced instead of executing it.
      last_error: ValueError | None = None
      attempts: list[dict[str, Any]] = []
      for attempt in range(2):
        attempt_prompt = prompt
        if attempt:
          assert last_error is not None
          attempt_prompt += PLANNER_RETRY_SUFFIX.format(reason=str(last_error))
        screenshot_png = (
            screenshot_to_png(observation.screenshot)
            if observation.screenshot is not None else None
        )
        response = self._model.complete(
            attempt_prompt,
            screenshot_png,
            temperature=0.0,
            max_tokens=192,
            # Appendix P18 uses prompt-level tool calling. API-level guided
            # JSON conflicts with the required <tool_call> envelope.
            structured_output=False,
        )
        if _is_complete_goal_call(response):
          # P18: the Planner's completion tool is the global completion
          # decision. It is deliberately not rewritten as a verification plan.
          self._goal_completed = True
          self._last_audit = {
              "attempts": attempts + [{"attempt": attempt + 1, "response": response, "accepted": True}],
              "decision": "complete_goal",
          }
          return PlanUnit("Overall goal achieved", "No further action required")
        try:
          plans = parse_plan_batch(response)
          self._pending.extend(plans)
          self._last_audit = {
              "attempts": attempts + [{"attempt": attempt + 1, "response": response, "accepted": True}],
              "decision": "set_tasks_with_agents",
              "plans": [
                  {"precondition": plan.precondition, "goal": plan.goal} for plan in plans
              ],
          }
          break
        except ValueError as error:
          last_error = error
          attempts.append({"attempt": attempt + 1, "response": response, "accepted": False, "error": str(error)})
      else:
        assert last_error is not None
        self._last_audit = {"attempts": attempts, "error": str(last_error)}
        raise ValueError(
            "Planner produced no semantically valid plan after one retry: "
            f"{last_error}"
        ) from last_error
    return self._pending.popleft()

  def record_result(self, plan: PlanUnit, *, completed: bool, detail: str = "") -> None:
    """Keep complete history; discard remaining cycle entries on Actor failure."""
    status = "completed" if completed else "failed"
    self._history.append(f"{status}: Precondition={plan.precondition}; Goal={plan.goal}; {detail}")
    if not completed:
      self._pending.clear()

  def record_execution(self, plan: PlanUnit, *, detail: str = "") -> None:
    """Record a PA-Lite Actor turn and force replanning from the next screen."""
    self._history.append(
        f"executed: Precondition={plan.precondition}; Goal={plan.goal}; {detail}"
    )
    self._pending.clear()

  def reset(self) -> None:
    self._pending.clear()
    self._history.clear()
    self._goal_completed = False
    self._last_audit = {}
    self._static_memory_history = "No historical interaction trajectories."
