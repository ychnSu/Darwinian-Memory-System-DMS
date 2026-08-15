"""Qwen2.5-VL CodeAct adapter for AndroidWorld action blocks."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import re
from typing import TYPE_CHECKING, Any, Protocol

from android_world.env import json_action
from dms.memory import PlanUnit, TrajectoryStep

if TYPE_CHECKING:
  from dms.androidworld_agent import Observation


class VisionLanguageModel(Protocol):
  def complete(
      self, prompt: str, screenshot_png: bytes | None = None, *,
      temperature: float = 0.0, json_schema: dict[str, object] | None = None,
      structured_output: bool | None = None,
  ) -> str: ...


ACTOR_PROMPT = """You are a helpful Android assistant. Write and execute Python
code to complete the current functional task. The code is executed through the
following action tools only:

- swipe(start_x, start_y, end_x, end_y, duration_ms)
- input_text(text, clear)
- press_key(press_key)
- tap(index, x, y, duration_ms, expected_text)
- start_app(package)
- remember(information)
- complete(success, reason)

Output your reasoning followed by exactly one Python code block. The code block
must contain exactly one action-tool call; do not import modules, define
functions, access files, use loops, or call any other API.

ONE SCREEN = ONE TOOL CALL:
- Use exactly one tool call for the current screen.
- Stop immediately before any action that requires observing a new screen, page
  load, animation, dialog, keyboard change, or app transition.
- Do not write actions for elements that are not currently visible.

Decision order:
1. Check whether the current foreground app/page matches the sub-plan.
2. If you are on the launcher/home screen and the task-scoped app is not open,
   call start_app(...) using the task-scoped app name.
3. If a permission, confirmation, or setup dialog is visible, handle that
   visible dialog first.
4. Otherwise choose exactly one currently visible control that advances the
   sub-plan, call tap(index=N) for that control, then stop.

If the current sub-task precondition is not met and cannot be reached, call
`complete(success=False, reason=...)`. When the current sub-task is achieved,
call `complete(success=True, reason=...)`. Use the current screenshot and UI
elements as ground truth. The attached screenshot is a Set-of-Mark view: green
boxes and their numbers are the same indices as the Visible UI elements list.
For a visible UI control, prefer `tap(index=N)`. Never guess an index from
element order or history: choose a number shown both on this screenshot and in
the current list. When the target has visible text, content description, or a
hint, include it exactly as `expected_text="..."`, for example
`tap(index=6, expected_text="Stopwatch")`. Do not invent elements, repeat a
failed action without a state change, or predict a future screen.

Use `press_key` only for BACK, HOME, DEL, ESCAPE, or TAB. Never use numeric
keycodes or ENTER to activate an on-screen button/tab such as Stopwatch,
shutter, save, start, run, pause, done, ok, or play: tap the current UI element
instead.

Overall task:
{task_goal}
AndroidWorld task template:
{task_name}
AndroidWorld task family:
{task_family}
Task-scoped app(s):
{task_app_names}
Task parameters:
{task_params}
Unless the Overall task explicitly requires another app, stay within these
task-scoped app(s). Do not open or navigate to an unrelated application.
For `start_app(package)`, prefer the AndroidWorld app name from Task-scoped
app(s), such as `clock`, `settings`, `chrome`, or `pro expense`. If exactly one
task-scoped app is listed and the current sub-plan is to open, launch, or start
an app, use that scoped app name rather than inventing a package name.
Current sub-plan:
Precondition: {precondition}
Goal: {subgoal}
Current foreground activity: {activity}
Visible UI elements:
{ui_text}
Accessibility UI tree:
{ui_tree_text}
Current sub-plan action history:
{history}

Chronological static-memory interaction history (context only; never execute
these historical actions directly):
{static_memory_history}
"""

ACTOR_RETRY_SUFFIX = """

Your preceding action was rejected before execution for this reason:
{reason}
Return a corrected replacement Python code block for the same current screen.
Use exactly one allowed action-tool call and no other Python statements.
"""


def _element_label(element: Any) -> str:
  """Return the user-visible identity used for index/goal grounding."""
  return " ".join(
      str(value) for value in (
          getattr(element, "text", None),
          getattr(element, "content_description", None),
          getattr(element, "hint_text", None),
      ) if value
  ).casefold()


def _normalize_text(value: str) -> str:
  return re.sub(r"\s+", " ", value.strip()).casefold()


def _element_text_candidates(element: Any) -> tuple[str, ...]:
  """Exact visible labels that can be used to rebind a stale UI index."""
  seen: set[str] = set()
  values: list[str] = []
  for attr in ("text", "content_description", "hint_text"):
    value = getattr(element, attr, None)
    if value is None:
      continue
    text = str(value).strip()
    normalized = _normalize_text(text)
    if text and normalized not in seen:
      seen.add(normalized)
      values.append(text)
  return tuple(values)


def _is_visible(element: Any) -> bool:
  for attr in ("is_visible", "is_visible_to_user"):
    value = getattr(element, attr, None)
    if value is not None:
      return bool(value)
  return True


def _is_clickable_control(element: Any) -> bool:
  return bool(
      getattr(element, "is_clickable", False)
      or getattr(element, "is_focusable", False)
      or getattr(element, "is_editable", False)
  )


def _has_visible_clickable_control(observation: Observation) -> bool:
  return any(
      _is_visible(element) and bool(getattr(element, "is_clickable", False))
      for element in getattr(observation, "ui_elements", ()) or ()
  )


_ALLOWED_PRESS_KEYS = frozenset({
    "KEYCODE_BACK",
    "KEYCODE_HOME",
    "KEYCODE_DEL",
    "KEYCODE_ESCAPE",
    "KEYCODE_TAB",
})
_TAP_REQUIRED_GOAL_RE = re.compile(
    r"\b(?:click|tap|select|press|button|tab|start|run|pause|resume|stop|"
    r"shutter|save|done|ok|play)\b",
    flags=re.IGNORECASE,
)


def _candidate_matches_expected(element: Any, expected_text: str) -> bool:
  expected = _normalize_text(expected_text)
  return any(
      _normalize_text(candidate) == expected
      for candidate in _element_text_candidates(element)
  )


def _visible_indices_matching_text(
    observation: Observation,
    expected_text: str,
    *,
    clickable_only: bool = False,
) -> list[int]:
  indices: list[int] = []
  for index, element in enumerate(observation.ui_elements):
    if observation.grounding_indices and index not in observation.grounding_indices:
      continue
    if not _is_visible(element):
      continue
    if clickable_only and not _is_clickable_control(element):
      continue
    if _candidate_matches_expected(element, expected_text):
      indices.append(index)
  return indices


def _action_with_index(
    action: json_action.JSONAction,
    index: int,
) -> json_action.JSONAction:
  return json_action.JSONAction(
      action_type=action.action_type,
      index=index,
      duration_ms=action.duration_ms,
  )


def _rebind_tap_by_expected_text(
    action: json_action.JSONAction,
    observation: Observation,
    expected_text: str | None,
) -> json_action.JSONAction:
  if action.action_type not in (
      json_action.CLICK, json_action.LONG_PRESS, json_action.DOUBLE_TAP,
  ):
    return action
  if expected_text is None:
    return action
  expected = str(expected_text).strip()
  if not expected:
    raise ValueError("tap expected_text must be non-empty when supplied.")
  if not observation.ui_elements:
    return action
  if action.index is not None and 0 <= action.index < len(observation.ui_elements):
    if (
        (not observation.grounding_indices or action.index in observation.grounding_indices)
        and _is_visible(observation.ui_elements[action.index])
        and _candidate_matches_expected(observation.ui_elements[action.index], expected)
    ):
      return action
  matches = _visible_indices_matching_text(observation, expected)
  if len(matches) == 1:
    return _action_with_index(action, matches[0])
  if not matches:
    raise ValueError(
        f"tap expected_text {expected!r} is not visible as a unique current UI element."
    )
  raise ValueError(
      f"tap expected_text {expected!r} matches multiple visible UI elements; "
      "use the current index for the intended target."
  )


def _unique_goal_label_match(
    observation: Observation,
    plan: PlanUnit | None,
) -> int | None:
  """Conservative repair for key presses when a visible goal label is unique."""
  if plan is None or not observation.ui_elements:
    return None
  goal_text = _normalize_text(plan.goal)
  matches: set[int] = set()
  for index, element in enumerate(observation.ui_elements):
    if observation.grounding_indices and index not in observation.grounding_indices:
      continue
    if not _is_visible(element) or not _is_clickable_control(element):
      continue
    for candidate in _element_text_candidates(element):
      normalized = _normalize_text(candidate)
      if len(normalized) < 2 or not re.search(r"[a-z]", normalized):
        continue
      if re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", goal_text):
        matches.add(index)
        break
  if len(matches) == 1:
    return next(iter(matches))
  return None


def _validate_current_ui_grounding(
    action: json_action.JSONAction,
    observation: Observation,
    plan: PlanUnit | None = None,
    expected_text: str | None = None,
) -> json_action.JSONAction:
  """Verify that a model-selected tap refers to this exact observed element."""
  # Production observations always carry this list. Retain parser-only/offline
  # compatibility when an older caller supplies textual UI state only.
  if not observation.ui_elements:
    return action
  plan_text = "" if plan is None else f"{plan.precondition} {plan.goal}"
  if action.action_type == json_action.PRESS_KEY:
    if (
        _TAP_REQUIRED_GOAL_RE.search(plan_text)
        and _has_visible_clickable_control(observation)
    ):
      repaired_index = _unique_goal_label_match(observation, plan)
      if repaired_index is not None:
        return json_action.JSONAction(
            action_type=json_action.CLICK,
            index=repaired_index,
        )
    keycode = (action.keycode or "").upper()
    if keycode not in _ALLOWED_PRESS_KEYS:
      raise ValueError(
          f"press_key may only use BACK, HOME, DEL, ESCAPE, or TAB; got {keycode}."
      )
    if (
        _TAP_REQUIRED_GOAL_RE.search(plan_text)
        and _has_visible_clickable_control(observation)
    ):
      raise ValueError(
          "press_key is not allowed for a sub-goal that requires selecting a "
          "visible button/tab/control; use tap(index=N, expected_text=...) instead."
      )
    return action
  action = _rebind_tap_by_expected_text(action, observation, expected_text)
  if (
      action.action_type
      in (json_action.CLICK, json_action.LONG_PRESS, json_action.DOUBLE_TAP)
      and action.index is not None
  ):
    if not 0 <= action.index < len(observation.ui_elements):
      raise ValueError(
          f"tap index {action.index} is absent from the current UI state "
          f"(valid indices: 0..{len(observation.ui_elements) - 1})."
      )
    if observation.grounding_indices and action.index not in observation.grounding_indices:
      raise ValueError(
          f"tap index {action.index} was not shown in this turn's grounded UI list."
      )
  return action


# Constraining the remote decoder is important here: JSON-object mode alone
# permits plausible but non-AndroidWorld fields such as ``status`` alongside
# an action.  ``oneOf`` also moves each action's required arguments into
# decoding, so a bare click is not a representable output.  Runtime validation
# below remains the final safety boundary.
ACTOR_JSON_SCHEMA: dict[str, object] = {
    "oneOf": [
        {
            "type": "object",
            "properties": {"action_type": {"const": action}, "index": {"type": "integer"}},
            "required": ["action_type", "index"],
            "additionalProperties": False,
        }
        for action in ("click", "double_tap", "long_press")
    ] + [
        {
            "type": "object",
            "properties": {
                "action_type": {"const": action}, "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["action_type", "x", "y"],
            "additionalProperties": False,
        }
        for action in ("click", "double_tap", "long_press")
    ] + [
        {
            "type": "object",
            "properties": {"action_type": {"const": "input_text"}, "index": {"type": "integer"}, "text": {"type": "string"}},
            "required": ["action_type", "index", "text"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"action_type": {"const": "open_app"}, "app_name": {"type": "string"}},
            "required": ["action_type", "app_name"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"action_type": {"const": "scroll"}, "direction": {"type": "string", "enum": ["up", "down", "left", "right"]}, "index": {"type": "integer"}},
            "required": ["action_type", "direction"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"action_type": {"const": "swipe"}, "direction": {"type": "string", "enum": ["up", "down", "left", "right"]}},
            "required": ["action_type", "direction"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"action_type": {"const": "answer"}, "text": {"type": "string"}},
            "required": ["action_type", "text"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"action_type": {"const": "status"}, "goal_status": {"type": "string", "enum": ["complete", "infeasible"]}},
            "required": ["action_type", "goal_status"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"action_type": {"enum": ["keyboard_enter", "navigate_back", "navigate_home", "wait"]}},
            "required": ["action_type"],
            "additionalProperties": False,
        },
    ],
}

_OPEN_APP_GOAL_RE = re.compile(
    r"\bopen\s+(?:the\s+)?(?P<app>[A-Za-z][A-Za-z0-9 ._-]*?)\s+"
    r"(?:app|application)\b",
    flags=re.IGNORECASE,
)


def _schema_for_plan(plan: PlanUnit) -> dict[str, object]:
  """Return the narrowest action schema consistent with the chosen sub-plan.

  A Planner goal that explicitly says to open an application already fixes the
  abstract action.  Requiring AndroidWorld's ``open_app`` preserves the
  Planner→Actor hierarchy and avoids arbitrary coordinate exploration of an
  unrelated launcher icon.  Other plans retain the full action schema.
  """
  match = _OPEN_APP_GOAL_RE.search(plan.goal)
  if match is None:
    return ACTOR_JSON_SCHEMA
  app_name = match.group("app").strip()
  return {
      "type": "object",
      "properties": {
          "action_type": {"const": "open_app"},
          "app_name": {"const": app_name},
      },
      "required": ["action_type", "app_name"],
      "additionalProperties": False,
  }


def _extract_json_object(text: str) -> dict[str, Any]:
  """Extract exactly the first balanced JSON object, including model prose."""
  start = text.find("{")
  if start < 0:
    raise ValueError("Actor response does not contain a JSON action.")
  depth, in_string, escaped = 0, False, False
  for offset, char in enumerate(text[start:], start):
    if in_string:
      if escaped:
        escaped = False
      elif char == "\\\\":
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
        value = json.loads(text[start : offset + 1])
        if not isinstance(value, dict):
          raise ValueError("Actor JSON must be an object.")
        return value
  raise ValueError("Actor response contains unterminated JSON.")


def parse_json_action(response: str) -> json_action.JSONAction:
  """Parse and validate a single, executable AndroidWorld JSONAction."""
  payload = _extract_json_object(response)
  # Qwen commonly emits the semantically equivalent ``action`` key despite
  # the prompt requesting AndroidWorld's canonical ``action_type`` name.
  # Normalize only this unambiguous alias; all values remain strictly checked
  # by JSONAction and the action-specific guards below.
  if "action_type" not in payload and "action" in payload:
    payload = dict(payload)
    payload["action_type"] = payload.pop("action")
  # Qwen may abbreviate a terminal status as {"status": "complete"}.
  # AndroidWorld represents the same intent as a status action with the
  # ``goal_status`` field, so normalize this equivalent form before strict
  # JSONAction validation.
  if "action_type" not in payload and "status" in payload:
    payload = dict(payload)
    payload["action_type"] = "status"
    payload["goal_status"] = payload.pop("status")
  elif payload.get("action_type") == "status" and "goal_status" not in payload and "status" in payload:
    payload = dict(payload)
    payload["goal_status"] = payload.pop("status")
  action = json_action.JSONAction(**payload)
  kind = action.action_type
  if kind in (json_action.CLICK, json_action.LONG_PRESS, json_action.DOUBLE_TAP):
    if action.index is None and (action.x is None or action.y is None):
      raise ValueError(f"{kind} requires index or both x and y.")
  if kind == json_action.INPUT_TEXT and (action.index is None or action.text is None):
    raise ValueError("input_text requires index and text.")
  if kind == json_action.SCROLL and action.direction is None:
    raise ValueError("scroll requires direction.")
  if kind == json_action.OPEN_APP and not action.app_name:
    raise ValueError("open_app requires app_name.")
  if kind == json_action.ANSWER and action.text is None:
    raise ValueError("answer requires text.")
  if kind == json_action.STATUS and action.goal_status not in ("complete", "infeasible"):
    raise ValueError("status requires goal_status complete or infeasible.")
  if kind != json_action.STATUS and action.goal_status is not None:
    raise ValueError("goal_status is only valid for a status action.")
  return action


_CODEACT_TOOLS = {
    "swipe", "input_text", "press_key", "tap", "start_app", "remember", "complete",
}
# Paper CodeAct exposes ``start_app(package)``. AndroidWorld's open_app action
# instead routes well-known apps through semantic keys (for example, ``clock``)
# before resolving their device-specific package/activity pair. Preserve the
# paper tool interface while adapting the Clock package variants seen across
# AOSP and Google system images to that AndroidWorld key.
_ANDROIDWORLD_PACKAGE_ALIASES = {
    "com.android.deskclock": "clock",
    "com.google.android.deskclock": "clock",
}


@dataclass(frozen=True)
class _ParsedCodeActAction:
  action: json_action.JSONAction
  expected_text: str | None = None


def _androidworld_app_name(package: str) -> str:
  """Resolve known CodeAct package aliases without rewriting unknown packages."""
  normalized = package.strip().casefold().split("/", maxsplit=1)[0]
  return _ANDROIDWORLD_PACKAGE_ALIASES.get(normalized, package)


_OPEN_APP_INTENT_RE = re.compile(r"\b(?:open|launch|start)\b.*\b(?:app|application)\b", re.IGNORECASE)


def _expected_package_for_app_name(app_name: str) -> str | None:
  """Known AndroidWorld app package for scoped-app hallucination correction."""
  normalized = app_name.strip().casefold()
  mapping = {
      "pro expense": "com.arduia.expense",
      "clock": "com.google.android.deskclock",
      "camera": "com.android.camera2",
      "audio recorder": "com.dimowner.audiorecorder",
      "markor": "net.gsantner.markor",
      "files": "com.google.android.documentsui",
      "simple calendar pro": "com.simplemobiletools.calendar.pro",
      "simple gallery pro": "com.simplemobiletools.gallery.pro",
      "simple sms messenger": "com.simplemobiletools.smsmessenger",
      "simple draw pro": "com.simplemobiletools.draw.pro",
      "chrome": "com.android.chrome",
      "contacts": "com.google.android.contacts",
      "settings": "com.android.settings",
      "broccoli": "com.flauschcode.broccoli",
      "vlc": "org.videolan.vlc",
      "retro music": "code.name.monkey.retromusic",
      "osmand": "net.osmand",
      "tasks": "org.tasks",
      "joplin": "net.cozic.joplin",
  }
  return mapping.get(normalized)


def _normalize_start_app_action(
    action: json_action.JSONAction,
    plan: PlanUnit,
    task_app_names: tuple[str, ...],
) -> json_action.JSONAction:
  """Correct scoped start_app package hallucinations before AndroidWorld sees them."""
  if action.action_type != json_action.OPEN_APP or not action.app_name:
    return action
  if len(task_app_names) != 1:
    return action
  if not _OPEN_APP_INTENT_RE.search(f"{plan.precondition} {plan.goal}"):
    return action
  requested = action.app_name.strip()
  scoped = task_app_names[0].strip()
  requested_package = requested.casefold().split("/", maxsplit=1)[0]
  expected_package = _expected_package_for_app_name(scoped)
  if "." in requested and expected_package and requested_package == expected_package.casefold():
    return json_action.JSONAction(action_type=json_action.OPEN_APP, app_name=scoped)
  if "." in requested:
    return json_action.JSONAction(action_type=json_action.OPEN_APP, app_name=scoped)
  return action


def _code_fence(response: str) -> str:
  match = re.search(r"```(?:python)?\s*\n?(.*?)```", response, re.DOTALL | re.IGNORECASE)
  if match is None:
    raise ValueError("CodeAct response must contain one Python code block.")
  return match.group(1).strip()


def _literal_calls(code: str) -> list[tuple[str, list[Any], dict[str, Any]]]:
  """Parse literal tool invocations; never execute model-generated code."""
  try:
    module = ast.parse(code, mode="exec")
  except SyntaxError as error:
    raise ValueError(f"Invalid CodeAct Python: {error.msg}") from error
  if not module.body:
    raise ValueError("CodeAct code block must contain at least one tool call.")
  parsed: list[tuple[str, list[Any], dict[str, Any]]] = []
  for statement in module.body:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
      raise ValueError("CodeAct code block may contain only action-tool calls.")
    call = statement.value
    if not isinstance(call.func, ast.Name) or call.func.id not in _CODEACT_TOOLS:
      raise ValueError("CodeAct code must call a supported named action tool.")
    if any(keyword.arg is None for keyword in call.keywords):
      raise ValueError("CodeAct tool calls cannot use **kwargs.")
    try:
      args = [ast.literal_eval(argument) for argument in call.args]
      kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
    except (TypeError, ValueError) as error:
      raise ValueError("CodeAct tool arguments must be literal values.") from error
    parsed.append((call.func.id, args, kwargs))
  if len(parsed) != 1:
    raise ValueError("CodeAct code block must contain exactly one tool call.")
  return parsed


def _argument(args: list[Any], kwargs: dict[str, Any], name: str, position: int, *, required: bool = True, default: Any = None) -> Any:
  if name in kwargs:
    return kwargs[name]
  if len(args) > position:
    return args[position]
  if required:
    raise ValueError(f"CodeAct tool requires `{name}`.")
  return default


def _parse_codeact_call(tool: str, args: list[Any], kwargs: dict[str, Any]) -> _ParsedCodeActAction:
  """Translate one Table-3 CodeAct call into an AndroidWorld action."""
  if tool == "tap":
    index = _argument(args, kwargs, "index", 0, required=False)
    x = _argument(args, kwargs, "x", 1, required=False)
    y = _argument(args, kwargs, "y", 2, required=False)
    expected_text = _argument(args, kwargs, "expected_text", 4, required=False)
    if expected_text is None:
      for extra in args[1:]:
        if isinstance(extra, str):
          expected_text = extra
          break
    # Small models often emit tap(index, x, y, duration_like_noise). Treat
    # positional extras after index as noisy grounding attempts. A long press
    # must be explicit via the duration_ms keyword.
    duration_ms = kwargs.get("duration_ms")
    action_type = json_action.CLICK
    if duration_ms is not None and int(duration_ms) >= 500:
      action_type = json_action.LONG_PRESS
    if index is not None:
      return _ParsedCodeActAction(
          json_action.JSONAction(
              action_type=action_type, index=index, duration_ms=duration_ms
          ),
          None if expected_text is None else str(expected_text),
      )
    if x is None or y is None:
      raise ValueError("tap requires an index or both x and y.")
    return _ParsedCodeActAction(
        json_action.JSONAction(
            action_type=action_type, x=int(x), y=int(y), duration_ms=duration_ms
        ),
        None if expected_text is None else str(expected_text),
    )
  if tool == "swipe":
    return _ParsedCodeActAction(
        json_action.JSONAction(
            action_type=json_action.SWIPE,
            start_x=int(_argument(args, kwargs, "start_x", 0)),
            start_y=int(_argument(args, kwargs, "start_y", 1)),
            end_x=int(_argument(args, kwargs, "end_x", 2)),
            end_y=int(_argument(args, kwargs, "end_y", 3)),
            duration_ms=int(_argument(args, kwargs, "duration_ms", 4)),
        )
    )
  if tool == "input_text":
    return _ParsedCodeActAction(
        json_action.JSONAction(
            action_type=json_action.INPUT_TEXT,
            text=str(_argument(args, kwargs, "text", 0)),
            clear_text=bool(_argument(args, kwargs, "clear", 1, required=False, default=False)),
        )
    )
  if tool == "press_key":
    key = str(_argument(args, kwargs, "press_key", 0))
    keycode = key if key.startswith("KEYCODE_") else f"KEYCODE_{key.upper()}"
    return _ParsedCodeActAction(
        json_action.JSONAction(action_type=json_action.PRESS_KEY, keycode=keycode)
    )
  if tool == "start_app":
    package = str(_argument(args, kwargs, "package", 0))
    return _ParsedCodeActAction(
        json_action.JSONAction(
            action_type=json_action.OPEN_APP,
            app_name=_androidworld_app_name(package),
        )
    )
  if tool == "remember":
    return _ParsedCodeActAction(
        json_action.JSONAction(
            action_type=json_action.REMEMBER,
            information=str(_argument(args, kwargs, "information", 0)),
        )
    )
  return _ParsedCodeActAction(
      json_action.JSONAction(
          action_type=json_action.COMPLETE,
          success=bool(_argument(args, kwargs, "success", 0)),
          reason=str(_argument(args, kwargs, "reason", 1)),
      )
  )


def _parse_codeact_calls(
    response: str,
) -> list[_ParsedCodeActAction]:
  return [
      _parse_codeact_call(*call)
      for call in _literal_calls(_code_fence(response))
  ]


def parse_codeact_actions(response: str) -> list[json_action.JSONAction]:
  """Translate one CodeAct Python block into one executable action."""
  return [parsed.action for parsed in _parse_codeact_calls(response)]


def parse_codeact_action(response: str) -> json_action.JSONAction:
  """Compatibility helper for one-action CodeAct responses."""
  actions = parse_codeact_actions(response)
  if len(actions) != 1:
    raise ValueError("Expected one CodeAct action; use parse_codeact_actions for a block.")
  return actions[0]


def screenshot_to_png(pixels: Any) -> bytes:
  """Encode AndroidWorld RGB pixels lazily to avoid a hard import at module load."""
  try:
    from PIL import Image
  except ImportError as error:
    raise RuntimeError("Pillow is required to send Android screenshots to Qwen2.5-VL.") from error
  from io import BytesIO
  image = Image.fromarray(pixels)
  buffer = BytesIO()
  image.save(buffer, format="PNG")
  return buffer.getvalue()


class QwenVLActor:
  """Concrete Actor: prompt Qwen2.5-VL then return a CodeAct action block."""

  def __init__(self, model: VisionLanguageModel):
    self._model = model
    self._last_audit: dict[str, Any] = {}
    self._task_name = ""
    self._task_family = "android_world"
    self._task_params: dict[str, Any] = {}
    self._task_app_names: tuple[str, ...] = ()
    self._static_memory_history = "No historical interaction trajectories."

  def set_task_context(
      self,
      *,
      task_name: str,
      task_params: dict[str, Any],
      task_app_names: tuple[str, ...],
      task_family: str = "android_world",
  ) -> None:
    """Set per-instance AndroidWorld metadata for action generation."""
    self._task_name = task_name
    self._task_family = task_family
    self._task_params = dict(task_params)
    self._task_app_names = tuple(task_app_names)

  def reset(self) -> None:
    self._last_audit = {}
    self._static_memory_history = "No historical interaction trajectories."

  def set_static_memory_context(self, history: str) -> None:
    """Set Static baseline's chronological interaction context for acting."""
    self._static_memory_history = history or "No historical interaction trajectories."

  @property
  def last_audit(self) -> dict[str, Any]:
    """Structured, non-executable evidence from the latest Actor decision."""
    return dict(self._last_audit)

  def next_actions(
      self, goal: str, plan: PlanUnit, observation: Observation,
      history: list[TrajectoryStep],
  ) -> list[json_action.JSONAction]:
    action_history = "\n".join(
        f"Observation: {step.observation}\nAction: {step.action}" for step in history
    ) or "No actions yet."
    prompt = ACTOR_PROMPT.format(
        task_goal=goal,
        task_name=self._task_name or "Not specified",
        task_family=self._task_family or "android_world",
        task_app_names=", ".join(self._task_app_names) or "Not specified",
        task_params=json.dumps(self._task_params, ensure_ascii=False, sort_keys=True, default=str),
        precondition=plan.precondition,
        subgoal=plan.goal,
        activity=observation.foreground_activity,
        ui_text=observation.ui_text or "Not available",
        ui_tree_text=observation.ui_tree_text or "Not available",
        history=action_history,
        static_memory_history=self._static_memory_history,
    )
    screenshot_png = screenshot_to_png(
        observation.grounding_screenshot
        if observation.grounding_screenshot is not None else observation.screenshot
    )
    last_error: ValueError | TypeError | None = None
    attempts: list[dict[str, Any]] = []
    for attempt in range(2):
      attempt_prompt = prompt
      if attempt:
        assert last_error is not None
        attempt_prompt += ACTOR_RETRY_SUFFIX.format(reason=str(last_error))
      response = self._model.complete(
          attempt_prompt, screenshot_png, temperature=0.0,
          structured_output=False,
      )
      try:
        parsed_codeact_actions = _parse_codeact_calls(response)
        if len(parsed_codeact_actions) != 1:
          raise ValueError("Actor must return exactly one action-tool call.")
        parsed_actions: list[json_action.JSONAction] = []
        for parsed in parsed_codeact_actions:
          action = _normalize_start_app_action(
              parsed.action, plan, self._task_app_names
          )
          parsed_actions.append(
              _validate_current_ui_grounding(
                  action, observation, plan, parsed.expected_text
              )
          )
        attempts.append({"attempt": attempt + 1, "response": response, "accepted": True})
        self._last_audit = {
            "attempts": attempts,
            "actions": [action.as_dict() for action in parsed_actions],
        }
        return list(parsed_actions)
      except (TypeError, ValueError) as error:
        last_error = error
        attempts.append({"attempt": attempt + 1, "response": response, "accepted": False, "error": str(error)})
    assert last_error is not None
    self._last_audit = {"attempts": attempts, "error": str(last_error)}
    raise ValueError(
        "Actor produced no executable action block after one retry: " f"{last_error}"
    ) from last_error

  def next_action(
      self, goal: str, plan: PlanUnit, observation: Observation,
      history: list[TrajectoryStep],
  ) -> json_action.JSONAction:
    """Compatibility wrapper for older single-action callers."""
    actions = self.next_actions(goal, plan, observation, history)
    if len(actions) != 1:
      raise ValueError(
          "Expected one CodeAct action; use next_actions for action blocks."
      )
    return actions[0]
