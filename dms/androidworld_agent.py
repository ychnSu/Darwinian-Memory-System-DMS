"""AndroidWorld adapter for the DMS planner/actor/memory loop.

This module intentionally contains no model client.  Concrete planner, actor,
and verifier implementations can use a local fake for tests or the remote VLM
later, while the interaction and memory lifecycle stays deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Protocol

from android_world.agents import base_agent
from android_world.agents import m3a_utils
from android_world.env import interface, json_action
from dms.memory import DMSMemoryStore, MemoryUnit, PlanUnit, TrajectoryStep
from dms.policy import DMSPolicy


@dataclass(frozen=True)
class Observation:
  """Portable state view handed to planner, actor, and verifier."""
  ui_text: str
  ui_tree_text: str
  foreground_activity: str
  screenshot: Any
  # Structured current-screen elements are used for action grounding only;
  # ui_text remains the model-facing and persisted representation.
  ui_elements: tuple[Any, ...] = ()
  # Actor-only Set-of-Mark view. Planner/Verifier retain the raw screenshot.
  grounding_screenshot: Any | None = None
  grounding_indices: frozenset[int] = frozenset()

  def memory_snapshot(self) -> str:
    """Textual o_t persisted with a_t; the current screenshot stays VLM input."""
    return (
        f"Foreground activity: {self.foreground_activity}\n"
        f"Visible UI elements:\n{self.ui_text}\n"
        f"Accessibility UI tree:\n{self.ui_tree_text}"
    )

  def screen_fingerprint(self) -> str:
    """Stable page fingerprint for loop detection."""
    payload = (
        f"{self.foreground_activity}\n"
        f"{self.ui_text}\n"
        f"{self.ui_tree_text[:4000]}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Verification:
  subplan_complete: bool
  task_complete: bool = False
  verified: bool = True
  subplan_failed: bool = False


class Planner(Protocol):
  def plan(self, goal: str, observation: Observation) -> PlanUnit: ...


class Actor(Protocol):
  def next_action(
      self, goal: str, plan: PlanUnit, observation: Observation,
      history: list[TrajectoryStep],
  ) -> json_action.JSONAction: ...


class Verifier(Protocol):
  def verify(
      self, goal: str, plan: PlanUnit, before: Observation,
      after: Observation, action: json_action.JSONAction,
      history: list[TrajectoryStep],
  ) -> Verification: ...


_MAX_GROUNDED_UI_ELEMENTS = 80
_UI_CHANGING_ACTIONS = frozenset({
    json_action.CLICK,
    json_action.DOUBLE_TAP,
    json_action.INPUT_TEXT,
    json_action.KEYBOARD_ENTER,
    json_action.LONG_PRESS,
    json_action.NAVIGATE_BACK,
    json_action.NAVIGATE_HOME,
    json_action.OPEN_APP,
    json_action.PRESS_KEY,
    json_action.SCROLL,
    json_action.SWIPE,
    json_action.WAIT,
})
_SIMPLE_DOWNLOADS_OPEN_RE = re.compile(
    r"\b(?:open|show|view|go to|navigate to)\b.*\bdownloads?\b|"
    r"\bdownloads?\b.*\b(?:folder|directory)\b",
    flags=re.IGNORECASE,
)
_COMPOUND_DOWNLOADS_BLOCKERS_RE = re.compile(
    r"\b(?:and|then|html|chrome|browser|locate|find|search|copy|save|export|"
    r"submit|draw|create|write|move|rename|delete|file\s+\w+\.)\b",
    flags=re.IGNORECASE,
)
_APP_LAUNCH_RE = re.compile(
    r"\b(?:open|launch|access|go to|navigate to)\b",
    flags=re.IGNORECASE,
)
_AUDIO_RECORDER_RE = re.compile(r"\baudio\s*recorder\b|audiorecorder", re.IGNORECASE)
_AUDIO_RECORD_GOAL_RE = re.compile(
    r"\b(?:record|recording|audio clip|start recording)\b", re.IGNORECASE
)
_MARKOR_NEWEST_NOTE_RE = re.compile(r"\bnewest note\b|\bdelete\b", re.IGNORECASE)
_CONTACT_ADD_GOAL_RE = re.compile(
    r"create\s+a\s+new\s+contact\s+for\s+(?P<name>.+?)\.\s*"
    r"their\s+number\s+is\s+(?P<number>[^.]+)\.",
    re.IGNORECASE,
)
_CLOCK_STOPWATCH_RUN_RE = re.compile(
    r"\b(?:run|start|begin)\b.*\bstopwatch\b|\bstopwatch\b.*\b(?:run|start|begin)\b",
    re.IGNORECASE,
)
_CLOCK_STOPWATCH_PAUSE_RE = re.compile(
    r"\bpause\b.*\bstopwatch\b|\bstopwatch\b.*\bpause\b",
    re.IGNORECASE,
)


def _is_actionable(element: Any) -> bool:
  return bool(
      getattr(element, "is_clickable", False)
      or getattr(element, "is_focusable", False)
      or getattr(element, "is_editable", False)
      or getattr(element, "is_scrollable", False)
  )


def _element_summary(index: int, element: Any) -> str:
  """Compact, index-stable UI record for visual/textual grounding."""
  bbox = getattr(element, "bbox_pixels", None)
  bounds = (
      f"({int(bbox.x_min)},{int(bbox.y_min)},{int(bbox.x_max)},{int(bbox.y_max)})"
      if bbox is not None else "unknown"
  )
  label = getattr(element, "text", None) or getattr(element, "content_description", None) or ""
  return (
      f"[{index}] text={label!r}; desc={getattr(element, 'content_description', None)!r}; "
      f"bounds={bounds}; clickable={bool(getattr(element, 'is_clickable', False))}; "
      f"editable={bool(getattr(element, 'is_editable', False))}; "
      f"scrollable={bool(getattr(element, 'is_scrollable', False))}; "
      f"selected={bool(getattr(element, 'is_selected', False))}"
  )


def _visible_label(element: Any) -> str:
  return " ".join(
      str(value) for value in (
          getattr(element, "text", None),
          getattr(element, "content_description", None),
      ) if value
  ).casefold()


def _observe(env: interface.AsyncEnv, state: interface.State) -> Observation:
  """Build raw and Set-of-Mark observations from one immutable UI state."""
  logical_size = env.logical_screen_size
  valid = [
      index for index, element in enumerate(state.ui_elements)
      if m3a_utils.validate_ui_element(element, logical_size)
  ]
  # Keep the list readable while favouring controls a user can operate. Sort
  # back by original index so screenshot labels, prompt list and executor share
  # exactly the AndroidWorld index space.
  ranked = sorted(
      valid,
      key=lambda index: (
          not _is_actionable(state.ui_elements[index]),
          not bool(getattr(state.ui_elements[index], "text", None)
                   or getattr(state.ui_elements[index], "content_description", None)),
          index,
      ),
  )[:_MAX_GROUNDED_UI_ELEMENTS]
  grounding_indices = frozenset(ranked)
  ui_text = "\n".join(
      _element_summary(index, state.ui_elements[index]) for index in sorted(ranked)
  ) or "No valid visible UI elements."
  grounded_screenshot = state.pixels.copy()
  for index in sorted(grounding_indices):
    m3a_utils.add_ui_element_mark(
        grounded_screenshot,
        state.ui_elements[index],
        index,
        logical_size,
        env.physical_frame_boundary,
        env.orientation,
    )
  tree_text = repr(state.forest)
  if len(tree_text) > 12000:
    tree_text = tree_text[:12000] + "\n[UI tree truncated]"
  return Observation(
      ui_text,
      tree_text,
      env.foreground_activity_name,
      state.pixels,
      tuple(state.ui_elements),
      grounded_screenshot,
      grounding_indices,
  )


class DMSAndroidWorldAgent(base_agent.EnvironmentInteractingAgent):
  """One-action-per-step AndroidWorld adapter for DMS.

  A retrieved macro is replayed action-by-action.  A generated sequence is
  accumulated until the verifier declares its sub-plan complete, then saved as
  a reusable non-trivial memory.  Failures are fed back to the retrieved entry.
  """

  def __init__(
      self,
      env: interface.AsyncEnv,
      planner: Planner,
      actor: Actor,
      verifier: Verifier | None,
      store: DMSMemoryStore,
      policy: DMSPolicy,
      name: str = "DMSAndroidWorldAgent",
      memory_mode: str = "dms",
  ):
    if memory_mode not in ("dms", "static", "zero_shot"):
      raise ValueError("memory_mode must be 'dms', 'static', or 'zero_shot'.")
    super().__init__(env, name)
    self.planner, self.actor, self.verifier = planner, actor, verifier
    self.store, self.policy = store, policy
    self.memory_mode = memory_mode
    self._plan: PlanUnit | None = None
    self._trace: list[TrajectoryStep] = []
    self._subplan_history: list[TrajectoryStep] = []
    self._task_trace: list[TrajectoryStep] = []
    # Failed generations are audit evidence, never reusable DMS memories.
    self._failure_trace: list[TrajectoryStep] = []
    self._replay: list[TrajectoryStep] = []
    self._replayed_memory_id: str | None = None
    self._mutation_memory_id: str | None = None
    self._active_memory_ids: set[str] = set()
    self._task_name = ""
    self._task_goal = ""
    self._task_params: dict[str, Any] = {}
    self._task_app_names: tuple[str, ...] = ()
    self._task_family = "android_world"
    self._recent_action_fingerprints: list[tuple[str, str]] = []

  def set_task_context(
      self,
      *,
      task_name: str,
      task_params: dict[str, Any],
      task_app_names: tuple[str, ...],
      task_family: str = "android_world",
  ) -> None:
    """Set AndroidWorld metadata once before the episode resets the agent.

    This is benchmark context for Planner/Actor grounding, rather than a DMS
    memory: it is deliberately refreshed for every task instance.
    """
    self._task_name = task_name
    self._task_params = dict(task_params)
    self._task_app_names = tuple(task_app_names)
    self._task_family = task_family
    for component in (self.planner, self.actor):
      set_component_context = getattr(component, "set_task_context", None)
      if callable(set_component_context):
        try:
          set_component_context(
              task_name=self._task_name,
              task_params=dict(self._task_params),
              task_app_names=self._task_app_names,
              task_family=self._task_family,
          )
        except TypeError:
          set_component_context(
              task_name=self._task_name,
              task_params=dict(self._task_params),
              task_app_names=self._task_app_names,
          )

  def reset(self, go_home: bool = False) -> None:
    super().reset(go_home=go_home)
    if hasattr(self.planner, "reset"):
      self.planner.reset()  # type: ignore[attr-defined]
    if hasattr(self.actor, "reset"):
      self.actor.reset()  # type: ignore[attr-defined]
    self._clear_subplan()
    self._task_trace = []
    self._task_goal = ""
    self._failure_trace = []
    self._active_memory_ids.clear()
    self._recent_action_fingerprints = []

  def finalize_task(self, *, task_succeeded: bool) -> set[str]:
    """Apply global feedback after the task runner gets ground truth."""
    active = set(self._active_memory_ids)
    if self.memory_mode == "dms":
      if task_succeeded and not active and len(self._task_trace) >= 2:
        memory = self.store.add(MemoryUnit(
            plan=PlanUnit(
                precondition=(
                    f"Task template: {self._task_name or 'unknown'}; "
                    f"task_apps={','.join(self._task_app_names) or 'unknown'}"
                ),
                goal=self._task_goal or "unknown",
            ),
            trajectory=list(self._task_trace),
            success=True,
            description=(
                "task_level_fallback=True; "
                "created because AndroidWorld evaluator succeeded but no "
                "sub-plan memory was retained; "
                f"task_template={self._task_name or 'unknown'}; "
                f"task_apps={','.join(self._task_app_names) or 'unknown'}"
            ),
        ))
        active.add(memory.id)
      self.store.record_global_feedback(active, task_succeeded=task_succeeded)
    elif self.memory_mode == "static" and len(self._task_trace) >= 2:
      memory = self.store.add(MemoryUnit(
          plan=PlanUnit(
              precondition=f"Task template: {self._task_name or 'unknown'}",
              goal=self._task_goal or "unknown",
          ),
          trajectory=list(self._task_trace),
          success=bool(task_succeeded),
          description=(
              f"AndroidWorld evaluator success={bool(task_succeeded)}; "
              f"task_template={self._task_name or 'unknown'}; "
              f"task_apps={','.join(self._task_app_names) or 'unknown'}"
          ),
      ))
      active.add(memory.id)
    self._active_memory_ids.clear()
    self._task_trace = []
    return active

  @staticmethod
  def _component_audit(component: Any) -> dict[str, Any]:
    """Read optional model-component evidence without constraining test fakes."""
    audit = getattr(component, "last_audit", {})
    return dict(audit) if isinstance(audit, dict) else {}

  @staticmethod
  def _environment_audit(before: Observation, after: Observation | None = None) -> dict[str, Any]:
    """Compact environment evidence; UI text stays in the normal checkpoint."""
    def summary(observation: Observation) -> dict[str, Any]:
      return {
          "activity": observation.foreground_activity,
          "visible_element_count": len(observation.ui_elements),
          "ui_sha256": hashlib.sha256(observation.ui_text.encode("utf-8")).hexdigest()[:16],
      }
    evidence: dict[str, Any] = {"before": summary(before)}
    if after is not None:
      evidence["after"] = summary(after)
      evidence["ui_changed"] = evidence["before"]["ui_sha256"] != evidence["after"]["ui_sha256"]
    return evidence

  def _audit(self, before: Observation, after: Observation | None = None) -> dict[str, Any]:
    """Per-step Planner/Actor/Verifier/environment audit record."""
    return {
        "planner": self._component_audit(getattr(self, "planner", None)),
        "actor": self._component_audit(getattr(self, "actor", None)),
        "verifier": self._component_audit(getattr(self, "verifier", None)),
        "environment": self._environment_audit(before, after),
    }

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    self._task_goal = goal
    before_state = self.get_post_transition_state()
    before = _observe(self.env, before_state)
    if self._plan is None:
      try:
        self._begin_subplan(goal, before)
      except Exception as error:
        return self._planning_failure(before_state, before, error)
    assert self._plan is not None

    # Appendix P18 makes ``complete_goal`` the Planner's global-completion
    # tool. Do not turn it into a synthetic sub-plan or send it to the Actor.
    if bool(getattr(self.planner, "goal_completed", False)):
      completed_plan = self._plan
      self._clear_subplan()
      return base_agent.AgentInteractionResult(True, {
          "raw_screenshot": before_state.pixels,
          "ui_elements": before_state.ui_elements,
          "plan": completed_plan,
          "planner_complete_goal": True,
          "audit": self._audit(before),
          "memory_size": len(self.store.memories),
      })

    actions: list[json_action.JSONAction] = []
    origin = ""
    if self._replay:
      actions = self._next_replay_actions(before)
      origin = "memory_replay" if actions else ""
    if not actions:
      shortcut = (
          self._maybe_clock_stopwatch_shortcut(goal, self._plan, before)
          or self._maybe_contacts_add_shortcut(goal, self._plan, before)
          or self._maybe_audio_recorder_shortcut(self._plan, before)
          or self._maybe_markor_delete_shortcut(self._plan, before)
          or self._maybe_task_app_launch_shortcut(self._plan, before)
          or self._maybe_downloads_shortcut(goal, self._plan, before)
      )
      if shortcut is not None:
        actions = [shortcut]
        origin = "framework_shortcut"
      else:
        try:
          next_actions = getattr(self.actor, "next_actions", None)
          if callable(next_actions):
            actions = next_actions(goal, self._plan, before, list(self._trace))
          else:
            actions = [self.actor.next_action(goal, self._plan, before, list(self._trace))]
        except Exception as error:  # Model/parse/grounding failures are recoverable.
          return self._recover_from_failure(
              goal, before, before_state, phase="actor", error=error
          )
        origin = "mutation_generation" if self._mutation_memory_id else "actor_generation"

    if not actions:
      return self._recover_from_failure(
          goal, before, before_state, phase="actor",
          error=ValueError("Actor returned an empty action block.")
      )

    executed_actions: list[json_action.JSONAction] = []
    current_state, current_observation = before_state, before
    for action in actions:
      if self._is_repeated_no_progress_action(action, current_observation):
        return self._recover_from_failure(
            goal,
            current_observation,
            current_state,
            phase="anti_loop",
            error=ValueError(
                "Repeated the same UI-changing action on an unchanged screen."
            ),
            action=action,
        )
      step = TrajectoryStep(current_observation.memory_snapshot(), action.as_dict())
      self._trace.append(step)
      # ``complete`` is the CodeAct agent's internal sub-task status signal,
      # not an Android/ADB operation.
      if action.action_type == json_action.COMPLETE:
        return self._handle_actor_completion(
            action, current_observation, current_state, origin
        )
      self._subplan_history.append(step)
      self._task_trace.append(step)
      executed_actions.append(action)
      before_action_observation = current_observation
      try:
        self.env.execute_action(action)
      except Exception as error:  # Stale UI/index or device actuation failure.
        return self._recover_from_failure(
            goal, current_observation, current_state, phase="execution",
            error=error, action=action
        )
      current_state = self.get_post_transition_state()
      current_observation = _observe(self.env, current_state)
      self._record_action_progress(
          action,
          before_observation=before_action_observation,
          after_observation=current_observation,
      )
      if self._should_stop_action_block_after(action):
        break

    after_state = current_state
    after = current_observation
    action_data = executed_actions[-1].as_dict()
    actions_data = [action.as_dict() for action in executed_actions]
    if self.verifier is None:
      # PA-Lite/static baselines should not use the DMS verifier to advance
      # sub-plans; global success remains solely AndroidWorld evaluator
      # responsibility in suite_utils._termination_fn_for_task.
      plan_data = self._plan
      record_execution = getattr(self.planner, "record_execution", None)
      if callable(record_execution):
        record_execution(
            plan_data,
            detail=f"Executed action block: {actions_data}",
        )
      self._clear_subplan()
      return base_agent.AgentInteractionResult(False, {
          "raw_screenshot": before_state.pixels,
          "ui_elements": before_state.ui_elements,
          "plan": plan_data,
          "action": action_data,
          "actions": actions_data,
          "origin": origin,
          "verification": None,
          "audit": self._audit(before, after),
          "memory_size": len(self.store.memories),
      })
    try:
      verdict = self.verifier.verify(
          goal, self._plan, before, after, executed_actions[-1],
          list(self._subplan_history)
      )
    except Exception as error:  # Verification service/parse failures also replan.
      return self._recover_from_failure(
          goal, after, after_state, phase="verifier", error=error, action=action
      )
    plan_data = self._plan
    self._finish_subplan_if_needed(verdict)

    # Appendix P18: only Planner.complete_goal ends the overall task.
    done = False
    return base_agent.AgentInteractionResult(done, {
        "raw_screenshot": before_state.pixels,
        "ui_elements": before_state.ui_elements,
        "plan": plan_data,
        "action": action_data,
        "actions": actions_data,
        "origin": origin,
        "verification": verdict,
        "audit": self._audit(before, after),
        "memory_size": len(self.store.memories),
    })

  def _handle_actor_completion(
      self,
      action: json_action.JSONAction,
      observation: Observation | None,
      state: interface.State,
      origin: str,
  ) -> base_agent.AgentInteractionResult:
    """Apply a CodeAct completion signal without sending it to ADB."""
    assert self._plan is not None
    # Generated status calls are appended before the shared action branch.
    # They are control metadata, never replayable Android actions, so remove
    # the just-recorded marker before possible memory creation.
    if self._trace and self._trace[-1].action == action.as_dict():
      self._trace.pop()
    asserted_success = bool(action.success)
    detail = action.reason or "Actor returned a completion status."
    plan_data = self._plan
    if asserted_success and self._completion_requires_target_app(observation):
      return self._recover_from_failure(
          self._task_goal,
          observation or Observation("", "", "", state.pixels, tuple(state.ui_elements)),
          state,
          phase="actor",
          error=ValueError(
              "Actor tried to complete the task outside the target app; "
              "current UI state must be grounded before completion."
          ),
          action=action,
      )
    if (
        action.success
        and self.verifier is None
        and self._task_family == "information_retrieval"
        and action.reason
    ):
      answer_action = json_action.JSONAction(
          action_type=json_action.ANSWER,
          text=action.reason,
      )
      self.env.execute_action(answer_action)
      answer_observation = observation or Observation(
          "", "", "", state.pixels, tuple(state.ui_elements)
      )
      self._task_trace.append(
          TrajectoryStep(answer_observation.memory_snapshot(), answer_action.as_dict())
      )
      self._clear_subplan()
      return base_agent.AgentInteractionResult(True, {
          "raw_screenshot": state.pixels,
          "ui_elements": state.ui_elements,
          "plan": plan_data,
          "action": answer_action.as_dict(),
          "origin": "actor_answer",
          "actor_origin": origin,
          "verification": None,
          "audit": self._audit(answer_observation),
          "memory_size": len(self.store.memories),
      })
    verdict = Verification(
        subplan_complete=asserted_success,
        verified=asserted_success,
        subplan_failed=not asserted_success,
    )
    self._finish_subplan_if_needed(verdict, detail=detail)
    if not asserted_success and hasattr(self.actor, "reset"):
      self.actor.reset()  # type: ignore[attr-defined]
    done = asserted_success and self.verifier is None
    return base_agent.AgentInteractionResult(done, {
        "raw_screenshot": state.pixels,
        "ui_elements": state.ui_elements,
        "plan": plan_data,
        "action": action.as_dict(),
        "origin": "actor_completion",
        "actor_origin": origin,
        "verification": verdict,
        "audit": self._audit(observation or Observation(
            "", "", "", state.pixels, tuple(state.ui_elements)
        )),
        "memory_size": len(self.store.memories),
    })

  def _begin_subplan(self, goal: str, observation: Observation) -> None:
    if self.memory_mode == "static":
      # Static baseline is history-context injection, not a trajectory/action
      # selector. Keep it at the Planner level only: the Actor must keep the
      # same constrained prompt surface as PA-Lite/zero-shot so historical
      # action traces cannot weaken one-tool-call compliance.
      static_history = self.store.chronological_context(
          task_name=self._task_name or None,
          task_app_names=self._task_app_names,
      )
      set_context = getattr(self.planner, "set_static_memory_context", None)
      if callable(set_context):
        set_context(static_history)
    self._plan = self.planner.plan(goal, observation)
    if self.memory_mode in ("zero_shot", "static"):
      return
    risk_threshold = (
        self.policy.effective_risk_threshold(self.store.global_failure_rate())
        if self.memory_mode == "dms" else 1.1
    )
    hit = self.store.retrieve(
        self._plan,
        risk_threshold=risk_threshold,
        task_app_names=self._task_app_names,
    )
    if self.memory_mode == "dms" and self.policy.should_replay(
        hit,
        risk_threshold=risk_threshold,
        global_failure_rate=self.store.global_failure_rate(),
    ):
      assert hit is not None
      self._replay = list(hit.trajectory)
      self._replayed_memory_id = hit.id
      self._active_memory_ids.add(hit.id)
      self.store.record_replay(hit.id)
    elif hit is not None and self.memory_mode == "dms":
      # Paper §3.2.2: with probability epsilon, Actor re-attempts the
      # sub-task from scratch. Keep the selected entry only as the potential
      # target for an in-place, strictly shorter successful replacement.
      self._mutation_memory_id = hit.id
      self.store.record_mutation(hit.id)

  def _maybe_downloads_shortcut(
      self,
      goal: str,
      plan: PlanUnit,
      observation: Observation,
  ) -> json_action.JSONAction | None:
    """Open Files for simple Downloads navigation without package guessing."""
    del goal
    text = f"{plan.precondition} {plan.goal}"
    if not _SIMPLE_DOWNLOADS_OPEN_RE.search(text):
      return None
    if _COMPOUND_DOWNLOADS_BLOCKERS_RE.search(text):
      return None
    if "com.google.android.documentsui" not in observation.foreground_activity.casefold():
      return json_action.JSONAction(action_type=json_action.OPEN_APP, app_name="files")
    matches = [
        index for index, element in enumerate(observation.ui_elements)
        if index in observation.grounding_indices
        and bool(getattr(element, "is_clickable", False))
        and _visible_label(element) in ("download", "downloads")
    ]
    if len(matches) == 1:
      return json_action.JSONAction(action_type=json_action.CLICK, index=matches[0])
    return None

  def _maybe_task_app_launch_shortcut(
      self,
      plan: PlanUnit,
      observation: Observation,
  ) -> json_action.JSONAction | None:
    """Open the single task-scoped app when the plan is plainly an app launch.

    Small models often turn "open the Clock app" into illegal key presses or
    stale launcher clicks. This shortcut keeps the Planner-Actor boundary, but
    uses AndroidWorld's app-name action for a deterministic app launch.
    """
    if len(self._task_app_names) != 1:
      return None
    app_name = self._task_app_names[0]
    if self._app_name_matches_activity(app_name, observation.foreground_activity):
      return None
    goal_text = plan.goal.casefold()
    app_tokens = self._app_name_tokens(app_name)
    if not app_tokens or not any(token in self._compact_text(goal_text) for token in app_tokens):
      return None
    if not _APP_LAUNCH_RE.search(goal_text):
      return None
    return json_action.JSONAction(
        action_type=json_action.OPEN_APP,
        app_name=app_name,
    )

  def _maybe_audio_recorder_shortcut(
      self,
      plan: PlanUnit,
      observation: Observation,
  ) -> json_action.JSONAction | None:
    """Click Audio Recorder's unlabeled record control from the recorder page."""
    if not _AUDIO_RECORDER_RE.search(observation.foreground_activity):
      return None
    text = f"{plan.precondition} {plan.goal}"
    if not _AUDIO_RECORD_GOAL_RE.search(text):
      return None
    candidates = []
    for index, element in enumerate(observation.ui_elements):
      if observation.grounding_indices and index not in observation.grounding_indices:
        continue
      if not bool(getattr(element, "is_clickable", False)):
        continue
      label = _visible_label(element)
      if "recording" in label or re.search(r"\brecord\b", label):
        candidates.append(index)
    if candidates:
      # Prefer the bottom-center record action over title/filename controls.
      return json_action.JSONAction(action_type=json_action.CLICK, index=candidates[-1])
    return None

  def _maybe_clock_stopwatch_shortcut(
      self,
      goal: str,
      plan: PlanUnit,
      observation: Observation,
  ) -> json_action.JSONAction | None:
    """Advance stable Clock stopwatch run/pause flows from current UI state."""
    if "clock" not in {name.casefold() for name in self._task_app_names}:
      return None
    text = f"{self._task_name} {goal} {plan.precondition} {plan.goal}"
    wants_pause = bool(_CLOCK_STOPWATCH_PAUSE_RE.search(text))
    wants_run = bool(_CLOCK_STOPWATCH_RUN_RE.search(text))
    if not (wants_run or wants_pause):
      return None
    if not self._app_name_matches_activity("clock", observation.foreground_activity):
      return json_action.JSONAction(action_type=json_action.OPEN_APP, app_name="clock")

    if wants_pause:
      pause = self._single_visible_text_match(
          observation, ("pause",), clickable_only=True
      )
      if pause is not None:
        return json_action.JSONAction(action_type=json_action.CLICK, index=pause)

    start = self._single_visible_text_match(
        observation, ("start", "resume"), clickable_only=True
    )
    if start is not None:
      return json_action.JSONAction(action_type=json_action.CLICK, index=start)

    stopwatch = self._single_visible_text_match(
        observation, ("stopwatch",), clickable_only=True
    )
    if stopwatch is not None:
      return json_action.JSONAction(action_type=json_action.CLICK, index=stopwatch)
    return None

  def _maybe_contacts_add_shortcut(
      self,
      goal: str,
      plan: PlanUnit,
      observation: Observation,
  ) -> json_action.JSONAction | None:
    """Fill the stable Contacts add-contact form from task parameters."""
    if "contacts" not in {name.casefold() for name in self._task_app_names}:
      return None
    contact = self._contact_target(goal)
    if contact is None:
      return None
    first_name, last_name, phone = contact
    text = f"{self._task_name} {goal} {plan.precondition} {plan.goal}"
    if "contact" not in text.casefold():
      return None
    if not self._app_name_matches_activity("contacts", observation.foreground_activity):
      return json_action.JSONAction(action_type=json_action.OPEN_APP, app_name="contacts")

    # If a previous wrong tap opened the account/settings sheet, close it and
    # return to the Contacts list instead of asking the small model to recover.
    close = self._single_visible_text_match(
        observation, ("close",), clickable_only=True
    )
    if close is not None and self._single_visible_text_match(
        observation, ("create contact",), clickable_only=True
    ) is None and self._field_index(observation, "First name") is None:
      return json_action.JSONAction(action_type=json_action.CLICK, index=close)

    create_contact = self._single_visible_text_match(
        observation, ("create contact",), clickable_only=True
    )
    first_index = self._field_index(observation, "First name")
    if create_contact is not None and first_index is None:
      return json_action.JSONAction(
          action_type=json_action.CLICK, index=create_contact
      )

    if first_index is not None and not self._field_has_value(
        observation, "First name", first_name
    ):
      return json_action.JSONAction(
          action_type=json_action.INPUT_TEXT,
          index=first_index,
          text=first_name,
          clear_text=True,
      )

    last_index = self._field_index(observation, "Last name")
    if last_index is not None and not self._field_has_value(
        observation, "Last name", last_name
    ):
      return json_action.JSONAction(
          action_type=json_action.INPUT_TEXT,
          index=last_index,
          text=last_name,
          clear_text=True,
      )

    phone_index = self._field_index(observation, "Phone")
    if phone_index is not None and not self._field_has_value(
        observation, "Phone", phone
    ):
      return json_action.JSONAction(
          action_type=json_action.INPUT_TEXT,
          index=phone_index,
          text=phone,
          clear_text=True,
      )

    save = self._single_visible_text_match(
        observation, ("save",), clickable_only=True
    )
    if (
        save is not None
        and self._field_has_value(observation, "First name", first_name)
        and self._field_has_value(observation, "Last name", last_name)
        and self._field_has_value(observation, "Phone", phone)
    ):
      return json_action.JSONAction(action_type=json_action.CLICK, index=save)
    return None

  def _maybe_markor_delete_shortcut(
      self,
      plan: PlanUnit,
      observation: Observation,
  ) -> json_action.JSONAction | None:
    """Advance Markor delete flow through stable app controls."""
    if "markor" not in observation.foreground_activity.casefold():
      return None
    text = f"{plan.precondition} {plan.goal}"
    if not _MARKOR_NEWEST_NOTE_RE.search(text):
      return None
    file_settings = self._single_visible_text_match(
        observation, ("file settings",), clickable_only=False
    )
    if file_settings is not None:
      return json_action.JSONAction(action_type=json_action.CLICK, index=file_settings)
    delete = self._single_visible_text_match(
        observation,
        ("delete", "delete file", "move to trash", "trash"),
        clickable_only=False,
    )
    if delete is not None:
      return json_action.JSONAction(action_type=json_action.CLICK, index=delete)
    if self._is_markor_note_editor(observation):
      more = self._single_visible_text_match(
          observation, ("more options",), clickable_only=True
      )
      if more is not None:
        return json_action.JSONAction(action_type=json_action.CLICK, index=more)
    return None

  @staticmethod
  def _is_markor_note_editor(observation: Observation) -> bool:
    labels = {
        _visible_label(element)
        for element in observation.ui_elements
        if _visible_label(element)
    }
    return bool({"undo", "redo", "save"} & labels)

  @staticmethod
  def _single_visible_text_match(
      observation: Observation,
      labels: tuple[str, ...],
      *,
      clickable_only: bool,
  ) -> int | None:
    normalized = {label.casefold() for label in labels}
    matches = []
    for index, element in enumerate(observation.ui_elements):
      if observation.grounding_indices and index not in observation.grounding_indices:
        continue
      if clickable_only and not bool(getattr(element, "is_clickable", False)):
        continue
      candidates = {
          str(value).casefold().strip()
          for value in (
              getattr(element, "text", None),
              getattr(element, "content_description", None),
              _visible_label(element),
          )
          if value
      }
      if candidates & normalized:
        matches.append(index)
    return matches[0] if len(matches) == 1 else None

  def _contact_target(self, goal: str) -> tuple[str, str, str] | None:
    name = str(self._task_params.get("name") or "").strip()
    phone = str(self._task_params.get("number") or "").strip()
    if not name or not phone:
      match = _CONTACT_ADD_GOAL_RE.search(goal)
      if match is None:
        return None
      name = match.group("name").strip()
      phone = match.group("number").strip()
    parts = name.split()
    if len(parts) < 2:
      return None
    return parts[0], " ".join(parts[1:]), phone

  @staticmethod
  def _field_index(observation: Observation, field_name: str) -> int | None:
    field = field_name.casefold()
    matches = []
    for index, element in enumerate(observation.ui_elements):
      if observation.grounding_indices and index not in observation.grounding_indices:
        continue
      if not bool(getattr(element, "is_editable", False)):
        continue
      candidates = {
          str(value).casefold().strip()
          for value in (
              getattr(element, "hint_text", None),
              getattr(element, "text", None),
              getattr(element, "content_description", None),
          )
          if value
      }
      if field in candidates:
        matches.append(index)
    return matches[0] if len(matches) == 1 else None

  @staticmethod
  def _field_has_value(
      observation: Observation, field_name: str, expected_value: str,
  ) -> bool:
    field = field_name.casefold()
    expected = expected_value.strip().casefold()
    expected_digits = re.sub(r"\D", "", expected_value)
    for element in observation.ui_elements:
      hint = str(getattr(element, "hint_text", "") or "").casefold().strip()
      if hint != field:
        continue
      text = str(getattr(element, "text", "") or "").strip()
      if field == "phone":
        if expected_digits and re.sub(r"\D", "", text) == expected_digits:
          return True
      elif text.casefold() == expected:
        return True
    return False

  def _completion_requires_target_app(
      self, observation: Observation | None,
  ) -> bool:
    """Reject successful completion when the current screen is not task-scoped."""
    if observation is None or not self._task_app_names:
      return False
    if self._task_family == "information_retrieval":
      return False
    return not any(
        self._app_name_matches_activity(app_name, observation.foreground_activity)
        for app_name in self._task_app_names
    )

  def _next_replay_actions(self, observation: Observation) -> list[json_action.JSONAction]:
    """Return one current-screen-safe replay action, skipping stale prefix steps."""
    while self._replay:
      replay_step = self._replay.pop(0)
      action = json_action.JSONAction(**replay_step.action)
      adapted = self._adapt_replayed_action(replay_step, action, observation)
      if adapted is not None:
        return [adapted]
    self._replayed_memory_id = None
    return []

  def _adapt_replayed_action(
      self,
      replay_step: TrajectoryStep,
      action: json_action.JSONAction,
      observation: Observation,
  ) -> json_action.JSONAction | None:
    """Align a retrieved trajectory action with the current observation.

    Task-level fallback memories can start from Launcher, Clock home, or
    Stopwatch. Replaying from step zero without checking the current screen can
    execute stale open-app/tab clicks. This keeps replay as an action policy,
    but only when the stored action is applicable to the visible state.
    """
    if (
        action.action_type == json_action.OPEN_APP
        and self._app_name_matches_activity(
            str(action.app_name or ""), observation.foreground_activity
        )
    ):
      return None
    if action.action_type == json_action.CLICK and action.index is not None:
      return self._rebase_replayed_click(replay_step, action, observation)
    return action

  def _rebase_replayed_click(
      self,
      replay_step: TrajectoryStep,
      action: json_action.JSONAction,
      observation: Observation,
  ) -> json_action.JSONAction | None:
    expected_label = self._stored_action_label(replay_step, int(action.index))
    if not expected_label:
      return action
    current_index = int(action.index)
    if current_index in observation.grounding_indices:
      current_label = _visible_label(observation.ui_elements[current_index])
      if current_label == expected_label:
        return action
    matches = [
        index for index, element in enumerate(observation.ui_elements)
        if index in observation.grounding_indices
        and bool(getattr(element, "is_clickable", False))
        and _visible_label(element) == expected_label
    ]
    if len(matches) != 1:
      return None
    payload = action.as_dict()
    payload["index"] = matches[0]
    return json_action.JSONAction(**payload)

  @staticmethod
  def _stored_action_label(step: TrajectoryStep, index: int) -> str:
    pattern = re.compile(
        rf"^\[{index}\]\s+text='([^']*)'; desc=(None|'([^']*)');",
        flags=re.MULTILINE,
    )
    match = pattern.search(step.observation)
    if match is None:
      return ""
    return (match.group(1) or match.group(3) or "").casefold()

  @staticmethod
  def _app_name_matches_activity(app_name: str, activity: str) -> bool:
    app_name = app_name.casefold().strip()
    activity_compact = DMSAndroidWorldAgent._compact_text(activity)
    aliases = {
        "clock": ("deskclock",),
        "files": ("documentsui",),
        "settings": ("settings",),
        "camera": ("camera",),
        "chrome": ("chrome",),
        "audio recorder": ("audiorecorder", "dimowneraudiorecorder"),
        "markor": ("markor", "gsantnermarkor"),
    }
    probes = (*DMSAndroidWorldAgent._app_name_tokens(app_name), *aliases.get(app_name, ()))
    return any(probe and probe in activity_compact for probe in probes)

  @staticmethod
  def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())

  @staticmethod
  def _app_name_tokens(app_name: str) -> tuple[str, ...]:
    compact = DMSAndroidWorldAgent._compact_text(app_name)
    return (compact,) if compact else ()

  def _recover_from_failure(
      self,
      goal: str,
      observation: Observation,
      state: interface.State,
      *,
      phase: str,
      error: Exception,
      action: json_action.JSONAction | None = None,
  ) -> base_agent.AgentInteractionResult:
    """Record a non-executable step, refresh state, and immediately replan.

    A failed model output is evidence for the Planner, but must never enter a
    reusable action memory. Returning a normal unfinished interaction result
    keeps AndroidWorld's episode runner alive instead of converting a correctable
    local failure into a skipped benchmark task.
    """
    assert self._plan is not None
    failure_action = {
        "action_type": "dms_recovery",
        "phase": phase,
        "reason": f"{type(error).__name__}: {error}",
    }
    if action is not None:
      failure_action["attempted_action"] = action.as_dict()
    failure_step = TrajectoryStep(observation.memory_snapshot(), failure_action)
    self._trace.append(failure_step)
    self._subplan_history.append(failure_step)
    self._failure_trace.append(failure_step)
    if self.memory_mode == "static" and phase == "execution" and action is not None:
      self._task_trace.append(failure_step)
    failed_plan = self._plan
    self._finish_subplan_if_needed(
        Verification(
            subplan_complete=False,
            verified=False,
            subplan_failed=True,
        ),
        detail=failure_action["reason"],
    )
    if hasattr(self.actor, "reset"):
      self.actor.reset()  # type: ignore[attr-defined]

    # Do not call the Planner from the exception handler.  When the VLM is
    # unavailable, immediately re-entering it turns a recoverable provider
    # outage into an uncaught nested exception.  The next runner step will
    # obtain a fresh observation and plan normally.
    return base_agent.AgentInteractionResult(False, {
        "raw_screenshot": state.pixels,
        "ui_elements": state.ui_elements,
        "plan": failed_plan,
        "action": failure_action,
        "origin": "recovery",
        "recovery": {
            "phase": phase,
            "reason": failure_action["reason"],
            "replanned": False,
            "next_plan": None,
        },
        "trajectory_step": failure_step,
        "audit": self._audit(observation),
        "memory_size": len(self.store.memories),
    })

  def _planning_failure(
      self, state: interface.State, observation: Observation, error: Exception
  ) -> base_agent.AgentInteractionResult:
    """Surface a Planner/provider error without executing or storing actions."""
    failure_action = {
        "action_type": "dms_recovery",
        "phase": "planner",
        "reason": f"{type(error).__name__}: {error}",
    }
    failure_step = TrajectoryStep(observation.memory_snapshot(), failure_action)
    self._failure_trace.append(failure_step)
    return base_agent.AgentInteractionResult(False, {
        "raw_screenshot": state.pixels,
        "ui_elements": state.ui_elements,
        "plan": None,
        "action": failure_action,
        "origin": "recovery",
        "recovery": {"phase": "planner", "reason": failure_action["reason"]},
        "trajectory_step": failure_step,
        "audit": self._audit(observation),
        "memory_size": len(self.store.memories),
    })

  def _finish_subplan_if_needed(
      self, verdict: Verification, *, detail: str = ""
  ) -> None:
    if (not verdict.subplan_complete and not verdict.subplan_failed) or self._plan is None:
      return
    if hasattr(self.planner, "record_result"):
      self.planner.record_result(  # type: ignore[attr-defined]
          self._plan,
          completed=verdict.subplan_complete and not verdict.subplan_failed,
          detail=detail,
      )
    successful = (
        verdict.subplan_complete and verdict.verified and not verdict.subplan_failed
    )
    if self._replayed_memory_id:
      if self.memory_mode == "dms":
        self.store.record_verification(self._replayed_memory_id, verified=verdict.verified)
    elif self._mutation_memory_id:
      # Mutations never create a duplicate entry. A failed/equal/longer
      # candidate leaves the retrieved memory intact, exactly as §3.2.2.
      if successful:
        self.store.replace_if_better(
            self._mutation_memory_id, self._trace, verified=True
        )
    elif self.memory_mode == "dms" and successful and len(self._trace) >= 2:
      memory = self.store.add(MemoryUnit(
          plan=self._plan,
          trajectory=list(self._trace),
          description=(
              f"task_template={self._task_name or 'unknown'}; "
              f"task_apps={','.join(self._task_app_names) or 'unknown'}; "
              f"goal={self._task_goal or 'unknown'}"
          ),
      ))
      self._active_memory_ids.add(memory.id)
    self._clear_subplan()

  def _clear_subplan(self) -> None:
    self._plan = None
    self._trace = []
    self._subplan_history = []
    self._replay = []
    self._replayed_memory_id = None
    self._mutation_memory_id = None

  @staticmethod
  def _canonical_action(action: json_action.JSONAction) -> str:
    """Serialize executable intent fields for repeat detection."""
    return json.dumps(action.as_dict(), sort_keys=True, ensure_ascii=False)

  def _is_repeated_no_progress_action(
      self, action: json_action.JSONAction, observation: Observation
  ) -> bool:
    if action.action_type not in _UI_CHANGING_ACTIONS:
      return False
    signature = (observation.screen_fingerprint(), self._canonical_action(action))
    return signature in self._recent_action_fingerprints

  def _record_action_progress(
      self,
      action: json_action.JSONAction,
      before_observation: Observation,
      after_observation: Observation,
  ) -> None:
    if action.action_type not in _UI_CHANGING_ACTIONS:
      return
    before_hash = before_observation.screen_fingerprint()
    after_hash = after_observation.screen_fingerprint()
    if before_hash == after_hash:
      self._recent_action_fingerprints.append((
          after_hash, self._canonical_action(action)
      ))
      self._recent_action_fingerprints = self._recent_action_fingerprints[-6:]
    else:
      self._recent_action_fingerprints.clear()

  @staticmethod
  def _should_stop_action_block_after(action: json_action.JSONAction) -> bool:
    return action.action_type in _UI_CHANGING_ACTIONS
