# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs a single task.

The minimal_run.py module is used to run a single task, it is a minimal version
of the run.py module. A task can be specified, otherwise a random task is
selected.
"""

from collections.abc import Sequence
import os
import random
import shutil
from typing import Type

from absl import app
from absl import flags
from absl import logging
from android_world import registry
from android_world import suite_utils
from android_world.agents import infer
from android_world.agents import t3a
from android_world.env import env_launcher
from android_world.task_evals import task_eval

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing


def _find_adb_directory() -> str:
  """Returns the directory where adb is located."""
  potential_paths = [
      os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
      os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
  ]
  for path in potential_paths:
    if os.path.isfile(path):
      return path
  adb_on_path = shutil.which('adb')
  if adb_on_path:
    return adb_on_path
  raise EnvironmentError(
      'adb not found in the common Android SDK paths. Please install Android'
      " SDK and ensure adb is in one of the expected directories. If it's"
      ' already installed, point to the installed location.'
  )


_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    _find_adb_directory(),
    'Path to adb. Set if not installed through SDK.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    False,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5554,
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
)

_TASK = flags.DEFINE_string(
    'task',
    None,
    'A specific task to run.',
)
_AGENT_NAME = flags.DEFINE_enum(
    'agent_name',
    't3a_gpt4',
    ['t3a_gpt4', 'zero_shot'],
    'Agent to run. Use zero_shot for the PA-Lite / Baseline A implementation.',
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    'task_random_seed',
    30,
    'Random seed used when generating task parameters.',
)
_DMS_VLM_BASE_URL = flags.DEFINE_string(
    'dms_vlm_base_url',
    'http://127.0.0.1:18000/v1',
    'OpenAI-compatible VLM endpoint reached through the SSH tunnel.',
)
_DMS_VLM_MODEL = flags.DEFINE_string(
    'dms_vlm_model',
    '/root/autodl-tmp/model',
    'Model identifier advertised by the remote vLLM server.',
)
_DMS_TRANSITION_PAUSE = flags.DEFINE_float(
    'dms_transition_pause',
    1.25,
    'Fixed seconds to wait before each Baseline A observation.',
)


def _build_agent(env):
  """Build the requested agent while keeping this runner minimal."""
  if _AGENT_NAME.value == 't3a_gpt4':
    return t3a.T3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
  if _AGENT_NAME.value == 'zero_shot':
    from dms.actor import QwenVLActor
    from dms.androidworld_agent import DMSAndroidWorldAgent
    from dms.policy import DMSPolicy
    from dms.vlm import OpenAICompatibleVLM
    from dms.zero_shot import ZeroShotMemoryStore
    from dms.planner import HierarchicalDMSPlanner

    model = OpenAICompatibleVLM(
        base_url=_DMS_VLM_BASE_URL.value,
        model=_DMS_VLM_MODEL.value,
    )
    agent = DMSAndroidWorldAgent(
        env=env,
        planner=HierarchicalDMSPlanner(model),
        actor=QwenVLActor(model),
        verifier=None,
        store=ZeroShotMemoryStore(),
        policy=DMSPolicy(epsilon=0.0),
        name='zero_shot',
        memory_mode='zero_shot',
    )
    agent.token_tracker = model  # pytype: disable=attribute-error
    return agent
  raise ValueError(f'Unsupported agent: {_AGENT_NAME.value}')


def _main() -> None:
  """Runs a single task."""
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
  )
  env.reset(go_home=True)
  random.seed(_TASK_RANDOM_SEED.value)
  task_registry = registry.TaskRegistry()
  aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
  if _TASK.value:
    if _TASK.value not in aw_registry:
      raise ValueError('Task {} not found in registry.'.format(_TASK.value))
    task_type: Type[task_eval.TaskEval] = aw_registry[_TASK.value]
  else:
    task_type: Type[task_eval.TaskEval] = random.choice(
        list(aw_registry.values())
    )
  params = task_type.generate_random_params()
  task = task_type(params)
  task.initialize_task(env)
  agent = _build_agent(env)
  if _AGENT_NAME.value == 'zero_shot':
    agent.transition_pause = _DMS_TRANSITION_PAUSE.value
  suite_utils._inject_dms_task_context(task, agent)  # pylint: disable=protected-access

  print('Goal: ' + str(task.goal))
  is_done = False
  max_steps = int(task.complexity * 10)
  for step_index in range(max_steps):
    response = agent.step(task.goal)
    print(f'Completed step {step_index + 1}.')
    if response.done:
      is_done = True
      break
  evaluator_successful = task.is_successful(env) == 1
  agent_successful = is_done and evaluator_successful
  token_tracker = getattr(agent, 'token_tracker', None)
  snapshot_usage = getattr(token_tracker, 'snapshot_usage', None)
  if callable(snapshot_usage):
    usage = snapshot_usage()
    print(
        'Token usage: '
        f'total={usage.total_tokens}, '
        f'prompt={usage.prompt_tokens}, '
        f'completion={usage.completion_tokens}'
    )
  print(
      f'Agent indicated done: {is_done}; '
      f'AndroidWorld evaluator success: {evaluator_successful}; '
      f'steps_used={step_index + 1 if max_steps else 0}; '
      f'max_steps={max_steps}'
  )
  print(
      f'{"Task Successful ✅" if agent_successful else "Task Failed ❌"};'
      f' {task.goal}'
  )
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
