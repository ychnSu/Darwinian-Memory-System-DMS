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

"""Run eval suite.

The run.py module is used to run a suite of tasks, with configurable task
combinations, environment setups, and agent configurations. You can run specific
tasks or all tasks in the suite and customize various settings using the
command-line flags.
"""

from collections.abc import Sequence
import json
import os
from pathlib import Path
import shutil

from absl import app
from absl import flags
from absl import logging
from android_world import checkpointer as checkpointer_lib
from android_world import constants
from android_world import registry
from android_world import suite_utils
from android_world.agents import base_agent
from android_world.agents import human_agent
from android_world.agents import infer
from android_world.agents import m3a
from android_world.agents import random_agent
from android_world.agents import seeact
from android_world.agents import t3a
from android_world.env import env_launcher
from android_world.env import interface
from dms.experiment_logging import IncrementalCsvLedger

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

_SUITE_FAMILY = flags.DEFINE_enum(
    'suite_family',
    registry.TaskRegistry.ANDROID_WORLD_FAMILY,
    [
        # Families from the paper.
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        # Other families for more testing.
        registry.TaskRegistry.MINIWOB_FAMILY,
        registry.TaskRegistry.ANDROID_FAMILY,
        registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
    ],
    'Suite family to run. See registry.py for more information.',
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    'task_random_seed', 30, 'Random seed for task randomness.'
)

_TASKS = flags.DEFINE_list(
    'tasks',
    None,
    'List of specific tasks to run in the given suite family. If None, run all'
    ' tasks in the suite family.',
)
_TASK_DIFFICULTY = flags.DEFINE_string(
    'task_difficulty',
    '',
    'Optional AndroidWorld difficulty filter: easy, medium, or hard.',
)
_BENCHMARK_CONFIG = flags.DEFINE_string(
    'benchmark_config',
    '',
    'Optional JSON benchmark config. When provided, it supplies suite_family, '
    'n_task_combinations, task_random_seed, tasks, and task_difficulty. '
    '--tasks and --task_difficulty override the config task list/filter.',
)
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    'n_task_combinations',
    5,
    'Number of task instances to run for each task template.',
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'The directory to save checkpoints and resume evaluation from. If the'
    ' directory contains existing checkpoint files, evaluation will resume from'
    ' the latest checkpoint. If the directory is empty or does not exist, a new'
    ' directory will be created.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    os.path.expanduser('~/android_world/runs'),
    'The path to save results to if not resuming from a checkpoint is not'
    ' provided.',
)
_RESULTS_CSV = flags.DEFINE_string(
    'results_csv', '',
    'Append-only per-task CSV ledger. Defaults to results.csv in the run checkpoint directory.',
)

# Agent specific.
_AGENT_NAME = flags.DEFINE_string('agent_name', 'm3a_gpt4v', help='Agent name.')

_DMS_EMBEDDING_URL = flags.DEFINE_string(
    'dms_embedding_url', 'http://127.0.0.1:18001/embed',
    'CPU BGE embedding endpoint, normally reached through the SSH tunnel.',
)
_DMS_EMBEDDING_PROVIDER = flags.DEFINE_enum(
    'dms_embedding_provider', 'local', ['local', 'remote'],
    'Embedding backend for DMS semantic retrieval. Use local for an on-disk '
    'sentence-transformers/BGE model, or remote for the HTTP embedding service.',
)
_DMS_EMBEDDING_MODEL_PATH = flags.DEFINE_string(
    'dms_embedding_model_path',
    os.path.expanduser('~/projects/android_world/embedding_model'),
    'Local sentence-transformers/BGE model directory used when '
    '--dms_embedding_provider=local.',
)
_DMS_EMBEDDING_DEVICE = flags.DEFINE_string(
    'dms_embedding_device',
    'cpu',
    'Local embedding device, for example cpu, cuda, or cuda:0.',
)
_DMS_MEMORY_DIR = flags.DEFINE_string(
    'dms_memory_dir', os.path.expanduser('~/.cache/android_world_dms'),
    'Directory for separate DMS and Static Memory JSON stores.',
)
_DMS_EPSILON = flags.DEFINE_float(
    'dms_epsilon', 0.10, 'DMS epsilon mutation probability.'
)
_DMS_RISK_THRESHOLD = flags.DEFINE_float(
    'dms_risk_threshold', 0.35, 'DMS replay risk threshold.'
)
_DMS_THRESHOLD_SENSITIVITY = flags.DEFINE_float(
    'dms_threshold_sensitivity', 0.30,
    'Paper dynamic risk-threshold sensitivity lambda.',
)
_DMS_PRUNE_INTERVAL_TASKS = flags.DEFINE_integer(
    'dms_prune_interval_tasks', 5,
    'Run DMS scheduled Elbow pruning after this many completed tasks.',
)
_DMS_MEMORY_CAPACITY_MIN = flags.DEFINE_integer(
    'dms_memory_capacity_min', 50,
    'DMS C_min: pruning/expansion check starts when memory size reaches this capacity.',
)
_DMS_MEMORY_CAPACITY_MAX = flags.DEFINE_integer(
    'dms_memory_capacity_max', 200,
    'DMS C_max: maximum adaptive memory capacity after high-quality saturation expansion.',
)
_DMS_MEMORY_CAPACITY_STEP = flags.DEFINE_integer(
    'dms_memory_capacity_step', 25,
    'DMS capacity expansion step used when elbow cutoff is above population mean.',
)
_DMS_VLM_BASE_URL = flags.DEFINE_string(
    'dms_vlm_base_url', 'http://127.0.0.1:18000/v1',
    'OpenAI-compatible VLM endpoint reached through the SSH tunnel.',
)
_DMS_VLM_MODEL = flags.DEFINE_string(
    'dms_vlm_model', '/root/autodl-tmp/model',
    'Model identifier advertised by the remote vLLM server.',
)
_DMS_TRANSITION_PAUSE = flags.DEFINE_float(
    'dms_transition_pause',
    1.25,
    'Fixed seconds to wait before each DMS/PA-Lite observation. Use a larger '
    'value for slow emulators with app launches, page transitions, or dialogs.',
)

_FIXED_TASK_SEED = flags.DEFINE_boolean(
    'fixed_task_seed',
    False,
    'Whether to use the same task seed when running multiple task combinations'
    ' (n_task_combinations > 1).',
)


# MiniWoB is very lightweight and new screens/View Hierarchy load quickly.
_MINIWOB_TRANSITION_PAUSE = 0.2

# Additional guidelines for the MiniWob tasks.
_MINIWOB_ADDITIONAL_GUIDELINES = [
    (
        'This task is running in a mock app, you must stay in this app and'
        ' DO NOT use the `navigate_home` action.'
    ),
]


def _get_agent(
    env: interface.AsyncEnv,
    family: str | None = None,
) -> base_agent.EnvironmentInteractingAgent:
  """Gets agent."""
  print('Initializing agent...')
  agent = None
  if _AGENT_NAME.value == 'human_agent':
    agent = human_agent.HumanAgent(env)
  elif _AGENT_NAME.value == 'random_agent':
    agent = random_agent.RandomAgent(env)
  # Gemini.
  elif _AGENT_NAME.value == 'm3a_gemini_gcp':
    agent = m3a.M3A(
        env, infer.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest')
    )
  elif _AGENT_NAME.value == 't3a_gemini_gcp':
    agent = t3a.T3A(
        env, infer.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest')
    )
  # GPT.
  elif _AGENT_NAME.value == 't3a_gpt4':
    agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
  elif _AGENT_NAME.value == 'm3a_gpt4v':
    agent = m3a.M3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
  # SeeAct.
  elif _AGENT_NAME.value == 'seeact':
    agent = seeact.SeeAct(env)
  elif _AGENT_NAME.value in ('dms', 'static_memory', 'zero_shot'):
    # Import only for DMS modes so existing AndroidWorld agents retain their
    # original dependency surface.
    from dms.actor import QwenVLActor
    from dms.androidworld_agent import DMSAndroidWorldAgent
    from dms.config import get_experiment_mode
    from dms.embeddings import LocalEmbeddingProvider, RemoteEmbeddingProvider
    from dms.memory import DMSMemoryStore
    from dms.planner import HierarchicalDMSPlanner
    from dms.policy import DMSPolicy
    from dms.verifier import AndroidWorldVerifierAdapter, QwenVLVerifier
    from dms.vlm import OpenAICompatibleVLM

    from dms.zero_shot import ZeroShotMemoryStore

    experiment = get_experiment_mode(_AGENT_NAME.value)
    if _DMS_EMBEDDING_PROVIDER.value == 'local':
      embedder = LocalEmbeddingProvider(
          _DMS_EMBEDDING_MODEL_PATH.value,
          device=_DMS_EMBEDDING_DEVICE.value,
      )
    else:
      embedder = RemoteEmbeddingProvider(_DMS_EMBEDDING_URL.value)
    if experiment.memory_filename is None:
      store = ZeroShotMemoryStore()
    else:
      memory_dir = Path(_DMS_MEMORY_DIR.value)
      memory_dir.mkdir(parents=True, exist_ok=True)
      store = DMSMemoryStore(
          memory_dir / experiment.memory_filename,
          embedder=embedder,
          static=(experiment.memory_mode == 'static'),
          retrieval_strategy=experiment.retrieval_strategy,
          capacity_min=_DMS_MEMORY_CAPACITY_MIN.value,
          capacity_max=_DMS_MEMORY_CAPACITY_MAX.value,
          capacity_step=_DMS_MEMORY_CAPACITY_STEP.value,
      )
    model = OpenAICompatibleVLM(
        base_url=_DMS_VLM_BASE_URL.value,
        model=_DMS_VLM_MODEL.value,
    )
    verifier = (
        None if experiment.memory_mode in ('zero_shot', 'static')
        else AndroidWorldVerifierAdapter(QwenVLVerifier(model))
    )
    agent = DMSAndroidWorldAgent(
        env=env,
        planner=HierarchicalDMSPlanner(model),
        actor=QwenVLActor(model),
        verifier=verifier,
        store=store,
        policy=DMSPolicy(
            epsilon=_DMS_EPSILON.value if experiment.enable_mutation else 0.0,
            risk_threshold=_DMS_RISK_THRESHOLD.value,
            threshold_sensitivity=_DMS_THRESHOLD_SENSITIVITY.value,
            seed=_TASK_RANDOM_SEED.value,
        ),
        name=_AGENT_NAME.value,
        memory_mode=experiment.memory_mode,
    )
    agent.token_tracker = model  # pytype: disable=attribute-error

  if not agent:
    raise ValueError(f'Unknown agent: {_AGENT_NAME.value}')

  if (
      agent.name in ['M3A', 'T3A', 'SeeAct']
      and family
      and family.startswith('miniwob')
      and hasattr(agent, 'set_task_guidelines')
  ):
    agent.set_task_guidelines(_MINIWOB_ADDITIONAL_GUIDELINES)
  agent.name = _AGENT_NAME.value

  return agent


def _load_benchmark_config(path: str) -> dict[str, object]:
  """Load a local JSON benchmark configuration."""
  if not path:
    return {}
  config_path = Path(path).expanduser()
  if not config_path.is_absolute():
    config_path = Path(__file__).parent / config_path
  with config_path.open('r', encoding='utf-8') as handle:
    config = json.load(handle)
  if not isinstance(config, dict):
    raise ValueError('--benchmark_config must point to a JSON object.')
  return config


def _benchmark_tasks(config: dict[str, object]) -> list[str] | None:
  """Return task names from a benchmark config, preserving declared order."""
  raw_tasks = config.get('tasks')
  if raw_tasks is None:
    return None
  if not isinstance(raw_tasks, list):
    raise ValueError('benchmark_config.tasks must be a list.')
  tasks: list[str] = []
  for item in raw_tasks:
    if isinstance(item, str):
      name = item
    elif isinstance(item, dict):
      name = str(item.get('task_name') or item.get('name') or '')
    else:
      raise ValueError('Each benchmark task must be a string or object.')
    if not name:
      raise ValueError('Each benchmark task requires task_name or name.')
    tasks.append(name)
  return tasks


def _filter_suite_by_difficulty(
    suite: suite_utils.Suite,
    difficulty: str,
) -> suite_utils.Suite:
  """Keep only task templates with the requested AndroidWorld difficulty."""
  difficulty = difficulty.strip().casefold()
  if difficulty not in ('easy', 'medium', 'hard'):
    raise ValueError("--task_difficulty must be one of: easy, medium, hard.")
  metadata_path = Path(__file__).parent / 'android_world' / 'task_metadata.json'
  with metadata_path.open('r', encoding='utf-8') as handle:
    metadata = json.load(handle)
  allowed = {
      str(item['task_name'])
      for item in metadata
      if str(item.get('difficulty', '')).casefold() == difficulty
  }
  filtered = suite_utils.Suite(
      (name, instances) for name, instances in suite.items() if name in allowed
  )
  filtered.suite_family = suite.suite_family
  if not filtered:
    raise ValueError(f'No tasks remain after --task_difficulty={difficulty}.')
  return filtered


def _main() -> None:
  """Runs eval suite and gets rewards back."""
  if _AGENT_NAME.value == 'dms' and _DMS_PRUNE_INTERVAL_TASKS.value < 1:
    raise ValueError('--dms_prune_interval_tasks must be positive.')
  if _DMS_MEMORY_CAPACITY_MIN.value < 2:
    raise ValueError('--dms_memory_capacity_min must be at least 2.')
  if _DMS_MEMORY_CAPACITY_MAX.value < _DMS_MEMORY_CAPACITY_MIN.value:
    raise ValueError('--dms_memory_capacity_max must be >= --dms_memory_capacity_min.')
  if _DMS_MEMORY_CAPACITY_STEP.value < 1:
    raise ValueError('--dms_memory_capacity_step must be positive.')
  benchmark_config = _load_benchmark_config(_BENCHMARK_CONFIG.value)
  suite_family = str(
      benchmark_config.get('suite_family', _SUITE_FAMILY.value)
  )
  n_task_combinations = int(
      benchmark_config.get('n_task_combinations', _N_TASK_COMBINATIONS.value)
  )
  task_random_seed = int(
      benchmark_config.get('task_random_seed', _TASK_RANDOM_SEED.value)
  )
  tasks = _TASKS.value or _benchmark_tasks(benchmark_config)
  task_difficulty = (
      _TASK_DIFFICULTY.value
      or str(benchmark_config.get('task_difficulty', '') or '')
  )
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
  )

  task_registry = registry.TaskRegistry()
  suite = suite_utils.create_suite(
      task_registry.get_registry(family=suite_family),
      n_task_combinations=n_task_combinations,
      seed=task_random_seed,
      tasks=tasks,
      use_identical_params=_FIXED_TASK_SEED.value,
  )
  suite.suite_family = suite_family
  if task_difficulty:
    suite = _filter_suite_by_difficulty(suite, task_difficulty)

  agent = _get_agent(env, suite_family)

  if _AGENT_NAME.value in ('dms', 'static_memory', 'zero_shot'):
    agent.transition_pause = _DMS_TRANSITION_PAUSE.value
  elif suite_family.startswith('miniwob'):
    # MiniWoB pages change quickly, don't need to wait for screen to stabilize.
    agent.transition_pause = _MINIWOB_TRANSITION_PAUSE
  else:
    agent.transition_pause = None

  if _CHECKPOINT_DIR.value:
    checkpoint_dir = _CHECKPOINT_DIR.value
  else:
    checkpoint_dir = checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)
  results_csv = (
      Path(_RESULTS_CSV.value)
      if _RESULTS_CSV.value else Path(checkpoint_dir) / 'results.csv'
  )
  ledger = IncrementalCsvLedger(results_csv, agent_name=_AGENT_NAME.value)

  print(
      f'Starting eval with agent {_AGENT_NAME.value} and writing to'
      f' {checkpoint_dir}\nIncremental results CSV: {results_csv}'
  )
  if benchmark_config:
    print(
        f"Benchmark config: {benchmark_config.get('name', _BENCHMARK_CONFIG.value)}; "
        f"tasks={len(tasks or [])}; n_task_combinations={n_task_combinations}; "
        f"seed={task_random_seed}"
    )
  completed_dms_tasks = 0
  last_usage_total = 0
  last_usage_prompt = 0
  last_usage_completion = 0
  last_usage_responses = 0
  completed_episode_metrics = []

  def _memory_stats() -> dict[str, object]:
    store = getattr(agent, 'store', None)
    stats = getattr(store, 'memory_stats', None)
    if callable(stats):
      return dict(stats())
    return {}

  def _record_task_result(task, task_success: float) -> None:
    nonlocal completed_dms_tasks
    del task
    if _AGENT_NAME.value in ('dms', 'static_memory'):
      agent.finalize_task(task_succeeded=task_success > 0.5)  # pytype: disable=attribute-error
    if _AGENT_NAME.value == 'dms':
      completed_dms_tasks += 1
      if completed_dms_tasks % _DMS_PRUNE_INTERVAL_TASKS.value == 0:
        agent.store.prune()  # pytype: disable=attribute-error

  def _record_episode(episodes, print_summary: bool = False) -> None:
    """Append the just-checkpointed result durably before the next task runs."""
    nonlocal last_usage_total, last_usage_prompt, last_usage_completion
    nonlocal last_usage_responses
    latest = episodes[-1]
    token_tracker = getattr(agent, 'token_tracker', None)
    snapshot_usage = getattr(token_tracker, 'snapshot_usage', None)
    if callable(snapshot_usage):
      usage = snapshot_usage()
      if usage.responses_with_usage > last_usage_responses:
        latest['token_total'] = usage.total_tokens - last_usage_total
        latest['token_prompt'] = usage.prompt_tokens - last_usage_prompt
        latest['token_completion'] = (
            usage.completion_tokens - last_usage_completion
        )
      last_usage_total = usage.total_tokens
      last_usage_prompt = usage.prompt_tokens
      last_usage_completion = usage.completion_tokens
      last_usage_responses = usage.responses_with_usage
    memory_stats = _memory_stats()
    latest.update(memory_stats)
    ledger.append_episode(latest)
    completed_episode_metrics.append({
        'trial_index': len(completed_episode_metrics) + 1,
        'task_template': latest.get(constants.EpisodeConstants.TASK_TEMPLATE),
        'k': int(latest.get(constants.EpisodeConstants.INSTANCE_ID, 0)) + 1,
        'success': latest.get(constants.EpisodeConstants.IS_SUCCESSFUL),
        'episode_length': latest.get(constants.EpisodeConstants.EPISODE_LENGTH),
        'token_total': latest.get('token_total'),
        **memory_stats,
    })
    if print_summary and _AGENT_NAME.value in ('dms', 'static_memory', 'zero_shot'):
      _print_focused_live_metrics(completed_episode_metrics)
    else:
      suite_utils.process_episodes(episodes, print_summary=print_summary)

  def _metric_float(metric: dict[str, object], key: str) -> float | None:
    value = metric.get(key)
    if value in ('', None):
      return None
    try:
      return float(value)
    except (TypeError, ValueError):
      return None

  def _mean(values: list[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None

  def _fmt(value: float | None, digits: int = 2) -> str:
    return '-' if value is None else f'{value:.{digits}f}'

  def _print_focused_live_metrics(metrics: list[dict[str, object]]) -> None:
    """Print DMS-focused online metrics instead of AndroidWorld's legacy table."""
    valid = [
        metric for metric in metrics
        if isinstance(metric.get('success'), (int, float, bool))
    ]
    if not valid:
      return
    successes = [float(metric['success']) for metric in valid]
    steps = [_metric_float(metric, 'episode_length') for metric in valid]
    tokens = [_metric_float(metric, 'token_total') for metric in valid]
    latest = valid[-1]
    overall_sr = sum(successes) / len(successes)
    print('\nDMS live metrics')
    print(
        'overall: '
        f'trials={len(valid)} '
        f'SR={overall_sr:.4f} '
        f'last={int(float(latest["success"]))} '
        f'avg_steps={_fmt(_mean(steps))} '
        f'avg_tokens={_fmt(_mean(tokens))} '
        f'memory_size={_fmt(_metric_float(latest, "memory_size"), 0)} '
        f'hit_rate={_fmt(_metric_float(latest, "memory_retrieval_hit_rate"), 3)} '
        f'reuse_rate={_fmt(_metric_float(latest, "memory_reuse_rate"), 3)} '
        f'mutations={_fmt(_metric_float(latest, "memory_total_mutations"), 0)} '
        f'pruned={_fmt(_metric_float(latest, "memory_total_pruned"), 0)}'
    )
    print(
        'task'.ljust(34)
        + 'trials  SR     last  avg_steps  avg_tokens  mem  hit    reuse  replay  mutate  pruned'
    )
    per_task: dict[str, list[dict[str, object]]] = {}
    for metric in valid:
      task = str(metric.get('task_template') or 'unknown')
      per_task.setdefault(task, []).append(metric)
    for task in sorted(per_task):
      rows = per_task[task]
      task_successes = [float(row['success']) for row in rows]
      row_steps = [_metric_float(row, 'episode_length') for row in rows]
      row_tokens = [_metric_float(row, 'token_total') for row in rows]
      last = rows[-1]
      print(
          task[:33].ljust(34)
          + f'{len(rows):>6}  '
          + f'{(sum(task_successes) / len(task_successes)):>5.2f}  '
          + f'{int(float(last["success"])):>4}  '
          + f'{_fmt(_mean(row_steps)):>9}  '
          + f'{_fmt(_mean(row_tokens)):>10}  '
          + f'{_fmt(_metric_float(last, "memory_size"), 0):>3}  '
          + f'{_fmt(_metric_float(last, "memory_retrieval_hit_rate"), 2):>5}  '
          + f'{_fmt(_metric_float(last, "memory_reuse_rate"), 2):>5}  '
          + f'{_fmt(_metric_float(last, "memory_total_replays"), 0):>6}  '
          + f'{_fmt(_metric_float(last, "memory_total_mutations"), 0):>6}  '
          + f'{_fmt(_metric_float(last, "memory_total_pruned"), 0):>6}'
      )
    print()

  suite_utils.run(
      suite,
      agent,
      checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
      demo_mode=False,
      process_episodes_fn=_record_episode,
      on_task_result=(
          _record_task_result
          if _AGENT_NAME.value in ('dms', 'static_memory') else None
      ),
  )
  valid_metrics = [
      metric for metric in completed_episode_metrics
      if isinstance(metric.get('success'), (int, float, bool))
  ]
  if valid_metrics:
    success_values = [float(metric['success']) for metric in valid_metrics]
    step_values = [
        float(metric['episode_length']) for metric in valid_metrics
        if metric.get('episode_length') not in ('', None)
    ]
    token_values = [
        float(metric['token_total']) for metric in valid_metrics
        if metric.get('token_total') not in ('', None)
    ]
    cumulative_successes = 0.0
    per_task_successes: dict[str, float] = {}
    per_task_counts: dict[str, int] = {}
    trial_metrics = []
    for index, metric in enumerate(valid_metrics, start=1):
      success = float(metric['success'])
      cumulative_successes += success
      task_template = str(metric.get('task_template') or 'unknown')
      per_task_successes[task_template] = (
          per_task_successes.get(task_template, 0.0) + success
      )
      per_task_counts[task_template] = per_task_counts.get(task_template, 0) + 1
      trial_metrics.append({
          'trial_index': index,
          'task_template': task_template,
          'k': metric.get('k'),
          'success': success,
          'cumulative_SR': cumulative_successes / index,
          'task_cumulative_SR': (
              per_task_successes[task_template] / per_task_counts[task_template]
          ),
          'episode_length': metric.get('episode_length'),
          'token_total': metric.get('token_total'),
          'memory_size': metric.get('memory_size'),
          'memory_survival_mean': metric.get('memory_survival_mean'),
          'memory_invalid_click_rate_mean': metric.get(
              'memory_invalid_click_rate_mean'
          ),
          'memory_last_pruned_count': metric.get('memory_last_pruned_count'),
          'memory_current_capacity': metric.get('memory_current_capacity'),
          'memory_total_created': metric.get('memory_total_created'),
          'memory_total_retrievals': metric.get('memory_total_retrievals'),
          'memory_total_retrieval_hits': metric.get(
              'memory_total_retrieval_hits'
          ),
          'memory_total_retrieval_misses': metric.get(
              'memory_total_retrieval_misses'
          ),
          'memory_total_suppressed': metric.get('memory_total_suppressed'),
          'memory_total_replays': metric.get('memory_total_replays'),
          'memory_total_mutations': metric.get('memory_total_mutations'),
          'memory_total_replacements': metric.get(
              'memory_total_replacements'
          ),
          'memory_total_pruned': metric.get('memory_total_pruned'),
          'memory_total_deleted_by_risk': metric.get(
              'memory_total_deleted_by_risk'
          ),
          'memory_total_embedding_failures': metric.get(
              'memory_total_embedding_failures'
          ),
          'memory_retrieval_hit_rate': metric.get(
              'memory_retrieval_hit_rate'
          ),
          'memory_reuse_rate': metric.get('memory_reuse_rate'),
          'memory_mutation_rate': metric.get('memory_mutation_rate'),
      })
    memory_size_values = [
        float(metric['memory_size']) for metric in valid_metrics
        if metric.get('memory_size') not in ('', None)
    ]
    summary = {
        'agent_name': _AGENT_NAME.value,
        'suite_family': suite_family,
        'task_difficulty': task_difficulty or 'all',
        'benchmark_config': _BENCHMARK_CONFIG.value or None,
        'num_tasks': len(valid_metrics),
        'SR': sum(success_values) / len(success_values),
        'avg_steps_per_task': (
            sum(step_values) / len(step_values) if step_values else None
        ),
        'avg_tokens_per_task': (
            sum(token_values) / len(token_values) if token_values else None
        ),
        'memory_size_final': (
            memory_size_values[-1] if memory_size_values else None
        ),
        'memory_size_max': (
            max(memory_size_values) if memory_size_values else None
        ),
        'trial_metrics': trial_metrics,
    }
    avg_steps = summary['avg_steps_per_task']
    avg_tokens = summary['avg_tokens_per_task']
    summary_path = Path(results_csv).parent / 'summary.json'
    with summary_path.open('w', encoding='utf-8') as handle:
      json.dump(summary, handle, indent=2, sort_keys=True)
    avg_steps_text = (
        f"{avg_steps:.2f}" if avg_steps is not None else "null"
    )
    avg_tokens_text = (
        f"{avg_tokens:.2f}" if avg_tokens is not None else "null"
    )
    print(
        f"Summary: SR={summary['SR']:.4f}, "
        f"avg_steps_per_task={avg_steps_text}, "
        f"avg_tokens_per_task={avg_tokens_text}, "
        f"num_tasks={summary['num_tasks']}. Wrote {summary_path}"
    )
  print(
      f'Finished running agent {_AGENT_NAME.value} on {suite_family}'
      f' family. Wrote to {checkpoint_dir}.'
  )
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
