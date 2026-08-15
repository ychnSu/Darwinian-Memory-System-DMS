# Darwinian Memory System (DMS) 复现技术报告

## 1. 项目介绍

本项目在AndroidWorld 任务环境中复现论文 **Darwinian Memory: A Training-Free Self-Regulating Memory System for GUI Agent Evolution** 中提出的 Darwinian Memory System（DMS），用于评估 GUI Agent 在多轮任务执行中的经验积累、记忆复用和自我调节能力。

本复现工程关注的问题是：在不训练模型参数的前提下，是否可以通过外部记忆系统提升 GUI Agent 在重复试验中的执行效率和成功率。为此，项目在相同模型、评测环境以及任务集合上比较三种方法：

|方法|简称|核心特征|
|-|-|-|
|Zero-shot Planner-Actor|Baseline A / PA-Lite|不使用历史记忆，仅基于当前任务、当前截图和当前 UI tree 规划并执行|
|Static Memory|Baseline B|将历史任务轨迹按时间顺序追加保存，并在 Planner 阶段作为上下文参考，不做检索、变异、替换或剪枝|
|Darwinian Memory System|DMS|构建可检索、可复用、可反馈更新、可变异替换、可动态剪枝的经验记忆库|

其中，Baseline A 用于衡量模型在无经验积累条件下的基础 GUI 操作能力；Baseline B 用于衡量传统静态历史上下文是否能带来收益；DMS 则用于验证论文提出的动态记忆机制是否能在多轮任务中产生更稳定的经验复用和性能演化。

本项目的复现目标是构建一套可重复运行、可审计、可对比的实验链路。系统记录任务成功率（Success Rate, SR）、平均执行步数、token 消耗、记忆库大小、检索命中率和剪枝行为，并通过可视化展示这些指标随 trial 增加的变化趋势。

## 2. 项目结构

本项目基于 AndroidWorld 官方代码进行扩展，核心改动集中在 `dms/`、`run.py`、`scripts/` 和 `configs/` 中。

```text
android_world/
├── dms/                         # DMS、PA-Lite、Static Memory 的核心实现
│   ├── androidworld_agent.py     # Planner-Actor-DMS 与 AndroidWorld 环境的适配层
│   ├── planner.py                # Planner：将任务拆解为当前 UI 下的子目标
│   ├── actor.py                  # Actor：根据子目标输出单步 AndroidWorld action
│   ├── verifier.py               # DMS 子任务验证模块
│   ├── memory.py                 # 分层记忆、检索、反馈、替换与剪枝
│   ├── policy.py                 # replay / mutation / risk threshold 策略
│   ├── embeddings.py             # 本地或远程 embedding 接口
│   └── experiment_logging.py     # 实验结果增量记录
│
├── configs/
│   └── mini_benchmark.json       # mini-benchmark 任务集合配置
│
├── scripts/
│   ├── run_mini_benchmark.sh             # Baseline A / PA-Lite 测试入口
│   ├── run_baseline_b_mini_benchmark.sh  # Baseline B / Static Memory 测试入口
│   ├── run_dms_mini_benchmark.sh         # DMS mini-benchmark 测试入口
│   ├── run_all_mini_benchmark.sh         # 三类方法按顺序运行的统一入口
│   ├── setup_verify_dms_env.sh           # 环境构建与链路验证脚本
│   └── dms_metrics_report.py             # 指标汇总与可视化生成
│
├── run.py                       # 统一实验入口，注册 zero_shot / static_memory / dms
├── minimal_task_runner.py        # 单任务环境检查入口
└── android_world/                # AndroidWorld 官方环境、任务、ADB 控制与 evaluator
```

主要运行模式如下：

|模式|`--agent_name`|说明|
|-|-|-|
|Baseline A|`zero_shot`|无记忆 PA-Lite，仅使用当前状态决策|
|Baseline B|`static_memory`|静态历史轨迹注入 Planner，不做 DMS 动态更新|
|DMS|`dms`|启用记忆检索、回放、变异、反馈与动态剪枝|

实验输出统一写入：

```text
~/android_world_results/
├── results.csv                  # 每个 episode 的增量结果
├── summary.json                 # 本轮实验总体指标
├── run.log                      # 完整终端日志
├── memory/
│   ├── dms_memory.json          # DMS 记忆元数据
│   └── memory_trajectories/     # 分离保存的轨迹文件
└── dms_metrics_report/
    ├── overall_dashboard.svg    # 总体指标可视化
    ├── sr_evolution.svg
    ├── token_steps_trend.svg
    ├── memory_size_curve.svg
    └── per_task/                # 每个任务的单独指标与图表
```

## 3. 环境搭建

本项目采用“本地 Windows + WSL2 + Android Emulator + 远程 VLM 推理”的混合环境。AndroidWorld 与实验脚本在 WSL 中运行，Android 模拟器由本机提供，VLM 推理由远程 GPU 服务器提供，embedding 当前使用本地模型。

环境说明以简要、可复现为原则：正文只记录核心参数，完整构建与验证由脚本执行。

### 3.1 本地 Windows 环境

本地主要承担 Android Emulator 与文件管理功能。

当前环境：

|项目|配置|
|-|-|
|操作系统|Windows|
|工程文档目录|`E:\桌面\my-project`|
|Android SDK|`~/Android/Sdk`，由 WSL 侧访问|
|模拟器设备|`emulator-5554`|
|AVD 名称| `AndroidWorldAvd`|


### 3.2 WSL 与 Python 环境

实验代码在 WSL2 中运行。

当前环境：

|项目|配置|
|-|-|
|WSL 发行版|Ubuntu-22.04|
|代码目录|`~/projects/android_world`|
|Conda 环境|`android_world`|
|Python|3.11|
|主要入口|`run.py`、`minimal_task_runner.py`、`scripts/*.sh`|
|`DEVICE_SERIAL`|`emulator-5554`|
|`ANDROID_SDK_ROOT`|`~/Android/Sdk`|


环境构建与验证：

```bash
cd ~/projects/android_world
bash scripts/setup_verify_dms_env.sh
```

该脚本覆盖 Conda 环境、Python 包、ADB 设备、关键 app、intent 映射、本地 embedding、远程 VLM endpoint 和一个简单的 AndroidWorld task测试。


### 3.3 本地 Embedding 环境

DMS 需要对记忆的 `precondition` 和 `goal` 进行语义检索。当前使用本地下载的 `bge-base-en-v1.5`，避免远程 embedding 服务带来的连接不稳定。

当前配置：

|项目|配置|
|-|-|
|Embedding provider|`local`|
|模型路径|`~/projects/android_world/embedding_model`|
|运行设备|`cpu`|
|向量维度|768|


### 3.4 远程 VLM 推理环境

主 VLM 部署在远程 GPU 服务器上，为 vGPU-48GB，通过 vLLM 提供 OpenAI-compatible API。本地 WSL 通过 SSH tunnel 访问远程服务。

当前配置：

|项目|配置|
|-|-|
|VLM|`Qwen2.5-VL-7B-Instruct`|
|远程模型目录|`/root/autodl-tmp/model`|
|远程 vLLM 端口|`127.0.0.1:8000`|
|WSL 本地转发端口|`127.0.0.1:18000`|
|API 类型|OpenAI-compatible|
|`max_model_len`|`16384`|
|`gpu_memory_utilization`|`0.80`|

## 4. 自动化运行脚本

mini-benchmark 由 `configs/mini_benchmark.json` 固定管理。完整对比实验使用一个总脚本，按顺序运行 Baseline A、Baseline B 和 DMS，并保存每类方法的日志与结果。

```bash
cd ~/projects/android_world

EMBEDDING_PROVIDER=local \
EMBEDDING_MODEL_PATH="$HOME/projects/android_world/embedding_model" \
EMBEDDING_DEVICE=cpu \
bash scripts/run_all_mini_benchmark.sh
```

脚本执行顺序：

```text
Baseline A / PA-Lite
Baseline B / Static Memory
DMS
```

结果目录：

```text
~/android_world_results/all_mini_benchmark_<timestamp>/
├── baseline_a/
├── baseline_b/
├── dms/
└── run_all.log
```

每个子目录中均包含 `results.csv`、`summary.json` 和 `run.log`；DMS 子目录还包含 `memory/` 与 `dms_metrics_report/`。

## 5. 核心机制实现

本项目的核心实现围绕 Planner-Actor 框架和外部记忆系统展开。三类方法共用同一 AndroidWorld 环境、同一远程 VLM 和同一任务集合，差异只体现在记忆模块是否启用。

### 5.1 Planner-Actor 框架

Planner 负责根据任务目标、当前截图和当前 UI tree 生成当前页面下的子目标；Actor 只负责执行单步动作。普通动作执行后会结束当前 Actor turn，由 Planner 基于新页面重新判断下一步，避免模型长期沿用旧 sub-plan。

对应实现：

|模块|作用|
|-|-|
|`dms/planner.py`|生成当前 UI 状态下的子目标|
|`dms/actor.py`|解析 CodeAct / JSON 动作，并约束为单步 AndroidWorld action|
|`dms/androidworld_agent.py`|连接 Planner、Actor、Verifier、Memory 与 AndroidWorld 环境|

### 5.2 分层记忆构建

DMS 记忆单元按论文中的层次结构保存：

```text
m = (p, tau, s_meta)
p   = intent-level plan，包括 precondition 和 goal
tau = action-level trajectory，包括 observation-action 序列
s_meta = success、reuse_count、feedback counters 等元信息
```

对应实现：

|代码位置|说明|
|-|-|
|`dms/memory.py` 的 `PlanUnit`、`TrajectoryStep`、`MemoryUnit`|定义意图层、动作层和记忆元数据|
|`dms/androidworld_agent.py` 的 `finalize_task()`|任务结束后根据 AndroidWorld evaluator 的全局成功信号写入或更新记忆|
|`dms/memory.py` 的 `_serialize_dynamic_memory()`|将 memory metadata 与 trajectory 分离保存|

Baseline B 也使用相同的记忆构建格式，但只作为静态上下文提供给 Planner，不启用检索、回放、变异、替换或剪枝。

### 5.3 检索、回放与变异

DMS 在 Planner 生成子目标后，使用 `precondition` 和 `goal` 的双因子语义相似度检索历史经验。若命中低风险记忆，则按动作轨迹进行 replay；否则由 Actor 重新生成动作。命中记忆后仍保留 epsilon mutation，用于探索更短或更稳定的执行轨迹。

对应实现：

|代码位置|说明|
|-|-|
|`dms/memory.py` 的 `retrieve()`|双因子检索，并结合 risk gate 过滤高风险记忆|
|`dms/policy.py` 的 `DMSPolicy`|控制 replay / mutation 选择和动态风险阈值|
|`dms/androidworld_agent.py` 的 replay 逻辑|将命中轨迹按当前 UI 状态逐步对齐并执行|
|`dms/memory.py` 的 `replace_if_better()`|当 mutation 得到更短且验证成功的轨迹时替换旧记忆|

### 5.4 Survival Value 实现

Survival Value 是 DMS 判断记忆是否值得保留的核心指标。当前实现综合了复用次数、时间衰减、验证失败、任务完成率、无效点击率和无进展动作率。

代码位置：

|代码位置|说明|
|-|-|
|`dms/memory.py:156` 的 `MemoryUnit.survival_value()`|返回最终 survival value|
|`dms/memory.py:174` 的 `MemoryUnit.survival_components()`|拆解 utility、adaptive decay、reliability、environment feedback 等分量|
|`dms/memory.py:226` 的 `task_completion_rate()`|由 AndroidWorld evaluator 的成功/失败反馈计算任务完成率|
|`dms/memory.py` 的 `invalid_click_rate()`、`no_progress_rate()`|将无效点击和卡住行为转化为惩罚项|



### 5.5 Risk Gate 与全局反馈

为避免重复使用失败经验，DMS 在检索阶段加入风险门控。风险分数使用 Beta-Binomial 平滑，并引入全局失败率 `T_global` 作为先验；当整体环境失败率升高时，系统会降低可接受风险阈值。

对应实现：

|代码位置|说明|
|-|-|
|`dms/memory.py:206` 的 `risk_lower_bound()`|实现 Beta-Binomial 风险估计，并使用 `T_global` 作为先验均值|
|`dms/memory.py:605` 的 `global_failure_rate()`|统计全局任务失败率|
|`dms/policy.py:17` 的 `effective_risk_threshold()`|根据 `T_global` 动态调整 replay 风险阈值|
|`dms/androidworld_agent.py:301` 的 `finalize_task()`|任务结束后写入全局成功/失败反馈|

### 5.6 Pruning 实现

Pruning 用于控制记忆库规模，并淘汰低 survival value 的经验。当前实现遵循容量触发原则：只有当记忆数量达到 `C_min` / 当前容量后，才执行 pruning 检查；若 elbow cutoff 高于 population mean，则认为当前记忆整体质量较高，优先扩容到 `C_max`，而不是强行裁剪。

代码位置：

|代码位置|说明|
|-|-|
|`dms/memory.py:637` 的 `DMSMemoryStore.prune()`|实现容量触发、elbow cutoff、扩容与低价值记忆删除|
|`dms/memory.py:791` 的 `elbow_threshold()`|根据 survival value 曲线寻找低价值尾部阈值|
|`run.py:191`、`run.py:195`|定义 pruning 间隔和容量参数|
|`run.py:493` 附近的 `_record_task_result()`|每完成固定数量任务后触发 `agent.store.prune()`|
|`scripts/run_dms_mini_benchmark.sh`|设置当前实验使用的 `PRUNE_INTERVAL=5`、`CAPACITY_MIN=18`、`CAPACITY_MAX=72`|

Pruning 产生的指标会写入 `results.csv` 和 `summary.json`，包括 `memory_size`、`memory_survival_mean`、`memory_last_pruned_count`、`memory_current_capacity` 和 `memory_total_pruned`，后续由 `scripts/dms_metrics_report.py` 生成可视化。

## 6. 实验结果展示与分析

完整合并结果保存在：

```text
~/android_world_results/dms_final_merged_summary_20260816/
├── dms_merged_metrics.csv
└── dms_merged_metrics.json
```

### 6.1 Mini-benchmark 任务选取

本项目没有直接运行 AndroidWorld 全量任务，而是构建了一个规模更小、可重复执行的 mini-benchmark。该测试集保存在 `configs/mini_benchmark.json`，每个任务默认运行 5 个 trials，与论文中的多轮评估设置保持一致。

mini-benchmark 共包含 22 个任务模板，覆盖全部 20 个真实应用场景，包括 Clock、Camera、Contacts、Files、Settings、Markor、Joplin、OsmAnd、VLC、Simple Calendar、Simple SMS、Audio Recorder 等。任务选择以 easy 为主，少量使用 medium 任务补足应用覆盖；同时保留 Baseline A 曾成功过的任务，便于观察 DMS 是否能在可学习任务上体现经验复用收益。

### 6.2 三类方法总体对比

|方法|总体 SR|平均 Step|平均 token|
|-|-:|-:|-:|
|Baseline A / PA-Lite|14.55%|13.44|262,223.07|
|Baseline B / Static Memory|13.64%|14.01|342,979.27|
|DMS|37.82%|9.84|237,963.19|
<img width="1680" height="762" alt="image" src="https://github.com/user-attachments/assets/d800ceb7-401c-45b1-96a2-3b7673775c2f" />


DMS 在 mini-benchmark 上取得最高总体 SR，同时平均步数和 token 消耗低于两个 baseline。Baseline B 的整体表现没有超过 Baseline A，说明静态历史轨迹如果缺少筛选、检索和风险控制，可能会干扰 7B 模型的当前状态判断。

### 6.3 代表任务对比

|任务|Baseline A SR|Baseline B SR|DMS SR|
|-|-:|-:|-:|
|ClockStopWatchRunning|100.00%|20.00%|100.00%|
|SystemWifiTurnOn|0.00%|20.00%|100.00%|

ClockStopWatchRunning 是短链路任务，Baseline A 本身已经能稳定完成，DMS 主要保持稳定性。SystemWifiTurnOn 则更能体现记忆复用收益：Baseline A 完全失败，Baseline B 只成功 1/5，而 DMS 达到 5/5。

### 6.4 Trial 动态变化

<img width="1702" height="836" alt="image" src="https://github.com/user-attachments/assets/9f415380-e81b-4ed3-9c29-3456084e0e5f" />
<img width="1714" height="836" alt="image" src="https://github.com/user-attachments/assets/1f3c16b7-84f0-470d-b796-4427bff8c5d3" />


从 trial 变化看，DMS 在这两个代表任务上表现为更稳定的跨轮执行。相比之下，Baseline B 即使偶尔成功，也没有形成稳定收敛；这说明单纯追加历史上下文不足以保证经验复用，必须配合 DMS 的检索、风险门控和 replay 机制。

<img width="1934" height="1008" alt="image" src="https://github.com/user-attachments/assets/6d54745b-21c8-46ca-9b9d-26a488824765" />

在SystemWifiTurnOn任务中，DMS系统展现了显著的进化效果：第一轮:执行 6 steps, 消耗 183k tokens；第2-5轮：稳定于 4 steps，消耗 74k-83k tokens。

### 6.5 记忆库增长
<img width="2224" height="1190" alt="image" src="https://github.com/user-attachments/assets/00148c21-0e2d-46d2-bdb3-08468bbe30dc" />

前期：memory size 从 C_min=18 附近逐步增长，表示成功轨迹被写入记忆库。
中期：曲线可能出现平台期，表示部分任务失败或没有产生可保存记忆。
后期：继续增长并接近 C_max=72，展示 DMS 的动态积累。

### 6.6 初步分析

DMS 的提升主要来自两点：一是成功轨迹可以被结构化保存并在后续 trial 中复用；二是 survival value 与 risk gate 能抑制低质量记忆继续参与决策。对于 Clock、Audio Recorder、System Wi-Fi、SimpleDraw 和 SimpleSMS 这类关键控件相对稳定的任务，DMS 更容易形成可复用经验。

失败任务仍集中在复杂页面理解、文本/文件定位、地图/音乐/日历等多状态 app 中。这些失败更多反映 7B Actor 的 GUI grounding 能力不足，而不是单纯的记忆机制缺失。当 Actor 无法产生可用成功轨迹时，DMS 也无法凭空形成高质量记忆。

## 7. Gap 分析

原论文主要使用更强的多模态模型作为 GUI Agent 后端，而本复现使用 `Qwen2.5-VL-7B-Instruct`。在 AndroidWorld 任务中，7B 模型与论文设置相比存在明显能力差距，主要体现在动作约束遵循、页面状态理解和跨步骤稳定性上。

### 7.1 观察到的差异

第一，7B 模型更容易违反动作格式约束。早期实验中，Actor 偶尔会输出多个动作、重复失败动作，或将普通点击误解析为长按。这会导致 AndroidWorld 执行层无法稳定接收动作，或者在同一页面浪费大量步数。

第二，7B 模型对当前 UI 状态的锚定能力较弱。典型现象是 Clock 任务中已经进入 Clock 首页，但模型无法稳定切换到 Stopwatch；或者已经进入 Stopwatch 页面，却没有点击开始按钮。类似问题也出现在 Audio Recorder、Contacts、Markor 等任务中：模型能够完成前半段导航，但在需要填写字段、确认按钮或处理二级页面时停滞。

第三，记忆对 7B 模型既可能帮助，也可能干扰。Static Memory 如果直接暴露完整历史轨迹给 Actor，会削弱动作约束，使模型把历史页面当作当前事实。因此 Baseline B 一度出现效果不如 zero-shot 的情况。这与论文中大模型能更稳定区分“历史经验”和“当前观测”的前提不同。

第四，DMS 的 replay 在小模型上需要更强的状态对齐。原论文中的强模型更容易根据当前截图判断某条历史轨迹是否可复用；而 7B 模型在初始页面不一致时，可能错误沿用旧轨迹，导致“前一轮成功、后一轮失败”的波动。

### 7.2 Prompt 工程补偿

为弥补 7B 模型的约束遵循不足，本项目对 Planner 和 Actor prompt 做了更严格的工程化约束。

Actor 侧主要强化为：

|补偿策略|目的|
|-|-|
|每个 Actor turn 只允许 exactly one action-tool call|降低多动作输出和隐式追加动作|
|普通动作执行后立即结束 Actor turn|让 Planner 在新页面重新规划，避免沿用旧 sub-plan|
|禁止重复已失败动作|减少卡在同一页面反复点击|
|优先使用当前 UI index|降低坐标猜测带来的无效点击|
|当 index 与可见文字不匹配时拒绝动作|避免旧 index 在页面刷新后误点|
|禁止 Actor 在非目标 app 页面直接 `complete`|避免模型过早自报完成|

Planner 侧主要强化为：

|补偿策略|目的|
|-|-|
|先做当前状态锚定，再使用 memory|防止把历史轨迹当作当前页面事实|
|将 memory 标注为“流程候选”|明确 memory 只能提供参考流程|
|按任务名和目标 app 过滤跨任务 memory|减少无关经验污染|
|Static Memory 只注入 Planner，不注入 Actor|保持 Actor 与 Baseline A 完全一致的动作约束|

这些修改并不改变 DMS 的核心思想，但提高了小模型在 AndroidWorld 中的可执行性。

### 7.3 参数与机制补偿

除了 prompt 约束，本项目还引入了更保守的参数和运行机制。

|调整|当前设置或做法|原因|
|-|-|-|
|页面切换等待|`TRANSITION_PAUSE=1.5`|模拟器启动、弹窗和页面切换较慢，过早观察会频繁报：Could not get a11y tree, retrying.|
|DMS 记忆容量|`C_min=18`，`C_max=72`|mini-benchmark 规模较小，过大的容量会让 pruning 长期不触发|
|Risk gate|Beta-Binomial + `T_global`|在整体失败率升高时更谨慎地 replay 历史经验|
|Task-level fallback memory|成功任务即使没有 sub-plan memory，也保存完整任务轨迹|避免早期 memory size 长期为 0，保证 DMS 有可进化对象|

### 7.4 与原论文仍存在的 Gap

尽管上述补偿提高了稳定性，当前复现与原论文仍存在差距。

第一，模型规模差距无法完全通过 prompt 弥补。7B 模型在 UI 文字识别、控件语义理解和长程任务规划上仍弱于 72B 级模型，因此部分任务即使有记忆也会出现 replay 不完整、漏点确认按钮、字段输入停滞等问题。

第二，AndroidWorld 环境本身存在 app 初始状态、intent 映射、snapshot 缺失和模拟器速度差异。

第三，当前 DMS 的动态进化已经包含 survival value、risk gate、mutation replacement 和 pruning，但效果仍受基础 Actor 成功率制约。如果 Actor 无法产生可用成功轨迹，DMS 无法凭空形成高质量记忆。因此，本复现中 DMS 的收益更依赖“先得到少量成功轨迹，再通过检索和 replay 降低后续成本”的过程。
