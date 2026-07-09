---
name: oa-fault-feature-discovery
description: 光放大器故障数据的隐藏故障特征发现 Skill。用于从光放大器遥测、传感器时序、设备运行记录、告警前后窗口或现场导出表格中整理数据、执行 predictor/generator/morph 分析、挖掘候选前兆模式、审查物理可信度，并生成中文报告。Use when the user asks for optical amplifier fault feature discovery, hidden precursor mining, coupling feature analysis, morph-based counterfactual analysis, or report generation from optical-amplifier fault data.
compatibility: opencode
metadata:
  domain: optical-amplifier-time-series
  language: zh-CN
---

# 光放大器故障特征挖掘 Skill

## 概述（Overview）

本 Skill 的目标是：**从光放大器故障相关数据中挖掘隐藏故障特征和候选前兆模式**。

数据格式不限定为 parquet。用户可能提供原始遥测表、传感器时序、设备运行记录、告警前后数据或现场导出表格。若数据尚不是当前项目可直接处理的 parquet，需要先整理成入口脚本可接受的输入，再运行分析流程。

## 何时使用 / 何时不适用（When to Use / When NOT to Use）

### 何时使用（When to Use）

- 用户有光放大器故障相关数据，希望挖掘隐藏故障特征、候选前兆模式、风险升高路径或可复核工程线索。
- 数据来自光放大器遥测、传感器时序、设备运行记录、告警前后窗口或现场导出表格。
- 用户希望把非 parquet 的原始数据整理成当前项目可分析的输入，再运行 predictor/generator/morph 流程。
- 用户希望审查字段单位、物理约束、语义耦合关系、报告展示规则或低可信 morph 结果。

### 何时不适用（When NOT to Use）

- 数据对象不是光放大器，也无法映射到当前光放大器字段、单位或物理耦合规则。
- 用户只需要普通数据清洗、通用机器学习建模、代码修复或与故障前兆挖掘无关的分析。
- 用户要求直接给出最终故障原因、硬告警阈值或诊断规则，但没有真实故障样本、工程机理或独立数据复核。
- 数据缺少时间、设备标识、状态/故障标签或可解释字段，且用户不打算先补充或整理数据。
- 用户提供的是完全不同设备类型的数据，应改用通用故障预测/特征挖掘流程，而不是套用本 Skill 的光放大器物理规则。

## 铁律（The Iron Law）

**morph 生成的高风险方向变化只能作为候选前兆线索，不能当作真实故障原因、物理事实或最终告警规则。**

**Phase I-IV checklist 必须显式展示给用户，用于说明当前分析进度；不要把阶段状态只藏在 agent 内部。**

必须区分：

| 类型 | 含义 | 允许表达 |
| --- | --- | --- |
| 候选前兆 | predictor 在风险升高方向上反复出现的字段或耦合变化 | 可能提示、值得复核 |
| 语义耦合证据 | agent 基于真实字段语义生成、并经代码校验计算后的工程关系变化 | 与风险升高相关、需要真实样本验证 |
| 真实故障原因 | 经真实故障样本、工程机理和独立数据验证后的结论 | 充分验证后才允许 |

## 全局 MUST

- 必须在运行入口脚本、训练脚本或分析脚本前先检查当前 Python 环境：
  `python scripts/check_env.py`。
- `scripts/check_env.py` 会检查当前解释器、PATH、conda 环境清单、当前项目 venv，以及 Windows 各盘常见 conda/venv 目录；如果环境在 D 盘等非当前盘，应先用 `python scripts/check_env.py --list-candidates` 确认是否已被发现。
- 如果 `scripts/check_env.py` 提示其它 Python/conda/venv 环境可用，必须优先使用该解释器运行 skill；不要因为当前 Python 缺依赖就直接安装 torch 或其它依赖。
- 必须优先确认数据是否属于光放大器故障相关场景。
- 必须先检查数据形态，再决定是否需要转换成当前入口脚本可处理的 parquet。
- 必须优先运行统一入口脚本生成最终报告；本次分析结果目录默认只作为临时运行目录。
- 必须默认使用 `--selection-mode low_mid`，从低/中风险窗口寻找高风险方向候选前兆路径。
- 必须先看最终报告中的物理可信度，再解释候选模式；只有保留调试证据时才读取 `physical_validation.json`。
- 必须按场景读取 `references/` 下对应 md 文件，而不是只依赖主 `SKILL.md`。
- 必须在运行前、运行中或最终交付中展示 Phase I-IV checklist，并随进度更新状态。
- Phase III 默认必须由 agent 读取 `references/semantic-coupling/field-semantics.md`，基于真实字段生成 `semantic_coupling_candidates.json`，并通过 `--semantic-couplings` 传给入口脚本；用户不需要手动提供该参数。
- Phase III 不允许只解释原始字段贡献后跳过语义耦合候选；除非字段语义解析失败或可用字段不足以形成任何合法耦合，否则必须生成并传入 `semantic_coupling_candidates.json`。
- Phase III 语义耦合候选必须高召回生成：在字段白名单、合法 operation、字段数量和单位约束内，尽可能多地覆盖二元关系与多字段关系；不要在生成阶段按主观判断少量挑选。
- 原始字段和语义耦合候选只能来自 `assets/feature_cols.txt` / `scripts/optical-amplifier-fault-discovery/configs/feature_cols.txt` 中定义的字段白名单；数据集中额外出现但不在白名单内的字段不参与分析。

## 全局 MUST NOT

- 不要在未执行环境检查、未确认缺失依赖、未获得用户确认的情况下直接运行 `pip install`。
- 不要在 `scripts/check_env.py` 已发现可用 Python/conda/venv 环境时继续安装依赖；应切换到该解释器运行。
- 不要把 morph 结果写成真实故障原因。
- 不要默认重新训练模型，除非用户明确要求。
- 不要把 `models/default_run` 当作某次新数据的本次分析结果目录。
- 不要重写、重排或自由发挥最终 `oa_fault_feature_report.md`。
- 不要从 `hidden_features/` 读取或交付 Markdown 报告；该目录只保存内部证据。
- 不要省略 Phase I-IV checklist。
- 不要把 Step 1/2/3 展示成额外 checklist；用户可见 checklist 只保留 Phase I-IV 四项。
- 不要把单字段 range/center/volatility 变化写成语义耦合。
- 不要把业务语义词写进 `operation`，例如 `symmetry`、`lsr_symmetry`、`consistency`、`optical_path`、`thermal_control` 都不是推荐 operation；它们只能写入 `type`、`category`、`name` 或 `meaning`。
- 不要给 `operation` 自造同义词、缩写或业务别名；生成候选 JSON 前必须逐项对照 `references/semantic-coupling/field-semantics.md` 的 Operation Contract。
- 不要为了凑数量展示 weak 或 unknown 证据。
- 不要把 agent 生成但尚未经过合法性校验、数值计算和 strong-only 筛选的耦合候选写入最终报告。
- 不要在没有 `semantic_coupling_candidates.json` 的情况下 fallback 使用旧的预定义具体耦合字段；此时只保留原始字段贡献和其他已计算证据。
- 不要因为原始字段贡献已经足够解释，就省略默认语义耦合候选生成步骤。
- 不要把数据集中额外存在、但不在 `feature_cols.txt` 白名单内的字段纳入原始字段贡献或语义耦合候选；例如 `P`、`FP` 开头的额外字段默认不考虑。

## 进度展示硬约束（Progress Checklist）

运行前、运行中和最终交付时，必须显式展示 Phase I-IV checklist。固定四项如下：

```text
[ ] Phase I: 检查当前目录数据形态和字段
[ ] Phase II: 检查模型参数是否存在
[ ] Phase III: 运行候选故障特征挖掘
[ ] Phase IV: 证据审查与生成最终报告
```

状态标记规则：

- `[•]` 表示当前正在执行的阶段。
- `[√]` 表示已完成阶段。
- `[ ]` 表示未开始阶段。
- 最终交付时四项应全部为 `[√]`；如果失败，当前失败阶段保留 `[•]` 并说明失败原因。
- 禁止使用 Markdown 默认完成标记 `[x]` 或 `[X]`；完成状态只能写成 `[√]`。
- 不要额外展示 Step 1/2/3 checklist，避免和 Phase checklist 重复。

最终完成示例：

```text
[√] Phase I: 检查当前目录数据形态和字段
[√] Phase II: 检查模型参数是否存在
[√] Phase III: 运行候选故障特征挖掘
[√] Phase IV: 证据审查与生成最终报告
```

## Phase I：数据检查与预处理（Data Inspection & Preprocessing）

目标：判断当前光放大器故障数据是否具备隐藏特征挖掘条件，并整理成当前项目可分析的输入。

本阶段必须读取并遵守：

```text
references/data-pipeline/data-inspection-and-preprocessing.md
```

本阶段涉及字段列、时间列、设备列、标签列、窗口长度、步长、标签策略或窗口构造时，必须读取 skill 资产中的默认配置快照：

```text
assets/default.yaml
assets/feature_cols.txt
```

如遇字段单位、字段语义、单调背景字段或物理约束问题，同时读取：

```text
references/physical-constraints.md
```

## Phase II：模型参数检查与准备（Model Check & Preparation）

目标：先确认 `models/default_run` 下是否已有可用 predictor/generator 模型参数；有则正常加载，没有则基于当前项目代码和数据重新训练或准备所需模型参数。

### Step 1：检查默认模型目录下的模型参数

先检查默认模型目录是否已有当前流程可直接使用的模型参数：

```text
models/default_run/predictor/model.best.pt
models/default_run/generator/vae.pt
```

判断重点：

- 是否存在 predictor 参数。
- 是否存在 generator 参数。
- 参数对应的特征数量、窗口长度和字段顺序是否与当前数据配置一致。
- 如果参数存在且兼容，直接加载，不重新训练。

如需判断 predictor/generator 架构或训练配置，读取：

```text
references/model/model-selection.md
```

### Step 2：缺失模型参数时重新训练或准备模型

如果 `models/default_run` 下没有可用模型参数，或参数与当前字段配置不兼容，则基于当前项目代码、当前数据和项目配置重新训练或准备所需模型参数。

训练相关脚本和项目代码位于：

```text
scripts/optical-amplifier-fault-discovery
```

其中训练入口脚本在：

```text
scripts/optical-amplifier-fault-discovery/scripts/train_predictor.py
scripts/optical-amplifier-fault-discovery/scripts/train_generator.py
```

项目配置和字段清单在：

```text
scripts/optical-amplifier-fault-discovery/configs/default.yaml
scripts/optical-amplifier-fault-discovery/configs/feature_cols.txt
```

Phase I 数据预处理和窗口构造说明必须先读取 `assets/` 下的默认配置快照。

训练、修改项目配置或直接运行训练脚本时，还必须确认 `scripts/optical-amplifier-fault-discovery/configs/` 下的实际运行配置与 `assets/` 基准一致，或明确记录差异。

训练完成后，入口脚本默认加载的模型参数必须放在：

```text
models/default_run/predictor/model.best.pt
models/default_run/generator/vae.pt
```

可选保存：

```text
models/default_run/predictor/threshold_metrics.json
models/default_run/run_summary.json
```

本步骤必须做到：

- 只有缺少可用模型参数或参数不兼容时，才进入重新训练/准备模型流程。
- 训练前必须确认数据已经完成字段、单位、时间顺序和标签检查。
- 不要把旧模型参数强行用于字段顺序或窗口长度不同的数据。
- 训练产物必须整理到 `models/default_run` 的固定结构下，否则后续入口脚本无法自动加载。

## Phase III：候选故障特征挖掘（Candidate Fault Feature Discovery）

目标：生成本次分析结果目录，并从 morph 后的字段变化、语义耦合候选和物理可信度审查中形成候选前兆模式。

### Step 1：生成本次分析结果目录

入口脚本会调用内置项目脚本：

```text
scripts/optical-amplifier-fault-discovery/scripts/prepare_analysis_run.py
scripts/optical-amplifier-fault-discovery/scripts/discover_hidden_features.py
```

对新数据，必须先生成新的本次分析结果目录，再在该目录上做隐藏特征发现。
该目录是运行时临时目录，默认在最终 `oa_fault_feature_report.md` 写出后清理，不作为普通用户交付物。
只有用户明确要求调试、复盘证据或保留中间结果时，才添加：

```text
--keep-artifacts
```

默认参数：

```text
--selection-mode low_mid
--morph-method multi-gradient
--max-windows 1000
--n-hf 5
--morph-steps 50
--step-size 0.1
--max-latent-norm 5
--target-logit-delta 2
```

默认运行时，agent 必须先按 `references/semantic-coupling/field-semantics.md` 形成字段语义和候选 JSON，再通过入口参数传入；普通用户只需要用自然语言要求运行 skill，不需要手写该参数：

```text
--semantic-couplings <semantic_coupling_candidates.json>
```

候选 JSON 建议写入本次临时分析目录：

```text
<数据目录>\.oa_fault_feature_discovery_artifacts\semantic_coupling_candidates.json
```

如果字段语义无法可靠解析，不要虚构耦合；可以不传 `--semantic-couplings` 继续运行，但必须说明本轮只使用原始字段贡献和其他已计算证据。

本步骤的强制执行顺序：

1. 从 `assets/feature_cols.txt` 读取字段白名单，并只在当前数据列名中确认这些字段是否存在；不要把数据集中额外列加入候选字段池。
2. 读取 `references/semantic-coupling/field-semantics.md`。
3. 只为字段白名单中的可解释字段形成字段语义表。
4. 生成召回优先的 `semantic_coupling_candidates.json`，覆盖二元关系和多字段关系；注意这是文件名，JSON 顶层字段必须是 `candidates` 数组，不要写成 `semantic_coupling_candidates`。
5. 运行入口脚本时传入 `--semantic-couplings <semantic_coupling_candidates.json>`。
6. 如果未生成候选 JSON，必须在交付回复中明确说明“本轮未进行语义耦合计算”的原因。

允许跳过语义耦合候选的情况仅限：

- 字段名乱码或缺失，无法可靠识别物理量、单位或通道。
- 可用数值字段少于两个。
- 所有潜在候选都违反 `field-semantics.md` 或 `physical-constraints.md` 的合法性边界。

只有用户明确要求分析高风险窗口内部差异时，才使用 `--selection-mode high`。

### Step 2：形成候选前兆证据

候选证据主要来自：

- 原始字段贡献：单字段在 morph 前后的均值、范围、波动、趋势等变化。
- 字段语义解析：先把字段整理成结构化语义，包括通道、部件、物理量、单位、统计类型、系统域、角色和背景字段标记。
- 语义耦合候选：默认必须生成。agent 基于字段语义生成尽可能多的候选耦合关系，但必须使用真实字段、可计算 operation 和可解释字段单位关系；不要在候选 JSON 中填写 `unit_rule`。
- 候选 JSON 的 `meaning` 必须由 agent 写成中文公式化说明，例如 `公式: PAOUT光功率MAX - PAIN光功率MAX。用于观察...`；不要留空，也不要写 `Agent-generated semantic coupling candidate`。
- 候选 JSON 的 `operation` 只能使用 `difference`、`abs_difference`、`ratio`、`product`、`co_movement`、`direction_agreement`。例如 LSR1/LSR2 对称性候选应写 `type/category: lsr_symmetry`，但 `operation` 必须写 `difference` 或 `abs_difference`；方向一致性候选不要写 `consistency`，要写 `direction_agreement`。
- 写入候选 JSON 前，必须按 `references/semantic-coupling/field-semantics.md` 的 Operation Contract 检查每个候选：operation 是否在白名单、字段数量是否匹配、单位是否合法、业务语义是否没有误放入 operation。
- 候选生成采用高召回、后筛选策略：先尽可能覆盖合法耦合候选，运行时非法候选可被跳过，最终报告只展示 `risk_corr > 0.75`、物理可信度通过且符合 strong-only 规则的结果。
- 候选生成阶段以召回优先，不要只生成少量最明显的耦合；最终报告仍按 `risk_corr > 0.75` 的 strong-only 规则筛选。
- 语义耦合计算：候选耦合必须通过代码合法性校验，并在 morph 前后窗口上计算变化强度、`risk_corr` 和 `relevance_level`。
- 跨模式共性：多个 HF 中反复出现的强相关字段或语义耦合关系。

生成字段语义和语义耦合候选时读取：

```text
references/semantic-coupling/field-semantics.md
```

解释字段贡献、耦合边界和报告展示时读取：

```text
references/physical-constraints.md
references/report-guidelines.md
```

必须遵守：

- 单字段 range/center/volatility 变化不是语义耦合。
- dB 口径的输入/输出光功率关系只保留“输出/输入增益 = 输出光功率 - 输入光功率”。
- agent 生成的候选耦合必须来自真实字段，且不能绕过 `references/semantic-coupling/field-semantics.md` 和 `references/physical-constraints.md` 的合法性边界。
- agent 生成的候选耦合只有完成代码计算并达到 `references/report-guidelines.md` 的 strong-only 筛选规则时，才能进入最终报告正文。
- predictor 风险升高只代表模型方向，不代表真实故障原因。

## Phase IV：证据审查与输出（Evidence Review & Output）

目标：根据物理可信度和相关性强度，决定最终报告中可展示的候选前兆。

本阶段的低可信 morph 处理必须调用：

```text
references/morph-correction.md
```

该文件定义了低可信触发条件、三轮调参策略、终止条件和降级输出要求。不要只凭经验临时调参。

### Step 1：处理低可信或失败结果

如果运行失败、字段单位异常、`physical_validity = low`、单调字段违规高、风险提升很小或 morph 结果不可信，读取：

```text
references/physical-constraints.md
references/morph-correction.md
```

由于默认运行会清理内部证据目录，若需要低可信 morph 调参或复盘 `physical_validation.json`、字段贡献、语义耦合证据，必须使用 `--keep-artifacts` 重新运行入口脚本，再读取内部证据并执行 `references/morph-correction.md`。

低可信结果不要输出强结论。必须读取并执行 `references/morph-correction.md` 的三轮策略：

1. 保守化已有路径。
2. 进一步降低目标风险提升和 latent 位移。
3. 扩大候选窗口覆盖，但继续保持保守 morph。

如果三轮后 morph 信号仍不满足要求，必须停止自动调参，降级输出“未发现稳定候选前兆”，并说明风险提升不足、物理可信度不足、单调背景字段违规或 strong 证据不足等原因。

如果任一轮调参后 morph 信号满足要求，不能直接输出结论；必须回到 Phase III 的 `Step 2：形成候选前兆证据`，重新计算原始字段贡献、语义耦合候选和跨模式共性，再按 strong-only 规则进入证据分级与最终报告。

### Step 2：证据分级

| 等级 | 当前规则 | 展示方式 |
| --- | --- | --- |
| strong | `risk_corr > 0.75` | 可进入最终报告正文 |
| weak | `risk_corr <= 0.75` | 只保留在内部证据 |
| unknown | 证据不足或不可计算 | 只保留在内部证据 |

语义耦合关系以 `risk_corr > 0.75` 作为主门槛；`abs_strength` 只做较宽松的二次筛选，避免把相关性很强但变化幅度不是最大的耦合过早过滤。

如需判断最终报告应展示哪些 HF、如何命名标题、如何处理 weak/unknown，读取：

```text
references/report-guidelines.md
```

## 红旗自查（Red Flag Checklist）

- 物理可信度为 `low` 时，不输出强结论，只能降级说明。
- 模式标题使用业务语义名称，`HF_001` 只作为内部追溯 ID。
- 单字段 range/center/volatility 变化只能作为原始字段贡献，不写成核心耦合关系。
- agent 生成但未通过合法性校验、数值计算和 strong-only 筛选的语义耦合，不写入最终报告正文。
- weak 或 unknown 证据只保留在内部文件，不进入主要发现。
- predictor 风险升高只表示模型方向，不写成真实故障原因。
- 普通交付回复只给报告路径、可信度和重点模式，不暴露完整 JSON/CSV 或内部调试路径。

## 输出格式（Output Format）

最终报告必须按照以下文件定义的结构和展示规则生成：

```text
references/report-guidelines.md
```

主报告由工具生成，agent 不要重写或重排 `oa_fault_feature_report.md`。
`hidden_features/` 只作为内部证据目录，不作为 Markdown 报告入口。

普通交付回复只需要包含最终报告路径、物理可信度和 1-3 个重点候选模式：

```text
<parquet所在目录>\oa_fault_feature_report.md
```

输出约束：

- 最终交付回复必须先展示“进度展示硬约束”中定义的 Phase I-IV checklist。
- 不要粘贴完整报告正文、完整 JSON 或完整 CSV。
- 不要在普通交付回复中展示内部证据路径。
- 不要把 low/weak/unknown 写成明确发现。
- 不要使用 `HF_001` 作为主标题；它只能作为内部追溯 ID。
- 不要写“已证明”“导致故障”“可直接告警”。
