# 字段语义与语义耦合候选（Field Semantics & Semantic Coupling）

本文件用于 Phase III 中的语义耦合候选生成。目标不是预定义具体耦合字段，而是先把原始字段解析成结构化语义，再由 agent 基于真实字段语义生成尽可能多的候选耦合，最后交由确定性代码校验、计算和 strong-only 筛选。

普通用户不需要手动编写或传入候选 JSON。使用本 skill 时，agent 默认负责生成 `semantic_coupling_candidates.json`，并把它传给入口脚本的 `--semantic-couplings` 参数。

## 1. 核心原则

- 候选生成阶段以召回优先。只要字段真实存在、operation 可计算、单位关系可解释，就应尽可能生成候选；不要因为候选看起来不一定 strong 就提前丢弃。
- 不要在候选生成阶段做“少量精选”。应先生成尽可能多的合法候选，再由代码计算 `risk_corr`、物理可信度和 strong-only 规则筛选最终有用结果。
- 多生成候选不等于多进报告。最终是否展示只由代码计算后的 `risk_corr > 0.75`、物理可信度和 strong-only 规则决定。
- 默认必须生成候选 JSON；不要只分析原始字段后跳过语义耦合。只有字段语义无法可靠解析、可用数值字段少于两个，或所有候选都违反合法性边界时，才允许不生成。
- 不要写死具体耦合字段，例如不要把 `LSR1管芯温度MAX - LSR2管芯温度MAX` 作为固定清单。
- 可以固定字段语义模板、候选耦合模板、合法性边界和可计算 operation。
- agent 只能基于真实存在的字段生成候选耦合，不得虚构字段、单位或工程含义。
- agent 只能基于 `feature_cols.txt` 字段白名单中的字段生成候选耦合；数据集中额外出现但不在白名单内的字段不得进入字段语义表、候选 JSON 或最终报告。
- 如果数据中存在 `P`、`FP` 等开头的额外字段，除非它们已经明确写入 `feature_cols.txt`，否则默认忽略。
- agent 生成的候选耦合只是待计算候选；未经过代码校验、数值计算和 strong-only 筛选前，不得进入最终报告正文。
- 单字段 range、center、volatility 是稳定性/波动证据，不是两个或更多字段之间的语义耦合。
- `operation` 只能表示代码支持的计算方法，不表示业务关系类型。`symmetry`、`lsr_symmetry`、`consistency`、`optical_path`、`thermal_control` 等都不是推荐 operation；这些词只能写入 `type`、`category`、`name` 或 `meaning`。如果要表达变化方向一致性，正式写法是 `direction_agreement`。

## 2. 字段语义模板

每个原始字段应先整理成如下结构。字段名必须来自 `feature_cols.txt` 字段白名单，并与当前分析使用的字段名完全一致。

```json
{
  "field_id": "F001",
  "raw_name": "LSR1TEC电压MAX(0.01V)",
  "display_name": "LSR1 TEC电压MAX",
  "base_name": "LSR1TEC电压",
  "channel": "LSR1",
  "component": "TEC",
  "physical_quantity": "电压",
  "quantity_type": "voltage",
  "stat": "MAX",
  "unit_raw": "0.01V",
  "unit": "V",
  "scale": 0.01,
  "system_domain": "thermal_control",
  "role": "actuator",
  "is_background": false,
  "is_monotonic_counter": false,
  "notes": "LSR1 TEC控制电压，反映温控执行量。"
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `field_id` | 稳定编号，用于候选耦合引用 |
| `raw_name` | 原始字段名，必须真实存在 |
| `display_name` | 报告展示名，去掉单位但保留 MAX/MIN |
| `base_name` | 去掉 MAX/MIN 和单位后的主体名称 |
| `channel` | 通道或对象，例如 LSR1、LSR2、PA、BA、EDFA、board、module、unknown |
| `component` | 部件，例如 TEC、drive、laser_die、optical_power、power_supply、unknown |
| `physical_quantity` | 中文物理量 |
| `quantity_type` | 标准类型，例如 temperature、current、voltage、power、gain、optical_power、time、count、unknown |
| `stat` | MAX、MIN、MEAN、RAW、unknown |
| `unit_raw` | 原始单位后缀，例如 0.01V |
| `unit` | 标准单位，例如 V、A、C、dB、min、count、raw |
| `scale` | 单位倍率，例如 0.01 |
| `system_domain` | thermal_control、optical_path、power_supply、runtime_background、unknown |
| `role` | sensor、actuator、response、background、unknown |
| `is_background` | 是否背景/工况字段 |
| `is_monotonic_counter` | 是否单调累计字段 |
| `notes` | 简短工程含义 |

## 3. 候选耦合模板

agent 生成候选耦合时必须写出一个完整 JSON 文件。文件名可以是 `semantic_coupling_candidates.json`，但 JSON 顶层字段必须使用 `candidates` 数组，不要使用 `semantic_coupling_candidates` 作为顶层字段名。候选应尽可能多，但必须可校验、可计算。`fields` 不限定为两个字段；字段数量由 `operation` 决定。

高召回生成要求：

- 对同单位、同统计类型、不同通道或不同位置的字段，尽量生成 `difference` 和/或 `abs_difference` 候选。
- 对 PA、BA 等输入/输出 dB 光功率字段，尽量生成输出减输入的 `difference` 候选。
- 对同一通道或同一系统域中可解释为执行量与响应量的字段组，尽量生成 `co_movement` 候选。
- 对跨通道同类字段组，尽量生成 `direction_agreement` 候选。
- 对电压字段与电流字段，只有工程含义可解释时生成 `product` 候选。
- 不要因为候选数量多而提前删减；非法候选运行时会被跳过，最终报告只展示 strong 结果。

生成候选时应尽量覆盖但不局限于以下方向：

- 同单位字段的 `difference` / `abs_difference`。
- dB 输入输出类字段的 `difference`，例如 `PAOUT光功率MAX(0.01dB) - PAIN光功率MAX(0.01dB)`。
- 有明确单位组合含义的 `product`，例如电压与电流。
- 同通道、同系统域或同部件字段的 `co_movement`。
- 跨通道同类字段的差值、绝对差或方向一致性。

候选召回清单：

- 同一链路或同一放大级的输入/输出光功率差值。
- 同一物理量、同一统计类型、不同通道之间的差值或绝对差。
- 同一通道内温度、电流、电压等执行量与响应量的协同变化。
- TEC 电压/电流、驱动电流/相关电压等可形成电功率含义的候选。
- 同系统域字段组的 `co_movement`，例如温控字段组、光路字段组、电源字段组。
- 跨通道同类字段组的 `direction_agreement`，用于观察多通道变化方向是否一致。

如果某一类候选因为字段缺失无法生成，跳过该类即可；不要因此跳过其它可生成候选。

同一 base_name 的 MAX/MIN `range` 和 `center` 只能作为单字段稳定性/波动证据，不得写入 `semantic_coupling_candidates.json`。

业务语义与 operation 的对应关系：

| 业务语义 | JSON 写法 |
| --- | --- |
| LSR1/LSR2 对称性、不对称性 | `type/category` 写 `lsr_symmetry`，`operation` 写 `difference` 或 `abs_difference` |
| 输入/输出光功率关系 | `type/category` 写 `optical_path`，`operation` 写 `difference` |
| 电压与电流形成电功率候选 | `type/category` 写 `electric_power`，`operation` 写 `product` |
| 多字段协同变化 | `type/category` 写业务域名称，`operation` 写 `co_movement` |
| 多字段方向一致性 | `type/category` 写业务域名称，`operation` 写 `direction_agreement` |

```json
{
  "candidates": [
    {
      "coupling_id": "C001",
      "name": "PA输出输入光功率差MAX",
      "fields": [
        {
          "field_id": "F001",
          "raw_name": "PAOUT光功率MAX(0.01dB)"
        },
        {
          "field_id": "F002",
          "raw_name": "PAIN光功率MAX(0.01dB)"
        }
      ],
      "operation": "difference",
      "type": "optical_path",
      "expected_output_unit": "dB",
      "meaning": "公式: PAOUT光功率MAX - PAIN光功率MAX。用于观察 PA 链路输出光功率与输入光功率的差值变化是否随风险升高而稳定增强。",
      "why_candidate": "两个字段同属 PA 光路，单位均为 dB，输出减输入可表示该链路的相对增益或损耗变化。"
    }
  ]
}
```

每个候选耦合必须包含：

- `coupling_id`
- `name`
- `fields`
- `operation`
- `meaning`
- `why_candidate`

`meaning` 必须使用中文，并且必须包含可读公式。公式使用去单位后的字段展示名，不要写内部英文变量名或下划线拼接名。示例：

- `公式: PAOUT光功率MAX - PAIN光功率MAX。用于观察 PA 链路输出光功率与输入光功率的差值变化。`
- `公式: |LSR1管芯温度MAX - LSR2管芯温度MAX|。用于观察 LSR1/LSR2 管芯温度不对称程度。`
- `公式: LSR1TEC电压MAX * LSR1驱动电流MAX。用于观察电功率类候选变化。`

字段数量规则：

- `difference`、`abs_difference`、`ratio`：当前只接受两个字段。
- `product`：当前只接受两个字段，且必须是一个电压字段和一个电流字段。
- `co_movement`、`direction_agreement`：接受两个或更多字段，适合表达同通道、同系统域、同部件或跨通道字段组的协同变化。

## 4. 可计算 Operation

operation 是代码支持的计算能力，不是必须生成的业务规则。agent 可以选择合适 operation，但不得 invent 未实现 operation。

合法 operation 只有以下六个：`difference`、`abs_difference`、`ratio`、`product`、`co_movement`、`direction_agreement`。如果想表达“对称性”，不要写 `"operation": "symmetry"`，应写 `"operation": "difference"` 或 `"operation": "abs_difference"`，并把 `type/category` 写成 `lsr_symmetry`。如果想表达“一致性”，不要写 `"operation": "consistency"`，应写 `"operation": "direction_agreement"`。

### Operation Contract

生成 `semantic_coupling_candidates.json` 前，必须逐个候选检查以下合同：

1. `operation` 必须逐字等于合法 operation 之一，不得使用同义词、缩写、业务词或临时命名。
2. `operation` 只决定计算方式；业务关系类型只能写入 `type`、`category`、`name`、`meaning` 或 `why_candidate`。
3. 字段数量必须与 operation 匹配：
   - `difference`、`abs_difference`、`ratio`：必须且只能有 2 个字段。
   - `product`：必须且只能有 2 个字段，且一个是电压字段、一个是电流字段。
   - `co_movement`、`direction_agreement`：必须有 2 个或更多字段。
4. 单位必须与 operation 匹配：
   - `difference`、`abs_difference`：字段单位必须一致；dB 字段允许差值。
   - `ratio`：不得用于 dB 字段，且当前只接受同单位字段。
   - `product`：只接受 V 和 A 形成电功率候选。
   - `co_movement`、`direction_agreement`：可以跨单位，但必须在 `meaning` 或 `why_candidate` 说明协同或方向一致性的工程理由。
5. 如果无法把业务想法映射到合法 operation，不要生成该候选；不要发明新 operation。

禁止写法示例：

```json
{"operation": "symmetry"}
{"operation": "consistency"}
{"operation": "thermal_control"}
{"operation": "optical_path"}
```

正确写法示例：

```json
{
  "type": "lsr_symmetry",
  "operation": "abs_difference",
  "meaning": "LSR1/LSR2 同类温度字段的不对称程度。"
}
```

| operation | 含义 | 基本约束 |
| --- | --- | --- |
| `difference` | 字段 A - 字段 B | 只接受两个字段，通常要求同单位或 dB 差值 |
| `abs_difference` | `abs(A - B)` | 只接受两个字段，通常要求同单位 |
| `ratio` | 字段 A / 字段 B | 只接受两个字段，分母必须非零且单位含义可解释 |
| `product` | 字段 A * 字段 B | 当前只接受一个电压字段和一个电流字段 |
| `co_movement` | 多字段变化协同强度 | 接受两个或更多字段，必须说明字段为何可能协同 |
| `direction_agreement` | 多字段变化方向一致性 | 接受两个或更多字段，必须说明字段方向关系 |

`range` 和 `center` 不属于语义耦合 operation。它们可以在原始字段贡献或稳定性证据中使用，但不得作为 agent 生成的候选耦合写入 JSON。

如果候选使用当前代码尚不支持的 operation，必须先实现确定性计算逻辑；否则该候选只能作为待实现建议，不能进入最终报告。运行时非法候选会被跳过并汇总警告，不应阻塞整轮隐藏特征挖掘；但不能把被跳过的候选写入最终报告。

## 5. 单位合法性自动校验

候选 JSON 不需要填写 `unit_rule`。agent 只提供真实字段、`operation`、中文名称和工程含义；代码会根据 `operation` 与字段单位自动校验是否合法。

自动校验规则：

| operation | 自动校验 |
| --- | --- |
| `difference` / `abs_difference` | 字段单位必须一致；dB 字段允许做差值，按输入/输出增益或相对变化理解 |
| `ratio` | 不允许用于 dB 字段；当前只接受同单位字段比例，分母必须可稳定计算 |
| `product` | 当前只接受一个 V 字段和一个 A 字段，用于电功率候选 |
| `co_movement` | 不要求同单位，但必须在 `meaning` / `why_candidate` 中说明字段为何可能协同 |
| `direction_agreement` | 不要求同单位，只表达变化方向一致性 |

`range` 和 `center` 不属于语义耦合 operation，不能写入候选 JSON。

## 6. 合法性校验

候选耦合进入计算前必须通过以下校验：

- 所有 `raw_name` 必须存在于当前字段清单。
- `field_id` 与 `raw_name` 必须一致。
- `raw_name` 必须属于 `feature_cols.txt` 字段白名单；不在白名单内的额外数据列即使真实存在，也不得作为候选耦合字段。
- `operation` 必须在可计算 operation 集合内。
- 字段单位必须通过代码的自动合法性校验。
- dB 字段优先做差值，不要做普通线性乘除。
- 不同单位字段不得直接做 `difference` 或 `abs_difference`。
- 背景字段、单调累计字段不得作为核心耦合主导项。
- `range` 和 `center` 不得出现在语义耦合候选 JSON 中。
- 候选必须能在 morph 前窗口和 morph 后窗口上稳定计算。

## 7. agent 生成边界

agent 可以做：

- 根据字段语义生成尽可能多的候选耦合。
- 为候选耦合提供中文业务命名和带公式的工程含义。
- 在 strong 结果中解释为什么该关系值得复核。

agent 不可以做：

- 直接把未计算候选写入最终报告。
- 为了增加候选数量虚构字段或单位。
- 把 weak、unknown 或无法计算的候选写成发现。
- 绕过代码校验，自行声明某个耦合 strong。

## 8. 进入最终报告的条件

语义耦合候选必须同时满足：

- 已通过合法性校验。
- 已完成 morph 前后数值计算。
- `risk_corr` 达到 `references/report-guidelines.md` 的 strong 阈值。
- `physical_validity` 不是 low。
- 不是背景字段或单调累计字段主导。

不满足条件的候选只可作为内部证据或待实现/待验证项，不进入最终报告正文。
