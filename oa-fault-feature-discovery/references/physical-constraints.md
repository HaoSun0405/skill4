# 物理约束（Physical Constraints）

## 概述（Overview）

morph 生成的是高风险方向的反事实窗口，不天然满足物理规律。所有候选前兆解释必须先经过物理可信度审查，再进入报告。

当前项目会在 morph 后对强单调背景字段执行 non-decreasing 物理投影。投影后重新计算 predictor 风险、字段贡献和耦合贡献，报告应以投影后的证据为准。

## 何时使用 / 何时不使用（When to Use / When NOT to Use）

### 何时使用

- 需要检查字段单位、字段语义或物理约束。
- 需要解释 `physical_validation.json`。
- 需要判断某个字段能否作为核心前兆。
- 需要判断某个耦合关系是否有物理含义。

### 何时不使用

- 只是在确认文件路径或入口命令。
- 用户要求的是模型训练架构选择，应使用 `references/model/model-selection.md`。

## Step 1：检查字段单位

当前项目使用以下单位规则：

- 温度：`0.1℃`
- TEC 电压、3V3 电压：`0.01V`
- LSR1/2 驱动电流 MAX/MIN：`0.1mA`
- LSR1/2 制冷电流 MAX/MIN：`0.1mA`
- LSR1/2 背光电流 MAX/MIN：`0.001mA`
- 光功率/增益：`0.01dB`

语义和单位必须匹配。电压字段不能使用 mA，温度字段不能使用 mA。历史旧字段单位错误不需要兼容。

## Step 2：识别字段类型

### state_signal

可作为核心指标：

- 温度、壳温、管芯温度
- 输入/输出光功率
- 增益
- 驱动电流
- 制冷电流
- 背光电流
- TEC 电压
- 3V3/电源电压

### monotonic_background

可作为背景风险信息，但不要写成直接故障原因：

- 累计运行时间
- 累计上电次数
- 累计超温运行时间

约束：morph 后不应下降。若下降，只能解释为生成可信度问题。

### resettable_runtime_background

最近一次上电后的运行时间可能因重新上电而清零，不按强单调字段处理。

### paired_component_signal

LSR1/LSR2 是成对组件字段。报告可以说明证据集中在 LSR1 或 LSR2，但不要直接断言某一侧硬件故障，除非真实样本验证支持。

## Step 3：检查语义耦合

语义耦合必须有可解释的物理含义，且必须来自 agent 基于真实字段语义生成的候选 JSON，并经过代码合法性校验和数值计算。

- LSR1/LSR2 对称性
- 光路增益：输出光功率 - 输入光功率
- 光路比值：仅在非 dB 线性读数时使用
- 增益一致性：测量增益与光路增益的关系
- 电功率：电流 * 电压，按单位换算后计算
- 电光效率：光输出相对电功率
- 热控负载：TEC/制冷量与温度状态关系

单字段 range/center/volatility 不是语义耦合。

agent 生成语义耦合候选时，必须先通过 `references/semantic-coupling/field-semantics.md` 中的字段真实性、operation、单位自动校验和背景字段校验。未通过校验或未完成数值计算的候选，不具备物理证据资格。

## Step 4：解释物理可信度

读取 `physical_validation.json`：

- `high`：可正常解释。
- `medium`：可解释，但需要保留可信度提示。
- `low`：不要输出强结论，应按 `references/morph-correction.md` 调参重跑或降级说明。

若存在单调字段下降、过大位移或风险没有正向提升，应在报告中说明相应限制。
