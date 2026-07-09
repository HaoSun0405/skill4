# Morph 纠正（Morph Correction）

## 概述（Overview）

本文件用于处理 morph 信号不可信、风险提升不足或生成方向违反物理约束的情况。低可信结果不能强行解释为故障前兆，必须按本文件调参、判断终止条件，并在必要时降级输出。

## 何时使用 / 何时不使用（When to Use / When NOT to Use）

### 何时使用

- `physical_validity = low`
- 单调背景字段下降比例高
- `risk_delta` 或 `logit_delta` 很小
- 核心证据主要来自累计运行时间、累计上电次数、累计超温运行时间等背景字段
- 生成方向明显违反光放大器物理约束
- 风险提升集中在少数异常字段，而不是稳定字段组或语义耦合关系

### 何时不使用

- 物理可信度为 high，且 strong 证据稳定。
- 用户只是要求查看已生成报告，不要求修正低可信结果。

## Step 1：保守化已有路径

目标：降低生成变化幅度，优先获得物理可信的候选方向。

使用：

```powershell
--selection-mode low_mid --morph-method multi-gradient --max-windows 300 --n-hf 5 --morph-steps 50 --step-size 0.1 --max-latent-norm 5 --target-logit-delta 2
```

如果原先已经是这组参数，但仍低可信，继续 Step 2。

## Step 2：进一步降低目标和位移

目标：检查是否存在更弱但更可信的风险方向。

使用：

```powershell
--selection-mode low_mid --morph-method conservative-gradient --max-windows 300 --n-hf 5 --morph-steps 40 --step-size 0.05 --max-latent-norm 3 --target-logit-delta 1.5
```

通过条件：

- `physical_validity` 不为 `low`
- 单调背景字段违规显著降低
- `risk_delta` 或 `logit_delta` 仍有正向提升
- strong 字段或语义耦合证据不是全部来自背景字段

不通过则继续 Step 3。

## Step 3：扩大候选覆盖但保持保守 morph

目标：确认信号不足是不是因为候选窗口太少或聚类过窄。

使用：

```powershell
--selection-mode low_mid --morph-method conservative-gradient --max-windows 500 --n-hf 8 --morph-steps 40 --step-size 0.05 --max-latent-norm 3 --target-logit-delta 1.5
```

注意：

- 只扩大 `max-windows` 和 `n-hf`，不要同时放大 `step-size` 或 `max-latent-norm`。
- `n-hf` 是候选聚类数，不等于最终报告必须展示的模式数。
- 最终仍只展示物理可信且相关性足够的模式。

## 信号达标后的回流要求

任一轮调参后，如果 morph 信号满足以下条件，必须停止继续调参，并回到候选证据形成流程：

- `physical_validity` 不为 `low`
- `risk_delta` 或 `logit_delta` 有稳定正向提升
- strong 字段或语义耦合证据不是全部来自背景字段
- 没有明显违反光放大器物理约束的生成方向

回流后必须重新形成候选前兆证据：

- 重新计算原始字段贡献。
- 重新生成并计算语义耦合候选。
- 根据字段工程含义补充必要的候选耦合关系。
- 重新检查跨模式共性信号。
- 按 `report-guidelines.md` 的 strong-only 规则筛选最终报告正文内容。

调参成功不等于已经得到最终结论；只有重新形成证据并通过证据分级后，才能生成或更新最终报告。

## 终止条件

完成三轮后，如果仍出现以下任一情况，必须停止自动调参：

- `physical_validity = low`
- `risk_delta` 或 `logit_delta` 持续很小
- strong 证据主要来自背景字段或明显异常字段
- 没有稳定的原始字段或语义耦合证据
- 生成方向与光放大器物理约束冲突

停止后不要继续硬调参数，不要为了得到报告而放大步长、位移或目标风险。

## 降级输出要求

如果三轮后 morph 信号仍不满足要求，最终输出必须降级为“未发现稳定候选前兆”，并说明原因。

必须包含：

- 已尝试的参数轮次
- 当前物理可信度
- 风险提升是否不足
- 是否存在单调背景字段违规
- 是否缺少稳定字段或语义耦合证据
- 下一步建议：补充真实故障样本、检查字段单位/标签、重新训练模型或调整数据窗口

禁止写成：

- “发现明确故障前兆”
- “某字段导致故障”
- “可直接作为告警规则”
- “发现多个 HF 但全部 unknown”
