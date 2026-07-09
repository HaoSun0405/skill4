# 模型选择（Model Selection）

## 概述（Overview）

本文件只在需要重新训练、替换或比较 predictor/generator 时使用。普通隐藏特征挖掘任务优先加载 `models/default_run` 中已有模型参数，不默认重训。

训练代码位于：

```text
scripts/optical-amplifier-fault-discovery
```

训练完成后的模型参数需要整理到：

```text
models/default_run/predictor/model.best.pt
models/default_run/generator/vae.pt
```

## 何时使用 / 何时不使用（When to Use / When NOT to Use）

### 何时使用

- `models/default_run` 下缺少 predictor 或 generator 参数。
- 当前模型参数与字段数量、字段顺序或窗口长度不兼容。
- 用户明确要求重新训练、替换模型或比较模型架构。
- 服务器训练完成后，需要判断新模型是否能替换默认模型。

### 何时不使用

- 只是对新数据运行已有模型做隐藏特征挖掘。
- 当前模型参数存在且与配置兼容。
- 用户没有要求重新训练或模型比较。

## Step 1：选择风险模型（Predictor）

### TCN

默认首选。适合固定长度、多变量工业时序窗口。

```yaml
predictor:
  model: tcn
  hidden_channels: 96
  num_blocks: 5
  kernel_size: 3
  dropout: 0.15
  learning_rate: 0.001
  batch_size: 256
  max_epochs: 80
  patience: 12
```

### GRU/LSTM

小数据、短窗口或希望快速验证时可作为备选。

```yaml
predictor:
  model: gru
  hidden_size: 64
  num_layers: 2
  dropout: 0.1
  learning_rate: 0.001
```

### Transformer Encoder

数据量足够、窗口较长且存在复杂长程依赖时再考虑。不建议作为默认首选。

## Step 2：选择生成模型（Generator）

### Window VAE

默认首选。适合窗口重构、latent morph 和反事实窗口生成。

```yaml
generator:
  model: window_vae
  latent_dim: 32
  hidden_channels: 64
  beta_kl: 0.001
  learning_rate: 0.0005
  gradient_clip_norm: 1.0
```

### Conditional VAE

只有在 low/high risk 或故障类型标签可靠时使用。

### Denoising Autoencoder

VAE 训练不稳定、KL 异常或重构质量差时作为稳定 baseline。

## Step 3：训练前检查

- 字段顺序必须与默认字段清单一致；Phase I 先读取 `assets/feature_cols.txt`，训练或修改项目配置时还必须确认 `scripts/optical-amplifier-fault-discovery/configs/feature_cols.txt` 与 assets 基准一致，或明确记录差异。
- `window_len` 必须与后续 morph 和模型加载配置一致。
- 训练集、验证集应按设备维度划分，避免同一设备窗口泄漏。
- 标签含义必须明确，不要把故障后信息混入故障前兆学习。
- 训练完成后必须更新或确认 `run_summary.json` 中的字段数量和窗口长度。
