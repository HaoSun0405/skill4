# Phase I：数据检查与预处理（Data Inspection & Preprocessing）

目标：先判断当前光放大器数据是否具备隐藏故障特征挖掘的基本条件，再把原始数据整理成可用于模型参数检查、候选特征挖掘和证据审查的稳定数据。

本阶段必须结合当前项目配置读取：

```text
assets/default.yaml
assets/feature_cols.txt
```

其中 `default.yaml` 定义时间列、设备列、标签列、窗口长度、步长、标签策略和预处理策略；`feature_cols.txt` 定义当前模型期望的光放大器特征列顺序。

`assets/` 下的文件是 Phase I 数据预处理必须读取的默认配置快照。直接运行项目脚本、重新训练或修改配置时，还必须确认实际运行配置与该快照一致，或明确记录差异：

```text
scripts/optical-amplifier-fault-discovery/configs/default.yaml
scripts/optical-amplifier-fault-discovery/configs/feature_cols.txt
```

## Step 1：数据检查（Data Inspection）

### 1.1 识别数据形态

优先判断数据属于哪一类：

| 数据形态 | 例子 | 后续方向 |
| --- | --- | --- |
| 原始遥测时序 | 同一 `sn` 下光功率、增益、温度、电流、电压随时间采样 | 需要按设备和时间排序，再窗口化 |
| 设备事件/告警日志 | 告警、状态变化、维修记录、故障时间点 | 需要和遥测时间对齐，避免未来信息泄漏 |
| 现场导出表格 | csv/xlsx/parquet 中的光放大器运行记录 | 需要字段名、单位、缺失、重复和排序检查 |
| 已有窗口/统计特征表 | 已经按窗口计算好的 MAX/MIN/均值/斜率等特征 | 需要确认是否仍符合当前 `assets/feature_cols.txt` 和模型窗口配置 |

### 1.2 检查关键字段

必须尽量确认：

| 字段类型 | 光放大器场景常见字段 |
| --- | --- |
| 时间列 | `DateTime`、`timestamp`、`time`、采样时间 |
| 设备列 | `sn`、`device_id`、设备序列号 |
| 标签列 | `label`、`fault`、`target`、告警/故障标记 |
| 特征列 | PAIN/PAOUT/BAIN/BAOUT 光功率、EDFA/可见增益、LSR1/LSR2 管芯温度、驱动电流、制冷电流、背光电流、TEC 电压、主板/扣板/模块温度、电源电压、运行时间 |
| 工况/背景列 | 设备批次、软件版本、环境温度、站点、业务状态、运行时长、上电次数 |

当前默认配置通常使用：

```text
timestamp_col: DateTime
group_col: sn
label_col: label
feature_cols_file: assets/feature_cols.txt
```

### 1.3 检查数据质量

检查：

- 行数、列数、数值列数量。
- 缺失率最高的字段，特别是光功率、温度、电流、电压核心字段。
- 同一 `sn + DateTime` 是否存在重复记录。
- 时间列能否解析，是否按同一 `sn` 内时间递增。
- 每个 `sn` 的样本数量，是否至少满足 `assets/default.yaml` 中的 `min_history`。
- 标签比例、正负样本数量，以及标签是否可能包含故障后的未来信息。
- 采样间隔是否稳定；若不稳定，需要说明窗口中的 64 个点不一定对应固定物理时长。
- 特征列是否与 `assets/feature_cols.txt` 对齐，字段顺序是否和模型参数兼容。
- 字段单位是否符合光放大器语义，例如温度不能是 mA，电压不能是 dB。

### TEMPLATE：快速读取与概览表格

```python
from pathlib import Path
import pandas as pd

path = Path("data.csv")

if path.suffix.lower() == ".csv":
    df = pd.read_csv(path)
elif path.suffix.lower() in {".xlsx", ".xls"}:
    df = pd.read_excel(path)
elif path.suffix.lower() == ".parquet":
    df = pd.read_parquet(path)
else:
    raise ValueError(f"Unsupported file type: {path.suffix}")

print(df.shape)
print(df.head())
print(df.dtypes)
print(df.isna().mean().sort_values(ascending=False).head(20))
```

### TEMPLATE：识别常见字段

```python
cols = list(df.columns)

time_like = [c for c in cols if any(k in c.lower() for k in ["time", "date", "timestamp"])]
id_like = [
    c for c in cols
    if c.lower() in {"sn", "id", "device_id", "unit_id"}
    or "device" in c.lower()
]
label_like = [c for c in cols if c.lower() in {"label", "fault", "target", "y", "alarm"}]
numeric_cols = df.select_dtypes(include="number").columns.tolist()

oa_signal_like = [
    c for c in cols
    if any(k in str(c) for k in [
        "PAIN", "PAOUT", "BAIN", "BAOUT", "EDFA", "可见增益",
        "LSR1", "LSR2", "管芯温度", "驱动电流", "制冷电流",
        "背光电流", "TEC", "光功率", "主板温度", "扣板温度",
        "模块壳温", "电源", "3V3",
    ])
]

print("time_like:", time_like)
print("id_like:", id_like)
print("label_like:", label_like)
print("numeric_cols:", numeric_cols[:50])
print("oa_signal_like:", oa_signal_like[:80])
```

### 可用脚本

正式进入当前项目流程时，使用内置预处理入口：

```powershell
python scripts\optical-amplifier-fault-discovery\scripts\prepare_data.py --config scripts\optical-amplifier-fault-discovery\configs\default.yaml --input <input-table> --output <output-dir>
```

普通 skill 使用时优先运行统一入口脚本；它会间接调用上述预处理逻辑：

```powershell
python <skill目录>\scripts\run_oa_fault_discovery.py --input <parquet文件或目录> --cpu
```

## Step 2：数据预处理（Data Preprocessing）

### 2.1 基础清洗

根据数据情况执行：

- 标准化设备 ID，例如去空格、统一大小写。
- 解析时间列，删除无法解析时间的记录。
- 按 `sn` 和 `DateTime` 排序。
- 删除重复记录，默认策略参考 `assets/default.yaml` 中的 `duplicate_policy`。
- 处理缺失值，记录填充方式。
- 剔除明显非特征列，例如文件名、备注、未来告警时间、来源文件、人工说明。
- 确认 `assets/feature_cols.txt` 中的字段都能在数据中找到；不要随意改变字段顺序。
- 当前模型和特征挖掘只使用 `assets/feature_cols.txt` / `configs/feature_cols.txt` 中定义的字段；数据集中额外存在但不在字段清单内的列默认忽略，不进入窗口、原始字段贡献或语义耦合候选。
- 例如 `P`、`FP` 开头的额外字段，如果没有写入字段清单，默认不参与本 skill 分析。

### TEMPLATE：按设备和时间排序

```python
device_col = "sn"
time_col = "DateTime"

df[device_col] = (
    df[device_col]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", "", regex=True)
    .str.upper()
)

df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.dropna(subset=[device_col, time_col])
df = df.sort_values([device_col, time_col]).reset_index(drop=True)
```

### 2.2 缺失值与异常值处理

优先使用保守策略：

- 同设备内前向填充。
- 同设备内后向填充。
- 仍缺失时用训练集或当前数据统计量填充。
- 对极端值使用分位数裁剪或保留并标记，避免无解释删除。
- 对温度、电压、电流、光功率字段，先确认单位和量纲，再解释异常值。

当前默认预处理会参考 `assets/default.yaml`：

```text
clip_quantiles: [0.001, 0.999]
standardize: true
add_missing_indicators: false
```

### TEMPLATE：保守缺失值填充

```python
feature_cols = df.select_dtypes(include="number").columns.tolist()
device_col = "sn"

df[feature_cols] = (
    df.groupby(device_col, group_keys=False)[feature_cols]
    .apply(lambda g: g.ffill().bfill())
)

medians = df[feature_cols].median(numeric_only=True)
df[feature_cols] = df[feature_cols].fillna(medians).fillna(0.0)
```

### 2.3 构造窗口或样本

当前项目的窗口构造参数必须优先从 `assets/default.yaml` 读取，不要凭经验假设：

```text
window_len: 64
stride: 64
min_history: 64
label_strategy: last
max_windows_total: 80000
window_sample_mode: uniform
```

默认窗口构造方式：

- 按 `sn` 分组，不跨设备构造窗口。
- 每个 `sn` 内按 `DateTime` 升序排序。
- 只有样本数达到 `min_history` 的设备才参与窗口构造。
- 每个窗口包含同一设备连续 `window_len` 个时间点。
- 当前 `stride=64`，因此默认是不重叠窗口。
- 窗口标签使用 `label_strategy: last`，即取窗口最后一个时间点的标签。
- 输出数组形状为 `(窗口数, 特征数, window_len)`。
- 如果窗口总数超过 `max_windows_total`，按 `window_sample_mode: uniform` 均匀抽样。

### TEMPLATE：窗口构造逻辑示意

```python
window_len = 64
stride = 64

windows = []
for sn, part in df.groupby("sn", sort=False):
    part = part.sort_values("DateTime").reset_index(drop=True)
    if len(part) < window_len:
        continue
    for end in range(window_len, len(part) + 1, stride):
        start = end - window_len
        window = part.iloc[start:end]
        x = window[feature_cols].to_numpy().T
        y = float(window["label"].iloc[-1])
        windows.append((sn, start, end - 1, x, y))
```

### 2.4 输出为可运行输入

统一入口脚本当前优先接受：

```text
具体 parquet 文件
只包含一个 parquet 的目录
```

转换后的 parquet 至少应保留：

```text
DateTime
sn
label
assets/feature_cols.txt 中定义的光放大器遥测字段
```

## 必须停止的情况

- 电压字段使用 mA、温度字段使用 mA、光功率字段使用温度单位等语义和单位明显不匹配。
- 缺少时间列、设备列或主要光放大器遥测字段，且无法从用户上下文推断。
- 标签定义不清楚，且用户要求输出强故障结论。
- 数据不是光放大器字段体系，却要求套用当前光放大器物理耦合规则。
- 当前数据字段顺序、窗口长度或特征数量与已有模型参数不兼容，且用户不打算重新准备模型。
