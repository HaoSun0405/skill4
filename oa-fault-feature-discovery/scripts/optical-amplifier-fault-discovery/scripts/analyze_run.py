import argparse
import json
import math
from pathlib import Path

import pandas as pd


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value, digits=4):
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def best_predictor_row(history):
    if "auprc" in history.columns and history["auprc"].notna().any():
        return history.loc[history["auprc"].idxmax()].to_dict()
    return history.loc[history["val_loss"].idxmin()].to_dict()


def predictor_comment(history):
    if len(history) < 2:
        return "训练轮数较少，暂不判断趋势。"
    first = history.iloc[0]
    last = history.iloc[-1]
    comments = []
    if last["train_loss"] < first["train_loss"] and last["val_loss"] > first["val_loss"]:
        comments.append("训练损失下降但验证损失上升，存在过拟合风险。")
    elif last["train_loss"] < first["train_loss"] and last["val_loss"] <= first["val_loss"]:
        comments.append("训练损失和验证损失整体同步改善。")
    if "auprc" in history and history["auprc"].notna().any():
        baseline = max(float(history["positive_rate"].dropna().max()), 1e-12)
        if history["auprc"].max() <= baseline * 1.2:
            comments.append("AUPRC 只略高于正样本比例，模型可能接近随机。")
        else:
            comments.append("AUPRC 明显高于正样本比例，模型学到了一定区分信号。")
    return " ".join(comments) if comments else "指标不足，暂不判断趋势。"


def generator_summary(history):
    if history.empty:
        return ["- 未找到 generator/train_history.csv。"]

    lines = []
    metric_col = "val_loss" if "val_loss" in history.columns else "loss"
    best_idx = history[metric_col].idxmin()
    best = history.loc[best_idx].to_dict()
    first = history.iloc[0].to_dict()
    last = history.iloc[-1].to_dict()

    lines.append(f"- 最佳 epoch: {fmt(best.get('epoch'), 0)}")
    lines.append(f"- 最佳 {metric_col}: {fmt(best.get(metric_col))}")
    lines.append(
        "- 首轮 train loss/reconstruction/kl: "
        f"{fmt(first.get('loss'))} / {fmt(first.get('reconstruction'))} / {fmt(first.get('kl'))}"
    )
    lines.append(
        "- 末轮 train loss/reconstruction/kl: "
        f"{fmt(last.get('loss'))} / {fmt(last.get('reconstruction'))} / {fmt(last.get('kl'))}"
    )
    if "val_loss" in history.columns:
        lines.append(
            "- 末轮 val loss/reconstruction/kl: "
            f"{fmt(last.get('val_loss'))} / {fmt(last.get('val_reconstruction'))} / {fmt(last.get('val_kl'))}"
        )

    numeric_cols = [c for c in ["loss", "reconstruction", "kl", "val_loss", "val_reconstruction", "val_kl"] if c in history.columns]
    if history[numeric_cols].isna().any().any():
        lines.append("- 存在 NaN，生成模型训练不稳定。")
    elif any((history[c].abs() > 1e8).any() for c in numeric_cols):
        lines.append("- 指标出现极大值，生成模型可能发生数值爆炸。")
    else:
        lines.append("- 未发现 NaN 或明显数值爆炸。")

    best_epoch = int(best.get("epoch", 0))
    last_epoch = int(last.get("epoch", 0))
    if last_epoch > best_epoch:
        lines.append(f"- 最佳 epoch 之后又训练了 {last_epoch - best_epoch} 轮；如果差值接近 patience，说明 early stopping 生效。")
    if len(history) >= 5:
        recent = history[metric_col].tail(min(5, len(history)))
        if recent.max() - recent.min() < max(abs(float(recent.mean())) * 0.01, 1e-6):
            lines.append("- 最近几轮变化很小，生成模型基本进入平台期。")
        elif float(last.get(metric_col)) > float(best.get(metric_col)) * 1.2:
            lines.append("- 末轮明显差于最佳轮，建议使用保存的最佳 vae.pt，不要使用末轮状态。")
        else:
            lines.append("- 仍有波动，建议结合重建样本或下游 morph 效果判断。")
    return lines


def make_report(run_dir):
    run_dir = Path(run_dir)
    run_summary = read_json(run_dir / "run_summary.json") if (run_dir / "run_summary.json").exists() else {}
    preprocess = read_json(run_dir / "processed_data" / "preprocess_summary.json") if (run_dir / "processed_data" / "preprocess_summary.json").exists() else {}
    predictor_history = pd.read_csv(run_dir / "predictor" / "train_history.csv") if (run_dir / "predictor" / "train_history.csv").exists() else pd.DataFrame()
    generator_history = pd.read_csv(run_dir / "generator" / "train_history.csv") if (run_dir / "generator" / "train_history.csv").exists() else pd.DataFrame()
    threshold_metrics = read_json(run_dir / "predictor" / "threshold_metrics.json") if (run_dir / "predictor" / "threshold_metrics.json").exists() else {}
    validation_pred_path = run_dir / "predictor" / "validation_predictions.csv"

    lines = ["# 模型训练分析报告", ""]
    lines.append("## 数据与窗口")
    lines.append(f"- 窗口数量: {fmt(preprocess.get('n_windows'))}")
    lines.append(f"- 特征数量: {fmt(preprocess.get('n_features'))}")
    lines.append(f"- 窗口形状: {preprocess.get('window_shape', run_summary.get('window_shape', 'NA'))}")
    lines.append(f"- 窗口数组估算内存 GiB: {fmt(preprocess.get('window_array_gib'))}")
    lines.append(f"- 正样本比例: {fmt(preprocess.get('label_positive_rate'))}")
    lines.append(f"- 划分数量: {preprocess.get('splits', 'NA')}")
    lines.append("")

    lines.append("## 预测模型")
    if predictor_history.empty:
        lines.append("- 未找到 predictor/train_history.csv。")
    else:
        best = best_predictor_row(predictor_history)
        lines.append(f"- 最佳 epoch: {fmt(best.get('epoch'), 0)}")
        lines.append(f"- AUC: {fmt(best.get('auc'))}")
        lines.append(f"- AUPRC: {fmt(best.get('auprc'))}")
        lines.append(f"- 验证损失: {fmt(best.get('val_loss'))}")
        lines.append(f"- 验证集正样本比例: {fmt(best.get('positive_rate'))}")
        lines.append(f"- 趋势判断: {predictor_comment(predictor_history)}")
    lines.append("")

    if threshold_metrics:
        lines.append("## 阈值评估")
        lines.append(f"- best_threshold: {fmt(threshold_metrics.get('best_threshold'))}")
        lines.append(f"- best_f1: {fmt(threshold_metrics.get('best_f1'))}")
        lines.append(f"- best_precision: {fmt(threshold_metrics.get('best_precision'))}")
        lines.append(f"- best_recall: {fmt(threshold_metrics.get('best_recall'))}")
        lines.append(
            "- 混淆矩阵: "
            f"TN={fmt(threshold_metrics.get('tn'), 0)}, "
            f"FP={fmt(threshold_metrics.get('fp'), 0)}, "
            f"FN={fmt(threshold_metrics.get('fn'), 0)}, "
            f"TP={fmt(threshold_metrics.get('tp'), 0)}"
        )
        for key in ["precision_at_recall_80", "precision_at_recall_90"]:
            if key in threshold_metrics:
                lines.append(f"- {key}: {fmt(threshold_metrics[key])}")
        lines.append("")

    lines.append("## 验证集预测明细")
    if validation_pred_path.exists():
        pred = pd.read_csv(validation_pred_path)
        lines.append(f"- 明细文件: {validation_pred_path}")
        lines.append(f"- 验证窗口数: {len(pred)}")
        if {"sn", "end_time", "true_label", "pred_score"}.issubset(pred.columns):
            top = pred.sort_values("pred_score", ascending=False).head(10)
            lines.append("- Top 10 高风险窗口:")
            lines.append("")
            lines.append("| sn | end_time | true_label | pred_score |")
            lines.append("|---|---|---:|---:|")
            for _, row in top.iterrows():
                lines.append(f"| {row['sn']} | {row['end_time']} | {int(row['true_label'])} | {float(row['pred_score']):.4f} |")
    else:
        lines.append("- 未找到 predictor/validation_predictions.csv。")
    lines.append("")

    lines.append("## 生成模型")
    lines.extend(generator_summary(generator_history))
    lines.append("")

    lines.append("## 建议")
    if not predictor_history.empty:
        best = best_predictor_row(predictor_history)
        pos = best.get("positive_rate")
        auprc = best.get("auprc")
        if pd.notna(auprc) and pd.notna(pos) and auprc > pos * 2:
            lines.append("- 预测模型初步有效，建议检查 validation_predictions.csv 中高分窗口是否符合业务直觉。")
        else:
            lines.append("- 预测模型优势不明显，建议检查 label 质量、窗口长度和特征列。")
    if not generator_history.empty:
        lines.append("- 生成模型是否真正有用，不能只看 loss；还需要看后续重建样本、latent 插值或 morph 后的特征变化是否合理。")
    lines.append("- 若误报成本高，重点看 precision；若漏报成本高，重点看 recall 或 precision_at_recall_90。")

    report = "\n".join(lines) + "\n"
    report_path = run_dir / "analysis_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run output directory, e.g. outputs/run_001")
    args = parser.parse_args()
    report_path, report = make_report(args.run)
    print(report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
