"""Metrics and plot helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def binary_metrics(labels: list[int], scores: list[float], threshold: float = 0.5) -> dict[str, float | int]:
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == 1 and label == 0)
    tn = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == 0 and label == 0)
    fn = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == 0 and label == 1)
    total = len(labels)
    positive = sum(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ap = average_precision(labels, scores)
    return {
        "count": total,
        "positive": positive,
        "negative": total - positive,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc(labels, scores),
        "average_precision": ap,
        "pr_auc": ap,
        "positive_prior": positive / total if total else 0.0,
    }


def auroc(labels: list[int], scores: list[float]) -> float:
    positive = sum(labels)
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return 0.0
    pairs = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        rank_sum += avg_rank * sum(label for _score, label in pairs[index:end])
        index = end
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def average_precision(labels: list[int], scores: list[float]) -> float:
    positive = sum(labels)
    if positive == 0:
        return 0.0
    pairs = sorted(zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True)
    tp = 0
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(pairs, start=1):
        if label == 1:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / positive


def precision_recall_curve(labels: list[int], scores: list[float]) -> tuple[list[float], list[float]]:
    positive = sum(labels)
    if positive == 0:
        return [0.0, 1.0], [0.0, 0.0]
    pairs = sorted(zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True)
    recall = [0.0]
    precision = [1.0]
    tp = 0
    fp = 0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == score:
            if pairs[index][1] == 1:
                tp += 1
            else:
                fp += 1
            index += 1
        recall.append(tp / positive)
        precision.append(tp / (tp + fp) if tp + fp else 0.0)
    return recall, precision


def plot_pr_curve(labels: list[int], scores: list[float], output_dir: Path, *, model_name: str) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "STIXGeneral"],
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    recall, precision = precision_recall_curve(labels, scores)
    ap = average_precision(labels, scores)
    prior = sum(labels) / len(labels) if labels else 0.0
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "pr_curve.png"
    pdf_path = output_dir / "pr_curve.pdf"
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.step(recall, precision, where="post", color="#0072B2", linewidth=2.2, label=f"{model_name} (AP = {ap:.3f})")
    ax.axhline(prior, color="#6E6E6E", linestyle="--", linewidth=1.2, label=f"Positive Prior = {prior:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"pr_curve_png": str(png_path), "pr_curve_pdf": str(pdf_path)}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
