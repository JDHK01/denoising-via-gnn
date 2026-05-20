#!/usr/bin/env python3
"""Train and evaluate the all-in HGT alert denoising model."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .data import AlertRecord, load_records
from .features import AlertTfidfVectorizer, IpInfoTfidfVectorizer, collect_ips_from_records
from .graph import AllInGraphBuilder, GraphBuilderConfig, NodeRef
from .ip_enrichment import IpEnrichment
from .metrics import binary_metrics, plot_pr_curve, write_json
from .model import AllInModelConfig, HGTOnlyClassifier
from .tensorize import NODE_TYPES, move_graph, to_tensor_graph


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset" / "allin" / "data_final"
DEFAULT_WORK_DIR = REPO_ROOT / "artifacts" / "allin" / "hgt_temporal"
Root = tuple[NodeRef, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    add_common_args(train)
    train.add_argument("--split", choices=("train",), default="train")
    train.add_argument("--val-ratio", type=float, default=0.2)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--best-metric", choices=("pr_auc", "f1", "auroc", "accuracy"), default="pr_auc")

    evaluate = sub.add_parser("eval")
    add_common_args(evaluate)
    evaluate.add_argument("--split", choices=("test", "train"), default="test")
    evaluate.add_argument("--checkpoint", type=Path, default=None)
    evaluate.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--tfidf-max-features", type=int, default=512)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--graph-learning", choices=("none", "temporal", "similarity", "both"), default="temporal")
    parser.add_argument("--temporal-window-seconds", type=int, default=3600)
    parser.add_argument("--temporal-prev", type=int, default=2)
    parser.add_argument("--similarity-candidate-window", type=int, default=8)
    parser.add_argument("--similarity-topk", type=int, default=2)
    parser.add_argument("--similarity-min-score", type=float, default=0.05)
    parser.add_argument("--no-src-dst-edges", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--hgt-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--no-alert-residual", action="store_true")
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16", "float16"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def train(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    set_seed(args.seed)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    save_command_script(args.work_dir, "command_train.sh")

    print(f"[1/6] loading records dataset={args.dataset_dir} split=train max_records={args.max_records}")
    records = load_records(args.dataset_dir, "train", max_records=args.max_records)
    summarize_records(records, args.work_dir / "train_dataset_summary.json")
    print_summary(records)
    if not records:
        raise RuntimeError("no training records loaded")

    print("[2/6] fitting TF-IDF on train records only")
    vectorizer = AlertTfidfVectorizer(
        max_features=args.tfidf_max_features,
        min_df=args.tfidf_min_df,
        ngram_range=(1, args.tfidf_ngram_max),
    )
    vectorizer.fit(records)
    alert_tfidf = vectorizer.transform(records)
    vectorizer_path = args.work_dir / "tfidf_vectorizer.pkl"
    vectorizer.save(vectorizer_path)
    print(
        f"  alert_tfidf fit_records={len(records)} "
        f"train_shape={alert_tfidf.shape} saved={vectorizer_path}"
    )

    print("[2.5/6] fitting IP info TF-IDF on train records only")
    ips = collect_ips_from_records(records)
    with IpEnrichment() as enrich:
        ip_info_map = enrich.lookup_batch(ips)
    ip_info_tfidf_vec = IpInfoTfidfVectorizer()
    ip_info_tfidf_vec.fit(records, ip_info_map)
    ip_info_tfidf_path = args.work_dir / "ip_info_tfidf_vectorizer.pkl"
    ip_info_tfidf_vec.save(ip_info_tfidf_path)
    print(
        f"  ip_info_tfidf fit_records={len(records)} "
        f"output_dim={ip_info_tfidf_vec.output_dim} saved={ip_info_tfidf_path}"
    )

    print(f"[3/6] building graph graph_learning={args.graph_learning}")
    graph = build_graph(records, alert_tfidf, ip_info_tfidf_vec, ip_info_map, args)
    print(f"  graph nodes={len(graph.nodes)} edges={len(graph.edges)} edge_types={len(graph.edge_types())}")

    print("[4/6] tensorizing graph")
    tensorized = to_tensor_graph(graph, alert_tfidf=alert_tfidf)
    roots = labeled_roots(records, tensorized)
    if not roots:
        raise RuntimeError("no labeled training roots found")
    train_roots, val_roots = split_roots(roots, val_ratio=args.val_ratio, seed=args.seed)
    print(f"  roots labeled={len(roots)} train={len(train_roots)} val={len(val_roots)}")
    print(f"  input_dims={tensorized.input_dims}")
    print(f"  edge_types={tensorized.edge_types}")

    device = torch.device(args.device)
    tensor_graph = move_graph(tensorized.graph, device)
    config = AllInModelConfig(
        node_types=NODE_TYPES,
        edge_types=tensorized.edge_types,
        input_dims=tensorized.input_dims,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        hgt_layers=args.hgt_layers,
        dropout=args.dropout,
        alert_residual=not args.no_alert_residual,
    )
    model = HGTOnlyClassifier(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_labels = [label for _root, label in train_roots]
    positive = sum(train_labels)
    negative = len(train_labels) - positive
    pos_weight = torch.tensor([negative / positive if positive else 1.0], dtype=torch.float32, device=device)
    amp_dtype = resolve_amp_dtype(args.amp_dtype)

    print(
        f"[5/6] training epochs={args.epochs} device={device} pos={positive} neg={negative} "
        f"pos_weight={float(pos_weight.item()):.3f}"
    )
    best_score = float("-inf")
    best_payload: dict[str, Any] = {}
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        random.Random(args.seed + epoch).shuffle(train_roots)
        train_loss, train_scores = train_step(model, tensor_graph, train_roots, optimizer, pos_weight, args, amp_dtype)
        train_metrics = binary_metrics([label for _root, label in train_roots], train_scores)
        val_loss, val_scores = evaluate_roots(model, tensor_graph, val_roots, pos_weight, args, amp_dtype)
        val_labels = [label for _root, label in val_roots]
        val_threshold, val_f1 = best_f1_threshold(val_labels, val_scores)
        val_metrics = binary_metrics(val_labels, val_scores, threshold=val_threshold if val_roots else 0.5)
        val_metrics["loss"] = val_loss
        val_metrics["best_threshold"] = val_threshold
        val_metrics["best_threshold_f1"] = val_f1
        train_metrics["loss"] = train_loss
        score = float(val_metrics.get(args.best_metric, 0.0) if val_roots else train_metrics.get(args.best_metric, 0.0))
        print(
            f"  epoch={epoch} train_loss={train_loss:.4f} train_ap={train_metrics['pr_auc']:.4f} "
            f"val_loss={val_loss:.4f} val_ap={val_metrics.get('pr_auc', 0.0):.4f} "
            f"val_f1={val_metrics.get('f1', 0.0):.4f} threshold={val_threshold:.3f} "
            f"elapsed={time.perf_counter() - epoch_started:.1f}s"
        )
        ckpt_path = save_checkpoint(args, model, config, epoch, train_metrics, val_metrics)
        if score >= best_score:
            best_score = score
            best_payload = {"epoch": epoch, "score": best_score, "train": train_metrics, "validation": val_metrics}
            shutil.copy2(ckpt_path, args.work_dir / "hgt_best.pt")
            print(f"    best checkpoint updated score={best_score:.4f}")

    write_json(args.work_dir / "metrics.json", best_payload)
    write_json(args.work_dir / "train_args.json", jsonable_args(args))
    print(f"[6/6] done elapsed={time.perf_counter() - started:.1f}s best_{args.best_metric}={best_score:.4f}")
    print(f"checkpoint={args.work_dir / 'hgt_best.pt'}")


def evaluate(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    save_command_script(args.work_dir, "command_eval.sh")
    checkpoint = args.checkpoint or args.work_dir / "hgt_best.pt"
    print(f"[1/5] loading checkpoint {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt.get("train_args", {})
    vectorizer_path = Path(train_args.get("work_dir", args.work_dir)) / "tfidf_vectorizer.pkl"
    vectorizer = AlertTfidfVectorizer.load(vectorizer_path)
    ip_info_tfidf_path = Path(train_args.get("work_dir", args.work_dir)) / "ip_info_tfidf_vectorizer.pkl"
    ip_info_tfidf_vec = IpInfoTfidfVectorizer.load(ip_info_tfidf_path)
    if args.threshold is None:
        args.threshold = float(ckpt.get("best_threshold", 0.5))
    print(f"  vectorizer={vectorizer_path} ip_info_tfidf={ip_info_tfidf_path} threshold={args.threshold:.4f}")

    print(f"[2/5] loading records split={args.split} max_records={args.max_records}")
    records = load_records(args.dataset_dir, args.split, max_records=args.max_records)
    summarize_records(records, args.work_dir / f"{args.split}_dataset_summary.json")
    print_summary(records)
    print("[3/5] transforming TF-IDF with training vectorizer")
    alert_tfidf = vectorizer.transform(records)
    ips = collect_ips_from_records(records)
    with IpEnrichment() as enrich:
        ip_info_map = enrich.lookup_batch(ips)
    print(f"  alert_tfidf shape={alert_tfidf.shape}")

    print(f"[4/5] building/tensorizing graph graph_learning={args.graph_learning}")
    graph = build_graph(records, alert_tfidf, ip_info_tfidf_vec, ip_info_map, args)
    tensorized = to_tensor_graph(graph, alert_tfidf=alert_tfidf, input_dims=ckpt["model_config"]["input_dims"])
    roots = labeled_roots(records, tensorized)
    print(f"  graph nodes={len(graph.nodes)} edges={len(graph.edges)} labeled_roots={len(roots)}")

    config = model_config_from_json(ckpt["model_config"])
    model = HGTOnlyClassifier(config)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device)
    model.to(device)
    tensor_graph = move_graph(tensorized.graph, device)
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)

    print(f"[5/5] evaluating device={device}")
    loss, scores = evaluate_roots(model, tensor_graph, roots, pos_weight, args, amp_dtype)
    labels = [label for _root, label in roots]
    metrics = binary_metrics(labels, scores, threshold=args.threshold)
    metrics["loss"] = loss
    metrics["threshold"] = args.threshold
    metrics.update(plot_pr_curve(labels, scores, args.work_dir, model_name="AllIn-HGT"))
    write_predictions(args.work_dir / "predictions.jsonl", records, tensorized, roots, scores, args.threshold)
    write_json(args.work_dir / "test_metrics.json", metrics)
    write_json(args.work_dir / "eval_args.json", jsonable_args(args))
    print(
        f"  acc={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
        f"auroc={metrics['auroc']:.4f} ap={metrics['pr_auc']:.4f}"
    )
    print(f"done elapsed={time.perf_counter() - started:.1f}s output={args.work_dir}")


# if_info is added, needs two dependences, ip_info_tdidf & ip_info_map
def build_graph(records: list[AlertRecord], alert_tfidf, ip_info_tfidf_vec, ip_info_map, args: argparse.Namespace):
    return AllInGraphBuilder(
        GraphBuilderConfig(
            graph_learning=args.graph_learning,
            temporal_window_seconds=args.temporal_window_seconds,
            temporal_prev=args.temporal_prev,
            similarity_candidate_window=args.similarity_candidate_window,
            similarity_topk=args.similarity_topk,
            similarity_min_score=args.similarity_min_score,
            add_src_dst_edges=not args.no_src_dst_edges,
        )
    ).build(records, alert_tfidf, ip_info_tfidf_vec, ip_info_map)


def train_step(
    model: HGTOnlyClassifier,
    graph,
    roots: list[Root],
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor,
    args: argparse.Namespace,
    amp_dtype: torch.dtype | None,
) -> tuple[float, list[float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    indices = [root_index for root_index, _label in roots]
    labels = torch.tensor([label for _root, label in roots], dtype=torch.float32, device=pos_weight.device)
    with autocast(pos_weight.device, amp_dtype):
        logits = model(graph, indices)
        loss = F.binary_cross_entropy_with_logits(logits.float(), labels, pos_weight=pos_weight)
    loss.backward()
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    scores = torch.sigmoid(logits.detach().float()).cpu().tolist()
    return float(loss.detach().cpu()), [float(score) for score in scores]


@torch.no_grad()
def evaluate_roots(
    model: HGTOnlyClassifier,
    graph,
    roots: list[Root],
    pos_weight: torch.Tensor,
    args: argparse.Namespace,
    amp_dtype: torch.dtype | None,
) -> tuple[float, list[float]]:
    if not roots:
        return 0.0, []
    model.eval()
    indices = [root_index for root_index, _label in roots]
    labels = torch.tensor([label for _root, label in roots], dtype=torch.float32, device=pos_weight.device)
    with autocast(pos_weight.device, amp_dtype):
        logits = model(graph, indices)
        loss = F.binary_cross_entropy_with_logits(logits.float(), labels, pos_weight=pos_weight)
    scores = torch.sigmoid(logits.float()).cpu().tolist()
    return float(loss.detach().cpu()), [float(score) for score in scores]


def labeled_roots(records: list[AlertRecord], tensorized) -> list[Root]:
    roots: list[Root] = []
    for record in records:
        if record.label is None:
            continue
        platform = record.platform_name or "__missing_platform__"
        ref = NodeRef("alert", (platform, f"alert:{record.index}"))
        if ref in tensorized.node_index:
            roots.append((tensorized.local_index(ref), int(record.label)))
    return roots


def split_roots(roots: list[Root], *, val_ratio: float, seed: int) -> tuple[list[Root], list[Root]]:
    rng = random.Random(seed)
    positives = [root for root in roots if root[1] == 1]
    negatives = [root for root in roots if root[1] == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def split_class(items: list[Root]) -> tuple[list[Root], list[Root]]:
        if len(items) < 2 or val_ratio <= 0:
            return items, []
        val_count = max(1, int(round(len(items) * val_ratio)))
        val_count = min(val_count, len(items) - 1)
        return items[val_count:], items[:val_count]

    train_pos, val_pos = split_class(positives)
    train_neg, val_neg = split_class(negatives)
    train = [*train_pos, *train_neg]
    val = [*val_pos, *val_neg]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def best_f1_threshold(labels: list[int], scores: list[float]) -> tuple[float, float]:
    if not labels:
        return 0.5, 0.0
    candidates = sorted(set(scores))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        f1 = float(binary_metrics(labels, scores, threshold=threshold)["f1"])
        if f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1
    return best_threshold, best_f1


def save_checkpoint(
    args: argparse.Namespace,
    model: HGTOnlyClassifier,
    config: AllInModelConfig,
    epoch: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
) -> Path:
    path = args.work_dir / f"hgt_epoch{epoch}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config_to_json(config),
            "epoch": epoch,
            "train_args": jsonable_args(args),
            "metrics": {"train": train_metrics, "validation": val_metrics},
            "best_threshold": val_metrics.get("best_threshold", 0.5),
        },
        path,
    )
    return path


def model_config_to_json(config: AllInModelConfig) -> dict[str, Any]:
    return {
        "node_types": list(config.node_types),
        "edge_types": [list(edge_type) for edge_type in config.edge_types],
        "input_dims": dict(config.input_dims),
        "hidden_dim": config.hidden_dim,
        "num_heads": config.num_heads,
        "hgt_layers": config.hgt_layers,
        "dropout": config.dropout,
        "alert_residual": config.alert_residual,
    }


def model_config_from_json(payload: dict[str, Any]) -> AllInModelConfig:
    return AllInModelConfig(
        node_types=tuple(payload["node_types"]),
        edge_types=tuple(tuple(edge_type) for edge_type in payload["edge_types"]),
        input_dims={key: int(value) for key, value in payload["input_dims"].items()},
        hidden_dim=int(payload["hidden_dim"]),
        num_heads=int(payload["num_heads"]),
        hgt_layers=int(payload["hgt_layers"]),
        dropout=float(payload["dropout"]),
        alert_residual=bool(payload.get("alert_residual", True)),
    )


def summarize_records(records: list[AlertRecord], path: Path) -> None:
    labeled = [record for record in records if record.label is not None]
    payload = {
        "records": len(records),
        "labeled": len(labeled),
        "positive": sum(int(record.label) for record in labeled),
        "negative": sum(1 - int(record.label) for record in labeled),
        "platforms": len({record.platform_name for record in records}),
        "uses_device_ip_list": False,
        "forbidden_model_fields_used": False,
    }
    write_json(path, payload)


def print_summary(records: list[AlertRecord]) -> None:
    labeled = [record for record in records if record.label is not None]
    positive = sum(int(record.label) for record in labeled)
    print(
        f"  records={len(records)} labeled={len(labeled)} pos={positive} "
        f"neg={len(labeled) - positive} platforms={len({record.platform_name for record in records})}"
    )


def write_predictions(path: Path, records: list[AlertRecord], tensorized, roots: list[Root], scores: list[float], threshold: float) -> None:
    by_root = {root_index: (label, score) for (root_index, label), score in zip(roots, scores, strict=True)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for root_index, (label, score) in by_root.items():
            ref = tensorized.index_node["alert"][root_index]
            record_index = int(ref.key[1].split("alert:", 1)[1])
            record = records[record_index]
            row = record.to_dict()
            row["label"] = label
            row["alert_node"] = ref.key[1]
            row["score"] = float(score)
            row["prediction"] = 1 if score >= threshold else 0
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def resolve_amp_dtype(name: str) -> torch.dtype | None:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return None


def autocast(device: torch.device, dtype: torch.dtype | None):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_command_script(output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(shell_quote(part) for part in [sys.executable, "-m", "allin.train", *sys.argv[1:]])
    (output_dir / name).write_text(command + "\n", encoding="utf-8")


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    if any(ch.isspace() or ch in "'\"$`\\" for ch in value):
        return "'" + value.replace("'", "'\"'\"'") + "'"
    return value


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
