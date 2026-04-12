from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef

from data.datamodule import MultiTaskMethylationDataModule, TYPE_GROUPS
from models.dnabert2_moe_prompt_binary import DNABert2PromptBinaryMoE, PromptBinaryLossWeights
from utils.class_weights import inverse_frequency_weights
from utils.seed import set_seed


DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the released MOSAIC checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/MOSAIC.yaml"), help="Config YAML.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/MOSAIC_model.ckpt"), help="Checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"), help="Directory for JSON output.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_datamodule(config: Dict[str, Any]) -> MultiTaskMethylationDataModule:
    dataset_cfg = config["dataset"]
    data_module = MultiTaskMethylationDataModule(
        dataset_root=Path(dataset_cfg["root"]),
        groups=dataset_cfg["groups"],
        model_name=config["model_name"],
        max_length=int(config["max_length"]),
        train_batch_size=int(config["train_batch_size"]),
        eval_batch_size=int(config["eval_batch_size"]),
        num_workers=int(dataset_cfg.get("num_workers", 0)),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.1)),
        seed=DEFAULT_SEED,
        type_sampling_weights=dataset_cfg.get("type_sampling_weights"),
        sampler_class_probs=dataset_cfg.get("sampler_class_probs"),
        include_dataset_labels=True,
    )
    data_module.setup(stage="test")
    return data_module


def build_model(config: Dict[str, Any], checkpoint: Path, data_module: MultiTaskMethylationDataModule) -> DNABert2PromptBinaryMoE:
    detect_weights = inverse_frequency_weights(data_module.detect_class_counts, num_classes=2)
    loss_cfg = config.get("loss_weights", {})
    loss_weights = PromptBinaryLossWeights(
        classify=float(loss_cfg.get("classify", 1.0)),
        gating=float(loss_cfg.get("gating", 0.3)),
    )
    expert_names: List[str] = [data_module.get_expert_name(idx) for idx in range(data_module.num_experts)]
    moe_cfg = config.get("moe", {})

    model = DNABert2PromptBinaryMoE.load_from_checkpoint(
        checkpoint,
        model_name=config["model_name"],
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        warmup_ratio=float(config["warmup_ratio"]),
        loss_weights=loss_weights,
        num_experts=data_module.num_experts,
        expert_names=expert_names,
        detect_class_weights=torch.tensor(detect_weights, dtype=torch.float),
        num_species=data_module.num_species,
        num_tasks=data_module.num_tasks,
        lora_rank=int(moe_cfg.get("lora_rank", 16)),
        lora_dropout=float(moe_cfg.get("lora_dropout", 0.1)),
        top_k=moe_cfg.get("top_k"),
        use_middle_token_only=bool(config.get("use_middle_token_only", False)),
        max_length=int(config["max_length"]),
    )
    model.eval()
    return model


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    if not y_true:
        return {"ACC": 0.0, "MCC": 0.0, "SN": 0.0, "SP": 0.0, "F1": 0.0}
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = cm[0, 0] if cm.shape[0] > 0 else 0
        tp = cm[-1, -1] if cm.shape[0] > 1 else 0
        fp = fn = 0
    sn = tp / (tp + fn + 1e-12)
    sp = tn / (tn + fp + 1e-12)
    return {"ACC": float(acc), "MCC": float(mcc), "SN": float(sn), "SP": float(sp), "F1": float(f1)}


def evaluate(model: DNABert2PromptBinaryMoE, data_module: MultiTaskMethylationDataModule) -> Dict[str, Any]:
    loader = data_module.test_dataloader()
    if loader is None:
        raise RuntimeError("Test dataloader is unavailable.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    y_true: List[int] = []
    y_pred: List[int] = []
    dataset_ids: List[int] = []
    type_ids: List[int] = []

    with torch.no_grad():
        for batch in loader:
            dataset_ids.extend(batch["expert_labels"].tolist())
            type_ids.extend(batch["task_ids"].tolist())
            labels = batch["detect_labels"]
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                species_ids=batch["species_ids"],
                task_ids=batch["task_ids"],
            )
            preds = outputs["detect_logits"].argmax(dim=-1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.cpu().tolist())

    results: Dict[str, Any] = {"overall": compute_metrics(y_true, y_pred)}

    per_dataset: List[Dict[str, Any]] = []
    by_dataset_pred: Dict[int, List[int]] = defaultdict(list)
    by_dataset_true: Dict[int, List[int]] = defaultdict(list)
    for idx, dataset_id in enumerate(dataset_ids):
        by_dataset_pred[dataset_id].append(y_pred[idx])
        by_dataset_true[dataset_id].append(y_true[idx])

    for dataset_id, preds in by_dataset_pred.items():
        metrics = compute_metrics(by_dataset_true[dataset_id], preds)
        per_dataset.append(
            {
                "dataset": data_module.get_expert_name(dataset_id),
                "samples": len(preds),
                **metrics,
            }
        )

    type_names = {value: key for key, value in TYPE_GROUPS.items()}
    per_type: List[Dict[str, Any]] = []
    by_type_pred: Dict[int, List[int]] = defaultdict(list)
    by_type_true: Dict[int, List[int]] = defaultdict(list)
    for idx, type_id in enumerate(type_ids):
        by_type_pred[type_id].append(y_pred[idx])
        by_type_true[type_id].append(y_true[idx])

    for type_id, preds in by_type_pred.items():
        metrics = compute_metrics(by_type_true[type_id], preds)
        per_type.append(
            {
                "type": type_names.get(type_id, str(type_id)),
                "samples": len(preds),
                **metrics,
            }
        )

    per_dataset.sort(key=lambda item: item["dataset"])
    per_type.sort(key=lambda item: item["type"])
    results["per_dataset"] = per_dataset
    results["per_type"] = per_type
    return results


def main() -> None:
    args = parse_args()
    set_seed(DEFAULT_SEED)

    config = load_config(args.config)
    data_module = build_datamodule(config)
    model = build_model(config, args.checkpoint, data_module)
    eval_results = evaluate(model, data_module)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"eval_{args.checkpoint.stem}.json"
    payload = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **eval_results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()
