from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

from data.MOSAIC_data import MultiTaskMethylationDataModule, TYPE_GROUPS
from models.MOSAIC import DNABert2PromptBinaryMoE, PromptBinaryLossWeights
from utils.MOSAIC_utils import inverse_frequency_weights, set_seed


DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MOSAIC inference on sequences.")
    parser.add_argument("--config", type=Path, default=Path("configs/MOSAIC.yaml"), help="Config YAML.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/MOSAIC_model.ckpt"), help="Checkpoint path.")
    parser.add_argument("--sequence", type=str, help="Single DNA sequence for inference.")
    parser.add_argument("--species", type=str, help="Species for single-sequence inference, e.g. A.thaliana.")
    parser.add_argument("--task", type=str, choices=sorted(TYPE_GROUPS), help="Methylation task for single-sequence inference.")
    parser.add_argument("--input-csv", type=Path, help="Batch CSV with columns: sequence,species,task.")
    parser.add_argument("--output-csv", type=Path, help="Optional output CSV path for batch inference.")
    parser.add_argument("--top-experts", type=int, default=4, help="How many routing experts to report.")
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


def normalize_text(value: str) -> str:
    return value.strip().replace(" ", "").lower()


def build_species_lookup(data_module: MultiTaskMethylationDataModule) -> Dict[str, str]:
    return {normalize_text(name): name for name in data_module.species_to_id}


def resolve_species(species_input: str, species_lookup: Dict[str, str]) -> str:
    key = normalize_text(species_input)
    if key not in species_lookup:
        valid = ", ".join(sorted(species_lookup.values()))
        raise ValueError(f"Unknown species '{species_input}'. Valid values: {valid}")
    return species_lookup[key]


def masked_routing_probs(gating_logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    probs = torch.softmax(gating_logits, dim=-1)
    if top_k is not None and top_k < probs.size(-1):
        top_values, top_indices = torch.topk(probs, top_k, dim=-1)
        mask = probs.new_zeros(probs.shape)
        mask.scatter_(1, top_indices, 1.0)
        probs = probs * mask
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
    return probs


def run_batch(
    rows: List[Dict[str, str]],
    data_module: MultiTaskMethylationDataModule,
    model: DNABert2PromptBinaryMoE,
    max_length: int,
    top_experts: int,
) -> List[Dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    species_lookup = build_species_lookup(data_module)
    tokenizer = data_module.tokenizer

    results: List[Dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            sequence = row["sequence"].strip().upper()
            canonical_species = resolve_species(row["species"], species_lookup)
            task_name = row["task"].strip()
            if task_name not in TYPE_GROUPS:
                raise ValueError(f"Unknown task '{task_name}'. Valid values: {sorted(TYPE_GROUPS)}")

            encoded = tokenizer(
                sequence,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            species_ids = torch.tensor([data_module.species_to_id[canonical_species]], dtype=torch.long, device=device)
            task_ids = torch.tensor([TYPE_GROUPS[task_name]], dtype=torch.long, device=device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                species_ids=species_ids,
                task_ids=task_ids,
            )
            detect_probs = torch.softmax(outputs["detect_logits"], dim=-1)[0]
            routing_probs = masked_routing_probs(outputs["gating_logits"], model.top_k)[0]
            top_n = min(top_experts, routing_probs.numel())
            top_values, top_indices = torch.topk(routing_probs, k=top_n)

            result: Dict[str, Any] = {
                "sequence": sequence,
                "species": canonical_species,
                "task": task_name,
                "pred_label": int(torch.argmax(detect_probs).item()),
                "pred_class": "methylated" if int(torch.argmax(detect_probs).item()) == 1 else "non-methylated",
                "prob_non_methylated": float(detect_probs[0].item()),
                "prob_methylated": float(detect_probs[1].item()),
            }
            for rank, (index, value) in enumerate(zip(top_indices.tolist(), top_values.tolist()), start=1):
                result[f"top{rank}_expert"] = data_module.get_expert_name(int(index))
                result[f"top{rank}_weight"] = float(value)
            results.append(result)
    return results


def load_rows(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.input_csv is not None:
        with args.input_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
        required = {"sequence", "species", "task"}
        if not rows:
            return []
        missing = required - set(rows[0].keys())
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
        return rows

    if args.sequence and args.species and args.task:
        return [{"sequence": args.sequence, "species": args.species, "task": args.task}]

    raise ValueError("Provide either --input-csv or the trio --sequence/--species/--task.")


def main() -> None:
    args = parse_args()
    set_seed(DEFAULT_SEED)

    config = load_config(args.config)
    data_module = build_datamodule(config)
    model = build_model(config, args.checkpoint, data_module)
    rows = load_rows(args)
    results = run_batch(rows, data_module, model, max_length=int(config["max_length"]), top_experts=int(args.top_experts))

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            if results:
                writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
            else:
                writer = csv.writer(handle)
                writer.writerow(["sequence", "species", "task"])
        print(f"Saved inference results to {args.output_csv}")
        return

    if len(results) == 1:
        print(json.dumps(results[0], indent=2))
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
