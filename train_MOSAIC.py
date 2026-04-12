from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from data.MOSAIC_data import MultiTaskMethylationDataModule
from models.MOSAIC import DNABert2PromptBinaryMoE, PromptBinaryLossWeights
from utils.MOSAIC_utils import inverse_frequency_weights, set_seed


DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MOSAIC on the processed methylation benchmark.")
    parser.add_argument("--config", type=Path, default=Path("configs/MOSAIC.yaml"), help="Path to the YAML config.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(DEFAULT_SEED)
    pl.seed_everything(DEFAULT_SEED, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    dataset_cfg = config["dataset"]
    paths_cfg = config["paths"]
    moe_cfg = config.get("moe", {})

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
    data_module.setup("fit")

    detect_weights = inverse_frequency_weights(data_module.detect_class_counts, num_classes=2)
    loss_cfg = config.get("loss_weights", {})
    loss_weights = PromptBinaryLossWeights(
        classify=float(loss_cfg.get("classify", 1.0)),
        gating=float(loss_cfg.get("gating", 0.3)),
    )
    expert_names: List[str] = [data_module.get_expert_name(idx) for idx in range(data_module.num_experts)]

    model = DNABert2PromptBinaryMoE(
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

    output_dir = Path(paths_cfg["output_dir"])
    checkpoint_dir = Path(paths_cfg["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="MOSAIC-{epoch:02d}-{val_detect_acc_epoch:.3f}",
        monitor="val/detect_acc_epoch",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer_cfg = config.get("trainer", {})
    trainer_kwargs = dict(
        max_epochs=int(config["max_epochs"]),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val"),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", "auto"),
        precision=trainer_cfg.get("precision"),
        accumulate_grad_batches=int(config.get("gradient_accumulation_steps", 1)),
        val_check_interval=trainer_cfg.get("val_check_interval", 1.0),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
        callbacks=[checkpoint_callback, lr_monitor],
        enable_checkpointing=trainer_cfg.get("enable_checkpointing", True),
        enable_progress_bar=trainer_cfg.get("enable_progress_bar", False),
        default_root_dir=output_dir,
    )
    for key in ("limit_train_batches", "limit_val_batches", "limit_test_batches"):
        if key in trainer_cfg:
            trainer_kwargs[key] = trainer_cfg[key]

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=data_module)

    if checkpoint_callback.best_model_path:
        trainer.test(ckpt_path="best", datamodule=data_module)
    else:
        trainer.test(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
