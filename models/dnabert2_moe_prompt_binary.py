from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, MulticlassAccuracy
from transformers import AutoModel, get_linear_schedule_with_warmup


@dataclass
class PromptBinaryLossWeights:
    classify: float = 1.0
    gating: float = 0.3


class LoRAAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.rank = max(1, rank)
        self.down = nn.Linear(hidden_size, self.rank, bias=False)
        self.up = nn.Linear(self.rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(self.dropout(x))) / self.rank


class DNABert2PromptBinaryMoE(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float,
        loss_weights: PromptBinaryLossWeights,
        num_experts: int,
        expert_names: List[str],
        detect_class_weights: Optional[torch.Tensor] = None,
        num_species: int = 0,
        num_tasks: int = 0,
        lora_rank: int = 16,
        lora_dropout: float = 0.1,
        top_k: Optional[int] = None,
        use_prompt_routing: bool = True,
        use_middle_token_only: bool = False,
        max_length: int = 41,
        trainable_base_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["detect_class_weights", "expert_names"])
        self.expert_names = expert_names

        if top_k is not None and (top_k <= 0 or top_k > num_experts):
            raise ValueError(f"Invalid top_k={top_k}; must be between 1 and num_experts ({num_experts}).")
        if num_species <= 0 or num_tasks <= 0:
            raise ValueError("num_species and num_tasks must be positive.")

        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = self.encoder.config.hidden_size

        self.species_embeddings = nn.Embedding(num_species, hidden_size)
        self.task_embeddings = nn.Embedding(num_tasks, hidden_size)
        self.adapters = nn.ModuleList([LoRAAdapter(hidden_size, lora_rank, lora_dropout) for _ in range(num_experts)])

        self.gating = nn.Linear(hidden_size, num_experts)
        self.detect_experts = nn.ModuleList([nn.Linear(hidden_size, 2) for _ in range(num_experts)])

        if detect_class_weights is not None:
            self.register_buffer("detect_class_weights", detect_class_weights)
        else:
            self.detect_class_weights = None

        self.loss_weights = loss_weights
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_prompt_routing = use_prompt_routing
        self.use_middle_token_only = use_middle_token_only
        self.middle_token_index = min(max_length // 2 + 1, max_length + 1)

        self.detect_accuracy = BinaryAccuracy()
        self.detect_f1 = BinaryF1Score()
        self.gating_accuracy = MulticlassAccuracy(num_classes=num_experts)

        self._configure_base(trainable_base_layers)

    def _configure_base(self, trainable_layers: Optional[int]) -> None:
        # trainable_layers=None => keep default (train all); >=0 => freeze all then unfreeze last N
        if trainable_layers is None:
            return
        for param in self.encoder.parameters():
            param.requires_grad = False
        if trainable_layers > 0:
            for layer in self.encoder.encoder.layer[-trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        species_ids: torch.Tensor,
        task_ids: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs)
        hidden_states = outputs["last_hidden_state"] if isinstance(outputs, dict) else outputs[0]

        if self.use_middle_token_only:
            index = min(self.middle_token_index, hidden_states.size(1) - 1)
            pooled = hidden_states[:, index, :]
        else:
            pooled = hidden_states[:, 0, :]

        if self.use_prompt_routing:
            gating_input = pooled + self.species_embeddings(species_ids) + self.task_embeddings(task_ids)
        else:
            gating_input = pooled
        gating_logits = self.gating(gating_input)
        gating_probs = torch.softmax(gating_logits, dim=-1)

        if self.top_k is not None and self.top_k < self.num_experts:
            topk_values, topk_indices = torch.topk(gating_probs, self.top_k, dim=-1)
            mask = gating_probs.new_zeros(gating_probs.shape)
            mask.scatter_(1, topk_indices, 1.0)
            gating_probs = gating_probs * mask
            gating_probs = gating_probs / (gating_probs.sum(dim=-1, keepdim=True) + 1e-8)

        expert_logits = []
        for idx in range(self.num_experts):
            adapted = pooled + self.adapters[idx](pooled)
            expert_logits.append(self.detect_experts[idx](adapted).unsqueeze(1))
        expert_logits = torch.cat(expert_logits, dim=1)
        detect_logits = torch.sum(expert_logits * gating_probs.unsqueeze(-1), dim=1)

        return {
            "detect_logits": detect_logits,
            "gating_logits": gating_logits,
        }

    def _compute_losses(
        self,
        detect_logits: torch.Tensor,
        detect_labels: torch.Tensor,
        gating_logits: torch.Tensor,
        expert_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        detect_loss = F.cross_entropy(detect_logits, detect_labels, weight=self.detect_class_weights)
        gating_loss = F.cross_entropy(gating_logits, expert_labels)
        total = self.loss_weights.classify * detect_loss + self.loss_weights.gating * gating_loss
        return {"total": total, "detect": detect_loss, "gating": gating_loss}

    def _log_metrics(
        self,
        prefix: str,
        detect_logits: torch.Tensor,
        detect_labels: torch.Tensor,
        gating_logits: torch.Tensor,
        expert_labels: torch.Tensor,
    ) -> None:
        detect_probs = torch.softmax(detect_logits, dim=-1)
        gating_probs = torch.softmax(gating_logits, dim=-1)
        self.detect_accuracy(detect_probs[:, 1], detect_labels)
        self.detect_f1(detect_probs[:, 1], detect_labels)
        self.gating_accuracy(gating_probs, expert_labels)

        self.log(f"{prefix}/detect_acc_epoch", self.detect_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}/detect_f1_epoch", self.detect_f1, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}/gating_acc_epoch", self.gating_accuracy, on_step=False, on_epoch=True, sync_dist=True)

        detect_step_acc = (detect_probs.argmax(dim=-1) == detect_labels).float().mean()
        gating_step_acc = (gating_probs.argmax(dim=-1) == expert_labels).float().mean()
        self.log(f"{prefix}/detect_acc_step", detect_step_acc, prog_bar=(prefix == "train"), on_step=True, on_epoch=False, sync_dist=True)
        self.log(f"{prefix}/gating_acc_step", gating_step_acc, on_step=True, on_epoch=False, sync_dist=True)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            species_ids=batch["species_ids"],
            task_ids=batch["task_ids"],
        )
        losses = self._compute_losses(outputs["detect_logits"], batch["detect_labels"], outputs["gating_logits"], batch["expert_labels"])
        self.log("train/loss", losses["total"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss_detect", losses["detect"], on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/loss_gating", losses["gating"], on_step=True, on_epoch=False, sync_dist=True)
        self._log_metrics("train", outputs["detect_logits"], batch["detect_labels"], outputs["gating_logits"], batch["expert_labels"])
        return losses["total"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            species_ids=batch["species_ids"],
            task_ids=batch["task_ids"],
        )
        losses = self._compute_losses(outputs["detect_logits"], batch["detect_labels"], outputs["gating_logits"], batch["expert_labels"])
        self.log("val/loss", losses["total"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._log_metrics("val", outputs["detect_logits"], batch["detect_labels"], outputs["gating_logits"], batch["expert_labels"])
        return losses["total"]

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            species_ids=batch["species_ids"],
            task_ids=batch["task_ids"],
        )
        losses = self._compute_losses(outputs["detect_logits"], batch["detect_labels"], outputs["gating_logits"], batch["expert_labels"])
        self.log("test/loss", losses["total"], on_step=False, on_epoch=True, sync_dist=True)
        self._log_metrics("test", outputs["detect_logits"], batch["detect_labels"], outputs["gating_logits"], batch["expert_labels"])
        return losses["total"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(warmup_steps, 0),
            num_training_steps=max(total_steps, 1),
        )
        scheduler_config = {"scheduler": scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
