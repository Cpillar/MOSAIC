from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import pytorch_lightning as pl
from transformers import AutoTokenizer


TYPE_GROUPS = {
    "4mC": 0,
    "5hmC": 1,
    "6mA": 2,
}


@dataclass
class Sample:
    sequence: str
    detect_label: int
    type_label: int  # -100 when not applicable
    multi_class_label: int
    dataset_id: int
    species_id: int
    task_id: int


class MultiTaskMethylationDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], tokenizer: AutoTokenizer, max_length: int) -> None:
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        encoded = self.tokenizer(
            sample.sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["detect_labels"] = torch.tensor(sample.detect_label, dtype=torch.long)
        item["type_labels"] = torch.tensor(sample.type_label, dtype=torch.long)
        item["multi_class_labels"] = torch.tensor(sample.multi_class_label, dtype=torch.long)
        item["expert_labels"] = torch.tensor(sample.dataset_id, dtype=torch.long)
        item["species_ids"] = torch.tensor(sample.species_id, dtype=torch.long)
        item["task_ids"] = torch.tensor(sample.task_id, dtype=torch.long)
        return item


class MultiTaskMethylationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: Path,
        groups: Sequence[str],
        model_name: str,
        max_length: int,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int = 0,
        val_ratio: float = 0.1,
        seed: int = 42,
        type_sampling_weights: Sequence[float] | None = None,
        sampler_class_probs: Sequence[float] | None = None,
        include_dataset_labels: bool = True,
        allowed_datasets: Sequence[Tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_root = dataset_root
        self.groups = list(groups)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = max_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed
        self.type_sampling_weights = list(type_sampling_weights) if type_sampling_weights is not None else None
        self.sampler_class_probs = list(sampler_class_probs) if sampler_class_probs is not None else None
        self.include_dataset_labels = include_dataset_labels
        self.allowed_datasets = {(g, d) for g, d in allowed_datasets} if allowed_datasets is not None else None

        self.train_dataset: MultiTaskMethylationDataset | None = None
        self.val_dataset: MultiTaskMethylationDataset | None = None
        self.test_dataset: MultiTaskMethylationDataset | None = None

        self.detect_class_counts: Dict[int, int] = {}
        self.type_class_counts: Dict[int, int] = {}
        self.multi_class_counts: Dict[int, int] = {}
        self._train_sampler: WeightedRandomSampler | None = None
        self.dataset_to_id: Dict[Tuple[str, str], int] = {}
        self.id_to_dataset: Dict[int, Tuple[str, str]] = {}
        self.dataset_metadata: Dict[int, Tuple[int, int]] = {}
        self.species_to_id: Dict[str, int] = {}
        self.id_to_species: Dict[int, str] = {}

    def _load_split(self, split: str) -> List[Sample]:
        samples: List[Sample] = []
        for group_name in self.groups:
            if group_name not in TYPE_GROUPS:
                raise ValueError(f"Unknown group name '{group_name}'. Expected one of {list(TYPE_GROUPS)}.")
            group_dir = self.dataset_root / group_name
            if not group_dir.exists():
                continue
            for dataset_dir in sorted(d for d in group_dir.iterdir() if d.is_dir()):
                if self.allowed_datasets is not None and (group_name, dataset_dir.name) not in self.allowed_datasets:
                    continue
                csv_path = dataset_dir / f"{split}.csv"
                if not csv_path.exists():
                    continue
                dataset_id = self._register_dataset(group_name, dataset_dir.name)
                species_id, task_id = self.dataset_metadata[dataset_id]
                with csv_path.open("r", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        seq = row["sequence"].strip().upper()
                        detect_label = int(row["label"])
                        if detect_label not in {0, 1}:
                            raise ValueError(f"Unexpected label {detect_label} in {csv_path}")
                        if detect_label == 1:
                            type_label = TYPE_GROUPS[group_name]
                            multi_class_label = type_label + 1
                        else:
                            type_label = -100
                            multi_class_label = 0
                        samples.append(
                            Sample(
                                sequence=seq,
                                detect_label=detect_label,
                                type_label=type_label,
                                multi_class_label=multi_class_label,
                                dataset_id=dataset_id,
                                species_id=species_id,
                                task_id=task_id,
                            )
                        )
        return samples

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and stage == "fit":
            return

        train_samples = self._load_split("train")
        random.Random(self.seed).shuffle(train_samples)

        val_size = int(len(train_samples) * self.val_ratio)
        if val_size == 0 and len(train_samples) > 1:
            val_size = 1
        if val_size > 0:
            val_samples = train_samples[:val_size]
            train_samples = train_samples[val_size:]
        else:
            val_samples = []

        self.train_dataset = MultiTaskMethylationDataset(train_samples, self.tokenizer, self.max_length)
        self.val_dataset = MultiTaskMethylationDataset(val_samples, self.tokenizer, self.max_length)

        test_samples = self._load_split("test")
        self.test_dataset = MultiTaskMethylationDataset(test_samples, self.tokenizer, self.max_length)

        self.detect_class_counts = self._count_detect_classes(train_samples)
        self.type_class_counts = self._count_type_classes(train_samples)
        self.multi_class_counts = self._count_multi_class(train_samples)
        self._train_sampler = self._build_train_sampler(train_samples)

    @staticmethod
    def _count_detect_classes(samples: Sequence[Sample]) -> Dict[int, int]:
        counts: Dict[int, int] = {0: 0, 1: 0}
        for sample in samples:
            counts[sample.detect_label] = counts.get(sample.detect_label, 0) + 1
        return counts

    @staticmethod
    def _count_type_classes(samples: Sequence[Sample]) -> Dict[int, int]:
        counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        for sample in samples:
            if sample.type_label >= 0:
                counts[sample.type_label] = counts.get(sample.type_label, 0) + 1
        return counts

    @staticmethod
    def _count_multi_class(samples: Sequence[Sample]) -> Dict[int, int]:
        counts: Dict[int, int] = {idx: 0 for idx in range(len(TYPE_GROUPS) + 1)}
        for sample in samples:
            counts[sample.multi_class_label] = counts.get(sample.multi_class_label, 0) + 1
        return counts

    def _build_train_sampler(self, samples: Sequence[Sample]) -> WeightedRandomSampler | None:
        if not samples:
            return None
        weights: List[float] = []
        if self.sampler_class_probs is not None:
            if len(self.sampler_class_probs) != len(TYPE_GROUPS) + 1:
                raise ValueError(
                    f"sampler_class_probs length {len(self.sampler_class_probs)} does not match number of multi-class labels {len(TYPE_GROUPS) + 1}."
                )
            for sample in samples:
                weights.append(float(self.sampler_class_probs[sample.multi_class_label]))
            return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

        if self.type_sampling_weights is None:
            return None
        if len(self.type_sampling_weights) != len(TYPE_GROUPS):
            raise ValueError(
                f"type_sampling_weights length {len(self.type_sampling_weights)} does not match number of type classes {len(TYPE_GROUPS)}."
            )
        for sample in samples:
            if sample.detect_label == 0 or sample.type_label < 0:
                weights.append(1.0)
            else:
                weights.append(float(self.type_sampling_weights[sample.type_label]))
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    def _register_dataset(self, group_name: str, dataset_name: str) -> int:
        key = (group_name, dataset_name)
        if key not in self.dataset_to_id:
            dataset_id = len(self.dataset_to_id)
            self.dataset_to_id[key] = dataset_id
            self.id_to_dataset[dataset_id] = key
            species_name = self._normalize_species_name(group_name, dataset_name)
            species_id = self._register_species(species_name)
            task_id = TYPE_GROUPS[group_name]
            self.dataset_metadata[dataset_id] = (species_id, task_id)
        return self.dataset_to_id[key]

    def _register_species(self, species_name: str) -> int:
        if species_name not in self.species_to_id:
            species_id = len(self.species_to_id)
            self.species_to_id[species_name] = species_id
            self.id_to_species[species_id] = species_name
        return self.species_to_id[species_name]

    @staticmethod
    def _normalize_species_name(group_name: str, dataset_name: str) -> str:
        prefix = f"{group_name}_"
        if dataset_name.startswith(prefix):
            return dataset_name[len(prefix) :]
        return dataset_name

    @property
    def num_experts(self) -> int:
        return len(self.dataset_to_id)

    def get_expert_name(self, expert_id: int) -> str:
        group_name, dataset_name = self.id_to_dataset[expert_id]
        return f"{group_name}/{dataset_name}"

    @property
    def num_species(self) -> int:
        return len(self.species_to_id)

    @property
    def num_tasks(self) -> int:
        return len(TYPE_GROUPS)

    def get_species_name(self, species_id: int) -> str:
        return self.id_to_species[species_id]

    def get_dataset_metadata(self, expert_id: int) -> Tuple[int, int]:
        return self.dataset_metadata[expert_id]

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule has not been set up. Call setup() before requesting dataloaders.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self._train_sampler is None,
            sampler=self._train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DataModule has not been set up. Call setup() before requesting dataloaders.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("DataModule has not been set up. Call setup() before requesting dataloaders.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
