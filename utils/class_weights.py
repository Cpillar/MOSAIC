from __future__ import annotations

from typing import Dict, List


def inverse_frequency_weights(class_counts: Dict[int, int], num_classes: int) -> List[float]:
    total = sum(class_counts.get(i, 0) for i in range(num_classes))
    if total == 0:
        return [1.0] * num_classes
    weights = []
    for idx in range(num_classes):
        count = class_counts.get(idx, 0)
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))
    return weights
