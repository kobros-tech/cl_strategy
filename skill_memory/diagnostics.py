"""Optional, intentionally expensive diagnostics for anonymous routing."""

from __future__ import annotations

from collections import Counter
from typing import Any


def routing_rank_diagnostics(routes: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute rank and confusion diagnostics from retained route records.

    This helper is deliberately separate from evaluation so normal routing does
    not need to compute or retain diagnostic aggregates. Call it only when
    ``diagnose=True`` and route history is available.
    """
    if not routes:
        return {
            "samples": 0,
            "top1_accuracy": 0.0,
            "top2_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "mean_reciprocal_rank": 0.0,
            "mean_correct_class_rank": 0.0,
        }

    ranks: list[int] = []
    confusion = Counter()
    for route in routes:
        true_class = route.get("evaluation_y")
        candidates = route.get("candidates", [])
        if true_class is None or not candidates:
            continue
        ordered = sorted(
            candidates,
            key=lambda item: item.get("score", float("-inf")),
            reverse=True,
        )
        rank = next(
            (
                index
                for index, candidate in enumerate(ordered, start=1)
                if int(candidate.get("class", -1)) == int(true_class)
            ),
            len(ordered) + 1,
        )
        ranks.append(rank)
        predicted = int(ordered[0].get("class", -1))
        if predicted != int(true_class):
            confusion[(int(true_class), predicted)] += 1

    if not ranks:
        return {"samples": 0}

    n = len(ranks)
    return {
        "samples": n,
        "top1_accuracy": sum(rank <= 1 for rank in ranks) / n,
        "top2_accuracy": sum(rank <= 2 for rank in ranks) / n,
        "top3_accuracy": sum(rank <= 3 for rank in ranks) / n,
        "mean_reciprocal_rank": sum(1.0 / rank for rank in ranks) / n,
        "mean_correct_class_rank": sum(ranks) / n,
        "confusion_pairs": [
            {"true_class": true_class, "wrong_class": wrong_class, "count": count}
            for (true_class, wrong_class), count in confusion.most_common()
        ],
    }
