"""Export SemPOI-RL predictions through a city-independent planning seam."""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "cross-city-profile/v1"
MODEL_STAGES = {"base", "sft", "rl", "unavailable"}
STYLE_SOURCES = {"model_generated", "destination_reference", "unavailable"}


def resolve_model_stage(
    *,
    use_llm: bool,
    use_lora: bool,
    secondary_lora_path: str | None,
    use_vllm: bool = False,
) -> str:
    """Label the style generator used for a profile without relying on filenames."""
    if not use_llm:
        return "unavailable"
    if not use_lora:
        return "base"
    if use_vllm:
        raise ValueError(
            "vLLM profile exports cannot be labeled SFT/RL because this project "
            "does not load PEFT adapters through the vLLM backend."
        )
    if secondary_lora_path:
        return "rl"
    return "sft"


def build_cross_city_profile(
    *,
    user_id: int | str,
    dataset_name: str,
    origin_city: str,
    destination_city: str,
    model_stage: str,
    predicted_poi_ids: Sequence[int],
    poi_meta: Mapping[int | str, Mapping[str, Any]],
    travel_style: str | None = None,
    style_source: str = "unavailable",
    sequence_checkpoint: str | None = None,
    style_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Build a portable profile without leaking destination ground truth.

    Source POI identifiers are retained only for traceability. The global travel
    style is the city-independent bridge to a destination planner. The predicted
    category sequence is destination-conditioned and should only be enabled as a
    labelled ablation, because POI identifier spaces are not shared with TRIP.
    """
    if model_stage not in MODEL_STAGES:
        raise ValueError(
            f"model_stage must be one of {sorted(MODEL_STAGES)}, got {model_stage!r}."
        )
    if style_source not in STYLE_SOURCES:
        raise ValueError(
            f"style_source must be one of {sorted(STYLE_SOURCES)}, "
            f"got {style_source!r}."
        )

    candidates = []
    categories = []
    valid_poi_ids = [int(poi_id) for poi_id in predicted_poi_ids if int(poi_id)]
    last_position = len(valid_poi_ids) - 1
    for position, numeric_id in enumerate(valid_poi_ids):
        metadata = poi_meta.get(numeric_id) or poi_meta.get(str(numeric_id)) or {}
        category = metadata.get("main_category") or "Unknown"
        is_query_boundary = position in {0, last_position}
        if not is_query_boundary:
            categories.append(category)
        candidates.append(
            {
                "rank": len(candidates) + 1,
                "source_poi_id": numeric_id,
                "role": (
                    "query_boundary" if is_query_boundary else "predicted_middle"
                ),
                "external_id": metadata.get("bid"),
                "category": category,
                "region": metadata.get("main_region"),
                "latitude": metadata.get("lat"),
                "longitude": metadata.get("lon"),
            }
        )

    category_counts = Counter(categories)
    total = len(categories)
    ordered_categories = sorted(
        category_counts,
        key=lambda category: (-category_counts[category], categories.index(category)),
    )
    category_weights = [
        {
            "category": category,
            "count": category_counts[category],
            "weight": round(category_counts[category] / total, 6),
        }
        for category in ordered_categories
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "user_id": user_id,
        "source": {
            "producer": "SemPOI-RL",
            "dataset": dataset_name,
            "origin_city": origin_city,
            "destination_city": destination_city,
            "model_stage": model_stage,
            "sequence_checkpoint": sequence_checkpoint,
            "style_checkpoint": style_checkpoint,
            "style_source": style_source,
        },
        "travel_style": {"summary": travel_style},
        "preferences": {
            "category_sequence": categories,
            "category_weights": category_weights,
            "candidate_count": total,
            "source": "destination_conditioned_sequence_prediction",
            "uses_destination_endpoints": True,
            "uses_destination_length": True,
        },
        "candidate_pois": candidates,
        "planning_policy": {
            "strength": "soft",
            "explicit_request_precedence": True,
            "hard_constraint_precedence": True,
            "destination_sequence_default": "diagnostic_only",
        },
    }


def write_cross_city_profile(
    profile: Mapping[str, Any], output_path: str | Path
) -> Path:
    """Persist one profile and return its resolved output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path.resolve()
