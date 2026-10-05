import json
import tempfile
import unittest
from pathlib import Path

from cross_city_profile import (
    build_cross_city_profile,
    resolve_model_stage,
    write_cross_city_profile,
)


class CrossCityProfileTest(unittest.TestCase):
    def test_profile_translates_ranked_pois_into_city_independent_preferences(self):
        poi_meta = {
            7: {
                "bid": "poi-7",
                "lat": 30.1,
                "lon": 120.1,
                "main_region": "Hangzhou",
                "main_category": "Park",
            },
            8: {
                "bid": "poi-8",
                "lat": 30.2,
                "lon": 120.2,
                "main_region": "Hangzhou",
                "main_category": "Museum",
            },
        }

        profile = build_cross_city_profile(
            user_id=42,
            dataset_name="Foursquare",
            origin_city="New York",
            destination_city="Hangzhou",
            model_stage="rl",
            predicted_poi_ids=[7, 8, 7, 0],
            poi_meta=poi_meta,
            travel_style="Prefers green space and a relaxed cultural route.",
            style_source="model_generated",
            sequence_checkpoint="model_best.xhr",
            style_checkpoint="rl-checkpoint",
        )

        self.assertEqual(profile["schema_version"], "cross-city-profile/v1")
        self.assertEqual(
            profile["preferences"]["category_sequence"],
            ["Museum"],
        )
        self.assertEqual(
            profile["preferences"]["category_weights"],
            [
                {"category": "Museum", "count": 1, "weight": 1.0},
            ],
        )
        self.assertEqual(
            [candidate["source_poi_id"] for candidate in profile["candidate_pois"]],
            [7, 8, 7],
        )
        self.assertNotIn("target_poi_ids", profile)
        self.assertEqual(
            [candidate["role"] for candidate in profile["candidate_pois"]],
            ["query_boundary", "predicted_middle", "query_boundary"],
        )
        self.assertEqual(
            profile["preferences"]["source"],
            "destination_conditioned_sequence_prediction",
        )
        self.assertEqual(profile["source"]["model_stage"], "rl")
        self.assertEqual(profile["source"]["style_checkpoint"], "rl-checkpoint")

    def test_profile_is_written_as_utf8_json_with_parent_directory(self):
        profile = build_cross_city_profile(
            user_id="用户-1",
            dataset_name="Yelp",
            origin_city="Phoenix",
            destination_city="Las Vegas",
            model_stage="base",
            predicted_poi_ids=[],
            poi_meta={},
            travel_style="偏好轻松的美食体验",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "profiles" / "用户-1.json"
            write_cross_city_profile(profile, output_path)

            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved, profile)

    def test_unknown_model_stage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "model_stage"):
            build_cross_city_profile(
                user_id=1,
                dataset_name="Yelp",
                origin_city="Phoenix",
                destination_city="Las Vegas",
                model_stage="final",
                predicted_poi_ids=[],
                poi_meta={},
            )

    def test_model_stage_is_derived_from_loaded_style_model(self):
        self.assertEqual(
            resolve_model_stage(
                use_llm=True, use_lora=False, secondary_lora_path=None
            ),
            "base",
        )
        self.assertEqual(
            resolve_model_stage(
                use_llm=True, use_lora=True, secondary_lora_path=None
            ),
            "sft",
        )
        self.assertEqual(
            resolve_model_stage(
                use_llm=True, use_lora=True, secondary_lora_path="rl-checkpoint"
            ),
            "rl",
        )
        self.assertEqual(
            resolve_model_stage(
                use_llm=False, use_lora=False, secondary_lora_path=None
            ),
            "unavailable",
        )

    def test_vllm_lora_stage_is_rejected_instead_of_mislabeled(self):
        with self.assertRaisesRegex(ValueError, "vLLM"):
            resolve_model_stage(
                use_llm=True,
                use_lora=True,
                secondary_lora_path="rl-checkpoint",
                use_vllm=True,
            )


if __name__ == "__main__":
    unittest.main()
