from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from preprocess_logistics_data import CSV_ENCODING, compute_travel_time_lookup  # noqa: E402


SCENARIO_SCALES: tuple[float, ...] = (0.90, 0.95, 1.00, 1.05, 1.10)


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_name: str
    speed_scale: float


def analysis_root() -> Path:
    return Path(__file__).resolve().parent


def baseline_input_root() -> Path:
    return ROOT_DIR / "preprocess_artifacts"


def scenario_specs() -> list[ScenarioSpec]:
    return [ScenarioSpec(f"speed_scale_{int(scale * 100):03d}", scale) for scale in SCENARIO_SCALES]


def rebuild_speed_profile(speed_profile_path: Path, scale: float) -> pd.DataFrame:
    speed_df = pd.read_csv(speed_profile_path)
    speed_df["speed_kmh"] = speed_df["speed_kmh"].astype(float) * scale
    speed_df["speed_km_per_min"] = speed_df["speed_kmh"] / 60.0
    speed_df.to_csv(speed_profile_path, index=False, encoding=CSV_ENCODING)
    return speed_df


def rebuild_travel_time_lookup(input_root: Path, speed_df: pd.DataFrame) -> None:
    distance_df = pd.read_csv(input_root / "tables" / "distance_matrix_clean.csv")
    travel_time_lookup, departure_minutes, node_ids = compute_travel_time_lookup(distance_df, speed_df)
    np.savez_compressed(
        input_root / "travel_time_lookup.npz",
        travel_time_minutes=travel_time_lookup.astype(np.float32),
        departure_minutes=departure_minutes.astype(np.int16),
        node_ids=node_ids.astype(np.int16),
    )


def build_scenarios() -> list[dict[str, object]]:
    scenario_input_root = analysis_root() / "inputs"
    scenario_input_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for spec in scenario_specs():
        target_root = scenario_input_root / spec.scenario_name / "preprocess_artifacts"
        if target_root.exists():
            shutil.rmtree(target_root)
        shutil.copytree(baseline_input_root(), target_root)
        speed_profile_path = target_root / "tables" / "speed_profile.csv"
        speed_df = rebuild_speed_profile(speed_profile_path, spec.speed_scale)
        rebuild_travel_time_lookup(target_root, speed_df)
        row = {
            "scenario_name": spec.scenario_name,
            "speed_scale": spec.speed_scale,
            "input_root": str(target_root),
            "min_speed_kmh": float(speed_df["speed_kmh"].min()),
            "max_speed_kmh": float(speed_df["speed_kmh"].max()),
        }
        manifest_rows.append(row)
    pd.DataFrame(manifest_rows).to_csv(
        analysis_root() / "scenario_manifest.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    (analysis_root() / "scenario_manifest.json").write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_rows


def main() -> None:
    manifest_rows = build_scenarios()
    print(
        json.dumps(
            {
                "scenario_count": len(manifest_rows),
                "scenarios": [row["scenario_name"] for row in manifest_rows],
                "analysis_root": str(analysis_root()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
