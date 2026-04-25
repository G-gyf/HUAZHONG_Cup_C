from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parent
SCENARIO_NAMES: tuple[str, ...] = (
    "speed_scale_090",
    "speed_scale_095",
    "speed_scale_100",
    "speed_scale_105",
    "speed_scale_110",
)
SEEDS: tuple[int, ...] = (11, 17, 23)


def run_command(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def main() -> None:
    start_time = time.time()
    run_command([sys.executable, "build_speed_scenarios.py"], ANALYSIS_DIR)
    run_records: list[dict[str, object]] = []
    for scenario_name in SCENARIO_NAMES:
        input_root = ANALYSIS_DIR / "inputs" / scenario_name / "preprocess_artifacts"
        for seed in SEEDS:
            output_root = ANALYSIS_DIR / "outputs" / scenario_name / f"seed_{seed}"
            output_root.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "solve_question1_speed_sensitivity.py",
                "--workspace",
                str(ANALYSIS_DIR),
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--seed-list",
                str(seed),
                "--max-generations",
                "2",
                "--particle-count",
                "1",
                "--top-route-candidates",
                "8",
                "--analysis-scenario",
                scenario_name,
            ]
            step_start = time.time()
            run_command(command, ANALYSIS_DIR)
            run_records.append(
                {
                    "scenario_name": scenario_name,
                    "seed": seed,
                    "input_root": str(input_root),
                    "output_root": str(output_root),
                    "elapsed_sec": time.time() - step_start,
                }
            )
    total_elapsed = time.time() - start_time
    run_log_path = ANALYSIS_DIR / "run_log.json"
    run_log_path.write_text(
        json.dumps(
            {
                "scenario_count": len(SCENARIO_NAMES),
                "seed_count": len(SEEDS),
                "run_count": len(run_records),
                "total_elapsed_sec": total_elapsed,
                "runs": run_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_count": len(run_records),
                "total_elapsed_sec": total_elapsed,
                "run_log": str(run_log_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
