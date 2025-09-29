"""Adaptive benchmark tests with line-by-line explanations."""

from pathlib import Path  # Used to create a temporary file path for the benchmark storage

from swarm_system.learning.adaptive_benchmark import AdaptiveFeedbackBenchmark  # Class under test


def test_adaptive_benchmark_updates(tmp_path):  # tmp_path fixture supplies a temporary directory
    storage = tmp_path / "adaptive.json"  # Decide where the benchmark should persist its state
    bench = AdaptiveFeedbackBenchmark(storage_path=str(storage))  # Instantiate the benchmark pointing at the temp file

    run_payload = {  # Build a minimal fake workflow payload to feed into the benchmark
        "run": {  # Information describing the workflow run itself
            "id": "run-1",  # Unique identifier for the run
            "file_path": "example.py",  # File analysed during the run
            "guidance_history": ["hint one", "hint two"],  # Guidance applied so the benchmark can track memory usage
        },
        "history": {"reward": {"delta": 0.2}},  # Reward history showing improvement across runs
        "iterations": [  # Store a single iteration containing discriminator metrics
            {
                "discriminator_score": {"reward": 0.85},  # Reward score to calculate averages
                "generator_output": {"overall_summary": "demo"},  # Summary field retained for completeness
            }
        ],
        "timestamp": 123.0,  # Timestamp used when persisting the record
    }

    bench.update_from_run(run_payload)  # Feed the payload into the benchmark so it records the run

    assert bench.records[0].file_path == "example.py"  # Confirm the record stored the correct file path
    assert bench.records[0].guidance_count == 2  # Ensure the benchmark counted guidance entries

    challenges = bench.build_challenge_set(top_n=1)  # Request the strongest challenge scenario from the benchmark
    assert challenges[0]["file_path"] == "example.py"  # Validate that the challenge references the expected file

    summary = bench.summarise()  # Fetch summary statistics over all recorded runs
    assert summary["records"] == 1  # Verify that exactly one record is being tracked

