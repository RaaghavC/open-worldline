"""Independent aggregate path checks using measured-score-shaped CPU fixtures."""
import copy
import importlib.util
from pathlib import Path

from experiments.room_world.memory_evaluate import compare_evaluations


def fixture():
    source = Path(__file__).with_name("test_room_memory_independent.py")
    spec = importlib.util.spec_from_file_location("independent_memory_score_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.measured_score_fixture()[0]


def score(reports):
    return compare_evaluations(reports, matched_budgets=True,
        visibility_audit={"status": "verified", "capture_manifest_sha256": "2" * 64})


def test_mixed_seed_training_paths_cannot_produce_aggregate_gates():
    reports = fixture()
    assert score(reports)["status"] == "complete"
    reports["20260908-carry"]["provenance"]["sequence_path"] = "batched"
    result = score(reports)
    assert result["status"] == "unavailable"
    assert not result["gates_available"]
    assert any("sequence paths differ" in reason for reason in result["reasons"])


def test_consistent_batched_labels_preserve_all_legacy_score_arithmetic():
    legacy = fixture()
    batched = copy.deepcopy(legacy)
    for key, value in batched.items():
        value["provenance"]["sequence_path"] = None if key == "frozen" else "batched"
        value["provenance"]["training_sequence_path"] = "batched"
    assert score(batched) == score(legacy)


def test_frozen_reference_from_a_different_study_path_is_rejected():
    reports = fixture()
    reports["frozen"]["provenance"]["training_sequence_path"] = "batched"
    result = score(reports)
    assert result["status"] == "unavailable"
    assert not result["gates_available"]
    assert any("study sequence path differs" in reason for reason in result["reasons"])
