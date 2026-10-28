import json

import pytest

import core.export_demo as export_demo
import core.validate as validate
from core.adjusted import regression_adjusted_effect
from core.report import build_report
from core.synthetic import generate_synthetic


@pytest.fixture(scope="module")
def small():
    return generate_synthetic(n=12_000, seed=31)


def test_chunked_full_data_adjustment_matches_the_core_function(small, monkeypatch):
    monkeypatch.setattr(validate, "ADJUST_CHUNK", 2_500)
    chunked = validate.full_adjusted_effects(small, ("conversion", "visit"))
    for outcome in ("conversion", "visit"):
        direct = regression_adjusted_effect(small, outcome)
        assert chunked[outcome]["adjusted_ate"] == pytest.approx(direct.adjusted_ate, abs=1e-9)
        assert chunked[outcome]["std_error"] == pytest.approx(direct.std_error, rel=1e-6)


def test_full_data_checks_report_structure(small):
    checks = validate.full_data_checks(small)
    assert checks["rows"] == len(small)
    assert checks["control_exposure_rate"] == 0
    assert set(checks["smd"]) == set(f"f{i}" for i in range(12))
    assert checks["unadjusted"]["conversion"]["ate"] > 0
    assert checks["cace"]["visit"]["one_sided"] is True
    json.dumps(checks)


def test_sample_coverage_counts_hits_and_alarms(small):
    truth = {o: float(small[o].mean()) for o in ("conversion", "visit")}
    result = validate.sample_coverage(small, truth, draws=6, size=3_000, seed=5)
    assert result["draws"] == 6 and result["sample_size"] == 3_000
    assert all(0 <= v <= 1 for v in result["ci_coverage_of_full_data_ate"].values())
    assert 0 <= result["srm_false_alarm_rate"] <= 1
    assert result["estimate_sd"]["visit"] > 0


def test_code_simulations_are_near_nominal_even_when_small():
    result = validate.simulate_code_correctness(seed=3, runs=300, fdr_runs=20)
    assert 0.90 <= result["ate_ci_coverage"] <= 0.99
    assert 0.90 <= result["relative_uplift_ci_coverage"] <= 0.99
    assert result["type_i_error_at_5_percent"] < 0.10
    assert result["fdr_null_runs"] == 20


def test_demo_export_builds_every_section_from_a_loader(small, monkeypatch, tmp_path):
    monkeypatch.setattr(export_demo, "load_criteo_sample", lambda size, seed: small)
    record = tmp_path / "validation.json"
    record.write_text(
        json.dumps(
            {
                "full_data": {
                    "unadjusted": {o: {"ate": 0.1} for o in export_demo.OUTCOMES},
                    "adjusted": {},
                    "cace": {},
                    "max_abs_smd": 0.04,
                },
                "sampling": {"draws": 1},
                "code_simulations": {"runs": 1},
            }
        )
    )
    data = export_demo.build_demo_data(
        12_000, 42, record, qini_bootstrap=5, learner_bootstrap=5, max_iter=5
    )
    assert set(data["outcomes"]) == {"conversion", "visit"}
    visit = data["outcomes"]["visit"]
    assert len(visit["qini"]["curve_x"]) <= 201 and visit["policy_scale"]["n_test"] > 0
    assert set(visit["segments"]) == set(f"f{i}" for i in range(12))
    assert len(visit["learners"]["learners"]) == 4
    assert data["full_data"]["max_abs_smd"] == 0.04 and data["data_source"] == "criteo"
    json.dumps(data)
    missing = export_demo.build_demo_data(
        12_000, 42, tmp_path / "none.json", qini_bootstrap=3, learner_bootstrap=3, max_iter=3
    )
    assert "full_data" not in missing


def test_report_builder_covers_all_core_sections(small):
    report = build_report(small, "visit", "f2", n_bootstrap=5)
    assert set(report) == {"dataset", "sample_ratio", "ate", "relative_uplift", "segments", "qini"}
    json.dumps(report)
