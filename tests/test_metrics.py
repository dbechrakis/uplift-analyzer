import json
import math

import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from core.metrics import average_treatment_effect, relative_uplift
from core.segments import uplift_by_segment
from core.srm import sample_ratio_check


def two_arm(n_t, x_t, n_c, x_c, outcome="conversion"):
    treatment = [1] * n_t + [0] * n_c
    y = [1] * x_t + [0] * (n_t - x_t) + [1] * x_c + [0] * (n_c - x_c)
    df = pd.DataFrame({"treatment": treatment, outcome: y})
    for i in range(12):
        df[f"f{i}"] = 0.0
    for col in ("conversion", "visit", "exposure"):
        if col not in df:
            df[col] = 0
    return df


def test_ate_known_answer():
    df = two_arm(100, 20, 100, 10)
    r = average_treatment_effect(df)
    assert r.ate == pytest.approx(0.1)
    assert r.std_error == pytest.approx(0.05)
    z = stats.norm.ppf(0.975)
    assert r.ci_low == pytest.approx(0.1 - z * 0.05)
    assert r.ci_high == pytest.approx(0.1 + z * 0.05)
    assert r.n_treated == 100 and r.n_control == 100
    assert r.confidence_level == pytest.approx(0.95)


def test_ate_p_value_matches_statsmodels():
    df = two_arm(300, 60, 200, 25)
    r = average_treatment_effect(df)
    z, p = proportions_ztest([60, 25], [300, 200])
    assert r.z_statistic == pytest.approx(z)
    assert r.p_value == pytest.approx(p)


def test_ate_no_difference_gives_p_one():
    r = average_treatment_effect(two_arm(100, 10, 100, 10))
    assert r.ate == 0
    assert r.p_value == pytest.approx(1.0)


def test_ate_zero_variance_does_not_crash():
    r = average_treatment_effect(two_arm(50, 0, 50, 0))
    assert r.p_value == 1.0 and r.z_statistic == 0.0


def test_ate_other_outcome_and_alpha():
    df = two_arm(100, 30, 100, 10, outcome="visit")
    r = average_treatment_effect(df, outcome="visit", alpha=0.01)
    assert r.outcome == "visit"
    assert r.confidence_level == pytest.approx(0.99)
    assert r.ate == pytest.approx(0.2)


def test_relative_uplift_known_answer():
    r = relative_uplift(two_arm(100, 20, 100, 10))
    assert r.relative_uplift == pytest.approx(1.0)
    se = math.sqrt(0.8 / 20 + 0.9 / 10)
    z = stats.norm.ppf(0.975)
    assert r.ci_low == pytest.approx(2 * math.exp(-z * se) - 1)
    assert r.ci_high == pytest.approx(2 * math.exp(z * se) - 1)


def test_relative_uplift_degenerate_cases():
    assert relative_uplift(two_arm(100, 5, 100, 0)).relative_uplift is None
    r = relative_uplift(two_arm(100, 0, 100, 5))
    assert r.relative_uplift == pytest.approx(-1.0) and r.ci_low is None


@pytest.mark.parametrize("bad", ["clicks", ""])
def test_invalid_outcome_rejected(bad):
    with pytest.raises(ValueError):
        average_treatment_effect(two_arm(10, 1, 10, 1), outcome=bad)


def test_missing_arm_rejected():
    df = two_arm(10, 1, 10, 1)
    with pytest.raises(ValueError):
        average_treatment_effect(df[df["treatment"] == 1])


def test_invalid_alpha_rejected():
    with pytest.raises(ValueError):
        average_treatment_effect(two_arm(10, 1, 10, 1), alpha=1.5)


def test_data_source_label_propagates():
    df = two_arm(10, 1, 10, 1)
    assert average_treatment_effect(df).data_source == "unspecified"
    assert average_treatment_effect(df, data_source="constructed").data_source == "constructed"
    df.attrs["data_source"] = "synthetic"
    assert average_treatment_effect(df).data_source == "synthetic"
    with pytest.raises(ValueError):
        average_treatment_effect(df, data_source="made-up")


def segment_frame():
    rows = []
    for half, (conv_t, conv_c) in enumerate([(40, 40), (120, 20)]):
        for arm, conv in ((1, conv_t), (0, conv_c)):
            for i in range(200):
                rows.append(
                    {
                        "f0": half * 100 + len(rows) % 100,
                        "treatment": arm,
                        "conversion": int(i < conv),
                    }
                )
    df = pd.DataFrame(rows)
    for i in range(1, 12):
        df[f"f{i}"] = 0.0
    df["visit"] = 0
    df["exposure"] = 0
    return df


def test_segments_known_answer():
    r = uplift_by_segment(segment_frame(), "f0", n_bins=2)
    low, high = r.segments
    assert low.ate == pytest.approx(0.0) and not low.significant
    assert high.ate == pytest.approx(0.5) and high.significant
    assert low.n + high.n == 800
    assert low.n_treated == 200 and low.n_control == 200


def test_segments_bh_adjustment_is_conservative(synthetic_df):
    r = uplift_by_segment(synthetic_df, "f0", n_bins=5)
    for s in r.segments:
        assert s.p_value_adjusted >= s.p_value
        assert 0 <= s.p_value_adjusted <= 1
    assert r.adjustment.startswith("Benjamini")


def test_segments_finds_the_planted_heterogeneity(synthetic_df):
    r = uplift_by_segment(synthetic_df, "f0", n_bins=4)
    assert r.segments[-1].ate > r.segments[0].ate


def test_segments_bin_missing_an_arm_is_not_tested():
    df = segment_frame()
    df["treatment"] = (df["f0"] >= 100).astype(int)
    r = uplift_by_segment(df, "f0", n_bins=2)
    assert all(s.ate is None and s.p_value_adjusted is None for s in r.segments)


def test_segments_rejects_bad_input(synthetic_df):
    with pytest.raises(ValueError):
        uplift_by_segment(synthetic_df, "not_a_feature")
    with pytest.raises(ValueError):
        uplift_by_segment(synthetic_df, "f0", n_bins=1)


def test_srm_known_chi_square():
    df = pd.DataFrame({"treatment": [1] * 900 + [0] * 100})
    r = sample_ratio_check(df, expected_share_treated=0.85)
    assert r.chi2_statistic == pytest.approx(50**2 / 850 + 50**2 / 150)
    assert r.mismatch is True
    assert r.observed_share_treated == pytest.approx(0.9)


def test_srm_balanced_split_passes():
    df = pd.DataFrame({"treatment": [1] * 850 + [0] * 150})
    r = sample_ratio_check(df)
    assert r.mismatch is False and r.p_value == pytest.approx(1.0)


def test_srm_custom_expected_share():
    df = pd.DataFrame({"treatment": [1] * 500 + [0] * 500})
    assert sample_ratio_check(df, expected_share_treated=0.5).mismatch is False
    assert sample_ratio_check(df).mismatch is True


def test_srm_validates_inputs():
    df = pd.DataFrame({"treatment": [1, 0]})
    with pytest.raises(ValueError):
        sample_ratio_check(df, expected_share_treated=1.0)
    with pytest.raises(ValueError):
        sample_ratio_check(df.iloc[0:0])


def test_all_results_are_json_serializable(synthetic_df):
    small = synthetic_df.head(5000)
    for result in (
        average_treatment_effect(small),
        relative_uplift(small),
        uplift_by_segment(small, "f1"),
        sample_ratio_check(small),
    ):
        round_trip = json.loads(result.to_json())
        assert round_trip == json.loads(json.dumps(result.to_dict()))
