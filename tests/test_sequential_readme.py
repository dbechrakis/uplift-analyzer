from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORD = json.loads((ROOT / "outputs" / "sequential.json").read_text())
README = (ROOT / "README.md").read_text()
VALIDATION = (ROOT / "VALIDATION.md").read_text()
LABELS = ("fixed", "peek", "bonferroni", "pocock", "obf", "msprt")


def pick(rule, looks, lift=0.0, adjustment="none", outcome="visit"):
    for row in RECORD["rows"]:
        if (
            row["rule"] == rule
            and row["n_looks"] == looks
            and row["injected_lift"] == lift
            and row["adjustment"] == adjustment
            and row["outcome"] == outcome
        ):
            return row
    raise KeyError((rule, looks, lift))


def pct(value):
    return f"{value * 100:.1f}%"


@pytest.mark.parametrize("rule", LABELS)
def test_readme_table_row_matches_the_results_file(rule):
    cells = [pct(pick(rule, k)["rejection_rate"]) for k in (5, 20, 200)]
    cells.append(pct(pick(rule, 20, 0.05)["rejection_rate"]))
    expected = f"| `{rule}` | " + " | ".join(cells) + " |"
    assert expected in README


def test_readme_real_replay_rows_match():
    replay = RECORD["real_replay"]
    for row in replay["rows"]:
        expected = (
            f"| `{row['rule']}` | {row['look']} of 20 | {row['share_of_sample'] * 100:.0f}% | "
            f"{row['estimate'] * 100:+.2f} pp |"
        )
        assert expected in README
    assert f"{replay['final_estimate'] * 100:+.2f} pp" in README


def test_validation_grades_match_the_results_file():
    for name, graded in RECORD["expectations"].items():
        status = {True: "Passed", False: "Failed", None: "Not graded"}[graded["passed"]]
        assert f"- {name}. {status}." in VALIDATION


def test_results_file_follows_the_design():
    assert RECORD["data_source"] == "criteo"
    assert RECORD["n_arm"] == 50_000
    assert RECORD["null_replications"] == 5_000
    assert RECORD["lift_replications"] == 2_000
    assert RECORD["seed"] == 42
    assert RECORD["look_counts"] == [5, 20, 200]
    assert {row["data_source"] for row in RECORD["rows"]} == {"criteo", "constructed"}
    assert RECORD["real_replay"]["sample_size"] == 500_000
