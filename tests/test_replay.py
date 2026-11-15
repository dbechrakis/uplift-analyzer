from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from core.data import DatasetIntegrityError, _read_control_pool, load_control_pool
from core.replay import (
    cumulative_sums,
    draw_experiment,
    eligible_rows,
    holdout_rows,
    inject_lift,
    look_sizes,
)


def test_arms_never_share_a_row_and_holdout_is_never_drawn():
    n_pool = 5_000
    eligible = eligible_rows(n_pool)
    holdout = set(holdout_rows(n_pool).tolist())
    for seed in np.random.SeedSequence(42).spawn(20):
        a, b = draw_experiment(eligible, 500, seed)
        assert len(set(a) & set(b)) == 0
        assert len(set(a) | set(b)) == 1_000
        assert not (set(a) | set(b)) & holdout


def test_draw_is_reproducible_and_checks_size():
    eligible = eligible_rows(1_000)
    a1, b1 = draw_experiment(eligible, 100, np.random.SeedSequence(1))
    a2, b2 = draw_experiment(eligible, 100, np.random.SeedSequence(1))
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    with pytest.raises(ValueError):
        draw_experiment(eligible, 500, np.random.SeedSequence(1))


def test_inject_lift_hits_the_stated_lift():
    rng = np.random.default_rng(3)
    base = 0.05
    outcome = (rng.random(4_000_000) < base).astype(np.int8)
    lifted = inject_lift(outcome, 0.10, base, rng)
    assert lifted.mean() / outcome.mean() - 1 == pytest.approx(0.10, abs=0.01)
    assert np.all(lifted >= outcome)
    with pytest.raises(ValueError):
        inject_lift(outcome, 0.1, 0.0, rng)
    with pytest.raises(ValueError):
        inject_lift(outcome, 50.0, 0.5, rng)


def test_changing_rows_after_a_look_cannot_change_that_look():
    rng = np.random.default_rng(4)
    outcome = (rng.random(1_000) < 0.2).astype(np.int8)
    looks = look_sizes(1_000, 10)
    before = cumulative_sums(outcome, looks)
    changed = outcome.copy()
    changed[500:] = 1 - changed[500:]
    after = cumulative_sums(changed, looks)
    assert np.array_equal(before[0][:5], after[0][:5])
    assert np.array_equal(before[1][:5], after[1][:5])
    assert not np.array_equal(before[0][5:], after[0][5:])


def test_look_sizes_must_divide():
    assert look_sizes(100, 4).tolist() == [25, 50, 75, 100]
    with pytest.raises(ValueError):
        look_sizes(100, 3)


def make_file(path, rows):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(rows, 12)), columns=[f"f{i}" for i in range(12)])
    frame["treatment"] = (rng.random(rows) < 0.85).astype(int)
    frame["conversion"] = (rng.random(rows) < 0.01).astype(int)
    frame["visit"] = (rng.random(rows) < 0.05).astype(int)
    frame["exposure"] = 0
    with gzip.open(path, "wt") as handle:
        frame.to_csv(handle, index=False)
    return frame


def test_control_pool_keeps_only_control_rows_and_checks_row_count(tmp_path):
    source = tmp_path / "file.csv.gz"
    frame = make_file(source, 2_000)
    pool = _read_control_pool(source, 2_000)
    assert len(pool) == int((frame["treatment"] == 0).sum())
    assert (pool["treatment"] == 0).all()
    with pytest.raises(DatasetIntegrityError):
        _read_control_pool(source, 1_999)


def test_load_control_pool_reads_the_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    frame = make_file(tmp_path / "x.csv.gz", 300)
    frame.to_parquet(cache / "criteo_control_pool.parquet", index=False)
    pool = load_control_pool(tmp_path)
    assert pool.attrs["data_source"] == "criteo"
    assert len(pool) == 300
