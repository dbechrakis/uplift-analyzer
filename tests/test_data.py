import gzip

import pandas as pd
import pytest

import core.data as data
from core.synthetic import SYNTHETIC_BANNER, generate_synthetic


def test_synthetic_schema_and_label():
    df = generate_synthetic(n=5000, seed=0)
    assert list(df.columns) == data.SCHEMA_COLUMNS
    assert df.attrs["data_source"] == "synthetic"
    assert "SYNTHETIC" in SYNTHETIC_BANNER
    assert set(df["treatment"].unique()) == {0, 1}
    assert df["visit"].ge(df["conversion"]).all()
    assert abs(df["treatment"].mean() - 0.85) < 0.03


def test_synthetic_is_deterministic_and_has_positive_effect():
    a = generate_synthetic(n=3000, seed=4)
    b = generate_synthetic(n=3000, seed=4)
    pd.testing.assert_frame_equal(a, b)
    big = generate_synthetic(n=50_000, seed=1)
    rates = big.groupby("treatment")["conversion"].mean()
    assert rates[1] > rates[0]


def test_missing_dataset_gives_actionable_message(tmp_path):
    with pytest.raises(data.DatasetNotFoundError) as err:
        data.load_criteo_sample(1000, 1, directory=tmp_path)
    message = str(err.value)
    assert data.DATASET_URL in message
    assert str(data.raw_path(tmp_path)) in message
    assert data.DATASET_SHA256 in message


def test_validate_schema_flags_missing_columns():
    with pytest.raises(ValueError, match="Missing columns"):
        data.validate_schema(pd.DataFrame({"f0": [1.0]}))


@pytest.fixture
def fake_raw(tmp_path, monkeypatch):
    rows = 2500
    monkeypatch.setattr(data, "DATASET_ROWS", rows)
    monkeypatch.setattr(data, "CHUNK_ROWS", 700)
    source = generate_synthetic(n=rows, seed=9)
    path = data.raw_path(tmp_path)
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as handle:
        source.to_csv(handle, index=False)
    return tmp_path, source


def test_sampling_is_exact_reproducible_and_cached(fake_raw):
    directory, source = fake_raw
    a = data.load_criteo_sample(600, 3, directory=directory)
    assert len(a) == 600
    assert a.attrs["data_source"] == "criteo"
    assert data.cache_path(600, 3, directory).exists()
    b = data.load_criteo_sample(600, 3, directory=directory)
    pd.testing.assert_frame_equal(a, b)
    c = data.load_criteo_sample(600, 4, directory=directory)
    assert not a.equals(c)
    assert str(a["f0"].dtype) == "float32" and str(a["treatment"].dtype) == "int8"
    merged = a.merge(source, on=data.SCHEMA_COLUMNS, how="inner")
    assert len(merged) >= 600


def test_row_count_mismatch_is_detected(fake_raw, monkeypatch):
    directory, _ = fake_raw
    monkeypatch.setattr(data, "DATASET_ROWS", 2600)
    with pytest.raises(data.DatasetIntegrityError):
        data.load_criteo_sample(100, 1, directory=directory)


def test_sample_size_bounds(tmp_path):
    with pytest.raises(ValueError):
        data.load_criteo_sample(0, 1, directory=tmp_path)
    with pytest.raises(ValueError):
        data.load_criteo_sample(data.DATASET_ROWS + 1, 1, directory=tmp_path)


def test_network_is_blocked_in_tests():
    import urllib.error

    with pytest.raises((RuntimeError, urllib.error.URLError)):
        data.download_dataset()


def test_describe_uses_dataset_label():
    info = data.describe(generate_synthetic(n=1000, seed=2))
    assert info.data_source == "synthetic" and info.rows == 1000
