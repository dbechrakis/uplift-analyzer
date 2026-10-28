from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [f"f{i}" for i in range(12)]
OUTCOMES = ("conversion", "visit", "exposure")
SCHEMA_COLUMNS = [*FEATURES, "treatment", "conversion", "visit", "exposure"]
DTYPES = {
    **dict.fromkeys(FEATURES, "float32"),
    "treatment": "int8",
    "conversion": "int8",
    "visit": "int8",
    "exposure": "int8",
}

DATASET_FILENAME = "criteo-research-uplift-v2.1.csv.gz"
DATASET_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/" + DATASET_FILENAME
)
DATASET_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
DATASET_ROWS = 13_979_592

DEFAULT_SAMPLE_SIZE = 500_000
DEFAULT_SEED = 42
CHUNK_ROWS = 1_000_000


class DatasetNotFoundError(FileNotFoundError):
    pass


class DatasetIntegrityError(ValueError):
    pass


def data_dir() -> Path:
    return Path(os.environ.get("UPLIFT_DATA_DIR", "data"))


def default_sample_size() -> int:
    return int(os.environ.get("UPLIFT_SAMPLE_SIZE", DEFAULT_SAMPLE_SIZE))


def default_seed() -> int:
    return int(os.environ.get("UPLIFT_SEED", DEFAULT_SEED))


def raw_path(directory: Path | None = None) -> Path:
    return (directory or data_dir()) / "raw" / DATASET_FILENAME


def cache_path(sample_size: int, seed: int, directory: Path | None = None) -> Path:
    return (directory or data_dir()) / "cache" / f"criteo_sample_{sample_size}_{seed}.parquet"


def manual_download_instructions(directory: Path | None = None) -> str:
    target = raw_path(directory)
    return (
        "Criteo Uplift v2.1 not found.\n"
        f"1. Download {DATASET_URL}\n"
        "   (Hugging Face mirror of the Criteo Uplift Prediction dataset, 311 MB, no login)\n"
        f"2. Save it as: {target}\n"
        "   Or run: python -m core.data --download\n"
        f"Expected SHA-256: {DATASET_SHA256}"
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_dataset(directory: Path | None = None, verify: bool = True) -> Path:
    target = raw_path(directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(DATASET_URL, timeout=60) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    if verify and sha256_of(partial) != DATASET_SHA256:
        partial.unlink()
        raise DatasetIntegrityError("Downloaded file failed the SHA-256 check")
    partial.replace(target)
    return target


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _read_sample(path: Path, indices: np.ndarray) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    offset = 0
    cursor = 0
    with gzip.open(path, "rt") as handle:
        for chunk in pd.read_csv(handle, dtype=DTYPES, chunksize=CHUNK_ROWS):
            end = offset + len(chunk)
            stop = np.searchsorted(indices, end, side="left")
            local = indices[cursor:stop] - offset
            if len(local):
                parts.append(chunk.iloc[local])
            cursor = stop
            offset = end
    if offset != DATASET_ROWS:
        raise DatasetIntegrityError(f"Expected {DATASET_ROWS} rows, found {offset}")
    return pd.concat(parts, ignore_index=True)


def load_criteo_sample(
    sample_size: int | None = None,
    seed: int | None = None,
    directory: Path | None = None,
    download: bool = False,
) -> pd.DataFrame:
    sample_size = default_sample_size() if sample_size is None else sample_size
    seed = default_seed() if seed is None else seed
    if not 0 < sample_size <= DATASET_ROWS:
        raise ValueError(f"sample_size must be between 1 and {DATASET_ROWS}")
    cached = cache_path(sample_size, seed, directory)
    if cached.exists():
        df = pd.read_parquet(cached)
    else:
        source = raw_path(directory)
        if not source.exists():
            if not download:
                raise DatasetNotFoundError(manual_download_instructions(directory))
            download_dataset(directory)
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(DATASET_ROWS, size=sample_size, replace=False))
        df = _read_sample(source, indices)
        cached.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cached, index=False)
    validate_schema(df)
    df.attrs["data_source"] = "criteo"
    return df


@dataclass(frozen=True)
class DatasetInfo:
    rows: int
    treated_share: float
    conversion_rate: float
    visit_rate: float
    exposure_rate: float
    data_source: str


def describe(df: pd.DataFrame, data_source: str = "unspecified") -> DatasetInfo:
    return DatasetInfo(
        rows=len(df),
        treated_share=float(df["treatment"].mean()),
        conversion_rate=float(df["conversion"].mean()),
        visit_rate=float(df["visit"].mean()),
        exposure_rate=float(df["exposure"].mean()),
        data_source=df.attrs.get("data_source", data_source),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Criteo Uplift dataset helper")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        print(download_dataset())
    else:
        print(manual_download_instructions())
