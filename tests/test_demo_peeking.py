import json
import shutil
import subprocess
from pathlib import Path

import pytest

from core.export_demo import peeking_data

node = shutil.which("node")
ROOT = Path(__file__).resolve().parent.parent


def test_peeking_data_matches_the_results_file():
    data = peeking_data(ROOT / "outputs" / "sequential.json")
    record = json.loads((ROOT / "outputs" / "sequential.json").read_text())
    peek20 = next(
        r["rejection_rate"]
        for r in record["rows"]
        if r["rule"] == "peek"
        and r["n_looks"] == 20
        and r["injected_lift"] == 0.0
        and r["adjustment"] == "none"
        and r["outcome"] == "visit"
    )
    assert data["false_positive"]["peek"]["20"] == peek20
    assert data["data_source"] == "criteo" and data["looks"] == [5, 20, 200]
    published = json.loads((ROOT / "demo" / "data.json").read_text())["peeking"]
    assert published == data


@pytest.mark.skipif(node is None, reason="node is not installed")
def test_js_series_matches_the_exported_data():
    data = peeking_data(ROOT / "outputs" / "sequential.json")
    script = f"""
const P = require({json.dumps(str((ROOT / "demo" / "peeking.js").resolve()))});
const data = {json.dumps(data)};
console.log(JSON.stringify([5, 20, 200].map(k => P.series(data, k))));
let bad = false;
try {{ P.series(data, 7); }} catch (e) {{ bad = e instanceof RangeError; }}
console.log(bad);
"""
    out = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, check=True
    ).stdout.split("\n")
    series = json.loads(out[0])
    for k, s in zip([5, 20, 200], series, strict=True):
        assert s["rules"] == data["rules"]
        assert s["falsePositive"] == [data["false_positive"][r][str(k)] for r in data["rules"]]
        assert s["power"] == [data["power_at_5_percent_lift"][r][str(k)] for r in data["rules"]]
    assert out[1] == "true"
