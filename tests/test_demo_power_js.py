import json
import shutil
import subprocess
from pathlib import Path

import pytest

from core.power import minimum_detectable_effect, required_sample_size

CASES = [
    (0.5, -0.2, 0.05, 0.8, 0.5),
    (0.00195, 0.1, 0.05, 0.8, 0.85),
    (0.0367, 0.25, 0.05, 0.9, 0.3),
    (0.02, 0.5, 0.01, 0.95, 0.7),
    (0.3, 0.05, 0.10, 0.6, 0.5),
]

node = shutil.which("node")


@pytest.mark.skipif(node is None, reason="node is not installed")
def test_js_power_calculator_matches_python():
    script = f"""
const P = require({json.dumps(str(Path("demo/power.js").resolve()))});
const out = {json.dumps(CASES)}.map(([b, l, a, p, s]) => {{
  const r = P.requiredSampleSize(b, l, a, p, s);
  const m = P.minimumDetectableEffect(b, r.nTotal, a, p, s);
  return [r.nTreated, r.nControl, m.absoluteEffect];
}});
console.log(JSON.stringify(out));
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    js = json.loads(result.stdout)
    for (b, lift, a, p, s), (nt, nc, mde) in zip(CASES, js, strict=True):
        py = required_sample_size(b, relative_lift=lift, alpha=a, power=p, treated_share=s)
        assert (nt, nc) == (py.n_treated, py.n_control)
        py_mde = minimum_detectable_effect(b, py.n_total, alpha=a, power=p, treated_share=s)
        assert mde == pytest.approx(py_mde.absolute_effect, rel=1e-4)
