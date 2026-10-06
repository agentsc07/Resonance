"""The harness must be correct BEFORE the engine exists: oracle = perfect score, null = zero. Run: python eval/test_harness.py"""
import subprocess
import sys
import json
import tempfile
from pathlib import Path

run = Path(__file__).parent / "run.py"


def go(pred):
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, str(run), "--split", "all", "--predictor", pred, "--out", td], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
        return json.loads((Path(td) / "summary.json").read_text())


o, n = go("oracle"), go("null")
assert o["invalid_predictions"] == 0 and n["invalid_predictions"] == 0
assert o["flaw_f1@0.5"] == 1.0 and o["flaw_f1@0.3"] == 1.0, o
assert o["category_f1@0.5"] == 1.0
assert o["median_onset_ms"] == 0.0 and o["median_offset_ms"] == 0.0
assert o["category_acc"] == 1.0
assert o["dose_response_min_spearman"] == 1.0, o
assert o["false_flags_per_min"] == 0.0
assert n["flaw_f1@0.5"] == 0.0 and n["false_flags_per_min"] == 0.0
print("harness OK: oracle F1=1.0 / timing 0 ms / category 100% / dose-response 1.0 / 0 false flags; null F1=0")
