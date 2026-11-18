import os
import re
import subprocess
import sys

SCRIPT = "GroundtruthPropagationTest2.py"  # your script's filename
BASE_ENV = {"DEBUG": "0"}

# One-at-a-time sweep: each knob is varied while the others stay at the current best.
GRID = {
    "THRESH": [0.35],
    "FB_SIGMA": [4],
    "GF_EPS": [25, 100, 400],
    "GF_RADIUS": [4, 8, 16],
    "MIN_COMPONENT_FRAC": [0.3],
    "PATCH_SIZE": [8, 12, 16],
    "MATCH_IOU": [0.05, 0.1, 0.2],
    "PATCH_STRIDE": [2],
    "REFINE_FLOW": [0, 1],
    "USE_CONFIDENCE": [0, 1],
    "MORPH": [0, 1],
    "GUIDED": [0, 1],
}
DEFAULTS = {
    "THRESH": 0.35, "FB_SIGMA": 4, "GF_EPS": 100, "GF_RADIUS": 8,
    "MIN_COMPONENT_FRAC": 0.3, "PATCH_SIZE": 8,
    "MATCH_IOU": 0.1, "PATCH_STRIDE": 2, "REFINE_FLOW": 1,
    "USE_CONFIDENCE": 1, "MORPH": 1, "GUIDED": 1,
}

# Run from the script's own folder so SCRIPT is found no matter where you launch this
HERE = os.path.dirname(os.path.abspath(__file__))


def run(params):
    env = {**os.environ, **BASE_ENV, **{k: str(v) for k, v in params.items()}}
    proc = subprocess.run([sys.executable, SCRIPT], env=env, cwd=HERE,
                          capture_output=True, text=True)
    m = re.search(r"Overall mean IoU: ([\d.]+)", proc.stdout)
    if not m:
        print("  (no score found; last stderr lines:)")
        print("  " + "\n  ".join(proc.stderr.strip().splitlines()[-3:]))
        return float("nan")
    return float(m.group(1))


best = dict(DEFAULTS)
best_score = run(best)
print(f"baseline {best_score:.4f}  {best}")

for knob, values in GRID.items():
    for v in values:
        if v == best[knob]:
            continue
        trial = {**best, knob: v}
        s = run(trial)
        flag = ""
        if s > best_score:  # NaN never compares greater, so failed runs are ignored
            best, best_score, flag = trial, s, "  <-- new best"
        print(f"{knob}={v:<6} -> {s:.4f}{flag}")

print(f"\nbest {best_score:.4f}: {best}")