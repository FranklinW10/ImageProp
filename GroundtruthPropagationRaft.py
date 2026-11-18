import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must be set before importing torch
import re
from collections import defaultdict
import numpy as np
import cv2
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

# ---------------- Config ----------------
BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "ImagesWithoutTruth")
GT_DIR = os.path.join(BASE, "Groundtruthmasks")
OUT_DIR = os.path.join(BASE, "PropagatedMasks")
PREFIX = "video-17-segment-1-frame-"
KNOWN_FRAMES = [0, 10]   # frames whose masks you "have"
EXTRAPOLATE = True       # also propagate before the first / after the last known frame

# ---- Tuned knobs (tuned with DIS; worth re-sweeping with RAFT) ----
THRESH = 0.35
FB_SIGMA = 4.0
GF_RADIUS = 8
GF_EPS = 100.0
MIN_COMPONENT_FRAC = 0.3
MATCH_IOU = 0.1

# ---- On/off switches ----
REFINE_FLOW = True       # compute flow directly to the keyframe instead of only composing steps
USE_CONFIDENCE = True
MORPH = True
GUIDED = False
EPS = 1e-3

# ---- RAFT settings ----
DEVICE = "cpu"           # "cpu" is safest; try "mps" for speed (fallback to CPU is enabled for unsupported ops)
RAFT_MAX_SIDE = 768      # flow is computed at most this size on the long side, then scaled back up
RAFT_ITERS = 20          # refinement iterations; 12 is faster, 32 is slightly more accurate

# ---- Debug ----
DEBUG = False
DEBUG_FRAMES = [3]

HAS_GUIDED = hasattr(cv2, "ximgproc")
if GUIDED and not HAS_GUIDED:
    print("WARNING: cv2.ximgproc missing, skipping guided filter. "
          "Run: pip uninstall opencv-python && pip install opencv-contrib-python")


# ---------------- Helpers ----------------
def npz_path(folder, frame):
    return os.path.join(folder, f"{PREFIX}{frame}.npz")


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    n = len(d["labels"])
    masks = np.stack([d[f"mask_{i}"] for i in range(n)]).astype(bool)
    labels = [str(l) for l in d["labels"]]
    return masks, labels


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def mask_to_box(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


# ---------------- Load frames ----------------
img_files = {}
for f in os.listdir(IMG_DIR):
    m = re.search(r"frame-(\d+)\.jpe?g$", f, re.IGNORECASE)
    if m:
        img_files[int(m.group(1))] = f
frame_ids = sorted(img_files)

first_masks, _ = load_npz(npz_path(GT_DIR, KNOWN_FRAMES[0]))
H, W = first_masks.shape[1:]

color = {}  # BGR frames at mask resolution (guide image for the guided filter)
for i in frame_ids:
    img = cv2.imread(os.path.join(IMG_DIR, img_files[i]))
    if img is None:
        raise IOError(f"Could not read image for frame {i}: {img_files[i]}")
    color[i] = cv2.resize(img, (W, H))
print(f"Loaded {len(frame_ids)} frames at {W}x{H}")

# ---------------- Optical flow (RAFT) ----------------
device = torch.device(DEVICE)
print(f"Loading RAFT on {device} (first run downloads the weights)...")
_raft = raft_large(weights=Raft_Large_Weights.DEFAULT).eval().to(device)

# RAFT needs height and width divisible by 8, so run at a nearby multiple-of-8 size
_scale = min(1.0, RAFT_MAX_SIDE / max(H, W))
RW = max(8, int(round(W * _scale / 8)) * 8)
RH = max(8, int(round(H * _scale / 8)) * 8)
print(f"RAFT runs at {RW}x{RH}")

_raft_input = {}
for i in frame_ids:
    rgb = cv2.cvtColor(cv2.resize(color[i], (RW, RH)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0 * 2.0 - 1.0
    _raft_input[i] = t.to(device)

_flow_cache = {}

grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


@torch.no_grad()
def flow(target, source):
    """Flow defined on `target` frame pixels, pointing to where they sit in the `source` frame.
    Both arguments are frame numbers. Results are cached."""
    key = (target, source)
    if key in _flow_cache:
        return _flow_cache[key]
    pred = _raft(_raft_input[target], _raft_input[source], num_flow_updates=RAFT_ITERS)[-1]
    fl = pred[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    if (RW, RH) != (W, H):
        fl = cv2.resize(fl, (W, H), interpolation=cv2.INTER_LINEAR)
        fl[..., 0] *= W / RW
        fl[..., 1] *= H / RH
    fl = np.ascontiguousarray(fl)
    _flow_cache[key] = fl
    return fl


def warp(mask, fl):
    return cv2.remap(mask, grid_x + fl[..., 0], grid_y + fl[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def fb_confidence(acc, start, t):
    """Confidence in [0,1] that flow t->start is reliable, from a forward-backward check."""
    back = flow(start, t)  # start -> t
    mx, my = grid_x + acc[..., 0], grid_y + acc[..., 1]
    bx = cv2.remap(back[..., 0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    by = cv2.remap(back[..., 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.sqrt((acc[..., 0] + bx) ** 2 + (acc[..., 1] + by) ** 2)
    conf = np.exp(-(err / FB_SIGMA) ** 2).astype(np.float32)
    inside = (mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)
    return conf * inside


def frames_between(start, end):
    if start not in color or end not in color:
        raise KeyError(f"Frame {start} or {end} has no image in {IMG_DIR}")
    lo, hi = sorted((start, end))
    ids = [f for f in frame_ids if lo <= f <= hi]
    return ids if end >= start else ids[::-1]


def chain(masks, start, end):
    """Carry soft masks from `start` to `end`.
    Returns {frame: (soft masks (n,H,W), confidence (H,W))}."""
    seq = frames_between(start, end)
    base = masks.astype(np.float32)
    out = {start: (base, np.ones((H, W), np.float32))}
    acc = np.zeros((H, W, 2), np.float32)  # flow from the current frame back to `start`
    for prev, t in zip(seq[:-1], seq[1:]):
        if REFINE_FLOW or prev == start:
            acc = flow(t, start) if REFINE_FLOW else flow(t, prev)
        else:
            step_flow = flow(t, prev)
            acc = step_flow + cv2.remap(
                acc, grid_x + step_flow[..., 0], grid_y + step_flow[..., 1],
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        conf = fb_confidence(acc, start, t) if USE_CONFIDENCE else np.ones((H, W), np.float32)
        out[t] = (np.stack([warp(m, acc) for m in base]), conf)
    return out


def match_tracks(fwd_end, B, labels_a, labels_b):
    pairs = []
    for i in range(len(labels_a)):
        for j in range(len(labels_b)):
            if labels_a[i] == labels_b[j]:
                s = iou(fwd_end[i] > 0.5, B[j])
                if s >= MATCH_IOU:
                    pairs.append((s, i, j))
    pairs.sort(reverse=True)
    used_i, used_j, matches = set(), set(), []
    for s, i, j in pairs:
        if i not in used_i and j not in used_j:
            matches.append((i, j))
            used_i.add(i)
            used_j.add(j)
    return matches


_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def finalize(soft, t):
    """Soft mask -> guided filter (snap to image edges) -> threshold -> morphology -> drop far-away islands."""
    soft = np.clip(soft.astype(np.float32), 0.0, 1.0)
    if GUIDED and HAS_GUIDED:
        soft = cv2.ximgproc.guidedFilter(color[t], soft, GF_RADIUS, GF_EPS)
    m = (soft > THRESH).astype(np.uint8)
    if MORPH:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _kernel)
    if MIN_COMPONENT_FRAC > 0 and m.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep_ids = 1 + np.where(areas >= MIN_COMPONENT_FRAC * areas.max())[0]
            m = np.isin(lab, keep_ids).astype(np.uint8)
    return m.astype(bool)


# ---------------- Propagate between each pair of known frames ----------------
results = {}  # frame -> list of (label, bool mask)

for a, b in zip(KNOWN_FRAMES[:-1], KNOWN_FRAMES[1:]):
    A, labels_a = load_npz(npz_path(GT_DIR, a))
    B, labels_b = load_npz(npz_path(GT_DIR, b))
    fwd = chain(A, a, b)
    bwd = chain(B, b, a)

    matches = match_tracks(fwd[b][0], B, labels_a, labels_b)
    matched_a = {i for i, _ in matches}
    matched_b = {j for _, j in matches}
    print(f"Frames {a}->{b}: {len(matches)} matched objects, "
          f"{len(labels_a) - len(matches)} only in frame {a}, {len(labels_b) - len(matches)} only in frame {b}")

    for t in (f for f in frame_ids if a < f < b):
        w = (t - a) / (b - a)  # 0 near frame a, 1 near frame b
        f_soft, f_conf = fwd[t]
        b_soft, b_conf = bwd[t]
        wf = (1 - w) * (f_conf + EPS)   # time proximity x flow reliability
        wb = w * (b_conf + EPS)
        norm = wf + wb
        objs = []
        for i, j in matches:
            soft = (wf * f_soft[i] + wb * b_soft[j]) / norm
            objs.append((labels_a[i], finalize(soft, t)))
        for i in range(len(labels_a)):
            if i not in matched_a:
                objs.append((labels_a[i], finalize(f_soft[i], t)))
        for j in range(len(labels_b)):
            if j not in matched_b:
                objs.append((labels_b[j], finalize(b_soft[j], t)))
        results[t] = objs

# ---------------- Extrapolate outside the known range ----------------
if EXTRAPOLATE:
    last = KNOWN_FRAMES[-1]
    after = [f for f in frame_ids if f > last]
    if after:
        A, labels = load_npz(npz_path(GT_DIR, last))
        fwd = chain(A, last, after[-1])
        for t in after:
            results[t] = [(labels[i], finalize(fwd[t][0][i], t)) for i in range(len(labels))]
        print(f"Extrapolated {len(after)} frames after frame {last}")

    first = KNOWN_FRAMES[0]
    before = [f for f in frame_ids if f < first]
    if before:
        A, labels = load_npz(npz_path(GT_DIR, first))
        bwd = chain(A, first, before[0])
        for t in before:
            results[t] = [(labels[i], finalize(bwd[t][0][i], t)) for i in range(len(labels))]
        print(f"Extrapolated {len(before)} frames before frame {first}")

# ---------------- Debug: per-object diagnostics ----------------
if DEBUG:
    for t in DEBUG_FRAMES:
        if t not in results:
            continue
        print(f"\nDebug frame {t}:")
        for k, (lab, m) in enumerate(results[t]):
            n_comp = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1
            box = mask_to_box(m).astype(int).tolist()
            print(f"  obj {k:>2} {lab:<14} area {int(m.sum()):>6}  components {n_comp}  box {box}")

# ---------------- Save ----------------
os.makedirs(OUT_DIR, exist_ok=True)
for t, objs in results.items():
    save = {f"mask_{k}": m for k, (_, m) in enumerate(objs)}
    save["labels"] = np.array([l for l, _ in objs])
    save["boxes"] = (np.stack([mask_to_box(m) for _, m in objs])
                     if objs else np.zeros((0, 4), dtype=np.float32))
    np.savez_compressed(npz_path(OUT_DIR, t), **save)
print(f"Saved propagated masks to {OUT_DIR}")

# ---------------- Evaluate against ground truth ----------------
print("\nEvaluation vs ground truth (mean best IoU per ground-truth object):")
all_scores = []
by_label = defaultdict(list)    # label -> [(iou, gt_area), ...]
pred_count = defaultdict(int)   # label -> predicted objects on scored frames
gt_count = defaultdict(int)     # label -> GT objects on scored frames

for t in sorted(results):
    if not os.path.exists(npz_path(GT_DIR, t)):
        continue
    gt_masks, gt_labels = load_npz(npz_path(GT_DIR, t))
    scores = []
    for gm, gl in zip(gt_masks, gt_labels):
        cands = [iou(gm, m) for l, m in results[t] if l == gl]
        s = max(cands) if cands else 0.0
        scores.append(s)
        by_label[gl].append((s, int(gm.sum())))
        gt_count[gl] += 1
    for l, _ in results[t]:
        pred_count[l] += 1
    all_scores.extend(scores)
    print(f"  frame {t:>2}: mean IoU {np.mean(scores):.3f} over {len(scores)} objects "
          f"({len(results[t])} predicted)")

if all_scores:
    arr = np.array(all_scores)
    print(f"\nOverall mean IoU: {arr.mean():.3f}   "
          f">0.5: {(arr > 0.5).mean():.0%}   >0.75: {(arr > 0.75).mean():.0%}")
    print("\nPer label (sorted worst first):")
    print(f"  {'label':<16}{'mean IoU':>9}{'n GT':>6}{'n pred':>8}{'avg GT area':>13}")
    for lab, vals in sorted(by_label.items(), key=lambda kv: np.mean([v[0] for v in kv[1]])):
        print(f"  {lab:<16}{np.mean([v[0] for v in vals]):>9.3f}{gt_count[lab]:>6}"
              f"{pred_count[lab]:>8}{np.mean([v[1] for v in vals]):>13.0f}")