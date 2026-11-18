import os
import re
import numpy as np
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "ImagesWithoutTruth")
GT_DIR = os.path.join(BASE, "Groundtruthmasks")
OUT_DIR = os.path.join(BASE, "PropagatedMasks")
PREFIX = "video-17-segment-1-frame-"
KNOWN_FRAMES = [0, 10]   # frames whose masks i have
EXTRAPOLATE = True       # also propagate before the first / after the last known frame

# ---- Tuned knobs (from the sweep) ----
THRESH = 0.35                # soft-mask threshold
FB_SIGMA = 4.0               # px; larger = more tolerant of flow inconsistency
GF_RADIUS = 8                # guided filter radius
GF_EPS = 100.0               # smaller = sticks harder to image edges
MIN_COMPONENT_FRAC = 0.3     # keep components >= this fraction of the largest
MATCH_IOU = 0.1              # min overlap to treat two objects as the same across keyframes
PATCH_SIZE = 8               # DIS patch size
PATCH_STRIDE = 2             # DIS patch stride

# ---- On/off switches ----
REFINE_FLOW = True           # re-estimate flow directly to the keyframe
USE_CONFIDENCE = True        # forward-backward consistency weighting in the blend
MORPH = True                 # 3x3 open + close on each output mask
GUIDED = False                # guided filter using the color frame
EPS = 1e-3

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

gray = {}
color = {}  # color frames, used as the guide image
for i in frame_ids:
    img = cv2.imread(os.path.join(IMG_DIR, img_files[i]))
    if img is None:
        raise IOError(f"Could not read image for frame {i}: {img_files[i]}")
    img = cv2.resize(img, (W, H))
    color[i] = img
    gray[i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Loaded {len(frame_ids)} frames at {W}x{H}")

# ---------------- Optical flow ----------------
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
dis.setFinestScale(0)                       # full-resolution flow (MEDIUM default is 1)
dis.setPatchSize(PATCH_SIZE)
dis.setPatchStride(PATCH_STRIDE)
dis.setGradientDescentIterations(40)
dis.setVariationalRefinementIterations(10)
dis.setUseSpatialPropagation(True)

grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def flow(target, source, init=None):
    """Flow defined on `target` pixels, pointing to where they sit in `source`."""
    return dis.calc(target, source, None if init is None else init.copy())


def warp(mask, fl):
    return cv2.remap(mask, grid_x + fl[..., 0], grid_y + fl[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def fb_confidence(acc, start, t):
    """Confidence in [0,1] that flow t->start is reliable, from a forward-backward check."""
    back = flow(gray[start], gray[t])  # start -> t
    mx, my = grid_x + acc[..., 0], grid_y + acc[..., 1]
    bx = cv2.remap(back[..., 0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    by = cv2.remap(back[..., 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.sqrt((acc[..., 0] + bx) ** 2 + (acc[..., 1] + by) ** 2)
    conf = np.exp(-(err / FB_SIGMA) ** 2).astype(np.float32)
    inside = (mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)
    return conf * inside


def frames_between(start, end):
    if start not in gray or end not in gray:
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
        step_flow = flow(gray[t], gray[prev])
        acc = step_flow + cv2.remap(
            acc, grid_x + step_flow[..., 0], grid_y + step_flow[..., 1],
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if REFINE_FLOW and prev != start:
            acc = flow(gray[t], gray[start], init=acc)
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
from collections import defaultdict

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