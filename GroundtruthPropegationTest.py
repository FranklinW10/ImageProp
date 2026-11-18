import os
import re
import numpy as np
import cv2

# ---------------- Config ----------------
BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "ImagesWithoutTruth")
GT_DIR = os.path.join(BASE, "Groundtruthmasks")
OUT_DIR = os.path.join(BASE, "PropagatedMasks")
PREFIX = "video-17-segment-1-frame-"
KNOWN_FRAMES = [0, 10]   # frames whose masks you "have" (add more if you have a third)
MATCH_IOU = 0.1          # min overlap to treat two objects as the same one across known frames


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
frame_ids = sorted(
    int(m.group(1))
    for f in os.listdir(IMG_DIR)
    if (m := re.search(r"frame-(\d+)\.jpe?g$", f))
)
first_masks, _ = load_npz(npz_path(GT_DIR, KNOWN_FRAMES[0]))
H, W = first_masks.shape[1:]

gray = {}
for i in frame_ids:
    img = cv2.imread(os.path.join(IMG_DIR, f"{PREFIX}{i}.jpeg"))
    img = cv2.resize(img, (W, H))
    gray[i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Loaded {len(frame_ids)} frames at {W}x{H}")

dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def warp(mask, flow):
    """mask lives in the source frame; flow was computed from the target frame to the source frame."""
    return cv2.remap(mask, grid_x + flow[..., 0], grid_y + flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def chain(masks, start, end):
    """Carry soft masks from frame `start` to frame `end`, one frame at a time. Returns {frame: (n,H,W)}."""
    step = 1 if end > start else -1
    cur = masks.astype(np.float32)
    out = {start: cur}
    for t in range(start + step, end + step, step):
        flow = dis.calc(gray[t], gray[t - step], None)  # target -> source
        cur = np.stack([warp(m, flow) for m in cur])
        out[t] = cur
    return out


def match_tracks(fwd_end, B, labels_a, labels_b):
    """Greedy match of objects propagated from frame a against ground-truth objects at frame b."""
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


# ---------------- Propagate between each pair of known frames ----------------
results = {}  # frame -> list of (label, bool mask)

for a, b in zip(KNOWN_FRAMES[:-1], KNOWN_FRAMES[1:]):
    A, labels_a = load_npz(npz_path(GT_DIR, a))
    B, labels_b = load_npz(npz_path(GT_DIR, b))
    fwd = chain(A, a, b)
    bwd = chain(B, b, a)

    matches = match_tracks(fwd[b], B, labels_a, labels_b)
    matched_a = {i for i, _ in matches}
    matched_b = {j for _, j in matches}
    print(f"Frames {a}->{b}: {len(matches)} matched objects, "
          f"{len(labels_a) - len(matches)} only in frame {a}, {len(labels_b) - len(matches)} only in frame {b}")

    for t in range(a + 1, b):
        w = (t - a) / (b - a)  # 0 near frame a, 1 near frame b
        objs = []
        for i, j in matches:
            soft = (1 - w) * fwd[t][i] + w * bwd[t][j]
            objs.append((labels_a[i], soft > 0.5))
        for i in range(len(labels_a)):
            if i not in matched_a:
                objs.append((labels_a[i], fwd[t][i] > 0.5))
        for j in range(len(labels_b)):
            if j not in matched_b:
                objs.append((labels_b[j], bwd[t][j] > 0.5))
        results[t] = objs

# ---------------- Save ----------------
os.makedirs(OUT_DIR, exist_ok=True)
for t, objs in results.items():
    save = {f"mask_{k}": m for k, (_, m) in enumerate(objs)}
    save["labels"] = np.array([l for l, _ in objs])
    save["boxes"] = np.stack([mask_to_box(m) for _, m in objs])
    np.savez_compressed(npz_path(OUT_DIR, t), **save)
print(f"Saved propagated masks to {OUT_DIR}")

# ---------------- Evaluate against ground truth ----------------
print("\nEvaluation vs ground truth (mean best IoU per ground-truth object):")
all_scores = []
for t in sorted(results):
    if not os.path.exists(npz_path(GT_DIR, t)):
        continue
    gt_masks, gt_labels = load_npz(npz_path(GT_DIR, t))
    scores = []
    for gm, gl in zip(gt_masks, gt_labels):
        cands = [iou(gm, m) for l, m in results[t] if l == gl]
        scores.append(max(cands) if cands else 0.0)
    all_scores.extend(scores)
    print(f"  frame {t:>2}: mean IoU {np.mean(scores):.3f} over {len(scores)} objects")
if all_scores:
    print(f"\nOverall mean IoU: {np.mean(all_scores):.3f}")