import os
import re
import zlib
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # saves PNGs instead of opening dozens of windows
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------- Config ----------------
BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "ImagesWithoutTruth")
GT_DIR = os.path.join(BASE, "Groundtruthmasks")
PROP_DIR = os.path.join(BASE, "PropagatedMasks")
OUT_DIR = os.path.join(BASE, "RaftOptimal2")
PREFIX = "video-17-segment-1-frame-"
KNOWN_FRAMES = [0, 10]
SIDE_BY_SIDE = True    # show ground truth next to the propagated result when it exists
MIN_SCORE = 0.0        # hides low-confidence ground-truth objects (propagated masks have no scores)
ALPHA = 0.5            # mask opacity
CMAP = plt.get_cmap("tab20")


# ---------------- Helpers ----------------
def npz_path(folder, frame):
    return os.path.join(folder, f"{PREFIX}{frame}.npz")


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    labels = [str(l) for l in d["labels"]]
    n = len(labels)
    masks = np.stack([d[f"mask_{i}"] for i in range(n)]).astype(bool)
    scores = d["scores"] if "scores" in d.files else None
    boxes = d["boxes"] if "boxes" in d.files else None
    return masks, labels, scores, boxes


def label_color(label):
    """Same label -> same color in every frame."""
    return np.array(CMAP(zlib.crc32(label.encode()) % 20)[:3])


def mask_to_box(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def draw(ax, img, masks, labels, scores, boxes, title):
    overlay = img.astype(float).copy()
    order = np.argsort(-masks.reshape(len(masks), -1).sum(axis=1))  # big objects first, small on top
    keep = [i for i in order if scores is None or scores[i] >= MIN_SCORE]

    for i in keep:
        c = label_color(labels[i]) * 255
        overlay[masks[i]] = (1 - ALPHA) * overlay[masks[i]] + ALPHA * c
    ax.imshow(overlay.astype(np.uint8))

    for i in keep:
        c = label_color(labels[i])
        box = boxes[i] if boxes is not None else mask_to_box(masks[i])
        if box is None or not np.any(masks[i]):
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor=c, linewidth=1.5))
        text = labels[i] if scores is None else f"{labels[i]} {scores[i]:.2f}"
        ax.text(x1, max(y1 - 2, 8), text, fontsize=6, color="black",
                bbox=dict(facecolor="white", alpha=0.8, pad=1, edgecolor="none"))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


# ---------------- Main ----------------
frame_ids = sorted(
    int(m.group(1))
    for f in os.listdir(IMG_DIR)
    if (m := re.search(r"frame-(\d+)\.jpe?g$", f))
)
os.makedirs(OUT_DIR, exist_ok=True)

for t in frame_ids:
    # Known frames come from your ground truth, the rest from the propagation output
    if t in KNOWN_FRAMES:
        pred_path, pred_title = npz_path(GT_DIR, t), f"Frame {t} (known ground truth)"
    else:
        pred_path, pred_title = npz_path(PROP_DIR, t), f"Frame {t} (propagated)"
    if not os.path.exists(pred_path):
        print(f"Skipping frame {t}: {pred_path} not found")
        continue

    masks, labels, scores, boxes = load_npz(pred_path)
    H, W = masks.shape[1:]
    img = cv2.cvtColor(cv2.imread(os.path.join(IMG_DIR, f"{PREFIX}{t}.jpeg")), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H))

    panels = [(masks, labels, scores, boxes, pred_title)]
    gt_path = npz_path(GT_DIR, t)
    if SIDE_BY_SIDE and t not in KNOWN_FRAMES and os.path.exists(gt_path):
        gm, gl, gs, gb = load_npz(gt_path)
        panels.append((gm, gl, gs, gb, f"Frame {t} (ground truth)"))

    fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * W / 70, H / 70))
    axes = np.atleast_1d(axes)
    for ax, (m, l, s, b, title) in zip(axes, panels):
        draw(ax, img, m, l, s, b, title)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"frame-{t}.png")
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")

print(f"\nDone. Open the {OUT_DIR} folder to flip through the frames.")