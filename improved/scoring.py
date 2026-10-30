# %% [markdown]
# # Scoring and clean-up
#
# The competition score (Panoptic Quality) and the clean-up that turns the network's
# probability map into separate filaments.

# %%
import cv2
import numpy as np
from numpy.typing import NDArray
from pycocotools import mask as mask_utils
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from improved.config import SIZE

# %% [markdown]
# ## The Panoptic Quality (PQ) score
#
# This is the competition score. A predicted filament only counts if it overlaps a real one
# by more than 50%. `pq_score` is the same code as in `code.ipynb`. `pq_fast` gives the
# same answer from label images and is used for quick checks while tuning.


# %%
def pq_score(truth, pred, threshold=0.5):
    """PQ between two lists of masks (RLE format)."""
    if len(truth) == 0 and len(pred) == 0:
        return dict(pq=1.0, sq=1.0, rq=1.0, tp=0, fp=0, fn=0, one_to_many=0, many_to_one=0)
    if len(truth) == 0 or len(pred) == 0:
        return dict(
            pq=0.0, sq=0.0, rq=0.0, tp=0, fp=len(pred), fn=len(truth), one_to_many=0, many_to_one=0
        )

    iou = np.array(mask_utils.iou(pred, truth, [0] * len(truth)))
    rows, cols = linear_sum_assignment(-iou)
    matched = iou[rows, cols]
    matched = matched[matched > threshold]

    tp = len(matched)
    fp = len(pred) - tp
    fn = len(truth) - tp
    sq = matched.mean() if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)

    touching = iou > 0
    return dict(
        pq=sq * rq,
        sq=sq,
        rq=rq,
        tp=tp,
        fp=fp,
        fn=fn,
        one_to_many=int((touching.sum(axis=0) > 1).sum()),
        many_to_one=int((touching.sum(axis=1) > 1).sum()),
    )


def pq_fast(truth_labels, pred_labels, threshold=0.5):
    """Same as pq_score, but from two label images (0 = background, 1..n = filaments)."""
    n_truth, n_pred = int(truth_labels.max()), int(pred_labels.max())
    if n_truth == 0 and n_pred == 0:
        return dict(pq=1.0, sq=1.0, rq=1.0, tp=0, fp=0, fn=0)
    if n_truth == 0 or n_pred == 0:
        return dict(pq=0.0, sq=0.0, rq=0.0, tp=0, fp=n_pred, fn=n_truth)

    # count overlapping pixels for every (truth, prediction) pair in one go
    pair_ids = truth_labels.ravel().astype(np.int64) * (n_pred + 1) + pred_labels.ravel()
    counts = np.bincount(pair_ids, minlength=(n_truth + 1) * (n_pred + 1))
    counts = counts.reshape(n_truth + 1, n_pred + 1)

    overlap = counts[1:, 1:].astype(np.float64)
    truth_area = counts[1:, :].sum(axis=1)[:, None]
    pred_area = counts[:, 1:].sum(axis=0)[None, :]
    iou = overlap / np.maximum(truth_area + pred_area - overlap, 1)

    rows, cols = linear_sum_assignment(-iou)
    matched = iou[rows, cols]
    matched = matched[matched > threshold]

    tp = len(matched)
    fp, fn = n_pred - tp, n_truth - tp
    sq = matched.mean() if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    return dict(pq=sq * rq, sq=sq, rq=rq, tp=tp, fp=fp, fn=fn)


def labels_to_rles(labels):
    """Label image -> list of full-size masks (RLE). Only touches each filament's bounding box."""
    rles = []
    boxes = ndimage.find_objects(labels.astype(np.int32), max_label=int(labels.max()))
    for filament_id, box in enumerate(boxes, start=1):
        if box is None:
            continue
        full = np.zeros((SIZE, SIZE), np.uint8, order="F")
        full[box] = labels[box] == filament_id
        rles.append(mask_utils.encode(full))
    return rles


# %% [markdown]
# ## Predicting and cleaning up the result
#
# The model gives, for every pixel, the chance that it is part of a filament. The clean-up
# turns that map into separate filaments.
#
# **Baseline clean-up:** keep pixels above 0.5, every connected blob is one filament.
#
# **Improved clean-up:**
# 1. *Two-level threshold (hysteresis).* A blob is only kept if part of it is very confident
#    (`thr_high`), but its outline follows the looser `thr_low`. Faint noise disappears,
#    thin faint filament ends stay attached.
# 2. *Optional gap closing* (`close_px`): fills tiny holes.
# 3. *Joining nearby pieces* (`join_px`): pieces closer than this get the same filament id,
#    without filling the gap. This fixes one filament predicted as several islands.
# 4. *Removing small blobs* (`min_area`).


# %%
def cleanup_baseline(prob, threshold=0.5, min_area=20, max_filaments=150):
    """The baseline's clean-up: threshold, then connected blobs."""
    mask = (prob > threshold).astype(np.uint8)
    _, blobs, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_first = np.argsort(-areas)[:max_filaments]

    labels = np.zeros(blobs.shape, np.int32)
    next_id = 1
    for index in largest_first:
        if areas[index] < min_area:
            continue
        labels[blobs == index + 1] = next_id
        next_id += 1
    return labels


def _ellipse(radius):
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _group_pieces(mask, join_px) -> NDArray[np.int32]:
    """Label connected pieces; with join_px, pieces that are close share one label."""
    if join_px == 0:
        pieces = cv2.connectedComponents(mask, connectivity=8)[1]
        return np.asarray(pieces, dtype=np.int32)
    grown = cv2.dilate(mask, _ellipse(join_px))
    groups = np.asarray(cv2.connectedComponents(grown, connectivity=8)[1], dtype=np.int32)
    pieces = groups * mask.astype(np.int32)  # keep only the original pixels: the mask does not grow
    return pieces


def cleanup(prob, thr_low=0.4, thr_high=0.6, min_area=60, join_px=0, close_px=0, max_filaments=150):
    """The improved clean-up. Returns a label image (0 = background, 1..n = filaments)."""
    mask = (prob > thr_low).astype(np.uint8)
    if close_px > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(close_px))

    pieces = _group_pieces(mask, join_px)
    n_pieces = int(pieces.max())
    if n_pieces == 0:
        return np.zeros(prob.shape, np.int32)

    area = np.bincount(pieces.ravel(), minlength=n_pieces + 1)
    confident_pixels = np.bincount(pieces[prob > thr_high], minlength=n_pieces + 1)
    keep = (area >= min_area) & (confident_pixels > 0)
    keep[0] = False

    kept_ids = np.nonzero(keep)[0]
    kept_ids = kept_ids[np.argsort(-area[kept_ids])][:max_filaments]  # largest first
    new_id = np.zeros(n_pieces + 1, np.int32)
    new_id[kept_ids] = np.arange(1, len(kept_ids) + 1)
    return new_id[pieces]
