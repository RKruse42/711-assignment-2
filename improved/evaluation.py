# %% [markdown]
# # Evaluation, test predictions and reports

# %%
import io
import json
import multiprocessing as mp
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from improved.config import CACHE, EVAL, MODELS, OUT, PREDICTIONS, REPORTS, TEST_DIR
from improved.data import (
    get_split,
    ground_truth_rles,
    load_brightness,
    load_index,
    make_folds,
    read_gray,
    scale_image,
)
from improved.model import build_model, load_weights, pick_device, predict_probability
from improved.scoring import (
    cleanup,
    cleanup_baseline,
    labels_to_rles,
    pq_fast,
    pq_score,
)

# %% [markdown]
# ## Evaluating and tuning the clean-up
#
# `evaluate` scores a trained model on its validation fold with the notebook's exact `pq_score`.
#
# With `tune=True` it also searches the clean-up settings. To keep the score honest, the
# validation observations are split into two halves: settings are chosen on one half and
# scored on the other, then the other way round. The reported **honest PQ** is the average
# of those two held-out scores.

# %%
SEARCH_GRID = dict(
    thr_low=[0.3, 0.4, 0.5, 0.6, 0.7],
    thr_high=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995],
    min_area=[20, 60, 120, 200, 300, 450, 600, 800],
    join_px=[0, 4, 8, 16, 24, 32, 48],
    close_px=[0, 2, 4],
)

_EVAL = {}  # probabilities, labels, ids: filled in before the helper processes start


def _worker_pool(workers=10):
    """Helper processes that share the data above by forking.

    Windows cannot fork. There we return None and the work runs in one process:
    slower, but the same answers."""
    if "fork" not in mp.get_all_start_methods():
        return nullcontext(None)
    return mp.get_context("fork").Pool(workers)


def _score_observation(job):
    obs, settings = job
    prob = _EVAL["probs"][obs].astype(np.float32)
    predicted = cleanup_baseline(prob) if settings == "baseline" else cleanup(prob, **settings)
    return [pq_fast(_EVAL["labels"][i], predicted)["pq"] for i in _EVAL["ids_of"][obs]]


def _quick_score(observations, settings, pool):
    jobs = [(obs, settings) for obs in observations]
    per_observation = pool.map(_score_observation, jobs) if pool else map(_score_observation, jobs)
    return float(np.mean([pq for scores in per_observation for pq in scores]))


def search_cleanup_settings(observations, pool):
    """Try one setting at a time (two rounds) and keep whatever raises PQ."""
    best = dict(thr_low=0.5, thr_high=0.5, min_area=20, join_px=0, close_px=0)
    best_score = _quick_score(observations, best, pool)
    for _ in range(2):
        for name, values in SEARCH_GRID.items():
            for value in values:
                candidate = dict(best, **{name: value})
                if candidate["thr_high"] < candidate["thr_low"]:
                    continue
                score = _quick_score(observations, candidate, pool)
                if score > best_score + 1e-5:
                    best, best_score = candidate, score
    return best


def exact_scores(observations, settings, ids_of, anns):
    """Exact PQ (pq_score on the original polygons) for every annotation set."""
    wanted = [i for obs in observations for i in ids_of[obs]]
    truth = ground_truth_rles(anns[anns["image_id"].isin(wanted)])
    rows, cleanup_seconds = [], []
    for obs in observations:
        started = time.time()
        prob = _EVAL["probs"][obs].astype(np.float32)
        labels = cleanup_baseline(prob) if settings == "baseline" else cleanup(prob, **settings)
        predicted = labels_to_rles(labels)
        cleanup_seconds.append(time.time() - started)
        for annotation_id in ids_of[obs]:
            row = pq_score(truth.get(annotation_id, []), predicted)
            row.update(id=annotation_id, obs=obs, n_pred=len(predicted))
            rows.append(row)
    return pd.DataFrame(rows), float(np.mean(cleanup_seconds))


def evaluate(
    checkpoint,
    recipe=None,
    tta=False,
    tune=False,
    post=None,
    baseline_cleanup=False,
    tag="",
):
    """Score a trained model. Writes outputs/eval/<tag>.json and <tag>_per_image.csv."""
    checkpoint = Path(checkpoint)
    record_path = checkpoint.with_suffix(".json")
    config = json.loads(record_path.read_text())["config"] if record_path.exists() else {}
    recipe = recipe or config.get("recipe", "baseline")
    fold = config.get("fold", 0)
    norm = config.get("norm", "fixed" if recipe == "baseline" else "image")

    device = pick_device()
    imgs, anns = load_index()
    _, val_obs = get_split(make_folds(imgs), fold)
    ids_of = imgs.groupby("obs")["id"].apply(list).to_dict()
    brightness = load_brightness()

    model = build_model(config.get("encoder", "resnet34"), None, config.get("act", "relu"))
    model = load_weights(model, checkpoint).to(device).eval()

    # 1. network output for every validation image (timed)
    started = time.time()
    probs = {}
    for obs in val_obs:
        image = np.load(CACHE / "img" / f"{obs}.npy")
        x = scale_image(image, norm, brightness[obs])
        probs[obs] = predict_probability(model, x, device, tta).astype(np.float16)
    network_seconds = (time.time() - started) / len(val_obs)
    print(f"network time per image: {network_seconds:.3f}s (tta={tta})", flush=True)

    _EVAL.update(
        probs=probs,
        ids_of=ids_of,
        labels={i: np.load(CACHE / "lab" / f"{i}.npy") for o in val_obs for i in ids_of[o]},
    )

    # 2. choose clean-up settings
    default_settings = dict(thr_low=0.4, thr_high=0.6, min_area=60, join_px=0, close_px=0)
    result: dict[str, Any] = dict(ckpt=str(checkpoint), recipe=recipe, fold=fold, tta=tta)
    settings: Any
    with _worker_pool() as pool:
        if baseline_cleanup:
            settings = "baseline"
        elif post:
            settings = json.loads(post)
        else:
            settings = default_settings

        if tune:
            half_a, half_b = val_obs[0::2], val_obs[1::2]
            from_a = search_cleanup_settings(half_a, pool)
            from_b = search_cleanup_settings(half_b, pool)
            held_out = pd.concat(
                [
                    exact_scores(half_b, from_a, ids_of, anns)[0],
                    exact_scores(half_a, from_b, ids_of, anns)[0],
                ]
            )
            settings = search_cleanup_settings(val_obs, pool)  # used from now on (e.g. test set)
            result.update(
                post_from_half_a=from_a,
                post_from_half_b=from_b,
                honest_pq=float(held_out.pq.mean()),
                honest_sq=float(held_out.sq.mean()),
                honest_rq=float(held_out.rq.mean()),
            )
            print("settings tuned on half A:", from_a, "\nsettings tuned on half B:", from_b)
            print(
                f"HONEST PQ (tuned on one half, scored on the other): {held_out.pq.mean():.4f}",
                flush=True,
            )

    # 3. final exact score
    scores, cleanup_seconds = exact_scores(val_obs, settings, ids_of, anns)
    result.update(
        post=settings,
        pq=float(scores.pq.mean()),
        pq_std_over_images=float(scores.pq.std()),
        sq=float(scores.sq.mean()),
        rq=float(scores.rq.mean()),
        fp=float(scores.fp.mean()),
        fn=float(scores.fn.mean()),
        splits=float(scores.one_to_many.mean()),
        merges=float(scores.many_to_one.mean()),
        n_pred=float(scores.n_pred.mean()),
        net_s_per_img=network_seconds,
        post_s_per_img=cleanup_seconds,
        n_sets=len(scores),
    )
    print(json.dumps(result, indent=1), flush=True)

    tag = tag or checkpoint.stem
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / f"{tag}.json").write_text(json.dumps(result, indent=1))
    scores.to_csv(EVAL / f"{tag}_per_image.csv", index=False)
    return result


# %% [markdown]
# ## Predicting the test set
#
# Predicts the 180 test images and times the whole pipeline (read image, scale, network,
# clean-up, encode). Predictions are written in the standard COCO "results" format
# (one entry per filament, mask as RLE).
#
# **Note:** check the exact `submission.csv` layout on the Kaggle Data page and convert these
# predictions to it before uploading.


# %%
def predict_test(checkpoint, post_json, tta=False):
    checkpoint = Path(checkpoint)
    config = json.loads(checkpoint.with_suffix(".json").read_text())["config"]
    settings = json.loads(Path(post_json).read_text())["post"]
    device = pick_device()

    started = time.time()
    model = build_model(config["encoder"], None, config["act"])
    model = load_weights(model, checkpoint).to(device).eval()
    load_seconds = time.time() - started

    results, seconds_per_image = [], []
    files = sorted(p.name for p in TEST_DIR.iterdir())
    for file_name in files:
        image_started = time.time()
        image = read_gray(TEST_DIR / file_name)
        x = scale_image(image, config["norm"])
        prob = predict_probability(model, x, device, tta)
        labels = cleanup_baseline(prob) if settings == "baseline" else cleanup(prob, **settings)
        for rle in labels_to_rles(labels):
            results.append(
                dict(
                    image_id=file_name.rsplit(".", 1)[0],
                    file_name=file_name,
                    category_id=1,
                    segmentation=dict(size=rle["size"], counts=rle["counts"].decode("ascii")),
                )
            )
        seconds_per_image.append(time.time() - image_started)

    timing = dict(
        images=len(files),
        filaments=len(results),
        model_load_s=load_seconds,
        mean_s_per_image=float(np.mean(seconds_per_image)),
        total_s=time.time() - started,
        tta=tta,
        device=device,
    )
    name = checkpoint.stem + ("_tta" if tta else "")
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    (PREDICTIONS / f"test_predictions_{name}.json").write_text(json.dumps(results))
    (PREDICTIONS / f"test_timing_{name}.json").write_text(json.dumps(timing, indent=1))
    print(json.dumps(timing, indent=1))
    return timing


# %% [markdown]
# ## Reports
#
# * `size_report`: number of weights, share of zero weights and file size of each model
#   (32-bit, 16-bit, and 16-bit zipped; zip shrinks pruned models because zeros compress well).


# %%
def _saved_size_mb(state, zipped=False):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    if not zipped:
        return buffer.tell() / 1e6
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("model.pt", buffer.getvalue())
    return archive.tell() / 1e6


def size_report(names=None):
    names = names or [
        "baseline_f0",
        "improved_f0",
        "pruned5_f0",
        "pruned7_f0",
        "pruned85_f0",
        "student_r18_f0",
    ]
    rows = []
    for name in names:
        path = OUT / f"{name}.pt" if name == "baseline_f0" else MODELS / f"{name}.pt"
        if not path.exists():
            continue
        state = torch.load(path, map_location="cpu")
        weights = [v for v in state.values() if v.dtype.is_floating_point]
        total = sum(v.numel() for v in weights)
        zeros = sum(int((v == 0).sum()) for v in weights)
        full = {k: v.float() if v.dtype.is_floating_point else v for k, v in state.items()}
        half = {k: v.half() if v.dtype.is_floating_point else v for k, v in state.items()}
        rows.append(
            dict(
                model=name,
                params_m=total / 1e6,
                zero_percent=100 * zeros / total,
                mb_fp32=_saved_size_mb(full),
                mb_fp16=_saved_size_mb(half),
                mb_fp16_zipped=_saved_size_mb(half, zipped=True),
            )
        )
        print(rows[-1], flush=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "size_report.json").write_text(json.dumps(rows, indent=1))
    return rows
