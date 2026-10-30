# %% [markdown]
# # Data
#
# Reading the data, the validation folds, image scaling, the fast data cache, facts about
# the data, and the random crops used for training.

# %%
import json
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from improved.config import ANN_FILE, CACHE, IMG_DIR, REPORTS, SIZE, SKY

# %% [markdown]
# ## Data: index and validation folds
#
# The folds are exactly the same as in `code.ipynb`: five folds by calendar month, and a
# two-day gap between training and validation days. All annotations of one observation
# always stay on the same side of the split.


# %%
def load_index():
    """Read the COCO json. Returns (images table, annotations table)."""
    coco = json.loads(ANN_FILE.read_text())
    imgs = pd.DataFrame(coco["images"])
    anns = pd.DataFrame(coco["annotations"])
    parts = imgs["id"].str.split("-", n=1)
    imgs["annotator"] = parts.str[0]
    imgs["obs"] = parts.str[1]
    imgs["time"] = pd.to_datetime(imgs["obs"].str[:14], format="%Y%m%d%H%M%S")
    imgs["instrument"] = imgs["obs"].str[14:]
    return imgs, anns


def make_folds(imgs, n_folds=5):
    """Give every observation a fold number, keeping whole months together."""
    obs = imgs.drop_duplicates("obs")[["obs", "time", "instrument"]].copy()
    obs["month"] = obs["time"].dt.to_period("M").astype(str)

    month_sizes = obs["month"].value_counts()
    shuffled = month_sizes.sample(frac=1, random_state=42)
    biggest_first = shuffled.sort_values(ascending=False, kind="stable").index

    load = np.zeros(n_folds)
    month_fold = {}
    for month in biggest_first:
        fold = int(np.argmin(load))  # put the month in the emptiest fold
        month_fold[month] = fold
        load[fold] += month_sizes[month]

    obs["fold"] = obs["month"].map(month_fold)
    return obs


def get_split(obs, fold, gap_days=2):
    """Lists of training and validation observations for one fold."""
    val = obs[obs["fold"] == fold]
    train = obs[obs["fold"] != fold]

    gaps = np.abs(train["time"].values[:, None] - val["time"].values[None, :])
    far_enough = gaps.min(axis=1) > np.timedelta64(gap_days, "D")
    return train[far_enough]["obs"].tolist(), val["obs"].tolist()


def ground_truth_rles(anns):
    """image id -> list of filament masks (one per annotated filament)."""
    rles = {}
    for image_id, group in anns.groupby("image_id"):
        rles[image_id] = [
            mask_utils.merge(mask_utils.frPyObjects(polygon, SIZE, SIZE))
            for polygon in group["segmentation"]
        ]
    return rles


# %% [markdown]
# ## Image scaling
#
# The baseline scales every image with the same fixed numbers. We scale each image by the
# brightness and contrast of its own solar disk, so images from the six telescopes look
# more alike to the model.


# %%
def scale_fixed(img):
    """The baseline's scaling."""
    return (img.astype(np.float32) / 255.0 - 0.45) / 0.25


def disk_stats(img):
    """Mean and spread of the brightness on the solar disk (the black sky is ignored)."""
    on_disk = img > SKY
    if on_disk.sum() < 1000:
        return float(img.mean()), float(img.std() + 1e-6)
    pixels = img[on_disk].astype(np.float32)
    return float(pixels.mean()), float(pixels.std() + 1e-6)


def scale_per_image(img, mean, std):
    """Improved scaling: use the image's own disk brightness."""
    x = (img.astype(np.float32) - mean) / std
    x[img <= SKY] = -3.0  # the sky is always one fixed 'very dark' value
    return np.clip(x, -6, 6).astype(np.float32)


def scale_image(image, norm, mean_std=None):
    """Scale an image the way the model was trained: norm is 'fixed' or 'image'.

    For 'image' scaling, mean_std is the (mean, spread) of the whole image's solar disk.
    Pass it when `image` is only a crop; otherwise it is measured from the image itself."""
    if norm == "fixed":
        return scale_fixed(image)
    mean, std = mean_std if mean_std else disk_stats(image)
    return scale_per_image(image, mean, std)


def load_brightness():
    """Disk brightness (mean, spread) of every observation, from the data cache."""
    table = pd.read_csv(CACHE / "stats.csv")
    rows = zip(table["obs"], table["mean"], table["std"])
    return {str(obs): (float(mean), float(std)) for obs, mean, std in rows}


def read_gray(path):
    """Read an image as grey levels, failing loudly if the file is missing or broken."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


# %% [markdown]
# ## Preparing the data cache and analysing the data
#
# Reading JPEGs and drawing polygons for every training sample is slow. `prepare_cache`
# converts everything once into fast `.npy` files in `cache/improved/`:
#
# * `img/<observation>.npy` the image
# * `lab/<annotation set>.npy` one person's filaments (0 = background, 1..n = filament id)
# * `soft/<observation>.npy` the **fused label**: the share of annotators (0-255) who marked
#   each pixel
# * `stats.csv` brightness of each image's solar disk

# %%
_POLYGONS = {}  # annotation set id -> list of polygons (filled before the worker pool starts)


def _cache_one_observation(job):
    obs, annotation_ids, file_name = job
    image = read_gray(IMG_DIR / file_name)
    np.save(CACHE / "img" / f"{obs}.npy", image)

    votes = np.zeros((SIZE, SIZE), np.float32)
    for annotation_id in annotation_ids:
        labels = np.zeros((SIZE, SIZE), np.uint8)
        for number, polygon in enumerate(_POLYGONS.get(annotation_id, []), start=1):
            rle = mask_utils.merge(mask_utils.frPyObjects(polygon, SIZE, SIZE))
            labels[mask_utils.decode(rle) > 0] = number
        np.save(CACHE / "lab" / f"{annotation_id}.npy", labels)
        votes += labels > 0

    fused = np.round(255 * votes / len(annotation_ids)).astype(np.uint8)
    np.save(CACHE / "soft" / f"{obs}.npy", fused)

    mean, std = disk_stats(image)
    return dict(obs=obs, mean=mean, std=std, disk_frac=float((image > SKY).mean()))


def prepare_cache(workers=8):
    imgs, anns = load_index()
    _POLYGONS.update({k: list(g["segmentation"]) for k, g in anns.groupby("image_id")})

    for folder in ("img", "soft", "lab"):
        (CACHE / folder).mkdir(parents=True, exist_ok=True)

    jobs = [
        (obs, list(group["id"]), group["file_name"].iloc[0]) for obs, group in imgs.groupby("obs")
    ]
    with Pool(min(workers, 12)) as pool:
        rows = []
        for done, row in enumerate(pool.imap_unordered(_cache_one_observation, jobs), start=1):
            rows.append(row)
            if done % 100 == 0:
                print(f"{done}/{len(jobs)}", flush=True)

    pd.DataFrame(rows).to_csv(CACHE / "stats.csv", index=False)
    print("done:", len(rows), "observations")


def _touches_another(labels, filament_id, neighbourhood=np.ones((3, 3), np.uint8)):
    """Does this filament come within 2 pixels of a different filament?"""
    mask = (labels == filament_id).astype(np.uint8)
    if mask.sum() == 0:
        return False
    ring = cv2.dilate(mask, neighbourhood, iterations=2).astype(bool) & ~mask.astype(bool)
    neighbours = labels[ring]
    return bool(((neighbours > 0) & (neighbours != filament_id)).any())


def analyse_data():
    """Facts about the training data that motivate the improvements -> reports/data_facts.json."""
    imgs, anns = load_index()
    stats = pd.read_csv(CACHE / "stats.csv")
    stats["instrument"] = stats["obs"].str[14:]
    ids_of = imgs.groupby("obs")["id"].apply(list).to_dict()

    filament_share, agreement = [], []
    touching, total_filaments = 0, 0
    for obs, annotation_ids in ids_of.items():
        label_images = [np.load(CACHE / "lab" / f"{i}.npy") for i in annotation_ids]
        disk_pixels = (np.load(CACHE / "img" / f"{obs}.npy") > SKY).sum()

        for labels in label_images:
            filament_share.append((labels > 0).sum() / max(disk_pixels, 1))
            total_filaments += int(labels.max())
            touching += sum(_touches_another(labels, k) for k in range(1, int(labels.max()) + 1))

        # how much do two annotators of the same image agree? (overlap / union)
        for i, a in enumerate(label_images):
            for b in label_images[i + 1 :]:
                union = ((a > 0) | (b > 0)).sum()
                agreement.append(((a > 0) & (b > 0)).sum() / max(union, 1))

    brightness_by_instrument = {}
    for name, group in stats.groupby("instrument"):
        means = group["mean"].to_numpy(dtype=float)
        brightness_by_instrument[str(name)] = dict(
            mean_brightness=float(means.mean()),
            brightness_range=[float(means.min()), float(means.max())],
            contrast=float(group["std"].to_numpy(dtype=float).mean()),
        )
    all_means = stats["mean"].to_numpy(dtype=float)
    areas = anns["area"].to_numpy(dtype=float)
    facts = dict(
        disk_brightness_by_instrument=brightness_by_instrument,
        disk_brightness_overall_range=[float(all_means.min()), float(all_means.max())],
        filament_pixels_percent_of_disk=dict(
            mean=100 * float(np.mean(filament_share)), median=100 * float(np.median(filament_share))
        ),
        annotator_pixel_agreement_iou=dict(
            pairs=len(agreement),
            mean=float(np.mean(agreement)),
            median=float(np.median(agreement)),
            share_below_0_5=float(np.mean(np.array(agreement) < 0.5)),
        ),
        filaments_touching_another_within_2px_percent=100 * touching / total_filaments,
        small_filaments=dict(
            area_below_200px_percent=100 * float((areas < 200).mean()),
            area_below_60px_percent=100 * float((areas < 60).mean()),
        ),
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "data_facts.json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))
    return facts


# %%
CROP = 512  # training crops are 512 x 512 pixels


class TrainingData(Dataset):
    """Random 512 x 512 crops, with augmentation. Cropping favours areas with filaments."""

    def __init__(self, config, train_obs, ids_of, brightness, n_samples):
        self.config = config
        self.brightness = brightness
        self.n_samples = n_samples
        if config["labels"] == "single":
            # baseline: every annotation set is its own example
            self.items = [
                (obs, annotation_id) for obs in train_obs for annotation_id in ids_of[obs]
            ]
        else:
            # improved: one example per observation, all annotators fused
            self.items = [(obs, None) for obs in train_obs]

    def __len__(self):
        return self.n_samples

    def _scale(self, image, obs):
        if self.config["norm"] == "fixed":
            return scale_fixed(image)
        return scale_image(image, self.config["norm"], self.brightness[obs])

    @staticmethod
    def _crop_corner(rng, target):
        """Top-left corner of a crop. 70% of the time it contains a filament."""
        random_corner = rng.integers(0, SIZE - CROP + 1, size=2)
        if rng.random() >= 0.7:
            return random_corner
        ys, xs = np.nonzero(target)
        if len(ys) == 0:
            return random_corner
        k = rng.integers(len(ys))
        y0 = int(np.clip(ys[k] - rng.integers(CROP), 0, SIZE - CROP))
        x0 = int(np.clip(xs[k] - rng.integers(CROP), 0, SIZE - CROP))
        return y0, x0

    def _stronger_picture_changes(self, rng, image):
        """Extra augmentation: gamma (like different telescopes) and blur (poor air quality)."""
        if rng.random() < 0.5:
            gamma = rng.uniform(0.7, 1.4)
            image = (255.0 * (image / 255.0) ** gamma).astype(np.uint8)
        if rng.random() < 0.3:
            image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.5, 1.5))
        return image

    def __getitem__(self, _index):
        rng = np.random.default_rng()
        obs, annotation_id = self.items[rng.integers(len(self.items))]
        image = np.load(CACHE / "img" / f"{obs}.npy", mmap_mode="r")
        if annotation_id is None:
            target = np.load(CACHE / "soft" / f"{obs}.npy", mmap_mode="r")
        else:
            target = np.load(CACHE / "lab" / f"{annotation_id}.npy", mmap_mode="r")

        y0, x0 = self._crop_corner(rng, target)
        image = np.array(image[y0 : y0 + CROP, x0 : x0 + CROP])
        target = np.array(target[y0 : y0 + CROP, x0 : x0 + CROP])
        if annotation_id is None:
            target = target.astype(np.float32) / 255.0  # share of annotators
        else:
            target = (target > 0).astype(np.float32)

        quarter_turns = rng.integers(4)
        image, target = np.rot90(image, quarter_turns), np.rot90(target, quarter_turns)
        if rng.random() < 0.5:
            image, target = image[:, ::-1], target[:, ::-1]
        image = np.ascontiguousarray(image)

        strong = self.config["aug"] == "strong"
        if strong:
            image = self._stronger_picture_changes(rng, image)

        x = self._scale(image, obs)
        x = x * rng.uniform(0.85, 1.15) + rng.uniform(-0.2, 0.2)  # contrast and brightness
        if strong and rng.random() < 0.3:
            x = x + rng.normal(0, rng.uniform(0.02, 0.1), x.shape)

        return (
            torch.from_numpy(x.astype(np.float32))[None],
            torch.from_numpy(np.ascontiguousarray(target))[None],
        )
