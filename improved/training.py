# %% [markdown]
# # Training

# %%
import copy
import json
import math
import time
from typing import Any

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from improved.config import (
    CACHE,
    MODELS,
)
from improved.data import (
    TrainingData,
    get_split,
    load_brightness,
    load_index,
    make_folds,
    scale_image,
)
from improved.model import (
    build_model,
    load_weights,
    pick_device,
    predict_probability,
    prune_conv_weights,
    weights_to_save,
)
from improved.scoring import cleanup, cleanup_baseline, pq_fast

# %% [markdown]
# ## Training
#
# Two recipes:
#
# * `baseline`: what `code.ipynb` does, reproduced so we can compare fairly on the same machine.
# * `improved`: our recipe. Every idea can also be switched off on its own (see the
#   options), which is how the "what helped how much" table is made.
#
# | Idea | Baseline | Improved |
# |---|---|---|
# | Labels | each annotator separately | all annotators fused into one soft label |
# | Image scaling | fixed numbers | per-image disk brightness |
# | Augmentation | rotate, flip, brightness | + gamma, blur, noise |
# | Learning rate | 3e-4, cosine step per epoch | warm-up, then cosine step per batch |
# | Encoder learning rate | same as decoder | 0.3 x decoder (it is pretrained) |
# | Weight averaging (EMA) | no | yes (0.999) |
# | Gradient clipping | no | yes (1.0) |
# | Decoder activation | ReLU | SiLU |
# | Mixed precision (16-bit maths) | no | yes |
# | Length | 30 epochs x 80 batches | 40 epochs x 250 batches |
#
# Compression options: `prune` zeroes the smallest weights and fine-tunes, `teacher` trains
# a (smaller) model to copy a bigger one (distillation).


# %%
RECIPES: dict[str, dict[str, Any]] = {
    "baseline": dict(
        epochs=30,
        steps=80,
        bs=8,
        lr=3e-4,
        enc_lr_mult=1.0,
        warmup=0,
        ema=0.0,
        act="relu",
        labels="single",
        norm="fixed",
        aug="basic",
        amp=0,
        encoder="resnet34",
        eval_every=5,
        sched="cosine_epoch",
    ),
    "improved": dict(
        epochs=40,
        steps=250,
        bs=8,
        lr=1e-3,
        enc_lr_mult=0.3,
        warmup=500,
        ema=0.999,
        act="silu",
        labels="fused",
        norm="image",
        aug="strong",
        amp=1,
        encoder="resnet34",
        eval_every=5,
        sched="cosine_step",
    ),
}


def make_config(recipe="improved", **overrides):
    """Start from a recipe and change any setting by name, e.g. make_config('improved', act='relu')."""
    config: dict[str, Any] = dict(RECIPES[recipe])
    config.update(
        recipe=recipe,
        fold=0,
        seed=0,
        init=None,
        prune=0.0,
        teacher=None,
        teacher_encoder="resnet34",
        kd_weight=1.0,
        workers=6,
    )
    config.update({k: v for k, v in overrides.items() if v is not None})
    return config


# %%
def learning_rate_factor(config, step, epoch):
    """How much of the base learning rate to use right now (1.0 = all of it)."""
    if config["sched"] == "cosine_epoch":  # baseline: changes once per epoch
        return 0.5 * (1 + math.cos(math.pi * epoch / config["epochs"]))

    warmup = config["warmup"]
    if step < warmup:  # warm-up: start tiny so the pretrained weights are not wrecked
        return (step + 1) / warmup
    total_steps = config["epochs"] * config["steps"]
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizer(model, config):
    """AdamW. The pretrained encoder gets a smaller learning rate than the new decoder."""
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    other_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    return torch.optim.AdamW(
        [
            dict(params=encoder_params, lr=config["lr"] * config["enc_lr_mult"]),
            dict(params=other_params, lr=config["lr"]),
        ],
        weight_decay=1e-4,
    )


@torch.no_grad()
def update_average(average_model, model, decay):
    """Exponential moving average of the weights: the average is steadier than the latest weights."""
    for average, current in zip(average_model.state_dict().values(), model.state_dict().values()):
        if average.dtype.is_floating_point:
            average.mul_(decay).add_(current.detach(), alpha=1 - decay)
        else:
            average.copy_(current)


class Validator:
    """Scores a model on the validation observations (quick PQ, used while training)."""

    # The improved recipe tries a few clean-up settings and keeps the best, so that choosing
    # the best epoch is not thrown off by the model being a bit more or less confident.
    IMPROVED_SETTINGS: list[dict[str, Any]] = [
        dict(thr_low=low, thr_high=high, min_area=200, join_px=16)
        for low, high in [(0.3, 0.5), (0.4, 0.7), (0.5, 0.8), (0.6, 0.9)]
    ]

    def __init__(self, config, val_obs, ids_of, brightness, device):
        self.config, self.val_obs, self.ids_of = config, val_obs, ids_of
        self.brightness, self.device = brightness, device
        self.images = {obs: np.load(CACHE / "img" / f"{obs}.npy") for obs in val_obs}
        self.labels = {i: np.load(CACHE / "lab" / f"{i}.npy") for o in val_obs for i in ids_of[o]}

    def _scale(self, obs):
        return scale_image(self.images[obs], self.config["norm"], self.brightness[obs])

    def _cleanup_options(self):
        if self.config["recipe"] == "baseline":
            return [None]  # None = the baseline clean-up
        return self.IMPROVED_SETTINGS

    def score(self, model):
        model.eval()
        # the baseline recipe checks on 40 observations, exactly like the notebook
        observations = self.val_obs[:40] if self.config["recipe"] == "baseline" else self.val_obs
        options = self._cleanup_options()
        rows = [[] for _ in options]

        for obs in observations:
            prob = predict_probability(model, self._scale(obs), self.device)
            for option_rows, settings in zip(rows, options):
                predicted = (
                    cleanup_baseline(prob) if settings is None else cleanup(prob, **settings)
                )
                option_rows.extend(pq_fast(self.labels[i], predicted) for i in self.ids_of[obs])

        model.train()
        averages = [pd.DataFrame(r).mean(numeric_only=True) for r in rows]
        return max(averages, key=lambda a: a.pq)


# %%
def train(config, name=None):
    """Train one model. Saves outputs/models/<name>.pt and <name>.json. Returns the run record."""
    name = name or f"{config['recipe']}_f{config['fold']}"
    print("config:", json.dumps(config), flush=True)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = pick_device()
    print(
        "GPU:" if device == "cuda" else "WARNING: no GPU, training on CPU is very slow:",
        torch.cuda.get_device_name(0) if device == "cuda" else "",
        flush=True,
    )

    # data
    imgs, _ = load_index()
    train_obs, val_obs = get_split(make_folds(imgs), config["fold"])
    ids_of = imgs.groupby("obs")["id"].apply(list).to_dict()
    brightness = load_brightness()
    print(f"fold {config['fold']}: train obs {len(train_obs)}, val obs {len(val_obs)}", flush=True)

    # model
    model = build_model(config["encoder"], "imagenet", config["act"])
    if config["init"]:
        model.load_state_dict(torch.load(config["init"], map_location="cpu"))
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.2f} M", flush=True)

    if config["prune"] > 0:
        assert config["ema"] == 0, "use ema=0 together with pruning"
        prune_conv_weights(model, config["prune"])
    pruned = config["prune"] > 0

    teacher = None
    if config["teacher"]:
        teacher = build_model(config["teacher_encoder"], None, config["act"])
        teacher = load_weights(teacher, config["teacher"]).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    average_model = None
    if config["ema"] > 0:
        average_model = copy.deepcopy(model).eval()
        for p in average_model.parameters():
            p.requires_grad_(False)

    optimizer = make_optimizer(model, config)
    base_rates = [group["lr"] for group in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"]) and device == "cuda")
    amp_on = scaler.is_enabled()
    bce = nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss("binary", from_logits=True)
    validator = Validator(config, val_obs, ids_of, brightness, device)

    loader = DataLoader(
        TrainingData(config, train_obs, ids_of, brightness, config["steps"] * config["bs"]),
        batch_size=config["bs"],
        num_workers=config["workers"],
        drop_last=True,
        persistent_workers=config["workers"] > 0,
    )

    MODELS.mkdir(parents=True, exist_ok=True)
    checkpoint = MODELS / f"{name}.pt"
    history, best_pq, step, start = [], -1.0, 0, time.time()
    model.train()

    for epoch in range(config["epochs"]):
        epoch_start, loss_sum = time.time(), 0.0
        for x, y in loader:
            factor = learning_rate_factor(config, step, epoch)
            for group, base in zip(optimizer.param_groups, base_rates):
                group["lr"] = base * factor

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp_on):
                logits = model(x)
            logits = logits.float()
            loss = bce(logits, y) + dice(logits, y)

            if teacher is not None:  # distillation: also copy the teacher's softer predictions
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp_on):
                    teacher_prob = torch.sigmoid(teacher(x).float())
                loss = loss + config["kd_weight"] * bce(logits, teacher_prob)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if config["recipe"] == "improved":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if average_model is not None:
                update_average(average_model, model, config["ema"])

            loss_sum += loss.item()
            step += 1

        record = dict(
            epoch=epoch + 1,
            loss=loss_sum / config["steps"],
            lr=optimizer.param_groups[1]["lr"],
            epoch_s=time.time() - epoch_start,
            total_s=time.time() - start,
        )
        message = (
            f"epoch {epoch + 1}/{config['epochs']} loss {record['loss']:.4f} "
            f"({record['epoch_s']:.0f}s, total {record['total_s'] / 60:.1f} min)"
        )

        is_check_epoch = (epoch + 1) % config["eval_every"] == 0 or epoch + 1 == config["epochs"]
        if is_check_epoch:
            scored_model = average_model if average_model is not None else model
            result = validator.score(scored_model)
            record.update(
                val_pq=result.pq,
                val_sq=result.sq,
                val_rq=result.rq,
                val_fp=result.fp,
                val_fn=result.fn,
            )
            message += (
                f" | val PQ {result.pq:.4f} SQ {result.sq:.3f} RQ {result.rq:.3f} "
                f"fp {result.fp:.1f} fn {result.fn:.1f}"
            )
            if result.pq > best_pq:
                best_pq = result.pq
                to_save = (
                    weights_to_save(scored_model)
                    if pruned and scored_model is model
                    else scored_model.state_dict()
                )
                torch.save(to_save, checkpoint)
                message += "  (saved)"
        history.append(record)
        print(message, flush=True)

    minutes = (time.time() - start) / 60
    run = dict(
        config=config,
        name=name,
        best_val_pq_quick=best_pq,
        train_minutes=minutes,
        params_m=n_params / 1e6,
        log=history,
    )
    (MODELS / f"{name}.json").write_text(json.dumps(run, indent=1))
    print(
        f"finished in {minutes:.1f} min, best quick val PQ {best_pq:.4f}, saved {checkpoint}",
        flush=True,
    )
    return run
