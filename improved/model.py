# %% [markdown]
# # The neural network
#
# Only the network itself: how it is built, loaded, run, and made smaller.
# Nothing here knows about scoring, validation or the test set.

# %%
import copy
from typing import Any

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# %% [markdown]
# ## The model
#
# A U-Net with a ResNet34 encoder, as in the baseline. Two small differences: the decoder can
# use a smoother activation (SiLU instead of ReLU), and the model can be built without
# downloading the ImageNet starting weights (used when loading our own trained weights).


# %%
def pick_device():
    """'cuda' is also the name PyTorch uses for AMD GPUs when it is built for ROCm."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(
    encoder: str = "resnet34",
    weights: str | None = "imagenet",
    act: str = "relu",
) -> nn.Module:
    options: dict[str, Any] = dict(
        encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=1
    )
    model = smp.Unet(**options)
    if act == "silu":
        use_silu_in_decoder(model)
    return model


def use_silu_in_decoder(model):
    """Replace ReLU by the smoother SiLU in the decoder only.

    The encoder keeps ReLU because its pretrained weights were learned with it."""
    for part in (model.decoder, model.segmentation_head):
        for module in part.modules():
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(module, name, nn.SiLU())


def load_weights(model, path):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict({key: value.float() for key, value in state.items()})
    return model


# %%
FLIPS = [[], [3], [2], [2, 3]]  # as is, left-right, up-down, both


@torch.no_grad()
def predict_probability(model, x, device, tta=False):
    """Chance of 'filament' for every pixel. With tta=True, average over 4 flipped views."""
    x = torch.from_numpy(x)[None, None].to(device)
    views = FLIPS if tta else FLIPS[:1]
    probabilities = []
    for dims in views:
        view = torch.flip(x, dims) if dims else x
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            probability = torch.sigmoid(model(view).float())
        probabilities.append(torch.flip(probability, dims) if dims else probability)
    average = torch.stack(probabilities).mean(dim=0)
    return average[0, 0].float().cpu().numpy()


def prune_conv_weights(model, amount):
    """Zero the smallest `amount` (0-1) of all convolution weights, across the whole model."""
    targets = [(m, "weight") for m in model.modules() if isinstance(m, nn.Conv2d)]
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=amount)
    zeros = sum(int((m.weight == 0).sum()) for m, _ in targets)
    total = sum(m.weight.numel() for m, _ in targets)
    print(f"pruned {100 * zeros / total:.1f}% of conv weights", flush=True)


def weights_to_save(model):
    """State dict with pruning made permanent (plain weight tensors that contain zeros)."""
    pruned_model = copy.deepcopy(model)
    for module in pruned_model.modules():
        if isinstance(module, nn.Conv2d) and prune.is_pruned(module):
            prune.remove(module, "weight")
    return pruned_model.state_dict()
