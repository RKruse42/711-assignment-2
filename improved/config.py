# %% [markdown]
# # Paths and settings
#
# Every file location and the two image constants live here.

# %%
from pathlib import Path

# %%
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:  # running inside the notebook, which sits in the repo root
    ROOT = Path.cwd()

IMG_DIR = ROOT / "data/train/train_images"
TEST_DIR = ROOT / "data/test/test_images"
ANN_FILE = ROOT / "data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
CACHE = ROOT / "cache/improved"

OUT = ROOT / "outputs"
MODELS = OUT / "models"  # trained weights + a record of each training run
EVAL = OUT / "eval"  # scores
PREDICTIONS = OUT / "predictions"  # predicted filaments for the test images
REPORTS = OUT / "reports"  # data facts, model sizes, results table

SIZE = 2048  # every image is 2048 x 2048 pixels
SKY = 20  # pixels darker than this are the black sky around the sun
