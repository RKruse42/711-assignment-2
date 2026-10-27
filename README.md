## Running the tests

### 1. Set up the environment

```bash
python3 -m venv venv
source venv/bin/activate
```

> On Windows, use `python` instead of `python3` for every command below.

### 2. Install dependencies

```bash
pip install numpy pandas pycocotools matplotlib scipy opencv-python
pip install --no-deps segmentation-models-pytorch
pip install timm safetensors huggingface_hub --no-deps
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install tqdm pyyaml click hf-xet httpx2
```
