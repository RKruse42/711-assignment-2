## Running the tests

### 1. Set up the environment

```bash
python3 -m venv venv
source venv/bin/activate
```

> On Windows, use `python` instead of `python3` for every command below.

### 2. Install dependencies

```bash
pip install numpy pandas pycocotools matplotlib scipy
pip install --no-deps segmentation-models-pytorch
pip install timm safetensors huggingface_hub --no-deps
```
