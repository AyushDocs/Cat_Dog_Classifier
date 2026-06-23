---
title: Cat vs Dog Classifier
emoji: 🐱
colorFrom: pink
colorTo: blue
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: true
license: mit
---

# Cat vs Dog Classifier

A PyTorch image classifier that distinguishes cats from dogs, trained on the [Kaggle Cat and Dog dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog).

**[Live Demo](https://huggingface.co/spaces/AyushDocs/cat-dog-classifier)**

## Models

Four architectures were trained and compared:

| Model | Val Accuracy | Params | Description |
|-------|-------------|--------|-------------|
| MLP Classifier | 61.1% | ~77M | Fully connected baseline |
| Simple CNN | 72.6% | ~3.6M | 2-layer CNN from scratch |
| Transfer Learning (FC only) | 97.6% | ~11.2M | Frozen ResNet18, FC head only |
| Transfer Learning (Layer4 + FC) | **98.1%** | ~11.2M | ResNet18 with layer4 unfrozen |

## Results

| Model | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| MLP | 66.1% | 61.1% | 123.8 | 36.2 |
| CNN | 88.6% | 70.4% | 53.0 | 36.0 |
| Transfer Learning (FC only) | 96.5% | 97.2% | 18.1 | 3.7 |
| Transfer Learning (Layer4 + FC) | **98.6%** | **98.1%** | 8.1 | 3.0 |

## How It Works

1. **Data** — 8,007 training + 2,025 test images from Kaggle (cats/dogs)
2. **Preprocessing** — Resize to 224x224, normalize to [-1, 1]
3. **Training** — Adam optimizer (lr=0.001), CrossEntropyLoss
4. **Best model** — ResNet18 with layer4 + FC unfrozen achieves 98% accuracy

## Local Setup

```bash
git clone https://github.com/AyushDocs/Cat_Dog_Classifier.git
cd Cat_Dog_Classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the Gradio app
python app.py
```

## Inference

```python
from src.utils import load_model_weights, predict
from PIL import Image

model = load_model_weights("Transfer Learning (Layer4 + FC)")
image = Image.open("path/to/image.jpg")
result = predict(model, image)
print(f"Prediction: {result['class']} ({result['confidence']:.1%})")
```

## Training

See the notebook in `research/01_AyushDocs_Classifier.ipynb` for full training code. The dataset is the [Kaggle Cat and Dog dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog) (`tongpython/cat-and-dog`).

## Model Files

Model weights are hosted on HuggingFace Hub: [`AyushDocs/cat-dog-classifier`](https://huggingface.co/AyushDocs/cat-dog-classifier)
