import os
import torch

IMG_SIZE = 224
CLASSES = ["cat", "dog"]
NUM_CLASSES = len(CLASSES)
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HF_MODEL_REPO = "AyushDocs/cat-dog-classifier"

MODEL_MAP = {
    "MLP Classifier": "mlp_classifier.pth",
    "Simple CNN": "simple_cnn.pth",
    "Transfer Learning (FC only)": "transfer_learning_fc_only.pth",
    "Transfer Learning (Layer4 + FC)": "transfer_learning_layer4_fc.pth",
}
