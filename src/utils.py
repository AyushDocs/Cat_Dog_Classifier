import torch
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download

from src.config import IMG_SIZE, MEAN, STD, CLASSES, HF_MODEL_REPO, MODEL_MAP, DEVICE


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = get_transform()
    return transform(image).unsqueeze(0)


def load_model_weights(model_name: str) -> torch.nn.Module:
    from src.model import get_model

    filename = MODEL_MAP[model_name]
    model = get_model(model_name)

    cache_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=filename)
    state_dict = torch.load(cache_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict(model: torch.nn.Module, image: Image.Image) -> dict:
    tensor = preprocess_image(image).to(DEVICE)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).cpu().squeeze()
    confidence, predicted_idx = torch.max(probs, dim=0)
    predicted_class = CLASSES[predicted_idx.item()]
    return {
        "class": predicted_class,
        "confidence": confidence.item(),
        "probabilities": {CLASSES[i]: probs[i].item() for i in range(len(CLASSES))},
    }
