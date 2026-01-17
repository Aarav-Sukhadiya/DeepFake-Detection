# src/image_infer.py

import torch
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image

# Load pretrained model
model = resnet18(weights="IMAGENET1K_V1")
model.eval()

# ImageNet normalization
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def infer_image(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    # Heuristic: treat "person-related" confidence as fake proxy
    fake_confidence = float(1 - probs.max().item())

    return {
        "input_type": "image",
        "is_fake": fake_confidence > 0.5,
        "confidence": round(fake_confidence, 2)
    }
