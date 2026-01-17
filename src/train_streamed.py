# src/train_streamed.py
# Streamed training with recursive subfolder support (CPU/GPU safe)

import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

# ---------------- CONFIG ----------------
DATA_DIR = "data/videos"     # expects data/videos/real and data/videos/fake (any depth)
FPS = 5                      # frame sampling rate
EPOCHS = 2
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/image_model.pth"
MAX_FRAMES_PER_VIDEO = 300   # safety cap (optional but recommended)
# ---------------------------------------

print(f"Training on device: {DEVICE}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- Model ----------------
model = resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def train_on_video(video_path: str, label: int):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(int(video_fps // FPS), 1)

    frame_id = 0
    used = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x = transform(img).unsqueeze(0).to(DEVICE)
            y = torch.tensor([label], dtype=torch.long).to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            used += 1
            if used >= MAX_FRAMES_PER_VIDEO:
                break

        frame_id += 1

    cap.release()


def train():
    model.train()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for label, cls in enumerate(["real", "fake"]):
            cls_dir = os.path.join(DATA_DIR, cls)
            if not os.path.isdir(cls_dir):
                continue

            for root, _, files in os.walk(cls_dir):
                for fname in files:
                    if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                        video_path = os.path.join(root, fname)
                        print(f"Training on: {video_path}")
                        train_on_video(video_path, label)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
