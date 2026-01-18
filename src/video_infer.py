# src/video_infer.py
# FINAL PATCHED VERSION
# - Correct video-level decision logic
# - Uses segment evidence first
# - Handles real videos cleanly (no zero-confidence segments)
# - FPS-safe, edge-case safe

import cv2
import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from src.temporal_utils import temporal_localization

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- CONFIG ----------------
FPS = 12
TEMPERATURE = 2.0
VIDEO_FAKE_THRESHOLD = 0.75
# --------------------------------------

# ---------------- MODEL ----------------
model = resnet18(weights="IMAGENET1K_V1")
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/image_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- PREPROCESS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def infer_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "input_type": "video",
            "video_is_fake": False,
            "overall_confidence": 0.0,
            "manipulated_segments": []
        }

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = FPS

    interval = max(int(video_fps // FPS), 1)

    frame_times = []
    frame_probs = []

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            time_sec = frame_id / video_fps

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x = transform(img).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                logits = model(x) / TEMPERATURE
                probs = torch.softmax(logits, dim=1)

            fake_prob = float(probs[0][1].item())
            frame_times.append(time_sec)
            frame_probs.append(fake_prob)

        frame_id += 1

    cap.release()

    # ---------------- TEMPORAL LOCALIZATION ----------------
    segments = temporal_localization(
        frame_times=frame_times,
        frame_probs=frame_probs,
        fps=FPS
    )

    # ---------------- VIDEO-LEVEL DECISION ----------------
    if segments:
        overall_confidence = max(s["confidence"] for s in segments)
    else:
        overall_confidence = float(sum(frame_probs) / len(frame_probs)) if frame_probs else 0.0

    overall_confidence = round(overall_confidence, 3)
    video_is_fake = overall_confidence >= VIDEO_FAKE_THRESHOLD

    return {
        "input_type": "video",
        "video_is_fake": video_is_fake,
        "overall_confidence": overall_confidence,
        "manipulated_segments": segments
    }
