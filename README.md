# Deepfake Detection in Images and Videos with Timestamp Localization

## Overview

This project implements an end-to-end deepfake detection system that works on both images and videos.  
In addition to classifying content as real or fake, the system localizes manipulated regions in videos and reports precise timestamps with confidence scores.

The emphasis of this project is on temporal reasoning, explainability, and practical usability rather than black-box classification.

---

## Key Features

- Image-level deepfake detection with confidence score
- Video-level deepfake detection with partial manipulation support
- Timestamp localization of manipulated video segments
- Stable temporal predictions (no flickering)
- Single calibrated confidence value per detected segment
- Runs on CPU and GPU (CUDA) automatically
- Streamlit-based frontend for easy demonstration
- Fully local execution (no external APIs)

---

## System Pipeline

1. Input image or video
2. Frame extraction (videos only)
3. Face detection and lightweight tracking
4. CNN-based feature extraction (ResNet-18)
5. Frame-level fake probability estimation
6. Sliding window temporal aggregation
7. Hysteresis-based segment merging
8. Final prediction with timestamps and confidence

---

## Data Assumptions

- Videos are stored inside:
  - `data/videos/real`
  - `data/videos/fake`
- Subfolders inside `real` and `fake` directories are supported
- Videos may contain both real and fake segments
- At least one visible face is assumed in most frames
- Performance may degrade under extreme compression or very low resolution

---

## Model Design

### Feature Extraction

- CNN backbone: ResNet-18
- Operates on cropped face regions
- Learns spatial artifacts introduced by deepfake generation

### Temporal Reasoning

Temporal reasoning is implemented algorithmically:

- Sliding window aggregation over frame probabilities
- Median window scoring for robustness
- Hysteresis thresholds to avoid flickering predictions
- Minimum segment duration filtering
- Segment-level confidence re-scoring

This approach prioritizes interpretability and computational efficiency.

### Confidence Estimation

- Frame-level probabilities are calibrated using temperature scaling
- Segment confidence is computed from stable window-level evidence
- A single interpretable confidence value is reported per segment

---

## Training

Training is streamed and memory-efficient:

- Videos are processed sequentially
- Frames are processed in mini-batches
- Frames are never stored on disk
- GPU acceleration is used automatically if available

### Train the Model

```bash
python src/train_streamed.py
The trained model is saved to:

models/image_model.pth
Inference
Image Inference
Input: single image

Output: fake probability (confidence score)

Video Inference
Input: video file

Output:

video-level fake prediction

overall confidence score

timestamped manipulated segments

Example Output
{
  "input_type": "video",
  "video_is_fake": true,
  "overall_confidence": 0.82,
  "manipulated_segments": [
    {
      "start_time": "00:01",
      "end_time": "00:04",
      "confidence": 0.81
    }
  ]
}
Installation
Single Command (Windows / Linux, CPU or GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 opencv-python numpy pillow streamlit
GPU acceleration is used automatically if CUDA is available.

Evaluation Metrics (Conceptual)
Precision and recall

Temporal Intersection over Union (tIoU)

Segment-level precision and recall

Temporal stability

Detection delay

Robustness under compression and lighting variations

Limitations
Uses algorithmic temporal reasoning instead of learned temporal models

Assumes a dominant face per video

Not intended for production deployment

Extreme video degradation may reduce accuracy

Ethical Considerations
Outputs are probabilistic and not absolute judgments

Intended for human-in-the-loop analysis

Not suitable for automated legal or disciplinary decisions

Conclusion
This project demonstrates a practical and explainable approach to deepfake detection with timestamp localization by combining CNN-based spatial analysis with robust temporal reasoning and calibrated confidence estimation.

```
