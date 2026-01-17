Deepfake Detection in Images and Videos with Timestamp Localization
Overview

This project implements an end-to-end deepfake detection system that works on both images and videos.
Unlike simple binary classifiers, the system localizes manipulated segments in videos with timestamps and confidence scores, making it suitable for forensic analysis and explainable AI use cases.

The system is designed to:

run on a standard laptop

optionally use GPU (CUDA) if available

provide interpretable JSON outputs

handle partially manipulated videos

Key Features

Image deepfake detection with confidence score

Video deepfake detection with timestamp localization

Handles videos containing both real and fake segments

Temporal reasoning for stable, non-flickering predictions

Calibrated confidence scores

Lightweight Streamlit frontend for demo

CPU-safe and GPU-accelerated training and inference

Project Pipeline
Input Image / Video
   ↓
Frame Extraction (video only)
   ↓
Face Detection & Lightweight Tracking
   ↓
CNN-based Feature Extraction
   ↓
Frame-level Fake Probability
   ↓
Temporal Window Aggregation
   ↓
Segment Merging & Confidence Estimation
   ↓
Final Prediction + Timestamps (JSON)

Folder Structure
deepfake_project/
├── data/
│   └── videos/
│       ├── real/
│       └── fake/
├── models/
│   └── image_model.pth
├── src/
│   ├── image_infer.py
│   ├── video_infer.py
│   ├── preprocess.py
│   ├── temporal_utils.py
│   └── train_streamed.py
├── frontend.py
├── requirements.txt
└── README.md

Data Assumptions

Videos are placed under:

data/videos/real/
data/videos/fake/


Subfolders inside real and fake are supported

Labels are video-level, but detection is segment-level

Faces are assumed to be present in most frames

Extreme compression or very low resolution may reduce performance

Model Design
Feature Extraction

CNN backbone: ResNet-18

Trained on video frames

Operates on face crops for better signal-to-noise ratio

Temporal Reasoning

Sliding window aggregation over frame probabilities

Median window scoring for robustness

Hysteresis thresholds for temporal stability

Segment merging with minimum duration constraints

Confidence Estimation

CNN outputs are calibrated using temperature scaling

Segment confidence is computed from stable window-level evidence

Outputs a single, interpretable confidence number

Training

Training is streamed and memory-efficient:

Videos are processed one at a time

Frames are processed one batch at a time

Frames are not stored on disk

Supports GPU acceleration with:

mini-batching

mixed precision (AMP)

cuDNN benchmarking

Train the Model
python src/train_streamed.py


The trained model is saved to:

models/image_model.pth

Inference
Image Inference

Single image

Outputs fake/real confidence

Video Inference

Processes video frames sequentially

Aggregates predictions temporally

Outputs:

overall video confidence

timestamped manipulated segments

Example output:

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

Frontend Demo

A simple Streamlit UI is provided.

Run the Frontend
streamlit run frontend.py


Features:

Upload image or video

Run real inference

View JSON output directly

Installation
Windows (CUDA / NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 opencv-python numpy pillow streamlit

CPU-only (Any OS)
pip install torch torchvision opencv-python numpy pillow streamlit

Evaluation Metrics (Conceptual)

Precision / Recall

Temporal Intersection over Union (tIoU)

Segment-level precision and recall

Temporal stability (non-flickering segments)

Detection delay

Robustness to compression and lighting variations

Limitations

Uses algorithmic temporal reasoning, not learned temporal models

Assumes a dominant face per video

Performance degrades under extreme compression or occlusion

Not intended for production use

Ethical Considerations

Outputs are probabilistic, not absolute truth

Intended for human-in-the-loop analysis

Not suitable for automated legal or disciplinary decisions

Future Work

Learned temporal models (1D CNN / LSTM)

Multi-face tracking

Domain-specific fine-tuning

Improved robustness under extreme degradation

Conclusion

This project demonstrates a practical, explainable deepfake detection system that goes beyond binary classification by providing timestamp-level localization and calibrated confidence scores, while remaining computationally feasible and interpretable.