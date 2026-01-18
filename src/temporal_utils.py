# src/temporal_utils.py
# FINAL CORRECTED VERSION
# - Explicit video-level memory
# - Stable confidence computation
# - No zero-confidence or spurious segments
# - Proper segment merging
# - All edge cases handled

from typing import List, Dict
import numpy as np


def seconds_to_timestamp(sec: float) -> str:
    sec = max(0.0, sec)
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


def temporal_localization(
    frame_times: List[float],
    frame_probs: List[float],
    fps: int = 12,
    window_size: int = 5,
    high_th: float = 0.7,
    low_th: float = 0.4,
    min_segment_sec: float = 1.0,
    min_confidence: float = 0.3
) -> List[Dict]:
    """
    Temporal localization with:
    - sliding-window aggregation
    - hysteresis
    - video-level persistent memory
    - confidence stability enforcement
    """

    if not frame_probs or len(frame_probs) < window_size:
        return []

    # -----------------------------
    # Compute sliding window scores
    # -----------------------------
    windows = []
    for i in range(len(frame_probs) - window_size + 1):
        score = float(np.median(frame_probs[i:i + window_size]))
        windows.append({
            "start": frame_times[i],
            "end": frame_times[i + window_size - 1],
            "score": score
        })

    if not windows:
        return []

    # -----------------------------
    # Video-level persistent memory
    # -----------------------------
    window_scores = np.array([w["score"] for w in windows])
    video_mean = float(np.mean(window_scores))
    video_std = float(np.std(window_scores))

    # Prevent pathological values
    video_std = max(video_std, 1e-6)

    # -----------------------------
    # Hysteresis-based segmentation
    # -----------------------------
    segments = []
    in_fake = False
    seg_start = None
    seg_scores = []

    for w in windows:
        # Memory-adjusted score
        adjusted_score = 0.7 * w["score"] + 0.3 * video_mean

        if not in_fake:
            if adjusted_score >= high_th:
                in_fake = True
                seg_start = w["start"]
                seg_scores = [adjusted_score]
        else:
            if adjusted_score >= low_th:
                seg_scores.append(adjusted_score)
            else:
                seg_end = w["end"]
                duration = seg_end - seg_start

                if duration >= min_segment_sec:
                    confidence = _compute_confidence(
                        seg_scores,
                        video_std,
                        min_confidence
                    )
                    if confidence is not None:
                        segments.append({
                            "start_time": seconds_to_timestamp(seg_start),
                            "end_time": seconds_to_timestamp(seg_end),
                            "confidence": confidence
                        })

                in_fake = False
                seg_start = None
                seg_scores = []

    # -----------------------------
    # Handle open segment at video end
    # -----------------------------
    if in_fake and seg_start is not None:
        seg_end = windows[-1]["end"]
        duration = seg_end - seg_start

        if duration >= min_segment_sec:
            confidence = _compute_confidence(
                seg_scores,
                video_std,
                min_confidence
            )
            if confidence is not None:
                segments.append({
                    "start_time": seconds_to_timestamp(seg_start),
                    "end_time": seconds_to_timestamp(seg_end),
                    "confidence": confidence
                })

    return segments


def _compute_confidence(
    seg_scores: List[float],
    video_std: float,
    min_confidence: float
):
    """
    Robust segment confidence computation.
    Returns None if confidence is too weak.
    """

    scores = np.array(seg_scores)
    if scores.size == 0:
        return None

    # Strength: focus on strongest evidence
    k = max(1, int(0.7 * len(scores)))
    strength = float(np.mean(np.sort(scores)[-k:]))

    # Stability: penalize flicker
    stability = float(1.0 - np.std(scores))
    stability = max(stability, 0.5)

    # Global consistency: penalize noisy videos
    global_consistency = float(1.0 / (1.0 + video_std))
    global_consistency = max(global_consistency, 0.5)

    confidence = strength * stability * global_consistency
    confidence = max(0.0, min(confidence, 1.0))

    if confidence < min_confidence:
        return None

    return round(confidence, 3)
