# src/temporal_utils.py
# Improved temporal reasoning with windowing + hysteresis

from typing import List, Dict
import numpy as np


def seconds_to_timestamp(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


def temporal_localization(
    frame_times: List[float],
    frame_probs: List[float],
    fps: int = 5,
    window_size: int = 5,          # frames per window (~1s)
    high_th: float = 0.7,          # enter fake
    low_th: float = 0.4,           # exit fake
    min_segment_sec: float = 1.0   # discard very short segments
) -> List[Dict]:
    """
    Improved temporal reasoning:
    - sliding window aggregation
    - hysteresis thresholding
    - stable segment confidence
    """

    if len(frame_probs) < window_size:
        return []

    # ---------- 1. Window-level aggregation ----------
    windows = []
    for i in range(0, len(frame_probs) - window_size + 1):
        w_probs = frame_probs[i:i + window_size]

        # robust aggregation (median is stable against spikes)
        score = float(np.median(w_probs))

        windows.append({
            "start": frame_times[i],
            "end": frame_times[i + window_size - 1],
            "score": score
        })

    # ---------- 2. Hysteresis-based segmentation ----------
    segments = []
    in_fake = False
    curr_start = None
    curr_scores = []

    for w in windows:
        if not in_fake:
            if w["score"] >= high_th:
                in_fake = True
                curr_start = w["start"]
                curr_scores = [w["score"]]
        else:
            if w["score"] >= low_th:
                curr_scores.append(w["score"])
            else:
                # close segment
                end_time = w["end"]
                duration = end_time - curr_start

                if duration >= min_segment_sec:
                    segments.append({
                        "start_time": seconds_to_timestamp(curr_start),
                        "end_time": seconds_to_timestamp(end_time),
                        "confidence": round(
                            float(np.mean(np.sort(curr_scores)[-max(1, int(0.7 * len(curr_scores))):])),
                            2
                        )
                    })

                in_fake = False
                curr_start = None
                curr_scores = []

    # ---------- 3. Handle open segment ----------
    if in_fake and curr_start is not None:
        end_time = windows[-1]["end"]
        duration = end_time - curr_start

        if duration >= min_segment_sec:
            segments.append({
                "start_time": seconds_to_timestamp(curr_start),
                "end_time": seconds_to_timestamp(end_time),
                "confidence": round(
                    float(np.mean(np.sort(curr_scores)[-max(1, int(0.7 * len(curr_scores))):])),
                    2
                )
            })

    return segments
