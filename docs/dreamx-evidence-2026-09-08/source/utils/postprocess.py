"""
Post-processing utilities for generated video frames.
Includes Lab color correction, temporal Gaussian smoothing, and chunk-boundary blending.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def postprocess_video_frames(video_tensor, reference_frame=None,
                             color_correction_strength=0.3):
    """
    Post-process video: Lab color correction, temporal Gaussian smoothing, chunk-boundary blending.

    Args:
        video_tensor: [B, T, H, W, C] in range [0, 255].
        reference_frame: [H, W, C] reference for color correction. Defaults to first frame.
        color_correction_strength: 0.0=disabled, 1.0=full correction.

    Returns:
        Processed video [B, T, H, W, C] in [0, 255].
    """
    if video_tensor.shape[1] <= 1:
        return video_tensor

    batch_size, num_frames, height, width, channels = video_tensor.shape
    processed = video_tensor.clone()

    # --- Color correction in Lab space (chunk-relative) ---
    if color_correction_strength > 0.0:
        if reference_frame is None:
            reference_frame = processed[0, 0]

        per_frame_strength = [color_correction_strength] * num_frames
        first_chunk_end = 9
        subsequent_chunk_size = 12

        def _lab_stats(frame_tensor):
            np_frame = frame_tensor.cpu().numpy().astype(np.float64) / 255.0
            lab = cv2.cvtColor(np_frame.astype(np.float32), cv2.COLOR_RGB2LAB)
            return lab.mean(axis=(0, 1)), lab.std(axis=(0, 1))

        ref_np = reference_frame.cpu().numpy().astype(np.float64) / 255.0
        ref_lab = cv2.cvtColor(ref_np.astype(np.float32), cv2.COLOR_RGB2LAB)
        global_ref_mean = ref_lab.mean(axis=(0, 1))
        global_ref_std = ref_lab.std(axis=(0, 1))

        for b in range(batch_size):
            cur_mean, cur_std = global_ref_mean.copy(), global_ref_std.copy()
            for t in range(num_frames):
                if t == first_chunk_end:
                    cur_mean, cur_std = _lab_stats(processed[b, first_chunk_end - 1])
                elif t > first_chunk_end and (t - first_chunk_end) % subsequent_chunk_size == 0:
                    cur_mean, cur_std = _lab_stats(processed[b, t - 1])

                frame_np = processed[b, t].cpu().numpy().astype(np.float64) / 255.0
                frame_lab = cv2.cvtColor(frame_np.astype(np.float32), cv2.COLOR_RGB2LAB)
                f_mean = frame_lab.mean(axis=(0, 1))
                f_std = frame_lab.std(axis=(0, 1))

                corrected = frame_lab.copy()
                for ch in range(3):
                    if f_std[ch] > 1e-6:
                        corrected[:, :, ch] = (corrected[:, :, ch] - f_mean[ch]) * (cur_std[ch] / f_std[ch]) + cur_mean[ch]
                    else:
                        corrected[:, :, ch] = cur_mean[ch]

                corrected_rgb = np.clip(cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB), 0.0, 1.0)
                strength = per_frame_strength[t]
                blended = (1.0 - strength) * frame_np + strength * corrected_rgb.astype(np.float64)
                processed[b, t] = torch.from_numpy((blended * 255.0).clip(0, 255)).to(video_tensor.dtype)

    return processed.clamp(0, 255)
