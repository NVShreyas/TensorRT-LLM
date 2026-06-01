# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cosmos3 vision conditioning helpers (pixel-space, pre-VAE)."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF

PathLike = Union[str, Path]
ImageInput = Union[PIL.Image.Image, torch.Tensor, str]
VideoInput = Union[str, bytes]


def resolve_video_path(video: VideoInput) -> tuple[str, Optional[str]]:
    """Return a filesystem path for ``video`` and an optional temp path to delete."""
    if isinstance(video, str):
        return video, None

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video)
    tmp.close()
    return tmp.name, tmp.name


def _resize_and_center_crop_frames(
    frames: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Aspect-ratio-preserving resize + center crop.

    Args:
        frames: [..., H, W] or [T, C, H, W]

    Returns:
        Tensor with spatial dims ``(target_h, target_w)``.
    """
    orig_h, orig_w = frames.shape[-2], frames.shape[-1]
    scaling_ratio = max(target_w / orig_w, target_h / orig_h)
    resize_h = int(math.ceil(scaling_ratio * orig_h))
    resize_w = int(math.ceil(scaling_ratio * orig_w))
    frames = TF.resize(frames, [resize_h, resize_w])
    return TF.center_crop(frames, [target_h, target_w])


def load_conditioning_image(
    image_path: PathLike,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Load an image for I2V conditioning.

    Returns:
        ``[3, 1, H, W]`` float tensor in ``[-1, 1]``.
    """
    with Path(image_path).open("rb") as f:
        img = PIL.Image.open(f).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0)
    img_tensor = _resize_and_center_crop_frames(img_tensor, target_h, target_w)
    img_tensor = img_tensor.squeeze(0) / 127.5 - 1.0
    return img_tensor.unsqueeze(1).contiguous()


def preprocess_conditioning_image(
    image: ImageInput,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Preprocess a conditioning image to ``[1, 3, H, W]`` in ``[-1, 1]``.

    String paths use the same resize/normalize path as ``load_conditioning_image``.
    Preprocessed ``torch.Tensor`` inputs are returned unchanged.
    """
    if isinstance(image, torch.Tensor):
        return image

    if isinstance(image, str):
        frames = load_conditioning_image(image, target_h, target_w)
        return frames.squeeze(1).unsqueeze(0)

    image = image.convert("RGB")
    img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().unsqueeze(0)
    img_tensor = _resize_and_center_crop_frames(img_tensor, target_h, target_w)
    img_tensor = img_tensor.squeeze(0) / 127.5 - 1.0
    return img_tensor.unsqueeze(0).contiguous()


def load_conditioning_video(
    video_path: PathLike,
    target_h: int,
    target_w: int,
    max_frames: int,
) -> torch.Tensor:
    """Load a conditioning clip for V2V.

    Returns:
        ``[3, T, H, W]`` float tensor in ``[-1, 1]``.
    """
    from torchvision.io import read_video

    frames, _, _ = read_video(str(video_path), pts_unit="sec")
    frames = frames[:max_frames]
    if frames.numel() == 0:
        raise ValueError(f"Video '{video_path}' contains no readable frames.")

    frames_tchw = frames.permute(0, 3, 1, 2).float()
    frames_resized = _resize_and_center_crop_frames(frames_tchw, target_h, target_w)
    frames_normalized = frames_resized / 127.5 - 1.0
    return frames_normalized.permute(1, 0, 2, 3).contiguous()


def build_conditioned_pixel_video(
    conditioning_frames: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Build a full-length pixel video batch from conditioning frames.

    Matches ``cosmos3-internal`` ``build_conditioned_video_batch``: copy the
    available conditioning frames, then pad the remainder by repeating the last
    conditioning frame.

    Args:
        conditioning_frames: ``[3, T_cond, H, W]`` or ``[1, 3, T_cond, H, W]``.
        num_frames: Target pixel frame count for generation.
        height: Target height (must match conditioning spatial size).
        width: Target width (must match conditioning spatial size).

    Returns:
        ``[1, 3, num_frames, H, W]`` in ``[-1, 1]``.
    """
    if conditioning_frames.ndim == 5:
        if conditioning_frames.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported for conditioning video.")
        conditioning_frames = conditioning_frames[0]

    if conditioning_frames.ndim != 4:
        raise ValueError(
            "conditioning_frames must be [3, T, H, W] or [1, 3, T, H, W], "
            f"got shape {tuple(conditioning_frames.shape)}"
        )

    _, t_cond, h, w = conditioning_frames.shape
    if h != height or w != width:
        raise ValueError(
            f"Conditioning spatial size {(h, w)} does not match target {(height, width)}"
        )

    video = torch.zeros(1, 3, num_frames, height, width, dtype=conditioning_frames.dtype)
    t_fill = min(t_cond, num_frames)
    video[0, :, :t_fill] = conditioning_frames[:, :t_fill]
    if t_fill < num_frames:
        video[0, :, t_fill:] = video[0, :, t_fill - 1 : t_fill].expand(
            -1, num_frames - t_fill, -1, -1
        )
    return video


def build_pixel_video_from_image(
    image_tensor: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    """Repeat a single preprocessed image across time for WAN VAE encoding.

    Args:
        image_tensor: ``[1, 3, H, W]`` in ``[-1, 1]``.

    Returns:
        ``[1, 3, num_frames, H, W]``.
    """
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError(
            f"image_tensor must be [1, 3, H, W], got shape {tuple(image_tensor.shape)}"
        )
    return image_tensor.unsqueeze(2).expand(-1, -1, num_frames, -1, -1).contiguous()


def max_pixel_frames_for_condition_indexes(
    condition_frame_indexes: list[int],
    temporal_compression: int,
) -> int:
    """Pixel frames to load for the given latent conditioning indexes."""
    if not condition_frame_indexes:
        return 1
    return (max(condition_frame_indexes) + 1) * temporal_compression


def build_conditioning_pixel_video(
    *,
    image: Optional[ImageInput],
    video: Optional[VideoInput],
    height: int,
    width: int,
    num_frames: int,
    condition_frame_indexes: list[int],
    temporal_compression: int,
) -> tuple[torch.Tensor, Optional[str]]:
    """Build pixel-space conditioning video for I2V or V2V.

    Returns:
        ``[1, 3, num_frames, H, W]`` tensor and an optional temp video path to delete.
    """
    if video is not None:
        video_path, tmp_path = resolve_video_path(video)
        max_frames = max_pixel_frames_for_condition_indexes(
            condition_frame_indexes,
            temporal_compression,
        )
        conditioning_frames = load_conditioning_video(
            video_path,
            target_h=height,
            target_w=width,
            max_frames=max_frames,
        )
        pixel_video = build_conditioned_pixel_video(
            conditioning_frames,
            num_frames=num_frames,
            height=height,
            width=width,
        )
        return pixel_video, tmp_path

    image_tensor = preprocess_conditioning_image(image, target_h=height, target_w=width)
    return build_pixel_video_from_image(image_tensor, num_frames=num_frames), None
