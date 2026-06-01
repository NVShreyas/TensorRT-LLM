"""Unit tests for Cosmos3 vision conditioning helpers."""

import torch
from PIL import Image

from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
    COSMOS3_CONDITION_FRAME_INDEXES_I2V,
    COSMOS3_CONDITION_FRAME_INDEXES_V2V,
    cosmos3_condition_frame_indexes_for_input,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.vision import (
    build_conditioned_pixel_video,
    build_conditioning_pixel_video,
    build_pixel_video_from_image,
    max_pixel_frames_for_condition_indexes,
    preprocess_conditioning_image,
)


def test_cosmos3_condition_frame_indexes_for_input():
    assert cosmos3_condition_frame_indexes_for_input(has_video_input=False) == (
        COSMOS3_CONDITION_FRAME_INDEXES_I2V
    )
    assert cosmos3_condition_frame_indexes_for_input(has_video_input=True) == (
        COSMOS3_CONDITION_FRAME_INDEXES_V2V
    )


def test_max_pixel_frames_for_condition_indexes():
    assert max_pixel_frames_for_condition_indexes([0], temporal_compression=4) == 4
    assert max_pixel_frames_for_condition_indexes([0, 1], temporal_compression=4) == 8


def test_build_pixel_video_from_image():
    image = torch.ones(1, 3, 8, 16)
    video = build_pixel_video_from_image(image, num_frames=5)
    assert video.shape == (1, 3, 5, 8, 16)
    assert torch.all(video[:, :, 0] == 1.0)
    assert torch.all(video[:, :, 4] == 1.0)


def test_build_conditioned_pixel_video_pads_with_last_frame():
    conditioning = torch.arange(12, dtype=torch.float32).reshape(3, 1, 2, 2).expand(3, 2, 2, 2)
    video = build_conditioned_pixel_video(conditioning, num_frames=4, height=2, width=2)
    assert video.shape == (1, 3, 4, 2, 2)
    assert torch.equal(video[0, :, 0], conditioning[:, 0])
    assert torch.equal(video[0, :, 1], conditioning[:, 1])
    assert torch.equal(video[0, :, 2], conditioning[:, 1])
    assert torch.equal(video[0, :, 3], conditioning[:, 1])


def test_preprocess_conditioning_image_from_pil():
    pil_image = Image.new("RGB", (20, 10), color=(255, 0, 0))
    tensor = preprocess_conditioning_image(pil_image, target_h=4, target_w=8)
    assert tensor.shape == (1, 3, 4, 8)
    assert tensor.min() >= -1.0 and tensor.max() <= 1.0


def test_build_conditioning_pixel_video_from_image():
    pil_image = Image.new("RGB", (16, 16), color=(128, 128, 128))
    video, tmp_path = build_conditioning_pixel_video(
        image=pil_image,
        video=None,
        height=8,
        width=8,
        num_frames=3,
        condition_frame_indexes=[0],
        temporal_compression=4,
    )
    assert tmp_path is None
    assert video.shape == (1, 3, 3, 8, 8)
