# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Per-model default generation parameters for Cosmos3 pipelines.

Shared by the Cosmos3 OmniMoT text-to-video and image-to-video generation paths.
"""

from typing import Dict

from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

# ---------------------------------------------------------------------------
# Constant tables
# ---------------------------------------------------------------------------

COSMOS3_720P_PARAMS = {
    "height": 720,
    "width": 1280,
    "num_inference_steps": 35,
    "guidance_scale": 6.0,
    "max_sequence_length": 1024,
    "num_frames": 189,
    "frame_rate": 24.0,
}

COSMOS3_EXTRA_SPECS: Dict[str, ExtraParamSchema] = {
    "use_duration_template": ExtraParamSchema(
        type="bool",
        default=True,
        description="Whether to use the duration template.",
    ),
    "use_resolution_template": ExtraParamSchema(
        type="bool",
        default=True,
        description="Whether to use the resolution template.",
    ),
    "use_system_prompt": ExtraParamSchema(
        type="bool",
        default=False,
        description="Whether to use the system prompt.",
    ),
    "use_guardrails": ExtraParamSchema(
        type="bool",
        default=True,
        description="Whether to use the guardrails.",
    ),
    "enable_audio": ExtraParamSchema(
        type="bool",
        default=False,
        description="Whether to enable audio generation.",
    ),
}

# Internal Cosmos3 conditioning defaults (latent frame indices). Not exposed via
# extra_param_specs; the pipeline selects these based on the conditioning input.
COSMOS3_CONDITION_FRAME_INDEXES_I2V = [0]
COSMOS3_CONDITION_FRAME_INDEXES_V2V = [0, 1]


def cosmos3_condition_frame_indexes_for_input(*, has_video_input: bool) -> list[int]:
    """Return default latent vision conditioning indexes for I2V vs V2V."""
    return list(
        COSMOS3_CONDITION_FRAME_INDEXES_V2V
        if has_video_input
        else COSMOS3_CONDITION_FRAME_INDEXES_I2V
    )


def cosmos3_condition_frame_indexes_inverse_dynamics(
    num_frames: int, temporal_compression: int = 4
) -> list[int]:
    """All latent frames are conditioned for inverse-dynamics (video in, actions out)."""
    latent_frames = (num_frames - 1) // temporal_compression + 1
    return list(range(latent_frames))
