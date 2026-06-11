#!/usr/bin/env python3
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
r"""Cosmos3 Text(+Image)-to-Video and action generation.

Cosmos3 OmniMoT supports text-only (T2V), image-conditioned (I2V/TI2V), optional
audio, and action generation (policy / forward dynamics / inverse dynamics) from
the same checkpoint when ``action_gen`` is enabled.

Checkpoints (pass the Hub ID or local path via ``--model``):

- `nvidia/Cosmos3-Nano <https://huggingface.co/nvidia/Cosmos3-Nano>`_
- `nvidia/Cosmos3-Super <https://huggingface.co/nvidia/Cosmos3-Super>`_

Guardrails are enabled by default (required by the
`NVIDIA Open Model License Agreement
<https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license>`_).
Install and authenticate as follows::

    pip install cosmos_guardrail==0.3.0 && pip uninstall opencv-python

Accept the terms for the guardrail checkpoint at
https://huggingface.co/nvidia/Cosmos-1.0-Guardrail and set a valid ``HF_TOKEN``
(the checkpoint is downloaded automatically on first run).

To run without guardrails (you are responsible for safe deployment)::

    export TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1

Deployment configs (``examples/visual_gen/configs/``):

- ``cosmos3-nano-1gpu.yaml`` — 1 GPU, FP8 dynamic quant
- ``cosmos3-super-4gpu.yaml`` — 4 GPU, CFG + Ulysses + parallel VAE

Usage:
    # T2V
    python cosmos3_ti2v.py --model nvidia/Cosmos3-Nano \
        --prompt "A serene mountain lake at sunrise." \
        --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

    # T2V + audio
    python cosmos3_ti2v.py --model nvidia/Cosmos3-Nano \
        --prompt "A low-angle tracking shot follows a man riding a vintage black motorcycle " \\
        "across a lush green grassy yard. Sunlight filters through overhead trees, casting " \\
        "dappled shadows across the vibrating chrome exhaust and the rider's leather jacket. " \\
        "He kicks up small blades of grass as he maneuvers the bike. He gradually decelerates, " \\
        "the front fork compressing slightly as he brakes to a smooth halt beside another " \\
        "individual standing in the shade. The camera settles into a medium two-shot, capturing " \\
        "the rider lifting his visor to speak, his face framed by a matte helmet. The video is " \\
        "8 seconds long and is of 24 FPS. This video is of 1280x720 resolution. Audio description: " \\
        "The rhythmic, mechanical chugging of a four-stroke motorcycle engine dominates the " \\
        "foreground, characterized by a throaty, guttural timbre. Periodic high-pitched revs " \\
        "punctuate the steady idle as the throttle is twisted. The sound of tires crunching " \\
        "softly over dry grass and twigs provides a textured background layer. As the vehicle " \\
        "slows, the engine note drops to a low-frequency rumble before clicking into neutral. " \\
        "A muffled, mid-range male voice begins speaking, accompanied by the metallic clink of " \\
        "a helmet visor snapping upward and the faint chirping of distant birds in an open-air " \\
        "environment." \
        --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml \
        --enable_audio

    # Action — policy (first frame + instruction -> predicted action + rollout video)
    python cosmos3_ti2v.py --model nvidia/Cosmos3-Nano \
        --prompt "Pick up the pear and place it in the bag." \
        --image_path first_frame.png \
        --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml \
        --action_mode policy \
        --domain_name bridge_orig_lerobot \
        --raw_action_dim 10 \
        --output_path policy_rollout.mp4 \
        --action_output_path policy_action.json

    # Action — forward dynamics (first frame + action trajectory -> rollout video)
    python cosmos3_ti2v.py --model nvidia/Cosmos3-Nano \
        --prompt "Robot manipulation rollout." \
        --image_path first_frame.png \
        --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml \
        --action_mode forward_dynamics \
        --domain_name av \
        --action_json action_trajectory.json \
        --output_path forward_dynamics.mp4

    # Action — inverse dynamics (video -> predicted action)
    python cosmos3_ti2v.py --model nvidia/Cosmos3-Nano \
        --prompt "Recover the robot action from this clip." \
        --video_path /path/to/clip.mp4 \
        --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml \
        --action_mode inverse_dynamics \
        --domain_name bridge_orig_lerobot \
        --raw_action_dim 10 \
        --output_path inverse_video.mp4 \
        --action_output_path inverse_action.json

    # Text-to-image
    python cosmos3_ti2v.py --model nvidia/Cosmos3-Nano \
        --prompt "A cute puppy playing with a ball in a park" \
        --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml \
        --output_type image \
        --output_path output.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

from tensorrt_llm import VisualGen, VisualGenArgs
from tensorrt_llm._torch.visual_gen.models.cosmos3.action import VIDEO_RES_SIZE_INFO
from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
    COSMOS3_ACTION_PARAMS,
    get_domain_preset,
    resolve_domain_action_config,
)

ACTION_MODES = frozenset({"policy", "forward_dynamics", "inverse_dynamics"})


def _validate_action_args(args: argparse.Namespace) -> None:
    if args.action_mode is None:
        return

    mode = args.action_mode.strip().lower()
    if mode not in ACTION_MODES:
        raise SystemExit(
            f"Invalid --action_mode {args.action_mode!r}; expected one of {sorted(ACTION_MODES)}."
        )
    if args.enable_audio:
        raise SystemExit("Cosmos3 does not support joint action and audio generation.")
    if args.output_type != "video":
        raise SystemExit("Action generation requires --output_type video.")

    if mode == "forward_dynamics":
        if args.action_json is None:
            raise SystemExit(f"{mode} requires --action_json.")
        if args.image_path is None and args.video_path is None:
            raise SystemExit(f"{mode} requires --image_path or --video_path for the first frame.")
    elif mode == "policy":
        if args.image_path is None and args.video_path is None:
            raise SystemExit(f"{mode} requires --image_path or --video_path for the first frame.")
        preset = get_domain_preset(args.domain_name, args.domain_id)
        effective_raw_dim = args.raw_action_dim or (preset or {}).get("raw_action_dim")
        if effective_raw_dim is None:
            raise SystemExit(
                f"{mode} requires --raw_action_dim or a known --domain_name with a preset."
            )
    elif mode == "inverse_dynamics":
        if args.video_path is None:
            raise SystemExit(
                f"{mode} requires --video_path (frame directory, .mp4/.avi, or image)."
            )
        preset = get_domain_preset(args.domain_name, args.domain_id)
        effective_raw_dim = args.raw_action_dim or (preset or {}).get("raw_action_dim")
        if effective_raw_dim is None:
            raise SystemExit(
                f"{mode} requires --raw_action_dim or a known --domain_name with a preset."
            )


def _apply_action_generation_params(params, args: argparse.Namespace) -> None:
    """Set action defaults on the request; domain presets override generic 480p."""
    cfg = resolve_domain_action_config(
        domain_name=args.domain_name,
        domain_id=args.domain_id,
        raw_action_dim=args.raw_action_dim,
        action_chunk_size=args.action_chunk_size,
        action_resolution=args.action_resolution,
    )
    bucket = str(cfg["action_resolution"])
    width, height = VIDEO_RES_SIZE_INFO[bucket]["16,9"]
    params.width = width
    params.height = height
    params.num_frames = cfg["num_frames"]
    params.num_inference_steps = COSMOS3_ACTION_PARAMS["num_inference_steps"]
    params.guidance_scale = COSMOS3_ACTION_PARAMS["guidance_scale"]
    params.frame_rate = cfg["frame_rate"]
    params.extra_params["action_chunk_size"] = cfg["action_chunk_size"]
    if cfg["raw_action_dim"] is not None:
        params.extra_params["raw_action_dim"] = cfg["raw_action_dim"]
    params.extra_params["action_resolution"] = cfg["action_resolution"]


def _default_action_output_path(video_path: str) -> str:
    stem = Path(video_path)
    if stem.suffix:
        return str(stem.with_name(f"{stem.stem}_action.json"))
    return f"{video_path}_action.json"


def _save_action_output(output, path: str) -> None:
    if output.action is None:
        return

    action = output.action
    if action.ndim == 3 and action.shape[0] == 1:
        action_data = action[0].tolist()
        shape = list(action.shape[1:])
    else:
        action_data = action.tolist()
        shape = list(action.shape)

    payload = {
        "action_mode": output.action_mode,
        "domain_id": output.domain_id,
        "raw_action_dim": output.raw_action_dim,
        "shape": shape,
        "dtype": str(action.dtype).replace("torch.", ""),
        "data": action_data,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_negative_prompt(path: Optional[str]) -> Any:
    if path is None:
        return None
    if os.path.isfile(path) and path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return path


def main():
    parser = argparse.ArgumentParser(description="Cosmos3 Text(+Image)-to-Video example")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos3-Nano",
        help="Model path or HuggingFace Hub ID (nvidia/Cosmos3-Nano, nvidia/Cosmos3-Super)",
    )
    parser.add_argument(
        "--visual_gen_args",
        "--extra_visual_gen_options",
        dest="visual_gen_args",
        type=str,
        default=None,
        help="Path to YAML config (same as trtllm-serve --visual_gen_args)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="cosmos3_negative_prompt.json",
        help="Text prompt or path to JSON file for negative prompt",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Conditioning image for I2V/TI2V or action policy/forward_dynamics",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="cosmos3_ti2v_output.mp4",
        help="Path to save the output video or image",
    )
    parser.add_argument(
        "--enable_duration_template", action="store_true", help="Enable duration template in prompt"
    )
    parser.add_argument(
        "--enable_resolution_template",
        action="store_true",
        help="Enable resolution template in prompt",
    )
    parser.add_argument(
        "--use_system_prompt", action="store_true", help="Use system prompt in prompt"
    )
    parser.add_argument("--enable_audio", action="store_true", help="Enable audio generation")
    parser.add_argument(
        "--action_mode",
        type=str,
        default=None,
        choices=sorted(ACTION_MODES),
        help="Action mode: policy, forward_dynamics, or inverse_dynamics",
    )
    parser.add_argument(
        "--domain_name",
        type=str,
        default=None,
        help="Embodiment domain name (e.g. bridge_orig_lerobot, av, droid_lerobot)",
    )
    parser.add_argument(
        "--domain_id",
        type=int,
        default=None,
        help="Embodiment domain id (alternative to --domain_name)",
    )
    parser.add_argument(
        "--raw_action_dim",
        type=int,
        default=None,
        help="Raw action DOF for policy/inverse_dynamics",
    )
    parser.add_argument(
        "--action_chunk_size",
        type=int,
        default=None,
        help=f"Action tokens to generate (default {COSMOS3_ACTION_PARAMS['action_chunk_size']})",
    )
    parser.add_argument(
        "--action_json",
        type=str,
        default=None,
        help="JSON file with action trajectory [T, D] for forward_dynamics",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Frame directory, .mp4/.avi video, or image path for inverse_dynamics",
    )
    parser.add_argument(
        "--action_resolution",
        type=int,
        default=None,
        choices=[256, 480, 704, 720],
        help=("Resolution bucket for action image sizing. Defaults to the domain preset or 480."),
    )
    parser.add_argument(
        "--action_output_path",
        type=str,
        default=None,
        help="Path to save predicted action JSON (default: <output_stem>_action.json)",
    )
    parser.add_argument(
        "--output_type", type=str, default="video", choices=["video", "image"], help="Output type"
    )

    # Guardrails
    parser.add_argument(
        "--disable_guardrails", action="store_true", help="NOT RECOMMENDED: Disable guardrails"
    )
    args = parser.parse_args()
    _validate_action_args(args)

    # Engine config from shared YAML (optional); model-specific defaults apply otherwise.
    extra_args = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None
    visual_gen = VisualGen(model=args.model, args=extra_args)

    params = visual_gen.default_params
    if args.image_path is not None:
        params.image = args.image_path

    if args.action_mode is not None:
        _apply_action_generation_params(params, args)

    negative_prompt = _load_negative_prompt(args.negative_prompt)

    params.extra_params["use_duration_template"] = args.enable_duration_template
    params.extra_params["use_resolution_template"] = args.enable_resolution_template
    params.extra_params["use_system_prompt"] = args.use_system_prompt
    params.extra_params["enable_audio"] = args.enable_audio
    params.extra_params["use_guardrails"] = not args.disable_guardrails
    params.extra_params["output_type"] = args.output_type

    if args.action_mode is not None:
        params.extra_params["action_mode"] = args.action_mode
    if args.domain_name is not None:
        params.extra_params["domain_name"] = args.domain_name
    if args.domain_id is not None:
        params.extra_params["domain_id"] = args.domain_id
    if args.raw_action_dim is not None:
        params.extra_params["raw_action_dim"] = args.raw_action_dim
    if args.action_chunk_size is not None:
        params.extra_params["action_chunk_size"] = args.action_chunk_size
    if args.action_json is not None:
        with open(args.action_json, encoding="utf-8") as f:
            params.extra_params["action"] = json.load(f)
    if args.video_path is not None:
        params.extra_params["video"] = args.video_path

    if negative_prompt is None:
        params.negative_prompt = None
    elif isinstance(negative_prompt, str):
        params.negative_prompt = negative_prompt
    else:
        params.negative_prompt = json.dumps(negative_prompt)

    output = visual_gen.generate(
        inputs=args.prompt,
        params=params,
    )

    if output.error is not None:
        raise SystemExit(f"Generation failed: {output.error}")

    output.save(args.output_path)
    print(f"Saved: {args.output_path}")

    if args.action_mode is not None:
        action_path = args.action_output_path or _default_action_output_path(args.output_path)
        _save_action_output(output, action_path)
        if output.action is not None:
            print(f"Saved action: {action_path}")
            print(
                f"Action shape: {tuple(output.action.shape)}, "
                f"raw_action_dim={output.raw_action_dim}, domain_id={output.domain_id}"
            )
        else:
            print("Warning: action_mode was set but the output carried no action tensor.")

    print(output.metrics)


if __name__ == "__main__":
    main()
