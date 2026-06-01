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

import math
import os
import time
from typing import List, Optional, Union

import PIL.Image
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import Qwen2Tokenizer

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger

from .defaults import (
    COSMOS3_720P_PARAMS,
    COSMOS3_EXTRA_SPECS,
    cosmos3_condition_frame_indexes_for_input,
)
from .guardrails import check_video_safety, download_guardrail_checkpoint
from .sound_tokenizer import LatentAutoEncoderV2
from .transformer_cosmos3 import Cosmos3VFMTransformer
from .vision import build_conditioning_pixel_video

COSMOS3_DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, "
    "low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, "
    "unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of "
    "poor quality."
)
COSMOS3_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant who will generate videos from a given prompt."
)
COSMOS3_DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
COSMOS3_DEFAULT_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."
TRTLLM_DISABLE_COSMOS3_GUARDRAILS = os.environ.get("TRTLLM_DISABLE_COSMOS3_GUARDRAILS", "0") == "1"


# TODO: add hf_ids
@register_pipeline("Cosmos3OmniMoTPipeline")
class Cosmos3OmniMoTPipeline(BasePipeline):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.audio_gen = False
        self.action_gen = False
        if getattr(
            model_config.pretrained_config,
            "audio_gen",
            getattr(model_config.pretrained_config, "sound_gen", False),
        ):
            logger.info("Initializing Cosmos3OmniMoTPipeline with audio generation.")
            self.audio_gen = True

        if getattr(model_config.pretrained_config, "action_gen", False):
            logger.info("Initializing Cosmos3OmniMoTPipeline with action generation.")
            self.action_gen = True

    def _init_transformer(self) -> None:
        logger.info("Initializing Cosmos3VFMTransformer")
        self.transformer = Cosmos3VFMTransformer(self.model_config)

    def load_weights(self, weights: dict) -> None:
        if self.transformer is not None and hasattr(self.transformer, "load_weights"):
            transformer_weights = weights.get("transformer", weights)
            self.transformer.load_weights(transformer_weights)
            self.transformer.eval()

    def load_standard_components(
        self, checkpoint_dir: str, device: torch.device, skip_components: Optional[list] = []
    ) -> None:
        skip_components = skip_components or []

        if self.audio_gen and PipelineComponent.SOUND_TOKENIZER not in skip_components:
            logger.info("Loading audio tokenizer...")
            self.audio_tokenizer = (
                LatentAutoEncoderV2.from_pretrained(
                    checkpoint_dir,
                    subfolder=PipelineComponent.SOUND_TOKENIZER,
                )
                .to(device)
                .to(self.dtype)
                .eval()
            )

        if PipelineComponent.TOKENIZER not in skip_components:
            logger.info("Loading tokenizer...")
            self.tokenizer = Qwen2Tokenizer.from_pretrained(
                checkpoint_dir,
                subfolder="text_tokenizer",
            )

        # Cosmos3 canonical defaults — overwritten if VAE is loaded
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 16

        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading VAE...")
            self.vae = AutoencoderKLWan.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.VAE,
                torch_dtype=torch.bfloat16,  # load VAE in BF16 for memory saving
            ).to(device)

            self.vae_scale_factor_temporal = getattr(
                self.vae.config, "scale_factor_temporal", self.vae_scale_factor_temporal
            )
            self.vae_scale_factor_spatial = getattr(
                self.vae.config, "scale_factor_spatial", self.vae_scale_factor_spatial
            )
            self.transformer.temporal_compression_factor = self.vae_scale_factor_temporal

        if PipelineComponent.SCHEDULER not in skip_components:
            logger.info("Loading scheduler...")
            self.scheduler = UniPCMultistepScheduler.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.SCHEDULER,
            )
            if self.audio_gen:
                # Separate instance so video and audio scheduler states don't collide
                # (UniPC mutates internal correction buffers on every .step() call).
                self.audio_scheduler = UniPCMultistepScheduler.from_config(self.scheduler.config)

        if not TRTLLM_DISABLE_COSMOS3_GUARDRAILS:
            # lazy import
            try:
                from cosmos_guardrail import CosmosSafetyChecker
            except (ImportError, ModuleNotFoundError):
                raise ValueError(
                    "Cosmos Guardrail is not installed. This is in violation of the "
                    "[NVIDIA Open Model License Agreement]"
                    "(https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). "
                    "Please run the following installation commands or "
                    "explicitly disable guardrails by setting TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1 "
                    "(user is responsible for deploying the model without guardrails). "
                    "- `pip install cosmos_guardrail==0.3.0 && pip uninstall opencv-python`"
                )
            # Guardrails are only evaluated on rank 0; load them only there to avoid
            # dead model weights occupying GPU memory on every other rank.
            if self.rank == 0:
                # the download guardrail checkpoint will bypass CosmosSafetyChecker's checkpoint download.
                # Both will use HF_HOME as the cache directory.
                download_guardrail_checkpoint()
                self.safety_checker = CosmosSafetyChecker()
                self.safety_checker.to(device)

    @property
    def default_warmup_resolutions(self):
        return [(720, 1280)]

    @property
    def default_warmup_num_frames(self):
        return [189]

    @property
    def default_generation_params(self):
        return dict(COSMOS3_720P_PARAMS)

    @property
    def extra_param_specs(self):
        return dict(COSMOS3_EXTRA_SPECS)

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        with torch.no_grad():
            self.forward(
                prompt="warmup",
                negative_prompt="",
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=COSMOS3_720P_PARAMS["guidance_scale"],
                seed=42,
                max_sequence_length=COSMOS3_720P_PARAMS["max_sequence_length"],
                use_guardrails=False,
                image=None,
                enable_audio=False,
            )

    def infer(self, req):
        video = req.params.video
        if isinstance(video, list):
            video = video[0] if video else None

        return self.forward(
            prompt=req.prompt,
            negative_prompt=req.params.negative_prompt,
            image=req.params.image,
            video=video,
            height=req.params.height,
            width=req.params.width,
            num_frames=req.params.num_frames,
            num_inference_steps=req.params.num_inference_steps,
            guidance_scale=req.params.guidance_scale,
            seed=req.params.seed,
            max_sequence_length=req.params.max_sequence_length,
            frame_rate=req.params.frame_rate,
            use_duration_template=req.params.extra_params.get("use_duration_template", True),
            use_resolution_template=req.params.extra_params.get("use_resolution_template", True),
            use_system_prompt=req.params.extra_params.get("use_system_prompt", False),
            use_guardrails=req.params.extra_params.get("use_guardrails", True),
            enable_audio=req.params.extra_params.get("enable_audio", False),
        )

    def _format_prompt_with_template(
        self,
        prompt: str,
        *,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        use_duration_template: bool = True,
        use_resolution_template: bool = True,
    ) -> str:
        prompt = prompt.strip()

        if use_duration_template and num_frames > 1:
            duration = num_frames / frame_rate
            dur_text = COSMOS3_DURATION_TEMPLATE.format(duration=duration, fps=frame_rate)
            prompt = prompt.rstrip(".") + ". " + dur_text

        prompt = prompt.strip()
        if use_resolution_template:
            res_text = COSMOS3_DEFAULT_RESOLUTION_TEMPLATE.format(height=height, width=width)
            prompt = prompt.rstrip(".") + ". " + res_text

        return prompt

    @nvtx_range("_tokenize_prompt", color="blue")
    def _tokenize_prompt(
        self, text: str, max_sequence_length: int, use_system_prompt: bool = False
    ):
        """Tokenize a prompt using the Qwen2 chat template.

        Returns (input_ids, attention_mask) as [1, S] tensors on device.
        """
        conversations = (
            [{"role": "system", "content": COSMOS3_DEFAULT_SYSTEM_PROMPT}]
            if use_system_prompt
            else []
        )
        conversations.append(
            {"role": "user", "content": text},
        )
        token_ids = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        reserved_tokens = 2
        if max_sequence_length < reserved_tokens:
            raise ValueError(
                f"max_sequence_length must be at least {reserved_tokens}, got {max_sequence_length}"
            )
        token_ids = token_ids[: max_sequence_length - reserved_tokens]
        token_ids.append(self.tokenizer.eos_token_id)  # 151645
        token_ids.append(self.tokenizer.convert_tokens_to_ids("<|vision_start|>"))  # 151652
        seq_len = len(token_ids)

        # Pad to max_sequence_length
        pad_len = max_sequence_length - seq_len
        attention_mask = [1] * seq_len + [0] * pad_len
        token_ids = token_ids + [self.tokenizer.pad_token_id or 0] * pad_len

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
        return input_ids, attention_mask

    # =========================================================================
    # Latent preparation
    # =========================================================================

    @nvtx_range("_prepare_latents", color="blue")
    def _prepare_latents(self, height, width, num_frames, generator):
        num_channels_latents = self.transformer.latent_channel_size
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            1,
            num_channels_latents,
            num_latent_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        return randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)

    # -- I2V / V2V latent preparation -----------------------------------------

    def _encode_pixel_video(self, pixel_video: torch.Tensor) -> torch.Tensor:
        """VAE-encode a pixel-space video batch.

        Args:
            pixel_video: ``[1, 3, T, H, W]`` in ``[-1, 1]``.

        Returns:
            ``[1, C, T_latent, H_latent, W_latent]`` normalized latent.
        """
        video = pixel_video.to(device=self.device, dtype=self.vae.dtype)
        latent = self.vae.encode(video).latent_dist.mode()

        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, -1, 1, 1, 1)
                .to(latent.device, latent.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, -1, 1, 1, 1)
                .to(latent.device, latent.dtype)
            )
            latent = (latent - latents_mean) / latents_std
        else:
            scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
            latent = latent * scaling_factor

        return latent.to(self.dtype)

    def _prepare_latents_conditioned(
        self,
        pixel_video: torch.Tensor,
        condition_frame_indexes: list[int],
        height: int,
        width: int,
        num_frames: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare latents with selected latent frames kept as clean conditioning.

        Returns:
            latents: initial noisy/clean mixture
            velocity_mask: ``1`` on frames to denoise, ``0`` on conditioned frames
            cond_latent: full VAE-encoded latent used for decode re-injection
            condition_mask: ``1`` on conditioned latent frames
        """
        C = self.transformer.latent_channel_size
        T_lat = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        noise = randn_tensor(
            (
                1,
                C,
                T_lat,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            ),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        cond_latent = self._encode_pixel_video(pixel_video)

        condition_mask = torch.zeros(1, 1, T_lat, 1, 1, device=self.device, dtype=self.dtype)
        for idx in condition_frame_indexes:
            if 0 <= idx < T_lat:
                condition_mask[:, :, idx, :, :] = 1.0

        if condition_mask.sum() == 0:
            raise ValueError(
                f"No valid conditioning latent frames in {condition_frame_indexes} "
                f"for T_lat={T_lat}."
            )

        latents = condition_mask * cond_latent + (1.0 - condition_mask) * noise
        velocity_mask = 1.0 - condition_mask
        return latents, velocity_mask, cond_latent, condition_mask

    # =========================================================================
    # VAE decode
    # =========================================================================

    @nvtx_range("_decode_latents", color="blue")
    def _decode_latents(self, latents):
        latents = latents.to(self.vae.dtype)

        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            if not hasattr(self, "_latents_mean"):
                self._latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, -1, 1, 1, 1)
                    .to(self.device, self.vae.dtype)
                )
                self._latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, -1, 1, 1, 1)
                    .to(self.device, self.vae.dtype)
                )
            latents = (latents * self._latents_std) + self._latents_mean
        else:
            scaling_factor = self.vae.config.get("scaling_factor", 1.0)
            latents = latents / scaling_factor

        video = self.vae.decode(latents, return_dict=False)[0]
        video = postprocess_video_tensor(video)
        return video

    # =========================================================================
    # Audio generation
    # =========================================================================

    def decode_audio(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode audio latent tokens back to waveform.

        Args:
            latent: Audio latent tensor of shape (B, C, T).

        Returns:
            Waveform tensor of shape (B, audio_channels, N_samples).
        """
        return self.audio_tokenizer.decode(latent)  # [B, audio_channels, N_samples]

    # =========================================================================
    # Forward (main generation entry point)
    # =========================================================================

    @nvtx_range("Cosmos3OmniMoTPipeline.forward")
    @torch.inference_mode()
    def forward(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[str] = None,
        image: Optional[Union[PIL.Image.Image, torch.Tensor, str]] = None,
        video: Optional[Union[str, bytes]] = None,
        height: int = COSMOS3_720P_PARAMS["height"],
        width: int = COSMOS3_720P_PARAMS["width"],
        num_frames: int = COSMOS3_720P_PARAMS["num_frames"],
        num_inference_steps: int = COSMOS3_720P_PARAMS["num_inference_steps"],
        guidance_scale: float = COSMOS3_720P_PARAMS["guidance_scale"],
        seed: int = 42,
        max_sequence_length: int = COSMOS3_720P_PARAMS["max_sequence_length"],
        frame_rate: float = COSMOS3_720P_PARAMS["frame_rate"],
        use_duration_template: bool = COSMOS3_EXTRA_SPECS["use_duration_template"].default,
        use_resolution_template: bool = COSMOS3_EXTRA_SPECS["use_resolution_template"].default,
        use_system_prompt: bool = COSMOS3_EXTRA_SPECS["use_system_prompt"].default,
        use_guardrails: bool = COSMOS3_EXTRA_SPECS["use_guardrails"].default,
        enable_audio: bool = COSMOS3_EXTRA_SPECS["enable_audio"].default,
    ):
        pipeline_start = time.time()
        timer = CudaPhaseTimer()
        timer.mark_pre_start()

        use_guardrails = use_guardrails and not TRTLLM_DISABLE_COSMOS3_GUARDRAILS

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        if batch_size > 1:
            # TODO: support batch generation
            raise ValueError("Batch generation is not supported for Cosmos3")

        if image is not None and video is not None:
            raise ValueError("Only one of `image` or `video` may be provided for conditioning.")

        if image is not None and not isinstance(image, (PIL.Image.Image, torch.Tensor, str)):
            raise ValueError(
                f"`image` must be a PIL.Image, torch.Tensor, or file path string, "
                f"got {type(image)}."
            )

        if video is not None and not isinstance(video, (str, bytes)):
            raise ValueError(f"`video` must be a file path string or raw bytes, got {type(video)}.")

        has_conditioning = image is not None or video is not None
        # Text guardrail — check both positive and user-supplied negative prompts.
        # None negative_prompt means the hardcoded default will be used (safe); skip it.
        text_blocked = torch.zeros((), device=self.device, dtype=torch.int32)
        if self.rank == 0 and use_guardrails and self.safety_checker is not None:
            prompts_to_check = list(prompt)
            if negative_prompt is not None:
                prompts_to_check.append(negative_prompt)
            for p in prompts_to_check:
                is_safe = self.safety_checker.check_text_safety(p)
                if not is_safe:
                    logger.warning("Text guardrail blocked prompt")
                    text_blocked.fill_(1)
                    break

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(text_blocked, src=0)

        if text_blocked.item():
            timer.mark_end()
            return timer.fill(PipelineOutput())

        generator = torch.Generator(device=self.device).manual_seed(seed)

        if negative_prompt is None:
            negative_prompt = COSMOS3_DEFAULT_NEGATIVE_PROMPT

        negative_prompt = self._format_prompt_with_template(
            negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            use_duration_template=use_duration_template,
            use_resolution_template=use_resolution_template,
        )

        prompt = [
            self._format_prompt_with_template(
                p,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                use_duration_template=use_duration_template,
                use_resolution_template=use_resolution_template,
            )
            for p in prompt
        ]
        logger.info(f"Prompt with metadata: '{prompt}'")

        prompt = prompt[0]

        # 1. Tokenize prompts (no separate text encoder — transformer embeds internally)
        logger.info("Tokenizing prompts...")
        cond_ids, cond_mask = self._tokenize_prompt(prompt, max_sequence_length, use_system_prompt)
        uncond_ids, uncond_mask = self._tokenize_prompt(
            negative_prompt, max_sequence_length, use_system_prompt
        )

        # 2. Prepare latents
        condition_mask = None
        cond_latent = None
        tmp_video_path = None
        if has_conditioning:
            condition_frame_indexes = cosmos3_condition_frame_indexes_for_input(
                has_video_input=video is not None
            )
            try:
                pixel_video, tmp_video_path = build_conditioning_pixel_video(
                    image=image,
                    video=video,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    condition_frame_indexes=condition_frame_indexes,
                    temporal_compression=self.vae_scale_factor_temporal,
                )
                latents, velocity_mask, cond_latent, condition_mask = (
                    self._prepare_latents_conditioned(
                        pixel_video,
                        condition_frame_indexes=condition_frame_indexes,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        generator=generator,
                    )
                )
            finally:
                if tmp_video_path is not None and os.path.exists(tmp_video_path):
                    os.unlink(tmp_video_path)
        else:
            latents = self._prepare_latents(height, width, num_frames, generator)
            velocity_mask = None

        # Compute video shape in latent space
        T_latent = latents.shape[2]
        H_latent = latents.shape[3]
        W_latent = latents.shape[4]
        video_shape = (T_latent, H_latent, W_latent)

        # 3. Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # 3b. Audio noise init
        # T_audio = ceil(duration_s * audio_latent_fps / temporal_compression_factor_audio)
        # Duration derived from num_frames / frame_rate; matches cosmos3-internal.
        do_audio = enable_audio and self.audio_gen and hasattr(self, "audio_tokenizer")
        audio_latents = None
        if do_audio:
            duration_s = num_frames / frame_rate
            T_audio = math.ceil(
                duration_s
                * self.transformer.audio_latent_fps
                / self.transformer.temporal_compression_factor_audio
            )
            audio_latents = randn_tensor(
                (1, self.transformer.audio_dim, T_audio),
                generator=generator,
                device=self.device,
                dtype=latents.dtype,
            )
            # Audio uses the same scheduler type/config as video.
            self.audio_scheduler.set_timesteps(num_inference_steps, device=self.device)

        # 4. Build forward_fn for the denoise loop
        def forward_fn(
            latent_input, extra_stream_latents, timestep, encoder_hidden_states, extra_tensors
        ):
            """Cosmos3 forward function for BasePipeline.denoise().

            Since Cosmos3 embeds text internally, we pass token IDs via extra_tensors
            rather than through encoder_hidden_states.
            """
            current_audio = extra_stream_latents.get("audio") if extra_stream_latents else None

            result = self.transformer(
                hidden_states=latent_input,
                timestep=timestep,
                text_ids=extra_tensors["text_ids"],
                text_mask=extra_tensors["text_mask"],
                video_shape=video_shape,
                fps=frame_rate,
                noisy_frame_mask=velocity_mask,
                audio_latents=current_audio,
            )

            video_noise_pred = result.video
            audio_noise_pred = result.audio

            if velocity_mask is not None:
                video_noise_pred = video_noise_pred * velocity_mask

            if audio_noise_pred is not None:
                return video_noise_pred, {"audio": audio_noise_pred}
            return video_noise_pred

        # 5. Build CFG tensors — text_ids and text_mask need to be split for CFG
        #    BasePipeline.denoise batches [uncond, cond] when guidance_scale > 1
        #    We pass text IDs/masks through extra_cfg_tensors so they get split correctly
        extra_cfg_tensors = {
            "text_ids": (cond_ids, uncond_ids),
            "text_mask": (cond_mask, uncond_mask),
        }

        self.transformer.reset_cache()

        # 6. Denoise
        timer.mark_denoise_start()
        extra_streams = {"audio": (audio_latents, self.audio_scheduler)} if do_audio else None
        denoise_result = self.denoise(
            latents=latents,
            scheduler=self.scheduler,
            prompt_embeds=cond_ids,  # placeholder — actual conditioning via extra_cfg_tensors
            neg_prompt_embeds=uncond_ids,
            guidance_scale=guidance_scale,
            forward_fn=forward_fn,
            extra_cfg_tensors=extra_cfg_tensors,
            extra_streams=extra_streams,
        )

        if extra_streams is not None:
            latents, extra_latents = denoise_result
            audio_latents = extra_latents.get("audio")
        else:
            latents = denoise_result
            audio_latents = None

        timer.mark_post_start()

        # 7. Decode video
        logger.info("Decoding video...")
        decode_start = time.time()

        if condition_mask is not None:
            latents = latents.clone()
            latents = torch.where(
                condition_mask.bool(),
                cond_latent.to(device=latents.device, dtype=latents.dtype),
                latents,
            )

        video = self.decode_latents(latents, self._decode_latents)

        # 7b. Decode audio
        waveform = None
        if do_audio and audio_latents is not None:
            logger.info("Decoding audio...")
            waveform = self.decode_audio(audio_latents)  # [B, audio_channels, N_samples]

        # Video guardrail
        if self.rank == 0:
            logger.info(f"Video decoded in {time.time() - decode_start:.2f}s")
            logger.info(f"Total pipeline time: {time.time() - pipeline_start:.2f}s")

            if use_guardrails and self.safety_checker is not None:
                video = check_video_safety(video, self.safety_checker)

        timer.mark_end()
        return timer.fill(
            PipelineOutput(
                video=video,
                frame_rate=frame_rate,
                audio=waveform,
                audio_sample_rate=self.audio_tokenizer.model_config["sampling_rate"]
                if waveform is not None
                else None,
            )
        )
