# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""ASDSV speculative denoising for WAN video generation.

Implements the core speculative loop from:
  "ASDSV: Multimodal Generation Made Efficient with Approximate Speculative
  Diffusion and Speculative Verification" (NeurIPS 2025).

The loop replaces BasePipeline.denoise() for WAN pipelines when a draft
transformer (1.3B) is loaded alongside the target transformer (14B).

Scheduler state design
----------------------
FlowMatchEulerDiscreteScheduler.step() auto-increments an internal
_step_index on every call.  In a speculative round we make K draft calls
+ 2 target anchor calls before knowing accept/reject — that is K+2
increments regardless of outcome, leaving _step_index wrong for both
branches (accept wants +K, reject wants +1).

Fix: bypass scheduler.step() entirely.  After scheduler.set_timesteps(N),
scheduler.sigmas is a precomputed tensor of shape [N+1].  The FlowMatch
Euler update is:

    x_{i+1} = x_i + (sigmas[i+1] - sigmas[i]) * noise_pred_i

We compute this directly using the loop variable i as the sigma index.
On accept i += K fast-forwards the index by K; on reject i += 1 discards
the draft steps with no state to roll back.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import torch

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class SpecStats:
    accepted_rounds: int = 0
    rejected_rounds: int = 0
    total_target_calls: int = 0
    total_draft_calls: int = 0
    # Per-round L1 errors at start/end anchors (for diagnostics)
    l1_starts: List[float] = field(default_factory=list)
    l1_ends: List[float] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        n = self.accepted_rounds + self.rejected_rounds
        return self.accepted_rounds / n if n else 0.0


# ---------------------------------------------------------------------------
# CFG noise wrapper
# ---------------------------------------------------------------------------


def _make_cfg_noise_fn(
    forward_fn: Callable,
    guidance_scale: float,
    prompt_embeds: torch.Tensor,
    neg_prompt_embeds: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Wrap a raw forward_fn into (x, t) -> noise_pred with CFG applied.

    Does NOT call scheduler.step() — the caller owns all Euler updates.

    forward_fn signature matches WAN's closure:
        (latents, extra_stream_latents, timestep, encoder_hidden_states, extra_tensors)
    """
    do_cfg = guidance_scale > 1.0

    def noise_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if do_cfg:
            x_in = torch.cat([x, x])
            enc_hs = torch.cat([neg_prompt_embeds, prompt_embeds])
        else:
            x_in = x
            enc_hs = prompt_embeds

        noise_pred = forward_fn(x_in, None, t, enc_hs, {})

        if do_cfg:
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        return noise_pred

    return noise_fn


# ---------------------------------------------------------------------------
# Speculative denoising loop
# ---------------------------------------------------------------------------


def speculative_denoise(
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    target_noise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    draft_noise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    spec_config,  # SpeculativeConfig from config.py
) -> Tuple[torch.Tensor, SpecStats]:
    """ASDSV speculative denoising loop with explicit sigma-based stepping.

    Each speculative round at step i with window K:

      1. noise_pred = target(latents_i, t_i)
         anchor_1 = euler(noise_pred, latents_i, i)        <- T_{i+1}

      2. draft^K from same latents_i, each with its own sigma index:
         D_{i+1}, D_{i+2}, ..., D_{i+K}

      3. noise_pred = target(D_{i+K-1}, t_{i+K-1})
         anchor_K = euler(noise_pred, D_{i+K-1}, i+K-1)   <- T_{i+K}
         Checking T_{i+K} vs D_{i+K} verifies whether the draft's final
         step matched what the target would produce from the same input.

      4. verify:
           L1(D_{i+1}, anchor_1) <= delta   (first-step alignment)
           L1(D_{i+K}, anchor_K) <= delta   (last-step alignment)

      5. accept -> latents = D_{i+K}, i += K   (sigma index jumps K)
         reject -> latents = anchor_1,  i += 1 (sigma index moves 1, draft discarded)

    On rejection latents = anchor_1 = target(latents_i), which is always
    on-distribution — the next round's draft starts from a clean target output.
    """
    N = len(timesteps)
    warmup_steps = int(spec_config.warmup_ratio * N)
    stage1_end = warmup_steps + int(spec_config.stage1_ratio * N)
    stats = SpecStats()

    def euler_step(noise_pred: torch.Tensor, x: torch.Tensor, idx: int) -> torch.Tensor:
        # Cast sigma diff to latent dtype (sigmas are float32, latents are bfloat16)
        sigma_diff = (sigmas[idx + 1] - sigmas[idx]).to(x.dtype)
        return x + sigma_diff * noise_pred

    i = 0
    while i < N:
        t = timesteps[i]

        # Stage-0: target-only warmup — high-noise steps are too volatile for speculation
        if i < warmup_steps:
            noise_pred = target_noise_fn(latents, t)
            latents = euler_step(noise_pred, latents, i)
            stats.total_target_calls += 1
            i += 1
            continue

        K = spec_config.gamma_1 if i < stage1_end else spec_config.gamma_2
        # Don't overshoot; also skip speculation for K<=1 (2 target calls per step is never a win)
        K = min(K, N - i - 1)
        if K <= 1:
            noise_pred = target_noise_fn(latents, t)
            latents = euler_step(noise_pred, latents, i)
            stats.total_target_calls += 1
            i += 1
            continue

        # ── Speculative round ──────────────────────────────────────────────

        # First anchor: target at step i -> T_{i+1}
        noise_pred = target_noise_fn(latents, t)
        anchor_1 = euler_step(noise_pred, latents, i)
        stats.total_target_calls += 1

        # Draft: K steps from same latents_i, each using explicit sigma index i+k
        d = latents
        draft_states = []
        for k in range(K):
            noise_pred = draft_noise_fn(d, timesteps[i + k])
            d = euler_step(noise_pred, d, i + k)
            stats.total_draft_calls += 1
            draft_states.append(d)
        # draft_states[0]  = D_{i+1}  (euler at sigma index i   -> i+1)
        # draft_states[-1] = D_{i+K}  (euler at sigma index i+K-1 -> i+K)

        # Second anchor: target at step i+K-1 applied to draft's (K-1)th output -> T_{i+K}
        noise_pred = target_noise_fn(draft_states[-2], timesteps[i + K - 1])
        anchor_K = euler_step(noise_pred, draft_states[-2], i + K - 1)
        stats.total_target_calls += 1

        # Verification: per-element L1 mean at both endpoints
        l1_start = (draft_states[0] - anchor_1).abs().mean().item()
        l1_end = (draft_states[-1] - anchor_K).abs().mean().item()
        stats.l1_starts.append(l1_start)
        stats.l1_ends.append(l1_end)

        if l1_start <= spec_config.delta and l1_end <= spec_config.delta:
            latents = draft_states[-1]  # accept: keep D_{i+K}
            i += K  # sigma index jumps K steps forward
            stats.accepted_rounds += 1
        else:
            latents = anchor_1  # reject: keep T_{i+1}, draft discarded
            i += 1  # sigma index moves 1 step
            stats.rejected_rounds += 1

    return latents, stats
