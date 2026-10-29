"""Self-Forcing and LongLive historical KV conditioning.

Self-Forcing uses its stock rotated-K cache with a bounded local window.
LongLive requires memory_indices and unrotated CPU history-bank hooks.
Use serially between generation chunks, after native state is initialized.
"""
import inspect
import math

import torch

from .data import validate_times
from .self_forcing_kv import SelfForcingHistoryAdapter


@torch.inference_mode()
def encode_wan_references(vae, rgb):
    """Encode [K,3,H,W] float RGB in [0,1] as K independent native frames.

    Self-Forcing and LongLive WanVAEWrapper take [B,3,T,H,W] in [-1,1]
    and return [B,T,C,h,w]. Independent encoding avoids temporal compression
    across unrelated reference images. Inputs follow the VAE's device/dtype;
    this helper does not move or reconfigure the VAE itself.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3 or len(rgb) == 0:
        raise ValueError('references must be nonempty [K,3,H,W] RGB')
    if not rgb.is_floating_point() or not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError('references must be finite floating-point RGB in [0,1]')
    parameter = next(vae.parameters(), None) if hasattr(vae, 'parameters') else None
    if parameter is not None:
        rgb = rgb.to(device=parameter.device, dtype=parameter.dtype)
    latents = []
    for frame in rgb:
        value = vae.encode_to_latent((frame * 2 - 1)[None, :, None])
        if value.ndim != 5 or value.shape[:2] != (1, 1) or not torch.isfinite(value).all():
            raise ValueError('native VAE must return one finite [1,1,C,h,w] latent per reference')
        latents.append(value)
    return torch.cat(latents, dim=1)



def _wan_caches(pipeline):
    caches = pipeline.kv_cache1
    cross = pipeline.crossattn_cache
    if not caches or not cross or len(caches) != len(cross):
        raise ValueError('initialize native KV and cross-attention caches before refreshing references')
    if any(cache['k'].ndim != 4 or cache['k'].shape[0] != 1 or
           cache['v'].shape != cache['k'].shape for cache in caches):
        raise ValueError('expected batch-one native [1,tokens,heads,dim] KV caches')
    return caches, cross



def _selected_indices(history_rgb, history_times, indices, recent_start, max_references):
    if history_rgb.ndim != 4 or history_rgb.shape[1] != 3 or len(history_rgb) != len(history_times):
        raise ValueError('history must be [N,3,H,W] RGB with N timestamps')
    if not math.isfinite(recent_start):
        raise ValueError('recent_start must be finite')
    validate_times(history_times, [recent_start])
    if torch.is_tensor(indices):
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError('selector indices must be a one-dimensional integer tensor')
        indices = indices.tolist()
    else:
        indices = list(indices)
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(history_rgb) for i in indices):
        raise ValueError('indices must be original candidate-array integer indices')
    if len(set(indices)) != len(indices) or len(indices) > max_references:
        raise ValueError('reference indices must be unique and fit the backend reference budget')
    indices.sort(key=lambda i: history_times[i])
    return indices


class LongLiveHistoryAdapter:
    """Select frames from the native unrotated historical KV bank.

    refresh() maps selector indices to original latent frames and native bank indices.
    generator_kwargs() supplies the real memory_indices argument accepted by
    the LongLive generator. Pass it to BOTH denoising and clean-context cache updates.
    Disable the pipeline's old retrieval block so it does not overwrite these
    indices. Refresh again after a native cache reset or prompt change.

    Supports one continuous rollout (batch=1) without resetting its native cache.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.memory_indices = None
        self.selected_indices = []
        self.timestamps = []
        model = pipeline.generator.model
        if getattr(model, 'training', False):
            raise ValueError('put the native generator in eval mode before using history conditioning')
        if 'memory_indices' not in inspect.signature(pipeline.generator.forward).parameters:
            raise ValueError('generator lacks the LongLive memory_indices history hook')
        attentions = [block.self_attn for block in model.blocks]
        if not attentions:
            raise ValueError('generator has no attention blocks')
        for attention in attentions:
            if not type(attention).__module__.endswith('causal_model_latentmem'):
                raise ValueError('adapter requires the inspected unrotated-K LongLive attention contract')
        self.max_references = min(attention.memory_size for attention in attentions)
        if self.max_references < 1:
            raise ValueError('configure a positive native memory_size')
        sink_sizes = {attention.sink_size for attention in attentions}
        if len(sink_sizes) != 1:
            raise ValueError('all attention layers must use the same sink size')
        self.sink_size = sink_sizes.pop()
        self._seen_bank_size = 0

    def _caches(self):
        return _wan_caches(self.pipeline)

    def available_latent_indices(self):
        """Return global latent indices currently stored in every native KV bank."""
        caches, _ = self._caches()
        sizes = [(len(cache.get('cpu_k_frames', [])), len(cache.get('cpu_v_frames', []))) for cache in caches]
        if len(set(sizes)) != 1 or sizes[0][0] != sizes[0][1]:
            raise ValueError('native historical K/V banks must be aligned across layers')
        size = sizes[0][0]
        if size < self._seen_bank_size:
            raise ValueError('native historical KV bank was reset; create a new adapter')
        self._seen_bank_size = size
        return range(self.sink_size, self.sink_size + size)

    @torch.inference_mode()
    def refresh(self, history_rgb, history_times, indices, *, recent_start, conditional_dict,
                history_latent_indices):
        caches, _ = self._caches()
        available = self.available_latent_indices()
        if len(history_latent_indices) != len(history_times):
            raise ValueError('candidate latent indices must match candidate timestamps')
        indices = _selected_indices(history_rgb, history_times, indices, recent_start, self.max_references)
        selected_latents = [history_latent_indices[i] for i in indices]
        if (any(isinstance(i, bool) or not isinstance(i, int) or i not in available for i in selected_latents)
                or len(set(selected_latents)) != len(selected_latents)):
            raise ValueError('selected frames must map to unique native historical KV entries')
        self.memory_indices = (torch.tensor([[i - self.sink_size for i in selected_latents]],
                                            device=caches[0]['k'].device, dtype=torch.long)
                               if indices else None)
        self.selected_indices = indices
        self.timestamps = [float(history_times[i]) for i in indices]
        return {'indices': indices, 'timestamps': self.timestamps, 'reference_count': len(indices)}

    def generator_kwargs(self):
        """Extra kwargs for the native generator; existing cache/current_start kwargs stay unchanged."""
        caches, _ = self._caches()
        if self.memory_indices is not None:
            available = self.available_latent_indices()
            if int(self.memory_indices.max()) >= len(available):
                raise ValueError('selected historical KV is no longer available')
        return {'memory_indices': self.memory_indices}

    @torch.inference_mode()
    def forward(self, *, noisy_image_or_video, conditional_dict, timestep, current_start,
                use_history=True, **kwargs):
        """Forward through LongLive with the selected history attached."""
        caches, cross = self._caches()
        # The upstream CPU path offloads views into a temporary cache that it then rolls.
        # Preserve the evicted entries before that roll; GPU-to-CPU offload already copies.
        evicted = []
        tokens = self.pipeline.frame_seq_length
        new_tokens = noisy_image_or_video.shape[1] * tokens
        for cache in caches:
            count = 0
            if (cache['k'].device.type == 'cpu' and current_start + new_tokens > int(cache['global_end_index'])
                    and new_tokens + int(cache['local_end_index']) > cache['k'].shape[1]):
                count = new_tokens + int(cache['local_end_index']) - cache['k'].shape[1]
            if count:
                if count % tokens:
                    raise ValueError('evicted KV does not align with latent frames')
                start = self.sink_size * tokens
                shape = (1, count // tokens, tokens, *cache['k'].shape[2:])
                k = cache['k'][:, start:start + count].reshape(shape).clone()
                v = cache['v'][:, start:start + count].reshape(shape).clone()
                evicted.append((len(cache.get('cpu_k_frames', [])),
                                [k[:, i:i + 1] for i in range(count // tokens)],
                                [v[:, i:i + 1] for i in range(count // tokens)]))
            else:
                evicted.append(None)
        result = self.pipeline.generator(
            noisy_image_or_video=noisy_image_or_video, conditional_dict=conditional_dict,
            timestep=timestep, current_start=current_start,
            kv_cache=caches, crossattn_cache=cross,
            **(self.generator_kwargs() if use_history else {'memory_indices': None}), **kwargs)
        for cache, saved in zip(caches, evicted):
            if saved is not None:
                old_size, keys, values = saved
                if (len(cache['cpu_k_frames']) != old_size + len(keys) or
                        len(cache['cpu_v_frames']) != old_size + len(values)):
                    raise ValueError('native LongLive history bank did not append the evicted KV')
                cache['cpu_k_frames'][old_size:] = keys
                cache['cpu_v_frames'][old_size:] = values
        return result
