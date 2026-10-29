"""Self-Forcing KV history and LongLive RGB reference conditioning.

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

def _wan_reference_banks(pipeline, rgb, caches, cross, conditional_dict):
    count, native = len(rgb), caches[0]['k']
    latents = encode_wan_references(pipeline.vae, rgb).to(device=native.device, dtype=native.dtype)
    patch = pipeline.generator.model.patch_size
    if (latents.shape[-2] % patch[1] or latents.shape[-1] % patch[2] or patch[0] != 1):
        raise ValueError('reference latent dimensions differ from native patch geometry')
    frame_tokens = (latents.shape[-2] // patch[1]) * (latents.shape[-1] // patch[2])
    if frame_tokens != pipeline.frame_seq_length:
        raise ValueError('reference RGB resolution must match the native generation resolution')
    if any(count * frame_tokens > cache['k'].shape[1] for cache in caches):
        raise ValueError('reference prefill exceeds native context cache capacity')
    scratch = [{
        'k': cache['k'].new_zeros((1, count * frame_tokens, *cache['k'].shape[2:])),
        'v': cache['v'].new_zeros((1, count * frame_tokens, *cache['v'].shape[2:])),
        'global_end_index': torch.zeros_like(cache['global_end_index']),
        'local_end_index': torch.zeros_like(cache['local_end_index']),
        'cpu_k_frames': [], 'cpu_v_frames': [],
    } for cache in caches]
    scratch_cross = [{**cache, 'k': torch.zeros_like(cache['k']),
                      'v': torch.zeros_like(cache['v']), 'is_init': False} for cache in cross]
    for index in range(count):
        kwargs = dict(
            noisy_image_or_video=latents[:, index:index + 1],
            conditional_dict=conditional_dict,
            timestep=torch.zeros((1, 1), dtype=torch.int64, device=native.device),
            kv_cache=scratch, crossattn_cache=scratch_cross,
            current_start=index * frame_tokens)
        if "memory_indices" in inspect.signature(pipeline.generator.forward).parameters:
            kwargs["memory_indices"] = None
        pipeline.generator(**kwargs)
    banks = []
    for cache in scratch:
        if int(cache['local_end_index'].item()) != count * frame_tokens:
            raise ValueError('native reference prefill did not retain every reference token')
        k = cache['k'].reshape(1, count, frame_tokens, *cache['k'].shape[2:]).to('cpu', copy=True)
        v = cache['v'].reshape(1, count, frame_tokens, *cache['v'].shape[2:]).to('cpu', copy=True)
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():
            raise ValueError('native reference KV contains nonfinite values')
        banks.append(([k[:, i:i + 1] for i in range(count)], [v[:, i:i + 1] for i in range(count)]))
    return banks


class LongLiveHistoryAdapter:
    """Replace only the historical bank; leave live local/sink KV untouched.

    refresh() consumes original RGB candidates, selector indices and timestamps.
    generator_kwargs() supplies the real memory_indices argument accepted by
    the LongLive generator. Pass it to BOTH denoising and clean-context cache updates.
    Disable the pipeline's old retrieval block so it does not overwrite these
    indices. Refresh again after a native cache reset or prompt change.

    Initial implementation supports one rollout (batch=1). Original timestamps
    remain in metadata; historical RoPE placement follows the native backend.
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

    def _caches(self):
        return _wan_caches(self.pipeline)

    @torch.inference_mode()
    def refresh(self, history_rgb, history_times, indices, *, recent_start, conditional_dict):
        caches, cross = self._caches()
        indices = _selected_indices(history_rgb, history_times, indices, recent_start, self.max_references)
        count = len(indices)
        native = caches[0]['k']
        banks = [([], []) for _ in caches]
        memory_indices = None
        if count:
            banks = _wan_reference_banks(self.pipeline, history_rgb[indices], caches, cross, conditional_dict)
            memory_indices = torch.arange(count, device=native.device)[None]
        # Publish only after encoding and prefill succeed; never rewrite live k/v or counters.
        for cache, (keys, values) in zip(caches, banks):
            cache['cpu_k_frames'], cache['cpu_v_frames'] = keys, values
            for name in ('_selected_gpu_signature', '_selected_gpu_k', '_selected_gpu_v'):
                cache.pop(name, None)
        self.memory_indices = memory_indices
        self.selected_indices = indices
        self.timestamps = [float(history_times[i]) for i in indices]
        return {'indices': indices, 'timestamps': self.timestamps, 'reference_count': count}

    def generator_kwargs(self):
        """Extra kwargs for the native generator; existing cache/current_start kwargs stay unchanged."""
        caches, _ = self._caches()
        if self.memory_indices is not None:
            count = self.memory_indices.shape[1]
            if any(min(len(cache['cpu_k_frames']), len(cache['cpu_v_frames'])) < count for cache in caches):
                raise ValueError('native history bank was reset; refresh selected references before generation')
        return {'memory_indices': self.memory_indices}

    @torch.inference_mode()
    def forward(self, *, noisy_image_or_video, conditional_dict, timestep, current_start, **kwargs):
        """Forward through LongLive with the selected history attached."""
        return self.pipeline.generator(
            noisy_image_or_video=noisy_image_or_video, conditional_dict=conditional_dict,
            timestep=timestep, current_start=current_start,
            kv_cache=self.pipeline.kv_cache1, crossattn_cache=self.pipeline.crossattn_cache,
            **self.generator_kwargs(), **kwargs)
