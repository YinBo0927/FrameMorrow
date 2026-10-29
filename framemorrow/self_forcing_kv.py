"""Reuse the native Self-Forcing KV cache for selected historical frames."""
import math

import torch


class SelfForcingHistoryAdapter:
    """Archive clean-context KV and prepend selected history to the native cache."""

    def __init__(self, pipeline, max_references=4):
        self.pipeline = pipeline
        self.max_references = max_references
        self.banks = None
        self.timestamps, self.selected_indices = [], []
        model = pipeline.generator.model
        if model.training or isinstance(max_references, bool) or not isinstance(max_references, int) or max_references < 1:
            raise ValueError('use an eval-mode Self-Forcing model and a positive integer reference budget')
        self.attentions = [block.self_attn for block in model.blocks]
        if not self.attentions or any(not type(a).__module__.endswith('.causal_model') or
                                      a.local_attn_size < 1 for a in self.attentions):
            raise ValueError('requires stock Self-Forcing rotated-K attention with a bounded local window')
        self.history = [[] for _ in self.attentions]

    def _caches(self):
        caches, cross = self.pipeline.kv_cache1, self.pipeline.crossattn_cache
        if not caches or len(caches) != len(self.attentions) or not cross or len(cross) != len(caches):
            raise ValueError('initialize one native KV and cross-attention cache per layer')
        if any(c['k'].ndim != 4 or c['k'].shape[0] != 1 or c['v'].shape != c['k'].shape or
               c['k'].shape[1] != a.max_attention_size for c, a in zip(caches, self.attentions)):
            raise ValueError('native KV capacity must match the batch-one attention window')
        return caches, cross

    @torch.inference_mode()
    def capture(self, start_frame, count):
        """Save the final clean-context KV after generating a contiguous block."""
        caches, _ = self._caches()
        tokens = self.pipeline.frame_seq_length
        if (isinstance(start_frame, bool) or not isinstance(start_frame, int) or
                isinstance(count, bool) or not isinstance(count, int) or count < 1 or
                start_frame != len(self.history[0])):
            raise ValueError('capture requires contiguous, nonempty latent-frame blocks')
        snapshots = []
        for cache in caches:
            end = int(cache['local_end_index'].item())
            if int(cache['global_end_index'].item()) != (start_frame + count) * tokens or end < count * tokens:
                raise ValueError('native cache does not contain the requested clean-context block')
            shape = (1, count, tokens, *cache['k'].shape[2:])
            keys = cache['k'][:, end - count * tokens:end].reshape(shape).to('cpu', copy=True)
            values = cache['v'][:, end - count * tokens:end].reshape(shape).to('cpu', copy=True)
            snapshots.append([(keys[:, i:i + 1], values[:, i:i + 1]) for i in range(count)])
        for history, frames in zip(self.history, snapshots):
            history.extend(frames)

    @torch.inference_mode()
    def refresh(self, history_rgb, history_times, indices, *, recent_start,
                conditional_dict, history_latent_indices):
        """Select archived KV by the original latent index, without re-encoding RGB."""
        self._caches()
        if history_rgb.ndim != 4 or history_rgb.shape[1] != 3 or len(history_rgb) != len(history_times):
            raise ValueError('history must be [N,3,H,W] RGB with N timestamps')
        history_times = [float(t) for t in history_times]
        if (not math.isfinite(recent_start) or len(history_times) != len(history_latent_indices) or
                any(not math.isfinite(float(t)) for t in history_times) or
                any(a >= b for a, b in zip(history_times, history_times[1:])) or
                (history_times and history_times[-1] >= recent_start)):
            raise ValueError('history timestamps must be finite, ordered, and before recent context')
        if torch.is_tensor(indices):
            if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
                raise ValueError('selector indices must be a one-dimensional integer tensor')
            indices = indices.tolist()
        else:
            indices = list(indices)
        if (any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(history_times) for i in indices) or
                len(set(indices)) != len(indices) or len(indices) > self.max_references):
            raise ValueError('selector indices must be unique valid candidate indices')
        indices.sort(key=lambda i: history_times[i])
        latent_indices = [history_latent_indices[i] for i in indices]
        if (any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(self.history[0])
                for i in latent_indices) or len(set(latent_indices)) != len(latent_indices)):
            raise ValueError('selected frames must map to unique archived latent frames')
        banks = []
        for history in self.history:
            frames = [history[i] for i in latent_indices]
            banks.append(([k for k, _ in frames], [v for _, v in frames]))
        self.banks = banks if indices else None
        self.selected_indices = indices
        self.timestamps = [float(history_times[i]) for i in indices]
        return {'indices': indices, 'timestamps': self.timestamps, 'reference_count': len(indices)}

    @torch.inference_mode()
    def forward(self, *, noisy_image_or_video, conditional_dict, timestep, current_start, **kwargs):
        caches, cross = self._caches()
        native = dict(noisy_image_or_video=noisy_image_or_video, conditional_dict=conditional_dict,
                      timestep=timestep, current_start=current_start, **kwargs)
        if not self.banks:
            return self.pipeline.generator(**native, kv_cache=caches, crossattn_cache=cross)
        count = len(self.selected_indices)
        prefix_tokens = count * self.pipeline.frame_seq_length
        combined = []
        for cache, (keys, values) in zip(caches, self.banks):
            key = torch.cat(keys, dim=1).flatten(1, 2).to(cache['k'])
            value = torch.cat(values, dim=1).flatten(1, 2).to(cache['v'])
            combined.append({**cache, 'k': torch.cat((key, cache['k']), dim=1),
                             'v': torch.cat((value, cache['v']), dim=1),
                             'global_end_index': cache['global_end_index'].clone(),
                             'local_end_index': cache['local_end_index'].clone() + prefix_tokens})
        scratch_cross = [{**c, 'k': c['k'].clone(), 'v': c['v'].clone()} for c in cross]
        settings = [(a.sink_size, a.max_attention_size) for a in self.attentions]
        try:
            for a, (sink, window) in zip(self.attentions, settings):
                a.sink_size, a.max_attention_size = sink + count, window + prefix_tokens
            result = self.pipeline.generator(**native, kv_cache=combined, crossattn_cache=scratch_cross)
        finally:
            for a, (sink, window) in zip(self.attentions, settings):
                a.sink_size, a.max_attention_size = sink, window
        for live, generated in zip(caches, combined):
            live['k'].copy_(generated['k'][:, prefix_tokens:])
            live['v'].copy_(generated['v'][:, prefix_tokens:])
            live['global_end_index'].copy_(generated['global_end_index'])
            live['local_end_index'].copy_(generated['local_end_index'] - prefix_tokens)
        for live, generated in zip(cross, scratch_cross):
            live['k'].copy_(generated['k'])
            live['v'].copy_(generated['v'])
            live['is_init'] = generated['is_init']
        return result
