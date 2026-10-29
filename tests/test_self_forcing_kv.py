import unittest
from types import SimpleNamespace

import torch

from framemorrow.self_forcing_kv import SelfForcingHistoryAdapter


class Attention:
    __module__ = 'wan.modules.causal_model'
    local_attn_size = 3
    sink_size = 1
    max_attention_size = 6


class Generator:
    def __init__(self):
        self.model = SimpleNamespace(training=False, blocks=[SimpleNamespace(self_attn=Attention())])
        self.seen = None

    def __call__(self, *, kv_cache, crossattn_cache, current_start, **kwargs):
        cache = kv_cache[0]
        self.seen = cache['k'].clone()
        count = 2
        end = min(int(cache['local_end_index']) + count, cache['k'].shape[1])
        cache['k'][:, end - count:end] = current_start + 10
        cache['v'][:, end - count:end] = current_start + 20
        cache['local_end_index'].fill_(end)
        cache['global_end_index'].fill_(current_start + count)
        return torch.tensor(1)


class HistoryKVTest(unittest.TestCase):
    def test_archives_clean_kv_and_uses_it_after_native_cache_changes(self):
        generator = Generator()
        cache = {'k': torch.zeros(1, 6, 1, 1), 'v': torch.zeros(1, 6, 1, 1),
                 'global_end_index': torch.tensor(0), 'local_end_index': torch.tensor(0)}
        cross = {'k': torch.zeros(1, 1), 'v': torch.zeros(1, 1), 'is_init': False}
        pipe = SimpleNamespace(generator=generator, kv_cache1=[cache], crossattn_cache=[cross],
                               frame_seq_length=2)
        adapter = SelfForcingHistoryAdapter(pipe, max_references=2)
        generator(kv_cache=[cache], crossattn_cache=[cross], current_start=0)
        adapter.capture(0, 1)
        generator(kv_cache=[cache], crossattn_cache=[cross], current_start=2)
        adapter.capture(1, 1)
        cache['k'].fill_(99)
        cache['v'].fill_(99)
        selected = adapter.refresh(torch.zeros(2, 3, 4, 4), [0.0, 0.25], [1, 0],
                                   recent_start=1.0, conditional_dict={}, history_latent_indices=[0, 1])
        self.assertEqual(selected['indices'], [0, 1])
        self.assertEqual(adapter.banks[0][0][0].flatten().tolist(), [10.0, 10.0])
        self.assertEqual(adapter.banks[0][0][1].flatten().tolist(), [12.0, 12.0])
        adapter.forward(noisy_image_or_video=torch.zeros(1), conditional_dict={},
                        timestep=torch.zeros(1), current_start=4)
        self.assertEqual(generator.seen[0, :4, 0, 0].tolist(), [10.0, 10.0, 12.0, 12.0])
        self.assertEqual(generator.model.blocks[0].self_attn.sink_size, 1)
        self.assertEqual(generator.model.blocks[0].self_attn.max_attention_size, 6)
        self.assertEqual(int(cache['global_end_index']), 6)
        with self.assertRaises(ValueError):
            adapter.capture(1, 1)
        with self.assertRaises(ValueError):
            adapter.refresh(torch.zeros(1, 3, 4, 4), [0.0], [0], recent_start=1.0,
                            conditional_dict={}, history_latent_indices=[99])


if __name__ == '__main__':
    unittest.main()
