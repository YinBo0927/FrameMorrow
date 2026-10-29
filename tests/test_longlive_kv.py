import unittest
from types import SimpleNamespace

import torch

from framemorrow.adapters import LongLiveHistoryAdapter


class Attention:
    __module__ = 'wan.modules.causal_model_latentmem'
    memory_size = 2
    sink_size = 1


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(training=False, blocks=[SimpleNamespace(self_attn=Attention())])
        self.received = None

    def forward(self, *, kv_cache, crossattn_cache, memory_indices=None, **kwargs):
        self.received = (kv_cache, memory_indices)
        return torch.tensor(1)


class LongLiveKVTest(unittest.TestCase):
    def test_selects_native_entries_without_rebuilding_or_overwriting_bank(self):
        generator = Generator()
        keys = [torch.full((1, 1, 2, 1, 1), float(i)) for i in range(5)]
        values = [k + 10 for k in keys]
        cache = {'k': torch.zeros(1, 6, 1, 1), 'v': torch.zeros(1, 6, 1, 1),
                 'global_end_index': torch.tensor(0), 'local_end_index': torch.tensor(0),
                 'cpu_k_frames': keys, 'cpu_v_frames': values}
        cross = {'k': torch.zeros(1, 1), 'v': torch.zeros(1, 1), 'is_init': False}
        pipe = SimpleNamespace(generator=generator, kv_cache1=[cache], crossattn_cache=[cross], frame_seq_length=2)
        adapter = LongLiveHistoryAdapter(pipe)
        self.assertEqual(list(adapter.available_latent_indices()), [1, 2, 3, 4, 5])
        selected = adapter.refresh(torch.zeros(3, 3, 4, 4), [0.25, 0.5, 1.0], [2, 1],
                                   recent_start=2.0, conditional_dict={}, history_latent_indices=[1, 2, 4])
        self.assertEqual(selected['indices'], [1, 2])
        self.assertEqual(adapter.memory_indices.tolist(), [[1, 3]])
        self.assertIs(cache['cpu_k_frames'], keys)
        self.assertIs(cache['cpu_v_frames'], values)
        adapter.forward(noisy_image_or_video=torch.zeros(1, 1, 1, 1, 1), conditional_dict={},
                        timestep=torch.zeros(1), current_start=4)
        self.assertIs(generator.received[0][0], cache)
        self.assertEqual(generator.received[1].tolist(), [[1, 3]])
        with self.assertRaises(ValueError):
            adapter.refresh(torch.zeros(1, 3, 4, 4), [0.0], [0], recent_start=2.0,
                            conditional_dict={}, history_latent_indices=[0])
        keys.clear()
        values.clear()
        with self.assertRaises(ValueError):
            adapter.generator_kwargs()


if __name__ == '__main__':
    unittest.main()
