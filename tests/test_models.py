"""CPU checks for tensor shapes, parameterized windows, and gradients."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.loss import forecasting_loss
from models import MODELS, ModelConfig


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_requested_windows_and_horizons(self):
        for name, model_class in MODELS.items():
            for window in (8, 16, 24, 32, 48):
                for horizon in (96, 192, 336, 720):
                    with self.subTest(model=name, window=window, horizon=horizon):
                        torch.manual_seed(2021)
                        config = ModelConfig(window_size=window, pred_len=horizon,
                                             enc_in=3, embed_size=4, d_model=16,
                                             n_heads=2, d_ff=32, dropout=0.0)
                        model = model_class(config)
                        prediction = model(torch.randn(2, 96, 3))
                        self.assertEqual(tuple(prediction.shape), (2, horizon, 3))
                        loss = forecasting_loss(prediction, torch.randn_like(prediction), 0.3)
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        gradients = [p.grad for p in model.parameters() if p.grad is not None]
                        self.assertTrue(gradients)
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_padding_and_single_region(self):
        for name, model_class in MODELS.items():
            for length, window in ((97, 24), (32, 32)):
                config = ModelConfig(window_size=window, seq_len=length, pred_len=12,
                                     enc_in=2, d_model=16, n_heads=2, embed_size=4, d_ff=32)
                result = model_class(config)(torch.randn(1, length, 2))
                self.assertEqual(tuple(result.shape), (1, 12, 2))
                self.assertTrue(torch.isfinite(result).all())

    def test_invalid_configuration(self):
        for values in ({'window_size': 0}, {'window_size': 100},
                       {'window_size': 8, 'd_model': 15, 'n_heads': 2},
                       {'window_size': 8, 'local_kernel_size': 4}):
            with self.assertRaises(ValueError):
                ModelConfig(**values)

    def test_zero_gate_and_variable_independence(self):
        config = ModelConfig(window_size=8, seq_len=32, pred_len=12, enc_in=3,
                             embed_size=4, d_model=16, n_heads=2, d_ff=32,
                             dropout=0.0, channel_mixer_gate_init=0.0)
        x = torch.randn(2, 32, 3)
        perturbed = x.clone()
        perturbed[:, :, 1] *= -2
        for name, model_class in MODELS.items():
            model = model_class(config).eval()
            with torch.no_grad():
                before, after = model(x), model(perturbed)
            torch.testing.assert_close(before[:, :, 0], after[:, :, 0], rtol=0, atol=0)
        mixer = MODELS['tomi-v'](config).channel_mixer
        features = torch.randn(2, 3, 32, 4)
        torch.testing.assert_close(mixer(features), features, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
