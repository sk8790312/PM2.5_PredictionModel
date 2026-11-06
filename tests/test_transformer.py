import tempfile
import unittest
from pathlib import Path
import torch
from transformer_model import CausalTransformer, load_checkpoint, predict_aqi


class TransformerChecks(unittest.TestCase):
    def test_future_history_does_not_change_earlier_hidden_states(self):
        torch.manual_seed(42)
        model = CausalTransformer(dropout=0).eval()
        original = torch.randn(2, 7, 1)
        changed = original.clone()
        changed[:, 4:] += 100
        with torch.inference_mode():
            a, b = model.hidden_states(original), model.hidden_states(changed)
        torch.testing.assert_close(a[:, :4], b[:, :4], rtol=0, atol=1e-6)
        self.assertFalse(torch.allclose(a[:, 4:], b[:, 4:]))

    def test_save_reload_and_aqi_inverse_transform(self):
        torch.manual_seed(42)
        model = CausalTransformer().eval()
        history = [50, 60, 70, 80, 90, 100, 110]
        saved = dict(model_config=dict(), model_state=model.state_dict(),
                     scaler=dict(mean=80., scale=20.))
        expected = predict_aqi(model, saved, history)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            torch.save(saved, path)
            restored, metadata = load_checkpoint(path)
            self.assertEqual(expected, predict_aqi(restored, metadata, history))
        with torch.inference_mode():
            x = (torch.tensor(history).float() - 80) / 20
            direct = model(x.reshape(1, 7, 1)).item() * 20 + 80
        self.assertAlmostEqual(expected, direct, places=6)

    def test_invalid_history_rejected(self):
        model = CausalTransformer()
        saved = dict(scaler=dict(mean=0., scale=1.))
        for history in ([1, 2], [1, 2, 3, 4, 5, 6, float('nan')], [-1]*7):
            with self.assertRaises(ValueError):
                predict_aqi(model, saved, history)


if __name__ == '__main__':
    unittest.main()
