import tempfile
import unittest
from pathlib import Path
import torch
from lstm_model import LSTMRegressor, load_checkpoint, predict_aqi


class LSTMChecks(unittest.TestCase):
    def test_samples_do_not_share_recurrent_state(self):
        torch.manual_seed(42)
        model = LSTMRegressor().eval()
        x = torch.randn(3, 7, 1)
        with torch.inference_mode():
            batch = model(x)
            separate = torch.cat([model(row.unsqueeze(0)) for row in x])
            reordered = model(x.flip(0)).flip(0)
        self.assertEqual(batch.shape, (3,))
        torch.testing.assert_close(batch, separate, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(batch, reordered, atol=1e-6, rtol=1e-5)

    def test_checkpoint_reload_and_inverse_scaling(self):
        torch.manual_seed(42)
        model = LSTMRegressor().eval()
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
        model = LSTMRegressor()
        saved = dict(scaler=dict(mean=0., scale=1.))
        for history in ([1, 2], [1, 2, 3, 4, 5, 6, float('nan')], [-1]*7):
            with self.assertRaises(ValueError):
                predict_aqi(model, saved, history)


if __name__ == '__main__':
    unittest.main()
