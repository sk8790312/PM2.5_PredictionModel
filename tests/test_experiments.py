import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from sequence_training import prepare_data
from run_stage4 import select_candidate, aggregate_metrics


class ExperimentChecks(unittest.TestCase):
    def daily(self):
        dates = pd.date_range('2021-01-01', periods=100, name='date')
        return pd.DataFrame({'pm25': np.arange(100, dtype=float)}, index=dates)

    def prepare(self, frame):
        with patch('sequence_training.load_daily', return_value=frame):
            return prepare_data('unused.xlsx')

    def test_scaler_and_train_tensors_ignore_all_future_periods(self):
        daily = self.daily()
        a = self.prepare(daily)
        daily.iloc[70:, 0] += 10000
        b = self.prepare(daily)
        self.assertEqual(a['mean'], b['mean'])
        self.assertEqual(a['scale'], b['scale'])
        self.assertAlmostEqual(a['mean'], np.arange(70).mean())
        self.assertAlmostEqual(a['scale'], np.arange(70).std())
        torch.testing.assert_close(a['xt'], b['xt'], rtol=0, atol=0)
        torch.testing.assert_close(a['yt'], b['yt'], rtol=0, atol=0)
        self.assertFalse(torch.equal(a['yv'], b['yv']))

    def test_validation_tensors_ignore_test_period_values(self):
        daily = self.daily()
        a = self.prepare(daily)
        daily.iloc[85:, 0] += 10000
        b = self.prepare(daily)
        for key in ['xt', 'yt', 'xv', 'yv']:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_selection_uses_validation_rmse_and_declared_tie_order(self):
        reports = [dict(validation_metrics=dict(RMSE=3, MAE=1), test_metrics=dict(RMSE=0)),
                   dict(validation_metrics=dict(RMSE=2, MAE=9), test_metrics=dict(RMSE=100)),
                   dict(validation_metrics=dict(RMSE=2, MAE=0), test_metrics=dict(RMSE=1))]
        self.assertEqual(select_candidate(reports), 1)

    def test_statistics_use_sample_std_and_all_three_runs(self):
        reports = [dict(validation_metrics=dict(MAE=x, RMSE=x+2, R2=x/10)) for x in [1,2,3]]
        result = aggregate_metrics(reports)
        self.assertEqual(result['MAE']['mean'], 2)
        self.assertEqual(result['MAE']['std'], 1)
        self.assertEqual(result['MAE']['minimum'], 1)
        self.assertEqual(result['MAE']['maximum'], 3)


if __name__ == '__main__':
    unittest.main()
