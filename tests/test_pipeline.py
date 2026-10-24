import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from pipeline import make_samples, split_samples, load_daily


class PipelineChecks(unittest.TestCase):
    def test_missing_day_never_bridges_window(self):
        idx = pd.date_range('2021-01-01', periods=20)
        daily = pd.DataFrame({'pm25': np.arange(20, dtype=float)}, index=idx)
        daily.loc[idx[9], 'pm25'] = np.nan
        result = make_samples(daily)
        self.assertFalse(result.index.isin(idx[9:17]).any())
        self.assertEqual(len(result), 5)
        self.assertEqual(result.loc[idx[17], 'lag_7'], 10)
        self.assertEqual(result.loc[idx[17], 'target'], 17)

    def test_partition_by_label_date_allows_observed_history(self):
        idx = pd.date_range('2021-01-01', periods=100)
        daily = pd.DataFrame({'pm25': np.arange(100)}, index=idx)
        result = split_samples(make_samples(daily), idx)
        self.assertEqual(result.loc[idx[69], 'split'], 'train')
        self.assertEqual(result.loc[idx[70], 'split'], 'validation')
        self.assertEqual(result.loc[idx[85], 'split'], 'test')
        self.assertEqual(result.loc[idx[70], 'lag_1'], 69)
        self.assertEqual(result.loc[idx[70], 'target'], 70)

    def test_loader_distinguishes_absent_dates_and_spaces(self):
        with patch('pipeline.pd.read_csv', return_value=pd.DataFrame({'date': ['2021-01-03', '2021-01-01'], ' pm25': [' ', 100]})):
            daily = load_daily('input.csv', '2021-01-01', '2021-01-03')
            self.assertEqual(daily['missing_reason'].tolist(), ['', 'date_absent', 'pm25_empty'])
            self.assertEqual(daily['pm25'].iloc[0], 100)

    def test_duplicate_dates_rejected(self):
        with patch('pipeline.pd.read_csv', return_value=pd.DataFrame({'date': ['2021-01-01', '2021-01-01'], 'pm25': [20, 30]})):
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_daily('input.csv', '2021-01-01', '2021-01-03')


if __name__ == '__main__':
    unittest.main()
