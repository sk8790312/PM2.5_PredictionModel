import unittest
import numpy as np
import pandas as pd
from analyze_validation import diagnostic_thresholds, error_metrics, group_masks


class AnalysisChecks(unittest.TestCase):
    def test_thresholds_ignore_future_values_and_do_not_bridge_missing_dates(self):
        data=pd.DataFrame({'pm25':np.arange(100,dtype=float)})
        data.loc[20,'pm25']=np.nan
        a=diagnostic_thresholds(data)
        data.loc[70:,'pm25']=100000
        self.assertEqual(a,diagnostic_thresholds(data))
        self.assertEqual(a['large_daily_change'],1.)

    def test_error_sign_and_group_boundaries(self):
        metrics=error_metrics([10,20],[12,16])
        self.assertEqual(metrics['MAE'],3)
        self.assertEqual(metrics['mean_error'],-1)
        self.assertAlmostEqual(metrics['RMSE'],np.sqrt(10))
        frame=pd.DataFrame({'daily_change':[-45,-44,0,44,45], 'actual':[1,2,3,4,5]})
        masks=group_masks(frame,dict(large_daily_change=45,high_aqi=4))
        self.assertEqual(masks['large_change'].tolist(),[True,False,False,False,True])
        self.assertEqual(int(masks['large_rise'].sum()),1)
        self.assertEqual(int(masks['large_fall'].sum()),1)
        self.assertTrue((masks['ordinary_change'] ^ masks['large_change']).all())
        self.assertEqual(int(masks['high_aqi'].sum()),2)


if __name__=='__main__':
    unittest.main()
