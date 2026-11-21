"""Targeted synthetic and temporal contract tests; no real-series model fits."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import engine as e


def synthetic():
    index=pd.date_range("2021-01-01","2024-03-31")
    t=np.arange(len(index))
    return pd.Series(500+.03*t+25*np.sin(2*np.pi*t/7)+np.random.default_rng(11).normal(0,12,len(t)),index=index,name="count")


class Contract(unittest.TestCase):
    def setUp(self):self.y=synthetic();self.issue="2024-01-28";self.o=e.at_origin(self.y,self.issue,scheduled=False)
    def test_schedule_and_boundaries(self):
        s=e.schedule()
        self.assertEqual(s.groupby("stage").size().to_dict(),{"calibration":4,"development":12,"retrospective":13,"selection":7})
        self.assertTrue((pd.to_datetime(s.issue)-pd.to_datetime(s.cutoff)).dt.days.eq(2).all())
        self.assertTrue((pd.to_datetime(s.target_end)-pd.to_datetime(s.issue)).dt.days.eq(27).all())
        self.assertLessEqual(pd.to_datetime(s.loc[s.stage=="selection","target_end"]).max(),pd.Timestamp("2025-06-13"))
    def test_delay_and_bridge(self):
        self.assertEqual(self.o.cutoff,pd.Timestamp("2024-01-26"))
        self.assertEqual(self.o.bridge[0],pd.Timestamp("2024-01-27"))
        self.assertTrue(self.o.bridge[1:].equals(self.o.targets))
    def test_affine_level_is_training_only_and_shift_equivariant(self):
        h=e.original.training_history(self.o,"three_years")
        z,m,s=e.affine(h);zz,mm,ss=e.affine(h+1000)
        np.testing.assert_allclose(z,zz,atol=1e-12)
        self.assertAlmostEqual(mm-m,1000)
        self.assertAlmostEqual(s,ss)
        np.testing.assert_allclose(m+s*z,h)
    def test_constant_level_affine(self):
        z,m,s=e.affine(pd.Series([700.]*100))
        self.assertEqual(m,700);self.assertEqual(s,1);self.assertTrue(z.eq(0).all())
    def test_future_tail_does_not_change_any_engine_inputs(self):
        other=self.y.copy();other.loc[other.index>self.o.cutoff]=1e9
        seen=[]
        def spy(origin,spec,*args):
            seen.append(origin.history.copy());return np.repeat(origin.history.iloc[-1],28),{"converged":True}
        with patch.object(e,"fit_ets",side_effect=spy),patch.object(e,"fit_sarimax",side_effect=spy),patch.object(e,"fit_lgbm",side_effect=spy):
            for spec in e.roster():
                a,ia=e.predict(self.y,self.issue,spec,scheduled=False);b,ib=e.predict(other,self.issue,spec,scheduled=False)
                np.testing.assert_array_equal(a,b);self.assertEqual(ia["mase_scale"],ib["mase_scale"])
        self.assertTrue(all(h.index[-1]==self.o.cutoff for h in seen))
    def test_weekday_mean_independently(self):
        h=self.y.loc[:self.o.cutoff].iloc[-56:]
        expected=[h.loc[h.index.weekday==d.weekday()].mean() for d in self.o.targets]
        np.testing.assert_allclose(e.weekday_values(self.y,self.issue),expected)
    def test_baseline_training_targets_are_origin_specific(self):
        x,raw,b,meta=e.training_table(self.o)
        self.assertTrue(meta.groupby("training_issue").size().eq(28).all())
        self.assertLessEqual(meta.target_date.max(),self.o.cutoff)
        for issue in meta.training_issue.drop_duplicates().iloc[[0,-1]]:
            positions=np.flatnonzero(meta.training_issue.eq(issue));c=issue-pd.Timedelta(days=2)
            history=self.y.loc[:c].iloc[-56:]
            expected=[history.loc[history.index.weekday==d.weekday()].mean() for d in meta.iloc[positions].target_date]
            np.testing.assert_allclose(b[positions],expected)
            np.testing.assert_allclose(raw[positions],self.y.loc[meta.iloc[positions].target_date])
            np.testing.assert_allclose(x.iloc[positions].count_c_minus_0,self.y.loc[c])
        np.testing.assert_allclose((raw-b)+b,raw)
    def test_day20_features_ignore_day13_actual(self):
        a=e.original.feature_path(self.y,self.issue)
        altered=self.y.copy();altered.loc[self.o.targets[12]]=1e10
        pd.testing.assert_frame_equal(a,e.original.feature_path(altered,self.issue))
    def test_mase_common_reference(self):
        h=e.original.training_history(self.o,"three_years").to_numpy()
        self.assertAlmostEqual(e.original.mase_scale(self.o)[0],np.abs(h[7:]-h[:-7]).mean())
    def test_metrics_pool_squares_and_signed_errors(self):
        g=pd.DataFrame({"prediction":[0.,10.,0.],"actual":[0.,0.,5.],"mase_scale":[2.,2.,2.]})
        m=e.metrics(g);self.assertEqual(m["mae"],5);self.assertAlmostEqual(m["rmse"],np.sqrt(125/3));self.assertEqual(m["mase"],2.5)
        self.assertEqual(m["signed_error"],5/3)
    def test_zero_mase_explicit(self):
        g=pd.DataFrame({"prediction":[1.],"actual":[2.],"mase_scale":[0.]})
        self.assertTrue(np.isnan(e.metrics(g)["mase"]));self.assertEqual(e.metrics(g)["mase_undefined_days"],1)
    def test_calibration_ranks_and_no_interpolation(self):
        g=pd.DataFrame({"date":pd.date_range("2025-06-15",periods=112),"actual":np.arange(112.),"prediction":0.})
        c=e.calibrate(g,"2025-11-02")
        self.assertEqual((c["lower_rank"],c["upper_rank"]),(11,102))
        self.assertEqual((c["lower_offset"],c["upper_offset"]),(10.,101.))
        with self.assertRaises(ValueError):e.calibrate(g.iloc[:-1],"2025-11-02")
        with self.assertRaises(ValueError):e.calibrate(g,"2025-08-01")
    def test_interval_clipping_score_and_misses(self):
        lo,hi=e.interval([5.,50.],{"lower_offset":-10.,"upper_offset":20.})
        np.testing.assert_array_equal(lo,[0.,40.]);np.testing.assert_array_equal(hi,[25.,70.])
        m=e.interval_metrics([0,100],lo,hi)
        self.assertEqual(m["coverage"],.5);self.assertEqual(m["above"],1);self.assertEqual(m["below"],0)
        self.assertEqual(m["interval_score80"],177.5)
    def test_fallback_logged_and_programming_errors_propagate(self):
        spec=e.roster()[0]
        with patch.object(e,"fit_ets",side_effect=e.original.ModelFailure("synthetic failure")):
            p,info=e.predict(self.y,self.issue,spec,scheduled=False)
            self.assertTrue(info["fallback"]);np.testing.assert_array_equal(p,e.original.weekly_naive(self.o))
        with patch.object(e,"fit_ets",side_effect=TypeError("bug")):
            with self.assertRaises(TypeError):e.predict(self.y,self.issue,spec,scheduled=False)
    def test_selection_tie_and_failure_eligibility(self):
        rows=pd.DataFrame({"id":["ets_level","weekday_mean","weekly_naive"],"mae":[1.,1.,2.],"failures":[0,0,0]})
        self.assertEqual(e.choose(rows),"weekday_mean")
        rows.loc[rows.id=="weekday_mean","failures"]=1
        self.assertEqual(e.choose(rows),"ets_level")


class RealSyntheticEngines(unittest.TestCase):
    def test_all_six_engines_forecast_29_then_score_28_and_ignore_future(self):
        y=synthetic();issue="2024-01-28";o=e.at_origin(y,issue,scheduled=False)
        altered=y.copy();altered.loc[altered.index>o.cutoff]=1e8
        for spec in e.roster()[:6]:
            with self.subTest(procedure=spec["id"]):
                a,ia=e.predict(y,issue,spec,scheduled=False);b,ib=e.predict(altered,issue,spec,scheduled=False)
                self.assertFalse(ia["fallback"],ia.get("error"));self.assertFalse(ib["fallback"],ib.get("error"))
                self.assertEqual(len(a),28);np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-12)
                self.assertEqual(ia["cutoff"],"2024-01-26")


if __name__=="__main__":unittest.main(verbosity=2)
