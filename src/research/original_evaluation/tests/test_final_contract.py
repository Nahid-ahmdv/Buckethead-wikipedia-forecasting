import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_experiment as e


class FinalContract(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2015-07-01", "2026-10-31")
        self.y = pd.Series(100 + np.arange(len(dates)) % 19, index=dates, dtype=float)
        self.issue = "2025-11-02"

    def test_exact_schedule_and_bridge(self):
        s = e.schedule()
        self.assertEqual(len(s), 13)
        all_dates = []
        for _, row in s.iterrows():
            o = e.origin_at(self.y, row.target_start)
            self.assertEqual(str(o.cutoff.date()), row.information_cutoff_date)
            self.assertEqual(str(o.targets[-1].date()), row.target_end)
            self.assertEqual(len(o.bridge), 29)
            self.assertTrue(o.bridge[1:].equals(o.targets))
            all_dates.extend(o.targets)
        self.assertEqual(len(set(all_dates)), 364)
        self.assertEqual(min(all_dates), pd.Timestamp("2025-11-02"))
        self.assertEqual(max(all_dates), pd.Timestamp("2026-10-31"))

    def test_roster_and_schedule_reject_additions(self):
        with self.assertRaises(ValueError): e.origin_at(self.y, "2025-11-03")
        with self.assertRaises(ValueError): e.origin_at(self.y, "2025-11-02 12:00")
        with self.assertRaises(ValueError): e.predict(self.y, self.issue, {"family": "prophet"})
        with self.assertRaises(ValueError): e.f.at_origin(self.y, self.issue)

    def test_all_engines_receive_only_available_history(self):
        altered = self.y.copy()
        altered.loc["2025-11-01":] = 999999
        seen = []
        def engine(o, window, params):
            self.assertEqual(o.history.index[-1], pd.Timestamp("2025-10-31"))
            seen.append(o.history.copy())
            return np.repeat(o.history.iloc[-1], 28), {"converged": True}
        with patch.object(e.f, "fit_sarimax", side_effect=engine):
            for spec in e.roster():
                a, ia = e.predict(self.y, self.issue, spec)
                b, ib = e.predict(altered, self.issue, spec)
                np.testing.assert_array_equal(a, b)
                self.assertEqual(ia["mase_scale"], ib["mase_scale"])
        self.assertEqual(len(seen), 2)

    def test_later_issue_uses_only_mature_final_observations(self):
        y = self.y.copy()
        y.loc["2025-11-28"] = 12345
        o = e.origin_at(y, "2025-11-30")
        self.assertEqual(o.history.iloc[-1], 12345)
        self.assertNotIn(pd.Timestamp("2025-11-29"), o.history.index)

    def test_baselines_and_common_scale_independently(self):
        o = e.origin_at(self.y, self.issue)
        hist = self.y.loc[:"2025-10-31"]
        a, b = e.f.weekly_naive(o), e.f.weekday_mean(o, 8)
        for i, d in enumerate(o.targets):
            same = hist.loc[hist.index.weekday == d.weekday()]
            self.assertEqual(a[i], same.iloc[-1])
            self.assertEqual(b[i], hist.iloc[-56:].loc[hist.iloc[-56:].index.weekday == d.weekday()].mean())
        r = hist.loc[hist.index > o.cutoff - pd.DateOffset(years=3)].to_numpy()
        self.assertEqual(e.f.mase_scale(o)[0], np.abs(r[7:] - r[:-7]).mean())

    def test_failure_is_recorded_and_not_retried(self):
        with patch.object(e.f, "fit_sarimax", side_effect=e.f.ModelFailure("test failure")) as engine:
            p, info = e.predict(self.y, self.issue, e.roster()[0])
        self.assertTrue(info["fallback"])
        self.assertEqual(engine.call_count, 1)
        np.testing.assert_array_equal(p, e.f.weekly_naive(e.origin_at(self.y, self.issue)))
        with patch.object(e.f, "fit_sarimax", side_effect=TypeError("programming error")):
            with self.assertRaises(TypeError): e.predict(self.y, self.issue, e.roster()[0])

    def test_nonfinite_forecast_falls_back(self):
        with patch.object(e.f, "fit_sarimax", return_value=(np.repeat(np.nan, 28), {})):
            _, info = e.predict(self.y, self.issue, e.roster()[0])
        self.assertTrue(info["fallback"])

    def test_interval_frozen_and_clipping_width(self):
        q = e.proposal()["interval"]["half_width"]
        lo, hi = e.interval_bounds([100, 1000], q)
        np.testing.assert_allclose(lo, [0, 1000-q])
        np.testing.assert_allclose(hi-lo, [100+q, 2*q])

    def test_sensitivity_daily_weight_and_pooled_rmse(self):
        dates = pd.date_range("2025-11-02", periods=364)
        df = pd.DataFrame({"date": dates, "actual": np.arange(364), "prediction": 0., "mase_scale": 2.})
        sub = e.populations(df)["exclude_november_2025"]
        self.assertEqual(len(sub), 335)
        score = e.summarize(sub)
        self.assertEqual(score["mae"], np.arange(29, 364).mean())
        self.assertEqual(score["rmse"], np.sqrt(np.mean(np.arange(29, 364, dtype=float)**2)))

    def test_missing_history_rejected(self):
        self.y.loc["2025-10-31"] = np.nan
        with self.assertRaises(ValueError): e.origin_at(self.y, self.issue)


if __name__ == "__main__": unittest.main()
