"""Read-only independent reconciliation; writes a report only when run as a script."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import final_experiment as e


def independent_scores(g):
    errors = g.actual.to_numpy() - g.prediction.to_numpy()
    return [np.mean(np.abs(errors)), np.sqrt(np.dot(errors, errors) / len(errors)),
            np.mean(np.abs(errors) / g.mase_scale.to_numpy())]


def independent_fourier(index):
    t = (index - pd.Timestamp("2015-01-01")).days.to_numpy()
    return pd.DataFrame({f"{name}_{j}": func(2*np.pi*j*t/365.25)
        for j in (1,2) for name,func in (("sin", np.sin), ("cos", np.cos))}, index=index)


def verify():
    e.approved_contract()
    token = e.digest(e.HERE / "execution_contract.json")
    y = pd.read_csv(e.ROOT / "data/processed/buckethead_daily.csv", parse_dates=["date"]).set_index("date")["count"].astype(float)
    p = pd.read_csv(e.RESULTS / "final_forecasts.csv")
    audits = json.loads((e.RESULTS / "fit_audits.json").read_text())
    assert len(p) == 1092 and len(audits) == 39
    assert set(p.procedure_id) == {e.f.spec_id(s) for s in e.roster()}
    assert p.groupby("procedure_id").date.nunique().eq(364).all()
    assert p.groupby(["origin_id", "procedure_id"]).size().eq(28).all()
    q = e.proposal()["interval"]["half_width"]
    reconstruction = []
    for _, s in e.schedule().iterrows():
        cutoff = pd.Timestamp(s.information_cutoff_date)
        hist = y.loc[:cutoff]
        train = hist.loc[hist.index > cutoff - pd.DateOffset(years=3)].asfreq("D")
        values = train.to_numpy()
        scale = np.mean(np.abs(values[7:] - values[:-7]))
        targets = pd.date_range(s.target_start, s.target_end)
        for spec in e.roster():
            identifier = e.f.spec_id(spec)
            g = p.loc[(p.origin_id == s.origin_id) & (p.procedure_id == identifier)]
            assert list(g.date) == [str(d.date()) for d in targets]
            assert list(g.horizon) == list(range(1, 29))
            np.testing.assert_array_equal(g.actual, y.loc[targets])
            np.testing.assert_allclose(g.mase_scale, scale, rtol=1e-13)
            info = next(a for a in audits if a["origin_id"] == s.origin_id and a["procedure_id"] == identifier)
            record = json.loads((e.RESULTS / "cache" / f"{s.origin_id}__{identifier}.json").read_text())
            assert record["contract_sha256"] == token and record["spec"] == spec
            np.testing.assert_allclose(g.prediction, record["prediction"], rtol=1e-13)
            assert info["cutoff"] == s.information_cutoff_date
            assert info["bridge_days"] == 29 and info["bridge_start"] == str((cutoff + pd.Timedelta(days=1)).date())
            assert info["scale_start"] == str(train.index[0].date()) and info["scale_end"] == str(cutoff.date())
            assert info["scale_pairs"] == len(train)-7
            if spec["family"] == "weekly_naive" or info["fallback"]:
                ref = np.array([hist.loc[hist.index.weekday == d.weekday()].iloc[-1] for d in targets])
            elif spec["family"] == "weekday_mean":
                recent = hist.iloc[-56:]
                ref = np.array([recent.loc[recent.index.weekday == d.weekday()].mean() for d in targets])
            else:
                assert info["converged"] and info["optimizer_result"]["warnflag"] == 0
                assert info["train_start"] == str(train.index[0].date()) and info["train_end"] == str(cutoff.date())
                assert info["train_rows"] == len(train)
                # Reconstruct from saved parameters, with no additional optimizer fit.
                model = SARIMAX(train, exog=independent_fourier(train.index), order=(1,0,1), seasonal_order=(0,1,1,7),
                    trend=None, enforce_stationarity=True, enforce_invertibility=True)
                result = model.filter([info["fitted_parameters"][name] for name in model.param_names])
                bridge = pd.date_range(cutoff + pd.Timedelta(days=1), targets[-1])
                ref = np.maximum(result.get_forecast(29, exog=independent_fourier(bridge)).predicted_mean.iloc[1:], 0)
                reconstruction.append(float(np.max(np.abs(np.asarray(ref) - g.prediction.to_numpy()))))
                assert np.min(np.abs(result.maroots)) > 1 and np.min(np.abs(result.arroots)) > 1
            np.testing.assert_allclose(g.prediction, ref, rtol=0, atol=1e-6)
    np.testing.assert_allclose(p.absolute_error, np.abs(p.actual - p.prediction), rtol=1e-12)
    np.testing.assert_allclose(p.squared_error, (p.actual - p.prediction)**2, rtol=1e-12)
    selected = p.loc[p.family == "sarimax"]
    np.testing.assert_allclose(selected.lower, np.maximum(selected.prediction - q, 0), rtol=1e-13)
    np.testing.assert_allclose(selected.upper, selected.prediction + q, rtol=1e-13)
    assert selected.interval_half_width.eq(q).all()
    summary = pd.read_csv(e.RESULTS / "summary.csv")
    origins = pd.read_csv(e.RESULTS / "origin_metrics.csv")
    intervals = pd.read_csv(e.RESULTS / "interval_metrics.csv")
    pairs = pd.read_csv(e.RESULTS / "paired_differences.csv")
    assert len(summary) == 6 and len(origins) == 75 and len(intervals) == 27 and len(pairs) == 54
    for population, expected_days, expected_paths in (("primary",364,13),("exclude_november_2025",335,12)):
        # An independent date-string mask, applied to the same predictions.
        sub = p if population == "primary" else p.loc[~p.date.str.startswith("2025-11-")]
        for _, row in summary.loc[summary.population == population].iterrows():
            g = sub.loc[sub.procedure_id == row.procedure_id]
            assert row.days == expected_days == len(g) and row.paths == expected_paths
            np.testing.assert_allclose([row.mae,row.rmse,row.mase], independent_scores(g), rtol=1e-12)
        for _, row in origins.loc[origins.population == population].iterrows():
            g = sub.loc[(sub.procedure_id == row.procedure_id) & (sub.origin_id == row.origin_id)]
            assert len(g) == row.days
            np.testing.assert_allclose([row.mae,row.rmse,row.mase], independent_scores(g), rtol=1e-12)
        sel = sub.loc[sub.family == "sarimax"]
        for _, row in intervals.loc[intervals.population == population].iterrows():
            g = sel if row.origin_id == "all" else sel.loc[sel.origin_id == row.origin_id]
            covered = (g.actual.to_numpy() >= g.lower.to_numpy()) & (g.actual.to_numpy() <= g.upper.to_numpy())
            np.testing.assert_allclose([row.coverage,row.mean_width], [np.mean(covered), np.mean(g.upper-g.lower)], rtol=1e-12)
            assert row.days == len(g) and row.below == (g.actual < g.lower).sum() and row.above == (g.actual > g.upper).sum()
        for _, row in pairs.loc[pairs.population == population].iterrows():
            a = sel if row.origin_id == "all" else sel.loc[sel.origin_id == row.origin_id]
            b = sub.loc[sub.family == row.baseline]
            if row.origin_id != "all": b = b.loc[b.origin_id == row.origin_id]
            am, bm = independent_scores(a)[0], independent_scores(b)[0]
            np.testing.assert_allclose([row.selected_minus_baseline_mae,row.relative_mae_improvement_percent], [am-bm,100*(1-am/bm)], rtol=1e-12)
    nov = origins.loc[(origins.population == "exclude_november_2025") & (origins.origin_id == "final_02")]
    assert nov.days.eq(27).all()
    return {"status":"passed", "frozen_artifacts_verified":e.original_integrity(),
        "procedure_origin_records":39, "daily_forecasts":1092, "primary_days_per_procedure":364,
        "sensitivity_days_per_procedure":335, "sensitivity_paths":12,
        "partial_sensitivity_path":"final_02: 27 days; final_01 contributes zero days",
        "metric_reconciliation":"Independent NumPy arithmetic, baseline reconstruction, historical MASE pairs, all interval bounds and paired differences",
        "saved_parameter_sarimax_reconstructions":len(reconstruction),
        "maximum_sarimax_reconstruction_difference":max(reconstruction, default=0),
        "fallbacks":sum(a["fallback"] for a in audits), "recorded_python_warnings":sum(len(a["warnings"]) for a in audits),
        "final_tuning_or_interval_recalibration":False}


if __name__ == "__main__":
    report = verify()
    e.write_json(e.RESULTS / "verification.json", report)
    print(json.dumps(report, indent=2))
