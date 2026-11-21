#!/usr/bin/env python3
"""Independently reconcile recorded development results with raw fit caches."""
from pathlib import Path
import sys
import json
import hashlib
import math

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"src"))
import forecasting as f


def verify():
    n_frozen=f.phase0_integrity()
    results=HERE/"results"
    y=f.load_development()
    paths=pd.read_csv(results/"development_forecasts.csv")
    metrics=pd.read_csv(results/"origin_metrics.csv")
    schedule=pd.read_csv(f.ROOT/"protocol/split_schedule.csv")
    environment=json.loads((results/"environment.json").read_text())
    for relative,expected_hash in environment["inputs_and_sources"].items():
        assert hashlib.sha256((f.ROOT/relative).read_bytes()).hexdigest()==expected_hash,relative
    implementation=hashlib.sha256(json.dumps(environment["inputs_and_sources"],sort_keys=True).encode()).hexdigest()
    assert implementation==environment["implementation_sha256"]
    expected={r.origin_id:r for _,r in schedule.iterrows() if r.use_for_hyperparameter_tuning or r.use_for_procedure_selection}
    assert len(metrics)==324 and len(paths)==9072
    assert set(paths.stage)=={"tuning","selection"} and paths.date.max()=="2025-10-04"
    assert len(set(paths.loc[paths.stage=="tuning","procedure_id"]))==19
    assert len(set(paths.loc[paths.stage=="selection","procedure_id"]))==8
    assert not (pd.to_datetime(paths.date)>=f.FINAL_START).any()
    for _,row in metrics.iterrows():
        cached=json.loads((results/"cache"/(row.cache_key+".json")).read_text())
        assert cached["implementation_sha256"]==implementation
        assert cached["spec"]=={"family":row.family,"window":row.window,"params":json.loads(row.params)}
        dates=pd.to_datetime(cached["target_dates"])
        scheduled=expected[row.origin_id]
        assert str(dates[0].date())==scheduled.target_start and str(dates[-1].date())==scheduled.target_end
        assert len(dates)==28 and cached["info"]["cutoff"]==scheduled.information_cutoff_date
        assert cached["info"]["bridge_days"]==29 and cached["info"]["bridge_start"]<scheduled.target_start
        group=paths.loc[(paths.stage==row.stage)&(paths.origin_id==row.origin_id)&(paths.procedure_id==row.procedure_id)]
        assert group.date.tolist()==cached["target_dates"]
        np.testing.assert_allclose(group.prediction,cached["prediction"],rtol=1e-13)
        np.testing.assert_array_equal(group.actual,y.loc[dates].to_numpy())
        errors=group.actual.to_numpy()-group.prediction.to_numpy()
        np.testing.assert_allclose([row.mae,row.rmse,row.mase],[np.abs(errors).mean(),np.sqrt(np.square(errors).mean()),np.abs(errors).mean()/row.mase_scale],rtol=1e-12)
        info=cached["info"]
        if "train_end" in info:assert info["train_end"]<=scheduled.information_cutoff_date
        if "label_end" in info:assert info["label_end"]<=scheduled.information_cutoff_date
        if "train_origin_start" in info and row.window=="three_years":
            assert pd.Timestamp(info["train_origin_start"])>pd.Timestamp(row.cutoff)-pd.DateOffset(years=3)
        if row.family in ("sarimax","prophet") and not row.fallback:
            hist=f.training_history(f.at_origin(y,scheduled.target_start),row.window)
            assert info["train_start"]==str(hist.index[0].date())
        assert bool(info["fallback"])==bool(row.fallback)
    assert metrics.groupby("origin_id").mase_scale.nunique().eq(1).all()
    for identifier,group in metrics.groupby("origin_id"):
        cutoff=pd.Timestamp(group.cutoff.iloc[0])
        reference=y.loc[(y.index>cutoff-pd.DateOffset(years=3))&(y.index<=cutoff)].to_numpy()
        expected_scale=np.mean(np.abs(reference[7:]-reference[:-7]))
        np.testing.assert_allclose(group.mase_scale,expected_scale,rtol=1e-13)
    summary=pd.read_csv(results/"selection_summary.csv")
    for _,row in summary.iterrows():
        group=paths.loc[(paths.stage=="selection")&(paths.procedure_id==row.procedure_id)]
        np.testing.assert_allclose([row.mae,row.rmse,row.mase],[group.absolute_error.mean(),np.sqrt(group.squared_error.mean()),(group.absolute_error/group.mase_scale).mean()],rtol=1e-12)
    config=json.loads((results/"proposed_final_configuration.json").read_text())
    assert config["implementation_sha256"]==implementation
    chosen=summary.loc[summary.procedure_id==config["selected_procedure_id"]].iloc[0]
    assert chosen.eligible and chosen.mae<=summary.loc[summary.eligible,"mae"].min()+1e-9
    calibration=pd.read_csv(results/"calibration_errors.csv")
    assert len(calibration)==336 and set(calibration.procedure_id)=={config["selected_procedure_id"]}
    assert calibration.date.max()=="2025-10-04"
    rank=math.ceil((len(calibration)+1)*.8)
    assert rank==270 and config["interval"]["half_width"]==np.sort(calibration.absolute_error)[rank-1]
    assert hashlib.sha256((results/"development_forecasts.csv").read_bytes()).hexdigest()==config["forecast_results_sha256"]
    registry=json.loads((results/"candidate_registry.json").read_text())
    assert len(registry["specifications"])==19
    assert config["final_outcomes_loaded"] is False and config["final_forecasts_computed"] is False
    return {"status":"passed","phase0_frozen_files_verified":n_frozen,"fit_records_verified":324,
        "daily_predictions_verified":9072,"scored_tuning_origins":12,"scored_selection_origins":12,
        "final_predictions":0,"final_performance_rows":0,"common_scale_verified":True,
        "calibration_n":336,"calibration_rank":270,"latest_scored_date":"2025-10-04"}


if __name__=="__main__":
    report=verify()
    (HERE/"results/verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
