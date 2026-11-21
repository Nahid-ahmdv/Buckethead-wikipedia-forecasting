"""Independent raw-data, metric, paired-control and saved-parameter verification.

No optimizer fits. This module never rewrites earlier phases or their results.
"""
import engine as e
import json,math
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


def scores(g):
    x=g.actual.to_numpy()-g.prediction.to_numpy()
    return [np.abs(x).sum()/len(x),np.sqrt(np.dot(x,x)/len(x)),np.mean(np.abs(x)/g.mase_scale.to_numpy()),-x.mean(),np.mean(x<0)]


def basis(index):
    t=(index-pd.Timestamp("2015-01-01")).days.to_numpy()
    return pd.DataFrame({f"{name}_{j}":fn(2*np.pi*j*t/365.25) for j in (1,2) for name,fn in (("sin",np.sin),("cos",np.cos))},index=index)


def verify():
    e.protect()
    lock=e.digest(e.HERE/"implementation_lock.json")
    for p,digest in json.loads((e.HERE/"implementation_lock.json").read_text())["files"].items():assert e.digest(e.ROOT/p)==digest,p
    raw=json.loads((e.ROOT/"data/raw/pageviews_all-access.json").read_text())["items"]
    y=pd.Series([x["views"] for x in raw],index=pd.to_datetime([x["timestamp"] for x in raw],format="%Y%m%d%H")).sort_index().astype(float)
    p=pd.read_csv(e.RESULTS/"forecasts.csv")
    audits=json.loads((e.RESULTS/"fit_audits.json").read_text())
    assert len(p)==8064 and len(audits)==288 and p.groupby(["origin_id","id"]).size().eq(28).all()
    assert set(p.id)=={s["id"] for s in e.roster()}
    replays=[];mlpairs=[]
    for _,origin in e.schedule().iterrows():
        issue=pd.Timestamp(origin.issue);cutoff=pd.Timestamp(origin.cutoff)
        assert cutoff==issue-pd.Timedelta(days=2)
        dates=pd.date_range(issue,periods=28);bridge=pd.date_range(cutoff+pd.Timedelta(days=1),dates[-1])
        hist=y.loc[:cutoff];train=hist.loc[hist.index>cutoff-pd.DateOffset(years=3)].asfreq("D")
        values=train.to_numpy();scale=np.abs(values[7:]-values[:-7]).mean()
        for spec in e.roster():
            g=p.loc[(p.origin_id==origin.origin_id)&(p.id==spec["id"])]
            a=next(a for a in audits if a["origin_id"]==origin.origin_id and a["id"]==spec["id"])
            cache=json.loads((e.RESULTS/"cache"/(a["cache_key"]+".json")).read_text())
            assert cache["lock_sha256"]==lock and cache["spec"]==spec
            assert list(g.date)==[str(d.date()) for d in dates]
            assert list(g.horizon)==list(range(1,29))
            np.testing.assert_array_equal(g.actual,y.loc[dates])
            np.testing.assert_allclose(g.prediction,cache["prediction"],rtol=1e-13)
            np.testing.assert_allclose(g.mase_scale,scale,rtol=1e-13)
            assert a["cutoff"]==str(cutoff.date()) and a["bridge_days"]==29 and a["bridge_start"]==str(bridge[0].date())
            if a["fallback"] or spec["id"]=="weekly_naive":
                ref=np.array([hist.loc[hist.index.weekday==d.weekday()].iloc[-1] for d in dates])
            elif spec["id"]=="weekday_mean":
                recent=hist.iloc[-56:];ref=np.array([recent.loc[recent.index.weekday==d.weekday()].mean() for d in dates])
            elif spec["family"] in ("ets","sarimax"):
                assert a["train_rows"]==len(train) and a["train_start"]==str(train.index[0].date()) and a["train_end"]==str(cutoff.date())
                assert a["converged"]
                if spec["family"]=="sarimax":
                    m=np.mean(values);std=np.std(values,ddof=0) or 1.
                    np.testing.assert_allclose([a["training_mean"],a["training_std"]],[m,std],rtol=1e-13)
                    model=SARIMAX((train-m)/std,exog=basis(train.index),order=(1,0,1),seasonal_order=(0,spec["params"]["D"],1,7),trend=None,enforce_stationarity=True,enforce_invertibility=True)
                    result=model.filter([a["parameters"][n] for n in model.param_names])
                    ref=m+std*result.get_forecast(29,exog=basis(bridge)).predicted_mean.iloc[1:].to_numpy()
                    assert np.abs(result.arroots).min()>1 and np.abs(result.maroots).min()>1
                else:
                    m=values.mean();np.testing.assert_allclose(a["training_scale"],m,rtol=1e-13)
                    model=ETSModel(train/m,error="add",trend="add" if spec["params"]["trend"] else None,damped_trend=spec["params"]["damped"],seasonal="add",seasonal_periods=7,initialization_method="estimated")
                    result=model.smooth([a["parameters"][n] for n in model.param_names])
                    ref=m*result.forecast(29).iloc[1:].to_numpy()
                ref=np.maximum(ref,0)
                replays.append(float(np.max(np.abs(g.prediction.to_numpy()-ref))))
            else:
                assert a["label_end"]<=str(cutoff.date()) and a["window"]=="expanding"
                continue
            np.testing.assert_allclose(g.prediction,ref,rtol=0,atol=2e-5)
        # ML pairs must be exact matches in all inputs except label definition.
        aa={a["id"]:a for a in audits if a["origin_id"]==origin.origin_id}
        ra,co=aa["lgbm_raw"],aa["lgbm_correction"]
        if not ra["fallback"] and not co["fallback"]:
            for key in ("train_rows","training_paths","feature_names","X_sha256","raw_labels_sha256","training_baselines_sha256","configuration","label_end","first_training_issue","last_training_issue"):
                assert ra[key]==co[key],(origin.origin_id,key)
            assert ra["fit_labels_sha256"]==ra["raw_labels_sha256"]
            # Independent complete-label construction from the logged training issues.
            issues=pd.date_range(ra["first_training_issue"],ra["last_training_issue"],freq="7D")
            labels=[];offsets=[]
            for s in issues:
                d=pd.date_range(s,periods=28);past=y.loc[:s-pd.Timedelta(days=2)].iloc[-56:]
                labels.extend(y.loc[d].to_numpy());offsets.extend([past.loc[past.index.weekday==t.weekday()].mean() for t in d])
            assert max(issues)+pd.Timedelta(days=27)<=cutoff
            assert e.array_hash(labels)==ra["raw_labels_sha256"]
            assert e.array_hash(offsets)==ra["training_baselines_sha256"]
            assert e.array_hash(np.asarray(labels)-np.asarray(offsets))==co["fit_labels_sha256"]
            mlpairs.append(origin.origin_id)
    summ=pd.read_csv(e.RESULTS/"summary.csv");origins=pd.read_csv(e.RESULTS/"origin_metrics.csv");horizons=pd.read_csv(e.RESULTS/"horizon_metrics.csv")
    pairs=pd.read_csv(e.RESULTS/"paired_differences.csv");intervals=pd.read_csv(e.RESULTS/"interval_metrics.csv")
    assert len(summ)==40 and len(origins)==384 and len(horizons)==1120 and len(pairs)==848
    for name,df in (("summary",summ),("origin",origins),("horizon",horizons)):
        for _,row in df.iterrows():
            g=p.loc[(p.stage==row.stage)&(p.id==row.id)]
            if row.population!="all":g=g.loc[~g.date.str.startswith("2025-11-")]
            if name=="origin":g=g.loc[g.origin_id==row.origin_id]
            if name=="horizon":g=g.loc[g.horizon==row.horizon]
            assert len(g)==row.days
            np.testing.assert_allclose([row.mae,row.rmse,row.mase,row.signed_error,row.overprediction_frequency],scores(g),rtol=1e-11,atol=1e-11)
    selection=json.loads((e.RESULTS/"point_selection.json").read_text())
    eligible=summ.loc[(summ.stage=="selection")&(summ.failures==0)]
    expected=min(eligible.loc[eligible.mae<=eligible.mae.min()+1e-9,"id"],key=e.PRIORITY.index)
    assert selection["selected_id"]==expected
    cal=json.loads((e.RESULTS/"interval_calibration.json").read_text())["procedures"]
    assert set(cal)==set([expected,"weekday_mean"])
    for identifier,c in cal.items():
        calibration=p.loc[(p.stage=="calibration")&(p.id==identifier)]
        assert len(calibration)==112 and calibration.date.max()=="2025-10-04"
        residual=np.sort(calibration.actual-calibration.prediction)
        np.testing.assert_allclose([c["lower_offset"],c["upper_offset"]],[residual[10],residual[101]],rtol=1e-13)
        g=p.loc[(p.stage=="retrospective")&(p.id==identifier)]
        np.testing.assert_allclose(g.lower,np.maximum(0,g.prediction+c["lower_offset"]),rtol=1e-13)
        np.testing.assert_allclose(g.upper,np.maximum(0,g.prediction+c["upper_offset"]),rtol=1e-13)
    for _,row in intervals.iterrows():
        g=p.loc[(p.stage=="retrospective")&(p.id==row.id)]
        if row.population!="all":g=g.loc[~g.date.str.startswith("2025-11-")]
        if row.group_type=="simultaneous_complete_paths":
            groups=[z for _,z in g.groupby("origin_id") if len(z)==28]
            cov=np.mean([((z.actual>=z.lower)&(z.actual<=z.upper)).all() for z in groups])
            np.testing.assert_allclose(row.path_coverage,cov);assert row.paths==len(groups)
            continue
        if row.group_type=="horizon":g=g.loc[g.horizon==int(row['group'])]
        elif row['group']!="all":g=g.loc[g.origin_id==row['group']]
        below=g.actual<g.lower;above=g.actual>g.upper
        score=(g.upper-g.lower)+10*(g.lower-g.actual)*below+10*(g.actual-g.upper)*above
        np.testing.assert_allclose([row.coverage,row.mean_width,row.interval_score80],[np.mean(~(below|above)),np.mean(g.upper-g.lower),score.mean()],rtol=1e-12)
    for _,row in pairs.iterrows():
        sub=p.loc[p.stage==row.stage]
        if row.population!="all":sub=sub.loc[~sub.date.str.startswith("2025-11-")]
        if row.origin_id!="all":sub=sub.loc[sub.origin_id==row.origin_id]
        am=scores(sub.loc[sub.id==row.id])[0];bm=scores(sub.loc[sub.id==row.baseline])[0]
        np.testing.assert_allclose([row.mae_difference,row.relative_mae_improvement_percent],[am-bm,100*(1-am/bm)],rtol=1e-11,atol=1e-11)
    # The clean-environment raw control is compared with the original cached control on shared dates.
    legacy=pd.read_csv(e.ROOT/"phase1/results/development_forecasts.csv")
    legacy=legacy.loc[legacy.procedure_id=="lightgbm__expanding__68a53d80"] if "lightgbm__expanding__68a53d80" in set(legacy.procedure_id) else legacy.loc[legacy.procedure_id==pd.read_csv(e.ROOT/"phase1/results/selection_summary.csv").loc[lambda z:(z.family=="lightgbm")&(z.window=="expanding"),"procedure_id"].iloc[0]]
    control=p.loc[(p.id=="lgbm_raw")&(p.stage!="retrospective")].merge(legacy[["date","prediction"]],on="date",suffixes=("_phase4","_phase1"))
    max_control=float(np.max(np.abs(control.prediction_phase4-control.prediction_phase1)))
    e.protect()
    return {"status":"passed","protected_files":539,"records_verified":288,"daily_predictions_verified":8064,"saved_parameter_replays":len(replays),
        "maximum_replay_difference":max(replays,default=0),"matched_ml_pairs_verified":len(mlpairs),"raw_lgbm_original_control_shared_days":len(control),
        "maximum_raw_lgbm_control_difference":max_control,"metric_population_rows":len(summ),"selection_paths":7,"calibration_days":112,
        "fallbacks":sum(a["fallback"] for a in audits),"recorded_python_warnings":sum(len(a["warnings"]) for a in audits),
        "fresh_future_paths_evaluated":0,"new_optimizer_fits_in_verifier":0}


if __name__=="__main__":
    report=verify();e.dump(e.RESULTS/"verification.json",report);print(json.dumps(report,indent=2))
