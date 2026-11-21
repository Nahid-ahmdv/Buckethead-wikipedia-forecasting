#!/usr/bin/env python3
"""Execute the frozen compact grid on development only, with resumable caches."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[key]="1"
os.environ.setdefault("MPLCONFIGDIR","/private/tmp/buckethead-phase1-mpl")

from pathlib import Path
import sys
import hashlib
import json
import logging
import math
import platform
import datetime as dt
import importlib
import importlib.metadata

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"src"))
import forecasting as f

RESULTS=HERE/"results"
CACHE=RESULTS/"cache"
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,(Path,pd.Timestamp)):return str(value)
    raise TypeError(type(value).__name__)


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,default=serial,allow_nan=True)+"\n")


def signature():
    paths=[HERE/"src/forecasting.py",Path(__file__),HERE/"PROTOCOL_IMPLEMENTATION.md",f.ROOT/"data/processed/development_daily.csv"]
    digests={str(p.relative_to(f.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return hashlib.sha256(json.dumps(digests,sort_keys=True).encode()).hexdigest(),digests


def evaluate(y,schedule,spec,stage,implementation,records,forecasts,progress):
    identifier=f.spec_id(spec)
    for _,row in schedule.iterrows():
        assert row["split"] != "final"
        issue=pd.Timestamp(row.target_start)
        key=hashlib.sha256(json.dumps({"implementation":implementation,"issue":str(issue),"spec":spec},sort_keys=True).encode()).hexdigest()
        cache=CACHE/(key+".json")
        if cache.exists():
            saved=json.loads(cache.read_text())
            prediction=pd.Series(saved["prediction"],index=pd.to_datetime(saved["target_dates"]),name="prediction")
            info=saved["info"]
            cache_hit=True
        else:
            prediction,info=f.predict(y,issue,spec)
            write_json(cache,{"implementation_sha256":implementation,"spec":spec,"target_dates":[str(t.date()) for t in prediction.index],"prediction":prediction.tolist(),"info":info})
            cache_hit=False
        expected=pd.date_range(issue,periods=28,freq="D")
        assert prediction.index.equals(expected) and expected[-1]<f.FINAL_START
        actual=y.loc[expected]
        scores=f.metrics(actual,prediction,info["mase_scale"])
        records.append({"stage":stage,"origin_id":row.origin_id,"procedure_id":identifier,**spec,
                        **scores,"fallback":info["fallback"],"runtime_seconds":info["runtime_seconds"],"cache_key":key,"cache_hit":cache_hit,
                        "mase_scale":info["mase_scale"],"cutoff":info["cutoff"],"target_start":str(issue.date())})
        for h,(date,a,p) in enumerate(zip(expected,actual,prediction),1):
            forecasts.append({"stage":stage,"origin_id":row.origin_id,"procedure_id":identifier,"date":str(date.date()),"horizon":h,"actual":float(a),"prediction":float(p),"absolute_error":float(abs(a-p)),"squared_error":float((a-p)**2),"mase_scale":info["mase_scale"],"fallback":info["fallback"]})
        progress["completed"]+=1
        progress.update({"stage":stage,"procedure":identifier,"origin":row.origin_id,"last_fallback":info["fallback"]})
        write_json(RESULTS/"progress.json",progress)
        print(f'{progress["completed"]:03d} {stage} {identifier} {row.origin_id} MAE={scores["mae"]:.2f} fallback={info["fallback"]}',flush=True)


def summary(records,forecasts,stage):
    r=pd.DataFrame(records); r=r.loc[r.stage==stage]
    paths=pd.DataFrame(forecasts); paths=paths.loc[paths.stage==stage]
    out=[]
    for identifier,g in r.groupby("procedure_id"):
        errors=paths.loc[paths.procedure_id==identifier]
        out.append({"procedure_id":identifier,"family":g.family.iloc[0],"window":g.window.iloc[0],
                    "params":g.params.iloc[0],"mae":g.mae.mean(),"rmse":np.sqrt(errors.squared_error.mean()),
                    "mase":np.mean(errors.absolute_error/errors.mase_scale),"path_mae_std":g.mae.std(ddof=1),
                    "paths":len(g),"failures":int(g.fallback.sum()),"runtime_seconds":g.runtime_seconds.sum()})
    return pd.DataFrame(out).sort_values(["mae","procedure_id"])


def main():
    RESULTS.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    frozen_files=f.phase0_integrity()
    y=f.load_development()
    schedule=pd.read_csv(f.ROOT/"protocol/split_schedule.csv")
    tuning=schedule.loc[schedule.use_for_hyperparameter_tuning==1]
    selection=schedule.loc[schedule.use_for_procedure_selection==1]
    assert len(tuning)==len(selection)==12
    implementation,digests=signature()
    modules={}
    for name in ("numpy","pandas","scipy","statsmodels","prophet","lightgbm","matplotlib","nbformat","nbclient","sklearn","cmdstanpy"):
        module=importlib.import_module(name)
        distribution="scikit-learn" if name=="sklearn" else name
        modules[name]={"imported_version":getattr(module,"__version__",None),"distribution_version":importlib.metadata.version(distribution),"path":getattr(module,"__file__",None)}
    write_json(RESULTS/"environment.json",{"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"modules":modules,"threads":{key:os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS")},"implementation_sha256":implementation,"inputs_and_sources":digests,"phase0_frozen_files_verified":frozen_files})
    specs=f.candidates()
    write_json(RESULTS/"candidate_registry.json",{"implementation_sha256":implementation,"specifications":specs})
    records=[];forecasts=[];progress={"completed":0,"planned_tuning_fits":len(specs)*12,"planned_selection_fits":96,"implementation_sha256":implementation}
    for spec in specs:
        evaluate(y,tuning,spec,"tuning",implementation,records,forecasts,progress)
    tune_summary=summary(records,forecasts,"tuning")
    tune_summary.to_csv(RESULTS/"tuning_summary.csv",index=False)
    tuned=[]
    for (family,window),group in tune_summary.groupby(["family","window"]):
        best=group.loc[group.mae<=group.mae.min()+1e-9].copy()
        best["sort_params"]=best.params.map(lambda p:json.dumps(p,sort_keys=True))
        row=best.sort_values("sort_params").iloc[0]
        tuned.append({"family":family,"window":window,"params":row.params})
    assert len(tuned)==8
    write_json(RESULTS/"tuned_configurations.json",tuned)
    for spec in tuned:
        evaluate(y,selection,spec,"selection",implementation,records,forecasts,progress)
    select_summary=summary(records,forecasts,"selection")
    select_summary["eligible"]=select_summary.failures==0
    select_summary.to_csv(RESULTS/"selection_summary.csv",index=False)
    eligible=select_summary.loc[select_summary.eligible].copy()
    eligible=eligible.loc[eligible.mae<=eligible.mae.min()+1e-9].copy()
    ranks={name:i for i,name in enumerate(("weekly_naive","weekday_mean","sarimax","prophet","lightgbm"))}
    eligible["family_rank"]=eligible.family.map(ranks)
    eligible["window_rank"]=eligible.window.map({"three_years":0,"expanding":1,"fixed":0})
    eligible["param_order"]=eligible.params.map(lambda p:json.dumps(p,sort_keys=True))
    winner=eligible.sort_values(["family_rank","window_rank","param_order"]).iloc[0]
    chosen={"family":winner.family,"window":winner.window,"params":winner.params}
    allpaths=pd.DataFrame(forecasts); allrecords=pd.DataFrame(records)
    allpaths.to_csv(RESULTS/"development_forecasts.csv",index=False)
    allrecords.assign(params=allrecords.params.map(lambda p:json.dumps(p,sort_keys=True))).to_csv(RESULTS/"origin_metrics.csv",index=False)
    calibration=allpaths.loc[(allpaths.stage=="selection")&(allpaths.procedure_id==winner.procedure_id)].copy()
    assert len(calibration)==336 and calibration.date.max()=="2025-10-04"
    rank=math.ceil((len(calibration)+1)*.8)
    width=float(np.sort(calibration.absolute_error.to_numpy())[rank-1])
    calibration.to_csv(RESULTS/"calibration_errors.csv",index=False)
    frozen={"status":"proposed final configuration; final evaluation not executed",
            "selected_procedure_id":winner.procedure_id,"selected_specification":chosen,
            "selection_mae":winner.mae,"selection_rmse":winner.rmse,"selection_mase":winner.mase,
            "selection_paths":12,"selection_fallbacks":int(winner.failures),
            "baselines":[s for s in tuned if s["family"] in ("weekly_naive","weekday_mean")],
            "interval":{"nominal_coverage":.8,"method":"fixed pooled absolute-error order statistic","n":336,"rank":rank,"half_width":width,"calibration_first":"2024-11-03","calibration_last":"2025-10-04","selection_reuse_disclosed":True},
            "event_features":False,"event_reason":"Phase 0 gate failed; no event-ledger changes",
            "final_schedule":{"start":"2025-11-02","end":"2026-10-31","paths":13,"horizon":28,"observation_delay_days":2,"november_sensitivity_days":335},
            "implementation_sha256":implementation,"forecast_results_sha256":hashlib.sha256((RESULTS/"development_forecasts.csv").read_bytes()).hexdigest(),
            "final_outcomes_loaded":False,"final_forecasts_computed":False,"frozen_at_utc":dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(RESULTS/"proposed_final_configuration.json",frozen)
    progress["status"]="complete; awaiting review before final evaluation"
    write_json(RESULTS/"progress.json",progress)
    print("SELECTED",json.dumps(chosen),"selection MAE",winner.mae,"interval half-width",width,flush=True)


if __name__=="__main__":
    main()
