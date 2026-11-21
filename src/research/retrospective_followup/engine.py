"""Six frozen Phase 4 procedures; all transformations are origin-available."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[key]="1"
os.environ.setdefault("MPLCONFIGDIR","/private/tmp/buckethead-phase4-mpl")
from pathlib import Path
from dataclasses import dataclass
import sys,json,hashlib,time,traceback,warnings
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RESULTS=HERE/"results"
sys.path.insert(0,str(ROOT/"phase1/src"))
import forecasting as original


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,(Path,pd.Timestamp)):return str(value)
    raise TypeError(type(value).__name__)
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=serial,allow_nan=True)+"\n")
def protect():
    snapshot=json.loads((HERE/"protected_snapshot.json").read_text())["files"]
    for p,expected in snapshot.items():assert digest(ROOT/p)==expected,p
    freeze=json.loads((HERE/"protocol_freeze.json").read_text())
    for p,expected in freeze["files"].items():assert digest(HERE/p)==expected,p
    amendment=json.loads((HERE/"protocol_amendment_01.json").read_text())
    for p,expected in amendment["files"].items():assert digest(HERE/p)==expected,p
    return len(snapshot)
def roster():return json.loads((HERE/"candidates.json").read_text())
def schedule():return pd.read_csv(HERE/"schedule_v2.csv")
def load_y():
    protect()
    frame=pd.read_csv(ROOT/"data/processed/buckethead_daily.csv",parse_dates=["date"])
    y=frame.set_index("date")["count"].astype(float)
    assert y.index.equals(pd.date_range("2015-07-01","2026-10-31")) and np.isfinite(y).all() and (y>=0).all()
    return y.rename("count")


def at_origin(y,issue,scheduled=True):
    issue=pd.Timestamp(issue)
    if issue!=issue.normalize() or issue.tzinfo is not None:raise ValueError("Naive UTC midnight date required")
    if scheduled and str(issue.date()) not in set(schedule().issue):raise ValueError("Issue not in frozen Phase 4 schedule")
    cutoff=issue-pd.Timedelta(days=2)
    history=y.loc[y.index<=cutoff].copy().asfreq("D")
    if len(history)<56 or history.index[-1]!=cutoff or not np.isfinite(history).all():raise ValueError("Incomplete origin history")
    target=pd.date_range(issue,periods=28)
    bridge=pd.date_range(cutoff+pd.Timedelta(days=1),target[-1])
    assert len(bridge)==29 and bridge[1:].equals(target)
    return original.Origin(issue,cutoff,target,bridge,history)


def weekday_values(history,issue):
    cutoff=pd.Timestamp(issue)-pd.Timedelta(days=2)
    past=history.loc[:cutoff].asfreq("D").iloc[-56:]
    if len(past)!=56 or past.index[-1]!=cutoff or not np.isfinite(past).all():raise ValueError("Unavailable weekday baseline")
    means=past.groupby(past.index.weekday).mean()
    targets=pd.date_range(issue,periods=28)
    return np.array([means.loc[d.weekday()] for d in targets],dtype=float)


def affine(history):
    m=float(history.mean());s=float(history.std(ddof=0))
    if s==0:s=1.
    return (history-m)/s,m,s


def training_table(origin):
    x,raw,meta=original.supervised_paths(origin,"expanding")
    offsets=np.empty(len(meta))
    for issue,positions in meta.groupby("training_issue",sort=False).indices.items():
        offsets[positions]=weekday_values(origin.history,issue)
    assert meta.target_date.max()<=origin.cutoff
    return x,raw,offsets,meta


def array_hash(a):return hashlib.sha256(np.ascontiguousarray(np.asarray(a,dtype=np.float64)).tobytes()).hexdigest()


def fit_ets(origin,spec):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    train=original.training_history(origin,"three_years")
    scale=float(train.mean())
    if not np.isfinite(scale) or scale<=0:raise original.ModelFailure("ETS nonpositive training scale")
    model=ETSModel(train/scale,error="add",trend="add" if spec["params"]["trend"] else None,
        damped_trend=spec["params"]["damped"],seasonal="add",seasonal_periods=7,initialization_method="estimated")
    result=model.fit(maxiter=1000,disp=False)
    info={"converged":bool(result.mle_retvals.get("converged",False)),"optimizer_result":result.mle_retvals,
        "likelihood":float(result.llf),"parameters":dict(zip(model.param_names,np.asarray(result.params).tolist())),
        "training_scale":scale,"train_start":str(train.index[0].date()),"train_end":str(train.index[-1].date()),"train_rows":len(train),
        "configuration":{"error":"add","trend":"add" if spec["params"]["trend"] else None,"damped_trend":spec["params"]["damped"],
            "seasonal":"add","seasonal_periods":7,"initialization_method":"estimated","maxiter":1000,"bounds":"library defaults"}}
    if not info["converged"] or not np.isfinite(result.params).all() or not np.isfinite(result.llf):
        raise original.ModelFailure("ETS nonconverged/nonfinite: "+json.dumps(info,default=serial))
    p=result.forecast(29)
    assert p.index.equals(origin.bridge)
    return p.iloc[1:].to_numpy()*scale,info


def fit_sarimax(origin,spec):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    train=original.training_history(origin,"three_years")
    z,m,s=affine(train)
    model=SARIMAX(z,exog=original.annual_fourier(train.index),order=(1,0,1),seasonal_order=(0,spec["params"]["D"],1,7),
        trend=None,enforce_stationarity=True,enforce_invertibility=True)
    result=model.fit(method="lbfgs",maxiter=500,disp=False)
    info={"converged":bool(result.mle_retvals.get("converged",False)),"optimizer_result":result.mle_retvals,
        "likelihood":float(result.llf),"parameters":dict(zip(result.param_names,result.params.tolist())),
        "training_mean":m,"training_std":s,"train_start":str(train.index[0].date()),"train_end":str(train.index[-1].date()),"train_rows":len(train),
        "minimum_ar_root_modulus":float(np.min(np.abs(result.arroots))),"minimum_ma_root_modulus":float(np.min(np.abs(result.maroots))),
        "configuration":{"order":[1,0,1],"seasonal_order":[0,spec["params"]["D"],1,7],"trend":None,"annual_K":2,
            "annual_period":365.25,"annual_epoch":"2015-01-01","stationarity":True,"invertibility":True,"optimizer":"lbfgs","maxiter":500}}
    if not info["converged"] or not np.isfinite(result.params).all() or not np.isfinite(result.llf):
        raise original.ModelFailure("SARIMAX nonconverged/nonfinite: "+json.dumps(info,default=serial))
    p=result.get_forecast(29,exog=original.annual_fourier(origin.bridge)).predicted_mean
    assert p.index.equals(origin.bridge)
    return m+s*p.iloc[1:].to_numpy(),info


def fit_lgbm(origin,spec,table=None):
    from lightgbm import LGBMRegressor
    x,raw,offsets,meta=training_table(origin) if table is None else table
    labels=raw-offsets if spec["params"]["correction"] else raw
    kwargs=dict(objective="regression_l1",num_leaves=7,n_estimators=300,learning_rate=.05,min_child_samples=50,
        random_state=42,n_jobs=1,deterministic=True,force_col_wise=True,verbosity=-1)
    model=LGBMRegressor(**kwargs).fit(x,labels)
    p=model.predict(original.feature_path(origin.history,origin.issue))
    if spec["params"]["correction"]:p=p+weekday_values(origin.history,origin.issue)
    return p,{"converged":True,"configuration":model.get_params(),"trained_iterations":model.booster_.current_iteration(),
        "train_rows":len(x),"training_paths":len(meta)//28,"feature_names":list(x.columns),"X_sha256":array_hash(x),
        "raw_labels_sha256":array_hash(raw),"training_baselines_sha256":array_hash(offsets),"fit_labels_sha256":array_hash(labels),
        "label_end":str(meta.target_date.max().date()),"last_training_issue":str(meta.training_issue.max().date()),
        "first_training_issue":str(meta.training_issue.min().date()),"model_sha256":hashlib.sha256(model.booster_.model_to_string().encode()).hexdigest(),
        "feature_gain":model.booster_.feature_importance(importance_type="gain").tolist(),"residual_labels_clipped":False}


def predict(y,issue,spec,table=None,scheduled=True):
    if spec not in roster():raise ValueError("Unfrozen procedure")
    origin=at_origin(y,issue,scheduled=scheduled)
    scale,scale_meta=original.mase_scale(origin)
    info={"id":spec["id"],"issue":str(origin.issue.date()),"cutoff":str(origin.cutoff.date()),
        "target_end":str(origin.targets[-1].date()),"bridge_start":str(origin.bridge[0].date()),"bridge_days":29,
        "mase_scale":scale,**scale_meta,"fallback":False,"window":spec["window"]}
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            if spec["family"]=="ets":p,details=fit_ets(origin,spec)
            elif spec["family"]=="sarimax":p,details=fit_sarimax(origin,spec)
            elif spec["family"]=="lightgbm":p,details=fit_lgbm(origin,spec,table)
            elif spec["id"]=="weekday_mean":p,details=weekday_values(origin.history,origin.issue),{"converged":True}
            else:p,details=original.weekly_naive(origin),{"converged":True}
            p=np.asarray(p,dtype=float)
            if p.shape!=(28,) or not np.isfinite(p).all():raise original.ModelFailure("Invalid forecast shape/values")
            info.update(details)
        except (original.ModelFailure,ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            info.update({"fallback":True,"converged":False,"error_type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc()})
            p=original.weekly_naive(origin)
        info["warnings"]=[{"category":w.category.__name__,"message":str(w.message)} for w in captured]
    info["runtime_seconds"]=time.perf_counter()-started
    return np.maximum(p,0),info


def metrics(g):
    error=g.prediction.to_numpy()-g.actual.to_numpy();q=g.mase_scale.to_numpy()
    valid=np.isfinite(q)&(q>0)
    return {"days":len(g),"mae":float(np.abs(error).mean()),"rmse":float(np.sqrt(np.mean(error**2))),
        "mase":float(np.mean(np.abs(error)/q)) if valid.all() else np.nan,"mase_undefined_days":int((~valid).sum()),
        "signed_error":float(error.mean()),"overprediction_frequency":float(np.mean(error>0))}


PRIORITY=["weekday_mean","weekly_naive","ets_level","ets_damped","sarimax_d0","sarimax_d1","lgbm_raw","lgbm_correction"]
def choose(summary):
    eligible=summary.loc[summary.failures==0]
    best=eligible.loc[eligible.mae<=eligible.mae.min()+1e-9]
    return min(best.id,key=PRIORITY.index)


def calibrate(g,first_issue):
    if len(g)!=112 or not np.isfinite(g.actual-g.prediction).all():raise ValueError("Calibration requires all 112 finite errors")
    if pd.to_datetime(g.date).max()>pd.Timestamp(first_issue)-pd.Timedelta(days=2):raise ValueError("Immature calibration targets")
    residual=np.sort(g.actual.to_numpy()-g.prediction.to_numpy())
    lower_rank=int(np.floor(113*.10));upper_rank=int(np.ceil(113*.90))
    return {"n":112,"lower_rank":lower_rank,"upper_rank":upper_rank,"lower_offset":float(residual[lower_rank-1]),
        "upper_offset":float(residual[upper_rank-1]),"nominal_pointwise_coverage":.8,"updates":False}


def interval(p,cal):
    return np.maximum(0,np.asarray(p)+cal["lower_offset"]),np.maximum(0,np.asarray(p)+cal["upper_offset"])
def interval_metrics(actual,lo,hi):
    actual,lo,hi=map(np.asarray,(actual,lo,hi))
    if (lo>hi).any():raise ValueError("Inverted interval")
    width=hi-lo
    score=width+10*np.maximum(lo-actual,0)+10*np.maximum(actual-hi,0)
    return {"days":len(actual),"coverage":float(np.mean((actual>=lo)&(actual<=hi))),"mean_width":float(np.mean(width)),
        "below":int((actual<lo).sum()),"above":int((actual>hi).sum()),"interval_score80":float(np.mean(score))}
