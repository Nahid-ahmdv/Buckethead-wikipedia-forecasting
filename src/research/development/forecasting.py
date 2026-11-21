"""Development-only fixed-origin forecasting. No final outcomes are loaded here."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import traceback
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FINAL_START = pd.Timestamp("2025-11-02")
EPOCH = pd.Timestamp("2015-01-01")
HORIZON = 28


class ModelFailure(RuntimeError):
    pass


def phase0_integrity():
    frozen = json.loads((ROOT / "protocol/freeze_manifest.json").read_text())
    for relative, expected in frozen["files"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Phase 0 artifact changed: {relative}")
    return len(frozen["files"])


def load_development(path=None):
    expected = ROOT / "data/processed/development_daily.csv"
    path = expected if path is None else Path(path)
    if path.resolve() != expected.resolve():
        raise ValueError("Phase 1 accepts only the frozen development CSV.")
    frame = pd.read_csv(path, parse_dates=["date"])
    if (frame.date >= FINAL_START).any() or (frame.partition == "final").any():
        raise ValueError("Final-period rows are forbidden in Phase 1.")
    y = frame.set_index("date")["count"].astype(float).asfreq("D")
    if not y.index.is_unique or y.isna().any() or (y < 0).any():
        raise ValueError("Invalid development calendar or values.")
    return y.rename("count")


@dataclass
class Origin:
    issue: pd.Timestamp
    cutoff: pd.Timestamp
    targets: pd.DatetimeIndex
    bridge: pd.DatetimeIndex
    history: pd.Series


def at_origin(y, issue):
    issue = pd.Timestamp(issue).normalize()
    targets = pd.date_range(issue, periods=HORIZON, freq="D")
    if targets[-1] >= FINAL_START:
        raise ValueError("Final forecast origins/targets are disabled in Phase 1.")
    cutoff = issue - pd.Timedelta(days=2)
    history = y.loc[y.index <= cutoff].copy().asfreq("D")
    if len(history) < 56 or history.index[-1] != cutoff or history.isna().any():
        raise ValueError("Insufficient or incomplete available history.")
    bridge = pd.date_range(cutoff + pd.Timedelta(days=1), targets[-1], freq="D")
    assert len(bridge) == 29 and bridge[1:].equals(targets)
    return Origin(issue, cutoff, targets, bridge, history)


def training_history(origin, window):
    if window == "expanding":
        return origin.history.copy()
    if window == "three_years":
        return origin.history.loc[origin.history.index > origin.cutoff - pd.DateOffset(years=3)].copy()
    raise ValueError(window)


def mase_scale(origin):
    reference = training_history(origin, "three_years")
    changes = reference.diff(7).dropna()
    value = float(changes.abs().mean())
    return value if np.isfinite(value) and value > 0 else np.nan, {
        "scale_start": str(reference.index[0].date()), "scale_end": str(reference.index[-1].date()),
        "scale_pairs": len(changes),
    }


def annual_fourier(index, k=2):
    t = (pd.DatetimeIndex(index) - EPOCH).days.to_numpy() / 365.25
    return pd.DataFrame({f"{kind}_{j}": fn(2*np.pi*j*t) for j in range(1,k+1) for kind,fn in (("sin",np.sin),("cos",np.cos))}, index=index)


def weekly_naive(origin):
    last = origin.history.iloc[-7:]
    lookup = {day.weekday(): value for day,value in last.items()}
    return np.array([lookup[day.weekday()] for day in origin.targets], dtype=float)


def weekday_mean(origin, weeks):
    recent = origin.history.iloc[-7*weeks:]
    means = recent.groupby(recent.index.dayofweek).mean()
    return np.array([means.loc[day.weekday()] for day in origin.targets], dtype=float)


def feature_path(history, issue):
    """Only origin-relative target lags; no target-relative observed lags."""
    issue = pd.Timestamp(issue)
    cutoff = issue - pd.Timedelta(days=2)
    past = history.loc[:cutoff]
    if len(past) < 56 or past.index[-1] != cutoff:
        raise ValueError("Feature lookback unavailable.")
    targets = pd.date_range(issue, periods=28, freq="D")
    values = {f"count_c_minus_{k}": float(past.loc[cutoff-pd.Timedelta(days=k)]) for k in (0,1,6,7,13,27,55)}
    for length in (7,28):
        values[f"mean_{length}"] = float(past.iloc[-length:].mean())
        values[f"std_{length}"] = float(past.iloc[-length:].std(ddof=0))
    x = pd.DataFrame({name: np.repeat(value,28) for name,value in values.items()}, index=targets)
    x["horizon"] = np.arange(1,29)
    x["target_weekday"] = targets.dayofweek.to_numpy()
    return x.join(annual_fourier(targets))


def supervised_paths(origin, window):
    full = origin.history
    left = full.index[0] if window == "expanding" else origin.cutoff - pd.DateOffset(years=3) + pd.Timedelta(days=1)
    first = max(left, full.index[0]+pd.Timedelta(days=57))
    first += pd.Timedelta(days=(origin.issue.weekday()-first.weekday()) % 7)
    # A full target path must be observed before the current issue's cutoff.
    last = origin.cutoff - pd.Timedelta(days=27)
    issues = pd.date_range(first, last, freq="7D")
    xs, ys, meta = [], [], []
    for issue in issues:
        target = pd.date_range(issue, periods=28, freq="D")
        x = feature_path(full, issue)
        xs.append(x.reset_index(drop=True))
        ys.extend(full.loc[target].to_numpy())
        meta.extend({"training_issue": issue, "feature_cutoff": issue-pd.Timedelta(days=2), "target_date": day} for day in target)
    if not xs:
        raise ValueError("No mature training paths.")
    return pd.concat(xs,ignore_index=True), np.array(ys), pd.DataFrame(meta)


def fit_sarimax(origin, window, params):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    train = training_history(origin,window)
    model = SARIMAX(train, exog=annual_fourier(train.index), order=tuple(params["order"]),
                    seasonal_order=(0,1,1,7), enforce_stationarity=True,
                    enforce_invertibility=True, trend=None)
    result = model.fit(method="lbfgs", maxiter=200, disp=False)
    info = {"converged": bool(result.mle_retvals.get("converged",False)),
            "aic": float(result.aic), "llf": float(result.llf),
            "optimizer": "lbfgs", "optimizer_result": result.mle_retvals,
            "fitted_parameters": dict(zip(result.param_names,result.params.tolist())),
            "train_start": str(train.index[0].date()), "train_end": str(train.index[-1].date()), "train_rows": len(train)}
    if not info["converged"] or not np.isfinite(result.params).all() or not np.isfinite(result.llf):
        raise ModelFailure("SARIMAX nonconverged/nonfinite: " + json.dumps(info,default=str))
    p = result.get_forecast(steps=29, exog=annual_fourier(origin.bridge)).predicted_mean
    assert p.index.equals(origin.bridge)
    return p.iloc[1:].to_numpy(), info


def fit_prophet(origin, window, params):
    from prophet import Prophet
    train = training_history(origin,window)
    kwargs = dict(growth="linear",weekly_seasonality=3,yearly_seasonality=3,daily_seasonality=False,
                  seasonality_mode="additive",changepoint_prior_scale=params["changepoint_prior_scale"],
                  uncertainty_samples=0)
    model = Prophet(**kwargs)
    model.fit(pd.DataFrame({"ds":train.index,"y":train.to_numpy()}),seed=42)
    converged = bool(model.stan_fit.converged)
    if not converged:
        raise ModelFailure("Prophet optimizer did not converge.")
    predicted = model.predict(pd.DataFrame({"ds":origin.bridge}))
    assert pd.DatetimeIndex(predicted.ds).equals(origin.bridge)
    state = {key:np.asarray(value).tolist() for key,value in model.params.items()}
    return predicted.yhat.to_numpy()[1:], {
        "converged":converged,"configuration":kwargs,"fit_seed":42,"fitted_parameters":state,
        "train_start":str(train.index[0].date()),"train_end":str(train.index[-1].date()),"train_rows":len(train),
    }


def fit_lightgbm(origin, window, params):
    from lightgbm import LGBMRegressor
    x,y,meta = supervised_paths(origin,window)
    kwargs = dict(objective="regression_l1",num_leaves=params["num_leaves"],n_estimators=300,
                  learning_rate=.05,min_child_samples=50,random_state=42,n_jobs=1,
                  deterministic=True,force_col_wise=True,verbosity=-1)
    model = LGBMRegressor(**kwargs).fit(x,y)
    future = feature_path(origin.history,origin.issue)
    predicted = model.predict(future)
    return predicted, {
        "converged":True,"configuration":model.get_params(),"trained_iterations":model.booster_.current_iteration(),
        "train_rows":len(x),"training_paths":len(meta)//28,
        "train_origin_start":str(meta.training_issue.min().date()),"train_origin_end":str(meta.training_issue.max().date()),
        "label_end":str(meta.target_date.max().date()),"last_feature_cutoff":str(meta.feature_cutoff.max().date()),
        "feature_names":list(x.columns),"feature_gain":model.booster_.feature_importance(importance_type="gain").tolist(),
        "fitted_state_sha256":hashlib.sha256(model.booster_.model_to_string().encode()).hexdigest(),
    }


def predict(y, issue, spec):
    """All model engines receive this origin's available history only."""
    origin = at_origin(y,issue)
    scale,scale_meta = mase_scale(origin)
    info = {"issue":str(origin.issue.date()),"cutoff":str(origin.cutoff.date()),
            "target_start":str(origin.targets[0].date()),"target_end":str(origin.targets[-1].date()),
            "bridge_start":str(origin.bridge[0].date()),"bridge_end":str(origin.bridge[-1].date()),
            "bridge_days":29,"history_rows":len(origin.history),"mase_scale":scale,**scale_meta,
            "family":spec["family"],"window":spec["window"],"params":spec["params"],"fallback":False}
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            if spec["family"] == "weekly_naive":
                p,details = weekly_naive(origin), {"converged":True}
            elif spec["family"] == "weekday_mean":
                p,details = weekday_mean(origin,spec["params"]["weeks"]), {"converged":True}
            else:
                engines = {"sarimax":fit_sarimax,"prophet":fit_prophet,"lightgbm":fit_lightgbm}
                p,details = engines[spec["family"]](origin,spec["window"],spec["params"])
            p = np.asarray(p,dtype=float)
            if p.shape != (28,) or not np.isfinite(p).all():
                raise ModelFailure("Invalid point forecast shape/values.")
            info.update(details)
        except (ModelFailure, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            # Unexpected programming errors such as NameError/TypeError propagate.
            info.update({"fallback":True,"converged":False,"error_type":type(error).__name__,
                         "error":str(error),"traceback":traceback.format_exc()})
            p = weekly_naive(origin)
        info["warnings"] = [{"category":w.category.__name__,"message":str(w.message)} for w in captured]
    info["runtime_seconds"] = time.perf_counter()-started
    return pd.Series(np.maximum(p,0),index=origin.targets,name="prediction"),info


def metrics(actual, prediction, scale):
    if not actual.index.equals(prediction.index) or len(actual) != 28:
        raise ValueError("Scoring requires identical 28-date fixed-origin paths.")
    if actual.isna().any() or not np.isfinite(prediction).all():
        raise ValueError("Missing/nonfinite score input.")
    error = actual.to_numpy()-prediction.to_numpy()
    return {"mae":float(np.abs(error).mean()),"rmse":float(np.sqrt(np.square(error).mean())),
            "mase":float(np.abs(error).mean()/scale) if np.isfinite(scale) and scale>0 else np.nan}


def candidates():
    specs = [{"family":"weekly_naive","window":"fixed","params":{}}]
    specs += [{"family":"weekday_mean","window":"fixed","params":{"weeks":w}} for w in (4,8)]
    for window in ("three_years","expanding"):
        specs += [{"family":"sarimax","window":window,"params":{"order":list(order)}} for order in ((1,0,0),(1,1,0),(1,0,1),(1,1,1))]
        specs += [{"family":"prophet","window":window,"params":{"changepoint_prior_scale":v}} for v in (.01,.1)]
        specs += [{"family":"lightgbm","window":window,"params":{"num_leaves":n}} for n in (7,15)]
    return specs


def spec_id(spec):
    return spec["family"]+"__"+spec["window"]+"__"+hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:8]
