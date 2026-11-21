"""Authorized final entry point; the frozen Phase 1 engine remains untouched.

Only the thirteen scheduled final issues and three frozen procedures are allowed.
Run as a script AFTER preflight.py. Existing fit records are reused, never replaced.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import sys, json, hashlib, time, warnings, traceback, platform, importlib, importlib.metadata
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "phase1/src"))
import forecasting as f
RESULTS = HERE / "results"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x), allow_nan=False) + "\n")


def original_integrity():
    counts = {}
    for label, folder, manifest in (("phase0", ROOT, ROOT / "protocol/freeze_manifest.json"),
                                     ("phase1", ROOT / "phase1", ROOT / "phase1/freeze_manifest.json")):
        files = json.loads(manifest.read_text())["files"]
        for relative, expected in files.items():
            if digest(folder / relative) != expected:
                raise ValueError(f"Frozen {label} artifact changed: {relative}")
        counts[label] = len(files)
    return counts


def proposal():
    return json.loads((ROOT / "phase1/results/proposed_final_configuration.json").read_text())


def roster():
    p = proposal()
    return [p["selected_specification"], *p["baselines"]]


def schedule():
    s = pd.read_csv(ROOT / "protocol/split_schedule.csv")
    return s.loc[s["split"] == "final"].reset_index(drop=True)


def origin_at(y, issue):
    # Explicit final authorization replaces neither the Phase 1 loader nor its guard.
    issue = pd.Timestamp(issue)
    if issue.tzinfo is not None or issue != issue.normalize():
        raise ValueError("Use a naive UTC daily issue date at midnight.")
    if str(issue.date()) not in set(schedule().target_start):
        raise ValueError("Issue is outside the thirteen authorized final paths.")
    cutoff = issue - pd.Timedelta(days=2)
    targets = pd.date_range(issue, periods=28, freq="D")
    history = y.loc[y.index <= cutoff].copy().asfreq("D")
    if not history.index.is_unique or len(history) < 56 or history.index[-1] != cutoff:
        raise ValueError("Incomplete available history.")
    if not np.isfinite(history).all() or (history < 0).any():
        raise ValueError("Invalid history values.")
    bridge = pd.date_range(cutoff + pd.Timedelta(days=1), targets[-1], freq="D")
    assert len(bridge) == 29 and bridge[1:].equals(targets)
    return f.Origin(issue, cutoff, targets, bridge, history)


def predict(y, issue, spec):
    if spec not in roster():
        raise ValueError("Procedure is outside the frozen final roster.")
    origin = origin_at(y, issue)
    scale, scale_meta = f.mase_scale(origin)
    info = {"issue": str(origin.issue.date()), "cutoff": str(origin.cutoff.date()),
            "target_start": str(origin.targets[0].date()), "target_end": str(origin.targets[-1].date()),
            "bridge_start": str(origin.bridge[0].date()), "bridge_end": str(origin.bridge[-1].date()),
            "bridge_days": 29, "history_rows": len(origin.history), "mase_scale": scale,
            **scale_meta, **spec, "fallback": False}
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            if spec["family"] == "weekly_naive":
                p, details = f.weekly_naive(origin), {"converged": True}
            elif spec["family"] == "weekday_mean":
                p, details = f.weekday_mean(origin, spec["params"]["weeks"]), {"converged": True}
            else:
                p, details = f.fit_sarimax(origin, spec["window"], spec["params"])
            p = np.asarray(p, dtype=float)
            if p.shape != (28,) or not np.isfinite(p).all():
                raise f.ModelFailure("Invalid point forecast shape/values.")
            info.update(details)
        except (f.ModelFailure, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            info.update({"fallback": True, "converged": False, "error_type": type(error).__name__,
                         "error": str(error), "traceback": traceback.format_exc()})
            p = f.weekly_naive(origin)
        info["warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in captured]
    info["runtime_seconds"] = time.perf_counter() - started
    # No new numerical cutoff or optimizer retry is introduced in Phase 2.
    if "fitted_parameters" in info:
        a = info["fitted_parameters"]
        info["minimum_ma_root_modulus"] = float(min(1 / abs(a["ma.L1"]), (1 / abs(a["ma.S.L7"])) ** (1 / 7)))
        info["minimum_ar_root_modulus"] = float(1 / abs(a["ar.L1"]))
    return pd.Series(np.maximum(p, 0), index=origin.targets, name="prediction"), info


def interval_bounds(p, q):
    p = np.maximum(np.asarray(p, dtype=float), 0)
    return np.maximum(p - q, 0), p + q


def summarize(frame):
    """Daily weighting also for partial sensitivity paths; pooled RMSE."""
    error = frame.actual.to_numpy() - frame.prediction.to_numpy()
    scale = frame.mase_scale.to_numpy()
    scaled = np.divide(np.abs(error), scale, out=np.full(len(error), np.nan), where=np.isfinite(scale) & (scale > 0))
    return {"days": len(frame), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mase": float(np.mean(scaled)), "mase_undefined_days": int(np.isnan(scaled).sum())}


def populations(frame):
    dates = pd.to_datetime(frame.date)
    return {"primary": frame, "exclude_november_2025": frame.loc[~((dates.dt.year == 2025) & (dates.dt.month == 11))]}


def approved_contract():
    contract = json.loads((HERE / "execution_contract.json").read_text())
    assert contract["preflight_passed"] is True
    for relative, expected in contract["locked_sources"].items():
        assert digest(ROOT / relative) == expected, relative
    assert digest(RESULTS / "preflight.json") == contract["preflight_sha256"]
    original_integrity()
    return contract


def load_final():
    approved_contract()
    frame = pd.read_csv(ROOT / "data/processed/buckethead_daily.csv", parse_dates=["date"])
    y = frame.set_index("date")["count"].astype(float).asfreq("D")
    assert len(y) == 4141 and y.index.is_unique and np.isfinite(y).all() and (y >= 0).all()
    assert y.index[0] == pd.Timestamp("2015-07-01") and y.index[-1] == pd.Timestamp("2026-10-31")
    return y


def main():
    contract = approved_contract()
    token = digest(HERE / "execution_contract.json")
    y = load_final()
    rows, audits = [], []
    q = proposal()["interval"]["half_width"]
    RESULTS.mkdir(exist_ok=True)
    for _, s in schedule().iterrows():
        for spec in roster():
            identifier = f.spec_id(spec)
            path = RESULTS / "cache" / f"{s.origin_id}__{identifier}.json"
            if path.exists():
                record = json.loads(path.read_text())
                assert record["contract_sha256"] == token and record["spec"] == spec
                pred = pd.Series(record["prediction"], index=pd.to_datetime(record["dates"]))
                info = record["info"]
            else:
                pred, info = predict(y, s.target_start, spec)
                record = {"contract_sha256": token, "spec": spec, "dates": [str(d.date()) for d in pred.index],
                          "prediction": pred.tolist(), "info": info}
                write_json(path, record)
            assert info["cutoff"] == s.information_cutoff_date and info["target_end"] == s.target_end
            lo, hi = interval_bounds(pred, q)
            for h, (day, point) in enumerate(pred.items(), 1):
                row = {"origin_id": s.origin_id, "issue": s.target_start, "cutoff": info["cutoff"],
                       "procedure_id": identifier, "family": spec["family"], "date": str(day.date()),
                       "horizon": h, "actual": float(y.loc[day]), "prediction": point,
                       "mase_scale": info["mase_scale"], "fallback": info["fallback"]}
                if spec["family"] == "sarimax":
                    row.update({"lower": lo[h-1], "upper": hi[h-1], "interval_half_width": q})
                rows.append(row)
            audits.append({"origin_id": s.origin_id, "procedure_id": identifier, **info})
        print(f"Completed {s.origin_id}: {s.target_start} through {s.target_end}", flush=True)
    frame = pd.DataFrame(rows)
    frame["absolute_error"] = (frame.actual - frame.prediction).abs()
    frame["squared_error"] = (frame.actual - frame.prediction) ** 2
    frame.to_csv(RESULTS / "final_forecasts.csv", index=False)
    write_json(RESULTS / "fit_audits.json", audits)
    aggregate, by_origin, intervals, pairs = [], [], [], []
    selected_id = proposal()["selected_procedure_id"]
    for population, sub in populations(frame).items():
        for (identifier, family), g in sub.groupby(["procedure_id", "family"]):
            aggregate.append({"population": population, "procedure_id": identifier, "family": family,
                              "paths": g.origin_id.nunique(), **summarize(g)})
            for oid, path in g.groupby("origin_id"):
                by_origin.append({"population": population, "procedure_id": identifier, "family": family,
                                  "origin_id": oid, "issue": path.issue.iloc[0], **summarize(path)})
        chosen = sub.loc[sub.procedure_id == selected_id]
        for oid, g in [("all", chosen), *list(chosen.groupby("origin_id"))]:
            intervals.append({"population": population, "origin_id": oid, "days": len(g),
                              "nominal_coverage": .8, "coverage": float(((g.actual >= g.lower) & (g.actual <= g.upper)).mean()),
                              "mean_width": float((g.upper - g.lower).mean()),
                              "below": int((g.actual < g.lower).sum()), "above": int((g.actual > g.upper).sum())})
        for base in roster()[1:]:
            b = sub.loc[sub.procedure_id == f.spec_id(base)]
            for oid, g in [("all", chosen), *list(chosen.groupby("origin_id"))]:
                ref = b.loc[b.origin_id.isin(g.origin_id.unique())]
                assert len(g) == len(ref)
                selected_mae, baseline_mae = summarize(g)["mae"], summarize(ref)["mae"]
                pairs.append({"population": population, "origin_id": oid, "baseline": base["family"],
                              "days": len(g), "selected_mae": selected_mae, "baseline_mae": baseline_mae,
                              "selected_minus_baseline_mae": selected_mae - baseline_mae,
                              "relative_mae_improvement_percent": 100 * (1 - selected_mae / baseline_mae)})
    for name, data in (("summary", aggregate), ("origin_metrics", by_origin), ("interval_metrics", intervals), ("paired_differences", pairs)):
        pd.DataFrame(data).to_csv(RESULTS / f"{name}.csv", index=False)
    modules = {}
    for name in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "nbformat", "nbclient"):
        module = importlib.import_module(name)
        modules[name] = {"imported_version": module.__version__, "distribution_version": importlib.metadata.version(name), "path": module.__file__}
    write_json(RESULTS / "environment.json", {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "modules": modules, "threads": {key: os.environ[key] for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "contract_sha256": token, "raw_series_sha256": digest(ROOT / "data/processed/buckethead_daily.csv"),
        "phase1_environment": "phase1/results/environment.json"})
    print(pd.DataFrame(aggregate).to_string(index=False))


if __name__ == "__main__":
    main()
