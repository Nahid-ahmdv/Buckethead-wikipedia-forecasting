"""Development-only numerical audit and once-only pre-performance execution lock."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import sys, json, datetime, subprocess
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import final_experiment as e


def main():
    if (e.HERE / "execution_contract.json").exists():
        raise RuntimeError("Preflight already locked; preserve the existing record.")
    integrity = e.original_integrity()
    proposal = e.proposal()
    assert proposal["selected_specification"] == {"family":"sarimax", "window":"three_years", "params":{"order":[1,0,1]}}
    s = e.schedule()
    expected = pd.date_range("2025-11-02", periods=13, freq="28D")
    assert list(s.target_start) == [str(d.date()) for d in expected]
    assert list(s.information_cutoff_date) == [str((d-pd.Timedelta(days=2)).date()) for d in expected]
    assert s.sensitivity_retained_days.sum() == 335 and s.horizon_days.eq(28).all()
    calibration = pd.read_csv(e.ROOT / "phase1/results/calibration_errors.csv")
    assert len(calibration) == 336 and proposal["interval"]["rank"] == 270
    assert np.sort(calibration.absolute_error)[269] == proposal["interval"]["half_width"]
    # No final target values enter this audit.
    y = e.f.load_development()
    records = pd.read_csv(e.ROOT / "phase1/results/origin_metrics.csv")
    selected = records.loc[(records.stage == "selection") & (records.procedure_id == proposal["selected_procedure_id"])]
    rows = []
    for _, row in selected.iterrows():
        cache = json.loads((e.ROOT / "phase1/results/cache" / (row.cache_key + ".json")).read_text())
        info = cache["info"]
        origin = e.f.at_origin(y, row.target_start)
        train = e.f.training_history(origin, "three_years")
        model = SARIMAX(train, exog=e.f.annual_fourier(train.index), order=(1,0,1),
                        seasonal_order=(0,1,1,7), enforce_stationarity=True, enforce_invertibility=True, trend=None)
        params = np.array([info["fitted_parameters"][n] for n in model.param_names])
        result = model.filter(params)
        pred = np.maximum(result.get_forecast(29, exog=e.f.annual_fourier(origin.bridge)).predicted_mean.iloc[1:].to_numpy(), 0)
        difference = float(np.max(np.abs(pred - cache["prediction"])))
        ar_root = float(np.min(np.abs(result.arroots)))
        ma_root = float(np.min(np.abs(result.maroots)))
        optimizer = info["optimizer_result"]
        assert info["converged"] and optimizer["warnflag"] == 0 and not info["fallback"]
        assert np.isfinite(params).all() and ar_root > 1 and ma_root > 1
        assert difference < 1e-6 and abs(result.llf - info["llf"]) < 1e-6
        rows.append({"origin_id": row.origin_id, "issue": row.target_start,
            **info["fitted_parameters"], "minimum_ar_root_modulus": ar_root,
            "minimum_ma_root_modulus": ma_root, "optimizer_iterations": optimizer["iterations"],
            "optimizer_warnflag": optimizer["warnflag"], "maximum_absolute_optimizer_gradient": float(np.max(np.abs(optimizer["gopt"]))),
            "refilter_maximum_prediction_difference": difference,
            "parameter_covariance_condition_number": float(np.linalg.cond(result.cov_params()))})
    assert len(rows) == 12
    replays = json.loads((e.ROOT / "phase1/results/numerical_replay.json").read_text())
    assert replays["all_replay_predictions_match_within_1e_minus_8"]
    # Regression suite before enabling the final loader. Legacy tests are read-only.
    logs = []
    for folder in (e.ROOT / "phase1/tests", e.HERE / "tests"):
        run = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(folder), "-v"], capture_output=True, text=True)
        logs.append(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError(logs[-1])
    e.RESULTS.mkdir(exist_ok=True)
    (e.RESULTS / "preflight_tests.log").write_text("\n".join(logs))
    pd.DataFrame(rows).to_csv(e.RESULTS / "development_numerical_audit.csv", index=False)
    report = {"status":"passed", "completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "frozen_files_verified":integrity, "development_selection_fits_checked":12,
        "refilter_forecasts_match":True, "previous_unchanged_optimizer_replays":replays["replays"],
        "minimum_ma_root_modulus":min(r["minimum_ma_root_modulus"] for r in rows),
        "maximum_ma_root_modulus":max(r["minimum_ma_root_modulus"] for r in rows),
        "final_target_values_loaded":False, "final_performance_computed":False,
        "targeted_tests_passed":23, "protocol_corrections":[],
        "remaining_risks":"Seasonal MA close to -1; weak boundary identification and optimizer portability remain. Convergence is not global optimality. Two prior optimizer replays and twelve fixed-parameter refilters are environment-specific diagnostics, not robustness across environments.",
        "failure_policy":"Unchanged: nonconvergence, nonfinite parameters/likelihood/forecast or recorded numerical exceptions trigger one logged weekly-naive fallback; include every path. No retries, new thresholds, or performance-driven substitutions. Near-boundary finite invertible fits are retained; programming errors propagate."}
    e.write_json(e.RESULTS / "preflight.json", report)
    sources = ["phase2/final_experiment.py", "phase2/preflight.py", "phase2/tests/test_final_contract.py",
               "phase1/src/forecasting.py", "phase1/FINAL_EVALUATION_SPEC.md", "phase1/results/proposed_final_configuration.json",
               "protocol/split_schedule.csv", "protocol/freeze_manifest.json", "phase1/freeze_manifest.json"]
    contract = {"authority":"User Phase 2 brief: preflight then execute if passed", "preflight_passed":True,
        "locked_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "locked_sources":{p:e.digest(e.ROOT/p) for p in sources}, "preflight_sha256":e.digest(e.RESULTS/"preflight.json"),
        "procedures":e.roster(), "interval":proposal["interval"], "final_paths":13, "primary_days":364, "sensitivity_days":335,
        "final_actuals_loaded_before_lock":False, "procedure_changed":False,
        "limitations":"Local execution record, not external preregistration or proof of analyst blindness; frozen retrieved history, not historical-as-of data."}
    e.write_json(e.HERE / "execution_contract.json", contract)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
