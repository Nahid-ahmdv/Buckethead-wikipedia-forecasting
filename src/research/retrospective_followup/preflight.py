"""Validate clean environment and tests, then lock implementation before real fits."""
import engine as e
import sys,subprocess,datetime,platform,importlib,importlib.metadata,json
import pandas as pd
import numpy as np


def main():
    if (e.HERE/"implementation_lock.json").exists():raise RuntimeError("Implementation already locked")
    e.protect()
    assert str(e.HERE/".venv") in sys.prefix,"Use the isolated Phase 4 environment"
    modules={}
    for name,dist in (("numpy","numpy"),("pandas","pandas"),("scipy","scipy"),("statsmodels","statsmodels"),("lightgbm","lightgbm"),("sklearn","scikit-learn"),("matplotlib","matplotlib")):
        mod=importlib.import_module(name);actual=mod.__version__;metadata=importlib.metadata.version(dist)
        assert actual==metadata,(name,actual,metadata)
        assert str(e.HERE/".venv") in mod.__file__,mod.__file__
        modules[name]={"imported_version":actual,"distribution_version":metadata,"path":mod.__file__}
    freeze=subprocess.run(["/opt/homebrew/bin/uv","pip","freeze","--python",sys.executable],capture_output=True,text=True,check=True)
    (e.HERE/"requirements.lock").write_text(freeze.stdout)
    run=subprocess.run([sys.executable,str(e.HERE/"tests.py")],capture_output=True,text=True)
    e.RESULTS.mkdir(exist_ok=True)
    (e.RESULTS/"tests.log").write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError(run.stdout+run.stderr)
    s=e.schedule();cal=s.loc[s.stage=="calibration"]
    assert pd.Timestamp(cal.target_end.max())<=pd.Timestamp("2025-11-02")-pd.Timedelta(days=2)
    assert pd.Timestamp(s.loc[s.stage=="selection","target_end"].max())<=pd.Timestamp(cal.issue.min())-pd.Timedelta(days=2)
    # Data are read for validity only; no real model is fitted during preflight.
    y=e.load_y();assert len(y)==4141
    environment={"python":sys.version,"executable":sys.executable,"prefix":sys.prefix,"platform":platform.platform(),
        "modules":modules,"native_threads":1,"installation":"Fresh uv-created isolated venv; all direct and transitive dependencies installed from public package wheels",
        "scope":"Clean Python environment on the existing macOS system, not a clean operating-system/container image; native libomp may be system supplied",
        "real_series_candidate_fits_before_lock":0}
    e.dump(e.RESULTS/"environment.json",environment)
    report={"status":"passed","completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protected_files":e.protect(),"clean_environment_import_metadata_match":True,"targeted_tests_passed":16,
        "six_synthetic_engines_executed":True,"real_series_candidate_fits":0,"chronological_calibration_valid":True,
        "candidate_count":6,"baseline_count":2,"planned_records":288,"planned_daily_predictions":8064,
        "methodological_blockers":[],"protocol_departures":["AMENDMENT_01.md: corrected selection/calibration label availability before real fitting"]}
    e.dump(e.RESULTS/"preflight.json",report)
    files=["engine.py","run_experiment.py","tests.py","preflight.py","PROTOCOL.md","PROTOCOL_V2.md","candidates.json","schedule.csv","schedule_v2.csv","AMENDMENT_01.md","protocol_amendment_01.json","requirements.txt","requirements.lock","protocol_freeze.json","results/environment.json","results/preflight.json"]
    locked={"preflight_passed":True,"created_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files":{"phase4/"+p:e.digest(e.HERE/p) for p in files},
        "inherited_engine_sha256":e.digest(e.ROOT/"phase1/src/forecasting.py"),
        "canonical_input_sha256":e.digest(e.ROOT/"data/processed/buckethead_daily.csv"),"real_series_candidate_fits":0}
    e.dump(e.HERE/"implementation_lock.json",locked)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
