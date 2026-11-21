"""Execute only the frozen bounded Phase 4 roster, with immutable fit records."""
import engine as e
import json,datetime
import numpy as np
import pandas as pd


def locked():
    lock=json.loads((e.HERE/"implementation_lock.json").read_text())
    assert lock["preflight_passed"]
    for relative,expected in lock["files"].items():assert e.digest(e.ROOT/relative)==expected,relative
    e.protect()
    return e.digest(e.HERE/"implementation_lock.json")


def summary(frame,audits):
    rows=[]
    for (stage,identifier),g in frame.groupby(["stage","id"]):
        fs=[a for a in audits if a["stage"]==stage and a["id"]==identifier]
        rows.append({"stage":stage,"population":"all","id":identifier,"paths":g.origin_id.nunique(),
            "failures":sum(a["fallback"] for a in fs),**e.metrics(g)})
    sub=frame.loc[(frame.stage=="retrospective")&~frame.date.str.startswith("2025-11-")]
    for identifier,g in sub.groupby("id"):
        rows.append({"stage":"retrospective","population":"exclude_november_2025","id":identifier,
            "paths":g.origin_id.nunique(),"failures":sum(a["fallback"] for a in audits if a["stage"]=="retrospective" and a["id"]==identifier and a["origin_id"] in set(g.origin_id)),**e.metrics(g)})
    return pd.DataFrame(rows)


def main():
    token=locked();y=e.load_y();rows=[];audits=[]
    specs=e.roster();calibrations={};selected=None
    for stage in ("development","selection","calibration","retrospective"):
        if stage=="calibration":
            sums=summary(pd.DataFrame(rows),audits)
            selection=sums.loc[(sums.stage=="selection")&(sums.population=="all")]
            selected=e.choose(selection)
            record={"selected_id":selected,"selection_role":"Seven chronological selection paths only",
                "selection_scores":selection.to_dict("records"),"locked_before_calibration_scores":True,
                "created_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"implementation_lock_sha256":token}
            path=e.RESULTS/"point_selection.json"
            if path.exists():
                old=json.loads(path.read_text());assert old["selected_id"]==selected and old["implementation_lock_sha256"]==token
            else:e.dump(path,record)
        if stage=="retrospective":
            frame=pd.DataFrame(rows)
            for identifier in dict.fromkeys([selected,"weekday_mean"]):
                g=frame.loc[(frame.stage=="calibration")&(frame.id==identifier)]
                calibrations[identifier]=e.calibrate(g,"2025-11-02")
            path=e.RESULTS/"interval_calibration.json"
            if path.exists():assert json.loads(path.read_text())["procedures"]==calibrations
            else:e.dump(path,{"procedures":calibrations,"target_start":"2025-06-15","target_end":"2025-10-04", "point_selection_fixed":selected,
                "method":"fixed pooled signed-error order statistics, floor/ceil tail ranks", "created_before_retrospective_candidate_scores":True})
        for _,s in e.schedule().loc[lambda d:d.stage==stage].iterrows():
            origin=e.at_origin(y,s.issue)
            table=None
            for spec in specs:
                key=f"{s.origin_id}__{spec['id']}__{token[:12]}"
                path=e.RESULTS/"cache"/(key+".json")
                if path.exists():
                    saved=json.loads(path.read_text());assert saved["lock_sha256"]==token and saved["spec"]==spec and saved["issue"]==s.issue
                    pred=np.asarray(saved["prediction"]);info=saved["info"]
                else:
                    if spec["family"]=="lightgbm" and table is None:table=e.training_table(origin)
                    pred,info=e.predict(y,s.issue,spec,table=table)
                    e.dump(path,{"lock_sha256":token,"spec":spec,"issue":s.issue,"dates":[str(d.date()) for d in origin.targets],"prediction":pred.tolist(),"info":info})
                assert len(pred)==28 and np.isfinite(pred).all() and info["cutoff"]==s.cutoff
                audits.append({"stage":stage,"origin_id":s.origin_id,"cache_key":key,**info})
                lo,hi=e.interval(pred,calibrations[spec["id"]]) if stage=="retrospective" and spec["id"] in calibrations else (None,None)
                for h,(day,point) in enumerate(zip(origin.targets,pred),1):
                    row={"stage":stage,"origin_id":s.origin_id,"issue":s.issue,"cutoff":s.cutoff,"id":spec["id"],"date":str(day.date()),"horizon":h,
                        "actual":float(y.loc[day]),"prediction":float(point),"mase_scale":info["mase_scale"],"fallback":info["fallback"]}
                    if lo is not None:row.update({"lower":float(lo[h-1]),"upper":float(hi[h-1])})
                    rows.append(row)
            e.dump(e.RESULTS/"progress.json",{"completed_records":len(audits),"planned_records":288,"stage":stage,"latest_issue":s.issue,"lock_sha256":token})
            print(stage,s.origin_id,s.issue,"records",len(audits),flush=True)
        pd.DataFrame(rows).to_csv(e.RESULTS/"forecasts.csv",index=False)
        e.dump(e.RESULTS/"fit_audits.json",audits)
        summary(pd.DataFrame(rows),audits).to_csv(e.RESULTS/"summary.csv",index=False)
    frame=pd.DataFrame(rows);frame["signed_error"]=frame.prediction-frame.actual
    frame["absolute_error"]=frame.signed_error.abs();frame["squared_error"]=frame.signed_error**2
    frame.to_csv(e.RESULTS/"forecasts.csv",index=False)
    sums=summary(frame,audits);sums.to_csv(e.RESULTS/"summary.csv",index=False)
    origin_rows=[];horizon_rows=[];pair_rows=[];interval_rows=[]
    for stage in ("development","selection","calibration","retrospective"):
        a=frame.loc[frame.stage==stage]
        populations={"all":a}
        if stage=="retrospective":populations["exclude_november_2025"]=a.loc[~a.date.str.startswith("2025-11-")]
        for population,sub in populations.items():
            for (identifier,oid),g in sub.groupby(["id","origin_id"]):
                origin_rows.append({"stage":stage,"population":population,"id":identifier,"origin_id":oid,"issue":g.issue.iloc[0],**e.metrics(g)})
            for (identifier,h),g in sub.groupby(["id","horizon"]):
                horizon_rows.append({"stage":stage,"population":population,"id":identifier,"horizon":h,**e.metrics(g)})
            for identifier,g in sub.groupby("id"):
                for baseline in ("weekday_mean","weekly_naive"):
                    b=sub.loc[sub.id==baseline]
                    for oid,z in [("all",g),*list(g.groupby("origin_id"))]:
                        ref=b.loc[b.origin_id.isin(z.origin_id.unique())]
                        assert list(z.date)==list(ref.date)
                        am,bm=e.metrics(z)["mae"],e.metrics(ref)["mae"]
                        pair_rows.append({"stage":stage,"population":population,"id":identifier,"baseline":baseline,"origin_id":oid,"days":len(z),
                            "mae_difference":am-bm,"relative_mae_improvement_percent":100*(1-am/bm)})
            if stage=="retrospective":
                for identifier in calibrations:
                    chosen=sub.loc[sub.id==identifier]
                    for group,g in [("all",chosen),*list(chosen.groupby("origin_id"))]:
                        interval_rows.append({"population":population,"id":identifier,"group_type":"origin","group":group,**e.interval_metrics(g.actual,g.lower,g.upper)})
                    for h,g in chosen.groupby("horizon"):
                        interval_rows.append({"population":population,"id":identifier,"group_type":"horizon","group":str(h),**e.interval_metrics(g.actual,g.lower,g.upper)})
                    complete=[g for _,g in chosen.groupby("origin_id") if len(g)==28]
                    interval_rows.append({"population":population,"id":identifier,"group_type":"simultaneous_complete_paths","group":"all",
                        "paths":len(complete),"path_coverage":float(np.mean([((g.actual>=g.lower)&(g.actual<=g.upper)).all() for g in complete]))})
    for name,data in (("origin_metrics",origin_rows),("horizon_metrics",horizon_rows),("paired_differences",pair_rows),("interval_metrics",interval_rows)):
        pd.DataFrame(data).to_csv(e.RESULTS/(name+".csv"),index=False)
    retrospective=sums.loc[(sums.stage=="retrospective")&(sums.population=="all")]
    nominee=e.choose(retrospective)
    e.dump(e.RESULTS/"future_nomination.json",{"nominee":nominee,"basis":"Lowest retrospective exploratory MAE among procedures with no retrospective fit failures; frozen tie rule",
        "development_selected_id":selected,"nominee_has_evaluated_intervals":nominee in calibrations,
        "future_reference":"weekday_mean","fresh_confirmation":"pending; zero complete future paths in the frozen snapshot",
        "planned_issues":["2026-11-08","2026-12-06","2027-01-03","2027-01-31"],"forecasts_issued":False,
        "interpretation":"Exploratory nomination, not an independently validated winner or calibrated deployment recommendation"})
    e.protect();e.dump(e.RESULTS/"progress.json",{"status":"completed; stopping rule reached","completed_records":len(audits),"real_fit_attempts":216,"baseline_paths":72,
        "all_daily_predictions":len(frame),"post_evaluation":True,"future_confirmation":"pending","lock_sha256":token})
    print(retrospective.sort_values("mae").to_string(index=False))
    print("Development selected:",selected,"Exploratory future nominee:",nominee,flush=True)


if __name__=="__main__":main()
