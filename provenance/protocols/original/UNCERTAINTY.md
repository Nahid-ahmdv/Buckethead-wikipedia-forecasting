# One uncertainty procedure

Use an **80% nominal pooled absolute-error interval** for the selected pipeline.
Choose this construction now, not by comparing final coverage. No per-horizon
method search, alternative nominal levels, or extensive uncertainty benchmark.

After hyperparameters and history policy have been selected on development:

1. Retain that procedure's fixed-origin forecasts for the first twelve selection
   paths (2024-11-03–2025-10-04), with hyperparameters chosen in the earlier tuning
   block. Each forecast uses its own available history and the O-2 cutoff.
2. Compute the 336 absolute errors on the original count scale, after clipping
   point forecasts at zero. All these outcomes are available before the first
   final issue date, 2025-11-02. The thirteenth selection/boundary block is not
   used for calibration because its final target is not yet available then.
3. Let n=336, alpha=.2, k=ceil((n+1)*(1-alpha))=270. Take the kth smallest absolute
   error, without interpolating quantiles. If missing forecasts prevent this
   predefined calibration, report failure rather than silently changing n.
4. For every final point prediction p use [max(0,p-q),p+q]. Keep q fixed throughout
   final evaluation. Compute widths after lower-bound clipping.
5. Report empirical daily coverage, mean width, and per-path coverage. Repeat
   coverage/width scoring under the predefined November target-date mask using
   identical intervals. Do not recalibrate against final observations.

This is a conformal-style order-statistic construction, **not a demonstrated
distribution-free coverage guarantee for this series**. Temporal dependence,
changes in traffic, heterogeneous forecast horizons, limited paths, and pipeline
selection matter. The calibration block also selected the procedure/window, so
it is not independent calibration of a previously fixed pipeline. No development
coverage claim will be treated as independent evidence. The later final period
provides empirical assessment of the entire selected construction.

Pooled intervals are intentionally simple and can hide differences across
horizons or event conditions. Describe those limits; do not adapt the method
after viewing which final paths it misses. An interval around user-classified
pageview counts does not quantify uncertainty about the number of human fans.
