#!/usr/bin/env python3
"""Render foundation-model diagnostics from saved predictions only."""
from pathlib import Path
import os
import sys
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'buckethead-foundation-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import foundation_analysis as fm


def main():
    fm.verify_foundation()
    folder = ROOT / 'results/foundation_models'
    figures = ROOT / 'figures/foundation_models'
    figures.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(folder / 'comparison.csv')
    paired = pd.read_csv(folder / 'paired_paths.csv')
    intervals = pd.read_csv(folder / 'interval_metrics.csv')
    forecasts = pd.read_csv(folder / 'forecasts.csv')
    new = forecasts.loc[forecasts.scored & (forecasts.stage == 'retrospective')]
    all_predictions = pd.concat([new, fm.reference_forecasts()], ignore_index=True)
    colors = {'chronos2': '#007f79', 'timesfm25': '#8056a3', 'weekday_mean': '#d58a00',
              'weekly_naive': '#8297aa', 'original_sarimax': '#bd5a49', 'lgbm_correction': '#5278a1'}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'figure.facecolor': 'white'})

    def save(figure, name):
        figure.savefig(figures / name, dpi=170, bbox_inches='tight', facecolor='white')
        plt.close(figure)

    primary = comparison.loc[comparison.population == 'all'].sort_values('mae')
    labels = [fm.LABELS[k] for k in primary.model_id]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for axis, column, title in zip(axes, ['mae', 'rmse', 'signed_error'],
                                   ['Absolute error', 'Root mean squared error', 'Mean prediction minus actual']):
        axis.barh(labels, primary[column], color=[colors[k] for k in primary.model_id])
        axis.set_title(title)
        axis.set_xlabel('Views per daily prediction')
        axis.grid(axis='x', alpha=.18)
        axis.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle('Same 364 retrospective targets, different forecasting procedures', fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, '01_point_comparison.png')

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for k in ['chronos2', 'timesfm25']:
        frame = paired.loc[(paired.population == 'all') & (paired.model_id == k)].sort_values('issue')
        axes[0].plot(np.arange(1, 14), frame.difference, marker='o', color=colors[k], label=fm.LABELS[k])
    axes[0].axhline(0, color='#333333', lw=1)
    axes[0].set(xlabel='Chronological retrospective path', ylabel='Path MAE minus weekday-mean MAE',
                title='Negative favors the foundation model', xticks=np.arange(1, 14))
    for k in ['chronos2', 'timesfm25', 'weekday_mean', 'original_sarimax']:
        frame = all_predictions.loc[all_predictions.model_id == k].copy()
        frame['loss'] = (frame.prediction - frame.actual).abs()
        profile = frame.groupby('horizon').loss.mean()
        axes[1].plot(profile.index, profile.values, color=colors[k], label=fm.LABELS[k])
    axes[1].set(xlabel='Scored forecast step (1 to 28)', ylabel='MAE, views/day',
                title='Thirteen outcomes contribute at each step')
    for ax in axes:
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, '02_paired_and_horizon.png')

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for row in intervals.loc[intervals.population == 'all'].itertuples():
        axes[0].scatter(row.mean_width, 100 * row.coverage, color=colors[row.model_id], s=75)
        axes[0].annotate(fm.LABELS[row.model_id], (row.mean_width, 100 * row.coverage),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0].axhline(80, linestyle='--', color='#333333', lw=1)
    axes[0].set(xlabel='Mean clipped interval width, views', ylabel='Empirical pointwise coverage (%)',
                title='Read containment alongside width', ylim=(0, 105))
    for k in ['chronos2', 'timesfm25', 'weekday_mean']:
        frame = all_predictions.loc[all_predictions.model_id == k].copy()
        frame['hit'] = (frame.actual >= frame.lower) & (frame.actual <= frame.upper)
        coverage = frame.groupby('horizon').hit.mean()
        axes[1].plot(coverage.index, 100 * coverage, marker='.', color=colors[k], label=fm.LABELS[k])
    axes[1].axhline(80, linestyle='--', color='#333333', lw=1)
    axes[1].set(xlabel='Scored forecast step', ylabel='Empirical pointwise coverage (%)',
                title='Native FM bands versus residual baseline bands', ylim=(0, 105))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.tight_layout()
    save(fig, '03_native_intervals.png')

    issues = sorted(new.issue.unique())[:3]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, issue in zip(axes, issues):
        base = all_predictions.loc[(all_predictions.issue == issue) &
                                   (all_predictions.model_id == 'weekday_mean')].sort_values('horizon')
        ax.plot(base.horizon, base.actual, color='#222222', lw=1.7, label='Actual views')
        ax.plot(base.horizon, base.prediction, '--', color=colors['weekday_mean'], label='Weekday mean')
        for k in ['chronos2', 'timesfm25']:
            frame = new.loc[(new.issue == issue) & (new.model_id == k)].sort_values('horizon')
            ax.plot(frame.horizon, frame.prediction, color=colors[k], label=fm.LABELS[k])
            ax.fill_between(frame.horizon.to_numpy(), frame.lower.to_numpy(), frame.upper.to_numpy(),
                            color=colors[k], alpha=.12)
        ax.set(title=f'Issue {issue}: first three retrospective paths in chronological order',
               ylabel='Daily pageviews')
        ax.grid(alpha=.18)
    axes[0].legend(ncol=4, fontsize=9)
    axes[-1].set_xlabel('Forecast step, after discarding the intervening day')
    fig.tight_layout()
    save(fig, '04_first_three_paths.png')


if __name__ == '__main__':
    main()
