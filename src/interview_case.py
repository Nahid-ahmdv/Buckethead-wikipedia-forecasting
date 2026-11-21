"""Read-only description of the existing October 2017 historical case."""
import numpy as np
import pandas as pd

ANCHOR = pd.Timestamp('2017-10-12')


def describe(y):
    reference = y.loc['2017-09-14':'2017-10-11']
    follow = y.loc['2017-10-12':'2017-10-18']
    earlier = y.loc['2017-09-12':'2017-10-09']
    assert len(reference) == len(earlier) == 28 and len(follow) == 7
    assert reference.index.equals(pd.date_range('2017-09-14', '2017-10-11'))
    daily = y.loc['2017-10-09':'2017-10-18'].rename('pageviews').to_frame()
    daily['relative_to_prior28_mean'] = daily.pageviews / reference.mean()
    facts = {'anchor': str(ANCHOR.date()), 'reference_start': '2017-09-14',
        'reference_end': '2017-10-11', 'reference_days': 28,
        'reference_mean': float(reference.mean()), 'reference_median': float(reference.median()),
        'reference_min': int(reference.min()), 'reference_max': int(reference.max()),
        'oct10_views': int(y.loc['2017-10-10']), 'oct11_views': int(y.loc['2017-10-11']),
        'oct12_views': int(y.loc['2017-10-12']), 'oct13_views': int(y.loc['2017-10-13']),
        'oct12_relative_to_reference': float(y.loc['2017-10-12'] / reference.mean()),
        'oct13_relative_to_reference': float(y.loc['2017-10-13'] / reference.mean()),
        'following7_start': '2017-10-12', 'following7_end': '2017-10-18',
        'following7_mean': float(follow.mean()), 'following7_sum': int(follow.sum()),
        'following7_relative_to_reference': float(follow.mean() / reference.mean()),
        'earlier_reference_start': '2017-09-12', 'earlier_reference_end': '2017-10-09',
        'earlier_reference_mean': float(earlier.mean()),
        'following7_relative_to_earlier_reference': float(follow.mean() / earlier.mean()),
        'scope': 'Source-led, previously inspected retrospective illustration; no effect estimate or predictor'}
    return daily, facts

