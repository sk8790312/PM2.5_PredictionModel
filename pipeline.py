"""Seven-day PM2.5 AQI forecasting: audit data and run transparent baselines."""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

START = '2021-07-01'
END = '2026-07-01'
LOOKBACK = 7


def load_daily(path, start=START, end=END):
    """Keep the calendar intact; never bridge an unobserved day."""
    path = Path(path)
    raw = pd.read_excel(path) if path.suffix.lower() == '.xlsx' else pd.read_csv(path)
    raw.columns = raw.columns.str.strip().str.lower()
    if not {'date', 'pm25'}.issubset(raw.columns):
        raise ValueError('Input must contain date and pm25 columns.')
    raw = raw.dropna(how='all').copy()
    raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
    if raw['date'].isna().any():
        raise ValueError('Non-empty rows contain invalid dates; inspect the input.')
    raw['date'] = raw['date'].dt.normalize()
    selected = raw.loc[raw['date'].between(start, end)].copy()
    if selected['date'].duplicated().any():
        raise ValueError('Duplicate calendar dates found; resolve them before training.')
    original = selected['pm25'].astype('string').str.strip().replace('', pd.NA)
    selected['pm25'] = pd.to_numeric(original, errors='coerce')
    if (original.notna() & selected['pm25'].isna()).any():
        raise ValueError('Non-numeric PM2.5 values found; inspect the input.')
    if (selected['pm25'].dropna() < 0).any():
        raise ValueError('Negative AQI values found; inspect the input.')
    selected = selected.set_index('date').sort_index()
    daily = selected[['pm25']].reindex(pd.date_range(start, end, name='date'))
    daily['record_present'] = daily.index.isin(selected.index)
    daily['missing_reason'] = np.where(
        ~daily['record_present'], 'date_absent',
        np.where(daily['pm25'].isna(), 'pm25_empty', ''))
    return daily


def make_samples(daily, lookback=LOOKBACK):
    """Only windows containing lookback consecutive input days plus a valid label."""
    result = pd.DataFrame(index=daily.index)
    for lag in range(lookback, 0, -1):
        result[f'lag_{lag}'] = daily['pm25'].shift(lag)
    result['target'] = daily['pm25']
    return result.dropna().rename_axis('target_date')


def split_samples(samples, calendar):
    """Assign by label date on fixed calendar boundaries before dropping windows."""
    train_stop = int(len(calendar) * .70)
    validation_stop = int(len(calendar) * .85)
    train_end = calendar[train_stop - 1]
    validation_end = calendar[validation_stop - 1]
    labels = np.where(samples.index <= train_end, 'train',
                      np.where(samples.index <= validation_end, 'validation', 'test'))
    result = samples.copy()
    result['split'] = labels
    return result


def score(y, prediction):
    return {'MAE': float(mean_absolute_error(y, prediction)),
            'RMSE': float(np.sqrt(mean_squared_error(y, prediction))),
            'R2': float(r2_score(y, prediction))}


def run(data, output, evaluate_test=False):
    daily = load_daily(data)
    samples = split_samples(make_samples(daily), daily.index)
    features = [f'lag_{lag}' for lag in range(LOOKBACK, 0, -1)]
    subsets = {s: samples[samples['split'].eq(s)] for s in ['train', 'validation', 'test']}
    if any(len(s) < 2 for s in subsets.values()):
        raise ValueError('Each partition must have at least two complete samples.')
    train = subsets['train']
    # The scaler and regression are fitted only on training samples.
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(train[features], train['target'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    missing = daily[daily['pm25'].isna()][['missing_reason']]
    missing.to_csv(output / 'missing_dates.csv', encoding='utf-8-sig')
    daily.to_csv(output / 'daily_calendar.csv', encoding='utf-8-sig')
    samples.to_csv(output / 'samples.csv', encoding='utf-8-sig')
    report = {
        'target': 'PM2.5-specific AQI (not concentration or overall AQI)',
        'units': 'AQI index points for MAE and RMSE; R2 is unitless',
        'period': {'start': START, 'end': END, 'endpoints_included': True},
        'calendar_days': len(daily),
        'observed_date_rows': int(daily['record_present'].sum()),
        'valid_pm25_days': int(daily['pm25'].notna().sum()),
        'absent_dates': int((~daily['record_present']).sum()),
        'empty_pm25_dates': int((daily['record_present'] & daily['pm25'].isna()).sum()),
        'complete_windows': len(samples),
        'lookback_days': LOOKBACK,
        'split_rule': '70/15/15 calendar periods, assigned by target date',
        'forecast_protocol': 'Rolling next-day forecasts using already observed history; model fitted once.',
        'splits': {}, 'validation_metrics': {},
        'test_evaluated': bool(evaluate_test),
        'runtime': {'python': platform.python_version(), 'pandas': pd.__version__,
                    'numpy': np.__version__, 'scikit-learn': sklearn.__version__},
    }
    for label, part in subsets.items():
        report['splits'][label] = {'samples': len(part),
            'first_target': str(part.index.min().date()),
            'last_target': str(part.index.max().date())}
    for label in ['validation'] + (['test'] if evaluate_test else []):
        part = subsets[label]
        prediction = pd.DataFrame({'actual': part['target'],
            'persistence': part['lag_1'], 'linear_regression': model.predict(part[features])},
            index=part.index)
        report[label + '_metrics'] = {name: score(prediction['actual'], prediction[name])
                                    for name in ['persistence', 'linear_regression']}
        prediction.to_csv(output / f'{label}_predictions.csv', encoding='utf-8-sig')
    (output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    make_plots(daily, output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def make_plots(daily, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(daily.index, daily['pm25'], linewidth=.7)
    ax.set(xlabel='Date', ylabel='PM2.5 AQI', title='Chengdu PM2.5 AQI: selected historical period')
    fig.tight_layout()
    fig.savefig(output / 'data_overview.png', dpi=160)
    plt.close(fig)
    values = pd.read_csv(output / 'validation_predictions.csv', parse_dates=['target_date']).set_index('target_date')
    fig, ax = plt.subplots(figsize=(11, 3.8))
    values.iloc[:90].plot(ax=ax, linewidth=1)
    ax.set(xlabel='Target date', ylabel='PM2.5 AQI', title='Validation forecasts: first 90 valid target dates')
    fig.tight_layout()
    fig.savefig(output / 'validation_forecasts.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Local AQICN .xlsx or .csv file')
    parser.add_argument('--output', default='results/baselines')
    parser.add_argument('--evaluate-test', action='store_true',
                        help='Use only once the experiment design is final; default reserves test set.')
    args = parser.parse_args()
    run(args.data, args.output, args.evaluate_test)
