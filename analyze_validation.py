"""Task 5: retrospective validation diagnostics, without training or test evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import load_daily, make_samples, split_samples, score

MODELS = ['persistence', 'linear_regression', 'lstm', 'transformer']
LABELS = ['Persistence', 'Linear regression', 'LSTM', 'Transformer']
COLORS = ['#888888', '#009E73', '#E69F00', '#0072B2']


def diagnostic_thresholds(daily):
    """Use only the fixed training calendar; diff never bridges a missing day."""
    train = daily['pm25'].iloc[:int(len(daily) * .70)]
    return dict(large_daily_change=float(train.diff().abs().quantile(.90)),
                high_aqi=float(train.quantile(.90)))


def error_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    return dict(n=len(actual), MAE=float(np.abs(error).mean()),
                RMSE=float(np.sqrt(np.mean(error**2))),
                mean_error=float(error.mean()), median_absolute_error=float(np.median(np.abs(error))),
                maximum_absolute_error=float(np.max(np.abs(error))))


def group_masks(frame, thresholds):
    change = frame['daily_change']
    cutoff = thresholds['large_daily_change']
    return dict(ordinary_change=change.abs() < cutoff,
                large_change=change.abs() >= cutoff,
                large_rise=change >= cutoff,
                large_fall=change <= -cutoff,
                high_aqi=frame['actual'] >= thresholds['high_aqi'],
                other_aqi=frame['actual'] < thresholds['high_aqi'])


def plot_results(frame, grouped, history_paths, largest_date, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    axes[0].plot(frame.index, frame.actual, color='#222222', lw=1.6, label='Actual')
    for model, label, color in zip(MODELS[1:], LABELS[1:], COLORS[1:]):
        axes[0].plot(frame.index, frame[model], label=label, color=color, lw=1, alpha=.9)
    axes[0].set(title='Validation forecasts: fixed seed 42 for neural models', ylabel='PM2.5 AQI')
    axes[0].legend(ncol=4, fontsize=9)
    axes[1].plot(frame.index, frame.transformer-frame.actual, color=COLORS[-1], lw=1)
    axes[1].axhline(0, color='#777777', lw=.8)
    axes[1].set(ylabel='Transformer error\n(predicted - actual)', xlabel='Target date')
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.tight_layout()
    fig.savefig(output / 'validation_comparison.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, group in zip(axes, ['ordinary_change', 'large_change']):
        selected = grouped[grouped.group.eq(group)]
        means, stds = [], []
        for model in MODELS:
            values = selected[selected.model.eq(model)].MAE
            means.append(values.mean())
            stds.append(values.std(ddof=1) if len(values)>1 else 0)
        ax.bar(range(4), means, yerr=stds, color=COLORS, capsize=4, width=.65)
        ax.set_xticks(range(4), ['Persistence','Linear','LSTM','Transformer'], rotation=15)
        ax.set_title(f'{group.replace("_", " ").title()} (n={int(selected.n.iloc[0])})')
        ax.set_ylim(0, max(75, max(means)+10))
        for i, value in enumerate(means):
            ax.text(i, value+2, f'{value:.2f}', ha='center', fontsize=10)
    axes[0].set_ylabel('Mean absolute error (AQI points)')
    fig.suptitle('Daily-change diagnostics: neural means across seeds 42, 43, 44', fontsize=12)
    fig.tight_layout()
    fig.savefig(output / 'change_error_comparison.png', dpi=180)
    plt.close(fig)

    window = frame.loc[largest_date-pd.Timedelta(days=7):largest_date+pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(window.index, window.actual, color='#222222', lw=2, marker='o', label='Actual')
    for model, label, color in zip(MODELS, LABELS, COLORS):
        ax.plot(window.index, window[model], color=color, lw=1.2, label=label)
    ax.axvline(largest_date, color='#777777', linestyle='--', lw=1)
    ax.set(title=f'Largest Transformer absolute error (seed 42): {largest_date.date()}',
           ylabel='PM2.5 AQI', xlabel='Target date (retrospective context window)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / 'largest_error_case.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, kind in zip(axes, ['transformer', 'lstm']):
        history = pd.read_csv(history_paths[kind])
        ax.plot(history.epoch, history.train_mse, label='Train (eval mode)', color='#0072B2')
        ax.plot(history.epoch, history.validation_mse, label='Validation', color='#E69F00')
        best = int(history.loc[history.validation_rmse.idxmin(),'epoch'])
        ax.axvline(best, color='#777777', lw=1, ls='--', label=f'Best epoch {best}')
        ax.set(title=f'{kind.title()}: selected config, seed 42', xlabel='Epoch')
        ax.legend(fontsize=9)
    axes[0].set_ylabel('MSE (standardized AQI)')
    fig.tight_layout()
    fig.savefig(output / 'training_curves.png', dpi=180)
    plt.close(fig)


def analyze(data, experiments, baselines, output):
    experiments, output = Path(experiments), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new output directory.')
    plan = json.loads((experiments / 'plan.json').read_text(encoding='utf-8'))
    if hashlib.sha256(Path(data).read_bytes()).hexdigest() != plan['data_sha256']:
        raise ValueError('Data differ from task 4; do not mix experiments.')
    seed_runs = pd.read_csv(experiments / 'selected_seed_results.csv')
    if len(seed_runs) != 6 or seed_runs[['model','seed']].duplicated().any():
        raise ValueError('Expected exactly three unique seeds per neural model.')
    daily = load_daily(data)
    samples = split_samples(make_samples(daily), daily.index)
    validation = samples[samples['split'].eq('validation')]
    frame = pd.read_csv(Path(baselines) / 'validation_predictions.csv',
                        parse_dates=['target_date']).set_index('target_date')
    pd.testing.assert_index_equal(frame.index, validation.index)
    np.testing.assert_array_equal(frame.actual, validation.target)
    frame['previous_actual'] = validation.lag_1
    frame['daily_change'] = frame.actual-frame.previous_actual
    thresholds = diagnostic_thresholds(daily)
    predictions = {kind: [(None, frame[kind])] for kind in MODELS[:2]}
    history_paths, histories = {}, []
    source_paths = [experiments/'plan.json', experiments/'selection.json',
                    experiments/'selected_seed_results.csv', Path(baselines)/'validation_predictions.csv']
    for kind in ['transformer', 'lstm']:
        rows = seed_runs[seed_runs.model.eq(kind)]
        if sorted(rows.seed.tolist()) != [42,43,44]:
            raise ValueError('The fixed seed set is 42,43,44.')
        predictions[kind] = []
        for row in rows.itertuples():
            directory = experiments / row.directory
            pred_path = directory / 'validation_predictions.csv'
            pred = pd.read_csv(pred_path, parse_dates=['target_date']).set_index('target_date')
            pd.testing.assert_index_equal(pred.index, frame.index)
            np.testing.assert_array_equal(pred.actual, frame.actual)
            values = pred[kind]
            if not np.isfinite(values).all():
                raise ValueError('Non-finite predictions.')
            predictions[kind].append((int(row.seed), values))
            history_path = directory/'training_history.csv'
            history = pd.read_csv(history_path)
            summary_path = directory/'summary.json'
            report = json.loads(summary_path.read_text())
            assert report['test_evaluated'] is False
            assert report['seed'] == row.seed
            best = history.loc[history.validation_rmse.idxmin()]
            last = history.iloc[-1]
            histories.append(dict(model=kind, seed=int(row.seed),
                best_epoch=int(best.epoch), last_epoch=int(last.epoch),
                first_train_mse=float(history.train_mse.iloc[0]),
                best_train_mse=float(best.train_mse), best_validation_mse=float(best.validation_mse),
                best_validation_rmse=float(best.validation_rmse),
                final_train_mse=float(last.train_mse),
                final_validation_rmse=float(last.validation_rmse),
                final_minus_best_validation_rmse=float(last.validation_rmse-best.validation_rmse)))
            source_paths.extend([pred_path, history_path, summary_path])
            if row.seed == 42:
                frame[kind] = values
                history_paths[kind] = history_path
    masks = group_masks(frame, thresholds)
    overall, grouped, monthly = [], [], []
    for kind, runs in predictions.items():
        for seed, values in runs:
            overall.append(dict(model=kind, seed=seed, **error_metrics(frame.actual,values)))
            for group, mask in masks.items():
                if mask.any():
                    grouped.append(dict(model=kind, seed=seed, group=group,
                        **error_metrics(frame.loc[mask,'actual'],values.loc[mask])))
            for month, part in frame.groupby(frame.index.to_period('M')):
                monthly.append(dict(model=kind, seed=seed, month=str(month),
                    **error_metrics(part.actual,values.loc[part.index])))
    for kind in MODELS:
        frame[kind+'_error'] = frame[kind]-frame.actual
        frame[kind+'_absolute_error'] = frame[kind+'_error'].abs()
    top = frame.sort_values('transformer_absolute_error', ascending=False, kind='stable').head(10)
    worst_date = top.index[0]
    groups = pd.DataFrame(grouped)
    largest = top.iloc[0]
    report = dict(validation_samples=len(frame),
                  validation_start=str(frame.index.min().date()),
                  validation_end=str(frame.index.max().date()),
                  representative_seed=42, thresholds=thresholds,
                  threshold_source='90th percentile of fixed training calendar only',
                  group_counts={name:int(mask.sum()) for name,mask in masks.items()},
                  error_sign='predicted minus actual; positive means overprediction',
                  top10_transformer_squared_error_share=float(
                      np.square(top.transformer_error).sum()/np.square(frame.transformer_error).sum()),
                  largest_error=dict(date=str(worst_date.date()), actual=float(largest.actual),
                      previous_actual=float(largest.previous_actual),
                      daily_change=float(largest.daily_change), prediction=float(largest.transformer),
                      absolute_error=float(largest.transformer_absolute_error)),
                  test_evaluated=False,
                  interpretation='Retrospective validation diagnostics only. Target-day change '
                      'and level use observed outcomes to group errors; they are not forecasting inputs. '
                      'Thresholds are analysis definitions, not health-risk categories. '
                      'No retraining, new model selection, ensemble or test metrics.')
    output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'aligned_predictions_seed42.csv',encoding='utf-8-sig')
    pd.DataFrame(overall).to_csv(output/'overall_errors.csv',index=False)
    groups.to_csv(output/'group_errors.csv',index=False)
    pd.DataFrame(monthly).to_csv(output/'monthly_errors.csv',index=False)
    top.to_csv(output/'top10_transformer_errors.csv',encoding='utf-8-sig')
    pd.DataFrame(histories).to_csv(output/'training_diagnostics.csv',index=False)
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    manifest = dict(data_sha256=plan['data_sha256'],
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    (output/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    plot_results(frame, groups, history_paths, worst_date, output)
    print(json.dumps(report,indent=2))
    print(groups.groupby(['group','model']).agg(n=('n','first'),
          MAE=('MAE','mean'), mean_error=('mean_error','mean')).to_string())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--experiments',default='results/stage4')
    parser.add_argument('--baselines',default='results/baselines')
    parser.add_argument('--output',default='results/stage5')
    args=parser.parse_args()
    analyze(args.data,args.experiments,args.baselines,args.output)
