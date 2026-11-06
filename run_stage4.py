"""Predeclared validation search followed by three-seed stability experiments."""
import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from pipeline import score
from sequence_training import prepare_data, run
from transformer_model import load_checkpoint as load_transformer
from lstm_model import load_checkpoint as load_lstm

METRICS = ['MAE', 'RMSE', 'R2']
SOURCE_FILES = ['pipeline.py', 'sequence_training.py', 'transformer_model.py',
                'lstm_model.py', 'run_stage4.py']


def make_plan(data):
    source = Path(__file__).resolve().parent
    return dict(
        protocol='WEEK 13 task 4 validation-only search v1',
        data_sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest(),
        code_sha256={p: hashlib.sha256((source / p).read_bytes()).hexdigest()
                     for p in SOURCE_FILES},
        selection_seed=42, stability_seeds=[42, 43, 44],
        selection_metric='validation RMSE',
        tie_break='candidate order',
        standard_deviation='sample standard deviation, ddof=1',
        training=dict(epochs=150, patience=20, batch_size=32, threads=2),
        candidates={
            'transformer': [
                dict(id='baseline', lr=0.001, model_config=dict()),
                dict(id='lower_lr', lr=0.0003, model_config=dict()),
                dict(id='wider', lr=0.001,
                     model_config=dict(d_model=64, dim_feedforward=128)),
            ],
            'lstm': [
                dict(id='baseline', lr=0.001, model_config=dict()),
                dict(id='lower_lr', lr=0.0003, model_config=dict()),
                dict(id='wider', lr=0.001, model_config=dict(hidden_size=64)),
            ],
        },
        test_evaluated=False,
        note='Select each model configuration using seed 42 only; do not reselect '
             'after seeds 43/44. Seed 42 is reused, not counted twice. '
             'These validation estimates include model selection and are not final test results.')


def select_candidate(reports):
    # Python min preserves input order for exact ties. No test metric is read.
    return min(range(len(reports)), key=lambda i: reports[i]['validation_metrics']['RMSE'])


def aggregate_metrics(reports):
    return {metric: dict(mean=float(np.mean([r['validation_metrics'][metric] for r in reports])),
                         std=float(np.std([r['validation_metrics'][metric] for r in reports], ddof=1)),
                         minimum=float(min(r['validation_metrics'][metric] for r in reports)),
                         maximum=float(max(r['validation_metrics'][metric] for r in reports)))
            for metric in METRICS}


def verify_run(directory, report, kind, prepared):
    predictions = pd.read_csv(directory / 'validation_predictions.csv',
                              parse_dates=['target_date']).set_index('target_date')
    expected = prepared['validation']
    pd.testing.assert_index_equal(predictions.index, expected.index)
    np.testing.assert_array_equal(predictions['actual'], expected['target'])
    assert not report['test_evaluated']
    assert report['scaler'] == dict(mean=prepared['mean'], scale=prepared['scale'],
        fit_end=str(prepared['daily'].index[prepared['train_stop'] - 1].date()))
    loader = load_transformer if kind == 'transformer' else load_lstm
    model, saved = loader(directory / 'best_model.pt')
    assert saved['model_config'] == report['model_config']
    assert saved['seed'] == report['seed']
    with torch.inference_mode():
        reloaded = model(prepared['xv']).numpy() * prepared['scale'] + prepared['mean']
    np.testing.assert_allclose(reloaded, predictions[kind], rtol=1e-6, atol=1e-5)
    metrics = score(predictions['actual'], predictions[kind])
    for metric in METRICS:
        np.testing.assert_allclose(metrics[metric], report['validation_metrics'][metric], rtol=1e-6)
    history = pd.read_csv(directory / 'training_history.csv')
    assert saved['best_epoch'] == report['training']['best_epoch']
    assert int(history.loc[history.validation_rmse.idxmin(), 'epoch']) == saved['best_epoch']
    return dict(validation_dates_and_targets_match=True, training_only_scaler_matches=True,
                reloaded_predictions_match=True, metrics_recomputed=True,
                best_epoch_matches_history=True, test_evaluated=False)


def run_experiments(data, output, resume=False):
    output = Path(output)
    plan = make_plan(data)
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise ValueError('Output exists. Use a new directory or --resume.')
        previous = json.loads((output / 'plan.json').read_text(encoding='utf-8'))
        if previous != plan:
            raise ValueError('Plan, data or source changed; choose a new output directory.')
    output.mkdir(parents=True, exist_ok=True)
    # Freeze the search space before any candidate is trained.
    (output / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    (output / 'logs').mkdir(exist_ok=True)
    prepared = prepare_data(data)
    all_rows, checks = [], {}

    def train(kind, candidate, seed, phase):
        relative = Path(phase) / kind / candidate['id'] / f'seed{seed}'
        directory = output / relative
        print(f'{phase}: {kind}/{candidate["id"]}/seed{seed}', flush=True)
        args = SimpleNamespace(data=str(data), output=str(directory), seed=seed,
                               lr=candidate['lr'], model_config=candidate['model_config'],
                               **plan['training'])
        if resume and (directory / 'summary.json').exists():
            report = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        else:
            log_path = output / 'logs' / (str(relative).replace('\\', '_').replace('/', '_') + '.log')
            with log_path.open('w', encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
                report = run(args, kind)
        assert report['data_sha256'] == plan['data_sha256']
        assert report['seed'] == seed and report['training']['lr'] == candidate['lr']
        checks[str(relative)] = verify_run(directory, report, kind, prepared)
        row = dict(phase=phase, model=kind, candidate=candidate['id'], seed=seed,
                   directory=relative.as_posix(), lr=candidate['lr'],
                   model_config=json.dumps(report['model_config'], sort_keys=True),
                   best_epoch=report['training']['best_epoch'],
                   epochs_run=report['training']['epochs_run'],
                   parameter_count=report['parameter_count'],
                   **report['validation_metrics'])
        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(output / 'all_runs.csv', index=False)
        (output / 'verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
        print(f'  MAE={row["MAE"]:.4f}, RMSE={row["RMSE"]:.4f}, R2={row["R2"]:.4f}', flush=True)
        return report

    selected, selected_reports = {}, {}
    for kind, candidates in plan['candidates'].items():
        reports = [train(kind, c, plan['selection_seed'], 'search') for c in candidates]
        best = select_candidate(reports)
        selected[kind] = candidates[best]
        selected_reports[kind] = [reports[best]]
    # Freeze selection before the additional seeds; never pick the best seed.
    selection = dict(criterion=plan['selection_metric'], seed=42, selected=selected,
                     selection_frozen_before_stability=True, test_evaluated=False)
    (output / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    for kind, candidate in selected.items():
        for seed in plan['stability_seeds'][1:]:
            selected_reports[kind].append(train(kind, candidate, seed, 'stability'))
    statistics = {kind: aggregate_metrics(reports) for kind, reports in selected_reports.items()}
    comparisons = []
    # Fit deterministic baselines again from the same training samples.
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    features = [f'lag_{i}' for i in range(7, 0, -1)]
    train_part, validation = prepared['train'], prepared['validation']
    linear = make_pipeline(StandardScaler(), LinearRegression())
    linear.fit(train_part[features], train_part['target'])
    for name, values in [('persistence', validation['lag_1']),
                         ('linear_regression', linear.predict(validation[features]))]:
        metrics = score(validation['target'], values)
        comparisons.append(dict(model=name, candidate='deterministic baseline', seeds='N/A',
                                **{f'{k}_mean': v for k, v in metrics.items()},
                                **{f'{k}_std': None for k in METRICS}))
    for kind, stats in statistics.items():
        comparisons.append(dict(model=kind, candidate=selected[kind]['id'], seeds='42,43,44',
                                **{f'{k}_{v}': stats[k][v] for k in METRICS for v in ['mean', 'std']}))
    pd.DataFrame(comparisons).to_csv(output / 'model_comparison.csv', index=False)
    pd.DataFrame([r for r in all_rows if r['phase'] == 'search']).to_csv(
        output / 'search_results.csv', index=False)
    stability_rows = [r for r in all_rows if r['candidate'] == selected[r['model']]['id']]
    assert len(stability_rows) == 6
    pd.DataFrame(stability_rows).to_csv(output / 'selected_seed_results.csv', index=False)
    summary = dict(selected=selected, statistics=statistics, comparison=comparisons,
                   selection_criterion='seed-42 validation RMSE per model',
                   seeds=plan['stability_seeds'], std_ddof=1, unique_training_runs=len(all_rows),
                   validation_samples=len(validation), test_evaluated=False,
                   limitation='Validation used for early stopping and configuration selection; '
                              'three seeds describe initialization variability on this one split, '
                              'not final test performance or statistical significance.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'environment.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    print(pd.DataFrame(comparisons).to_string(index=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='results/stage4')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run_experiments(args.data, args.output, args.resume)
