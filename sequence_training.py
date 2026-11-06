"""Shared training protocol for Transformer and LSTM; validation only."""
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline import LOOKBACK, START, END, load_daily, make_samples, split_samples, score
from transformer_model import CausalTransformer, load_checkpoint as load_transformer
from lstm_model import LSTMRegressor, load_checkpoint as load_lstm


def plot_results(history, predictions, output, display_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history['epoch'], history['train_mse'], label='train (eval mode)')
    ax.plot(history['epoch'], history['validation_mse'], label='validation')
    ax.set(xlabel='Epoch', ylabel='MSE (standardized AQI)', title=f'{display_name} loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'loss_curve.png', dpi=160)
    plt.close(fig)
    ax = predictions.iloc[:90].plot(figsize=(11, 4), linewidth=1)
    ax.set(xlabel='Target date', ylabel='PM2.5 AQI',
           title=f'{display_name} validation: first 90 valid target dates')
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output / 'validation_forecasts.png', dpi=160)
    plt.close(fig)


def prepare_data(path):
    daily = load_daily(path)
    samples = split_samples(make_samples(daily), daily.index)
    train = samples[samples['split'].eq('train')]
    validation = samples[samples['split'].eq('validation')]
    if min(len(train), len(validation)) < 2:
        raise ValueError('Training and validation need at least two complete samples.')
    # One common AQI scale for all time steps and labels. Fit only training dates.
    train_stop = int(len(daily) * .70)
    observed_train = daily['pm25'].iloc[:train_stop].dropna()
    mean = float(observed_train.mean())
    scale = float(observed_train.std(ddof=0)) or 1.0
    features = [f'lag_{i}' for i in range(LOOKBACK, 0, -1)]

    def tensors(part):
        x = ((part[features].to_numpy(dtype=np.float32) - mean) / scale)
        y = ((part['target'].to_numpy(dtype=np.float32) - mean) / scale)
        return torch.tensor(x).unsqueeze(-1), torch.tensor(y)

    xt, yt = tensors(train)
    xv, yv = tensors(validation)
    return dict(daily=daily, samples=samples, train=train, validation=validation,
                mean=mean, scale=scale, train_stop=train_stop,
                xt=xt, yt=yt, xv=xv, yv=yv)


def run(args, model_kind):
    if model_kind not in {'transformer', 'lstm'}:
        raise ValueError('Unknown sequence model.')
    output = Path(args.output)
    # Keep separate runs reviewable; never overwrite an earlier experiment.
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new --output directory.')
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    prepared = prepare_data(args.data)
    daily, samples = prepared['daily'], prepared['samples']
    train, validation = prepared['train'], prepared['validation']
    mean, scale, train_stop = prepared['mean'], prepared['scale'], prepared['train_stop']
    xt, yt, xv, yv = (prepared[k] for k in ['xt', 'yt', 'xv', 'yv'])
    loader = DataLoader(TensorDataset(xt, yt), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    if model_kind == 'transformer':
        config = dict(lookback=LOOKBACK, d_model=32, nhead=2, num_layers=1,
                      dim_feedforward=64, dropout=0.1)
        config.update(getattr(args, 'model_config', None) or {})
        if config['lookback'] != LOOKBACK:
            raise ValueError('The fixed protocol requires seven-day inputs.')
        model = CausalTransformer(**config)
        load_checkpoint = load_transformer
        display_name = 'Transformer'
        description = 'causal decoder-only Transformer with regression head'
    else:
        config = dict(lookback=LOOKBACK, hidden_size=32, num_layers=1, dropout=0.1)
        config.update(getattr(args, 'model_config', None) or {})
        if config['lookback'] != LOOKBACK:
            raise ValueError('The fixed protocol requires seven-day inputs.')
        model = LSTMRegressor(**config)
        load_checkpoint = load_lstm
        display_name = 'LSTM'
        description = 'unidirectional LSTM with regression head' 
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    rows, best_rmse, stale = [], float('inf'), 0
    checkpoint = output / 'best_model.pt'
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_mse = float(loss_fn(model(xt), yt))
            vp = model(xv)
            validation_mse = float(loss_fn(vp, yv))
        rmse = float(np.sqrt(validation_mse) * scale)
        if not np.isfinite([train_mse, validation_mse, rmse]).all():
            raise RuntimeError('Non-finite loss; inspect data or training settings.')
        rows.append(dict(epoch=epoch, train_mse=train_mse,
                         validation_mse=validation_mse, validation_rmse=rmse))
        if rmse < best_rmse:
            best_rmse, best_epoch, stale = rmse, epoch, 0
            torch.save(dict(model_state=model.state_dict(), model_config=config,
                            scaler=dict(mean=mean, scale=scale,
                                        fit_end=str(daily.index[train_stop - 1].date())),
                            seed=args.seed, best_epoch=epoch,
                            period=dict(start=START, end=END),
                            target='PM2.5-specific AQI'), checkpoint)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}: train MSE={train_mse:.4f}, validation RMSE={rmse:.4f}', flush=True)
        if stale >= args.patience:
            break
    # Use the validation-selected checkpoint, not the last epoch.
    restored, saved = load_checkpoint(checkpoint)
    with torch.inference_mode():
        prediction = restored(xv).numpy() * scale + mean
    prediction_frame = pd.DataFrame({'actual': validation['target'], model_kind: prediction},
                                    index=validation.index)
    prediction_frame.to_csv(output / 'validation_predictions.csv', encoding='utf-8-sig')
    history = pd.DataFrame(rows)
    history.to_csv(output / 'training_history.csv', index=False)
    report = dict(model=description,
                  model_config=config, seed=args.seed, device='cpu',
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  training=dict(max_epochs=args.epochs, epochs_run=len(rows),
                                best_epoch=best_epoch, patience=args.patience,
                                lr=args.lr, batch_size=args.batch_size,
                                optimizer='AdamW', weight_decay=1e-4,
                                gradient_clip=1.0, selection_metric='validation RMSE',
                                deterministic_algorithms=True, threads=args.threads),
                  scaler=saved['scaler'], period=dict(start=START, end=END),
                  data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                  splits={s: dict(samples=int(samples['split'].eq(s).sum()),
                          first_target=str(samples[samples['split'].eq(s)].index.min().date()),
                          last_target=str(samples[samples['split'].eq(s)].index.max().date()))
                          for s in ['train', 'validation', 'test']},
                  validation_metrics=score(validation['target'], prediction),
                  units='MAE/RMSE in AQI points; R2 unitless', test_evaluated=False,
                  runtime=dict(python=platform.python_version(), torch=torch.__version__,
                               numpy=np.__version__, pandas=pd.__version__),
                  elapsed_seconds=time.perf_counter() - started)
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    plot_results(history, prediction_frame, output, display_name)
    print(json.dumps(report, indent=2))
    return report


def main(model_kind):
    parser = argparse.ArgumentParser(description=f'Train {model_kind}; evaluate validation only.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default=f'results/{model_kind}/seed42')
    parser.add_argument('--model-config', type=json.loads, default=None,
                        help='Optional JSON object overriding model dimensions/dropout')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.model_config is not None and not isinstance(args.model_config, dict):
        parser.error('--model-config must be a JSON object.')
    if min(args.epochs, args.patience, args.batch_size, args.threads, args.lr) <= 0:
        parser.error('Training parameters must be positive.')
    run(args, model_kind)
