from hashlib import sha256
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

SENSORS = [f'sensor_{i}' for i in range(1, 22)]
SETTINGS = [f'setting_{i}' for i in range(1, 4)]
COLUMNS = ['engine', 'cycle', *SETTINGS, *SENSORS]
MAX_BYTES = 10 * 1024 * 1024


def numeric_table(raw):
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError('Each file must be nonempty and at most 10 MiB.')
    try:
        table = pd.read_csv(StringIO(raw.decode('utf-8-sig')), sep=r'\s+', header=None, nrows=50001).astype(float)
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        raise ValueError('Expected whitespace-separated numeric NASA text without a header.') from error
    if len(table) > 50000 or not np.isfinite(table.to_numpy()).all():
        raise ValueError('Files must have at most 50,000 rows and only finite numbers.')
    if (np.abs(table.to_numpy()) > 1e9).any():
        raise ValueError('A numeric value exceeds the supported range.')
    return table


def parse_cycles(raw):
    frame = numeric_table(raw)
    if frame.shape[1] != 26:
        raise ValueError('Expected 26 columns: engine, cycle, 3 settings, 21 sensors.')
    frame.columns = COLUMNS
    for column in ('engine', 'cycle'):
        values = frame[column]
        if ((values < 1) | (values > 1000) | (values % 1 != 0)).any():
            raise ValueError('Engine IDs and cycles must be integers from 1 to 1000.')
        frame[column] = values.astype(int)
    if frame.duplicated(['engine', 'cycle']).any():
        raise ValueError('Duplicate engine/cycle rows are not allowed.')
    frame = frame.sort_values(['engine', 'cycle']).reset_index(drop=True)
    ids = frame.engine.unique()
    if len(ids) > 300 or not np.array_equal(ids, np.arange(1, len(ids) + 1)):
        raise ValueError('Use at most 300 consecutive engine IDs, starting at 1.')
    for _, engine in frame.groupby('engine'):
        if not np.array_equal(engine.cycle.to_numpy(), np.arange(1, len(engine) + 1)):
            raise ValueError('Each engine needs consecutive cycles starting at 1.')
    return frame


def parse_dataset(train_raw, test_raw, rul_raw):
    train, test = parse_cycles(train_raw), parse_cycles(test_raw)
    if train.engine.nunique() < 5 or train.groupby('engine').size().min() < 5:
        raise ValueError('Training requires at least 5 engines with 5 cycles each.')
    if train.equals(test):
        raise ValueError('Training and test files must be different.')
    labels = numeric_table(rul_raw)
    if labels.shape != (test.engine.nunique(), 1):
        raise ValueError('RUL needs one value per test engine, in engine ID order.')
    truth = labels.iloc[:, 0].to_numpy()
    if ((truth < 0) | (truth > 1000) | (truth % 1 != 0)).any():
        raise ValueError('RUL labels must be integer cycles from 0 to 1000.')
    return {'train': train, 'test': test, 'truth': truth,
            'fingerprints': {name: sha256(raw).hexdigest() for name, raw in [('train', train_raw), ('test', test_raw), ('rul', rul_raw)]}}


def features(frame):
    # Neither engine identity nor future failure time is a predictor.
    result = frame[['cycle', *SETTINGS, *SENSORS]].astype(float).copy()
    grouped = frame.groupby('engine', sort=False)
    for sensor in SENSORS:
        result[f'{sensor}_mean5'] = grouped[sensor].transform(lambda s: s.rolling(5, min_periods=1).mean())
        result[f'{sensor}_std5'] = grouped[sensor].transform(lambda s: s.rolling(5, min_periods=1).std(ddof=0))
        result[f'{sensor}_delta'] = grouped[sensor].diff().fillna(0)
    return result


def targets(frame):
    return frame.groupby('engine').cycle.transform('max') - frame.cycle


def validation_split(frame):
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(np.sort(frame.engine.unique()))
    count = max(1, int(np.ceil(len(shuffled) * .2)))
    val_ids, fit_ids = shuffled[:count], shuffled[count:]
    snapshots = []
    for engine_id in sorted(val_ids):
        rows = frame[frame.engine == engine_id]
        position = min(len(rows) - 2, max(0, int(len(rows) * rng.uniform(.5, .9)) - 1))
        snapshots.append(int(rows.index[position]))
    return frame.index[frame.engine.isin(fit_ids)].to_numpy(), np.array(snapshots), fit_ids, val_ids


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    error = prediction - truth
    penalty = np.where(error < 0, np.expm1(-error / 13), np.expm1(error / 10))
    return {'mae': float(mean_absolute_error(truth, prediction)),
            'rmse': float(np.sqrt(mean_squared_error(truth, prediction))),
            'nasa_score': float(penalty.sum()), 'n_engines': len(truth)}


def model_for(name, trees):
    estimator = DummyRegressor(strategy='mean') if name == 'Mean baseline' else XGBRegressor(
        n_estimators=trees, max_depth=4, learning_rate=.05, subsample=.9, colsample_bytree=.9,
        objective='reg:squarederror', tree_method='hist', random_state=42, n_jobs=2, importance_type='gain')
    return Pipeline([('varying', VarianceThreshold()), ('model', estimator)])


def predict(model, x):
    return np.clip(model.predict(x), 0, 1000)


def train_models(dataset, progress=lambda message: None, trees=180):
    frame = dataset['train']
    progress('Engineering causal five-cycle sensor features.')
    x, y = features(frame), targets(frame)
    fit_idx, val_idx, fit_ids, val_ids = validation_split(frame)
    scores, fitted = {}, {}
    for name in ('Mean baseline', 'XGBoost'):
        progress(f'Validating {name} on separate engines.')
        model = model_for(name, trees).fit(x.loc[fit_idx], y.loc[fit_idx])
        scores[name] = metrics(y.loc[val_idx], predict(model, x.loc[val_idx]))
    selected = min(scores, key=lambda name: scores[name]['rmse'])
    for name in scores:
        progress(f'Refitting {name} on all training engines; test labels remain unused.')
        fitted[name] = model_for(name, trees).fit(x, y)
    xgb = fitted['XGBoost']
    names = x.columns[xgb.named_steps['varying'].get_support()]
    importance = sorted([{'feature': name, 'gain': float(value)} for name, value in zip(names, xgb.named_steps['model'].feature_importances_)], key=lambda row: row['gain'], reverse=True)[:12]
    last = dataset['test'].groupby('engine', sort=True).tail(1)
    test_x = features(dataset['test']).loc[last.index]
    prediction = predict(fitted[selected], test_x)
    fleet = [{'engine': int(row.engine), 'observed_cycles': int(row.cycle), 'rul': float(value)} for row, value in zip(last.itertuples(), prediction)]
    report = {'selected_model': selected, 'validation': scores, 'feature_importance': importance,
              'feature_importance_type': 'Normalized XGBoost split gain, not causal explanation; shown even if baseline wins.',
              'split': {'training_engines': sorted(map(int, fit_ids)), 'validation_engines': sorted(map(int, val_ids)), 'validation_snapshot_cycles': frame.loc[val_idx, ['engine', 'cycle']].to_dict('records')},
              'protocol': 'Seed 42; 80/20 engine split; one 50–90%-life snapshot per validation engine; select by validation RMSE; refit on all train engines.',
              'target': 'Uncapped RUL = last training cycle minus current cycle; predictions clipped to [0,1000].',
              'fingerprints': dataset['fingerprints'], 'seed': 42, 'trees': trees,
              'test_evaluation': None, 'predictions': fleet}
    return {'models': fitted, 'report': report}


def evaluate_test(dataset, result):
    last = dataset['test'].groupby('engine', sort=True).tail(1)
    x, truth = features(dataset['test']).loc[last.index], dataset['truth']
    scores = {name: metrics(truth, predict(model, x)) for name, model in result['models'].items()}
    prediction = predict(result['models'][result['report']['selected_model']], x)
    return {'metrics': scores, 'points': [{'engine': int(engine), 'actual': float(actual), 'predicted': float(pred)} for engine, actual, pred in zip(last.engine, truth, prediction)]}
