"""Verify indexing, train matched CNN/Dense models, evaluate untouched test files."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
import numpy as np
import tensorflow as tf

BASE = Path(__file__).resolve().parent
OUT = BASE / 'artifacts'
SENSORS = ['Temp1', 'Press1', 'Temp2', 'Press2', 'Acc', 'Gas']
LOOKBACK, MAX_HORIZON, STRIDE, SEED = 100, 300, 5, 42
STARTS = np.arange(0, 3000 - LOOKBACK - MAX_HORIZON + 1, STRIDE, dtype=np.int32)


def save_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def build_model():
    # Only convolution and Dense layers; no pooling, Flatten, RNN, or other feature layers.
    # SAME padding retains the last observations: 100 -> 50 -> 25 -> 1.
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(100, 6)),
        tf.keras.layers.Conv1D(32, 5, strides=2, padding='same', activation='relu', name='conv_1'),
        tf.keras.layers.Conv1D(32, 5, strides=2, padding='same', activation='relu', name='conv_2'),
        tf.keras.layers.Conv1D(32, 25, activation='relu', name='conv_full_window'),
        tf.keras.layers.Dense(32, activation='relu', name='dense_hidden'),
        tf.keras.layers.Dense(6, activation='sigmoid', name='six_sensors'),
    ], name='cnn_dense_forecaster')
    assert all(isinstance(layer, (tf.keras.layers.Conv1D, tf.keras.layers.Dense)) for layer in model.layers)
    assert model.output_shape == (None, 1, 6)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def make_dataset(cache, ids, horizon, batch_size, training):
    with tf.device('/CPU:0'):
        file_ids = tf.constant(ids, dtype=tf.int32)
        starts = tf.constant(STARTS)
    n = len(ids) * len(STARTS)
    ds = tf.data.Dataset.range(n, output_type=tf.int32)
    if training:
        ds = ds.shuffle(min(n, 32768), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)

    def gather(tokens):
        with tf.device('/CPU:0'):
            f = tf.gather(file_ids, tokens // len(STARTS))
            s = tf.gather(starts, tokens % len(STARTS))
            rows = s[:, None] + tf.range(LOOKBACK)[None, :]
            fs = tf.broadcast_to(f[:, None], tf.shape(rows))
            x = tf.gather_nd(cache, tf.stack([fs, rows], axis=-1))
            target = s + LOOKBACK - 1 + horizon
            y = tf.gather_nd(cache, tf.stack([f, target], axis=-1))[:, None, :]
            x = tf.ensure_shape(x, [None, 100, 6])
            y = tf.ensure_shape(y, [None, 1, 6])
            return x, y

    ds = ds.map(gather, num_parallel_calls=2, deterministic=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    options.experimental_deterministic = True
    return ds.with_options(options).prefetch(2)


def verify(cache, manifest, batch_size):
    # Exact small, synthetic fixture: both file and row are encoded in the values.
    toy = np.arange(30 * 6).reshape(30, 6)
    n, length, horizons = 30, 4, (2, 7)
    starts = np.arange(n - length - max(horizons) + 1)
    assert len(starts) == 20
    for h in horizons:
        x = np.stack([toy[s:s + length] for s in starts])
        y = np.stack([toy[s + length - 1 + h] for s in starts])
        assert x.shape == (20, 4, 6) and y.shape == (20, 6)
        np.testing.assert_array_equal(x[0], toy[0:4])
        np.testing.assert_array_equal(y[0], toy[3 + h])
        np.testing.assert_array_equal(x[-1], toy[19:23])
        np.testing.assert_array_equal(y[-1], toy[22 + h])
    example = []
    ids = manifest['split']['train']
    for h in (30, 300):
        ds = make_dataset(cache, ids[:2], h, batch_size, False)
        tokens = 0
        for x, y in ds:
            x, y = x.numpy(), y.numpy()
            for j in sorted(set([0, len(x) - 1])):
                token = tokens + j
                f = ids[token // len(STARTS)]
                s = int(STARTS[token % len(STARTS)])
                np.testing.assert_array_equal(x[j], cache[f, s:s + LOOKBACK].numpy())
                np.testing.assert_array_equal(y[j, 0], cache[f, s + LOOKBACK - 1 + h].numpy())
            tokens += len(x)
        assert tokens == 2 * len(STARTS)
        example.append({'horizon_rows': h, 'first_X_rows': [0, 99], 'first_y_row': 99 + h,
                        'last_X_rows': [2600, 2699], 'last_y_row': 2699 + h})
    result = {'passed': True, 'synthetic_common_samples': 20,
              'lookback': LOOKBACK, 'stride': STRIDE,
              'per_file_samples': len(STARTS), 'first_last_examples': example,
              'split_samples': {k: len(v) * len(STARTS) for k, v in manifest['split'].items()},
              'X_batch_shape': ['batch', 100, 6], 'y_batch_shape': ['batch', 1, 6],
              'note': 'The singleton time axis permits CNN/Dense layers only.'}
    save_json('window_verification.json', result)
    print('WINDOW VERIFICATION PASS', json.dumps(result), flush=True)


def metric_summary(truth, pred, span):
    error = np.asarray(pred, dtype=np.float64) - truth
    sse = np.square(error).sum(axis=0)
    centered = np.square(truth - truth.mean(axis=0)).sum(axis=0)
    per_sensor = {}
    for i, sensor in enumerate(SENSORS):
        per_sensor[sensor] = {'MAE': float(np.abs(error[:, i]).mean()),
                              'RMSE': float(np.sqrt(np.square(error[:, i]).mean())),
                              'R2': float(1 - sse[i] / centered[i]) if centered[i] else None}
    return {'normalized_MAE': float(np.mean(np.abs(error) / span)),
            'normalized_RMSE': float(np.sqrt(np.mean(np.square(error / span)))),
            'per_sensor_original_units': per_sensor}


def evaluate(model, ds, raw, ids, horizon, low, span):
    pred = model.predict(ds, verbose=0)[:, 0, :].astype(np.float64) * span + low
    f = np.repeat(np.array(ids), len(STARTS))
    s = np.tile(STARTS, len(ids))
    truth = raw[f, s + LOOKBACK - 1 + horizon].astype(np.float64)
    persistence = raw[f, s + LOOKBACK - 1].astype(np.float64)
    assert pred.shape == truth.shape == (len(ids) * len(STARTS), 6)
    assert np.isfinite(pred).all()
    result = {'samples': len(truth), 'cnn': metric_summary(truth, pred, span),
              'persistence': metric_summary(truth, persistence, span)}
    np.savez_compressed(OUT / f'predictions_{horizon // 10}s.npz',
                        predictions=pred, truth=truth, persistence=persistence, file_ids=f, starts=s)
    return result


class EpochLog(tf.keras.callbacks.Callback):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def on_epoch_begin(self, epoch, logs=None):
        self.started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        line = {'horizon_seconds': self.horizon // 10, 'epoch': epoch + 1,
                'seconds': round(time.perf_counter() - self.started, 3),
                **{k: float(v) for k, v in logs.items()}}
        with (OUT / 'epochs.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(line) + '\n')
        print('EPOCH', json.dumps(line), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(SEED)
    manifest = json.loads((OUT / 'split_manifest.json').read_text())
    scaler = json.loads((OUT / 'scaler.json').read_text())
    low, high = np.array(scaler['min']), np.array(scaler['max'])
    span = high - low
    scaled = np.load(OUT / 'scaled.npy', mmap_mode='r')
    raw = np.load(OUT / 'raw.npy', mmap_mode='r')
    with tf.device('/CPU:0'):
        cache = tf.constant(scaled)
    verify(cache, manifest, args.batch_size)
    model = build_model()
    model.summary(print_fn=lambda line: print(line, flush=True))
    x, y = next(iter(make_dataset(cache, manifest['split']['train'][:1], 30, 4, False)))
    smoke = model.train_on_batch(x, y)
    assert np.isfinite(np.array(smoke)).all()
    device = model(x).device
    if gpus:
        assert 'GPU' in device, device
    save_json('smoke_test.json', {'passed': True, 'train_on_batch': [float(v) for v in smoke],
                                 'output_device': device, 'parameters': model.count_params(),
                                 'layers': [layer.__class__.__name__ for layer in model.layers]})
    print('MODEL/GPU SMOKE PASS', device, flush=True)
    if args.verify_only:
        return
    config = {'seed': SEED, 'lookback': LOOKBACK, 'sampling_seconds_assumed': 0.1,
              'horizon_rows': [30, 300], 'stride': STRIDE, 'epochs': args.epochs,
              'batch_size': args.batch_size, 'optimizer': 'Adam', 'learning_rate': 0.001,
              'loss': 'MSE over six scaled sensors', 'checkpoint_selection': 'minimum validation MSE',
              'shuffle_buffer': 32768, 'deterministic_ops': True,
              'python': sys.version, 'tensorflow': tf.__version__, 'numpy': np.__version__,
              'platform': platform.platform(), 'GPU': [str(gpu) for gpu in gpus],
              'build_info': tf.sysconfig.get_build_info()}
    save_json('experiment_config.json', config)
    results, initial_hash = {}, None
    for h in (30, 300):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        model = build_model()
        digest = hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()
        if initial_hash is None:
            initial_hash = digest
        assert digest == initial_hash, 'Initial weights must be identical.'
        train_ds = make_dataset(cache, manifest['split']['train'], h, args.batch_size, True)
        val_ds = make_dataset(cache, manifest['split']['validation'], h, args.batch_size, False)
        checkpoint = OUT / f'model_{h // 10}s.keras'
        start = time.perf_counter()
        print(f'TRAIN START horizon={h // 10}s, identical_initial_weights={digest}', flush=True)
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=0,
                            callbacks=[tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_loss', save_best_only=True), EpochLog(h)])
        train_seconds = time.perf_counter() - start
        history = {k: [float(x) for x in values] for k, values in history.history.items()}
        save_json(f'history_{h // 10}s.json', history)
        model = tf.keras.models.load_model(checkpoint)
        test_ds = make_dataset(cache, manifest['split']['test'], h, args.batch_size, False)
        metrics = evaluate(model, test_ds, raw, manifest['split']['test'], h, low, span)
        metrics.update({'horizon_seconds': h // 10, 'training_seconds': train_seconds,
                        'parameters': model.count_params(), 'initial_weights_sha256': digest,
                        'epochs_run': len(history['loss']),
                        'best_epoch': int(np.argmin(history['val_loss']) + 1),
                        'best_validation_MSE': float(min(history['val_loss']))})
        results[f'{h // 10}s'] = metrics
        save_json('results.json', results)
        print('TEST RESULT', json.dumps(metrics), flush=True)
    print('COMPLETE: two matched models trained and test-evaluated.', flush=True)


if __name__ == '__main__':
    main()
