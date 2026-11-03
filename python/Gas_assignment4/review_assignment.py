"""Read-only review of original CSVs and all submitted numeric artifacts."""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / 'artifacts'


def main():
    load = lambda name: json.loads((OUT / name).read_text(encoding='utf-8'))
    manifest, scaler, ranges, results = (load(name) for name in ('split_manifest.json', 'scaler.json', 'range_report.json', 'results.json'))
    raw, scaled = (np.load(OUT / name, mmap_mode='r') for name in ('raw.npy', 'scaled.npy'))
    low, high = np.array(scaler['min']), np.array(scaler['max'])
    span = high - low
    assert raw.shape == scaled.shape == (1000, 3000, 6)
    assert np.all(span > 0)
    assert len(manifest['files']) == 1000
    splits = {name: set(ids) for name, ids in manifest['split'].items()}
    assert [len(splits[k]) for k in ('train', 'validation', 'test')] == [700, 150, 150]
    assert not (splits['train'] & splits['validation'] or splits['train'] & splits['test'] or splits['validation'] & splits['test'])
    assert set.union(*splits.values()) == set(range(1000))
    train_min, train_max = np.full(6, np.inf), np.full(6, -np.inf)
    zero_counts = np.zeros(6, dtype=np.int64)
    out_of_range = {name: np.zeros(6, dtype=np.int64) for name in splits}
    for i, filename in enumerate(manifest['files']):
        path = Path(manifest['data_dir']) / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sha256'][str(i)]
        data = np.loadtxt(path, delimiter=',', encoding='utf-8-sig', ndmin=2)
        assert data.shape == (3000, 7) and np.isfinite(data).all()
        np.testing.assert_array_equal(data[:, 0], np.arange(3000))
        x = data[:, 1:]
        np.testing.assert_array_equal(raw[i], x.astype(np.float32))
        np.testing.assert_array_equal(scaled[i], np.clip((x - low) / span, 0, 1).astype(np.float32))
        zero_counts += (x == 0).sum(0)
        name = next(name for name, ids in splits.items() if i in ids)
        out_of_range[name] += ((x < low) | (x > high)).sum(0)
        if name == 'train':
            train_min, train_max = np.minimum(train_min, x.min(0)), np.maximum(train_max, x.max(0))
        if (i + 1) % 200 == 0:
            print(f'SOURCE/CACHE REVIEW {i + 1}/1000', flush=True)
    np.testing.assert_array_equal(low, train_min)
    np.testing.assert_array_equal(high, train_max)
    for name in splits:
        np.testing.assert_array_equal(out_of_range[name], np.array(ranges[name]['below_train_min']) + ranges[name]['above_train_max'])
    report = (BASE / 'report.md').read_text(encoding='utf-8')
    metric_checks = {}
    first_inputs = None
    for key, horizon in [('3s', 30), ('30s', 300)]:
        saved = np.load(OUT / f'predictions_{key}.npz')
        f, s = saved['file_ids'], saved['starts']
        np.testing.assert_array_equal(f, np.repeat(sorted(splits['test']), 521))
        np.testing.assert_array_equal(s, np.tile(np.arange(0, 2601, 5), 150))
        if first_inputs is None:
            first_inputs = f.copy(), s.copy()
        else:
            np.testing.assert_array_equal(f, first_inputs[0])
            np.testing.assert_array_equal(s, first_inputs[1])
        np.testing.assert_array_equal(saved['truth'], raw[f, s + 99 + horizon])
        for method, predictions in [('cnn', saved['predictions']), ('persistence', saved['persistence'])]:
            error = predictions - saved['truth']
            np.testing.assert_allclose(np.abs(error / span).mean(), results[key][method]['normalized_MAE'], rtol=1e-12)
            np.testing.assert_allclose(np.sqrt(np.square(error / span).mean()), results[key][method]['normalized_RMSE'], rtol=1e-12)
            for i, sensor in enumerate(scaler['sensors']):
                expected = results[key][method]['per_sensor_original_units'][sensor]
                mse = np.square(error[:, i]).mean()
                r2 = 1 - np.square(error[:, i]).sum() / np.square(saved['truth'][:, i] - saved['truth'][:, i].mean()).sum()
                np.testing.assert_allclose([np.abs(error[:, i]).mean(), np.sqrt(mse), r2],
                                           [expected['MAE'], expected['RMSE'], expected['R2']], rtol=1e-12)
        assert f"{results[key]['cnn']['normalized_RMSE']:.6f}" in report
        history = load(f'history_{key}.json')
        assert len(history['loss']) == results[key]['epochs_run'] == 20
        assert np.argmin(history['val_loss']) + 1 == results[key]['best_epoch']
        with zipfile.ZipFile(OUT / f'model_{key}.keras') as z:
            config = json.loads(z.read('config.json'))
        types = [layer['class_name'] for layer in config['config']['layers']]
        assert types == ['InputLayer', 'Conv1D', 'Conv1D', 'Conv1D', 'Dense', 'Dense'], types
        metric_checks[key] = {'all_sensor_metrics_verified': True, 'test_samples': len(f), 'saved_layers': types}
    assert results['3s']['initial_weights_sha256'] == results['30s']['initial_weights_sha256']
    with zipfile.ZipFile(BASE / 'assignment4_submission.zip') as z:
        assert z.testzip() is None
        checksums = json.loads(z.read('artifacts/delivery_checksums.json'))
        for name, digest in checksums.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
            assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
    result = {'verified': True, 'original_CSVs_and_hashes_checked': 1000,
              'raw_and_scaled_cache_match_originals': True, 'training_only_scaler_refit_matches': True,
              'zero_counts': zero_counts.tolist(), 'split_disjoint': True,
              'out_of_range': {k: v.tolist() for k, v in out_of_range.items()},
              'models': metric_checks, 'submission_zip_and_current_files_match': True,
              'requirement_interpretation': 'User clarified Util -> Preprocessing -> Prepatch describes the process, not a mandatory tool.',
              'sampling_interval_evidence': '_Merged_Data_data_set_00000.png x-axis: Timestamp(100 ms), visually checked',
              'limitations': ['No separate physical timestamp column in CSV',
                              'Statistical independence of different experiment files is not established']}
    (OUT / 'requirements_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ASSIGNMENT REVIEW PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
