"""Independent audit of saved predictions, checkpoint reload, and source row alignment."""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

BASE = Path(__file__).resolve().parent
OUT = BASE / 'artifacts'


def main():
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    get = lambda name: json.loads((OUT / name).read_text(encoding='utf-8'))
    results, manifest, scaler = get('results.json'), get('split_manifest.json'), get('scaler.json')
    low, span = np.array(scaler['min']), np.array(scaler['max']) - np.array(scaler['min'])
    raw = np.load(OUT / 'raw.npy', mmap_mode='r')
    scaled = np.load(OUT / 'scaled.npy', mmap_mode='r')
    previous = None
    audit = {}
    for key, h in [('3s', 30), ('30s', 300)]:
        saved = np.load(OUT / f'predictions_{key}.npz')
        f, starts = saved['file_ids'], saved['starts']
        assert len(f) == 78150 and len(set(f.tolist())) == 150
        assert set(f.tolist()) == set(manifest['split']['test'])
        assert np.all(starts >= 0) and np.all(starts + 99 + h < 3000)
        if previous is not None:
            np.testing.assert_array_equal(f, previous[0])
            np.testing.assert_array_equal(starts, previous[1])
        previous = f.copy(), starts.copy()
        np.testing.assert_array_equal(saved['truth'], raw[f, starts + 99 + h].astype(np.float64))
        np.testing.assert_array_equal(saved['persistence'], raw[f, starts + 99].astype(np.float64))
        for method, array in [('cnn', saved['predictions']), ('persistence', saved['persistence'])]:
            diff = (array - saved['truth']) / span
            rmse = float(np.linalg.norm(diff) / np.sqrt(diff.size))
            np.testing.assert_allclose(rmse, results[key][method]['normalized_RMSE'], rtol=1e-12)
        # Match the original inference batches and graph execution. cuDNN may
        # choose a different numeric algorithm for a four-item eager batch.
        positions = np.array([0, 520, 521, len(f) - 1])
        model = tf.keras.models.load_model(OUT / f'model_{key}.keras')
        assert [type(layer).__name__ for layer in model.layers] == ['Conv1D'] * 3 + ['Dense'] * 2
        max_delta = 0.0
        for batch_id in sorted(set((positions // 512).tolist())):
            begin, end = batch_id * 512, min((batch_id + 1) * 512, len(f))
            x = np.stack([scaled[f[i], starts[i]:starts[i] + 100] for i in range(begin, end)])
            pred_scaled = model.predict(x, batch_size=len(x), verbose=0)[:, 0, :]
            expected_scaled = (saved['predictions'][begin:end] - low) / span
            max_delta = max(max_delta, float(np.abs(pred_scaled - expected_scaled).max()))
            np.testing.assert_allclose(pred_scaled, expected_scaled, rtol=2e-5, atol=2e-6)
        history = get(f'history_{key}.json')
        assert len(history['loss']) == results[key]['epochs_run'] == 20
        assert np.argmin(history['val_loss']) + 1 == results[key]['best_epoch']
        audit[key] = {'passed': True, 'source_target_rows_checked': len(f),
                      'same_test_inputs': True, 'checkpoint_reloaded': True,
                      'checkpoint_max_scaled_difference_same_batch': max_delta,
                      'metrics_recomputed': True, 'layer_restriction_verified': True}
    (OUT / 'independent_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('INDEPENDENT AUDIT PASS', json.dumps(audit))


if __name__ == '__main__':
    main()
