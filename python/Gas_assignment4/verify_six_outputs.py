"""Confirm that each saved horizon model predicts all six sensors in one call."""
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
    manifest = json.loads((OUT / 'split_manifest.json').read_text())
    scaler = json.loads((OUT / 'scaler.json').read_text())
    low, span = np.array(scaler['min']), np.array(scaler['max']) - np.array(scaler['min'])
    scaled = np.load(OUT / 'scaled.npy', mmap_mode='r')
    raw = np.load(OUT / 'raw.npy', mmap_mode='r')
    ids = manifest['split']['test']
    assert len(scaler['sensors']) == 6
    report = {'requirement': 'One model inference must return all six sensors together.',
              'sensor_order': scaler['sensors'], 'models': {}}
    for key, h in [('3s', 30), ('30s', 300)]:
        model = tf.keras.models.load_model(OUT / f'model_{key}.keras')
        assert model.input_shape == (None, 100, 6)
        assert model.output_shape == (None, 1, 6)
        assert isinstance(model.layers[-1], tf.keras.layers.Dense)
        assert model.layers[-1].units == 6
        kernel, bias = model.layers[-1].get_weights()
        assert kernel.shape == (32, 6) and bias.shape == (6,)
        assert model.loss == 'mse'
        # One history window, one forward call, six predictions together.
        file_id = ids[0]
        x = scaled[file_id, :100][None, ...]
        prediction = model(x, training=False).numpy()
        assert x.shape == (1, 100, 6)
        assert prediction.shape == (1, 1, 6) and np.isfinite(prediction).all()
        truth = raw[file_id, 99 + h][None, None, :]
        assert truth.shape == prediction.shape
        original_prediction = prediction[0, 0].astype(np.float64) * span + low
        # Match the original batch configuration when comparing saved inference.
        starts = np.arange(0, 2560, 5)
        batch = np.stack([scaled[file_id, s:s + 100] for s in starts])
        batched_prediction = model.predict(batch, batch_size=512, verbose=0)[:, 0, :]
        saved = np.load(OUT / f'predictions_{key}.npz')
        assert saved['predictions'].shape == saved['truth'].shape == (78150, 6)
        expected = (saved['predictions'][:512] - low) / span
        np.testing.assert_allclose(batched_prediction, expected, rtol=2e-5, atol=2e-6)
        result = {'passed': True, 'input_shape': list(x.shape), 'single_call_output_shape': list(prediction.shape),
                  'output_units': model.layers[-1].units, 'training_loss': model.loss,
                  'test_prediction_shape': list(saved['predictions'].shape),
                  'test_file': manifest['files'][file_id], 'input_rows': [0, 99], 'target_row': 99 + h,
                  'single_call_original_unit_predictions': dict(zip(scaler['sensors'], original_prediction.tolist())),
                  'original_unit_truth': dict(zip(scaler['sensors'], truth[0, 0].astype(float).tolist())),
                  'saved_batch_predictions_match': True}
        report['models'][key] = result
        print(key, json.dumps(result), flush=True)
    report['passed'] = True
    (OUT / 'six_output_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SIX SENSOR JOINT PREDICTION PASS', flush=True)


if __name__ == '__main__':
    main()
