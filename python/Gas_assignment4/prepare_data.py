"""File split, training-only Min-Max fit, range audit, and compact caches."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DATA = Path('/workspace/DataSet_00000_00999/_Merged_Data')
OUT = BASE / 'artifacts'
SENSORS = ['Temp1', 'Press1', 'Temp2', 'Press2', 'Acc', 'Gas']


def read_csv(path):
    # No header: skiprows stays zero. Equivalent to pandas header=None.
    a = np.loadtxt(path, delimiter=',', encoding='utf-8-sig', ndmin=2)
    if a.shape != (3000, 7) or not np.isfinite(a).all():
        raise ValueError(f'Invalid numeric CSV: {path}, shape={a.shape}')
    if not np.array_equal(a[:, 0], np.arange(3000)):
        raise ValueError(f'Unexpected index: {path}')
    return a[:, 1:]


def main():
    OUT.mkdir(exist_ok=True)
    files = sorted(p for p in DATA.rglob('*') if p.is_file() and p.suffix.lower() == '.csv')
    if len(files) != 1000:
        raise ValueError(f'Expected 1000 files, got {len(files)}')
    order = np.random.default_rng(42).permutation(1000)
    split = {k: sorted(v.tolist()) for k, v in {
        'train': order[:700], 'validation': order[700:850], 'test': order[850:]
    }.items()}
    assert len(set().union(*(set(v) for v in split.values()))) == 1000
    assert sum(len(v) for v in split.values()) == 1000
    low, high = np.full(6, np.inf), np.full(6, -np.inf)
    print('Fitting bounds on 700 TRAIN files only...', flush=True)
    for i in split['train']:
        x = read_csv(files[i])
        low, high = np.minimum(low, x.min(0)), np.maximum(high, x.max(0))
    span = high - low
    if np.any(span == 0):
        raise ValueError(f'Constant training column requires review: {np.array(SENSORS)[span == 0]}')
    scaler = {'sensors': SENSORS, 'min': low.tolist(), 'max': high.tolist(),
              'fit_split': 'train', 'policy': 'clip scaled inputs and training/validation labels to [0,1]; evaluate against original unclipped test truth'}
    raw = np.lib.format.open_memmap(OUT / 'raw.npy', mode='w+', dtype='float32', shape=(1000, 3000, 6))
    scaled = np.lib.format.open_memmap(OUT / 'scaled.npy', mode='w+', dtype='float32', shape=(1000, 3000, 6))
    report, hashes = {}, {}
    for name, ids in split.items():
        below, above, zeros = np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64)
        observed_min, observed_max = np.full(6, np.inf), np.full(6, -np.inf)
        for i in ids:
            x = read_csv(files[i])
            h = hashlib.sha256(files[i].read_bytes()).hexdigest()
            hashes.setdefault(h, []).append((name, i))
            below += (x < low).sum(0)
            above += (x > high).sum(0)
            zeros += (x == 0).sum(0)
            observed_min, observed_max = np.minimum(observed_min, x.min(0)), np.maximum(observed_max, x.max(0))
            raw[i] = x
            scaled[i] = np.clip((x - low) / span, 0, 1)
        report[name] = {'files': len(ids), 'rows': len(ids) * 3000,
                        'below_train_min': below.tolist(), 'above_train_max': above.tolist(),
                        'outside_percent': ((below + above) / (len(ids) * 3000) * 100).tolist(),
                        'raw_min': observed_min.tolist(), 'raw_max': observed_max.tolist(),
                        'zero_counts': zeros.tolist()}
        print(name, json.dumps(report[name]), flush=True)
    raw.flush()
    scaled.flush()
    cross = [v for v in hashes.values() if len({name for name, _ in v}) > 1]
    if cross:
        raise ValueError(f'Exact duplicate contents across splits: {cross[:5]}')
    assert float(scaled.min()) >= 0 and float(scaled.max()) <= 1
    manifest = {'data_dir': str(DATA), 'seed': 42, 'split': split,
                'files': [str(p.relative_to(DATA)) for p in files],
                'sha256': {str(i): h for h, items in hashes.items() for _, i in items},
                'exact_duplicate_groups': sum(len(v) > 1 for v in hashes.values()),
                'cross_split_exact_duplicates': 0}
    for name, obj in [('split_manifest', manifest), ('scaler', scaler), ('range_report', report)]:
        (OUT / f'{name}.json').write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('PASS: splits disjoint; training-only bounds; cache range [0,1]; no cross-split exact duplicates.', flush=True)
    print('Training bounds:', json.dumps(scaler), flush=True)


if __name__ == '__main__':
    main()
