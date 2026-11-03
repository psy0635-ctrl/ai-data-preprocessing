"""Package the real report, code, models, metrics and audit evidence, without data caches."""
import hashlib
import json
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'artifacts'


def main():
    audit = json.loads((OUT / 'independent_audit.json').read_text(encoding='utf-8'))
    assert set(audit) == {'3s', '30s'} and all(v['passed'] for v in audit.values())
    required = ['report.md', 'report.html', 'README.md', 'figures/learning_curves.png',
                'figures/test_comparison.png', 'figures/sensor_rmse.png',
                'figures/forecast_3s.png', 'figures/forecast_30s.png']
    for name in required:
        assert (BASE / name).is_file(), name
    paths = sorted(set([BASE / name for name in required]
                       + list(BASE.glob('*.py'))
                       + [p for p in OUT.iterdir() if p.suffix in ('.json', '.jsonl', '.log', '.csv', '.keras', '.npz')
                          and p.name != 'delivery_checksums.json']))
    native_paths = BASE / 'submission' / 'document_paths.json'
    if native_paths.is_file():
        native = json.loads(native_paths.read_text(encoding='utf-8'))
        native_name = native['output_hwp'].replace('\\', '/').rsplit('/', 1)[-1]
        native_report = BASE / 'submission' / native_name
        if native_report.is_file():
            paths.append(native_report)
    checksums = {str(p.relative_to(BASE)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    integrity = OUT / 'delivery_checksums.json'
    integrity.write_text(json.dumps(checksums, ensure_ascii=False, indent=2), encoding='utf-8')
    destination = BASE / 'assignment4_submission.zip'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in paths + [integrity]:
            z.write(path, arcname=str(path.relative_to(BASE)))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        for name, digest in checksums.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    print('PACKAGE PASS:', destination, 'bytes=', destination.stat().st_size, 'files=', len(paths) + 1)


if __name__ == '__main__':
    main()
