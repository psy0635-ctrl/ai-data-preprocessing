"""Verify saved artifacts, reproducibility, inference and the final report."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
from pathlib import Path
import json
import subprocess
import ast
import re
import hashlib
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
import pymupdf
from Iris_assignment3 import CONDITIONS, build_model, dataset

ROOT=Path(__file__).resolve().parent
d=json.loads((ROOT/'iris_results/training_history.json').read_text(encoding='utf-8'))
repeat=json.loads((ROOT/'tmp/iris_recheck/training_history.json').read_text(encoding='utf-8'))
assert len(d['runs'])==5 and d['epochs']==100
assert d['runs'][0]['histories']==repeat['runs'][0]['histories'], 'Non-reproducible curves'
iris=load_iris(); seed=d['seeds'][0]; run=d['runs'][0]
x=((iris.data[run['validation_indices']]-run['scaler_mean'])/run['scaler_scale']).astype('float32')
y=iris.target[run['validation_indices']]
order_reference=None
for name,(aug,drop) in CONDITIONS.items():
    h=run['histories'][name]
    model=tf.keras.models.load_model(ROOT/f'iris_results/{name}.keras')
    pred=model(x,training=False).numpy()
    acc=float((pred.argmax(axis=1)==y).mean())
    loss=float(-np.log(pred[np.arange(len(y)),y]).mean())
    np.testing.assert_allclose(acc,h['val_accuracy'][-1],atol=1e-6)
    np.testing.assert_allclose(loss,h['val_loss'][-1],atol=1e-6)
    tf.keras.utils.set_random_seed(seed)
    fresh=build_model(seed,aug,drop)
    if aug or drop: _=build_model(seed)
    ordered=[batch_y.numpy().tolist() for _,batch_y in dataset(x,y,seed)]
    if order_reference is None:order_reference=ordered
    else:assert ordered==order_reference
    for layer in fresh.layers:
        if isinstance(layer,tf.keras.layers.GaussianNoise):
            np.testing.assert_array_equal(layer(x,training=False).numpy(),x)
            assert not np.array_equal(layer(x,training=True).numpy(),x)
        if isinstance(layer,tf.keras.layers.Dropout):
            ones=np.ones((30,32),dtype='float32')
            np.testing.assert_array_equal(layer(ones,training=False).numpy(),ones)
            assert (layer(ones,training=True).numpy()==0).any()

paths=subprocess.check_output(['rg','--files','--hidden','--no-ignore',
    '-g','!**/.git/**','-g','!**/.venv/**','-g','!**/__pycache__/**',
    '-g','*.py','-g','*.ipynb','D:/dlProject'],encoding='utf-8').splitlines()
scan=json.loads((ROOT/'project_scan.json').read_text(encoding='utf-8'))
original=[]; matches=[]
for path in paths:
    p=Path(path)
    text=p.read_text(encoding='utf-8-sig')
    if p.suffix=='.py':ast.parse(text)
    if p.parent==ROOT and p.name in ('Iris_assignment3.py','make_iris_report.py','verify_iris.py'):continue
    if 'form_revision' in p.parts:continue
    original.append(str(p))
    if re.search(r'iris|붓꽃|setosa|versicolor|virginica',text,re.I):matches.append(str(p))
assert not matches, matches
scan['complete_source_scan']={'paths':original,'count':len(original),'existing_iris_matches':matches,
    'includes_gitignored_sources':True}
scan['additional_review']['original_python_file_count']=len(original)
(ROOT/'project_scan.json').write_text(json.dumps(scan,ensure_ascii=False,indent=2),encoding='utf-8')

pdf=next((ROOT/'과제').glob('*.pdf'))
with pymupdf.open(pdf) as doc:
    assert len(doc)==6
    text=''.join(p.get_text() for p in doc)
    for expected in ['2601110288','박수용','93.33%','92.00%','GaussianNoise','Dropout']:
        assert expected in text,expected
    assert sum(len(p.get_images()) for p in doc)==6
    assert all(len(p.get_text())>100 for p in doc)
    for page in doc:
        for block in page.get_text('dict')['blocks']:
            if block['type']!=0:continue
            for line in block['lines']:
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    assert x0>=0 and y0>=0 and x1<=page.rect.width+1 and y1<=page.rect.height+1
hashes={}
for name in ['Flower.py','Flower2.py']:
    p=ROOT/name
    baseline=subprocess.check_output(['git','-C','D:/dlProject','show',f'HEAD:python/Flower/{name}'])
    assert p.read_bytes().replace(b'\r\n',b'\n')==baseline.replace(b'\r\n',b'\n')
    hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
result={'status':'PASS','training_runs':20,'epochs_per_run':100,'primary_seed_rerun':'All four full histories match exactly',
        'saved_models':'Loaded predictions match report metrics','inference_modes':'Noise and dropout inactive',
        'batch_order':'Same for all four model conditions','existing_source_files_checked':len(original),
        'pdf_pages':6,'pdf_embedded_graphs':6,'pdf_text_and_bounds':'PASS',
        'original_files_unchanged_sha256':hashes}
(ROOT/'iris_results/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
