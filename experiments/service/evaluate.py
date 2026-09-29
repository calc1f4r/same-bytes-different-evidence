#!/usr/bin/env python3
"""Run real HTTP requests against server.py; pilot inputs and score rows are read-only."""
import hashlib
import io
import json
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PILOT = ROOT / 'raster_study/pilot_v1'
OUT = HERE / 'evaluation'
OUT.mkdir(exist_ok=True)
CONTROL = HERE / 'controls'
CONTROL.mkdir(exist_ok=True)

def sha(b): return hashlib.sha256(b).hexdigest()

def send(data, idx, mode, key='own', label=''):
    url = f'http://127.0.0.1:8765/api/evaluate?path={mode}&idx={idx}&key={key}'
    start = time.perf_counter_ns()
    req = urllib.request.Request(url, data=data, headers={'Content-Type':'application/octet-stream'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            code, row = r.status, json.load(r)
    except urllib.error.HTTPError as e:
        code, row = e.code, json.load(e)
    client_ms = (time.perf_counter_ns()-start)/1e6
    assert row['source_sha256'] == sha(data)
    assert (code == 200) == (row['status'] == 'ok')
    if code == 200:
        for url_field, hash_field in [('artifact_url','returned_file_sha256'),
                                      ('raster_url','raster_sha256'),('tensor_url','tensor_sha256')]:
            with urllib.request.urlopen('http://127.0.0.1:8765'+row[url_field]) as r:
                assert sha(r.read()) == row[hash_field]
    row['http_status'] = code
    row['client_ms'] = client_ms
    row['label'] = label
    return row

scores = [json.loads(x) for x in (PILOT/'results/riva_gan.jsonl').read_text().splitlines()]
threshold = json.loads((PILOT/'results/riva_gan_calibration.json').read_text())['threshold']
cal = [r for r in scores if r['split'] == 'calibration']
values = sorted({r['score'] for r in cal})
mid = [(x+y)/2 for x,y in zip(values[:-1], values[1:])]
neg = [r['score'] for r in cal if r['kind'] in ('unmarked','wrong_key')]
pos = [r['score'] for r in cal if r['kind'] == 'marked']
literal_threshold = max(v for v in mid if all(n<=v for n in neg) and any(p>v for p in pos))
assert threshold == 0.75 and literal_threshold == 0.984375
rows = []
for idx in range(46,52):
    fixture = PILOT/'fixtures'/f'riva_gan_{idx}_exif6.jpg'
    blob = fixture.read_bytes()
    assert sha(blob) == next(r for r in scores if r['idx']==idx and r['kind']=='exif_verifier_raw')['file_sha256']
    raw = send(blob,idx,'raw',label='exif_original')
    canonical = send(blob,idx,'canonical',label='exif_canonical')
    rows.extend([raw,canonical])
    for result, kind in [(raw,'exif_verifier_raw'),(canonical,'baked_visible')]:
        old = next(r for r in scores if r['idx']==idx and r['kind']==kind)
        assert result['score'] == old['score'] and result['tensor_sha256'] == old['tensor_hash'] and result['raster_sha256'] == old['rgb_hash']
        assert result['raw_output'] == old['raw_output'] and result['decoded_bits'] == old['decoded_bits']
    assert raw['returned_file_sha256']==raw['source_sha256']
    assert canonical['raster_sha256'] != raw['raster_sha256']
    # Same source, two distinct key controls on fresh identity JPEG and one clean unmarked JPEG.
    marked = PILOT/'fixtures'/f'riva_gan_{idx}_marked.png'
    clean = ROOT/'bench/data/clean/sd21_base'/f'sd21_base_{idx:03d}.png'
    for source, label in [(marked,'marked'),(clean,'unmarked')]:
        out=CONTROL/f'{label}_{idx}.jpg'
        with Image.open(source) as im:
            im.convert('RGB').save(out,format='JPEG',quality=95,subsampling=0)
        data=out.read_bytes()
        for mode in ('raw','canonical'):
            result=send(data,idx,mode,label=label)
            rows.append(result)
            if label=='marked':
                rows.append(send(data,idx,mode,key='wrong',label='wrong_key'))
    # Reingestion of returned PNG is encoded + raster idempotent (and scores repeat).
    png = (HERE/'artifacts'/canonical['artifact_url'].split('/')[-1]).read_bytes()
    once = send(png,idx,'canonical',label='canonical_reingest')
    twice = send(png,idx,'canonical',label='canonical_repeat')
    rows.extend([once,twice])
    for r in (once,twice):
        assert r['returned_file_sha256']==canonical['returned_file_sha256']
        assert r['raster_sha256']==canonical['raster_sha256']
        assert r['tensor_sha256']==canonical['tensor_sha256'] and r['score']==canonical['score']
    print(idx,'raw',raw['score'],'canonical',canonical['score'],flush=True)

# Invalid, unsupported, and resource policy are counted as rejects, not successful detection.
for blob,mode,label in [(b'not jpeg','raw','invalid_raw'),(b'not jpeg','canonical','unsupported_format'),
                        (b'\xff\xd8garbage','canonical','invalid_jpeg'),(b'x'*(8*1024*1024+1),'canonical','oversized')]:
    # The 413 path rejects the body before reading; it intentionally has no persisted row.
    if label == 'oversized':
        import http.client
        conn=http.client.HTTPConnection('127.0.0.1',8765,timeout=30)
        # Header-only limit check: server intentionally does not consume oversized body.
        conn.request('POST','/api/evaluate?path=canonical&idx=46',body=None,
                     headers={'Content-Length':str(len(blob))})
        res=conn.getresponse()
        assert res.status==413
        rows.append({'label':label,'status':'rejected','http_status':413,'response':json.loads(res.read()),
                     'declared_content_length':len(blob),'body_sent':False})
        conn.close()
    else:
        r=send(blob,46,mode,label=label)
        assert r['status']=='rejected'
        rows.append(r)

for r in rows:
    if r['status']=='ok':
        r['accept_implemented_deviation']=r['score']>threshold
        r['accept_literal_frozen_sensitivity']=r['score']>literal_threshold
(OUT/'http_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
selected=lambda label,mode: [r for r in rows if r.get('label')==label and r.get('path')==mode and r['status']=='ok']
summary={'status':'exploratory live CPU model evaluation; v1 implementation threshold materially deviates from frozen protocol',
 'threshold_executed_deviation':threshold,'threshold_literal_frozen_sensitivity':literal_threshold,
 'number_http_rows':len(rows),'number_ok':sum(r['status']=='ok' for r in rows),
 'number_rejected':sum(r['status']=='rejected' for r in rows), 'heldout_indices':list(range(46,52)), 'groups':{}}
for label in ('exif_original','exif_canonical','marked','unmarked','wrong_key'):
    for mode in ('raw','canonical'):
        group=selected(label,mode)
        if not group: continue
        summary['groups'][f'{label}_{mode}']={'n':len(group),'accepted_implemented':sum(r['accept_implemented_deviation'] for r in group),
           'accepted_literal_sensitivity':sum(r['accept_literal_frozen_sensitivity'] for r in group),
           'mean_score':statistics.mean(r['score'] for r in group),
           'mean_service_ms':statistics.mean(r['elapsed_ms'] for r in group),
           'median_service_ms':statistics.median(r['elapsed_ms'] for r in group),
           'mean_client_ms':statistics.mean(r['client_ms'] for r in group)}
summary['encoded_and_raster_idempotence']='6/6 repeated canonical PNGs exact; scores and tensors exact'
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
