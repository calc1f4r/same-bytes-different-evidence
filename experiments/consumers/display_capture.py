#!/usr/bin/env python3
"""v3 browser probe: pinned Chromium canvas + UI screenshot vs scored v3 EXIF and ICC/gamma fixtures."""
import base64, hashlib, io, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parents[2]
PILOT = BASE/'raster_study/pilot_v3'
OUT = BASE/'raster_study/browser_probe/v3'; OUT.mkdir(exist_ok=True)
BINARY = Path.home()/'.cache/ms-playwright/chromium-1234/chrome-linux64/chrome'
FAMS = ['riva_gan', 'tree_ring_rand', 'tree_ring_ring', 'dwt_dct_svd']

def h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def probe(page, family, idx, path, ref, tag):
    data = path.read_bytes()
    uri = 'data:image/' + ('jpeg' if path.suffix == '.jpg' else 'png') + ';base64,' + base64.b64encode(data).decode()
    page.set_content('<html><head><style>html,body{margin:0;background:white}img{display:block;width:256px;height:256px;object-fit:fill}</style></head><body><img id="r"></body></html>')
    page.locator('img').evaluate('(e,src)=>{e.src=src}', uri)
    page.locator('img').evaluate('async e=>await e.decode()')
    pixel = page.evaluate('''() => {let i=document.querySelector('img'), c=document.createElement('canvas');c.width=i.naturalWidth;c.height=i.naturalHeight;let x=c.getContext('2d',{willReadFrequently:true,colorSpace:'srgb'});x.drawImage(i,0,0);return {w:c.width,h:c.height,p:Array.from(x.getImageData(0,0,c.width,c.height).data)}}''')
    canvas = np.asarray(pixel['p'], dtype=np.uint8).reshape(pixel['h'], pixel['w'], 4)[:, :, :3]
    shot = page.locator('img').screenshot()
    with Image.open(io.BytesIO(shot)) as im:
        shot_rgb = np.asarray(im.convert('RGB')).copy()
        if shot_rgb.shape != canvas.shape:
            shot_rgb = np.asarray(im.convert('RGB').resize((canvas.shape[1], canvas.shape[0]))).copy()
    return {'family': family, 'idx': idx, 'tag': tag, 'file_sha256': hashlib.sha256(data).hexdigest(),
            'chromium_canvas_rgb_hash': h(canvas), 'chromium_screenshot_rgb_hash': h(shot_rgb),
            'scored_rgb_hash': ref['rgb_hash'],
            'canvas_equal_scored': h(canvas) == ref['rgb_hash'],
            'screenshot_equal_scored': h(shot_rgb) == ref['rgb_hash'],
            'browser_version': ver,
            'css': 'margin:0;background:white;img display:block;width:256px;height:256px;object-fit:fill;force-color-profile=srgb'}

rows = []
with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path=str(BINARY), headless=True,
                                 args=['--no-sandbox', '--disable-gpu', '--force-color-profile=srgb'])
    page = browser.new_page(viewport={'width': 320, 'height': 320}, device_scale_factor=1, color_scheme='light')
    ver = browser.version
    for family in FAMS:
        f = PILOT/'results'/f'{family}.jsonl'
        if not f.exists(): continue
        score = [json.loads(x) for x in f.read_text().splitlines()]
        # EXIF display rows + color managed rows
        exif_rows = {x['idx']: x for x in score if x['split'] == 'evaluation' and x['kind'] == 'exif_display'}
        color_rows = [x for x in score if x['split'] == 'evaluation' and x['kind'].startswith('color_') and x['kind'].endswith('_pillow_managed')]
        for idx, ref in sorted(exif_rows.items()):
            path = PILOT/'fixtures'/f'{family}_{idx}_exif6.jpg'
            if not path.exists(): continue
            rows.append(probe(page, family, idx, path, ref, 'exif6'))
        for ref in color_rows:
            axis = ref['axis']
            path = PILOT/'fixtures'/f'{family}_{ref["idx"]}_{axis}.png'
            if not path.exists(): continue
            rows.append(probe(page, family, ref['idx'], path, ref, axis))
    browser.close()

(OUT/'results.json').write_text(json.dumps({'browser_version': rows[0]['browser_version'] if rows else None,
                                            'binary_sha256': hashlib.sha256(BINARY.read_bytes()).hexdigest(),
                                            'rows': rows}, indent=1))
ok_c = sum(r['canvas_equal_scored'] for r in rows); ok_s = sum(r['screenshot_equal_scored'] for r in rows)
print(f'canvas==scored {ok_c}/{len(rows)} | screenshot==scored {ok_s}/{len(rows)}')
for r in rows:
    if not r['canvas_equal_scored']:
        print('MISMATCH', r['family'], r['idx'], r['tag'])
