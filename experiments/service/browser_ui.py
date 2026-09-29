#!/usr/bin/env python3
"""Real pinned Chromium UI interaction, returned-image screenshot and sRGB canvas hashes."""
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PILOT=ROOT/'raster_study/pilot_v1'
BINARY=Path.home()/'.cache/ms-playwright/chromium-1234/chrome-linux64/chrome'
assert BINARY.exists()
rows=json.loads((HERE/'evaluation/http_rows.json').read_text())
reference={(r['idx'],r['path']):r for r in rows if r.get('label') in ('exif_original','exif_canonical')}
out=HERE/'evaluation/screenshots';out.mkdir(parents=True,exist_ok=True)
results=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path=str(BINARY),headless=True,args=['--no-sandbox','--disable-gpu','--force-color-profile=srgb'])
    page=browser.new_page(viewport={'width':1000,'height':950},device_scale_factor=1,color_scheme='light')
    page.goto('http://127.0.0.1:8765/',wait_until='networkidle')
    for idx in range(46,52):
        fixture=PILOT/'fixtures'/f'riva_gan_{idx}_exif6.jpg'
        for path in ('raw','canonical'):
            page.locator('#file').set_input_files(str(fixture))
            page.locator('#idx').fill(str(idx))
            page.locator('#path').select_option(path)
            page.locator('#result').evaluate('(el) => el.textContent = ""')
            page.locator('#run').click()
            page.wait_for_function('document.querySelector("#result").textContent.includes("request_id")')
            payload=json.loads(page.locator('#result').inner_text())
            assert payload['idx']==idx and payload['path']==path and payload['status']=='ok'
            assert payload['request_id'] != (results[-1]['request_id'] if results else None)
            expected=reference[(idx,path)]
            assert payload['score']==expected['score'] and payload['raster_sha256']==expected['raster_sha256']
            assert page.locator('#display').get_attribute('src')==payload['artifact_url']
            assert page.locator('#display').is_visible()
            assert ('canonical PNG' if path=='canonical' else 'original JPEG') in page.locator('#state').inner_text()
            canvas=page.evaluate('''() => {const image=document.querySelector('#display');const c=document.createElement('canvas');c.width=image.naturalWidth;c.height=image.naturalHeight;let ctx=c.getContext('2d',{willReadFrequently:true,colorSpace:'srgb'});ctx.drawImage(image,0,0);return {width:c.width,height:c.height,rgba:Array.from(ctx.getImageData(0,0,c.width,c.height).data)};}''')
            rgba=bytes(canvas['rgba'])
            rgb=bytes(rgba[i] for i in range(len(rgba)) if i%4!=3)
            hash_rgb=hashlib.sha256(rgb).hexdigest()
            # Chromium displays EXIF-aware image even for raw pathway; both displayed rasters equal canonical tensor.
            canonical=reference[(idx,'canonical')]['raster_sha256']
            assert hash_rgb==canonical
            screenshot=out/f'{idx}_{path}.png'
            page.screenshot(path=str(screenshot),full_page=True)
            results.append({'idx':idx,'path':path,'request_id':payload['request_id'],
                'returned_url':payload['artifact_url'],'returned_file_sha256':payload['returned_file_sha256'],
                'service_raster_sha256':payload['raster_sha256'],'ui_canvas_rgb_sha256':hash_rgb,
                'ui_canvas_matches_canonical_raster':hash_rgb==canonical,
                'canvas_width':canvas['width'],'canvas_height':canvas['height'],
                'screenshot':str(screenshot.relative_to(HERE)),
                'screenshot_sha256':hashlib.sha256(screenshot.read_bytes()).hexdigest()})
    errors=[]
    browser_version=browser.version
    browser.close()
result={'status':'actual UI file upload -> service HTTP -> returned artifact displayed in pinned Chromium; screenshot and sRGB canvas captured',
        'browser_version':browser_version,'browser_binary':str(BINARY),
        'browser_binary_sha256':hashlib.sha256(BINARY.read_bytes()).hexdigest(),
        'flags':['--no-sandbox','--disable-gpu','--force-color-profile=srgb'],
        'viewport':{'width':1000,'height':950},'device_scale_factor':1,'canvas_color_space':'srgb',
        'rows':results,'canvas_matches_canonical':sum(x['ui_canvas_matches_canonical_raster'] for x in results)}
(HERE/'evaluation/browser_ui.json').write_text(json.dumps(result,indent=2)+'\n')
print('Chromium',browser_version,'UI cases',len(results),'canvas matches canonical',result['canvas_matches_canonical'])
