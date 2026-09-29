#!/usr/bin/env python3
"""Loopback-only exploratory CPU RivaGAN service; never writes pilot inputs/results."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['ORT_DISABLE_ALL'] = '1'
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'service_tree'))
from canonical import canonicalize, Rejected, MAX_ENCODED
import cv2
import numpy as np
from PIL import Image
from imwatermark.rivaGan import RivaWatermark


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bits(idx, wrong):
    return np.random.default_rng((20270927 if wrong else 20260927) + idx).integers(0, 2, 32, dtype=np.uint8)


class Service(BaseHTTPRequestHandler):
    def send_bytes(self, status, body, mime):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/':
            self.send_bytes(200, (HERE / 'ui.html').read_bytes(), 'text/html; charset=utf-8')
        elif path.startswith('/arrays/') and path.count('/') == 2:
            name = path.split('/')[-1]
            if not name.endswith(('.rgb', '.f32')) or len(name.split('.')[0]) != 64 or not all(c in '0123456789abcdef' for c in name.split('.')[0]):
                self.send_error(404)
                return
            file = HERE / 'arrays' / name
            if not file.is_file():
                self.send_error(404)
                return
            self.send_bytes(200, file.read_bytes(), 'application/octet-stream')
        elif path.startswith('/artifacts/') and path.count('/') == 2:
            name = path.split('/')[-1]
            if not name.endswith(('.png', '.jpg')) or not all(c in '0123456789abcdef.' or c in 'pngjr' for c in name):
                self.send_error(404)
                return
            file = HERE / 'artifacts' / name
            if not file.is_file():
                self.send_error(404)
                return
            self.send_bytes(200, file.read_bytes(), 'image/png' if name.endswith('.png') else 'image/jpeg')
        else:
            self.send_error(404)

    def do_POST(self):
        begin = time.perf_counter_ns()
        query = parse_qs(urlsplit(self.path).query)
        if urlsplit(self.path).path != '/api/evaluate':
            self.send_error(404)
            return
        n = int(self.headers.get('Content-Length', '-1'))
        # Do not read oversized uploads into memory.
        if n < 0 or n > MAX_ENCODED:
            row = {'status': 'rejected', 'error': 'encoded_limit', 'declared_content_length': n,
                   'body_read': False, 'elapsed_ms': (time.perf_counter_ns()-begin)/1e6,
                   'request_id': str(time.time_ns()) + '_limit'}
            (HERE / 'requests').mkdir(exist_ok=True)
            (HERE / 'requests' / (row['request_id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
            self.send_bytes(413, json.dumps(row).encode(), 'application/json')
            return
        blob = self.rfile.read(n)
        mode = query.get('path', [''])[0]
        key = query.get('key', ['own'])[0]
        try:
            idx = int(query.get('idx', [''])[0])
            if idx < 0 or idx > 100000 or mode not in ('raw', 'canonical') or key not in ('own', 'wrong'):
                raise ValueError('invalid_parameter')
            if mode == 'raw':
                if not blob.startswith(b'\xff\xd8'):
                    raise Rejected('raw_requires_jpeg')
                with Image.open(io.BytesIO(blob)) as im:
                    if im.format != 'JPEG' or im.mode != 'RGB' or 'icc_profile' in im.info or getattr(im, 'n_frames', 1) != 1:
                        raise Rejected('unsupported_jpeg_color_or_frames')
                    w, h = im.size
                    if not 1 <= w <= 1024 or not 1 <= h <= 1024 or w*h > 1024*1024:
                        raise Rejected('dimensions_limit')
                    # Deliberately ignore EXIF orientation, matching v1 verifier policy.
                    rgb = np.asarray(im.convert('RGB')).copy()
                rendered = blob
                extension = 'jpg'
            else:
                rendered, raster, (w, h) = canonicalize(blob)
                rgb = np.frombuffer(raster, dtype=np.uint8).reshape(h, w, 3)
                extension = 'png'
            decode_ms = (time.perf_counter_ns()-begin)/1e6
            bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
            tensor = (np.asarray([bgr], dtype=np.float32)/127.5-1).transpose(3, 0, 1, 2)[None, :, :, :, :]
            tensor = np.ascontiguousarray(tensor)
            output = np.asarray(RivaWatermark.decoder.run(None, {'frame': tensor})[0][0]).ravel()
            decoded = (output > 0.52).astype(np.uint8)
            expected = bits(idx, key == 'wrong')
            score = float((decoded == expected).mean())
            model_ms = (time.perf_counter_ns()-begin)/1e6-decode_ms
            artifact_name = sha(rendered) + '.' + extension
            artifact = HERE / 'artifacts' / artifact_name
            artifact.parent.mkdir(exist_ok=True)
            artifact.write_bytes(rendered)
            raster_bytes = np.ascontiguousarray(rgb).tobytes()
            tensor_bytes = tensor.tobytes()
            raster_name = sha(raster_bytes) + '.rgb'
            tensor_name = sha(tensor_bytes) + '.f32'
            (HERE / 'arrays').mkdir(exist_ok=True)
            (HERE / 'arrays' / raster_name).write_bytes(raster_bytes)
            (HERE / 'arrays' / tensor_name).write_bytes(tensor_bytes)
            row = {'status': 'ok', 'model_execution': 'live_cpu_onnx', 'path': mode, 'key': key, 'idx': idx,
                   'source_sha256': sha(blob), 'returned_file_sha256': sha(rendered), 'artifact_url': '/artifacts/' + artifact_name,
                   'raster_sha256': sha(raster_bytes), 'raster_url': '/arrays/'+raster_name, 'raster_shape': list(rgb.shape),
                   'raster_dtype': 'uint8', 'raster_order': 'HWC RGB',
                   'tensor_sha256': sha(tensor_bytes), 'tensor_url': '/arrays/'+tensor_name, 'tensor_shape': list(tensor.shape), 'tensor_dtype': str(tensor.dtype),
                   'tensor_order': 'BGR [1,3,1,H,W] float32 /127.5-1',
                   'raw_output': output.tolist(), 'decoded_bits': decoded.tolist(), 'expected_bits': expected.tolist(),
                   'score': score, 'decode_ms': decode_ms, 'model_ms': model_ms,
                   'elapsed_ms': (time.perf_counter_ns()-begin)/1e6}
            status = 200
        except (ValueError, OSError, SyntaxError, Rejected) as e:
            row = {'status': 'rejected', 'error': str(e) if isinstance(e, Rejected) else 'decode_failure_or_invalid_parameter',
                   'path': mode, 'idx': query.get('idx', [''])[0], 'key': key,
                   'source_sha256': sha(blob), 'elapsed_ms': (time.perf_counter_ns()-begin)/1e6}
            status = 422
        request_id = f'{time.time_ns()}_{sha(blob)[:12]}'
        row['request_id'] = request_id
        (HERE / 'requests').mkdir(exist_ok=True)
        (HERE / 'requests' / (request_id + '.json')).write_text(json.dumps(row, indent=2) + '\n')
        self.send_bytes(status, json.dumps(row).encode(), 'application/json')


def main():
    RivaWatermark.loadModel()
    providers = RivaWatermark.decoder.get_providers()
    if providers != ['CPUExecutionProvider']:
        raise RuntimeError(f'CPU-only provider required: {providers}')
    print(json.dumps({'ready': True, 'providers': providers, 'address': '127.0.0.1:8765'}), flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8765), Service).serve_forever()


if __name__ == '__main__':
    main()
