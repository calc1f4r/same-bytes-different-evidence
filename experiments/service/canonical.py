"""Narrow deterministic Pillow ingestion profile; not a universal standards decoder.

Only 8-bit RGB/RGBA non-animated PNG and RGB JPEG are accepted. PNG color
management is rejected, not silently interpreted. No resizing is performed.
"""
from __future__ import annotations

import io
import struct
import zlib
from PIL import Image, ImageOps

MAX_ENCODED = 8 * 1024 * 1024
MAX_SIDE = 1024
MAX_PIXELS = 1024 * 1024
PNG_SIG = b"\x89PNG\r\n\x1a\n"


class Rejected(ValueError):
    pass


def _png_chunks(blob: bytes) -> list[bytes]:
    if not blob.startswith(PNG_SIG):
        raise Rejected("not_png")
    pos, names = 8, []
    while pos + 12 <= len(blob):
        length = struct.unpack_from(">I", blob, pos)[0]
        if length > MAX_ENCODED or pos + 12 + length > len(blob):
            raise Rejected("invalid_png_chunk")
        name = blob[pos + 4:pos + 8]
        body = blob[pos + 8:pos + 8 + length]
        crc = struct.unpack_from(">I", blob, pos + 8 + length)[0]
        if zlib.crc32(name + body) & 0xffffffff != crc:
            raise Rejected("invalid_png_crc")
        names.append(name)
        pos += length + 12
        if name == b"IEND":
            if pos != len(blob):
                raise Rejected("trailing_png_data")
            return names
    raise Rejected("missing_png_end")


def canonicalize(blob: bytes) -> tuple[bytes, bytes, tuple[int, int]]:
    """Return (RGB PNG encoding, exact contiguous HWC RGB uint8 raster, (w,h))."""
    if len(blob) > MAX_ENCODED:
        raise Rejected("encoded_limit")
    if blob.startswith(PNG_SIG):
        names = _png_chunks(blob)
        if any(n in names for n in (b"iCCP", b"gAMA", b"cHRM", b"sRGB", b"eXIf", b"acTL")):
            raise Rejected("unsupported_png_semantics")
        expected = "PNG"
    elif blob.startswith(b"\xff\xd8"):
        expected = "JPEG"
    else:
        raise Rejected("unsupported_format")
    try:
        with Image.open(io.BytesIO(blob)) as im:
            if im.format != expected or getattr(im, "n_frames", 1) != 1:
                raise Rejected("unsupported_frames_or_format")
            w, h = im.size
            if w < 1 or h < 1 or w > MAX_SIDE or h > MAX_SIDE or w * h > MAX_PIXELS:
                raise Rejected("dimensions_limit")
            if expected == "JPEG" and ("icc_profile" in im.info or im.mode != "RGB"):
                raise Rejected("unsupported_jpeg_color")
            if expected == "PNG" and im.mode not in ("RGB", "RGBA"):
                raise Rejected("unsupported_png_mode")
            im.load()
            if expected == "JPEG":
                # Apply JPEG EXIF orientation once, then discard source metadata.
                im = ImageOps.exif_transpose(im)
            if im.mode == "RGBA":
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(bg, im).convert("RGB")
            else:
                im = im.convert("RGB")
            raw = im.tobytes("raw", "RGB")
            output = io.BytesIO()
            im.save(output, format="PNG")
            return output.getvalue(), raw, im.size
    except (OSError, SyntaxError, ValueError) as exc:
        if isinstance(exc, Rejected):
            raise
        raise Rejected("decode_failure") from exc
