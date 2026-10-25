"""Draw the FlexScribe icon (no dependencies) and save it as a multi-size Windows .ico.

Design: teal rounded square, white hexagon outline (the hexaflex) with a sound wave inside (the transcription).
Usage: python tools/make_icon.py  ->  static/flexscribe.ico (also served as the page favicon)
"""
import math
import struct
import zlib
from pathlib import Path

TEAL = (47, 111, 98)
WHITE = (255, 255, 255)


def sd_round_rect(x, y, half, r):
    qx, qy = abs(x) - half + r, abs(y) - half + r
    return math.hypot(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0) - r


def sd_hexagon(x, y, r):  # pointy-top regular hexagon, circumradius r
    k = (-0.8660254, 0.5, 0.57735027)
    x, y = abs(y), abs(x)  # rotate so a vertex points up
    d = 2 * min(k[0] * x + k[1] * y, 0)
    x, y = x - d * k[0], y - d * k[1]
    x -= min(max(x, -k[2] * r * 0.8660254), k[2] * r * 0.8660254)
    y -= r * 0.8660254
    return math.hypot(x, y) * (1 if y > 0 else -1)


def sd_capsule_v(x, y, cx, half_h, w):
    dy = max(abs(y) - half_h, 0)
    return math.hypot(x - cx, dy) - w


def shade(u, v, px):
    """Coverage (0-1) of background and white foreground at normalized coords u, v in [-1, 1]."""
    bg = sd_round_rect(u, v, 0.96, 0.30)
    hexagon = abs(sd_hexagon(u, v, 0.70)) - 0.085           # outline
    bars = min(sd_capsule_v(u, v, cx, h, 0.055)
               for cx, h in ((-0.30, 0.10), (-0.15, 0.24), (0.0, 0.36), (0.15, 0.24), (0.30, 0.10)))
    fg = min(hexagon, bars)
    aa = 1.5 * px
    cov = lambda d: min(max(0.5 - d / aa, 0), 1)
    return cov(bg), cov(fg)


def render(size):
    ss = 4 if size < 64 else 2  # supersampling
    px = 2 / (size * ss)
    rows = []
    for j in range(size):
        row = bytearray([0])  # PNG filter: none
        for i in range(size):
            a_bg = a_fg = 0
            for sj in range(ss):
                for si in range(ss):
                    u = ((i + (si + 0.5) / ss) / size) * 2 - 1
                    v = ((j + (sj + 0.5) / ss) / size) * 2 - 1
                    b, f = shade(u, v, px)
                    a_bg += b
                    a_fg += f * b
            a_bg /= ss * ss
            a_fg /= ss * ss
            t = a_fg / a_bg if a_bg else 0
            rgb = [round(TEAL[c] * (1 - t) + WHITE[c] * t) for c in range(3)]
            row += bytes(rgb + [round(a_bg * 255)])
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def write_ico(path, sizes=(16, 24, 32, 48, 64, 128, 256)):
    images = [render(s) for s in sizes]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = b""
    for s, png in zip(sizes, images):
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(png), offset)
        offset += len(png)
    Path(path).write_bytes(header + entries + b"".join(images))
    Path(path).with_suffix(".png").write_bytes(images[-1])  # 256 px preview


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "static" / "flexscribe.ico"
    write_ico(out)
    print("written", out)
