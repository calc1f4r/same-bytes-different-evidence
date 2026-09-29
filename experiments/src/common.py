"""Common utilities: deterministic seed derivation, image IO, manifests, prompts.

Canonical image format everywhere: numpy uint8 array, shape (H, W, 3), RGB.
All randomness flows through derive_seed() so every artifact is reproducible
from config.json plus the script names.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"


def load_config() -> dict:
    with open(ROOT / "config.json") as f:
        return json.load(f)


def config_sha256() -> str:
    return hashlib.sha256((ROOT / "config.json").read_bytes()).hexdigest()


def derive_seed(*parts) -> int:
    """Stable seed derivation: blake2b over the labeled parts.

    Parts are stringified with their type tag so (1, 'a') != ('1a',).
    """
    mat = "|".join(f"{type(p).__name__}:{p}" for p in parts)
    h = hashlib.blake2b(mat.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")


def rng_for(*parts) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(derive_seed(*parts)))


def bits_to_hex(bits: list[int] | np.ndarray) -> str:
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    pad = (-len(bits)) % 8
    padded = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    bytes_ = np.packbits(padded).tobytes()
    return bytes_.hex()


def hex_to_bits(hex_str: str, n_bits: int) -> np.ndarray:
    bytes_ = bytes.fromhex(hex_str)
    bits = np.unpackbits(np.frombuffer(bytes_, dtype=np.uint8))
    return bits[:n_bits].copy()


def random_payload(n_bits: int, *seed_parts) -> np.ndarray:
    return rng_for("payload", *seed_parts).integers(0, 2, size=n_bits).astype(np.uint8)


def bit_accuracy(recovered: np.ndarray | list[int] | None, truth: np.ndarray) -> float:
    if recovered is None:
        return 0.0
    r = np.asarray(recovered, dtype=np.uint8).ravel()
    t = np.asarray(truth, dtype=np.uint8).ravel()
    if r.shape != t.shape:
        return 0.0
    return float((r == t).mean())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_png(img: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, format="PNG", compress_level=6)


def load_png(path: Path) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8).copy()


class Manifest:
    """Append-only JSONL manifest with a schema version."""

    def __init__(self, path: Path, schema: str):
        self.path = path
        self.schema = schema
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict) -> None:
        row = {"schema": self.schema, **row}
        with open(self.path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def read(self) -> list[dict]:
        rows = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def exists(self) -> bool:
        return self.path.exists()


# 64 prompts, fixed order, one per lineage index. Varied subjects/styles so
# failure analysis by content type is possible later. Order is frozen: prompt
# index i in family F always uses PROMPTS[i].
PROMPTS = [
    "a mountain lake at sunrise, mirror reflection, photo",
    "a red bicycle leaning on a brick wall, photo",
    "a plate of sushi on a wooden table, photo",
    "an old lighthouse on a cliff, stormy sea, photo",
    "a tabby cat sleeping on a windowsill, photo",
    "a field of sunflowers under a blue sky, photo",
    "a modern glass skyscraper seen from below, photo",
    "a cup of cappuccino with latte art, photo",
    "a hiking trail through a pine forest, photo",
    "a vintage typewriter on a desk, photo",
    "a hot air balloon over a desert canyon, photo",
    "a bowl of ramen with steam, photo",
    "a snowy street with warm streetlights at dusk, photo",
    "a sailboat on calm water at noon, photo",
    "a bouquet of tulips in a vase, photo",
    "a chessboard mid-game, close-up, photo",
    "a waterfall in a tropical jungle, photo",
    "a stack of old books in a library, photo",
    "a sandy beach with palm trees, aerial view, photo",
    "a violin resting on sheet music, photo",
    "a farmer's market vegetable stall, photo",
    "a northern lights sky over a frozen lake, photo",
    "a sports car on a racetrack, motion blur, photo",
    "a rooftop garden with potted plants, photo",
    "a wooden bridge over a stream in autumn, photo",
    "a ceramic teapot pouring tea, photo",
    "a city skyline at night, long exposure, photo",
    "a sheep pasture on rolling green hills, photo",
    "a science laboratory with glassware, photo",
    "a street musician playing guitar, photo",
    "a lavender field in Provence, photo",
    "a bowl of fresh strawberries, photo",
    "a windmill on a dutch plain, photo",
    "a campfire under a starry sky, photo",
    "an artist's studio with paintings, photo",
    "a train station platform with a steam train, photo",
    "a coral reef with tropical fish, underwater photo",
    "a bakery window with fresh bread, photo",
    "a zen rock garden with raked gravel, photo",
    "a basketball court at golden hour, photo",
    "a foggy harbor with fishing boats, photo",
    "a greenhouse full of tomato plants, photo",
    "a medieval castle on a hill, photo",
    "a dog catching a frisbee mid-air, photo",
    "a flower cart on a cobblestone street, photo",
    "an airplane wing above the clouds, photo",
    "a milk jar and cookies on a kitchen counter, photo",
    "a rice terrace in the mountains, photo",
    "a violin maker's workshop, wood shavings, photo",
    "a winter cabin with smoke from the chimney, photo",
    "a cactus garden at midday, photo",
    "a row of colorful houses on a canal, photo",
    "a meteor streaking over a mountain ridge, photo",
    "a potter shaping clay on a wheel, photo",
    "a vineyard rows at harvest time, photo",
    "a duck pond with lily pads, photo",
    "a motorcycle parked on a coastal road, photo",
    "a spice market with sacks of colored spices, photo",
    "a library reading lamp and open atlas, photo",
    "a koi pond in a japanese garden, photo",
    "a telescope dome under the milky way, photo",
    "a food truck serving tacos at night, photo",
    "a swing hanging from an oak tree, photo",
    "a scooter parked by a cafe in the rain, photo",
]
assert len(PROMPTS) == 64
