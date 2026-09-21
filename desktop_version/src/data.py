"""MNIST 데이터셋 다운로드 및 로딩.

원본 IDX 형식을 직접 파싱한다. 외부 데이터셋 라이브러리를 쓰지 않는 것이
이 프로젝트의 목적이다.
"""
from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# torchvision이 사용하는 미러. yann.lecun.com 원본은 접근이 불안정하다.
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(force: bool = False) -> None:
    """네 개의 MNIST 아카이브를 data/ 에 내려받는다. 이미 있으면 건너뛴다."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected_md5 in FILES.values():
        dest = DATA_DIR / name
        if dest.exists() and not force and _md5(dest) == expected_md5:
            print(f"  [skip] {name}")
            continue
        last_error: Exception | None = None
        for mirror in MIRRORS:
            url = mirror + name
            try:
                print(f"  [get ] {name} <- {mirror}")
                urllib.request.urlopen(url, timeout=60)
                urllib.request.urlretrieve(url, dest)
                actual = _md5(dest)
                if actual != expected_md5:
                    raise ValueError(f"MD5 불일치: {actual} != {expected_md5}")
                break
            except Exception as exc:  # 다음 미러로 넘어간다
                last_error = exc
                print(f"         실패: {exc}")
        else:
            raise RuntimeError(f"{name} 다운로드 실패") from last_error


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic == 2051:  # 이미지
            rows, cols = struct.unpack(">II", f.read(8))
            buf = f.read(count * rows * cols)
            return np.frombuffer(buf, dtype=np.uint8).reshape(count, rows, cols)
        if magic == 2049:  # 레이블
            buf = f.read(count)
            return np.frombuffer(buf, dtype=np.uint8)
        raise ValueError(f"알 수 없는 IDX magic: {magic}")


def load(split: str) -> tuple[np.ndarray, np.ndarray]:
    """('train' | 'test') -> (X, y).

    X: float32, (N, 784), 0.0~1.0 범위로 정규화.
    y: int64, (N,)
    """
    if split not in ("train", "test"):
        raise ValueError(f"split은 'train' 또는 'test'여야 합니다: {split}")
    images = _read_idx(DATA_DIR / FILES[f"{split}_images"][0])
    labels = _read_idx(DATA_DIR / FILES[f"{split}_labels"][0])
    X = images.reshape(len(images), -1).astype(np.float32) / 255.0
    y = labels.astype(np.int64)
    return X, y


if __name__ == "__main__":
    download()
    for split in ("train", "test"):
        X, y = load(split)
        print(f"{split}: X={X.shape} {X.dtype}  y={y.shape}  범위=[{X.min()}, {X.max()}]")
