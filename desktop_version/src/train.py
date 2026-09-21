"""MNIST MLP 학습 스크립트.

사용법:
    python -m src.train                 # 기본 설정으로 학습
    python -m src.train --epochs 40     # 에폭 수 변경
    python -m src.train --no-augment    # 증강 없이
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import data
from .model import MLP

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL = MODEL_DIR / "mlp.npz"

PAD = 2  # 증강에서 허용하는 최대 이동 픽셀


def _shift_index_maps(pad: int = PAD) -> np.ndarray:
    """28x28을 상하좌우로 옮길 때 쓸 gather 인덱스를 미리 만든다.

    이미지를 (28+2p)x(28+2p)로 0-패딩한 뒤 창을 옮겨 잘라내는 방식이라
    가장자리가 자연스럽게 0으로 채워진다.
    반환: (이동 가짓수, 784) — 패딩된 평면 배열에 대한 인덱스.
    """
    side = 28 + 2 * pad
    maps = []
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            rows = np.arange(28) + pad - dy
            cols = np.arange(28) + pad - dx
            maps.append((rows[:, None] * side + cols[None, :]).reshape(-1))
    return np.array(maps, dtype=np.int64)


class Augmenter:
    """작은 평행이동으로 데이터를 늘린다.

    사용자가 마우스로 그린 숫자는 MNIST만큼 정확히 중앙에 오지 않으므로,
    이동에 강인하게 만들어 두면 실제 입력에서 체감 정확도가 올라간다.
    """

    def __init__(self, pad: int = PAD, seed: int = 0):
        self.pad = pad
        self.side = 28 + 2 * pad
        self.maps = _shift_index_maps(pad)
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        n = len(batch)
        padded = np.zeros((n, self.side, self.side), dtype=batch.dtype)
        padded[:, self.pad : self.pad + 28, self.pad : self.pad + 28] = batch.reshape(n, 28, 28)
        flat = padded.reshape(n, -1)
        choice = self.rng.integers(0, len(self.maps), size=n)
        return flat[np.arange(n)[:, None], self.maps[choice]]


def evaluate(model: MLP, X: np.ndarray, y: np.ndarray, batch: int = 1000) -> tuple[float, float]:
    """(정확도, 평균 손실)을 계산한다. 메모리를 아끼려 나눠서 처리한다."""
    correct = 0
    loss_sum = 0.0
    for i in range(0, len(X), batch):
        xb, yb = X[i : i + batch], y[i : i + batch]
        probs, _ = model.forward(xb)
        correct += int((probs.argmax(axis=1) == yb).sum())
        loss_sum += float(-np.log(probs[np.arange(len(yb)), yb] + 1e-12).sum())
    return correct / len(X), loss_sum / len(X)


def train(
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    lr_decay: float = 0.92,
    weight_decay: float = 1e-5,
    augment: bool = True,
    val_size: int = 5000,
    seed: int = 42,
    out: Path = DEFAULT_MODEL,
) -> MLP:
    print("데이터 준비 중...")
    data.download()
    X_all, y_all = data.load("train")
    X_test, y_test = data.load("test")

    # 학습/검증 분할. 검증 정확도가 가장 높은 시점의 가중치를 남긴다.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_all))
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    print(f"  학습 {len(X_train)}  검증 {len(X_val)}  테스트 {len(X_test)}")

    model = MLP(sizes=[784, 128, 64, 10], seed=seed)
    augmenter = Augmenter(seed=seed) if augment else None
    print(f"  구조 {' -> '.join(map(str, model.sizes))}  증강 {'켬' if augment else '끔'}")

    best_val = -1.0
    best_weights: tuple[list[np.ndarray], list[np.ndarray]] | None = None
    n = len(X_train)
    started = time.time()

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        epoch_loss = 0.0
        batches = 0
        current_lr = lr * (lr_decay ** (epoch - 1))

        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            if augmenter is not None:
                xb = augmenter(xb)
            loss, gW, gb = model.loss_and_grads(xb, yb)
            model.adam_step(gW, gb, lr=current_lr, weight_decay=weight_decay)
            epoch_loss += loss
            batches += 1

        val_acc, val_loss = evaluate(model, X_val, y_val)
        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            best_weights = ([w.copy() for w in model.W], [b.copy() for b in model.b])
            marker = "  <- 최고"
        print(
            f"  에폭 {epoch:2d}/{epochs}  lr={current_lr:.5f}  "
            f"학습손실={epoch_loss / batches:.4f}  "
            f"검증손실={val_loss:.4f}  검증정확도={val_acc:.4f}{marker}"
        )

    if best_weights is not None:
        model.W, model.b = best_weights
    test_acc, test_loss = evaluate(model, X_test, y_test)
    elapsed = time.time() - started
    print(f"\n학습 완료 ({elapsed:.1f}초)")
    print(f"  최고 검증 정확도 {best_val:.4f}")
    print(f"  테스트 정확도   {test_acc:.4f}  (손실 {test_loss:.4f})")

    model.save(out)
    print(f"  저장: {out}")
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="NumPy MLP로 MNIST 학습")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-augment", action="store_true", help="평행이동 증강 끄기")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    args = p.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        augment=not args.no_augment,
        seed=args.seed,
        out=args.out,
    )


if __name__ == "__main__":
    main()
