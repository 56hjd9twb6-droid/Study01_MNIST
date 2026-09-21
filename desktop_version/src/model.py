"""순수 NumPy MLP. 순전파/역전파를 직접 구현한다.

구조: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (softmax)

웹 버전의 순수 JS 추론 코드(web_version/js/model.js)는 이 클래스의
forward()와 정확히 같은 연산을 수행한다. 한쪽을 바꾸면 반드시 다른 쪽도 바꿔야 한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """ReLU 계열에 맞는 He 정규분포 초기화."""
    std = np.sqrt(2.0 / fan_in)
    return (rng.standard_normal((fan_in, fan_out)) * std).astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    """행 단위 softmax. 최댓값을 빼서 overflow를 막는다."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """완전연결 신경망. 가중치는 W[i] (fan_in, fan_out), 편향은 b[i] (fan_out,)."""

    def __init__(self, sizes: list[int] | None = None, seed: int = 42):
        self.sizes = sizes or [784, 128, 64, 10]
        rng = np.random.default_rng(seed)
        self.W = [
            he_init(self.sizes[i], self.sizes[i + 1], rng)
            for i in range(len(self.sizes) - 1)
        ]
        self.b = [
            np.zeros(self.sizes[i + 1], dtype=np.float32)
            for i in range(len(self.sizes) - 1)
        ]
        self._init_adam()

    def _init_adam(self) -> None:
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    @property
    def n_layers(self) -> int:
        return len(self.W)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """X: (N, 784) -> (확률 (N, 10), 각 층의 활성값 캐시).

        마지막 층을 뺀 모든 층에 ReLU, 마지막 층에 softmax를 적용한다.
        """
        acts = [X]
        a = X
        for i in range(self.n_layers):
            z = a @ self.W[i] + self.b[i]
            a = softmax(z) if i == self.n_layers - 1 else np.maximum(z, 0.0)
            acts.append(a)
        return a, acts

    def predict(self, X: np.ndarray) -> np.ndarray:
        """가장 확률이 높은 클래스 인덱스."""
        probs, _ = self.forward(X)
        return probs.argmax(axis=1)

    def loss_and_grads(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        """교차 엔트로피 손실과 기울기를 계산한다 (역전파)."""
        n = len(X)
        probs, acts = self.forward(X)

        # 교차 엔트로피: -log p[정답]
        eps = 1e-12
        loss = float(-np.log(probs[np.arange(n), y] + eps).mean())

        # softmax + 교차 엔트로피의 결합 미분은 (p - onehot)로 간단해진다
        delta = probs.copy()
        delta[np.arange(n), y] -= 1.0
        delta /= n

        gW: list[np.ndarray | None] = [None] * self.n_layers
        gb: list[np.ndarray | None] = [None] * self.n_layers
        for i in range(self.n_layers - 1, -1, -1):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                # ReLU의 미분: 활성값이 0보다 클 때만 기울기를 통과시킨다
                delta = (delta @ self.W[i].T) * (acts[i] > 0)
        return loss, gW, gb  # type: ignore[return-value]

    def adam_step(
        self,
        gW: list[np.ndarray],
        gb: list[np.ndarray],
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """Adam 한 스텝. weight_decay는 가중치에만 적용한다(편향 제외)."""
        self.t += 1
        bc1 = 1.0 - beta1**self.t
        bc2 = 1.0 - beta2**self.t
        for i in range(self.n_layers):
            g = gW[i] + weight_decay * self.W[i] if weight_decay else gW[i]
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * g
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (g * g)
            self.W[i] -= lr * (self.mW[i] / bc1) / (np.sqrt(self.vW[i] / bc2) + eps)

            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * gb[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (gb[i] * gb[i])
            self.b[i] -= lr * (self.mb[i] / bc1) / (np.sqrt(self.vb[i] / bc2) + eps)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"sizes": np.array(self.sizes)}
        for i in range(self.n_layers):
            arrays[f"W{i}"] = self.W[i]
            arrays[f"b{i}"] = self.b[i]
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "MLP":
        data = np.load(Path(path))
        model = cls(sizes=[int(v) for v in data["sizes"]])
        for i in range(model.n_layers):
            model.W[i] = data[f"W{i}"].astype(np.float32)
            model.b[i] = data[f"b{i}"].astype(np.float32)
        model._init_adam()
        return model
