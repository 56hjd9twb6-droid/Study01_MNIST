"""역전파 기울기를 수치미분과 비교해 검증한다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import MLP, softmax


def numeric_grad(f, x, h=1e-5):
    """중앙차분 수치미분."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + h
        fp = f()
        x[idx] = orig - h
        fm = f()
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * h)
        it.iternext()
    return grad


def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


def test_softmax_rows_sum_to_one():
    z = np.random.default_rng(0).standard_normal((5, 10)).astype(np.float32) * 10
    p = softmax(z)
    assert np.allclose(p.sum(axis=1), 1.0), "softmax 행 합이 1이 아님"
    assert np.all(p >= 0), "softmax 음수 출력"
    print("PASS  softmax 행 합 = 1, 전부 비음수")


def test_softmax_overflow_safe():
    z = np.array([[1000.0, 1001.0, 999.0]], dtype=np.float32)
    p = softmax(z)
    assert np.all(np.isfinite(p)), "큰 입력에서 softmax가 발산"
    print("PASS  softmax overflow 안전")


def test_backprop_matches_numeric():
    """역전파와 수치미분을 float64로 비교한다.

    모델은 평소 float32로 학습하지만, float32에서 중앙차분을 하면 h가
    유효숫자에 묻혀 상대오차가 1e-2 수준까지 벌어진다. 검증 대상은
    수식이지 정밀도가 아니므로 여기서만 배정밀도로 올린다.
    """
    rng = np.random.default_rng(1)
    # 수치미분은 느리므로 작은 모델로 검증한다
    model = MLP(sizes=[6, 5, 4, 3], seed=7)
    # 편향이 0이면 검증이 약해지므로 흔들어 준다
    model.b = [rng.standard_normal(b.shape) * 0.1 for b in model.b]
    model.W = [w.astype(np.float64) for w in model.W]
    X = rng.standard_normal((8, 6))
    y = rng.integers(0, 3, size=8)

    # ReLU의 꺾이는 지점 근처에서는 수치미분 자체가 정의되지 않는다.
    # 사전활성값이 0에서 충분히 떨어져 있는지 먼저 확인한다.
    _, acts = model.forward(X)
    for i, a in enumerate(acts[1:-1], start=1):
        nonzero = np.abs(a[a != 0])
        assert nonzero.size == 0 or nonzero.min() > 1e-3, (
            f"층 {i} 활성값이 ReLU 꺾임점에 너무 가까워 수치미분을 신뢰할 수 없다"
        )

    loss, gW, gb = model.loss_and_grads(X, y)

    def f():
        return model.loss_and_grads(X, y)[0]

    worst = 0.0
    for i in range(model.n_layers):
        nW = numeric_grad(f, model.W[i])
        nb = numeric_grad(f, model.b[i])
        eW, eb = rel_error(gW[i], nW), rel_error(gb[i], nb)
        worst = max(worst, eW, eb)
        print(f"      층 {i}: W 상대오차={eW:.3e}  b 상대오차={eb:.3e}")
        assert eW < 1e-7, f"층 {i} W 기울기 불일치: {eW}"
        assert eb < 1e-7, f"층 {i} b 기울기 불일치: {eb}"
    print(f"PASS  역전파 == 수치미분 (최대 상대오차 {worst:.3e}, 손실 {loss:.4f})")


def test_loss_decreases_on_overfit():
    """샘플 몇 개를 외우게 해서 학습 루프가 실제로 동작하는지 본다."""
    rng = np.random.default_rng(2)
    model = MLP(sizes=[784, 32, 10], seed=3)
    X = rng.random((16, 784)).astype(np.float32)
    y = rng.integers(0, 10, size=16)
    first = model.loss_and_grads(X, y)[0]
    for _ in range(300):
        _, gW, gb = model.loss_and_grads(X, y)
        model.adam_step(gW, gb, lr=1e-3)
    last, _, _ = model.loss_and_grads(X, y)
    acc = (model.predict(X) == y).mean()
    assert last < first * 0.05, f"손실이 충분히 안 줄었다: {first:.4f} -> {last:.4f}"
    assert acc == 1.0, f"16개 샘플도 못 외운다: 정확도 {acc}"
    print(f"PASS  과적합 테스트: 손실 {first:.4f} -> {last:.6f}, 정확도 {acc:.0%}")


def test_save_load_roundtrip(tmp_path=Path("/tmp/mlp_roundtrip.npz")):
    model = MLP(sizes=[784, 32, 10], seed=5)
    X = np.random.default_rng(6).random((4, 784)).astype(np.float32)
    before, _ = model.forward(X)
    model.save(tmp_path)
    reloaded = MLP.load(tmp_path)
    after, _ = reloaded.forward(X)
    assert np.allclose(before, after, atol=1e-6), "저장/로드 후 출력이 달라짐"
    assert reloaded.sizes == model.sizes
    tmp_path.unlink(missing_ok=True)
    print("PASS  저장/로드 왕복 일치")


if __name__ == "__main__":
    for fn in [
        test_softmax_rows_sum_to_one,
        test_softmax_overflow_safe,
        test_backprop_matches_numeric,
        test_loss_decreases_on_overfit,
        test_save_load_roundtrip,
    ]:
        fn()
    print("\n전체 통과")
