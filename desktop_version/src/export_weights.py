"""학습한 가중치를 웹 버전이 읽을 JSON으로 내보낸다.

같이 생성하는 fixtures.json은 파이썬과 JS 추론 결과가 일치하는지
검증하기 위한 것이다(web_version/model/fixtures.json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import data
from .model import MLP
from .preprocess import normalize
from .train import DEFAULT_MODEL, evaluate

WEB_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "web_version" / "model"

# JSON에 기록할 소수점 자리수. 6자리면 파일 크기와 정확도가 모두 무난하다.
DECIMALS = 6


def round_weights(model: MLP, decimals: int = DECIMALS) -> MLP:
    """JSON에 쓸 정밀도로 가중치를 반올림한 사본을 만든다."""
    rounded = MLP(sizes=model.sizes)
    rounded.W = [np.round(w, decimals).astype(np.float32) for w in model.W]
    rounded.b = [np.round(b, decimals).astype(np.float32) for b in model.b]
    return rounded


def export_model(model: MLP, out: Path, decimals: int = DECIMALS) -> Path:
    """가중치를 평면 배열 형태로 저장한다. JS에서 Float32Array로 바로 쓴다."""
    payload = {
        "format": "mnist-mlp-v1",
        "sizes": [int(s) for s in model.sizes],
        "activations": ["relu"] * (model.n_layers - 1) + ["softmax"],
        "layers": [
            {
                "shape": [int(model.W[i].shape[0]), int(model.W[i].shape[1])],
                # 행 우선(row-major) 평면 배열: W[i][r * cols + c]
                "W": [round(float(v), decimals) for v in model.W[i].reshape(-1)],
                "b": [round(float(v), decimals) for v in model.b[i]],
            }
            for i in range(model.n_layers)
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    return out


def export_fixtures(model: MLP, out: Path, n: int = 12, decimals: int = DECIMALS) -> Path:
    """JS 구현을 검증할 표본을 저장한다.

    전처리 전 원본 28x28과 전처리 후 벡터, 그리고 기대 확률을 함께 담아
    JS 쪽의 전처리와 추론을 각각 따로 확인할 수 있게 한다.
    """
    X, y = data.load("test")
    rng = np.random.default_rng(0)
    # 0~9가 최소 한 번씩은 들어가도록 고른다
    picked: list[int] = []
    for digit in range(10):
        candidates = np.where(y == digit)[0]
        picked.append(int(rng.choice(candidates)))
    while len(picked) < n:
        extra = int(rng.integers(0, len(X)))
        if extra not in picked:
            picked.append(extra)

    samples = []
    for idx in picked:
        raw = X[idx].reshape(28, 28)
        vec = normalize(raw)
        assert vec is not None
        probs, _ = model.forward(vec.reshape(1, -1))
        samples.append(
            {
                "label": int(y[idx]),
                # 원본은 0~255 정수로 저장해 파일을 작게 유지한다
                "raw": [int(round(v * 255)) for v in X[idx]],
                "normalized": [round(float(v), decimals) for v in vec],
                "probs": [round(float(v), 8) for v in probs[0]],
                "predicted": int(probs[0].argmax()),
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"format": "mnist-fixtures-v1", "samples": samples}, separators=(",", ":")))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="가중치를 웹 버전용 JSON으로 내보내기")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--out-dir", type=Path, default=WEB_MODEL_DIR)
    p.add_argument("--decimals", type=int, default=DECIMALS)
    args = p.parse_args()

    model = MLP.load(args.model)
    print(f"모델 로드: {args.model}  구조 {' -> '.join(map(str, model.sizes))}")

    # 반올림이 정확도를 떨어뜨리지 않는지 확인한다
    X_test, y_test = data.load("test")
    acc_full, _ = evaluate(model, X_test, y_test)
    rounded = round_weights(model, args.decimals)
    acc_round, _ = evaluate(rounded, X_test, y_test)
    print(f"  원본 정확도        {acc_full:.4f}")
    print(f"  {args.decimals}자리 반올림 정확도 {acc_round:.4f}  (차이 {acc_round - acc_full:+.4f})")
    if abs(acc_round - acc_full) > 0.001:
        raise SystemExit("반올림으로 정확도가 유의하게 떨어졌습니다. --decimals를 늘리세요.")

    model_path = export_model(rounded, args.out_dir / "weights.json", args.decimals)
    fixture_path = export_fixtures(rounded, args.out_dir / "fixtures.json", decimals=args.decimals)
    for path in (model_path, fixture_path):
        print(f"  저장: {path}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
