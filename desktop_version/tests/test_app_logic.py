"""GUI를 실제로 띄우지 않고 그리기 -> 추론 경로를 검증한다.

창은 withdraw()로 숨기고, 마우스 이벤트 대신 _stroke()를 직접 호출한다.
"""
import math
import sys
import tkinter as tk
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.app import PAD_SIZE, RecognizerApp
from src.model import MLP
from src.preprocess import normalize
from src.train import DEFAULT_MODEL


def polyline(app: RecognizerApp, points: list[tuple[float, float]]) -> None:
    app._stroke(points[0][0], points[0][1], points[0][0], points[0][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        app._stroke(x0, y0, x1, y1)


def line(x1, y1, x2, y2, n=40):
    return [(x1 + (x2 - x1) * i / n, y1 + (y2 - y1) * i / n) for i in range(n + 1)]


def circle(cx, cy, rx, ry, n=60):
    return [(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n))
            for i in range(n + 1)]


# 웹 버전 테스트와 같은 궤적을 쓴다
SHAPES = {
    0: [circle(140, 140, 52, 72)],
    1: [line(120, 70, 145, 52), line(145, 52, 145, 215)],
    4: [line(170, 50, 95, 150), line(95, 150, 205, 150), line(170, 50, 170, 215)],
    7: [line(80, 60, 200, 60), line(200, 60, 120, 215)],
}


def main() -> None:
    if not DEFAULT_MODEL.exists():
        raise SystemExit(f"학습된 모델이 없습니다: {DEFAULT_MODEL}")

    model = MLP.load(DEFAULT_MODEL)
    root = tk.Tk()
    root.withdraw()  # 창을 띄우지 않는다
    app = RecognizerApp(root, model)

    failures = 0
    for expected, strokes in SHAPES.items():
        app.clear()
        for s in strokes:
            polyline(app, s)
        app.predict()
        got = app.digit_label.cget("text")
        conf = app.conf_label.cget("text")
        ok = got == str(expected)
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {expected} 그림 -> 예측 {got} @ {conf}")

    # 빈 캔버스에서 추론해도 예외가 나지 않아야 한다
    app.clear()
    app.predict()
    assert app.digit_label.cget("text") == "–", "빈 캔버스인데 숫자가 표시됨"
    print("PASS  빈 캔버스에서 예외 없음")

    # 전처리 결과가 유효 범위 안인지 확인
    app.clear()
    polyline(app, SHAPES[7][0])
    polyline(app, SHAPES[7][1])
    gray = 1.0 - np.asarray(app.image, dtype=np.float64) / 255.0
    vec = normalize(gray)
    assert vec is not None and vec.shape == (784,), f"전처리 출력 이상: {vec}"
    assert 0.0 <= vec.min() and vec.max() <= 1.0, f"값 범위 이탈: [{vec.min()}, {vec.max()}]"
    print(f"PASS  전처리 출력 (784,) 범위 [{vec.min():.3f}, {vec.max():.3f}]")

    root.destroy()
    if failures:
        raise SystemExit(f"\n{failures}건 실패")
    print("\n전체 통과")


if __name__ == "__main__":
    main()
