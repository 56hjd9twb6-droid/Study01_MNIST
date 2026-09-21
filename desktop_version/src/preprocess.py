"""손으로 그린 이미지를 MNIST 형식(28x28)으로 정규화한다.

MNIST의 원래 전처리 규칙을 그대로 따른다:
  1. 잉크가 있는 영역의 바운딩 박스로 자른다
  2. 종횡비를 유지한 채 긴 변이 20px이 되도록 축소한다
  3. 28x28 캔버스에 넣고, 무게중심이 중앙에 오도록 옮긴다

이 파일의 로직은 web_version/js/preprocess.js와 1:1로 대응한다.
한쪽을 고치면 반드시 다른 쪽도 고치고 tools/check_parity.py로 확인할 것.
직접 구현한 면적평균 리샘플링을 쓰는 이유는, PIL이나 캔버스의 내장 리사이즈에
의존하면 두 구현의 결과가 미묘하게 달라지기 때문이다.
"""
from __future__ import annotations

import numpy as np

INK_THRESHOLD = 0.05  # 이 값 이하는 배경으로 본다
BOX = 20  # 숫자를 담을 정사각형 한 변
CANVAS = 28


def resize_area(src: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    """면적평균(area-average) 리샘플링.

    출력 픽셀 하나가 덮는 입력 영역을 겹친 넓이로 가중평균한다.
    축소·확대 모두 동작하며, 부동소수점 연산 순서까지 JS 구현과 맞춘다.
    """
    src_h, src_w = src.shape
    dst = np.zeros((dst_h, dst_w), dtype=np.float64)
    for dy in range(dst_h):
        y0 = dy * src_h / dst_h
        y1 = (dy + 1) * src_h / dst_h
        sy_start, sy_end = int(np.floor(y0)), min(int(np.ceil(y1)), src_h)
        for dx in range(dst_w):
            x0 = dx * src_w / dst_w
            x1 = (dx + 1) * src_w / dst_w
            sx_start, sx_end = int(np.floor(x0)), min(int(np.ceil(x1)), src_w)
            total = 0.0
            weight = 0.0
            for sy in range(sy_start, sy_end):
                wy = min(y1, sy + 1.0) - max(y0, float(sy))
                if wy <= 0:
                    continue
                for sx in range(sx_start, sx_end):
                    wx = min(x1, sx + 1.0) - max(x0, float(sx))
                    if wx <= 0:
                        continue
                    w = wy * wx
                    total += float(src[sy, sx]) * w
                    weight += w
            dst[dy, dx] = total / weight if weight > 0 else 0.0
    return dst


def normalize(gray: np.ndarray) -> np.ndarray | None:
    """(H, W) 회색조 배열 -> (784,) float32 벡터.

    입력은 0.0~1.0, 1.0이 잉크(흰 글씨/검은 배경)여야 한다.
    잉크가 전혀 없으면 None을 반환한다.
    """
    gray = np.asarray(gray, dtype=np.float64)
    if gray.ndim != 2:
        raise ValueError(f"2차원 배열이어야 합니다: {gray.shape}")

    # 1. 바운딩 박스로 자르기
    mask = gray > INK_THRESHOLD
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    cropped = gray[top:bottom, left:right]

    # 2. 긴 변을 BOX에 맞춰 종횡비 유지 축소
    h, w = cropped.shape
    scale = BOX / max(h, w)
    new_h = max(1, min(BOX, int(round(h * scale))))
    new_w = max(1, min(BOX, int(round(w * scale))))
    small = resize_area(cropped, new_h, new_w)

    # 3. 28x28 중앙에 배치
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.float64)
    off_y = (CANVAS - new_h) // 2
    off_x = (CANVAS - new_w) // 2
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = small

    # 4. 무게중심을 캔버스 중앙으로 이동
    canvas = shift_to_center_of_mass(canvas)
    return canvas.reshape(-1).astype(np.float32)


def shift_to_center_of_mass(canvas: np.ndarray) -> np.ndarray:
    """잉크의 무게중심이 (13.5, 13.5)에 오도록 정수 픽셀만큼 평행이동."""
    total = canvas.sum()
    if total <= 0:
        return canvas
    ys, xs = np.mgrid[0 : canvas.shape[0], 0 : canvas.shape[1]]
    cy = float((canvas * ys).sum() / total)
    cx = float((canvas * xs).sum() / total)
    shift_y = int(round((CANVAS - 1) / 2.0 - cy))
    shift_x = int(round((CANVAS - 1) / 2.0 - cx))
    if shift_y == 0 and shift_x == 0:
        return canvas

    out = np.zeros_like(canvas)
    # 잘려나가지 않는 구간만 복사한다
    src_y0 = max(0, -shift_y)
    src_y1 = min(CANVAS, CANVAS - shift_y)
    src_x0 = max(0, -shift_x)
    src_x1 = min(CANVAS, CANVAS - shift_x)
    if src_y0 < src_y1 and src_x0 < src_x1:
        out[src_y0 + shift_y : src_y1 + shift_y, src_x0 + shift_x : src_x1 + shift_x] = (
            canvas[src_y0:src_y1, src_x0:src_x1]
        )
    return out


def to_ascii(vec: np.ndarray) -> str:
    """28x28 벡터를 터미널에서 눈으로 확인하기 위한 문자 그림."""
    chars = " .:-=+*#%@"
    img = np.asarray(vec).reshape(CANVAS, CANVAS)
    lines = []
    for row in img:
        lines.append("".join(chars[min(int(v * len(chars)), len(chars) - 1)] for v in row))
    return "\n".join(lines)
