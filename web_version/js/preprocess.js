/**
 * 손으로 그린 이미지를 MNIST 형식(28x28)으로 정규화한다.
 *
 * 이 파일은 desktop_version/src/preprocess.py의 1:1 대응 구현이다.
 * 한쪽을 고치면 반드시 다른 쪽도 고치고, test.html로 일치를 확인할 것.
 * 캔버스의 drawImage 리사이즈를 쓰지 않고 면적평균을 직접 구현한 이유는
 * 브라우저마다 리샘플링 알고리즘이 달라 파이썬과 결과가 어긋나기 때문이다.
 */

export const INK_THRESHOLD = 0.05; // 이 값 이하는 배경으로 본다
export const BOX = 20;             // 숫자를 담을 정사각형 한 변
export const CANVAS = 28;

/**
 * 면적평균(area-average) 리샘플링.
 * 출력 픽셀 하나가 덮는 입력 영역을 겹친 넓이로 가중평균한다.
 *
 * @param {Float64Array} src 입력 평면 배열 (srcH * srcW)
 * @returns {Float64Array} 출력 평면 배열 (dstH * dstW)
 */
export function resizeArea(src, srcH, srcW, dstH, dstW) {
  const dst = new Float64Array(dstH * dstW);
  for (let dy = 0; dy < dstH; dy++) {
    const y0 = (dy * srcH) / dstH;
    const y1 = ((dy + 1) * srcH) / dstH;
    const syStart = Math.floor(y0);
    const syEnd = Math.min(Math.ceil(y1), srcH);
    for (let dx = 0; dx < dstW; dx++) {
      const x0 = (dx * srcW) / dstW;
      const x1 = ((dx + 1) * srcW) / dstW;
      const sxStart = Math.floor(x0);
      const sxEnd = Math.min(Math.ceil(x1), srcW);
      let total = 0;
      let weight = 0;
      for (let sy = syStart; sy < syEnd; sy++) {
        const wy = Math.min(y1, sy + 1) - Math.max(y0, sy);
        if (wy <= 0) continue;
        for (let sx = sxStart; sx < sxEnd; sx++) {
          const wx = Math.min(x1, sx + 1) - Math.max(x0, sx);
          if (wx <= 0) continue;
          const w = wy * wx;
          total += src[sy * srcW + sx] * w;
          weight += w;
        }
      }
      dst[dy * dstW + dx] = weight > 0 ? total / weight : 0;
    }
  }
  return dst;
}

/**
 * 잉크의 무게중심이 (13.5, 13.5)에 오도록 정수 픽셀만큼 평행이동한다.
 * @param {Float64Array} canvas 28*28 평면 배열
 * @returns {Float64Array} 이동된 28*28 배열
 */
export function shiftToCenterOfMass(canvas) {
  let total = 0;
  let sumY = 0;
  let sumX = 0;
  for (let y = 0; y < CANVAS; y++) {
    for (let x = 0; x < CANVAS; x++) {
      const v = canvas[y * CANVAS + x];
      total += v;
      sumY += v * y;
      sumX += v * x;
    }
  }
  if (total <= 0) return canvas;

  const cy = sumY / total;
  const cx = sumX / total;
  const shiftY = Math.round((CANVAS - 1) / 2 - cy);
  const shiftX = Math.round((CANVAS - 1) / 2 - cx);
  if (shiftY === 0 && shiftX === 0) return canvas;

  const out = new Float64Array(CANVAS * CANVAS);
  // 잘려나가지 않는 구간만 복사한다
  const srcY0 = Math.max(0, -shiftY);
  const srcY1 = Math.min(CANVAS, CANVAS - shiftY);
  const srcX0 = Math.max(0, -shiftX);
  const srcX1 = Math.min(CANVAS, CANVAS - shiftX);
  for (let y = srcY0; y < srcY1; y++) {
    for (let x = srcX0; x < srcX1; x++) {
      out[(y + shiftY) * CANVAS + (x + shiftX)] = canvas[y * CANVAS + x];
    }
  }
  return out;
}

/**
 * 회색조 배열을 28x28 입력 벡터로 변환한다.
 *
 * @param {Float64Array} gray 0.0~1.0, 1.0이 잉크. 길이 h*w.
 * @returns {Float64Array|null} 784 길이 벡터. 잉크가 없으면 null.
 */
export function normalize(gray, h, w) {
  // 1. 바운딩 박스 찾기
  let top = Infinity;
  let bottom = -1;
  let left = Infinity;
  let right = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (gray[y * w + x] > INK_THRESHOLD) {
        if (y < top) top = y;
        if (y > bottom) bottom = y;
        if (x < left) left = x;
        if (x > right) right = x;
      }
    }
  }
  if (bottom < 0) return null; // 잉크 없음

  const cropH = bottom - top + 1;
  const cropW = right - left + 1;
  const cropped = new Float64Array(cropH * cropW);
  for (let y = 0; y < cropH; y++) {
    for (let x = 0; x < cropW; x++) {
      cropped[y * cropW + x] = gray[(top + y) * w + (left + x)];
    }
  }

  // 2. 긴 변을 BOX에 맞춰 종횡비 유지 축소
  const scale = BOX / Math.max(cropH, cropW);
  const newH = Math.max(1, Math.min(BOX, Math.round(cropH * scale)));
  const newW = Math.max(1, Math.min(BOX, Math.round(cropW * scale)));
  const small = resizeArea(cropped, cropH, cropW, newH, newW);

  // 3. 28x28 중앙에 배치
  const canvas = new Float64Array(CANVAS * CANVAS);
  const offY = Math.floor((CANVAS - newH) / 2);
  const offX = Math.floor((CANVAS - newW) / 2);
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      canvas[(offY + y) * CANVAS + (offX + x)] = small[y * newW + x];
    }
  }

  // 4. 무게중심 정렬
  return shiftToCenterOfMass(canvas);
}

/**
 * 캔버스의 픽셀을 회색조 잉크 배열로 바꾼다.
 * 캔버스는 흰 배경에 검은 글씨이므로 반전한다(잉크 = 1.0).
 *
 * @param {ImageData} imageData
 * @returns {Float64Array} 길이 width*height, 0.0~1.0
 */
export function imageDataToInk(imageData) {
  const { data, width, height } = imageData;
  const gray = new Float64Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const a = data[i * 4 + 3] / 255;
    // 알파를 흰 배경과 합성한 뒤 밝기를 구하고 반전한다
    const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    const composited = lum * a + 1 * (1 - a);
    gray[i] = 1 - composited;
  }
  return gray;
}
