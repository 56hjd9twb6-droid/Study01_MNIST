/**
 * 순수 자바스크립트 MLP 추론.
 *
 * desktop_version/src/model.py의 MLP.forward()와 같은 연산을 수행한다.
 * 학습은 하지 않는다 — 가중치는 데스크톱 버전에서 학습해 JSON으로 받는다.
 * 외부 라이브러리를 쓰지 않는다.
 */

/**
 * 행 단위 softmax. 최댓값을 빼서 overflow를 막는다(파이썬 구현과 동일).
 * @param {Float64Array} z
 * @returns {Float64Array}
 */
export function softmax(z) {
  let max = -Infinity;
  for (let i = 0; i < z.length; i++) if (z[i] > max) max = z[i];
  const out = new Float64Array(z.length);
  let sum = 0;
  for (let i = 0; i < z.length; i++) {
    out[i] = Math.exp(z[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < z.length; i++) out[i] /= sum;
  return out;
}

export class MLP {
  /**
   * @param {{sizes:number[], layers:{shape:number[], W:number[], b:number[]}[]}} weights
   */
  constructor(weights) {
    if (weights.format !== "mnist-mlp-v1") {
      throw new Error(`알 수 없는 가중치 형식: ${weights.format}`);
    }
    this.sizes = weights.sizes;
    this.layers = weights.layers.map((layer, i) => {
      const [rows, cols] = layer.shape;
      if (layer.W.length !== rows * cols) {
        throw new Error(`층 ${i}: W 길이 불일치 (${layer.W.length} != ${rows * cols})`);
      }
      if (layer.b.length !== cols) {
        throw new Error(`층 ${i}: b 길이 불일치 (${layer.b.length} != ${cols})`);
      }
      return {
        rows,
        cols,
        // 행 우선 평면 배열. Float32Array로 두면 파이썬 float32와 값이 맞는다.
        W: Float32Array.from(layer.W),
        b: Float32Array.from(layer.b),
      };
    });
  }

  /** 가중치 JSON을 받아 모델을 만든다. */
  static async load(url) {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`가중치를 불러오지 못했습니다 (${res.status} ${res.statusText}): ${url}`);
    }
    return new MLP(await res.json());
  }

  /** 파라미터 개수. */
  get paramCount() {
    return this.layers.reduce((n, l) => n + l.rows * l.cols + l.cols, 0);
  }

  /**
   * 순전파. 마지막 층 전까지 ReLU, 마지막 층에 softmax.
   * @param {Float64Array|number[]} input 길이 784
   * @returns {Float64Array} 길이 10의 확률
   */
  forward(input) {
    if (input.length !== this.sizes[0]) {
      throw new Error(`입력 길이가 ${this.sizes[0]}이어야 합니다: ${input.length}`);
    }
    let a = input;
    for (let li = 0; li < this.layers.length; li++) {
      const { rows, cols, W, b } = this.layers[li];
      const z = new Float64Array(cols);
      // z = a @ W + b — W가 행 우선이므로 입력을 바깥 루프에 둔다(캐시 효율).
      for (let c = 0; c < cols; c++) z[c] = b[c];
      for (let r = 0; r < rows; r++) {
        const av = a[r];
        if (av === 0) continue; // 입력은 대부분 0이라 이 가지치기가 크게 이득이다
        const base = r * cols;
        for (let c = 0; c < cols; c++) z[c] += av * W[base + c];
      }
      if (li === this.layers.length - 1) {
        a = softmax(z);
      } else {
        for (let c = 0; c < cols; c++) if (z[c] < 0) z[c] = 0; // ReLU
        a = z;
      }
    }
    return a;
  }

  /** 가장 확률이 높은 클래스. */
  predict(input) {
    const probs = this.forward(input);
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return { digit: best, confidence: probs[best], probs };
  }
}
