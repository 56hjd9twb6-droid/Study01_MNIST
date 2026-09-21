# study01_MNIST

손글씨 숫자 인식기. 같은 모델을 두 가지 방식으로 제공한다.

```
desktop_version/   NumPy로 직접 구현한 학습 + tkinter GUI   (파이썬)
web_version/       순수 자바스크립트 추론, 정적 배포        (의존성 0)
```

## 두 폴더의 관계

한쪽에서 학습하고, 다른 쪽은 그 결과를 쓴다.

```
desktop_version/models/mlp.npz          학습 산출물 (NumPy)
        │  python -m src.export_weights
        ▼
web_version/model/weights.json          웹이 읽는 가중치
web_version/model/fixtures.json         두 구현 일치 검증용 표본
```

**웹 버전에는 학습 코드가 없다.** 모델을 바꾸려면 항상 `desktop_version`에서
학습하고 내보내야 한다.

## 반드시 지킬 것

### 1. 전처리는 두 곳에 있고, 항상 같아야 한다

`desktop_version/src/preprocess.py` 와 `web_version/js/preprocess.js` 는
같은 알고리즘의 두 구현이다. 한쪽만 고치면 웹과 데스크톱의 인식 결과가
조용히 갈라진다 — 예외도 안 나고 테스트도 안 깨지므로 알아채기 어렵다.

한쪽을 고쳤다면:
1. 다른 쪽도 같이 고친다
2. `python -m src.export_weights` 로 fixtures.json 을 다시 만든다
3. `web_version/test.html` 을 열어 일치를 확인한다

리사이즈에 PIL이나 캔버스 `drawImage` 를 쓰지 않고 면적평균을 직접 구현한 것도
같은 이유다. 내장 리샘플링은 구현마다 결과가 달라 두 버전이 어긋난다.

### 2. 추론 연산도 두 곳에 있다

`model.py` 의 `MLP.forward()` 와 `model.js` 의 `MLP.forward()` 가 짝이다.
층 구조나 활성함수를 바꾸면 양쪽 모두 고치고, 내보내기 형식
(`weights.json` 의 `format` 필드)도 올려야 한다.

### 3. 가중치는 커밋한다

`web_version/model/weights.json` 이 저장소에 있어야 GitHub Pages가 동작한다.
`.gitignore` 가 무시하는 것은 MNIST 원본 데이터(`desktop_version/data/`)와
가상환경뿐이다.

## 검증

코드를 고쳤으면 해당하는 것을 돌린다.

```bash
# 파이썬: 역전파가 수치미분과 맞는지
cd desktop_version && .venv/bin/python tests/test_gradients.py

# 파이썬: GUI 그리기 -> 추론 경로 (창을 띄우지 않는다)
cd desktop_version && .venv/bin/python tests/test_app_logic.py

# JS: 파이썬과 결과가 같은지 — 브라우저에서 web_version/test.html 을 연다
python3 -m http.server 8765 --directory web_version
```

`test.html` 은 파일을 직접 열면(`file://`) 동작하지 않는다. ES 모듈과 fetch에
정적 서버가 필요하다.

## 현재 성능

784 → 128 → 64 → 10 다층 퍼셉트론, 파라미터 109,386개.
MNIST 테스트 정확도 **98.59%**.

정확도를 더 올리려면 CNN이 필요하지만, 그러면 `model.js` 에 합성곱을 직접
구현해야 한다. 지금 구조는 행렬곱 하나뿐이라 양쪽을 맞추기 쉽다는 게 장점이다.
