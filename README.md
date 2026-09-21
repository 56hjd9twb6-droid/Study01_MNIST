# Study01_MNIST

손글씨 숫자 인식기. NumPy로 직접 구현해 학습하고, 순수 자바스크립트로 추론한다.
딥러닝 프레임워크를 쓰지 않는다.

| | 학습 | 추론 | 의존성 |
|---|---|---|---|
| **desktop_version** | NumPy 직접 구현 | 파이썬 | numpy, Pillow |
| **web_version** | 없음 (가중치를 받아 씀) | 순수 JS | 없음 |

MNIST 테스트 정확도 **98.59%** · 784 → 128 → 64 → 10 · 파라미터 109,386개

## 웹 버전

```bash
python3 -m http.server 8765 --directory web_version
```

`http://localhost:8765/` 접속. 빌드 단계가 없고 외부 요청도 없어서,
`web_version/` 폴더를 그대로 정적 호스팅에 올리면 된다.

GitHub Pages는 저장소 Settings → Pages → Source 를 **GitHub Actions** 로
설정하면 `.github/workflows/pages.yml` 이 자동 배포한다.

## 데스크톱 버전

```bash
cd desktop_version
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python -m src.train           # 학습 (약 40초, MNIST 자동 다운로드)
.venv/bin/python -m src.app             # GUI 실행
.venv/bin/python -m src.export_weights  # 웹 버전용 가중치 내보내기
```

## 어떻게 동작하나

그린 그림은 MNIST와 같은 방식으로 정규화된다.

1. 잉크가 있는 영역만 잘라낸다
2. 종횡비를 유지한 채 긴 변이 20px이 되도록 줄인다
3. 28×28 캔버스에 넣고 무게중심을 한가운데로 옮긴다

이 전처리와 신경망 순전파가 파이썬과 자바스크립트에 각각 구현돼 있고,
`web_version/test.html` 이 두 구현의 결과가 같은지 검증한다
(현재 오차: 전처리 5.1e-7, 확률 1.0e-7).

## 검증

```bash
cd desktop_version
.venv/bin/python tests/test_gradients.py   # 역전파 vs 수치미분
.venv/bin/python tests/test_app_logic.py   # 그리기 -> 추론 경로
```

웹 쪽은 서버를 띄운 뒤 `http://localhost:8765/test.html` 을 연다.

## 라이선스 / 데이터

MNIST는 Yann LeCun 등이 배포한 데이터셋으로, 학습 시 자동으로 내려받는다
(약 11MB, 저장소에는 포함하지 않는다).
