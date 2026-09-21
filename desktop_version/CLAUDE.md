# desktop_version

NumPy만으로 구현한 MNIST 학습기와 tkinter GUI. 딥러닝 프레임워크를 쓰지 않는다.

## 설정

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

시스템 파이썬은 3.9다. **PyTorch를 추가하지 말 것** — 최신 torch는 3.10 이상을
요구하고, 애초에 직접 구현하는 것이 이 프로젝트의 목적이다.

## 명령

```bash
.venv/bin/python -m src.train            # 학습 -> models/mlp.npz  (약 40초)
.venv/bin/python -m src.train --epochs 5 # 짧게
.venv/bin/python -m src.export_weights   # 가중치를 web_version 으로 내보내기
.venv/bin/python -m src.app              # GUI 실행 (학습된 모델 필요)
.venv/bin/python -m src.data             # 데이터만 내려받기
```

모듈 실행(`-m src.xxx`)이어야 한다. `python src/train.py` 는 상대 임포트 때문에
동작하지 않는다.

## 파일 구조

```
src/
  data.py             MNIST 다운로드 + IDX 파서. MD5로 무결성 확인.
  model.py            MLP. 순전파/역전파/Adam을 직접 구현.
  preprocess.py       그린 이미지 -> 28x28 정규화.  ※ JS 쌍둥이 있음
  train.py            학습 루프, 평행이동 증강, 검증 기반 최적 가중치 선택
  export_weights.py   weights.json + fixtures.json 생성
  app.py              tkinter GUI
tests/
  test_gradients.py   역전파 vs 수치미분
  test_app_logic.py   창을 띄우지 않고 그리기->추론 검증
models/mlp.npz        학습 산출물 (커밋함)
data/                 MNIST 원본 (커밋 안 함)
```

## 알아 둘 것

### preprocess.py 는 혼자가 아니다

`web_version/js/preprocess.js` 와 한 쌍이다. 고칠 때는 루트 `CLAUDE.md` 의
"전처리는 두 곳에 있고, 항상 같아야 한다" 절차를 따를 것.

### 기울기를 고쳤으면 수치미분으로 확인한다

`tests/test_gradients.py` 가 역전파를 중앙차분과 비교한다. 이 테스트는
float64로 돌린다 — float32에서는 h가 유효숫자에 묻혀 상대오차가 1e-2까지
벌어져서 맞는 코드도 틀린 것처럼 보인다. 모델 자체는 float32로 학습한다.

### macOS 26 + 시스템 Tk 8.5.9 에서는 창이 그려지지 않는다

확인된 환경 문제다. macOS 26.6 기본 파이썬(3.9)이 쓰는 Tk는 8.5.9(2010년)이고,
이 조합에서는 창은 뜨지만 위젯이 그려지지 않아 **흰 화면만 보인다**.
앱 실행 시 나오는 `DEPRECATION WARNING: The system version of Tk is deprecated`
가 그 신호다.

코드 문제가 아니다 — `tests/test_app_logic.py` 는 통과하고, 실제로 그림을 그리면
추론도 정상 동작한다. 화면에 출력만 안 된다.

고치려면 Tk 8.6 이 포함된 파이썬이 필요하다. python.org 설치본을 쓰고
그 인터프리터로 가상환경을 다시 만들 것:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import tkinter; print(tkinter.TkVersion)"   # 8.6 이상이어야 한다
```

웹 버전은 이 문제와 무관하다. 같은 모델·같은 전처리를 쓰므로 결과도 동일하다.

### GUI는 캔버스를 두 번 그린다

tkinter Canvas는 픽셀을 되읽을 수 없다. 그래서 `app.py` 는 같은 획을
tkinter Canvas와 PIL Image에 각각 그리고, 추론할 때는 PIL 쪽을 읽는다.
`_stroke()` 를 고치면 두 그리기가 어긋나지 않는지 확인할 것.
PIL의 `line()` 은 끝을 둥글게 하지 않아서 양 끝에 원을 따로 찍는다.

### 증강은 평행이동만 한다

사용자가 마우스로 그린 숫자는 MNIST만큼 정확히 중앙에 오지 않는다.
±2px 이동 증강이 그 차이를 메운다. 회전이나 탄성변형은 넣지 않았다 —
`preprocess.py` 가 이미 크기와 무게중심을 정규화하므로 이득이 적다.

### 학습 결과는 시드에 고정된다

`--seed` 기본값 42. 재현이 필요하면 건드리지 말 것.
현재 기록: 테스트 정확도 98.59% (60에폭).

### 내보내기는 정확도를 검사한다

`export_weights.py` 는 JSON 반올림(소수 6자리) 후 테스트 정확도가
0.1%포인트 넘게 떨어지면 중단한다. 자릿수를 줄이려면 이 검사를 통과해야 한다.
