# web_version

브라우저에서 동작하는 손글씨 숫자 인식기. **외부 라이브러리를 쓰지 않는다.**
빌드 단계도 없다 — 이 폴더를 그대로 정적 호스팅에 올리면 된다.

## 절대 규칙

1. **의존성 0.** npm 패키지, CDN 스크립트, 웹폰트, 외부 이미지 모두 금지.
   추가하려는 순간 이 버전의 존재 이유가 사라진다.
2. **빌드 없음.** 번들러·트랜스파일러·전처리기를 도입하지 않는다.
   브라우저가 읽는 파일이 저장소에 있는 파일 그대로여야 한다.
3. **경로는 전부 상대경로.** `/js/app.js` 가 아니라 `./js/app.js`.
   GitHub Pages는 `사용자명.github.io/저장소명/` 서브경로로 서빙되므로
   절대경로를 쓰면 전부 404가 난다.
4. **학습 코드를 넣지 않는다.** 여기는 추론 전용이다.

## 실행

ES 모듈과 fetch 때문에 `file://` 로는 동작하지 않는다. 정적 서버가 필요하다.

```bash
python3 -m http.server 8765 --directory web_version
```

- `http://localhost:8765/` — 앱
- `http://localhost:8765/test.html` — 파이썬 구현과 일치하는지 검증

## 파일 구조

```
index.html          UI 골격
css/style.css       스타일. CSS 변수로 색을 모아 둠.
js/preprocess.js    그린 이미지 -> 28x28 정규화   ※ 파이썬 쌍둥이 있음
js/model.js         MLP 순전파                    ※ 파이썬 쌍둥이 있음
js/app.js           캔버스 입력 <-> 화면 출력 연결
model/weights.json  가중치 (desktop_version 이 생성, 약 1MB)
model/fixtures.json 일치 검증용 표본 12개
test.html           검증 페이지
.nojekyll           GitHub Pages가 Jekyll로 처리하지 않게 함
```

## 알아 둘 것

### preprocess.js / model.js 는 파이썬 코드의 번역본이다

각각 `desktop_version/src/preprocess.py`, `model.py` 와 짝이다.
한쪽만 고치면 **조용히** 결과가 갈라진다. 예외도 안 나고 UI도 멀쩡해 보인다.
고쳤다면 `test.html` 을 반드시 열어 볼 것. 허용 오차는 전처리 1e-5,
확률 1e-4 이며, 현재 실측은 각각 5.1e-7, 1.0e-7 이다.

이 테스트는 실제로 빨간불이 켜지는 것을 확인했다. `preprocess.js` 의 `BOX` 를
20에서 19로 바꾸자(파이썬은 20 유지) 12개 표본 중 12개가 실패했다.

### test.html 은 모듈을 동적으로 불러온다

정적 `import` 를 쓰면 브라우저 모듈 캐시 때문에 **방금 고친 파일이 아니라
이전 버전을 검증하고 초록불을 띄운다.** 실제로 겪은 문제다 — 디스크의
`BOX` 가 19인데 캐시된 20으로 테스트가 통과했다. 그래서 `test.html` 은
`await import(\`./js/model.js?t=${Date.now()}\`)` 형태로 매번 새로 받는다.
이 부분을 정적 import로 되돌리지 말 것.

앱(`index.html`)은 캐시를 그대로 쓴다. 배포된 페이지에서는 캐시가 이득이고,
사용자가 모듈을 고칠 일이 없기 때문이다.

### 리사이즈를 직접 구현한 이유

캔버스의 `drawImage` 축소는 브라우저마다 알고리즘이 다르다. 파이썬과
결과를 맞출 수 없어서 `resizeArea()` 로 면적평균을 직접 구현했다.
느려 보이지만 28x28로 줄이는 연산이라 체감되지 않는다.

### 가중치는 Float32Array 로 읽는다

JSON의 수는 float64지만 파이썬 모델이 float32이므로 `Float32Array.from()`
으로 맞춘다. 이걸 빼면 파이썬과 확률값이 미세하게 달라진다.

### forward() 의 0 건너뛰기

입력 784개 중 대부분이 0(배경)이라 `if (av === 0) continue` 로 첫 층
행렬곱을 크게 줄인다. ReLU 뒤에도 0이 많아 같은 최적화가 계속 효과가 있다.
수학적으로는 영향이 없다 — 0을 곱해 더하는 것을 생략할 뿐이다.

### 가중치를 바꿨다면

`desktop_version` 에서 `python -m src.export_weights` 를 돌려
`weights.json` 과 `fixtures.json` 을 **함께** 갱신한다. 하나만 바꾸면
test.html 이 실패한다(그게 의도다).

## GitHub Pages 배포

`.github/workflows/pages.yml` 이 `web_version/` 을 사이트 루트로 올린다.
저장소 Settings → Pages → Source 를 **GitHub Actions** 로 두어야 한다.

수동으로 확인하려면 서브경로에서 서빙해 본다:

```bash
mkdir -p /tmp/sim/저장소명 && cp -R web_version/. /tmp/sim/저장소명/
python3 -m http.server 8799 --directory /tmp/sim
# http://localhost:8799/저장소명/ 이 정상 동작해야 한다
```
