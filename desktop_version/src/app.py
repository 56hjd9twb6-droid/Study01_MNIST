"""tkinter 손글씨 인식 GUI.

실행:
    python -m src.app

웹 버전(web_version/js/app.js)과 같은 전처리·같은 가중치를 쓴다.
따라서 같은 그림에 대해 두 버전의 결과가 일치해야 한다.
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .model import MLP
from .preprocess import CANVAS as GRID
from .preprocess import normalize
from .train import DEFAULT_MODEL

PAD_SIZE = 280      # 그리기 영역 한 변(픽셀)
STROKE_WIDTH = 20   # 웹 버전과 같은 획 두께
PREVIEW_SCALE = 3   # 28x28 미리보기 확대 배율

BG = "#0f1116"
SURFACE = "#171a22"
SURFACE_2 = "#1f2430"
BORDER = "#2a3040"
TEXT = "#e7eaf0"
TEXT_DIM = "#99a2b3"
ACCENT = "#6ea8fe"
ACCENT_DIM = "#2b4a7d"


class RecognizerApp:
    def __init__(self, root: tk.Tk, model: MLP):
        self.root = root
        self.model = model
        self.last_xy: tuple[int, int] | None = None

        # tkinter Canvas는 픽셀을 되읽을 수 없어서, 같은 획을 PIL 이미지에도
        # 그려 두고 추론할 때는 그쪽을 읽는다.
        self.image = Image.new("L", (PAD_SIZE, PAD_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)

        root.title("손글씨 숫자 인식")
        root.configure(bg=BG)
        root.resizable(False, False)

        self._build_ui()
        self.clear()

    # ---------- 화면 구성 ----------

    def _build_ui(self) -> None:
        big = tkfont.Font(family="Helvetica", size=64, weight="bold")
        head = tkfont.Font(family="Helvetica", size=11, weight="bold")
        body = tkfont.Font(family="Helvetica", size=12)
        small = tkfont.Font(family="Helvetica", size=10)

        outer = tk.Frame(self.root, bg=BG, padx=20, pady=18)
        outer.pack(fill="both", expand=True)

        tk.Label(
            outer, text="손글씨 숫자 인식", bg=BG, fg=TEXT,
            font=tkfont.Font(family="Helvetica", size=18, weight="bold"),
        ).pack(anchor="w")
        tk.Label(
            outer,
            text=f"{' → '.join(map(str, self.model.sizes))}  ·  "
                 f"파라미터 {sum(w.size for w in self.model.W) + sum(b.size for b in self.model.b):,}개",
            bg=BG, fg=TEXT_DIM, font=small,
        ).pack(anchor="w", pady=(0, 14))

        columns = tk.Frame(outer, bg=BG)
        columns.pack(fill="both", expand=True)

        # --- 왼쪽: 그리기 ---
        left = tk.Frame(columns, bg=SURFACE, padx=14, pady=14,
                        highlightbackground=BORDER, highlightthickness=1)
        left.pack(side="left", anchor="n")

        tk.Label(left, text="그리기", bg=SURFACE, fg=TEXT_DIM, font=head).pack(anchor="w", pady=(0, 8))

        self.canvas = tk.Canvas(
            left, width=PAD_SIZE, height=PAD_SIZE,
            bg="white", highlightthickness=0, cursor="crosshair",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        tk.Button(
            left, text="지우기", command=self.clear,
            bg=SURFACE_2, fg=TEXT, font=body, relief="flat",
            activebackground=BORDER, activeforeground=TEXT,
            highlightbackground=SURFACE, cursor="hand2",
        ).pack(fill="x", pady=(10, 4))

        tk.Label(
            left, text="칸을 크게 채워 한 글자만 그리세요.",
            bg=SURFACE, fg=TEXT_DIM, font=small,
        ).pack(anchor="w")

        # --- 오른쪽: 결과 ---
        right = tk.Frame(columns, bg=SURFACE, padx=14, pady=14,
                         highlightbackground=BORDER, highlightthickness=1)
        right.pack(side="left", anchor="n", padx=(16, 0), fill="y")

        tk.Label(right, text="인식 결과", bg=SURFACE, fg=TEXT_DIM, font=head).pack(anchor="w", pady=(0, 8))

        result = tk.Frame(right, bg=SURFACE)
        result.pack(anchor="w", fill="x")
        self.digit_label = tk.Label(result, text="–", bg=SURFACE, fg=BORDER, font=big, width=2)
        self.digit_label.pack(side="left")
        meta = tk.Frame(result, bg=SURFACE)
        meta.pack(side="left", padx=(12, 0))
        self.conf_label = tk.Label(
            meta, text="—", bg=SURFACE, fg=TEXT,
            font=tkfont.Font(family="Helvetica", size=16, weight="bold"),
        )
        self.conf_label.pack(anchor="w")
        self.conf_note = tk.Label(meta, text="아직 그린 내용이 없습니다",
                                  bg=SURFACE, fg=TEXT_DIM, font=small)
        self.conf_note.pack(anchor="w")

        # --- 확률 막대 ---
        bars = tk.Frame(right, bg=SURFACE)
        bars.pack(anchor="w", pady=(14, 0), fill="x")
        self.bars: list[dict] = []
        for d in range(10):
            row = tk.Frame(bars, bg=SURFACE)
            row.pack(fill="x", pady=1)
            label = tk.Label(row, text=str(d), bg=SURFACE, fg=TEXT_DIM, font=small, width=2)
            label.pack(side="left")
            track = tk.Canvas(row, width=180, height=10, bg=SURFACE_2,
                              highlightthickness=0)
            track.pack(side="left", padx=6)
            fill = track.create_rectangle(0, 0, 0, 10, fill=ACCENT_DIM, width=0)
            pct = tk.Label(row, text="—", bg=SURFACE, fg=TEXT_DIM, font=small, width=6, anchor="e")
            pct.pack(side="left")
            self.bars.append({"label": label, "track": track, "fill": fill, "pct": pct})

        # --- 모델 입력 미리보기 ---
        tk.Label(right, text="모델이 보는 이미지", bg=SURFACE, fg=TEXT_DIM, font=head).pack(
            anchor="w", pady=(16, 8)
        )
        self.preview = tk.Canvas(
            right, width=GRID * PREVIEW_SCALE, height=GRID * PREVIEW_SCALE,
            bg="black", highlightbackground=BORDER, highlightthickness=1,
        )
        self.preview.pack(anchor="w")
        self._preview_image: ImageTk.PhotoImage | None = None

    # ---------- 그리기 ----------

    def on_press(self, event: tk.Event) -> None:
        self.last_xy = (event.x, event.y)
        # 점 하나만 찍어도 자국이 남도록 아주 짧은 선을 그린다
        self._stroke(event.x, event.y, event.x, event.y)

    def on_drag(self, event: tk.Event) -> None:
        if self.last_xy is None:
            self.last_xy = (event.x, event.y)
        x0, y0 = self.last_xy
        self._stroke(x0, y0, event.x, event.y)
        self.last_xy = (event.x, event.y)

    def on_release(self, _event: tk.Event) -> None:
        self.last_xy = None
        self.predict()

    def _stroke(self, x0: int, y0: int, x1: int, y1: int) -> None:
        r = STROKE_WIDTH / 2
        self.canvas.create_line(
            x0, y0, x1, y1, width=STROKE_WIDTH, fill="#101014",
            capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=True,
        )
        self.draw.line((x0, y0, x1, y1), fill=0, width=STROKE_WIDTH)
        # PIL의 line은 끝을 둥글게 하지 않으므로 양끝에 원을 덧그린다
        for cx, cy in ((x0, y0), (x1, y1)):
            self.draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=0)

    def clear(self) -> None:
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, PAD_SIZE, PAD_SIZE), fill=255)
        self.digit_label.config(text="–", fg=BORDER)
        self.conf_label.config(text="—")
        self.conf_note.config(text="아직 그린 내용이 없습니다")
        for d, bar in enumerate(self.bars):
            bar["track"].coords(bar["fill"], 0, 0, 0, 10)
            bar["track"].itemconfig(bar["fill"], fill=ACCENT_DIM)
            bar["pct"].config(text="—", fg=TEXT_DIM)
            bar["label"].config(fg=TEXT_DIM)
        self.preview.delete("all")
        self._preview_image = None

    # ---------- 추론 ----------

    def predict(self) -> None:
        # PIL 이미지는 흰 배경(255)에 검은 글씨(0)이므로 반전해 잉크를 1.0으로 만든다
        gray = 1.0 - np.asarray(self.image, dtype=np.float64) / 255.0
        vec = normalize(gray)
        if vec is None:
            return

        probs, _ = self.model.forward(vec.reshape(1, -1))
        probs = probs[0]
        digit = int(probs.argmax())
        confidence = float(probs[digit])

        self.digit_label.config(text=str(digit), fg=ACCENT)
        self.conf_label.config(text=f"{confidence * 100:.1f}%")
        self.conf_note.config(
            text="확실합니다" if confidence > 0.9
            else "아마도 이 숫자입니다" if confidence > 0.6
            else "헷갈립니다 — 더 크게 그려 보세요"
        )

        for d, bar in enumerate(self.bars):
            top = d == digit
            bar["track"].coords(bar["fill"], 0, 0, 180 * float(probs[d]), 10)
            bar["track"].itemconfig(bar["fill"], fill=ACCENT if top else ACCENT_DIM)
            bar["pct"].config(text=f"{probs[d] * 100:.1f}%", fg=TEXT if top else TEXT_DIM)
            bar["label"].config(fg=TEXT if top else TEXT_DIM)

        self._show_preview(vec)

    def _show_preview(self, vec: np.ndarray) -> None:
        arr = (np.clip(vec.reshape(GRID, GRID), 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L").resize(
            (GRID * PREVIEW_SCALE, GRID * PREVIEW_SCALE), Image.NEAREST
        )
        # PhotoImage는 참조를 붙들고 있지 않으면 가비지 컬렉션으로 사라진다
        self._preview_image = ImageTk.PhotoImage(img)
        self.preview.delete("all")
        self.preview.create_image(0, 0, anchor="nw", image=self._preview_image)


def main() -> None:
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    if not model_path.exists():
        raise SystemExit(
            f"학습된 모델이 없습니다: {model_path}\n"
            f"먼저 학습하세요:  python -m src.train"
        )
    model = MLP.load(model_path)
    root = tk.Tk()
    RecognizerApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
