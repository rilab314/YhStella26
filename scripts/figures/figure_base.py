"""그림 생성기의 공통 뼈대 (design 15.3절).

**미구현 — 설계만 적어 둔다.**

논문 그림은 성질이 두 갈래고, 그래서 베이스도 둘이다.

| 갈래 | 무엇 | 출력 | 예 |
| --- | --- | --- | --- |
| `SampleFigure` | 샘플마다 조건을 재고 통과한 것만 그린다 | 최대 100장 | Fig. 1·9·11·12 |
| `PlotFigure` | 데이터 전체를 집계해 그래프 한 장 | 1장 (+csv) | Fig. 5c·8c·10·13·14 |
| (베이스 없음) | 개념도 — 코드가 아니라 그리기 도구로 만든다 | — | Fig. 2·3·4·7 |

---

# 0. 출력 경로와 덮어쓰기 규약

    PAPER_ROOT  = .../Ongoing/2026_stella/paper       ← 데이터 작업 폴더. 저장소가 아니다
    FIGURE_ROOT = PAPER_ROOT / "figure"

    def figure_dir(name) -> Path:
        '''FIGURE_ROOT/name 을 **지우고 다시 만들어** 돌려준다.'''

**폴더 단위로 지우고 다시 만든다** (사용자 지시). `figure_01` 을 다시 그리면
`figure/figure_01/` 을 통째로 `rmtree` 하고 새로 `mkdir` 한다. 파일 하나씩 덮어쓰면
사고가 난다 — 지난 실행에서 3위였다가 이번에 조건을 통과하지 못한 그림이 그대로 남아
"이번 결과"에 섞인다. 지우는 단위는 **그림 하나**이므로 다른 그림 폴더는 건드리지 않는다.

그림은 실행마다 수백 MB 가 새로 만들어진다 — git 이력에 넣을 것이 아니다.
**논문에 실제로 고른 몇 장만** `docs/figures/` 로 옮겨 커밋한다.

> `PAPER_ROOT` 는 `tables/table_base.py` 에도 같은 값이 있다. `scripts/` 는 패키지가 아니라
> 두 폴더가 서로를 import 할 수 없어서다(`sys.path` 조작은 하지 않는다 — CLAUDE.md).
> **경로를 바꾸면 두 곳을 함께 고친다.** 상수 하나뿐이라 이 중복이 폴더를 하나 더 만드는
> 것보다 싸다고 판단했다.

---

# 1. `SampleFigure` — 정성 그림의 선별 규약

## 왜 선별하는가

**데이터 전체를 렌더링해 눈으로 고르지 않는다.** val 1,218장을 다 그려 놓고 넘겨 보면
(1) 시간이 오래 걸리고 (2) 무엇을 기준으로 골랐는지 논문에 쓸 수 없다.
그림마다 **그 현상이 실제로 일어난 프레임만** 정량 조건으로 고른다.

## 흐름

    run()
      ├─ figure_dir(name)                폴더를 지우고 다시 만든다
      ├─ scan()                          split 의 모든 stem 을 훑는다
      │    └─ measure(stem)              build_figure() 호출 → None 이면 탈락
      │         └─ build_figure(stem)    (이미지, 파일명 꼬리, 정렬 점수) 또는 None
      ├─ sort(점수 내림차순)
      ├─ write(상위 CAP개)               001_{stem}{suffix}.png
      └─ report()                        "훑은 N장 중 후보 M장, 상위 K장 저장"

## 규약

1. **상한 `CAP = 100`** (사용자 지시). 사람이 100장까지는 넘겨 보고 고를 수 있다.
   조건이 100장보다 훨씬 많이 통과하면 그것은 **조건이 헐거운 것**이므로 조건을 조인다.
2. **파일명 앞 세 자리가 순위**다 — `001_`이 가장 좋은 후보. 뒤에 타일 id 와 판정 수치를
   붙여, 파일 이름만 보고 왜 뽑혔는지 알 수 있게 한다.
   예: `003_126.6219,37.3851_f1-742.png` (프레임 F1 0.742)
3. **점수는 그림마다 다르다.** Fig. 1은 프레임 F1, Fig. 9는 ΔF1, Fig. 12는 유형 강도.
   무엇으로 정렬했는지 각 스크립트 docstring 첫 문단에 반드시 쓴다.
4. **이미지는 한 번만 만든다.** 조건 판정 중에 완성 이미지를 메모리에 PNG 로 인코딩해 두고,
   상위 100장만 디스크에 쓴다. 조건 판정과 렌더링을 두 번 돌리지 않는다.
5. **판넬 사이 여백은 검정 20 px**, 원본 해상도(768) 그대로 붙인다. 모든 그림이 같다.

## 입력은 예측 캐시다 — 파이프라인을 다시 돌리지 않는다

`scripts/dump_predictions.py`가 떨군 희소 캐시(`.../pred_cache/`)를 읽는다.
**GPU 를 쓰지 않고, 학습이 도는 중에도 돌아간다**(D 트랙 규약, CLAUDE.md).
캐시에는 영상이 없으므로 파일 이름(stem)으로 데이터셋에서 영상만 다시 읽는다 —
`scripts/viz_cache.py`가 이미 쓰는 방식과 같다.

**어느 split 을 쓰나.** 방법 설명용 그림(Fig. 1·4·6·7)은 `val`, 결과용 그림
(Fig. 9·11·12)은 **`test`** 다 — 논문이 수치를 보고하는 split 과 그림이 어긋나면 안 된다.

## 렌더링은 하나의 규칙을 공유한다

예측이든 GT 든, 우리 것이든 baseline 것이든 **같은 함수로 그린다**. 그래야 그림에서 보이는
차이가 알고리즘의 차이로 읽힌다. 클래스 색 폴리라인 + **양 끝점에 흰 테두리 원**.
끝점 원이 이 논문의 장치다 — 선만 그리면 한 선이 4조각 났는지 보이지 않지만,
끝점 원을 찍으면 점이 8개라 **조각남이 눈으로 세진다.**
`stella/train/viz.py`가 이미 하는 일이라 그것을 재사용한다.

## 설계할 클래스

    class SampleFigure:
        name: str          # "figure_01" — 폴더 이름이자 파일 이름
        split: str         # "val" | "test"
        cap: int = 100

        def run(self) -> None
        def scan(self) -> list[tuple[float, str, str, bytes]]
        def measure(self, stem) -> tuple | None
        def build_figure(self, stem, sample, pred) -> tuple | None   # 하위 클래스가 구현
        def write(self, ranked) -> int
        def report(self, kept, n_candidates, n_scanned) -> None

---

# 2. `PlotFigure` — 집계 그림의 규약

## 흐름

    run()
      ├─ figure_dir(name)
      ├─ collect()      수치를 모은다 (metrics.csv · 라벨 JSON · 캐시 · 판정 결과 CSV)
      ├─ save_csv()     그린 값을 그대로 CSV 로 남긴다   ← 여기가 요점
      └─ plot()         figure_XX.png (본문 확인용) + figure_XX.pdf (제출용 벡터)

## 규약

1. **그린 값을 반드시 CSV 로 같이 남긴다.** 그림의 숫자와 본문의 숫자가 어긋나는 사고는
   논문에서 가장 자주 나오고 가장 창피하다. CSV 가 있으면 본문을 쓸 때 그것을 보고 옮긴다.
2. **표가 이미 그 값을 갖고 있으면 표의 CSV 를 읽는다.** 다시 계산하지 않는다 —
   표와 그림이 어긋날 수 있는 경로 자체를 없앤다.
3. **PDF 를 같이 낸다.** IEEE 는 벡터 그림을 요구하고, 확대해도 글자가 깨지지 않아야 한다.
4. **폰트 크기는 본문 캡션보다 작지 않게.** 1단 그림은 폭 3.5 in, 2단은 7.16 in 로 그린다
   (IEEE Access 2단 규격). 그려 놓고 줄이면 글자가 읽히지 않는다.
5. 색은 흑백 인쇄에서도 갈리게 — 색만이 아니라 **선 모양·마커도 같이 바꾼다.**

## 설계할 클래스

    class PlotFigure:
        name: str
        width_in: float     # 3.5 (1단) | 7.16 (2단)
        height_in: float

        def run(self) -> None
        def collect(self) -> "pandas.DataFrame"    # 하위 클래스가 구현
        def plot(self, frame, axes) -> None        # 하위 클래스가 구현
        def save(self, frame, figure) -> None      # csv + png + pdf
"""
