"""표 생성기의 공통 뼈대 (design 15.4절).

**미구현 — 설계만 적어 둔다.**

## 왜 표도 스크립트로 만드나

표를 손으로 옮겨 적으면 **실험을 다시 돌릴 때마다 옮겨 적어야 하고, 한 번은 반드시 틀린다.**
그리고 논문 심사에서 "이 값은 어디서 나왔나"에 답할 수 없다. 표마다 스크립트를 두면
그 질문의 답이 곧 스크립트 경로다.

**사람이 정하는 값도 스크립트에 둔다.** Table I(표현 방식 비교)처럼 논문 저자의 판단으로
채우는 표도 그 내용을 스크립트 안의 상수 표로 두고 렌더링만 코드가 한다 — 그래야
"어느 파일을 고쳐야 하나"가 하나로 남는다.

## 출력 경로와 덮어쓰기 규약

    PAPER_ROOT = .../Ongoing/2026_stella/paper       ← 데이터 작업 폴더. 저장소가 아니다
    TABLE_ROOT = PAPER_ROOT / "table"

    def table_dir(name) -> Path:
        '''TABLE_ROOT/name 을 **지우고 다시 만들어** 돌려준다.'''

**폴더 단위로 지우고 다시 만든다** (사용자 지시). 지우는 단위는 **표 하나**다.

> `PAPER_ROOT` 는 `figures/figure_base.py` 에도 같은 값이 있다. `scripts/` 는 패키지가 아니라
> 두 폴더가 서로를 import 할 수 없어서다. **경로를 바꾸면 두 곳을 함께 고친다.**

## 흐름

    run()
      ├─ table_dir(name)           폴더를 지우고 다시 만든다
      ├─ collect()                 수치를 모은다 (하위 클래스가 구현)
      ├─ format()                  열 이름·자릿수·빈칸 기호를 논문 형식으로
      └─ save()                    table_XX.csv + table_XX.md
                                   (+ table_XX.tex 는 원고 작성 단계에서 추가)

## 규약

1. **두 파일을 낸다.** `.csv` 는 다른 스크립트(그림)가 읽는 기계용, `.md` 는 사람이 읽고
   원고에 붙여 넣는 용도다. 둘의 숫자는 같은 `format()` 을 거친 **같은 값**이다.
2. **자릿수는 표마다 고정한다.** F1 은 소수 넷째 자리(0.4303), 백분율은 소수 첫째 자리
   (+19.0%), 시간은 정수 ms. 반올림을 표시 단계에서만 하고 CSV 에는 원값을 함께 남긴다.
3. **빈칸은 `–`(en dash)** 로 쓴다. 0 과 "아직 안 쟀음"이 같은 칸에 보이면 안 된다.
4. **실험이 아직 없으면 빈 표를 내고 무엇이 없는지 출력한다.** 스크립트가 죽으면 안 된다 —
   "Table V 는 baseline 학습 2건이 없어 4행이 비었다"가 화면에 찍혀야 한다.
5. **비교하는 행은 규격이 같아야 한다.** U 규격(3,000장·10에폭)과 F 규격(전체 데이터)의
   절대값을 한 표에 섞지 않는다(CLAUDE.md). 표에 규격 열을 두거나, 표를 나눈다.
6. **읽는 곳을 하나로.** 학습 결과는 `stella/eval/runlog.py`, 디코더 결과는
   `stella/decode/sweep.py`를 통해서만 읽는다. 표가 `metrics.csv`를 직접 파싱하면
   판정 스크립트(`scripts/judge_round.py`)와 값이 갈라진다.

## 설계할 클래스

    class PaperTable:
        name: str                  # "table_05"
        columns: list[str]         # 논문에 실릴 열 이름
        align: list[str]           # markdown 정렬 (---, :---:, ---:)

        def run(self) -> None
        def collect(self) -> "pandas.DataFrame"    # 하위 클래스가 구현
        def format(self, frame) -> "pandas.DataFrame"
        def save(self, frame) -> None              # csv + md
        def report_missing(self, frame) -> None    # 규약 4

## 값의 출처 (표마다 다르다 — 각 스크립트 docstring 에 명시한다)

| 출처 | 무엇 | 읽는 곳 |
| --- | --- | --- |
| 학습 로그 | 실행별 val 지표 | `stella/eval/runlog.py` (`metrics.csv` 판독) |
| 예측 캐시 | 디코더 설정을 바꿔 다시 잰 지표 | `stella/decode/sweep.py` |
| 라벨 JSON | 클래스 빈도·길이 분포 | `scripts/stat_labels.py` |
| config | 파라미터 값 | `configs/` dataclass 를 직접 읽는다 |
| 코드 | 텐서 shape·dtype | `stella/model/stella.py` 의 출력 계약 |
| 손으로 | 관련연구 비교 매트릭스 | 스크립트 안의 상수 표 |
"""
