"""논문 그림·표의 출력 경로 단일 출처 (design 15.2절).

**미구현 — 설계만 적어 둔다.** 아래 규약을 확정한 뒤 구현한다.

## 어디에 저장하나

저장소가 아니라 **데이터 작업 폴더**에 쌓는다. 그림은 수백 MB가 되고 실행마다 통째로
다시 만들어지므로 git 이력에 넣을 것이 아니다. 저장소에는 **논문에 실제로 고른 몇 장만**
`docs/figures/`로 옮겨 커밋한다.

    {PAPER_ROOT} = .../Ongoing/2026_stella/paper/
    ├── figure/
    │   ├── figure_01/   001_{stem}_{score}.png ... (최대 100장)
    │   ├── figure_05/   figure_05.png · figure_05.pdf · figure_05.csv
    │   └── ...
    └── table/
        ├── table_01/    table_01.csv · table_01.md
        └── ...

## 덮어쓰기 규약 (사용자 지시)

**폴더 단위로 지우고 다시 만든다.** `figure_01`을 다시 그리면 `figure/figure_01/`을
통째로 `rmtree` 하고 새로 `mkdir` 한다 — 옛 실행의 잔재가 섞여 남지 않는다.
지난 실행의 그림이 랭킹 3위였는데 이번 실행에서 탈락했다면 **그 파일은 사라져야 한다.**
파일 하나씩 덮어쓰면 그 파일이 남아 "이번 결과"를 오염시킨다.

다른 그림(`figure_02`)은 건드리지 않는다 — 지우는 단위는 **그림 하나**다.

## 이름 규약

- 폴더·파일 이름은 **두 자리 0 채움**(`figure_01`, `table_09`)이다. 그림이 14개라
  `figure_1`과 `figure_14`가 이름순으로 붙어 버리는 것을 막는다.
- 논문의 표 번호는 로마숫자(Table I~IX)이고, 스크립트는 `table_01`~`table_09`가 그것에
  1:1 대응한다.

## 설계할 함수

    PAPER_ROOT   : Path      데이터 작업 폴더 아래 paper/
    FIGURE_ROOT  : Path      PAPER_ROOT / "figure"
    TABLE_ROOT   : Path      PAPER_ROOT / "table"

    def reset_dir(root: Path, name: str) -> Path:
        '''root/name 을 지우고 다시 만들어 돌려준다. 위 "덮어쓰기 규약"의 유일한 구현부.'''

    def figure_dir(name: str) -> Path:   # reset_dir(FIGURE_ROOT, name)
    def table_dir(name: str) -> Path:    # reset_dir(TABLE_ROOT, name)

`PAPER_ROOT`는 config가 아니라 이 파일이 정한다 — 실험 config(`configs/`)는 학습 설정이고,
논문 산출물 경로는 학습과 무관하기 때문이다. 경로를 바꿀 일이 생기면 여기 한 줄만 고친다.
"""
