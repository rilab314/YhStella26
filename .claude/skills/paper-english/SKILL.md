---
name: paper-english
description: 논문 영어 본문을 새로 쓰거나 고칠 때 지켜야 할 문체·문장·용어 규칙. abstract, introduction, related work, method, 실험 결과 분석, 결론, figure/table 캡션, cover letter, biography, 리뷰 답변서 등 영문 원고 텍스트를 작성·재작성·축약·폴리싱·번역할 때 항상 먼저 읽는다. English academic writing style for manuscripts.
---

# 논문 영어 본문 작성 규칙

이 규칙은 같은 연구실의 이전 원고 작업에서 실제로 받은 지적들을 일반화한 것이다.
영문 원고 텍스트를 **한 문장이라도** 만들거나 고칠 때 적용한다.
`CLAUDE.md`의 작성 규칙에 딸린 세부 규칙이며, 충돌하면 사용자의 그 세션 지시가 우선한다.

## 0. 이 저장소에서의 자리

- **영문 원고 본문은 `docs/manuscript/` 아래에만 쓴다.** 이 폴더는 `.gitignore` 대상이라 저장소에
  올라가지 않는다 — 원고는 파일로 남기되 버전 관리는 사용자가 따로 한다.
- **한글 설계안은 `docs/paper_outline.md`가 단일 출처다.** 절 구성·기여·그림/표 번호·초록 골격이
  거기 있다. 영문 본문은 그 설계안을 옮겨 쓰는 것이지 새로 정하는 것이 아니다.
  설계안과 다르게 쓸 이유가 생기면 먼저 설계안을 고친다.
- 구현 근거는 `docs/1_structure.md` ~ `docs/6_paper_assets.md`, 변경 내역은 `docs/history.md`.

## 0.1. 작업 순서

1. **고칠 파일을 먼저 다시 읽는다.** 기억한 내용으로 쓰지 않는다.
2. **수치는 원본에서 읽는다.** 이 저장소의 수치 원본은 실행 폴더의 `metrics.csv`
   (`.../2026_stella/log/{실행}/metrics.csv`, 판독은 `scripts/show_run.py`·`scripts/summarize_runs.py`)와
   `scripts/tables/table_NN.py`가 뽑는 표다. 설계안(`docs/paper_outline.md`)에 옮겨 적힌 값과
   다르면 **원본이 맞다.**
3. **아직 측정하지 않은 수치는 지어내지 않는다.** `XX.X` 같은 자리표시자를 그대로 두고,
   그 자리표시자 목록을 원고 맨 끝(또는 채팅)에 모아 밝힌다.
4. 쓴다.
5. **자기 점검**: 아래 §11 체크리스트를 실제로 훑는다. 특히 세미콜론·문장 길이·괄호 수치.
6. **출력 방식은 그 세션의 지시를 따른다.** 지시가 없으면 채팅에만 낸다(영문 본문 + 그 아래 한글 번역).
   원고 파일에 바로 반영하라는 지시가 있으면 `docs/manuscript/` 아래 파일에 쓰고,
   무엇을 고쳤는지 채팅에 한글로 요약한다.

번역을 붙이는 이유는 확인용만이 아니다. **영어 문장과 한글 문장이 1:1로 대응되지 않으면 영어가 너무 꼬인 것이다.**
한글로는 세 문장인데 영어가 한 문장이면 영어를 세 문장으로 나눈다. 실제로 이 지적을 받은 적이 있다.
파일에 바로 쓰는 경우에도 이 기준은 그대로다 — 한 문장이 한글로 옮겨지지 않으면 그 영어 문장을 나눈다.

## 1. 독자와 목표 문체

- 독자는 **비영어권 공대 대학생**이다. 공학 전공서적을 영어로 읽는 수준이면 막힘없이 읽혀야 한다.
- **기교 없이, 평이한 단어로, 논리적이고 사실적으로만** 쓴다.
- 문장이 멋있는지는 평가 기준이 아니다. **한 번에 읽고 뜻이 잡히는지**가 유일한 기준이다.

### 1.1. 쉬운 단어를 고른다

같은 뜻이면 짧고 흔한 단어를 쓴다. 학술적으로 보이려고 어려운 동의어를 쓰지 않는다.

| 쓰지 않는다                                       | 쓴다                   |
| -------------------------------------------- | -------------------- |
| utilize, employ, leverage                    | use                  |
| demonstrate, exhibit, illustrate(수치가 주어일 때)  | show                 |
| facilitate                                   | help, make easier    |
| in order to                                  | to                   |
| prior to / subsequent to                     | before / after       |
| due to the fact that                         | because              |
| in the event that                            | if                   |
| a large number of / a plurality of           | many, several        |
| the majority of                              | most                 |
| is capable of / is able to                   | can                  |
| perform a comparison / conduct an evaluation | compare / evaluate   |
| terminate, initiate                          | end, start           |
| aforementioned                               | this / these + 명사    |
| thereby, hence, whereby                      | 문장을 나누고 so / because |
| moreover, furthermore                        | also (또는 접속어 없이)     |

- `significant`는 통계 검정을 한 경우가 아니면 쓰지 않는다. `large`, `clear`가 맞다.
- `novel`, `robust`, `state-of-the-art`는 남발하지 않는다. 근거가 있을 때 한 번만.

### 1.2. 수사·비유·과장을 쓰지 않는다

- 비유·의인화·관용구 금지. 지표가 "오른다"는 표현도 사실 수치로 바꾼다.
  - 나쁜 예: `AP climbs sharply after merging.`
  - 좋은 예: `Merging raises the score by more than ten points.`
- `dramatically`, `remarkably`, `surprisingly`, `it is worth noting that`, `as we can see` 같은 군더더기는 지운다.
- 변명조로 쓰지 않는다. 한계는 **사실만** 한 문장으로 적는다.
  - 나쁜 예: `Unfortunately, we could not reproduce the baseline because the code is unavailable, which is beyond our control.`
  - 좋은 예: `The post-processing code of that method is not public, so we could not reproduce it on SEED-MAP.`

## 2. 문장 구조

### 2.1. 한 문장에 한 가지 생각

- 25단어를 넘으면 자를 곳이 있는지 본다. 평균 15~20단어를 목표로 한다.
- **세미콜론(`;`)은 원칙적으로 쓰지 않는다.** 두 문장으로 나눌 수 있으면 거의 항상 나눈다.
  - `... in one scene; the other rows show failures.` → `... in one scene. The other rows show failures.`
- 콜론(`:`)은 목록이나 패널 라벨(`Left:`, `green:`)에만 쓴다. 문장을 잇는 데 쓰지 않는다.
- 쉼표로 절을 계속 매달지 않는다. 관계절·분사구문이 두 개 이상 붙으면 문장을 쪼갠다.

실제 사례:

```
나쁨: For each subsequent line we resample it at a fixed spacing and measure, at every
      sample point, its distance to the reference lines, which is used in the next step.
좋음: Each subsequent line is resampled at a fixed spacing. At every sample point we
      measure its distance to the reference lines. This distance is denoted by $d_k$.
```

### 2.2. 어려운 구문을 쓰지 않는다

- **도치·가정법 금지.** 평서문으로 쓴다.
  - 나쁜 예: `Were the two rails recovered consistently, we could keep both; because they overlap irregularly, ...`
  - 좋은 예: `If both rails were extracted cleanly, we could keep them. In practice they overlap irregularly. We therefore keep one representative line.`
- **중첩 관계절 금지.** `the maximal run of weak samples that contains a contiguous strong sub-run` 같은 구조는 단계별 문장으로 푼다.
  - 좋은 예: `A strong run longer than $\ell_{\text{div}}$ becomes a core. The core is then extended over the weak samples on both sides. The extended range is taken as a free segment.`
- **명사 3개 이상 연속(noun stack) 금지.** `lane marking instance matching threshold` → `the threshold used to match lane marking instances`.
- 수동태는 행위자가 중요하지 않을 때만 쓴다. 주어가 명확하면 능동태가 낫다.
- 판정·정의에는 뉘앙스를 넣는다. `A sample is "overlapping"` → `A sample is considered "overlapping"`.

### 2.3. 지시어를 모호하게 두지 않는다

- `it`, `this`, `they`, `the former/latter`가 무엇을 가리키는지 앞 문장을 다시 읽어야 하면 실패다.
- `this` 뒤에는 항상 명사를 붙인다. `This improves ...` → `This trimming step improves ...`.

## 3. 문단 구조

- **결론·주장을 첫 문장에 쓴다.** 근거·수치·부연은 그 뒤에 붙인다.
- 한 문단은 한 주제, 3~6문장. 두 주제가 섞이면 문단을 나눈다.
- 절(subsection)이 시작되면 그 절이 무엇을 다루는지 한 문장으로 먼저 알린다.
- 같은 내용을 두 곳에서 설명하지 않는다. 원리는 한 번만 못 박고, 뒤에서는 그 원리를 참조만 한다.
- **바로 앞 문장을 되풀이하거나, 앞 문장이 이미 함의하는 내용을 풀어 쓰는 문장을 쓰지 않는다.**
  근거를 덧붙이려다 같은 말을 다시 하게 되면 앞 문장에 절 하나로 합치거나 지운다.
  - 나쁜 예: `... the heading is updated from a point one look-back span behind the tip.`
    다음 문단이 `Thinning a thick blob leaves small zigzags. A heading taken from the nearest point
    alone would follow these zigzags, whereas measuring it over a longer span averages them out.`
  - 좋은 예: `... the heading is updated from a point one look-back span behind the tip. The
    immediately preceding point would follow the small zigzags that thinning leaves along the
    skeleton, while a longer span averages them out.`
- **소제목은 정말 필요할 때만 둔다.** 문단 첫 문장이 이미 주제를 밝힌다면 볼드 소제목은 군더더기다.
  소제목이 약어 정의를 대신하고 있었다면, 소제목을 없애고 첫 문장에 동격구(`B1, the
  watershed-instances baseline, ...`)로 정의를 넣는다.
- **리뷰 지적에 전용 문단을 만들지 않는다.** 지적받은 내용은 그것이 원래 있어야 할 자리에
  한두 문장으로 넣는다. 오해를 풀겠다고 해명 문단을 새로 만들면 논문이 심사 기록처럼 읽힌다.
  - 나쁜 예: `The word *averaging* can mean two things here, and only one of them applies. ...` (6문장 문단)
  - 좋은 예: smoothing을 설명하는 자리에서 `This is the only place in the pipeline where points are
    averaged, and the average runs along one lane.` (2문장)

## 4. 뜻을 숨기지 않는다

독자가 뒤 문장까지 읽고 앞 문장을 역추론하게 만들면 안 된다. 가리키는 대상을 그 자리에서 밝힌다.

```
나쁨: Two post-processing steps are applied to the output only, so that they never
      restrict the merging that follows.
좋음: After the merging, noise removal and smoothing are applied to the output.
```

- **정의 없이 용어를 꺼내지 않는다.** `surviving group`처럼 처음 등장하는 용어는 그 앞에서 한 문장으로 정의한다.
- 약어는 첫 등장에서 풀어 쓴다. 이후에는 약어만 쓴다.
- 오해를 부르는 표현을 쓰지 않는다.
  - 나쁜 예: `Dashed markings are fragmented by design.` (일부러 끊었다는 뜻으로 읽힌다)
  - 좋은 예: `Because they are dashed markings, they naturally appear as many separate dashes.`

## 5. 인과 서술

- **무엇이 왜 그렇게 되는지를 평이한 말로 먼저** 전달하고, 수치는 근거로 뒤에 붙인다.
- 인과는 쓰기 전에 데이터로 검증한다. 그럴듯한 설명이 틀린 경우가 실제로 여러 번 있었다.
  - 예: `merge_ratio > 1`(과병합)인데 "조각남(fragmentation)"으로 서술한 오류.
    조각남이면 예측 수가 GT보다 많고 precision이 recall보다 훨씬 낮아야 한다. 표가 그렇지 않았다.
  - 지표 하나로 두 가지 해석이 가능하면, 다른 열을 근거로 어느 쪽인지 못 박는다.
- 근거 없는 추세 주장을 하지 않는다. `the gains grow with ...`처럼 실험이 뒷받침하지 않는 문장은 쓰지 않는다.
- 주장 범위를 데이터가 보증하는 만큼으로 좁힌다.
  - 나쁜 예: `Our method produces no zigzags.`
  - 좋은 예: `Our method does not create the zigzags that come from averaging or reordering points.`
- **근거(인용·확립된 관행)를 먼저 말하고 나서 "우리는 이렇게 했다"를 쓴다.** 순서를 바꾸면 즉흥적으로
  고른 것처럼 읽힌다. `Our baseline uses X. This is standard practice [23].` → `X is standard
  practice [23]. Our baseline follows it.`
- **부정문으로 에둘러 말하지 않는다.** 같은 뜻이면 긍정형이 더 분명하다.
  `cannot come from a friendlier segmentation` → `comes only from the algorithm`.

## 6. 실험 결과 서술

- **괄호 수치를 매 문장에 붙이지 않는다.** 그 값은 이미 표에 있다.
  핵심 결과를 강조하거나 두 값을 직접 비교할 때만 골라 넣는다.
- 클래스·행을 전수 나열하지 않는다. **원리를 먼저 세우고 대표 사례만** 든다.
  (예: 두 regime을 정의한 뒤 각 regime에서 클래스 2개씩만 설명)
- 표에 있는 숫자를 문장으로 옮겨 적기만 하는 문단은 지운다. 본문은 표가 말하지 않는 것을 말해야 한다.
- 수치는 항상 원본 CSV와 대조해서 쓴다. 손으로 옮겨 적지 않는다.
- **단위를 한 문단 안에서 섞지 않는다.** 앞이 픽셀이면 뒤도 픽셀이다. 미터는 그 값이 도로 위의
  실제 크기로 읽혀야 의미가 생기는 자리에서만 쓴다. 픽셀 값에 미터를 습관적으로 병기하지 않는다.
- **수치를 대면 "그래서 어떻다"를 같은 문단에서 끝낸다.** 비교 대상만 대고 결론을 빠뜨리지 않는다.
  - 나쁜 예: `All three are below the 0.765 m width of the band used for evaluation.` (그래서?)
  - 좋은 예: `A lane is rasterized with a stroke of $t = 3$ px for evaluation, so smoothing shifts
    the geometry by much less than the metric can resolve.`
- **측정의 기준(basis)이 결과를 바꾸면 무엇을 쟀는지 밝힌다.** 예를 들어 좌표를 정수 픽셀로 반올림한
  뒤 변위를 재면 중앙값이 0이 된다. 그것은 "안 움직였다"가 아니라 "반올림으로 되돌아왔다"는 뜻이므로,
  연산 자체의 효과를 보고할 때는 반올림 전 값을 쓴다.
- **같은 현상을 지표 두 개로 중복 보고하지 않는다.** 사실상 같은 실패를 다른 단위로 잰다면 하나만
  남긴다. 서로 다른 실패 방향을 정확히 반씩 나누는 대칭 짝이면 둘 다 남긴다.
- **비교 기준 없는 비율(%)보다 개수가 낫다.** `32.2%`보다 `6,799 pairs`가 바로 읽힌다. 비율을 쓰려면
  무엇 대비인지 같은 문장에서 밝힌다.
- 소수 자릿수·지표 표기는 표와 본문에서 동일하게 맞춘다(`F1@0.5`, `36.20`).
- 그림을 설명할 때는 **그 그림에 실제로 보이는 것만** 쓴다.
  보이지 않는 중간 단계는 일반화해서 말한다.
  - 나쁜 예: (segmentation 열이 없는 그림에서) `The segmentation is sparse and broken here.`
  - 좋은 예: `The prediction zigzags in this row. This usually traces back to the segmentation, which in scenes like this tends to be sparse.`

## 7. 버전 비교를 쓰지 않는다

논문 독자는 이 알고리즘을 처음 본다. 이전 버전·이전 실험과의 비교는 본문에 쓰지 않는다.

- `is now balanced` → `is balanced`
- `unlike our previous pipeline, ...`, `(updated)`, `we no longer use ...` 전부 삭제
- 현재 동작만 정확히 기술한다. 개정 경위와 기각한 선택지는 `docs/5_decisions.md`에, 수정 이력은
  `docs/history.md`에 남긴다. 실험 경과는 `experiment/`와 PR 본문(`report` 스킬)이 맡는다.
- **본문 문장을 특정 시점에 못박지 않는다.** `in August 2026` 같은 표현은 시간이 지나면 낡아 보인다.
  외부 자원을 확인한 시점은 각주 인용("Checked 12 August 2026")에만 남긴다.
- **리뷰어를 본문에 등장시키지 않는다.** *"as suggested by the reviewer"*, *"in response to the review"*
  같은 표현을 쓰면 그 내용이 논문의 논리에서 나온 것이 아니라 심사의 부산물로 읽힌다.

## 8. 알고리즘 서술 깊이

- 구현 디테일을 전부 옮기지 않는다. 독자가 **아이디어를 이해**하는 것이 우선이다.
  - 예: 평행 판정의 SVD 절차(중심점·주축·법선 계산)는 본문에서 빼고,
    "길이 방향으로 많이 겹치고 가로 방향 거리가 가까우면 평행으로 본다"는 의도만 남긴다.
- 단, **핵심 메커니즘·정의·기준은 수식으로 정확히** 못 박는다. 부수적인 세부는 산문으로 요약한다.
- 상수·임계값은 의미를 먼저 말하고 값을 뒤에 준다. 코드 변수명을 그대로 쓰지 않는다.
- 절차가 여러 단계면 번호 순서대로 문장을 나눈다. 한 문장에 두 단계를 넣지 않는다.
- **의사코드(Algorithm)를 실었으면 본문에서 해설한다.** 줄 번호를 묶어 "몇 줄~몇 줄은 어느 절의 무엇"을
  한 문단으로 짚는다. 의사코드만 던져두면 독자가 두 번 읽어야 한다.
- **논문에 넣지 않는 구현 디테일**: seed를 어디서 고르는지, 동점일 때 무엇을 먼저 잡는지, 루프를 어떻게
  빠져나오는지 같은 것. 꼭 필요하면 의사코드의 주석 한 줄이나 본문 한 문장으로 끝낸다. 이것들로 소절을
  만들지 않는다.

### 8.1. 절차가 아니라 결과를 쓴다

**가장 자주 어긴 규칙이다.** 코드를 읽고 쓰면 루프 한 바퀴를 그대로 문장으로 옮기게 된다.
독자가 알고 싶은 것은 "매 반복마다 무엇을 하는가"가 아니라 **"끝나면 무엇이 되어 있는가"**다.

- 반복문·재귀·그리디 선택을 단계별로 서술하지 않는다. **결과의 성질**로 말한다.
- 그 단계가 **왜 있는지(무엇이 잘못돼서 이 처리가 필요한지)**를 먼저 쓰고, 무엇을 하는지를 뒤에 쓴다.
- 자료구조·정렬 순서·캐시처럼 결과에 드러나지 않는 것은 쓰지 않는다.

```
나쁨(루프 서술): We then append whole lines one after another. Each time we take the line
      whose nearest endpoint is closest to the current end of the lane, and we reverse
      its point order when it is attached by its far end.
좋음(결과 서술): Its lines are joined at the endpoints that lie closest to each other, and
      every line is turned to run the same way as the one it is joined to. The lane is
      therefore a single polyline that runs from one end of the group to the other.
```

```
나쁨(코드 순서 그대로): The longest line is taken as a reference. Each subsequent line is
      resampled at a fixed spacing.
좋음(의도 순서로):     Every center line is first resampled at a fixed spacing. The resampled
      lines are next sorted by length.
```

- 판정 규칙은 **무엇을 거부/채택하는지를 첫 문장에** 쓴다. 소제목도 그 판정 이름으로 짓는다.
  `Parallel rejection`은 무엇을 거부하는지 드러나지 않는다 → `Rejecting side-by-side pairs`.
- 여러 조건을 순서대로 검사하는 규칙은, 검사 순서가 아니라 **어떤 것이 통과하고 어떤 것이 걸리는지**로 쓴다.

### 8.2. 수식 표기 (LaTeX)

- 수식은 일반 텍스트로 흉내내지 말고 **TeX로 정확히** 쓴다.
- **인라인 vs 블록**: 짧은 기호·변수는 인라인(`$ ... $`), 길고 복잡한 수식·행렬은 블록(`$$ ... $$`)으로 분리한다.
- **`$$`는 단독 줄에 쓰고 블록 위아래에 빈 줄을 넣는다.** 빈 줄이 없으면 마크다운 뷰어에서 렌더링이 깨진다.
- 행렬은 `\begin{bmatrix} ... \end{bmatrix}`를 쓴다.
- 문장 안에서 수식이 절을 이루면 수식 뒤에 쉼표·마침표를 문법에 맞게 찍는다.
  뒤에 `where`로 이어지면 수식 끝은 쉼표다.
- **식에 나오는 기호는 식보다 먼저 정의한다.** 이름만 대는 것으로는 부족하고, 그 기호가 **무엇을 재는
  값인지**를 말한다. 연산자(`∩`, `|·|`)도 처음 쓰면 `where` 절에서 뜻을 밝힌다.
  - 나쁜 예: `... gives the intervals $I_A$ and $I_B$ that the two lines span.` (구간이 뭔지 모호하다)
  - 좋은 예: `Projecting a point onto $\mathbf{a}$ reduces it to one number, its position along that
    axis. The projected points of line $A$ run from a smallest to a largest position, and we write
    that range as $I_A$.` + 식 뒤에 `where $I_A \cap I_B$ is the part the two ranges share and
    $|\cdot|$ is the length of a range.`
- 기호가 겹치지 않는지 원고 전체에서 확인한다. 같은 글자를 두 뜻으로 쓰면 하나를 바꾼다
  (예: 점 개수 $N$과 병합 횟수가 겹치면 후자를 $N_{\text{merge}}$로).

## 9. 용어·표기 일관성

- **한 개념에는 한 단어만** 쓴다. 동의어로 바꿔 쓰지 않는다.
  - 예: `satellite tile`과 `satellite image`를 섞지 않는다. 입력 처리 단위를 뜻할 때만 `tile`.
  - 예: `segmentation backbones`와 `segmentation models`를 섞지 않는다.
- **한 단어를 여러 뜻으로 쓰지 않는다.** 위 규칙의 반대 방향이고, 놓치기 더 쉽다.
  한 소절 안에서 `leave`를 "벗어나다"·"남기다"·"방치하다"로 돌려 쓴 적이 있다. 그 소절이 이미
  `divergence`라는 이름을 쓰고 있었으므로, 기하적인 뜻은 전부 `diverge from`으로 통일하고
  나머지 두 자리는 다른 단어로 바꿨다. **임계값·기호의 이름이 곧 그 개념의 단어다.**
- 클래스 이름은 데이터셋(SEED-MAP) 표기 그대로(`center_line`, `no_parking_stopping_line`) 쓰고
  이탤릭으로 표시한다.
  **클래스 이름과 표기가 같은 일반 명사를 다른 뜻으로 쓰지 않는다** — 예: "center line"을 일반적인
  "가늘어진 선"의 뜻으로 쓰면 클래스 `center_line`과 헷갈린다. 그 자리는 다른 단어(`vectorized line`)로 바꾼다.
- 지표 이름은 정의한 형태로 고정한다(`F1@0.5`를 `F1@50`, `F1 score at 0.5`로 섞지 않는다).
- 참고문헌 인용의 구두점을 지킨다. 비제한 관계절 앞에는 쉼표를 넣는다.
  - `Chen et al. [5] which ...` → `Chen et al. [5], which ...`
- 범위 인용은 저널 스타일을 따른다(IEEE: `[2]–[6]`).
- 저널 스타일 표기를 원고 전체에서 통일한다(IEEE: `Fig. 2`, `Section III-B`, 표 번호 로마자).

## 10. 캡션·표·그림

- **캡션에는 구성과 보는 법만** 쓴다. 해석·성능 주장은 본문에 쓴다.
- IEEE 계열은 표 캡션을 짧게 쓴다. 두 번째 문장부터는 본문으로 옮긴다.
- 캡션 안에서 패널을 나열할 때 구두점을 통일한다. 문장은 마침표로 끝낸다.
  세미콜론으로 패널을 잇지 않는다. `Left:` / `green:` 같은 라벨 콜론은 유지한다.
- 본문에서 세부 패널을 가리킬 때 표기를 통일한다(`Figure 2a` 또는 저널 스타일 `Fig. 2(a)`).
- 표 제목의 대소문자를 통일한다(`Per-Class Performance on the Validation Split`).
- **표에 그룹 구분용 빈 행을 만들지 않는다.** 스테이지·범주를 나누고 싶으면 행 순서로 나타내고,
  꼭 필요하면 열을 하나 더 둔다.
- 값과 단위는 각각 제 열에 둔다. 같은 값을 두 단위로 병기하는 열을 만들지 않는다.
- **파라미터 표에는 참조 열(`Ref.`)을 둔다.** 각 행이 어느 식·어느 절에서 쓰이는지 가리켜야
  표와 본문을 나란히 읽을 수 있다. 권장 열 구성: `Parameter | Sym. | Ref. | Value | Unit`.
- 표를 스크립트로 만든다면 **논문에 그대로 붙여넣을 수 있는 형식**까지 출력하게 한다.
  손으로 옮기는 순간 코드와 표가 어긋나기 시작한다.
- 표에 설명이 필요한 값(0, 빈칸, 특이값)이 있으면 본문에서 그 뜻을 밝히거나, 밝히지 않을 것이면
  행 이름을 자명하게 고친다. `One entry of the table is zero.`처럼 사실만 적고 끝내지 않는다.
- 캡션과 본문이 같은 문장을 반복하지 않게 한다.

## 11. 제출 전 자기 점검 체크리스트

작성·수정한 부분에 대해 아래를 실제로 훑는다.

1. 세미콜론이 남아 있는가? 있으면 마침표로 나눈다.
2. 25단어를 넘는 문장이 있는가? 쉼표로 매달린 절이 두 개 이상인가?
3. 한글 번역과 문장 수가 1:1로 맞는가?
4. `utilize`, `facilitate`, `in order to`, `thereby` 같은 어려운 표현이 남아 있는가?
5. `it` / `this` / `they`가 무엇을 가리키는지 그 문장만 읽고 알 수 있는가?
6. 정의 없이 등장한 용어·약어가 있는가?
7. 문단 첫 문장이 그 문단의 주장인가?
8. 괄호 수치가 습관적으로 붙어 있지 않은가? 핵심 비교에만 남겼는가?
9. 수치가 원본 CSV·표와 일치하는가? 자릿수 표기가 통일됐는가?
10. 이전 버전과 비교하는 표현이 있는가?
11. 같은 개념을 다른 단어로 부른 곳이 있는가?
12. 그림 설명이 그림에 실제로 보이는 것만 말하고 있는가?
13. 데이터가 뒷받침하지 않는 인과·추세 주장이 있는가?
14. 캡션에 해석이 들어가 있지 않은가?
15. 출력 형식이 그 세션의 지시와 맞는가? 파일에 썼다면 무엇을 고쳤는지 한글로 요약했는가?
16. 리뷰어·심사 과정을 가리키는 표현이 본문에 남아 있는가? 리뷰 지적 때문에 만든 해명 문단이 있는가?
17. 알고리즘을 **루프 한 바퀴**로 서술한 곳이 있는가? 결과의 성질로 바꿀 수 있는가?
18. 식에 쓰인 기호·연산자가 그 식 **앞에서** 정의됐는가? 같은 글자를 두 뜻으로 쓴 곳이 있는가?
19. 한 단어를 서로 다른 뜻으로 돌려 쓴 곳이 있는가?
20. 바로 앞 문장을 되풀이하는 문장이 있는가?
21. 한 문단 안에서 단위(픽셀·미터)가 섞였는가? 수치를 댔는데 결론 문장이 없는가?
22. 표에 빈 그룹 행이 있는가? 참조 열이 있는가? 스크립트 출력과 원고의 표가 같은가?
23. 소제목이 꼭 필요한가(약어 정의를 소제목이 대신하고 있지 않은가)? 본문 문장이 특정 시점에
    못박혀 있는가?
24. 같은 현상을 지표 두 개로 중복 보고하거나 비교 기준 없는 비율을 쓰지 않았는가? 인과를 부정문으로
    돌려 말하거나, 우리 방법 설명을 근거보다 먼저 쓰지 않았는가?
