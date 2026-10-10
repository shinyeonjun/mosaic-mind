# mosaic-mind

거대한 언어모델 하나 대신, **역할별 부품(읽기, 신뢰, 질문, 생각, 연결, 사령탑, 성장)을 키워서 얼리고, 부품 사이의 작은 연결만 배워서 조합하는** 지능 모델을 노트북 한 대(RTX 5070 Laptop 8GB)로 키워 가는 연구다. 단계마다 질문 하나, 비교 대상, 판정 기준을 결과 보기 전에 `design/`에 적고, 실패와 정정도 그대로 기록한다.

| 단계 | 할 수 있게 된 것 | 새 부품 | 결과 |
|---|---|---|---|
| 1 | 말투 속 신뢰 단서를 스스로 꺼내 씀 | 연결 머리 (6천) | 졸업 |
| 2 | 언제, 누구에게 물을지 정함 | 호기심 머리 (1만) | 졸업 |
| 3 | 3칸 고리만 배우고 6칸 고리를 더 오래 생각해서 풂 | 생각 루프 | 졸업 |
| 4 | 얼린 부품 둘을 연결 184개로 이어 새 문제를 풂 | 연결 장치 | 졸업 |
| 5 | 실제 한국어 기사(KLUE-MRC): 이 과제로 학습한 0.28B 전문가가 학습 안 한 1.2B 범용 모델보다 잘 읽음 (공정한 크기 비교 아님) | 읽기 전문가 | 절반 |
| 6 | 실제 기사 + 거짓말하는 출처: 읽기와 신뢰를 함께 써서 루머를 거름 | 실제 글 신뢰 부품 | 졸업 |
| 7 | "뭐가 모자라지?"를 보고 필요한 부품을 부름 | 사령탑 | 졸업 |
| 8 | 실패가 늘 때만 새 부품을 키움 | "다음 차례" 가지 | 졸업 |
| 9 | 새 능력의 모양을 가설로 찾음 (한 칸 돌리기, 해↔별) | 가설 탐색 | 졸업 |
| MK1 | 부품들을 칠판으로 묶어 하나로: 만든 세계 + 실제 기사 11,000문제를 섞어 풂, 점수 그대로 + 물어보기 +0.12 | 칠판, 읽기 기억 | 졸업 (A–D) |
| 10 | 실제 기사에서 읽은 답이 열쇠가 되어 사슬 추론으로 이어짐 (학습 없이, 전달 100%) | 없음 (이름표 맞추기 규칙) | 기준 3/4 (1개는 설계 실수) |
| 11 | 안내판과 기사가 엇갈릴 때 "어느 통로가 맞아 왔는지"로 저울질 (이상적 판단자의 98%) | 중재자 (2,242) | 기준 3/4 (1개는 효과 크기 계산 실수) |
| 12 | 세상이 확인해 준 사실을 세션을 넘어 기억: 날마다 똑똑해지고(0.18 → 0.50), 다른 종류 문제에도 씀(+0.21) | 장기 기억 (규칙) | 졸업 |
| 13 | 경험 밖(처음 보는 나쁜 안내판)에서도 무너지지 않는 믿음: 세는 표현 + 경험에서 배운 사전 믿음 (seed 3개, 새 세계) | 세는 중재자 (학습 숫자 2개) | 졸업 |
| 14 | 믿던 출처가 갑자기 배신하면 알아채고 끊기 (예상 못 한 불확실성, 변화점 검출; seed 3개, 새 세계) | 변화 감지 중재자 (학습 숫자 3개) | 졸업 |
| MK1 v2 | 외부 시험 RGB(봉인 1회): 작은 Gemma E2B를 품고 더 큰 Gemma E4B와 동점(0.441 vs 0.440), 안의 E2B 혼자보다 +0.094. 거절률 41% → 78% | 영어 읽기 위성 (SQuAD 2.0) | 비김 (E4B), 이김 (E2B) |
| MK1 v2 + 생각 위성 | 외부 시험 MuSiQue(봉인 1회, 여러 단계 추론): HotpotQA로 배운 선택기·독해기를 붙였지만 E2B와 비김(0.506 vs 0.513), E4B에 짐(0.587) | 단계별 문서 선택기, 다단계 독해기 | 짐 (E4B), 비김 (E2B) |
| MK1 v2 + 키우는 부품 | 외부 시험 새 과제 흐름(봉인 1회, 분류 과제 6개를 피드백만으로 배우기): 기호 라벨에서 E2B보다 +0.173, E4B와 비김(0.603 vs 0.615). 이름 라벨은 E4B에 짐(0.653 vs 0.761) | 과제마다 키우는 프로토타입 부품 (e5-small 위) | 비김/짐 (E4B), 이김 (E2B) |
| MK1 v2 + 키우는 부품 (bge 눈) | 외부 시험 새 과제 흐름 final2(봉인 1회): 기호 라벨에서 **E4B를 이김**(0.752 vs 0.630, +0.122), 이름 라벨은 비김(0.764 vs 0.788) | bge-large 눈 (0.34B), 정규화·중심 맞춘 프로토타입 | **이김**/비김 (E4B), 이김 (E2B) |
| MK1 v2 + 격자 생각 위성 | 외부 시험 ARC-AGI-1 evaluation(봉인 1회): 신경망 없는 프로그램 탐색 위성으로 **E4B를 이김**(0.074 vs 0.023, +0.051). 답했을 때 93% 맞음 | 격자 프로그램 탐색기 (기본 동작 약 90개, 깊이 3) | **이김** (E4B), 이김 (E2B) |
| MK1 v3 (하나의 순환) | 섞인 봉인 시험(MuSiQue + 과제 흐름 2,400개, 종류를 알려 주지 않음): 기호 라벨 **E4B를 이김**(0.725 vs 0.662), 이름 라벨 짐(0.750 vs 0.796), MuSiQue 비김(0.588 vs 0.583). 보정기는 시험 밖 데이터로만 학습, E2B 호출 최대 4번 → 1.4번 | 보정기 (P(맞음), P(답할 수 있음)), 기대 점수 결정, 말하기 위성 건너뛰기 | 이김/짐/비김 (E4B) |

**읽을 때 주의 (객관성 감사, `design/audit-2026-10-08.md`)**: 만든 세계(1–4, 7–11단계)의 시험 문장은 학습 문장과 같은 틀이다. 시험한 건 언어 이해가 아니라 처음 보는 조합(더 긴 사슬, 새 문제 종류, 새 부품 조합, 숨은 신뢰도)에서의 추론과 판단이다. 실제 기사 시험 문제의 40%는 읽기 전문가가 학습 때 본 기사에서 나왔다(점수는 부풀지 않았음). 장기 기억 효과의 크기는 질문이 자주 반복되는 세계 설계에 달려 있다. MK1 결과는 seed 3개와 세션 단위 구간으로 다시 확인했다.

전체 그림과 결론은 `notebook/00_research_overview.ipynb`, 단계별 결과는 `notebook/06–14`, 연구 기록은 `design/research-log-2026-10-06.md`에 있다.

## 처음 실험들 (Local Cognitive Lab)

역할별 부품(언어 해석, 기억, 판단)을 연결해서 판단하고 공부하는 시스템을 작은 통제 실험으로 만들고 측정한다. 초기 실험(001–010)은 정답 구조 관측으로 순환 코어를 비교했다. 규칙 세계 v1/v2부터는 한국어 문장 입력과 사전학습 모델 부품을 쓴다.

## 실행

프로젝트 venv가 이미 있다면:

```powershell
uv pip install --python .\venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match -e . "torch==2.14.1+cu130"
& .\venv\Scripts\python.exe -m cognitive_lab.selfcheck
& .\venv\Scripts\python.exe -m cognitive_lab.train --device auto
```

RTX 50 시리즈용 CUDA 13.0 휠을 사용한다. `unsafe-best-match`는 이 설치 명령에서 정확히 지정한 PyTorch CUDA 휠을 PyPI의 CPU 휠보다 선택하기 위해 넣었다. 결과 JSON과 체크포인트는 각각 `artifacts/results/`, `artifacts/checkpoints/`에 생긴다. GPU 자동 선택은 PyTorch에서 CUDA를 사용할 수 있을 때만 적용된다.

기본 학습 명령은 GRU 상태 크기 기준선, GRU 파라미터 수 기준선, RIMs형 코어를 한 seed에서 비교한다. 여러 seed는 CMD에서 다음처럼 연속 실행할 수 있다. 각 seed 결과는 별도 JSON으로 저장된다.

```cmd
for %S in (43 44 45 46 47) do python -m cognitive_lab.train --device cuda --seed %S
```

## 구조

- `design/`: 가설과 실험 정의
- `model/`: 내려받은 사전학습 모델. LFM2.5-1.2B와 mDeBERTa NLI를 규칙 세계 실험에서 쓴다.
- `src/cognitive_lab/task.py`, `core/`, `train.py`: 실험 001–005 (기억 갱신 과제, GRU/RIMs 비교)
- `src/cognitive_lab/active/`: 실험 006–010 (질문을 고르는 탐구형 에이전트)
- `src/cognitive_lab/world/`: 규칙 세계 v1. 한국어 문장 → 판단. 생성기, 정답 해결기, 지름길 검사, 점수판, L1/P1/P2 에이전트가 있다. 설계: `design/world-v1-korean-rule-world.md`
- `src/cognitive_lab/world2/`: 규칙 세계 v2. 누구를 믿을지 경험으로 배운다. 신탁, 코드 학습자(B2/B3), 신경망 판단기(N2), 분석 도구가 있다. 설계: `design/world-v2-learning-to-judge.md`
  - `hedged*.py`: 말투 세계와 연결 머리(1단계), `asking*.py`: 물어보기와 호기심 머리(2단계)
- `src/cognitive_lab/world3/`: 이어진 단서와 생각 루프(3단계). 설계: `design/world-v3-thinking.md`
- `src/cognitive_lab/world4/`: 신뢰 + 생각 복합 문제, 부품 연결(4단계). 설계: `design/world-v4-composite.md`
- `src/cognitive_lab/world5/`: KLUE-MRC, 범용 언어모델 vs 읽기 전문가(5단계). 설계: `design/world-v5-real-korean.md`
- `src/cognitive_lab/world6/`: 실제 기사 + 출처 신뢰(6단계). 설계: `design/world-v6-real-trust.md`
- `src/cognitive_lab/world7/`: 사령탑(7단계). 설계: `design/world-v7-routing.md`
- `src/cognitive_lab/world8/`: 스스로 성장과 모양 찾기(8–9단계). 설계: `design/world-v8-growth.md`
- `src/cognitive_lab/mk1/`: MK1 통합(부품 명함, 졸업 시험, 칠판, 통합 시험). 설계: `design/mk1-integration.md`
- `data/`(git 제외): KLUE-MRC parquet. `huggingface_hub`로 `klue/klue`의 `mrc/` 두 파일을 `data/klue-mrc/`에 받는다.
- `model/`, `artifacts/`(git 제외): 내려받은 모델, 체크포인트, 결과 JSON, 캐시
- `notebook/`: 결과 분석과 그래프 전용 노트북. 재사용할 로직은 `src`에 둔다.

규칙 세계 실험은 `transformers`가 필요하고(`pip install -e .[llm]`), 노트북은 `jupyter`와 `matplotlib`이 필요하다.

