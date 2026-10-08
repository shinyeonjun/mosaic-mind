# 뇌과학 조사 → MK1 설계 결정 (2026-10-09)

형식: 원리 / 근거 / MK1에 이미 있는 것 / 설계 결정 / 시험 가능한 예측. 뇌과학 이론은 권위가 아니라 **설계 가설**로 쓴다. 대부분 논쟁 중이고, 아래 근거 중 일부는 초록과 2차 자료로만 확인했다(표시함).

## 1. 전역 작업공간 (Global Workspace) → 칠판

- **근거**: Baars, Dehaene의 전역 작업공간 이론. Goyal 외(ICLR 2022 구두 발표)는 전문 모듈들이 **용량이 제한된 공유 작업공간**에 쓰기 위해 경쟁하고, 그 내용이 모두에게 방송되는 구조를 제안했다. 저자들의 주장은 용량 제한이 (1) 전문화와 조합성을 촉진하고 (2) 독립된 전문가들을 동기화한다는 것이다. VanRullen & Kanai(2021)는 모듈마다 다른 잠재 공간을 비지도 번역(순환 일관성)으로 잇는 공용 잠재 작업공간을 제안했다(구현 로드맵, 후속 구현 있음).
- **MK1에 있는 것**: 칠판(이름 붙은 칸, 쓰기 기록). 용량 제한과 쓰기 경쟁은 없다.
- **설계 결정**: v2 칠판에 **쓰기 예산**(한 단계에 k개 Claim만 쓰기, 점수 높은 순)과 **방송**(모든 위성이 같은 칠판 상태를 읽음)을 넣는다. 칸 주소는 구조(Entity, 술어), 칸 내용은 학습된 벡터(VanRullen식 공용 잠재 표현)를 함께 둔다.
- **예측**: 쓰기 예산이 있는 칠판이 무제한 칠판보다 깊이 일반화(ProofWriter 깊이 5)가 좋다. 반증되면 예산을 없앤다.

## 2. 기저핵 관문 (PBWM) → 사령탑

- **근거**: O'Reilly & Frank의 PBWM. 기저핵이 전전두엽 작업기억의 관문을 연다. "새 정보로 갱신할지, 지금 것을 유지할지"를 정하고, 확장판에서는 **유지 중인 여러 표상 중 무엇을 지금 쓸지**(출력 관문)도 정한다. 관문은 손으로 짠 게 아니라 도파민 기반 행동자-비평자(actor-critic) 강화학습으로 배운다. LSTM의 관문과 기능이 비슷하다.
- **MK1에 있는 것**: 사령탑(7단계). 물었을 때의 점수 이득을 회귀로 배워서, 비용보다 크면 부른다. 비평자 역할에 가깝다.
- **설계 결정**: 사령탑을 관문 세 개로 나눈다. (a) 입력 관문: 어느 위성을 부를지, (b) 갱신 관문: 칠판의 어느 Claim을 덮어쓸지, (c) 출력 관문: 답할 때 어느 Claim을 근거로 쓸지. 전부 "얻은 점수 − 비용"을 보상으로 배운다.
- **예측**: 출력 관문이 있으면 근거(support)가 짧고 정확해지고, 반사실 문서가 섞였을 때 덜 흔들린다.

## 3. 확신과 메타인지 (OFC) → 모르면 멈춤

- **근거**: Kepecs 외(Nature 2008). 쥐의 안와전두피질 뉴런 발화가 **결정 확신 모델의 예측**과 맞았다. 확신은 "증거가 주어졌을 때 내 선택이 맞을 확률"이라는 통계량이고, 단순한 연산으로 계산된다. Lak 외(2014)는 OFC를 끄면 정확도는 그대로인데 확신 보고만 망가진다는 걸 보였다. 즉 결정과 확신은 분리될 수 있다. 결정 회로 안에 확신이 섞여 있다는 반대 견해(Kiani & Shadlen)도 있다.
- **MK1에 있는 것**: 부품마다 확률 0.5 기준으로 "모름". 확신이 보정(calibration)됐는지는 재지 않았다.
- **설계 결정**: 모든 위성이 Answer에 **보정된 conf = P(맞음 | 증거)**를 붙인다. 보정은 위성 바깥의 작은 확신 판독기(검증셋으로 맞춤)가 한다. 답할지는 "conf × 맞힌 이득 − (1−conf) × 틀린 손해 > 0"으로 정한다.
- **예측과 지표**: 보정 오차(ECE)가 줄고, 같은 거절률에서 선택적 정확도가 단일 LLM보다 높다.

## 4. 예상된 불확실성 vs 예상 못 한 불확실성 (ACh / NE) → 중재자

- **근거**: Yu & Dayan(Neuron 2005, NIPS 2002). 아세틸콜린은 **이미 아는 단서의 불확실성**(이 출처는 원래 70%만 맞음)을, 노르에피네프린은 **예상 못 한 불확실성**(맥락이 갑자기 바뀌어서 생긴 놀람)을 신호한다는 베이즈 이론이다. NE는 "신경 인터럽트"처럼 지금까지의 믿음을 끊고 다시 배우게 한다(Dayan & Yu 2005).
- **MK1에 있는 것**: 세는 중재자(I단계)는 ACh 역할만 한다(Beta 사후분포로 알려진 신뢰도). **I단계에서 남은 약점, "나쁜 안내판을 늦게 의심함"이 바로 NE 역할이 없어서 생긴다.**
- **설계 결정**: 중재자에 **변화 감지**를 추가한다. 통로의 최근 적중률이 지금 믿음에서 크게 벗어나면(예상 못 한 놀람), 그 통로의 횟수를 줄이거나 초기화한다. 계산은 베이즈 변화점 검출(run-length) 또는 놀람 문턱이다. 학습할 숫자는 문턱과 망각률 2개다.
- **예측**: 세션 중간에 안내판 신뢰도가 바뀌는 세계(새로 만듦)에서, 변화 감지 중재자가 바뀐 뒤 회복 속도(문제 수)가 확실히 빠르다. 바뀌지 않는 세계에서는 손해가 0.01 이내다.

## 5. 상보적 학습 체계 (해마 / 신피질) + 스키마 → 기억과 성장

- **근거**: Kumaran, Hassabis & McClelland(TiCS 2016). 해마는 개별 경험을 빠르게 저장하고, 신피질은 구조화된 지식을 천천히 쌓는다. 새 사실을 가중치에 빨리 밀어 넣으면 옛 지식이 덮어써진다(파국적 간섭). 그래서 해마가 경험을 따로 저장했다가, 다른 경험과 섞어서 신피질에 **재생(replay)**한다. Tse 외(Science 2007)는 이미 **스키마(틀)가 있으면 새 정보가 한 번 만에 빠르게 굳는다**는 걸 보였다. McClelland 외는 기존 지식과 맞는 정보는 간섭 없이 빨리 흡수된다고 시뮬레이션으로 보였다.
- **MK1에 있는 것**: 장기 기억(H)은 해마 역할(빠른 사례 저장)만 한다. 성장(8–9단계)은 오답노트와 복습으로 재생을 일부 했다.
- **설계 결정**: (a) **빠른 저장소**(장기 기억, 사례 그대로)와 (b) **느린 저장소**(작은 학습 부품의 가중치)를 나눈다. 쉬는 시간마다 빠른 저장소에서 사례를 뽑아 옛 사례와 **섞어서** 느린 부품을 조금 학습한다(재생 응고). 응고 뒤에는 졸업 시험을 다시 돌려 옛 능력이 안 깨졌는지 확인한다. (c) **스키마 관문**: 기존 Claim 그래프와 맞는 새 사실은 빨리 응고하고, 안 맞는 건 사례로만 둔다.
- **예측**: 재생 응고를 하면 바꿔 말한 질문에도 답한다(사례 기억만으로는 못 함). 졸업 시험은 그대로 통과한다.

## 6. 신경조절과 메타학습 (Doya 2002) → 위성의 하이퍼파라미터 조절

- **근거**: Doya(Neural Networks 2002, 2차 요약으로 확인). 도파민은 보상 예측 오차, 세로토닌은 보상 시간 범위(할인율), 노르아드레날린은 탐색 폭(softmax 온도), 아세틸콜린은 학습률에 대응한다는 가설이다. 지나친 단순화라는 비판도 있다.
- **설계 결정**: 지금은 사람이 정한 상수인 "중재자 망각률, 사령탑 탐색, 성장 문턱"을 **전역 조절 신호**(최근 놀람, 최근 보상 추세) 몇 개로 조절한다. 우선순위는 낮게 두고, 4번(변화 감지)이 통하면 확장한다.

## 7. 예측 부호화 (Rao & Ballard, Friston) → 놀람이 배움과 주의를 이끈다

- **근거**: 위층은 아래층 활동을 예측하고, **예측 오차**만 위로 올려 보낸다. 오차는 **정밀도(분산의 역수)**로 가중되는데, 정밀도 조절이 곧 주의다. 자유 에너지는 단순화 가정 아래 정밀도로 가중한 예측 오차와 같다(리뷰 arXiv 2107.00140, 일부는 2차 자료).
- **MK1과의 대응**: 중재자의 통로별 신뢰도 가중은 **정밀도 가중**과 같은 구조다. 성장의 "실패율이 오르면 키운다"는 지속적 예측 오차로 성장을 촉발하는 셈이다.
- **설계 결정**: 위성마다 "다음에 칠판에 무엇이 올라올지"를 예측하게 하고, 예측 오차를 (a) 주의(어느 칸을 다시 볼지), (b) 성장 신호(오차가 계속 크면 새 가지), (c) 재생 우선순위(놀란 사례부터 응고)로 쓴다.

## 8. 천 개의 뇌 이론 (Hawkins, Numenta) → 위성들의 투표

- **근거**: 신피질의 각 기둥이 사물 전체의 모델을 따로 만들고, 장거리 연결로 **투표해서 합의**한다는 이론이다(논쟁 중이고, 투표 세부는 커뮤니티 해석 포함).
- **설계 결정**: 같은 질문에 여러 위성(읽기, 사실 판단, 기억, 말하기)이 각자 답 후보를 내면, 중재자가 **신뢰도 가중 투표**로 합의한다. 투표 결과를 다시 모두에게 방송해 한 번 더 고치게 한다(되풀이 합의, TRM식 반복과도 맞음).
- **예측**: 가중 투표가 가장 좋은 단일 위성보다 낫고, 위성 하나를 빼도 크게 무너지지 않는다(강건성).

## 9. 모듈형 딥러닝 서베이 (Pfeiffer, Ruder, Vulić, Ponti, TMLR 2023)

- **근거**: 모듈 설계는 "조건부 라우팅 → 일부 모듈 계산 → 결합(aggregation)"으로 정리된다. 계산과 라우팅을 분리하고 모듈을 따로 갱신하면 긍정 전이와 체계적 일반화에 유리하다는 주장이다(초록 수준 확인).
- **MK1과의 대응**: 사령탑 = 라우팅, 중재자·투표 = 결합, 위성 단위 학습 = 국소 갱신. 지금 구조가 이 분류 안에 정확히 들어간다. 라우팅·결합 방법 비교는 원문 본문을 읽고 추가한다.

## 우선순위 (기대 효과 / 노력)

| 순위 | 할 일 | 근거 | 노력 | 시험 |
|---|---|---|---|---|
| 1 | 중재자에 변화 감지 (NE) | 4 | 작음 | 세션 중간에 신뢰도가 바뀌는 v11 변형 |
| 2 | 위성마다 보정된 확신 + 기대 이득으로 답하기 | 3 | 작음 | ECE, 선택적 정확도 |
| 3 | v2 칠판에 쓰기 예산 + 방송 + 가중 투표 | 1, 8 | 중간 | ProofWriter 깊이 일반화, 위성 하나 빼기 |
| 4 | 사령탑을 관문 세 개로 (RL) | 2 | 중간 | 반사실 강건성, 근거 정확도 |
| 5 | 빠른/느린 기억 + 재생 응고 + 스키마 관문 | 5 | 큼 | 바꿔 말한 질문, 졸업 시험 유지 |
| 6 | 예측 오차로 주의·성장·재생 이끌기 | 7 | 큼 | 성장 시점, 재생 효율 |
| 7 | 전역 조절 신호 | 6 | 작음 (1번 뒤) | — |

## 출처
- Goyal 외, Coordination Among Neural Modules Through a Shared Global Workspace, ICLR 2022: https://arxiv.org/abs/2103.01197
- VanRullen & Kanai, Deep Learning and the Global Workspace Theory: https://arxiv.org/abs/2012.10390
- Kumaran, Hassabis & McClelland, What Learning Systems do Intelligent Agents Need?, TiCS 2016: https://web.stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf
- Tse 외, Schemas and memory consolidation, Science 2007: https://pmc.ncbi.nlm.nih.gov/articles/PMC2271071
- McClelland, Rapid neocortical learning (schema-consistent): https://stanford.edu/~jlmcc/papers/McClellandOF_JEPGRapidNeocorticalLearning.pdf
- Yu & Dayan, Expected and Unexpected Uncertainty, NIPS 2002: https://proceedings.neurips.cc/paper/2002/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html
- Dayan & Yu, Norepinephrine and Neural Interrupts, NIPS 2005: https://proceedings.neurips.cc/paper/2005/hash/b2ea5e977c5fc1ccfa74171a9723dd61-Abstract.html
- PBWM (O'Reilly & Frank; Hazy, Frank & O'Reilly): https://ccnlab.org/papers/HazyFrankOReilly06.pdf
- Kepecs 외, Neural correlates, computation and behavioural impact of decision confidence, Nature 2008: https://doi.org/10.1038/nature07200
- Lak 외 2014 (OFC 비활성화와 확신): https://pmc.ncbi.nlm.nih.gov/articles/PMC4364549
- Doya, Metalearning and neuromodulation (요약): https://en.wikipedia.org/wiki/Metalearning_(neuroscience)
- Free Energy Principle의 기계학습 응용 리뷰: https://arxiv.org/abs/2107.00140
- Predictive Coding Theories of Cortical Function: https://arxiv.org/abs/2112.10048
- Thousand Brains (Hawkins 세미나): https://redwood.berkeley.edu/seminars/jeff-hawkins-april-2021/
- Pfeiffer 외, Modular Deep Learning, TMLR 2023: https://arxiv.org/abs/2302.11529
- AI Meets Brain: memory systems survey (2025): https://arxiv.org/abs/2512.23343
