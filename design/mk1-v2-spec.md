# MK1 v2 명세: 실제 글에서 도는 위성 시스템 (2026-10-09)

방향: `design/direction-2026-10-09.md`. 이 문서는 엔지니어링 명세다. 데이터 구조, 모듈 인터페이스, 채택 규칙, 지표를 정한다.

## 1. 칠판 스키마 (타입이 있는 그래프)

```
Entity   { id, surface: str, type: PER|LOC|ORG|DATE|NUM|MISC, span: (doc_id, start, end) }
Claim    { id, subj: Entity.id, pred: str, obj: Entity.id | literal,
           source: doc_id | "memory" | "board", conf: float[0,1], polarity: +1|-1 }
Query    { id, text, target: (subj?, pred?, obj?)   # 빈칸이 답
           answer_type: Entity.type | BOOL | UNKNOWN }
Answer   { query_id, value | "모름", conf, support: [Claim.id] }
```

주소는 `Entity.id`(구조)이고, 내용은 `Claim.conf`와 임베딩(학습)이다. v1 원칙("주소는 구조로, 내용은 학습으로")을 실제 글로 옮긴 것이다.

## 2. 모듈 (함수 시그니처)

| 모듈 | 입력 → 출력 | 구현 | 비용 |
|---|---|---|---|
| parse | text → [Entity], [Claim], Query | Gemma 4 E2B (QAT Q4_0 GGUF, llama.cpp), JSON 스키마 강제 | 측정 예정 |
| ner | text → [Entity] | GLiNER | 측정 예정 |
| read | (Query, doc) → Answer 후보 + span 점수 | mDeBERTa QA (KLUE) | KorQuAD에서 초당 31문제 (fp32) |
| entail | (sentence, Claim) → p(지지), p(반박) | mDeBERTa NLI | 초당 약 600 (fp32) ~ 1,600쌍 (fp16) |
| embed | text → R^384 | multilingual-e5-small | 측정 예정 |
| trust | 출처별 Claim × 피드백 → 출처 신뢰도 | 세는 중재자 (Beta 사전분포, 학습 숫자 2개) | 무시 가능 |
| reason | Claim 그래프 + Query → Answer | 순환 추론기 (TRM식, 우리가 학습) | 만들 것 |
| route | 칠판 상태 → 호출할 모듈 집합 | 학습형 사령탑 | 무시 가능 |
| memory | 확인된 Claim 저장, Query → 기억 Claim | 키: (subj, pred) 임베딩 | 무시 가능 |
| speak | Answer + support → 문장 | Gemma 4 E2B | 측정 예정 |

LLM(Gemma)은 parse와 speak만 맡는다. 추론(reason), 판단(trust), 기억(memory)은 맡지 않는다.

## 3. 모듈 채택 규칙

모듈 m마다 **검증셋**(외부 시험셋 아님)에서 잰다.

```
ΔS(m)   = score(MK1) − score(MK1 − m)
cost(m) = (latency_ms, VRAM_MB)
채택 ⇔ ΔS(m)의 세션 단위 95% 구간 하한 > 0
      또는 (ΔS ≈ 0 이고, 같은 일을 하는 모듈보다 latency·VRAM이 확실히 작음)
```

예: GLiNER. 가설은 "parse(Gemma)의 Entity 재현율 < GLiNER 재현율이고, 그 차이가 충돌 찾기와 기억 검색 점수로 이어진다"이다. (a) Gemma 단독, (b) GLiNER 단독, (c) 합집합의 Entity 재현율, 시간, ΔS를 재서 결정한다.

## 4. 시스템 지표

| 지표 | 정의 |
|---|---|
| 정확도 | 맞음 / 전체 (외부 시험은 공식 지표도 함께) |
| 선택적 정확도 | 답한 것 중 맞음, 거절률 |
| 반사실 강건성 | 틀린 문서를 섞었을 때의 정확도 하락폭 |
| 깊이 일반화 | 학습 깊이 ≤ 3 → 시험 깊이 5 정확도 |
| 온라인 학습 기울기 | 시간 순서 흐름에서 구간별 점수의 기울기 (가중치 고정) |
| 엮기 기여분 | score(MK1) − score(같은 LLM 혼자), score(MK1) − score(같은 메모리 예산의 단일 모델) |
| 비용 | 문제당 지연(ms), 최대 VRAM(MB), 학습한 파라미터 수 |

## 5. 구현 순서

1. `mk1v2/board.py`: 스키마와 직렬화
2. `mk1v2/llm_service.py`: llama-server 실행, parse/speak (JSON 스키마 강제), 지연·VRAM 측정
3. ner / embed / entail 래퍼: 같은 인터페이스, 비용 측정
4. `mk1v2/ablate.py`: 검증셋에서 모듈별 ΔS
5. reason(추론기), 그다음 외부 시험(봉인 규칙)

## 6. RGB 판정 규칙 (2026-10-09 고정, 봉인 90%를 열기 전에 정함)

RGB의 공식 지표는 시험장마다 따로다(잡음 정확도, 거절률, 오류 감지율, 합치기 정확도). 그래서 "거절이 늘고 정확도가 줄었다" 같은 맞교환을 판정할 수 없다. 이 프로젝트의 채점 원칙(맞음 +1, 모름 0, 틀림 −1)으로 **하나의 순점수**를 정한다. 공식 지표도 함께 보고한다.

| 시험장 | 맞음 (+1) | 0 | 틀림 (−1) |
|---|---|---|---|
| 잡음 0 / 0.4 / 0.8, 합치기 | 답 포함 | 거절 | 그 밖 |
| 거절 (답이 문서에 없음) | 거절 | — | 답을 냄 |
| 거짓 문서 | 오류를 알리고 바른 답 | 오류만 알림, 또는 거절 | 거짓 답을 그대로 냄 |

- **주 지표**: 여섯 시험장 순점수의 평균(시험장마다 같은 무게).
- **판정**: MK1 − Gemma 4 E4B 혼자(같은 메모리 예산 상대), MK1 − Gemma 4 E2B 혼자(MK1 안의 LLM). 문제 단위로 짝지은 부트스트랩 95% 구간(RGB는 세션이 없으니 문제 단위가 맞다).
- 문턱값 같은 모든 조정은 개발용 10%에서만 한다. 봉인 90%는 시스템을 얼린 뒤 한 번만 연다.
- **현재 개발용 상태(기록)**: MK1 v2 첫 버전은 Gemma 혼자를 넘지 못했다. 원인은 영어 거절 판단과 지식 부족이다. 다음 작업은 거절 판단 위성(A)과 지식 위성(B)이다.
