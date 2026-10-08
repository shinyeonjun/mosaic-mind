# MK1 부품 명함

`src/cognitive_lab/mk1/registry.py`에서 만든다. 졸업 점수는 `python -m cognitive_lab.mk1.checks`가 다시 잰다.

## 읽기 (만든 세계) (`reader-v1`)

- 하는 일: 문장과 가설 쌍을 읽어 768차원 특징을 낸다
- 읽는 것: (문장, 가설) 쌍
- 쓰는 것: 768-d 특징 (모든 작은 부품의 입력)
- 코드: `mk1/reader.py (ReaderService)`
- 체크포인트: `world-v1_reader-anchored-learned-sourcefold-staged-tuned_composition_seed-42.pt`
- 학습: 규칙 세계 v1, 순서대로 키우기(읽기 → 판단 → 함께 다듬기)
- 크기: 2.78억, 얼림
- 졸업 시험: `reader-parity`, 점수 0.0 (design/interface-learning-v1.md)

## 신뢰 + 기억 (말투 포함) (`trust-v2h`)

- 하는 일: 사람마다 맞은 말을 기억하고 누구를 믿을지 판단한다. 말투 단서도 쓴다
- 읽는 것: 사람의 말(문장) × 기준 문 × 열쇠의 읽기 특징, 정답 피드백
- 쓰는 것: 열쇠별 확률 (기준 문마다)
- 코드: `world2/hedged_system.py (ConnectedSystem)`
- 체크포인트: `world-v2h_connected-m8_seed-{42,43,44}.pt`
- 학습: 말투 세계 v2-H, 세션 3,000개, 최종 답만
- 크기: 연결 머리 6,152 + 기억·판단 19,361, 얼림
- 졸업 시험: `stage1-trust`, 점수 0.4148 (design/world-v2h-tone-cues.md)

## 호기심 (물어보기) (`curiosity`)

- 하는 일: 누구에게 물으면 점수가 얼마나 오를지 예측해 질문할지 정한다
- 읽는 것: 신뢰 부품의 기억과 상태
- 쓰는 것: 물을 사람 (또는 안 물음)
- 코드: `world2/asking_system.py (CuriosityHead)`
- 체크포인트: `world-v2a_curiosity_cost-0.2_seed-{42,43,44}.pt`
- 학습: 물어보기 세계 v2-A, 물어본 결과의 점수 변화를 회귀
- 크기: 10,689, 얼림
- 졸업 시험: `stage2-curiosity`, 점수 0.5311 (design/world-v2a-asking.md)

## 생각 (고리 따라가기) (`thinker`)

- 하는 일: 문마다 상태를 두고 한 번에 한 칸씩 정보를 전달한다. 답이 더 안 바뀌면 멈춘다
- 읽는 것: 안내판 문장의 사실·연결 특징, (선택) 다른 부품이 넘긴 사실
- 쓰는 것: 문마다 열쇠 믿음, 질문 문의 답
- 코드: `world3/thinker.py (Thinker)`
- 체크포인트: `world-v3_thinker-grown_seed-{45,46,47}.pt`
- 학습: 이어진 단서 v3: 연결 학습(모두 켜고 가지치기) → 연결 얼리고 모든 문 채점
- 크기: 25,747, 얼림
- 졸업 시험: `stage3-thinker`, 점수 0.975 (design/world-v3-thinking.md)

## 연결 (신뢰 → 생각) (`adapter-trust-thinker`)

- 하는 일: 신뢰 부품의 결론을 생각 부품이 받는 사실 메시지로 바꾼다 (충실한 전달)
- 읽는 것: 기준 문마다 신뢰 부품의 결론
- 쓰는 것: 생각 부품의 사실 메시지
- 코드: `world4/system.py (CompositeSystem.adapter)`
- 체크포인트: `world-v4_grown_seed-{42,43,44}.pt`
- 학습: 복합 세계 v4, 정답 없이 신뢰 부품의 결론을 그대로 전달하도록
- 크기: 184, 얼림
- 졸업 시험: `stage4-composite`, 점수 0.4485 (design/world-v4-composite.md)

## 읽기 전문가 (실제 한국어 기사) (`reader-qa`)

- 하는 일: KLUE 기사에서 답 위치를 찾고 답 없음 점수를 낸다
- 읽는 것: (질문, 기사)
- 쓰는 것: 답, 답 점수, 답 없음 점수
- 코드: `world5/reader_qa.py (Reader)`
- 체크포인트: `world-v5_reader-qa.pt`
- 학습: KLUE-MRC 14,054문제 (사용자가 실행, 약 1시간)
- 크기: 2.78억, 얼림
- 졸업 시험: `stage5-reader`, 점수 0.726 (design/world-v5-real-korean.md)

## 신뢰 (실제 기사 출처) (`trust-real`)

- 하는 일: 출처별로 읽은 답과 읽기 점수를 보고, 누가 거짓말하는지 배우며 답을 고른다
- 읽는 것: 출처별 읽기 전문가의 답과 점수, 정답 피드백
- 쓰는 것: 후보 답의 확률 (또는 모름)
- 코드: `world6/trust.py (TrustPart)`
- 체크포인트: `world-v6_trust_seed-{42,43,44}.pt`
- 학습: 실제 기사 + 오보 출처 v6, 세션 2,000개
- 크기: 2,386, 얼림
- 졸업 시험: `stage6-trust-real`, 점수 0.1752 (design/world-v6-real-trust.md)

## 사령탑 (`router`)

- 하는 일: 생각 부품의 상태에서 무엇이 모자라는지 보고 신뢰 부품을 부를지 정한다
- 읽는 것: 기준 문마다 [질문 문과 이어짐, 열쇠 확신도, 답 확신도]
- 쓰는 것: 물어볼 기준 문들
- 코드: `world7/routing.py (Router)`
- 체크포인트: `world-v7_router_seed-{42,43,44}.pt`
- 학습: 사령탑 세계 v7, 물었을 때의 점수 변화를 회귀
- 크기: 1,249, 얼림
- 졸업 시험: `stage7-router`, 점수 0.4109 (design/world-v7-routing.md)

## 성장 (언제 키울지 + 모양 찾기) (`growth`)

- 하는 일: 실패율이 오르면 성장을 결정하고, 오답노트로 가설(열쇠 재배치 6가지 + 더 공부하기)을 시험해 새 가지를 붙인다
- 읽는 것: 문제마다 실패 여부
- 쓰는 것: 새 가지 (생각 부품에 덧붙음)
- 코드: `world8/stream9.py, world8/multibranch.py`
- 체크포인트: 없음 (아직 저장 안 됨)
- 학습: 규칙 (문턱 0.05, 2구간 연속), 가지는 그때그때 학습
- 크기: 가지 하나 약 6,200, 자라는 중
- 졸업 시험: `stage9-growth`, 점수 저장된 부품 없음 (design/world-v8-growth.md)
