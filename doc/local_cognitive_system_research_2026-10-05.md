# 로컬 인지 시스템: 모델·논문 적합성 조사 노트

조사 기준일: 2026-10-05

## 범위와 증거의 한계

목표는 한국어 텍스트를 입력받고, 유창한 문장 생성 없이도 기억·상태 갱신·추론·행동 선택을 수행하는 로컬 인지 시스템이다. 음성·시각은 필수 기능에서 제외했다.

이 문서는 공개 모델 카드, Hugging Face 저장소 메타데이터, 저자 공개 평가 자료, 논문 초록 및 접근 가능한 본문에 기반한 설계 조사다. 모든 논문의 전체 본문을 읽었다는 뜻이 아니며, 모델 가중치를 내려받아 추론·학습·속도 측정을 수행한 결과도 아니다. 아래의 채택·보류는 실험 우선순위이지 검증 완료된 성능 순위가 아니다.

핵심 결론: 사전학습 모델은 입력 해석·검색·후보 평가 부품으로 재사용하되, 상태 유지·기억 갱신·호출 순서·중단 판단을 담당하는 작은 학습 가능 코어를 별도로 설계하는 편이 연구 목적에 맞는다. 처음부터 모든 후보를 동시에 탑재하지 않는다.

## 1. 혼동하면 안 되는 기능

| 서로 다른 것 | 설계상 구분 |
|---|---|
| 검색 관련도와 작업기억 제어 | 관련도는 현재 쿼리와 문서의 관계다. 작업기억 제어는 무엇을 유지·교체·읽을지 시간에 걸쳐 결정한다. |
| 응답 선호와 사실성 | 선호 모델의 높은 점수는 사실이라는 보증이 아니다. |
| 텍스트 근거 지지와 세계의 진실 | NLI/grounding 모델은 주어진 근거에서 결론이 따라오는지 본다. 근거 자체가 잘못되었을 수 있다. |
| 제시된 후보 선택과 새로운 계획 구성 | 분류기는 후보에 없는 정답이나 임의 길이의 계획을 자동으로 만들어내지 않는다. |
| 외부 기억 저장과 가중치 학습 | DB에 경험을 저장하는 것은 모델 가중치의 지속학습과 다르다. |
| 구조화 메시지와 공유 잠재공간 | 같은 JSON 스키마를 쓰는 것과 서로 다른 신경망의 내부 표현을 정렬하는 것은 다르다. |
| 논리적 병렬성과 GPU 속도 | 한 GPU에서 여러 모델을 동시에 돌린다고 지연이 반드시 줄어들지는 않는다. |

## 2. 모델 조사 목록

크기는 반올림한 참고값이다. 모델명에 들어간 수, 모델 카드의 집계, 실제 저장된 텐서 수가 다를 수 있다. 체크포인트·정밀도·리비전을 확정한 후 다시 계산해야 한다.

### 2.1 표현 학습·입력 해석·판단

| 정확한 저장소 | 규모 | 본래 기능 | 이 프로젝트에서의 판단 |
|---|---:|---|---|
| [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) | 약 140M | 다국어 마스크 언어 모델 | 한국어를 포함한 공통 인코더 및 자체 분류/NLI 헤드 학습의 우선 후보. 다운로드만으로 범용 판단기가 되는 것은 아님. |
| [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) | 약 307M | 다국어 마스크 언어 모델 | small 대비 정확도·학습 비용 비교군. |
| [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) | 약 150M | 영어 중심 마스크 언어 모델 | 영어 내부 상태 실험에는 후보. 한국어 기본 인코더로는 후순위. |
| [klue/roberta-base](https://huggingface.co/klue/roberta-base) | 약 111M | 한국어 사전학습 모델 | 한국어 전용 소형 기준선. 역할별 지도학습 필요. |
| [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) | 144.3M | 동적 선택지 기반 결정 | 학습 없이 시험할 유한 선택지 판단 기준선. 기본 후보 수 2–20, 자체 런타임, 전체 학습 파이프라인 비공개라는 제약. |
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | 약 421M | 영어 선택·불리언·등급 판단 | 아키텍처 연구 후보. 기본 모델의 적용 성능을 별도 튜닝판 성적으로 대체해서 해석하면 안 됨. |
| [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | 약 322M, 계열 문서 표기 | 다국어 결정 모델 | 학습 실험 후보. 기본 체크포인트의 전이 정확도 및 보정 필요. 계열 문서 중심으로 검토. |
| [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 약 421M, 계열 문서 표기 | 특정 결정 데이터에 튜닝 | 기본 Laya와 별개의 실험 대상으로 관리. 저자 평가 문서가 공개 원시 결과의 부재를 명시한 부분이 있음. |
| [fastino/GLiNER2.5-multi-Decide](https://huggingface.co/fastino/GLiNER2.5-multi-Decide) | 약 287M | 다국어 동적 라벨 분류 | 초기 라우팅·여러 분류 헤드 실행의 우선 시험 후보. 공개 대표 벤치마크는 영어여서 한국어 검증 필요. |
| [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | 카드 340M; Hub 텐서 집계와 차이 | 영어 동적 라벨 분류 | 영어 비교군. 한국어 시스템의 기본값으로 선택하지 않음. 크기는 리비전 고정 후 재확인. |
| [fastino/gliner2.5-multi-v1](https://huggingface.co/fastino/gliner2.5-multi-v1) | 약 287M | 다국어 엔티티·관계·구조화 추출·분류 | 입력 구조화 우선 후보. Decide와 역할이 겹치므로 둘 다 상시 탑재하기 전에 하나로 충분한지 검사. |
| [fastino/gliner2-base-v1](https://huggingface.co/fastino/gliner2-base-v1) | 약 209M | 영어 중심 구조화 추출 | 이전 세대 비교군. 한국어 기본 선택으로 보류. |
| [fastino/gliner2-large-v1](https://huggingface.co/fastino/gliner2-large-v1) | Hub 약 486M | 구조화 추출 | 용량 증가 대비 이익과 언어 범위를 확인해야 함. 초기 구성에서는 제외. |
| [urchade/gliner_multi-v2.1](https://huggingface.co/urchade/gliner_multi-v2.1) | 약 289M | 다국어 엔티티 추출 | 이름·시간·대상 등 원문 span 추출 비교군. 완전한 의미 파서가 아님. |
| [numind/NuExtract-2.0-2B](https://huggingface.co/numind/NuExtract-2.0-2B) | 약 2.2B 전체 | 멀티모달 구조화 추출 | 메타데이터 수준의 확장 후보. 텍스트 전용 소형 실험에는 비용이 커 우선 보류. |

### 2.2 검색·연상 기억

| 저장소 | 규모 / 입력 | 적합성 |
|---|---|---|
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | 약 118M / 512 토큰 / 384차원 | 짧은 대화 사건·경험 검색의 경량 기준선. query/passage 규약과 잘림 처리 필요. |
| [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) | 약 300M / 2,048 토큰 | 온디바이스 다국어 검색 후보. Gemma 이용 조건, 출력 차원 및 실제 런타임을 확인. |
| [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) | 약 305M / 8,192 토큰 | 더 긴 기록 검색 후보. 사용자 정의 코드 의존성 관리 필요. |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 약 596M / 32K | 긴 문맥·지시 기반 검색 비교군. 짧은 사건 검색에서 추가 비용이 보상되는지 측정. |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 약 568M / 8,192 토큰 | dense·sparse·multi-vector 혼합 검색이 필요할 때 후보. 작은 경험 저장소에는 복잡성이 불필요할 수 있음. |
| [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | 약 118M | 다국어 문장 유사도 비교군. 질문→근거 검색 목적과 학습 목적의 차이를 확인해야 함. |

### 2.3 재순위화·근거 검증·품질 평가

| 저장소 | 규모 | 적합성 / 한계 |
|---|---:|---|
| [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) | 약 306M | 다국어 문서 관련도 우선 비교군. 작업기억 유지·교체 정책은 별도. |
| [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 약 596M | 지시 기반 관련도 비교군. CausalLM의 yes/no 로짓을 쓰는 점수화 방식이지 독립 encoder-only 모델은 아님. |
| [jinaai/jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) | 약 278M | 다국어 재순위화 후보. CC-BY-NC-4.0 표기와 긴 입력의 sliding-window 처리를 확인. |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 약 568M | 다국어 재순위화 비교군. 작은 시스템에서는 메모리 대비 이익 검증 필요. |
| [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) | 약 118M | 매우 작은 검색 비교군. 공개 학습 언어 목록에 한국어가 없어 한국어 우선 배치에는 신중. |
| [MoritzLaurer/mDeBERTa-v3-base-mnli-xnli](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) | 약 279M | 함의·모순·중립의 실제 NLI 학습 목적이 근거 검증과 맞음. 다국어 기반이라는 사실과 한국어 NLI 평가 근거를 구분. |
| [vectara/hallucination_evaluation_model](https://huggingface.co/vectara/hallucination_evaluation_model) | 약 110M | HHEM-2.1-Open: 근거→주장의 지지 평가. 공개 모델과 별도 플랫폼의 다국어 최신 버전을 혼동하지 말 것. |
| [Skywork/Skywork-Reward-V2-Qwen3-0.6B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B) | 약 596M | 응답 후보 선호·품질 평가 후보. 진실 확률이나 장기 행동 가치로 사용하지 않음. |
| [project-themis/Themis-RM-0.6B](https://huggingface.co/project-themis/Themis-RM-0.6B) | 약 596M | 코드 품질 다기준 보상 평가. 실제 실행·테스트·정적 검사 대체 불가. 코드가 범위에 들어올 때만 추가. |

### 2.4 선택적 생성·계획 후보 제안

| 저장소 | 규모 | 적합성 |
|---|---:|---|
| [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) | 약 1.17B | 텍스트 전용, 한국어 명시. 작은 구조화 후보 생성기의 비교 대상. LFM 별도 라이선스 확인. |
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 언어 0.8B, 전체 약 0.87B | 태스크 튜닝·프로토타이핑 후보. 기본 모델을 신뢰성 검증 없이 중앙 제어기로 삼지 않음. |
| [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 언어 2B, 전체 약 2.27B | 유한 후보로 다루기 어려운 계획·가설 제안의 비교 기준선. 전체 제어권은 별개 문제. |
| [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | 약 3.84B | 한국어 포함 범용 추론·코드 대안. 더 큰 메모리·연산 비용을 감수할 만큼 이익이 있는지 측정. |
| [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 약 3.08B | 공개 학습·영어 실험 비교군. 한국어를 핵심 지원으로 확인하지 못해 기본 선택에서 후순위. |
| [mistralai/Ministral-3-3B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512) | Hub 전체 약 4.25B | 추론 전문 비교군. 이름의 3B만으로 전체 적재 용량을 계산하면 안 됨. 초기 최소 구성에는 보류. |

## 3. 중요한 공개 평가 감사 결과

### Julia-1

[저자 평가 원자료](https://huggingface.co/SupersonicLabs/Julia-1/blob/main/metrics/accuracy-20260924.json)는 한국어 MASSIVE에서 2,974개 중 2,296개 정답, 약 77.20%를 기록한다. 하지만 모델 카드가 명시하듯 평가 대상은 18개 scenario 분류다. 60개 intent 전체, 슬롯 추출, 일반 추론 성적이 아니다. typed-decisions 결과는 2,000개 판단 중 1,463개, 73.15%다. Jev 비교값은 별도 공개 참고치이지 동일 환경 재실행 비교가 아니다. 학습 코드 전체가 공개된 모델과 재현성 수준이 같다고 보지 않는다.

### Laya

[저자 BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)에서 typed-decisions 기본 영어 모델은 정확도 0.362, 다수결 기준선은 0.461이다. 튜닝판은 0.766으로 보고되지만, 해당 문서는 그 튜닝 결과 행을 뒷받침하는 committed result file이 아직 없다고 명시한다. 따라서 기본 모델의 실패와 튜닝의 가능성을 함께 기록하되, 튜닝 성적을 독립 재현된 사실로 취급하지 않는다. 모델 카드와 상세 문서의 다국어 수치도 일부 다르므로 리비전 고정이 필요하다.

### GLiNER2.5 Decide

다국어 모델의 기능과 출력 형식은 프로젝트에 가깝지만 대표 공개 비교표는 영어 17개 도메인이다. 영어 평균을 한국어 성능으로 환산하지 않는다. 영어 모델의 카드 파라미터 수와 Hub 텐서 집계 차이는 체크포인트 고정 후 확인할 항목이다.

### HHEM

오픈 체크포인트 HHEM-2.1-Open의 범위와 Vectara 플랫폼에서 제공하는 후속 다국어 버전의 범위를 구분한다. 문서 내 한국어 지원 언급을 오픈 체크포인트의 검증 결과로 옮기면 안 된다.

### HRM

[논문 v3의 평가 설명](https://arxiv.org/html/2506.21734v3)에서 ARC는 테스트 입력당 1,000개 증강 변형을 풀어 투표한다. 27M 파라미터라는 크기만 보고 전체 추론 비용이 작다고 결론 내릴 수 없다. 학습·테스트 증강, 반복 깊이, task-specific training을 함께 비교해야 한다.

## 4. 논문 지도: 원리, 구현, 반례

| 논문 / 자료 | 이 프로젝트에서 읽는 이유 | 바로 등치하면 안 되는 것 |
|---|---|---|
| [Fedorenko et al., 2024 — Language is primarily a tool for communication rather than thought](https://www.nature.com/articles/s41586-024-07522-w) | 언어 처리와 다른 인지 기능의 구분을 검토 | 인간의 기능적 분리가 LLM의 추론 불가능성을 증명하지는 않음. Perspective 논문. |
| [Riveland & Pouget, 2024 — Natural language instructions induce compositional generalization in networks of neurons](https://www.nature.com/articles/s41593-024-01607-5) | 사전학습 언어 표현과 별도 과제 수행 신경망을 연결하는 직접적인 선례 | 심리물리 과제에서의 일반화를 범용 대화 지능으로 확대하지 않음. 접근 가능한 초록·논문 발췌 중심으로 검토. |
| [VanRullen & Kanai, 2021 — Deep Learning and the Global Workspace Theory](https://arxiv.org/abs/2012.10390) | 전문 모듈의 서로 다른 표현을 공통 잠재공간으로 연결하는 구상 | 완성된 범용 모델이 아니라 연구 로드맵. JSON 공유와 latent alignment는 다름. |
| [Goyal et al., 2022 — Coordination Among Neural Modules Through a Shared Global Workspace](https://arxiv.org/html/2103.01197v2) | 제한된 메모리, 쓰기 경쟁, broadcast를 실제 학습 가능한 신경망에 구현 | 함께 학습한 모듈의 결과가 임의의 frozen 모델 조합에서 자동 재현되는 것은 아님. |
| [O'Reilly & Frank, 2006 — Making Working Memory Work](https://pubmed.ncbi.nlm.nih.gov/16378516/) | 정보를 언제 유지·갱신할지 학습하는 gating | 순간적인 문서 relevance ranking과 시간적 기억 제어를 구분. |
| [Kumaran et al., 2016 — What Learning Systems do Intelligent Agents Need?](https://pubmed.ncbi.nlm.nih.gov/27315762/) | 빠른 경험 저장과 느린 일반화를 포함하는 CLS 업데이트 | 해마=DB, 대뇌피질=LLM이라는 일대일 대응은 과도한 단순화. |
| [Cole et al., 2013 — Multi-task connectivity reveals flexible hubs for adaptive task control](https://pmc.ncbi.nlm.nih.gov/articles/PMC3758404/) | 과제에 따라 연결 방식이 달라지는 통제 | 기능 연결 관측은 독립적인 모델 라우터가 뇌에 존재한다는 증거가 아님. |
| [Rigotti et al., 2013 — The importance of mixed selectivity in complex cognitive tasks](https://www.nature.com/articles/nature12160) | 하나의 표현에 여러 과제 변수가 섞일 수 있다는 근거 | 기능마다 별도 거대 모델을 하나씩 두어야 한다는 주장에 대한 제동. |
| [Shenhav et al., 2013 — The expected value of control](https://pmc.ncbi.nlm.nih.gov/articles/PMC3767969/) | 추가 연산의 기대 이익과 비용을 함께 고려하는 제어 | 응답 선호 reward와 장기 제어 가치는 다름. |
| [Mattar & Daw, 2018 — Prioritized memory access explains planning and hippocampal replay](https://www.nature.com/articles/s41593-018-0232-z) | 기억 접근을 미래 의사결정 효용과 연결 | 의미 유사도만으로 기억 우선순위를 결정하지 않기. |
| [Rao & Ballard, 1999 — Predictive coding in the visual cortex](https://pubmed.ncbi.nlm.nih.gov/10195184/) | 예측과 관측의 오차를 이용하는 계산 원리 | 두 언어 모델의 의견 불일치가 곧 predictive coding은 아님. |
| [Goyal et al., ICLR 2021 — Recurrent Independent Mechanisms](https://openreview.net/forum?id=mLcmdlEUxy-) | 작은 순환 모듈의 선택적 업데이트와 희소 통신 | 자체 학습해야 하는 구조이며 범용 사전학습된 ‘뇌 부품’은 아님. |
| [Graves et al., 2016 — Hybrid computing using a neural network with dynamic external memory](https://www.nature.com/articles/nature20101) | DNC: 외부 메모리 읽기·쓰기와 제어기를 학습하는 선례 | DB를 연결하기만 한 시스템과 구분. |
| [Wang et al., 2025 — Hierarchical Reasoning Model](https://arxiv.org/html/2506.21734v3) | 두 시간척도의 재귀 계산과 작업별 비언어 추론 | 숫자·격자용 체크포인트가 바로 자연어 추론 모델이 되는 것은 아님. |
| [Jolicoeur-Martineau, 2025 — Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/html/2510.04871v1) | TRM: 더 단순한 작은 재귀 모델이라는 강한 비교군 | 복잡한 뇌 비유보다 반복 계산과 학습법이 이익의 원인일 수 있음. |
| [Hafner et al., 2025 — Mastering diverse control tasks through world models](https://www.nature.com/articles/s41586-025-08744-2) | DreamerV3: 상태 전이 예측과 행동 가치 학습 | 영상·제어 환경에서의 성공을 범용 텍스트 인지로 즉시 전용하지 않음. |
| [Laird et al., 2017 — A Standard Model of the Mind](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2744) | 기존 cognitive architecture들의 구성과 선행 연구 확인 | 새로운 HF 모델을 붙였다는 이유만으로 아키텍처 자체의 신규성을 주장하지 않음. |
| [Guo et al., 2017 — On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) | 확률 보정과 temperature scaling | softmax 값이 곧 실제 정답 확률은 아님. |
| [Ovadia et al., 2019 — Can You Trust Your Model's Uncertainty?](https://arxiv.org/abs/1906.02530) | 분포 이동에서의 불확실성 검증 | 기존 검증셋에서의 보정이 새로운 도메인에 보장되지 않음. |
| [Cogitate Consortium et al., 2025 — Adversarial testing of GNW and IIT](https://www.nature.com/articles/s41586-025-08888-1) | 의식 이론의 실험적 검증과 한계 | 공유 작업공간을 만든 것을 의식의 구현으로 부르지 않음. 본문 접근 제한으로 상세 실험 수치 해석은 하지 않음. |
| [Naccache et al., 2025 — GNW theoretical framework and the adversarial testing](https://academic.oup.com/nc/article/2025/1/niaf037/8280147) | 위 실험 해석에 대한 이론 측의 응답 | 특정 이론이 단순히 입증/폐기되었다는 이분법을 피함. |
| [Park et al., 2021 — KLUE](https://arxiv.org/abs/2105.09680) | 한국어 NLI·관계 추출·대화 상태 추적 등 부품 평가 | 벤치마크 점수만으로 전체 인지 성능을 대신하지 않음. |

## 5. 권장 구성: 모델 조합과 학습 코어를 분리

### 첫 번째 실험: 작동하는 모듈식 시스템

- 입력 구조화: GLiNER2.5 multi-v1과 간단한 규칙 파서 비교.
- 유한 선택: Julia-1과 GLiNER2.5-multi-Decide 비교. 둘을 모두 최종 구성에 넣을 필요는 없음.
- 사건 검색: multilingual-e5-small을 경량 기준선으로, EmbeddingGemma를 대체 후보로 둠.
- 재순위화: 실제 필요성이 입증될 때만 gte-multilingual-reranker-base 추가.
- 근거 검사: NLI 모델 및 수치·시간·규칙의 결정적 검사를 분리.
- 작업기억: 출처·시간·관측/추측 구분·삭제/대체 기록을 갖춘 명시적 상태 구조.
- 제어: 우선 상태기계. 이후 성공 결과가 달린 episode로 GRU/작은 정책망 학습.
- 열린 후보 생성: 필요할 때만 LFM2.5-1.2B 또는 Qwen3.5-2B. 출력은 제한된 계획/가설로 관리.

### 두 번째 실험: 실제 비언어 인지 코어

자연어 입력을 태스크 표현으로 옮긴 뒤, 작은 GRU 기준선과 RIMs+shared workspace를 같은 과제·데이터·계산 예산으로 학습한다. 상태 유지, 모듈 활성화, 기억 읽기/쓰기, 선택, 중단이 행동 결과에 의해 개선되어야 한다. 한 네트워크에 여러 헤드를 붙인 구성도 필수 비교군이다.

LLM을 제거했을 때에도 정해진 세계에서 새 규칙을 기억하고 적용하거나 여러 행동을 조합할 수 있는지를 본다. 범위 밖의 자유 질문은 정직하게 미지원/추가 정보 필요로 처리한다.

## 6. 인터페이스와 학습 목표

명시적 공유 상태: goal, observations, source IDs, timestamps, beliefs, alternatives, memory references, valid actions, outcome, remaining budget.

latent 공간 공유는 후속 연구로 분리한다. E5 벡터와 Qwen hidden state는 차원이 같아도 의미 체계가 같지 않다. 대응 데이터와 adapter 학습, downstream 과제 손실이 필요하다.

제안하는 코어 학습 항목은 다음과 같다. 이것들은 확인된 기존 모델 성능이 아니라 새 시스템의 실험 설계다.

| 대상 | 입력과 출력 | 초기 지도 신호 |
|---|---|---|
| 작업기억 gate | 현재 상태·새 정보 → KEEP/WRITE/REPLACE | 미래 과제에 필요한 사실의 유지 여부 |
| 호출 정책 | 상태·예산 → 다음 모듈/중단 | 실제 성공률 향상과 비용 |
| 근거 판단 | 근거·주장 → 지지/모순/불명 | NLI와 도메인별 검증 정답 |
| 상태 전이 예측 | 상태·행동 → 다음 상태 | 실제 실행 후 관측 |
| 행동 가치 | 상태·행동 후보 → 예상 효용 | 환경에서 측정한 결과 |
| 기억 재학습 | 검증된 경험 → 정책/표현 갱신 | 성공·실패 episode와 별도 유지 검증셋 |

## 7. 평가 설계

필수 과제: 방해 정보 속 작업기억 유지, 규칙 전환, 오래된 사실 수정, 시간·출처 충돌, 신규 대상에 대한 기존 규칙 조합, 유한 행동 계획, 근거 부족 시 보류.

필수 비교군: 동일 정보·도구를 갖춘 단일 소형 LLM, 고정 파이프라인, 상태 없는 분류기, 단일 GRU, RIMs, 공유 작업공간 제거, gate 제거, 동일 모델 추가 추론, 서로 다른 모델 대신 공유 backbone+여러 head.

훈련/검증/테스트는 행 단위 무작위 분할만 하지 않는다. 문장 템플릿·세계 설정·규칙 조합·시간을 분리하고, 여러 질문이 한 사건에서 나온 경우 사건 단위로 신뢰구간을 계산한다. 한국어 부정·조건·시점 변경·대명사·영한 혼용·선택지 순서 변경을 포함한다.

평가값: 전체 성공률, 기억 유지/수정 정확도, 모순 탐지, 보류율과 보류 후 오류율, 확률 보정, p50/p95 지연, cold/warm 구분, peak RAM/VRAM, 모델 호출 횟수, 생성 토큰 수, 총 반복 계산량. 특정 장치의 실제 수치는 아직 없음.

## 8. 로컬 실행의 주의점

가중치 메모리의 단순 하한은 파라미터 수 × 비트 수 ÷ 8이다. 실행 시에는 activation, KV/recurrent state, tokenizer, batch, runtime, workspace 등이 추가된다. GPU 동시 실행과 논리적 병렬 실행을 구분하고, 매번 모델을 언로드/재로드하는 비용도 측정한다.

8GB VRAM 실험에서는 작은 인코더 일부를 CPU에 상주시키고 GPU의 큰 생성 모델은 하나 이하로 시작하는 방식을 검토한다. 16GB에서는 여유가 늘지만 모든 후보를 올리는 것 자체를 목표로 삼지 않는다. 이는 배치 제안이며 확인된 실측 용량이 아니다.

배포 전에는 저장소 리비전·가중치 해시·라이선스·토크나이저·정밀도를 고정하고, custom code와 종속성을 검토한다. 다운로드 이후의 네트워크 차단 상태에서도 동작하는지 확인한다. 별도 프로세스 모델 서버는 우선 loopback으로만 노출한다.

## 최종 연구 질문

‘작은 모델 여러 개를 연결하면 똑똑해지는가?’보다 다음 질문이 검증 가능하다.

**한국어 입력을 해석하는 사전학습 부품과 학습 가능한 작은 상태 제어 코어를 결합했을 때, 제한된 기억·연산 예산에서 단일 모델이나 고정 파이프라인보다 기억 유지·규칙 전환·근거 기반 행동 선택이 개선되는가?**

모듈을 늘렸을 때 이득이 없다면 그 결과도 유효하다. 모듈 수가 아니라, 어떤 정보 교환과 어떤 학습 신호가 성능을 만들었는지가 연구 결과다.
