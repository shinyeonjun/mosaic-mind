"""Part cards: every grown part of MK1, what it reads and writes, how it was trained, and the score it
graduated with. The graduation record is what `cognitive_lab.mk1.checks` re-measures.

python -m cognitive_lab.mk1.registry      # writes design/mk1-parts.md
"""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PartCard:
    name: str
    role: str  # plain words, as used with the user
    does: str
    reads: str
    writes: str
    code: str
    checkpoints: tuple[str, ...]
    trained: str
    parameters: str
    frozen: bool
    graduation: dict = field(default_factory=dict)  # {"check": name, "score": expected, "where": design doc}


PARTS: tuple[PartCard, ...] = (
    PartCard(
        name="reader-v1", role="읽기 (만든 세계)", does="문장과 가설 쌍을 읽어 768차원 특징을 낸다",
        reads="(문장, 가설) 쌍", writes="768-d 특징 (모든 작은 부품의 입력)", code="mk1/reader.py (ReaderService)",
        checkpoints=("world-v1_reader-anchored-learned-sourcefold-staged-tuned_composition_seed-42.pt",),
        trained="규칙 세계 v1, 순서대로 키우기(읽기 → 판단 → 함께 다듬기)", parameters="2.78억", frozen=True,
        graduation={"check": "reader-parity", "score": 0.0, "where": "design/interface-learning-v1.md"}),
    PartCard(
        name="trust-v2h", role="신뢰 + 기억 (말투 포함)",
        does="사람마다 맞은 말을 기억하고 누구를 믿을지 판단한다. 말투 단서도 쓴다",
        reads="사람의 말(문장) × 기준 문 × 열쇠의 읽기 특징, 정답 피드백", writes="열쇠별 확률 (기준 문마다)",
        code="world2/hedged_system.py (ConnectedSystem)", checkpoints=("world-v2h_connected-m8_seed-{42,43,44}.pt",),
        trained="말투 세계 v2-H, 세션 3,000개, 최종 답만", parameters="연결 머리 6,152 + 기억·판단 19,361", frozen=True,
        graduation={"check": "stage1-trust", "score": 0.4148, "where": "design/world-v2h-tone-cues.md"}),
    PartCard(
        name="curiosity", role="호기심 (물어보기)", does="누구에게 물으면 점수가 얼마나 오를지 예측해 질문할지 정한다",
        reads="신뢰 부품의 기억과 상태", writes="물을 사람 (또는 안 물음)", code="world2/asking_system.py (CuriosityHead)",
        checkpoints=("world-v2a_curiosity_cost-0.2_seed-{42,43,44}.pt",),
        trained="물어보기 세계 v2-A, 물어본 결과의 점수 변화를 회귀", parameters="10,689", frozen=True,
        graduation={"check": "stage2-curiosity", "score": 0.5311, "where": "design/world-v2a-asking.md"}),
    PartCard(
        name="thinker", role="생각 (고리 따라가기)",
        does="문마다 상태를 두고 한 번에 한 칸씩 정보를 전달한다. 답이 더 안 바뀌면 멈춘다",
        reads="안내판 문장의 사실·연결 특징, (선택) 다른 부품이 넘긴 사실", writes="문마다 열쇠 믿음, 질문 문의 답",
        code="world3/thinker.py (Thinker)", checkpoints=("world-v3_thinker-grown_seed-{45,46,47}.pt",),
        trained="이어진 단서 v3: 연결 학습(모두 켜고 가지치기) → 연결 얼리고 모든 문 채점", parameters="25,747", frozen=True,
        graduation={"check": "stage3-thinker", "score": 0.975, "where": "design/world-v3-thinking.md"}),
    PartCard(
        name="adapter-trust-thinker", role="연결 (신뢰 → 생각)",
        does="신뢰 부품의 결론을 생각 부품이 받는 사실 메시지로 바꾼다 (충실한 전달)",
        reads="기준 문마다 신뢰 부품의 결론", writes="생각 부품의 사실 메시지", code="world4/system.py (CompositeSystem.adapter)",
        checkpoints=("world-v4_grown_seed-{42,43,44}.pt",),
        trained="복합 세계 v4, 정답 없이 신뢰 부품의 결론을 그대로 전달하도록", parameters="184", frozen=True,
        graduation={"check": "stage4-composite", "score": 0.4485, "where": "design/world-v4-composite.md"}),
    PartCard(
        name="reader-qa", role="읽기 전문가 (실제 한국어 기사)", does="KLUE 기사에서 답 위치를 찾고 답 없음 점수를 낸다",
        reads="(질문, 기사)", writes="답, 답 점수, 답 없음 점수", code="world5/reader_qa.py (Reader)",
        checkpoints=("world-v5_reader-qa.pt",), trained="KLUE-MRC 14,054문제 (사용자가 실행, 약 1시간)", parameters="2.78억",
        frozen=True, graduation={"check": "stage5-reader", "score": 0.726, "where": "design/world-v5-real-korean.md"}),
    PartCard(
        name="trust-real", role="신뢰 (실제 기사 출처)",
        does="출처별로 읽은 답과 읽기 점수를 보고, 누가 거짓말하는지 배우며 답을 고른다",
        reads="출처별 읽기 전문가의 답과 점수, 정답 피드백", writes="후보 답의 확률 (또는 모름)", code="world6/trust.py (TrustPart)",
        checkpoints=("world-v6_trust_seed-{42,43,44}.pt",), trained="실제 기사 + 오보 출처 v6, 세션 2,000개", parameters="2,386",
        frozen=True, graduation={"check": "stage6-trust-real", "score": 0.1752, "where": "design/world-v6-real-trust.md"}),
    PartCard(
        name="router", role="사령탑", does="생각 부품의 상태에서 무엇이 모자라는지 보고 신뢰 부품을 부를지 정한다",
        reads="기준 문마다 [질문 문과 이어짐, 열쇠 확신도, 답 확신도]", writes="물어볼 기준 문들", code="world7/routing.py (Router)",
        checkpoints=("world-v7_router_seed-{42,43,44}.pt",), trained="사령탑 세계 v7, 물었을 때의 점수 변화를 회귀",
        parameters="1,249", frozen=True,
        graduation={"check": "stage7-router", "score": 0.4109, "where": "design/world-v7-routing.md"}),
    PartCard(
        name="growth", role="성장 (언제 키울지 + 모양 찾기)",
        does="실패율이 오르면 성장을 결정하고, 오답노트로 가설(열쇠 재배치 6가지 + 더 공부하기)을 시험해 새 가지를 붙인다",
        reads="문제마다 실패 여부", writes="새 가지 (생각 부품에 덧붙음)", code="world8/stream9.py, world8/multibranch.py",
        checkpoints=(), trained="규칙 (문턱 0.05, 2구간 연속), 가지는 그때그때 학습", parameters="가지 하나 약 6,200",
        frozen=False, graduation={"check": "stage9-growth", "score": None, "where": "design/world-v8-growth.md"}),
)


def markdown() -> str:
    lines = ["# MK1 부품 명함", "", "`src/cognitive_lab/mk1/registry.py`에서 만든다. 졸업 점수는 `python -m cognitive_lab.mk1.checks`가 다시 잰다.", ""]
    for p in PARTS:
        score = p.graduation.get("score")
        lines += [f"## {p.role} (`{p.name}`)", "",
                  f"- 하는 일: {p.does}", f"- 읽는 것: {p.reads}", f"- 쓰는 것: {p.writes}",
                  f"- 코드: `{p.code}`", f"- 체크포인트: {', '.join(f'`{c}`' for c in p.checkpoints) or '없음 (아직 저장 안 됨)'}",
                  f"- 학습: {p.trained}", f"- 크기: {p.parameters}, {'얼림' if p.frozen else '자라는 중'}",
                  f"- 졸업 시험: `{p.graduation.get('check')}`, 점수 {score if score is not None else '저장된 부품 없음'} ({p.graduation.get('where')})", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    out = PROJECT_ROOT / "design" / "mk1-parts.md"
    out.write_text(markdown(), encoding="utf-8")
    print(f"wrote {out} ({len(PARTS)} parts)")
