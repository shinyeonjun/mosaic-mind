"""Entities, sources, and Korean phrasing templates for rule world v1.

Door names end in 문 and key names end in 열쇠 for every entity set, so each
template can use fixed particles (문은/문을/문이, 열쇠로/열쇠가/열쇠를).
Doors are named by color and keys by unrelated nouns, so no door shares a name
with a key (a shared color invited a "red door -> red key" guess in L1).
"""

UNKNOWN = "모름"

# Index is the trust rank: a higher index wins when sources disagree.
SOURCES = ("지나가던 사람", "안내판", "관리인")

ENTITY_SETS = {
    "base": {"doors": ("빨간", "파란", "초록"), "keys": ("해", "달", "별")},
    "validation": {"doors": ("노란", "보라", "하얀"), "keys": ("구름", "바람", "번개")},
    "test": {"doors": ("검은", "주황", "분홍"), "keys": ("꽃", "돌", "나무")},
}

WORLD_RULES = (
    "모든 문은 정확히 한 종류의 열쇠로 열린다. 여러 문이 같은 열쇠를 쓸 수도 있다. "
    "반박되지 않은 말은 믿는다. 같은 사람이 정정하면 정정한 말을 따른다. "
    "말이 서로 다르면 관리인 > 안내판 > 지나가던 사람 순으로 믿는다. "
    "알 수 없으면 '모름'이라고 답한다."
)

TEMPLATES = {
    "assert": (
        ("assert.01", "{D}은 {K}로 열려."),
        ("assert.02", "{D}은 {K}로 열린다."),
        ("assert.03", "{D}은 {K}로 열립니다."),
        ("assert.04", "{D}은 {K}로 열어야 해요."),
        ("assert.05", "{K}로 {D}을 열 수 있어."),
        ("assert.06", "{D} 열쇠는 {K}야."),
        ("assert.07", "{D}에는 {K}가 맞아요."),
        ("assert.08", "{D}? 그건 {K}로 열면 돼."),
        ("assert.09", "{K}를 쓰면 {D}이 열린대요."),
        ("assert.10", "{D}을 여는 건 {K}다."),
    ),
    "negate": (
        ("negate.01", "{D}은 {K}로는 안 열려."),
        ("negate.02", "{D}은 {K}로 열리지 않는다."),
        ("negate.03", "{K}로는 {D}을 못 열어요."),
        ("negate.04", "{D}에 {K}는 안 맞아."),
        ("negate.05", "{K}는 {D} 열쇠가 아닙니다."),
        ("negate.06", "{D}을 {K}로 열려고 하면 안 열려요."),
        ("negate.07", "{D}은 {K}로 여는 게 아니다."),
    ),
    "correct": (
        ("correct.01", "아, 내가 잘못 말했다. {D}은 {K}야."),
        ("correct.02", "정정할게. {D}은 {K}로 열려."),
        ("correct.03", "아까 말한 거 틀렸어요. {D}은 {K}로 열어야 해요."),
        ("correct.04", "아니, 그게 아니라 {D}은 {K}로 열린다."),
        ("correct.05", "다시 말할게요. {D} 열쇠는 {K}예요."),
        ("correct.06", "방금 건 취소. {D}은 {K}가 맞아."),
        ("correct.07", "잘못 알려줬네. {K}로 {D}을 열면 돼."),
    ),
    "chatter": (
        ("chatter.01", "오늘 날씨 좋네요."),
        ("chatter.02", "여기 좀 춥다."),
        ("chatter.03", "점심 뭐 먹지?"),
        ("chatter.04", "이 복도는 꽤 길군요."),
        ("chatter.05", "어제 비가 많이 왔어."),
        ("chatter.06", "조명이 좀 어둡네."),
        ("chatter.07", "잠깐 쉬었다 가자."),
        ("chatter.08", "바닥이 미끄러우니 조심하세요."),
    ),
    "question": (
        ("question.01", "{D}은 무슨 열쇠로 열어?"),
        ("question.02", "{D}을 열려면 어떤 열쇠가 필요해?"),
        ("question.03", "{D}에 맞는 열쇠는?"),
        ("question.04", "{D}은 어떤 열쇠로 열리나요?"),
        ("question.05", "{D}을 여는 열쇠가 뭐야?"),
    ),
}


def door_name(label: str) -> str:
    return f"{label} 문"


def key_name(label: str) -> str:
    return f"{label} 열쇠"


def phrasing_part(index: int) -> str:
    """Assign template index to a phrasing-split part: every fifth goes to test."""
    if index % 5 == 4:
        return "test"
    if index % 5 == 3:
        return "validation"
    return "train"
