"""Thinking satellite for grids: program search over a small DSL (ARC-AGI-1, design/mk1-v2-spec.md, 12).

No learning from data: given a task's demonstration pairs, it searches for a program that maps every demonstration
input to its output exactly, then runs that program on the test input. Two solvers:

  compose    breadth-first search over compositions (depth <= 3) of grid primitives (geometry, cropping, objects,
             scaling, tiling, halves, gravity, filling), with intermediate results deduplicated across all
             demonstration inputs at once; a final colour map is learned from the demonstrations when consistent
  local      for same-shape tasks: a lookup table from a cell's 3x3 neighbourhood to its output colour, learned from
             the demonstrations; it answers only when every test neighbourhood was seen

"I don't know" is part of the answer: when no program fits every demonstration, the satellite says nothing.
"""

import itertools
import time
from collections import Counter

import numpy as np

MARK = 10  # a colour that does not exist in ARC: primitives paint new cells with it, the final colour map names it


def bg(g: np.ndarray) -> int:
    return 0 if (g == 0).any() else int(Counter(g.ravel().tolist()).most_common(1)[0][0])


def crop(g: np.ndarray, b: int | None = None):
    b = bg(g) if b is None else b
    ys, xs = np.nonzero(g != b)
    if len(ys) == 0:
        return None
    return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def components(g: np.ndarray, diagonal: bool, multicolour: bool):
    """Connected non-background components: list of (cells, colours)."""
    b = bg(g)
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    out = []
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if diagonal else [])
    for y in range(h):
        for x in range(w):
            if seen[y, x] or g[y, x] == b:
                continue
            stack, cells = [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in steps:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] != b and (multicolour or g[ny, nx] == g[y, x]):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append(cells)
    return out


def object_crop(g: np.ndarray, cells) -> np.ndarray:
    ys, xs = zip(*cells)
    sub = np.full((max(ys) - min(ys) + 1, max(xs) - min(xs) + 1), bg(g))
    for y, x in cells:
        sub[y - min(ys), x - min(xs)] = g[y, x]
    return sub


def pick(key, diagonal=True, multicolour=True, reverse=False):
    def f(g):
        objs = components(g, diagonal, multicolour)
        if not objs or len(objs) > 30:
            return None
        scores = [key(g, o) for o in objs]
        best = max(scores) if reverse else min(scores)
        if scores.count(best) != 1:
            return None
        return object_crop(g, objs[scores.index(best)])
    return f


def colour_count(g, o):
    return len({int(g[y, x]) for y, x in o})


def unique_colour(g, o):
    """Objects whose colour appears in only one object score 0."""
    colours = Counter(int(g[y, x]) for obj in components(g, True, False) for y, x in obj[:1])
    return 0 if colours[int(g[o[0]])] == 1 else 1


def scale_up(k):
    return lambda g: np.kron(g, np.ones((k, k), dtype=g.dtype)) if g.shape[0] * k <= 30 and g.shape[1] * k <= 30 else None


def scale_down(k):
    def f(g):
        h, w = g.shape
        if h % k or w % k:
            return None
        small = g[::k, ::k]
        return small if (np.kron(small, np.ones((k, k), dtype=g.dtype)) == g).all() else None
    return f


def tile(rows, cols, flip=False):
    def f(g):
        if g.shape[0] * rows > 30 or g.shape[1] * cols > 30:
            return None
        if not flip:
            return np.tile(g, (rows, cols))
        row = [g if c % 2 == 0 else g[:, ::-1] for c in range(cols)]
        lines = [np.hstack(row) if r % 2 == 0 else np.hstack(row)[::-1] for r in range(rows)]
        return np.vstack(lines)
    return f


def halves(axis, part):
    """Split into two halves, by a one-cell separator line if there is one, else exactly in the middle."""
    def f(g):
        a = g if axis == 0 else g.T
        n = a.shape[0]
        mid = n // 2
        if n % 2 == 1 and len(set(a[mid].tolist())) == 1:
            first, second = a[:mid], a[mid + 1:]
        elif n % 2 == 0:
            first, second = a[:mid], a[mid:]
        else:
            return None
        out = first if part == 0 else second
        return out if axis == 0 else out.T
    return f


def overlay(axis, how):
    """Combine the two halves cell-wise into a mask painted MARK (the colour map names it)."""
    def f(g):
        a, b = halves(axis, 0)(g), halves(axis, 1)(g)
        if a is None or b is None or a.shape != b.shape:
            return None
        ba, bb = a != bg(g), b != bg(g)
        m = {"and": ba & bb, "or": ba | bb, "xor": ba ^ bb, "nor": ~(ba | bb), "left": ba & ~bb, "right": bb & ~ba}[how]
        return np.where(m, MARK, 0)
    return f


def gravity(direction):
    def f(g):
        b = bg(g)
        a = np.rot90(g, direction)
        out = np.full_like(a, b)
        for x in range(a.shape[1]):
            col = [v for v in a[:, x] if v != b]
            if col:
                out[-len(col):, x] = col
        return np.rot90(out, -direction)
    return f


def fill_enclosed(g):
    b = bg(g)
    h, w = g.shape
    reach = np.zeros_like(g, dtype=bool)
    stack = [(y, x) for y in range(h) for x in range(w) if (y in (0, h - 1) or x in (0, w - 1)) and g[y, x] == b]
    for y, x in stack:
        reach[y, x] = True
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < h and 0 <= nx < w and not reach[ny, nx] and g[ny, nx] == b:
                reach[ny, nx] = True
                stack.append((ny, nx))
    holes = (g == b) & ~reach
    return np.where(holes, MARK, g) if holes.any() else None


def symmetrise(kind):
    def f(g):
        b = bg(g)
        m = {"lr": g[:, ::-1], "ud": g[::-1], "rot": np.rot90(g, 2), "t": g.T if g.shape[0] == g.shape[1] else None}[kind]
        if m is None:
            return None
        return np.where(g == b, m, g)
    return f


def remove_noise(g):
    b = bg(g)
    out = g.copy()
    for cells in components(g, True, False):
        if len(cells) == 1:
            out[cells[0]] = b
    return out if (out != g).any() else None


def outline(g):
    b = bg(g)
    h, w = g.shape
    out = g.copy()
    for y in range(h):
        for x in range(w):
            if g[y, x] == b and any(0 <= y + dy < h and 0 <= x + dx < w and g[y + dy, x + dx] != b
                                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))):
                out[y, x] = MARK
    return out


def mirror_cat(axis, flip_first=False):
    def f(g):
        a, b = (g[::-1] if axis == 0 else g[:, ::-1]), g
        if flip_first:
            a, b = b, a
        out = np.vstack([b, a]) if axis == 0 else np.hstack([b, a])
        return out if max(out.shape) <= 30 else None
    return f


def rays(directions):
    """Extend every non-background pixel in the given directions to the border, over background only."""
    steps = {"h": [(0, 1), (0, -1)], "v": [(1, 0), (-1, 0)], "d": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
             "right": [(0, 1)], "down": [(1, 0)], "left": [(0, -1)], "up": [(-1, 0)]}[directions]

    def f(g):
        b = bg(g)
        h, w = g.shape
        pts = list(zip(*np.nonzero(g != b)))
        if not pts or len(pts) > 40:
            return None
        out = g.copy()
        for y, x in pts:
            for dy, dx in steps:
                ny, nx = y + dy, x + dx
                while 0 <= ny < h and 0 <= nx < w:
                    if out[ny, nx] == b:
                        out[ny, nx] = g[y, x]
                    ny, nx = ny + dy, nx + dx
        return out
    return f


def connect(g):
    """Fill background between two same-coloured pixels on the same row or column."""
    b = bg(g)
    out = g.copy()
    for a in (out, out.T):
        src = a.copy()
        for i, line in enumerate(src):
            for c in set(line.tolist()) - {b}:
                idx = np.nonzero(line == c)[0]
                if len(idx) >= 2:
                    seg = a[i, idx.min():idx.max() + 1]
                    seg[seg == b] = c
    return out if (out != g).any() else None


def periodic(mask_colour_from):
    """Fill a masked area (one colour) by the smallest 2D period the unmasked cells obey; `patch` returns only the
    masked rectangle."""
    def f(g, patch=False):
        h, w = g.shape
        candidates = [0] if mask_colour_from == "zero" else [int(c) for c in np.unique(g)]
        for c in candidates:
            m = g == c
            if not m.any() or m.all():
                continue
            ys, xs = np.nonzero(~m)
            vals = g[ys, xs]
            for py, px in sorted(((py, px) for py in range(1, h + 1) for px in range(1, w + 1)), key=lambda p: p[0] * p[1]):
                if py * px > h * w // 2:
                    break
                idx = (ys % py) * px + xs % px
                lo = np.full(py * px, 99)
                hi = np.full(py * px, -1)
                np.minimum.at(lo, idx, vals)
                np.maximum.at(hi, idx, vals)
                if (lo == hi).all():
                    full = np.tile(lo.reshape(py, px), (h // py + 1, w // px + 1))[:h, :w]
                    if not patch:
                        return full
                    ys, xs = np.nonzero(m)
                    return full[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return None
    return f


def by_size(g):
    """Paint every object with a code for its size (10 + size); the final colour map names the codes."""
    objs = components(g, False, False)
    if not objs or len(objs) > 30:
        return None
    out = g.copy()
    for cells in objs:
        for y, x in cells:
            out[y, x] = 10 + min(len(cells), 80)
    return out


def size_rank(g):
    """Paint objects by size rank (largest = 10, next = 11, ...), ties make it fail."""
    objs = components(g, False, False)
    sizes = sorted({len(o) for o in objs}, reverse=True)
    if not objs or len(objs) > 12 or len(sizes) != len(objs):
        return None
    out = g.copy()
    for cells in objs:
        for y, x in cells:
            out[y, x] = 10 + sizes.index(len(cells))
    return out


def cells(g):
    """Split by full separator lines (rows/columns of one colour, not background); list of equal-shaped cells."""
    rows = [i for i in range(g.shape[0]) if len(set(g[i].tolist())) == 1 and g[i, 0] != bg(g)]
    cols = [j for j in range(g.shape[1]) if len(set(g[:, j].tolist())) == 1 and g[0, j] != bg(g)]
    if not rows and not cols:
        return None

    def spans(cuts, n):
        edges = [-1] + cuts + [n]
        return [(a + 1, b) for a, b in zip(edges, edges[1:]) if b > a + 1]

    out = [g[a:b, c:d] for a, b in spans(rows, g.shape[0]) for c, d in spans(cols, g.shape[1])]
    return out if len(out) >= 2 and len({c.shape for c in out}) == 1 else None


def cell_pick(kind):
    def f(g):
        cs = cells(g)
        if cs is None:
            return None
        keys = [c.tobytes() for c in cs]
        if kind == "odd":
            counts = Counter(keys)
            odd = [c for c, k in zip(cs, keys) if counts[k] == 1]
            return odd[0] if len(odd) == 1 else None
        score = [int((c != bg(g)).sum()) if kind in ("most", "least") else len(set(c.ravel().tolist())) for c in cs]
        best = max(score) if kind in ("most", "colourful") else min(score)
        return cs[score.index(best)] if score.count(best) == 1 else None
    return f


def cell_overlay(how):
    def f(g):
        cs = cells(g)
        if cs is None:
            return None
        masks = [c != bg(g) for c in cs]
        m = {"or": np.logical_or.reduce(masks), "and": np.logical_and.reduce(masks),
             "xor": np.sum(masks, 0) == 1, "nor": ~np.logical_or.reduce(masks)}[how]
        return np.where(m, MARK, 0)
    return f


def stack_cells(g):
    """Overlay cells, later cells drawn over earlier ones (non-background wins)."""
    cs = cells(g)
    if cs is None:
        return None
    out = cs[0].copy()
    for c in cs[1:]:
        out = np.where(c != bg(g), c, out)
    return out


PRIMITIVES = {
    "rot90": lambda g: np.rot90(g, 1), "rot180": lambda g: np.rot90(g, 2), "rot270": lambda g: np.rot90(g, 3),
    "flip_ud": lambda g: g[::-1], "flip_lr": lambda g: g[:, ::-1], "transpose": lambda g: g.T,
    "anti_transpose": lambda g: np.rot90(g, 2).T,
    "crop": crop, "crop0": lambda g: crop(g, 0),
    "largest": pick(lambda g, o: len(o), reverse=True), "smallest": pick(lambda g, o: len(o)),
    "largest_single": pick(lambda g, o: len(o), multicolour=False, reverse=True),
    "smallest_single": pick(lambda g, o: len(o), multicolour=False),
    "most_colours": pick(colour_count, reverse=True), "unique_colour": pick(unique_colour, multicolour=False),
    "up2": scale_up(2), "up3": scale_up(3), "down2": scale_down(2), "down3": scale_down(3),
    "tile1x2": tile(1, 2), "tile2x1": tile(2, 1), "tile2x2": tile(2, 2), "tile3x3": tile(3, 3),
    "tile2x2_flip": tile(2, 2, True), "tile3x3_flip": tile(3, 3, True),
    "mirror_down": mirror_cat(0), "mirror_right": mirror_cat(1), "mirror_up": mirror_cat(0, True), "mirror_left": mirror_cat(1, True),
    "top": halves(0, 0), "bottom": halves(0, 1), "left": halves(1, 0), "right": halves(1, 1),
    **{f"overlay_{a}_{h}": overlay(i, h) for i, a in enumerate(("rows", "cols")) for h in ("and", "or", "xor", "nor", "left", "right")},
    "gravity_down": gravity(0), "gravity_right": gravity(1), "gravity_up": gravity(2), "gravity_left": gravity(3),
    "fill_enclosed": fill_enclosed, "outline": outline, "remove_noise": remove_noise,
    "sym_lr": symmetrise("lr"), "sym_ud": symmetrise("ud"), "sym_rot": symmetrise("rot"), "sym_t": symmetrise("t"),
    **{f"rays_{d}": rays(d) for d in ("h", "v", "d", "right", "down", "left", "up")},
    "connect": connect,
    "periodic": periodic("any"), "periodic_patch": lambda g: periodic("any")(g, patch=True),
    "by_size": by_size, "size_rank": size_rank,
    **{f"cell_{k}": cell_pick(k) for k in ("odd", "most", "least", "colourful")},
    **{f"cells_{h}": cell_overlay(h) for h in ("or", "and", "xor", "nor")},
    "stack_cells": stack_cells,
}


def colour_map(preds: list[np.ndarray], targets: list[np.ndarray]) -> dict | None:
    m = {}
    for p, t in zip(preds, targets):
        if p.shape != t.shape:
            return None
        for a, b in zip(p.ravel().tolist(), t.ravel().tolist()):
            if m.setdefault(a, b) != b:
                return None
    return m


def apply_map(g: np.ndarray, m: dict) -> np.ndarray | None:
    if any(int(v) not in m for v in np.unique(g)):  # a colour the demonstrations never mapped: do not guess
        return None
    out = np.vectorize(lambda v: m.get(int(v), int(v)))(g)
    return None if (out >= 10).any() else out


def _safe(f, g):
    try:
        out = f(g)
    except Exception:
        return None
    if out is None or out.size == 0 or max(out.shape) > 30:
        return None
    return np.ascontiguousarray(out)


def compose(train, tests, depth=3, budget=10.0, want=2):
    """Programs (tuples of primitive names) consistent with every demonstration, shortest first, with their test
    outputs; up to `want` distinct test answers."""
    started = time.perf_counter()
    inputs = [np.array(x) for x, _ in train] + [np.array(t) for t in tests]
    targets = [np.array(y) for _, y in train]
    n = len(train)
    frontier = [((), inputs)]
    seen = {tuple(g.tobytes() + bytes(g.shape) for g in inputs[:n])}
    answers = []

    def check(prog, grids):
        exact = all(g.shape == t.shape and (g == t).all() for g, t in zip(grids[:n], targets))
        m = None if exact else colour_map(grids[:n], targets)
        if not exact and m is None:
            return
        outs = []
        for g in grids[n:]:
            o = g if exact else apply_map(g, m)
            if o is None or (o >= 10).any():
                return
            outs.append(o.tolist())
        if outs not in [a for _, a in answers]:
            answers.append((prog + (() if exact else ("colour_map",)), outs))

    check((), inputs)
    for _ in range(depth):
        nxt = []
        for prog, grids in frontier:
            for name, f in PRIMITIVES.items():
                if time.perf_counter() - started > budget or len(answers) >= want:
                    return answers
                out = [_safe(f, g) for g in grids]
                if any(o is None for o in out):
                    continue
                key = tuple(g.tobytes() + bytes(g.shape) for g in out[:n])
                if key in seen:
                    continue
                seen.add(key)
                check(prog + (name,), out)
                nxt.append((prog + (name,), out))
        frontier = nxt
    return answers


NEIGHBOURHOODS = {"cell": [(0, 0)], "cross": [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)],
                  "3x3": [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]}


def local_rule(train, tests):
    """Same-shape tasks: output colour = lookup of the input's neighbourhood (padded with -1); the smallest
    neighbourhood that is consistent on the demonstrations and has seen every test neighbourhood answers."""
    pairs = [(np.array(x), np.array(y)) for x, y in train]
    if any(x.shape != y.shape for x, y in pairs):
        return []

    def keys(g, offsets):
        p = np.pad(g, 1, constant_values=-1)
        h, w = g.shape
        return [[tuple(int(p[i + 1 + dy, j + 1 + dx]) for dy, dx in offsets) for j in range(w)] for i in range(h)]

    for name, offsets in NEIGHBOURHOODS.items():
        table, ok = {}, True
        for x, y in pairs:
            for row_k, row_y in zip(keys(x, offsets), y.tolist()):
                for k, v in zip(row_k, row_y):
                    if table.setdefault(k, v) != v:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok or all((x == y).all() for x, y in pairs):
            continue
        outs = []
        for t in tests:
            ks = keys(np.array(t), offsets)
            if any(k not in table for row in ks for k in row):
                outs = None
                break
            outs.append([[table[k] for k in row] for row in ks])
        if outs is not None:
            return [((f"local_rule_{name}",), outs)]
    return []


def solve(task: dict, budget: float = 10.0) -> list[tuple]:
    """Up to two (program, test outputs) answers; empty = "I don't know"."""
    train = [(p["input"], p["output"]) for p in task["train"]]
    tests = [p["input"] for p in task["test"]]
    answers = compose(train, tests, budget=budget)
    for a in local_rule(train, tests):
        if a[1] not in [b for _, b in answers]:
            answers.append(a)
    return answers[:2]
