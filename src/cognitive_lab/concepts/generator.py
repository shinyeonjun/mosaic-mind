"""Synthetic grid tasks from core-knowledge priors (design/build-from-scratch-2026-10-10.md, step 1).

Each FAMILY is one concept. Every call draws a new task of that concept: its parameters (colours, directions, sizes,
counts) change from task to task, so a learner must read the parameters off the demonstrations. A task has the ARC
format: {"train": [{"input", "output"}] * 3, "test": [{"input", "output"}]}, grids 6-14 wide, background 0.

Families come from Chollet's core-knowledge priors (objectness, geometry and topology, numbers and counting,
goal-directedness) plus comparison (same / different). They were written from that general list, not from any
benchmark's tasks; ConceptARC's 16 concept names and its 16 minimal tasks had been seen before this file was written,
so some names overlap (recorded in the design doc). The ConceptARC corpus stays sealed.

python -m cognitive_lab.concepts.generator --check
"""

import argparse
import random

import numpy as np

DEMOS = 3


# ----------------------------------------------------------------------------- helpers
def blank(rng, lo=6, hi=14):
    return np.zeros((rng.randint(lo, hi), rng.randint(lo, hi)), dtype=int)


def colours(rng, k, exclude=(0,)):
    return rng.sample([c for c in range(1, 10) if c not in exclude], k)


def shape(rng, h, w, fill=0.6):
    """A random 4-connected blob inside an h x w box (at least 2 cells)."""
    if h * w < 2:  # a 1 x 1 box can never hold 2 cells
        w = 2
    while True:
        m = np.zeros((h, w), dtype=bool)
        y, x = rng.randrange(h), rng.randrange(w)
        m[y, x] = True
        for _ in range(max(1, int(h * w * fill))):
            ys, xs = np.nonzero(m)
            i = rng.randrange(len(ys))
            dy, dx = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            ny, nx = ys[i] + dy, xs[i] + dx
            if 0 <= ny < h and 0 <= nx < w:
                m[ny, nx] = True
        if m.sum() >= 2:
            ys, xs = np.nonzero(m)
            return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def place(rng, g, mask, colour, tries=60, gap=1):
    """Put a mask on free cells with a free margin; returns (y, x) or None."""
    h, w = mask.shape
    H, W = g.shape
    for _ in range(tries):
        if H - h < 0 or W - w < 0:
            return None
        y, x = rng.randint(0, H - h), rng.randint(0, W - w)
        y0, x0, y1, x1 = max(0, y - gap), max(0, x - gap), min(H, y + h + gap), min(W, x + w + gap)
        if (g[y0:y1, x0:x1] == 0).all():
            g[y:y + h, x:x + w][mask] = colour
            return y, x
    return None


def components(g):
    """4-connected same-colour non-background objects: list of (colour, cells)."""
    seen = np.zeros_like(g, dtype=bool)
    out = []
    H, W = g.shape
    for y in range(H):
        for x in range(W):
            if g[y, x] == 0 or seen[y, x]:
                continue
            c, stack, cells = g[y, x], [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == c:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append((int(c), cells))
    return out


def scatter(rng, n, size=(1, 4), cols=None, lo=6, hi=14, distinct_sizes=False):
    """A grid with n separate random objects; returns (grid, [(colour, cells)])."""
    for _ in range(50):
        g = blank(rng, lo, hi)
        objs, used = [], set()
        for i in range(n):
            m = shape(rng, rng.randint(*size), rng.randint(*size))
            for _ in range(20):
                m = shape(rng, rng.randint(*size), rng.randint(*size))
                if not distinct_sizes or int(m.sum()) not in used:
                    break
            c = cols[i % len(cols)] if cols else rng.randint(1, 9)
            if place(rng, g, m, c) is None:
                break
            used.add(int(m.sum()))
        objs = components(g)
        if len(objs) == n:
            return g, objs
    return None


# ----------------------------------------------------------------------------- families
# Each family: f(rng) -> a function that makes ONE (input, output) pair with the task's fixed parameters, or None.

def fam_recolour_largest(rng):
    target = colours(rng, 1)[0]

    def pair():
        r = scatter(rng, rng.randint(2, 4), (1, 5), distinct_sizes=True)
        if r is None:
            return None
        g, objs = r
        sizes = [len(c) for _, c in objs]
        if sizes.count(max(sizes)) > 1:
            return None
        out = g.copy()
        for y, x in objs[sizes.index(max(sizes))][1]:
            out[y, x] = target
        return g, out
    return pair


def fam_keep_extreme(rng):
    keep_largest = rng.random() < 0.5

    def pair():
        r = scatter(rng, rng.randint(2, 4), (1, 5), distinct_sizes=True)
        if r is None:
            return None
        g, objs = r
        sizes = [len(c) for _, c in objs]
        pick = max(sizes) if keep_largest else min(sizes)
        if sizes.count(pick) > 1:
            return None
        out = np.zeros_like(g)
        for y, x in objs[sizes.index(pick)][1]:
            out[y, x] = g[y, x]
        return g, out
    return pair


def fam_remove_noise(rng):
    noise = colours(rng, 1)[0]

    def pair():
        r = scatter(rng, rng.randint(1, 3), (2, 5), cols=colours(rng, 3, exclude=(0, noise)))
        if r is None:
            return None
        g, _ = r
        out = g.copy()
        for _ in range(rng.randint(2, 6)):
            y, x = rng.randrange(g.shape[0]), rng.randrange(g.shape[1])
            if g[y, x] == 0:
                g[y, x] = noise
        return g, out
    return pair


def fam_recolour_by_size(rng):
    palette = colours(rng, 4)  # size 1, 2, 3, 4+

    def pair():
        r = scatter(rng, rng.randint(2, 5), (1, 3), cols=[5])
        if r is None:
            return None
        g, objs = r
        out = g.copy()
        for _, cells in objs:
            for y, x in cells:
                out[y, x] = palette[min(len(cells), 4) - 1]
        return g, out
    return pair


def fam_count_to_bar(rng):
    bar = colours(rng, 1)[0]

    def pair():
        n = rng.randint(1, 6)
        r = scatter(rng, n, (1, 3))
        if r is None:
            return None
        g, _ = r
        return g, np.full((1, n), bar)
    return pair


def fam_fill_holes(rng):
    fill = colours(rng, 1)[0]

    def pair():
        g = blank(rng)
        out_cols = colours(rng, 2, exclude=(0, fill))
        for i in range(rng.randint(1, 2)):
            h, w = rng.randint(3, 5), rng.randint(3, 5)
            m = np.ones((h, w), dtype=bool)
            if place(rng, g, m, out_cols[i]) is None:
                return None
        out = g.copy()
        for c, cells in components(g):
            ys, xs = zip(*cells)
            y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
            g[y0 + 1:y1, x0 + 1:x1] = 0
            out[y0 + 1:y1, x0 + 1:x1] = fill
        return g, out
    return pair


def fam_outline(rng):
    edge = colours(rng, 1)[0]

    def pair():
        g = blank(rng)
        c = colours(rng, 1, exclude=(0, edge))[0]
        for _ in range(rng.randint(1, 2)):
            if place(rng, g, np.ones((rng.randint(3, 5), rng.randint(3, 5)), dtype=bool), c) is None:
                return None
        out = g.copy()
        H, W = g.shape
        for y in range(H):
            for x in range(W):
                if g[y, x] and any(not (0 <= y + dy < H and 0 <= x + dx < W) or g[y + dy, x + dx] == 0
                                   for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    out[y, x] = edge
        return g, out
    return pair


def fam_remove_border_touching(rng):
    def pair():
        r = scatter(rng, rng.randint(3, 5), (1, 3))
        if r is None:
            return None
        g, objs = r
        H, W = g.shape
        out = g.copy()
        touched = 0
        for _, cells in objs:
            if any(y in (0, H - 1) or x in (0, W - 1) for y, x in cells):
                touched += 1
                for y, x in cells:
                    out[y, x] = 0
        return (g, out) if 0 < touched < len(objs) else None
    return pair


def fam_flip(rng):
    axis = rng.choice(["ud", "lr"])

    def pair():
        r = scatter(rng, rng.randint(1, 3), (1, 4))
        if r is None:
            return None
        g, _ = r
        return g, (g[::-1] if axis == "ud" else g[:, ::-1]).copy()
    return pair


def fam_rotate(rng):
    k = rng.choice([1, 2, 3])

    def pair():
        r = scatter(rng, rng.randint(1, 3), (1, 4))
        if r is None:
            return None
        g, _ = r
        return g, np.rot90(g, k).copy()
    return pair


def fam_mirror_complete(rng):
    side = rng.choice(["left", "top"])

    def pair():
        g = blank(rng)
        H, W = g.shape
        half = np.zeros_like(g)
        if side == "left":
            w = W // 2
            sub = (np.array([[rng.random() < 0.35 for _ in range(w)] for _ in range(H)]) * rng.randint(1, 9))
            half[:, :w] = sub
            full = half.copy()
            full[:, W - w:] = sub[:, ::-1]
        else:
            h = H // 2
            sub = (np.array([[rng.random() < 0.35 for _ in range(W)] for _ in range(h)]) * rng.randint(1, 9))
            half[:h] = sub
            full = half.copy()
            full[H - h:] = sub[::-1]
        return (half, full) if (half != full).any() else None
    return pair


def fam_extend_to_wall(rng):
    direction = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

    def pair():
        g = blank(rng)
        H, W = g.shape
        out = g.copy()
        pts = set()
        for _ in range(rng.randint(1, 4)):
            pts.add((rng.randrange(H), rng.randrange(W)))
        lines = {}
        for y, x in pts:
            c = rng.randint(1, 9)
            g[y, x] = c
            lines[(y, x)] = c
        out = g.copy()
        for (y, x), c in lines.items():
            ny, nx = y + direction[0], x + direction[1]
            while 0 <= ny < H and 0 <= nx < W and g[ny, nx] == 0:
                out[ny, nx] = c
                ny, nx = ny + direction[0], nx + direction[1]
        return (g, out) if (g != out).any() else None
    return pair


def fam_gravity(rng):
    k = rng.choice([0, 1, 2, 3])  # which wall, by rotation

    def pair():
        g = blank(rng)
        H, W = g.shape
        for _ in range(rng.randint(3, 10)):
            g[rng.randrange(H), rng.randrange(W)] = rng.randint(1, 9)
        a = np.rot90(g, k)
        out = np.zeros_like(a)
        for x in range(a.shape[1]):
            col = [v for v in a[:, x] if v]
            if col:
                out[-len(col):, x] = col
        out = np.rot90(out, -k)
        return (g, out.copy()) if (g != out).any() else None
    return pair


def fam_translate(rng):
    dy, dx = rng.choice([(0, 1), (0, 2), (1, 0), (2, 0), (0, -1), (-1, 0), (1, 1)])

    def pair():
        g = blank(rng)
        H, W = g.shape
        c = rng.randint(1, 9)
        m = shape(rng, rng.randint(1, 4), rng.randint(1, 4))
        if H - m.shape[0] - 2 < 2 or W - m.shape[1] - 2 < 2:
            return None
        y, x = rng.randint(2, H - m.shape[0] - 2), rng.randint(2, W - m.shape[1] - 2)
        g[y:y + m.shape[0], x:x + m.shape[1]][m] = c
        out = np.zeros_like(g)
        out[y + dy:y + dy + m.shape[0], x + dx:x + dx + m.shape[1]][m] = c
        return g, out
    return pair


def fam_scale_up(rng):
    k = rng.choice([2, 3])

    def pair():
        h, w = rng.randint(2, 4), rng.randint(2, 4)
        g = np.array([[rng.choice([0, 0, rng.randint(1, 9)]) for _ in range(w)] for _ in range(h)])
        return (g, np.kron(g, np.ones((k, k), dtype=int))) if g.any() else None
    return pair


def fam_crop(rng):
    def pair():
        g = blank(rng)
        m = shape(rng, rng.randint(2, 5), rng.randint(2, 5))
        c = rng.randint(1, 9)
        pos = place(rng, g, m, c, gap=0)
        if pos is None:
            return None
        y, x = pos
        return g, g[y:y + m.shape[0], x:x + m.shape[1]].copy()
    return pair


def fam_tile(rng):
    reps = rng.choice([(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)])

    def pair():
        h, w = rng.randint(2, 4), rng.randint(2, 4)
        g = np.array([[rng.choice([0, rng.randint(1, 9)]) for _ in range(w)] for _ in range(h)])
        return (g, np.tile(g, reps)) if g.any() else None
    return pair


def fam_connect_pairs(rng):
    line = colours(rng, 1)[0]

    def pair():
        g = blank(rng)
        H, W = g.shape
        out = None
        c = colours(rng, 1, exclude=(0, line))[0]
        rows = rng.sample(range(H), min(H, rng.randint(1, 3)))
        for y in rows:
            a, b = sorted(rng.sample(range(W), 2))
            if b - a < 2:
                continue
            g[y, a] = g[y, b] = c
        out = g.copy()
        for y in range(H):
            xs = np.nonzero(g[y] == c)[0]
            if len(xs) == 2:
                out[y, xs[0] + 1:xs[1]] = line
        return (g, out) if (g != out).any() else None
    return pair


def fam_inside_frame(rng):
    inside = colours(rng, 1)[0]

    def pair():
        g = blank(rng, 8, 14)
        H, W = g.shape
        frame = colours(rng, 1, exclude=(0, inside))[0]
        dots = colours(rng, 1, exclude=(0, inside, frame))[0]
        y0, x0 = rng.randint(0, H - 5), rng.randint(0, W - 5)
        y1, x1 = rng.randint(y0 + 4, H - 1), rng.randint(x0 + 4, W - 1)
        g[y0, x0:x1 + 1] = g[y1, x0:x1 + 1] = frame
        g[y0:y1 + 1, x0] = g[y0:y1 + 1, x1] = frame
        for _ in range(rng.randint(4, 9)):
            y, x = rng.randrange(H), rng.randrange(W)
            if g[y, x] == 0:
                g[y, x] = dots
        out = g.copy()
        sub = out[y0 + 1:y1, x0 + 1:x1]
        sub[sub == dots] = inside
        return (g, out) if (g != out).any() and ((g == dots) & (out == dots)).any() else None
    return pair


def fam_majority_colour(rng):
    def pair():
        g = blank(rng, 3, 8)
        H, W = g.shape
        cs = colours(rng, 3)
        g = np.array([[rng.choice(cs) for _ in range(W)] for _ in range(H)])
        counts = [(g == c).sum() for c in cs]
        if sorted(counts)[-1] == sorted(counts)[-2]:
            return None
        return g, np.full((1, 1), cs[int(np.argmax(counts))])
    return pair


def fam_size_parity(rng):
    even, odd = colours(rng, 2)

    def pair():
        r = scatter(rng, rng.randint(2, 5), (1, 3), cols=[5])
        if r is None:
            return None
        g, objs = r
        out = g.copy()
        for _, cells in objs:
            for y, x in cells:
                out[y, x] = even if len(cells) % 2 == 0 else odd
        return g, out
    return pair


def fam_sort_bars(rng):
    descending = rng.random() < 0.5

    def pair():
        n = rng.randint(3, 6)
        H = rng.randint(5, 10)
        heights = [rng.randint(1, H) for _ in range(n)]
        c = rng.randint(1, 9)
        g = np.zeros((H, n * 2 - 1), dtype=int)
        out = np.zeros_like(g)
        for i, h in enumerate(heights):
            g[H - h:, 2 * i] = c
        for i, h in enumerate(sorted(heights, reverse=descending)):
            out[H - h:, 2 * i] = c
        return (g, out) if (g != out).any() else None
    return pair


def fam_move_until_obstacle(rng):
    wall = colours(rng, 1)[0]
    direction = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

    def pair():
        g = blank(rng)
        H, W = g.shape
        mover = colours(rng, 1, exclude=(0, wall))[0]
        y, x = rng.randrange(1, H - 1), rng.randrange(1, W - 1)
        g[y, x] = mover
        steps = 0
        ny, nx = y, x
        while 0 <= ny + direction[0] < H and 0 <= nx + direction[1] < W:
            ny, nx = ny + direction[0], nx + direction[1]
            steps += 1
        if steps < 3:
            return None
        k = rng.randint(2, steps)
        wy, wx = y + direction[0] * k, x + direction[1] * k
        g[wy, wx] = wall
        out = g.copy()
        out[y, x] = 0
        out[wy - direction[0], wx - direction[1]] = mover
        return (g, out) if (g != out).any() else None
    return pair


def fam_ray_until_obstacle(rng):
    trail = colours(rng, 1)[0]
    direction = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

    def pair():
        r = scatter(rng, rng.randint(2, 4), (1, 2), cols=[5])
        if r is None:
            return None
        g, _ = r
        H, W = g.shape
        src = colours(rng, 1, exclude=(0, trail, 5))[0]
        for _ in range(20):
            y, x = rng.randrange(H), rng.randrange(W)
            if g[y, x] == 0:
                g[y, x] = src
                break
        else:
            return None
        out = g.copy()
        ny, nx = y + direction[0], x + direction[1]
        while 0 <= ny < H and 0 <= nx < W and g[ny, nx] == 0:
            out[ny, nx] = trail
            ny, nx = ny + direction[0], nx + direction[1]
        return (g, out) if (g != out).any() else None
    return pair


def fam_keep_same_shape(rng):
    def pair():
        g = blank(rng, 8, 14)
        ref = shape(rng, rng.randint(2, 3), rng.randint(2, 3))
        cs = colours(rng, 2)
        out = np.zeros_like(g)
        kept = 0
        for i in range(rng.randint(3, 5)):
            same = i == 0 or rng.random() < 0.4
            m = ref if same else shape(rng, rng.randint(2, 3), rng.randint(2, 3))
            if not same and m.shape == ref.shape and (m == ref).all():
                continue
            pos = place(rng, g, m, cs[0] if i == 0 else cs[1])
            if pos is None:
                return None
            if same and i > 0:
                y, x = pos
                out[y:y + m.shape[0], x:x + m.shape[1]][m] = cs[1]
                kept += 1
        # the reference (colour cs[0]) stays too
        out[g == cs[0]] = cs[0]
        return (g, out) if kept else None
    return pair


def fam_odd_one_out(rng):
    mark = colours(rng, 1)[0]

    def pair():
        g = blank(rng, 8, 14)
        common = shape(rng, rng.randint(2, 3), rng.randint(2, 3))
        odd = shape(rng, rng.randint(2, 3), rng.randint(2, 3))
        if odd.shape == common.shape and (odd == common).all():
            return None
        c = colours(rng, 1, exclude=(0, mark))[0]
        out = None
        n = rng.randint(3, 5)
        odd_i = rng.randrange(n)
        positions = []
        for i in range(n):
            m = odd if i == odd_i else common
            pos = place(rng, g, m, c)
            if pos is None:
                return None
            positions.append((pos, m))
        out = g.copy()
        (y, x), m = positions[odd_i]
        out[y:y + m.shape[0], x:x + m.shape[1]][m] = mark
        return g, out
    return pair


def fam_recolour_by_colour(rng):
    """A colour substitution read off the demonstrations (mapping changes per task)."""
    src = colours(rng, 3)
    dst = colours(rng, 3)

    def pair():
        r = scatter(rng, rng.randint(2, 4), (1, 4), cols=src)
        if r is None:
            return None
        g, _ = r
        out = g.copy()
        for a, b in zip(src, dst):
            out[g == a] = b
        return g, out
    return pair


def fam_centre_dot(rng):
    dot = colours(rng, 1)[0]

    def pair():
        h, w = rng.choice([3, 5, 7]), rng.choice([3, 5, 7])
        g = blank(rng, max(h, w) + 2, 14)
        c = colours(rng, 1, exclude=(0, dot))[0]
        pos = place(rng, g, np.ones((h, w), dtype=bool), c)
        if pos is None:
            return None
        y, x = pos
        out = g.copy()
        out[y + h // 2, x + w // 2] = dot
        return g, out
    return pair


FAMILIES = {
    # objectness
    "recolour_largest": ("objects", fam_recolour_largest),
    "keep_extreme": ("objects", fam_keep_extreme),
    "remove_noise": ("objects", fam_remove_noise),
    "fill_holes": ("objects", fam_fill_holes),
    "outline": ("objects", fam_outline),
    "remove_border_touching": ("objects", fam_remove_border_touching),
    "crop": ("objects", fam_crop),
    "recolour_by_colour": ("objects", fam_recolour_by_colour),
    # geometry and topology
    "flip": ("geometry", fam_flip),
    "rotate": ("geometry", fam_rotate),
    "mirror_complete": ("geometry", fam_mirror_complete),
    "translate": ("geometry", fam_translate),
    "scale_up": ("geometry", fam_scale_up),
    "tile": ("geometry", fam_tile),
    "inside_frame": ("geometry", fam_inside_frame),
    "centre_dot": ("geometry", fam_centre_dot),
    "connect_pairs": ("geometry", fam_connect_pairs),
    # numbers and counting
    "count_to_bar": ("number", fam_count_to_bar),
    "recolour_by_size": ("number", fam_recolour_by_size),
    "size_parity": ("number", fam_size_parity),
    "majority_colour": ("number", fam_majority_colour),
    "sort_bars": ("number", fam_sort_bars),
    # goal-directedness (things move until something stops them)
    "extend_to_wall": ("agent", fam_extend_to_wall),
    "gravity": ("agent", fam_gravity),
    "move_until_obstacle": ("agent", fam_move_until_obstacle),
    "ray_until_obstacle": ("agent", fam_ray_until_obstacle),
    # comparison
    "keep_same_shape": ("compare", fam_keep_same_shape),
    "odd_one_out": ("compare", fam_odd_one_out),
}


def task(family: str, seed) -> dict | None:
    """One task of a family: 3 demonstrations and 1 test pair sharing the task's parameters."""
    rng = random.Random(f"{family}|{seed}")
    make = FAMILIES[family][1](rng)
    pairs = []
    for _ in range(60):
        p = make()
        if p is None:
            continue
        x, y = p
        if x.shape == y.shape and (x == y).all():
            continue
        if any((x.shape == a.shape and (x == a).all()) for a, _ in pairs):
            continue
        pairs.append((x, y))
        if len(pairs) == DEMOS + 1:
            break
    if len(pairs) < DEMOS + 1:
        return None
    as_list = [{"input": x.tolist(), "output": y.tolist()} for x, y in pairs]
    return {"family": family, "train": as_list[:DEMOS], "test": as_list[DEMOS:]}


def check(per_family: int = 30, search_sample: int = 10) -> dict:
    """Step 1 gate: every family yields valid tasks; how many the current grid search satellite solves (diversity)."""
    from cognitive_lab.mk1v2.arc_search import solve

    report = {}
    for name, (group, _) in FAMILIES.items():
        made = [t for t in (task(name, s) for s in range(per_family)) if t]
        solved = 0
        for t in made[:search_sample]:
            ans = solve(t, budget=3.0)
            solved += any(o[0] == t["test"][0]["output"] for _, o in ans)
        sizes = [len(t["train"][0]["input"]) for t in made]
        report[name] = {"group": group, "valid": f"{len(made)}/{per_family}",
                        "search_solves": f"{solved}/{min(search_sample, len(made))}",
                        "rows": f"{min(sizes) if sizes else 0}-{max(sizes) if sizes else 0}"}
        print(f"{name:24s} {group:9s} valid {report[name]['valid']:6s} search {report[name]['search_solves']:6s} rows {report[name]['rows']}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()


if __name__ == "__main__":
    main()
