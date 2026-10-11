"""Eight new concept families for the family-count probe (design/next-after-c-2026-10-11.md section 6).

Written after the 28 families of generator.py were all used for pretraining (learner v2), so none of the probe models
has seen these. They come from the same core-knowledge list (objects, geometry, numbers, goal-directedness), not from
any benchmark's tasks; ConceptARC stays sealed.
"""

import numpy as np

from cognitive_lab.concepts.generator import blank, colours, components, place, scatter, shape


def fam_swap_colours(rng):
    """The two colours in the grid trade places."""
    def pair():
        a, b = colours(rng, 2)
        r = scatter(rng, rng.randint(3, 6), (1, 4), cols=[a, b])
        if r is None:
            return None
        g, _ = r
        if not ((g == a).any() and (g == b).any()):
            return None
        out = g.copy()
        out[g == a], out[g == b] = b, a
        return g, out
    return pair


def fam_draw_diagonal(rng):
    """Every dot shoots a diagonal line (the task's direction) to the border; rays never cross."""
    dy, dx = rng.choice([(1, 1), (1, -1), (-1, 1), (-1, -1)])

    def pair():
        g = blank(rng)
        H, W = g.shape
        out = g.copy()
        owner = np.zeros_like(g)
        for i in range(rng.randint(1, 3)):
            y, x, c = rng.randrange(H), rng.randrange(W), rng.randint(1, 9)
            cells = []
            while 0 <= y < H and 0 <= x < W:
                cells.append((y, x))
                y, x = y + dy, x + dx
            if len(cells) < 3 or any(owner[p] for p in cells):
                return None
            for p in cells:
                owner[p] = i + 1
                out[p] = c
            g[cells[0]] = c
        return g, out
    return pair


def fam_mark_corners(rng):
    """The four corners of every rectangle take the task's marker colour."""
    k = colours(rng, 1)[0]

    def pair():
        g = blank(rng)
        for _ in range(rng.randint(1, 3)):
            c = rng.choice([x for x in range(1, 10) if x != k])
            if place(rng, g, np.ones((rng.randint(2, 5), rng.randint(2, 5)), dtype=bool), c) is None:
                return None
        out = g.copy()
        for _, cells in components(g):
            ys, xs = [a for a, _ in cells], [b for _, b in cells]
            for y in (min(ys), max(ys)):
                for x in (min(xs), max(xs)):
                    out[y, x] = k
        return g, out
    return pair


def fam_cross_from_dot(rng):
    """Every dot fills its whole row, then every dot fills its whole column (columns on top)."""
    def pair():
        g = blank(rng)
        H, W = g.shape
        pts = []
        for _ in range(rng.randint(1, 2)):
            y, x = rng.randrange(H), rng.randrange(W)
            if any(y == a or x == b for a, b, _ in pts):
                return None
            pts.append((y, x, rng.randint(1, 9)))
        out = g.copy()
        for y, x, c in pts:
            g[y, x] = c
            out[y, :] = c
        for y, x, c in pts:
            out[:, x] = c
        return g, out
    return pair


def fam_copy_shape_to_markers(rng):
    """One shape and some single-cell markers: the shape is stamped with each marker as its top-left corner."""
    def pair():
        g = blank(rng, 8, 14)
        a, b = colours(rng, 2)
        m = shape(rng, rng.randint(2, 3), rng.randint(2, 3))
        if place(rng, g, m, a) is None:
            return None
        h, w = m.shape
        box = np.zeros((h, w), dtype=bool)
        box[0, 0] = True
        spots = []
        for _ in range(rng.randint(1, 3)):
            pos = place(rng, g, box, b)
            if pos is None:
                return None
            spots.append(pos)
        out = g.copy()
        for y, x in spots:
            out[y, x] = 0
            out[y:y + h, x:x + w][m] = a
        return g, out
    return pair


def fam_bounding_box(rng):
    """Every object gets a one-cell frame (the task's colour) around its bounding box."""
    k = colours(rng, 1)[0]

    def pair():
        g = blank(rng)
        boxes = []
        for _ in range(rng.randint(1, 2)):
            m = shape(rng, rng.randint(2, 4), rng.randint(2, 4))
            h, w = m.shape
            big = np.zeros((h + 2, w + 2), dtype=bool)
            big[1:-1, 1:-1] = m
            pos = place(rng, g, big, rng.choice([x for x in range(1, 10) if x != k]))
            if pos is None:
                return None
            boxes.append((pos, h, w))
        out = g.copy()
        for (y, x), h, w in boxes:
            out[y, x:x + w + 2] = k
            out[y + h + 1, x:x + w + 2] = k
            out[y:y + h + 2, x] = k
            out[y:y + h + 2, x + w + 1] = k
        return g, out
    return pair


def fam_invert_colours(rng):
    """Figure and ground swap: background cells take the object colour, object cells become background."""
    def pair():
        c = rng.randint(1, 9)
        r = scatter(rng, rng.randint(2, 4), (2, 4), cols=[c])
        if r is None:
            return None
        g, _ = r
        return g, np.where(g == 0, c, 0)
    return pair


def fam_pattern_continue(rng):
    """The first rows (or columns) show a repeating block; the output repeats it to fill the grid."""
    vertical = rng.random() < 0.5

    def pair():
        H, W, p = rng.randint(6, 14), rng.randint(6, 14), rng.randint(2, 3)
        pat = np.array([[rng.choice([0, 0, rng.randint(1, 9)]) for _ in range(W)] for _ in range(p)])
        if not pat.any() or (pat == 0).all(axis=1).any():
            return None
        full = np.tile(pat, (H // p + 1, 1))[:H]
        shown = p * rng.randint(1, 2)
        if shown >= H - 1:
            return None
        g = np.zeros((H, W), dtype=int)
        g[:shown] = full[:shown]
        return (g, full) if vertical else (g.T.copy(), full.T.copy())
    return pair


PROBE = {
    "swap_colours": ("objects", fam_swap_colours),
    "mark_corners": ("objects", fam_mark_corners),
    "copy_shape_to_markers": ("objects", fam_copy_shape_to_markers),
    "bounding_box": ("geometry", fam_bounding_box),
    "invert_colours": ("geometry", fam_invert_colours),
    "pattern_continue": ("number", fam_pattern_continue),
    "draw_diagonal": ("agent", fam_draw_diagonal),
    "cross_from_dot": ("agent", fam_cross_from_dot),
}
