"""Alphabet letter shapes for the drone-formation environment (v5: free-size).

Each letter is drawn as ASCII art on a 7-row x 5-col canvas, in a classic
5x7 bitmap-font style. Unlike v4 (every letter exactly 12 cells), here EACH
LETTER USES ITS NATURAL CELL COUNT (range MIN_CELLS .. MAX_CELLS).

The drone-formation environment uses a fixed fleet of MAX_CELLS drones; for
any letter, the remaining (MAX_CELLS - letter cells) drones go to padding
'wait' cells along the grid edge so a single network can handle all letters.

Helpers
-------
- letter_cells(name)      -> sorted [(row, col), ...] for the letter glyph
- letter_bbox(name)       -> (height, width)
- letter_cell_count(name) -> int
- render_letter(name)     -> ASCII string (for debugging)

Run this file directly to see all 26 letters + their cell counts:
    python letters.py
"""
from __future__ import annotations

# Bitmap-font-style ASCII art on a 7-row x 5-col canvas.
# Letters use their NATURAL cell count -- no padding to a fixed number here.
_LETTER_ART: dict[str, str] = {
    # A: 5x7, ~13 cells
    "A": """
.###.
#...#
#...#
#####
#...#
#...#
#...#
""",
    "B": """
####.
#...#
#...#
####.
#...#
#...#
####.
""",
    "C": """
.####
#....
#....
#....
#....
#....
.####
""",
    "D": """
####.
#...#
#...#
#...#
#...#
#...#
####.
""",
    "E": """
#####
#....
#....
####.
#....
#....
#####
""",
    "F": """
#####
#....
#....
####.
#....
#....
#....
""",
    "G": """
.####
#....
#....
#..##
#...#
#...#
.####
""",
    "H": """
#...#
#...#
#...#
#####
#...#
#...#
#...#
""",
    "I": """
#####
..#..
..#..
..#..
..#..
..#..
#####
""",
    "J": """
#####
....#
....#
....#
....#
#...#
.###.
""",
    "K": """
#...#
#..#.
#.#..
##...
#.#..
#..#.
#...#
""",
    "L": """
#....
#....
#....
#....
#....
#....
#####
""",
    "M": """
#...#
##.##
#.#.#
#.#.#
#...#
#...#
#...#
""",
    "N": """
#...#
##..#
#.#.#
#.#.#
#.#.#
#..##
#...#
""",
    "O": """
.###.
#...#
#...#
#...#
#...#
#...#
.###.
""",
    "P": """
####.
#...#
#...#
####.
#....
#....
#....
""",
    "Q": """
.###.
#...#
#...#
#...#
#.#.#
#..#.
.##.#
""",
    "R": """
####.
#...#
#...#
####.
#.#..
#..#.
#...#
""",
    "S": """
.####
#....
#....
.###.
....#
....#
####.
""",
    "T": """
#####
..#..
..#..
..#..
..#..
..#..
..#..
""",
    "U": """
#...#
#...#
#...#
#...#
#...#
#...#
.###.
""",
    "V": """
#...#
#...#
#...#
#...#
#...#
.#.#.
..#..
""",
    "W": """
#...#
#...#
#...#
#.#.#
#.#.#
##.##
#...#
""",
    "X": """
#...#
#...#
.#.#.
..#..
.#.#.
#...#
#...#
""",
    "Y": """
#...#
#...#
.#.#.
..#..
..#..
..#..
..#..
""",
    "Z": """
#####
....#
...#.
..#..
.#...
#....
#####
""",
}


def _parse_art(art: str) -> list[tuple[int, int]]:
    """Parse ASCII-art block into sorted (row, col) coords of '#' cells."""
    cells: list[tuple[int, int]] = []
    rows = [line for line in art.splitlines() if line != ""]
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch == "#":
                cells.append((r, c))
    return sorted(cells)


# Public: name -> list of (row, col) cells (RELATIVE, top-left = (0,0))
LETTERS: dict[str, list[tuple[int, int]]] = {
    name: _parse_art(art) for name, art in _LETTER_ART.items()
}

# Derived constants used by the environment
_COUNTS = [len(cells) for cells in LETTERS.values()]
MIN_CELLS = min(_COUNTS)
MAX_CELLS = max(_COUNTS)  # fleet size = MAX_CELLS drones


def letter_cells(name: str) -> list[tuple[int, int]]:
    """Return the (row, col) cells of a letter glyph (relative coords)."""
    return list(LETTERS[name])


def letter_cell_count(name: str) -> int:
    return len(LETTERS[name])


def letter_bbox(name: str) -> tuple[int, int]:
    """Return (height, width) bounding box of a letter."""
    cells = LETTERS[name]
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    return max_r + 1, max_c + 1


def render_letter(name: str) -> str:
    """Return an ASCII rendering of a letter (for visual debugging)."""
    cells = set(LETTERS[name])
    h, w = letter_bbox(name)
    lines = []
    for r in range(h):
        lines.append("".join("#" if (r, c) in cells else "." for c in range(w)))
    return "\n".join(lines)


def validate_letters(min_cells: int = 8, max_cells: int = 25) -> None:
    """Raise if any letter is wildly out of expected range. Lenient by default,
    only catching obvious typos in the art."""
    bad = []
    for name, cells in LETTERS.items():
        n = len(cells)
        if n < min_cells or n > max_cells:
            bad.append((name, n))
    if bad:
        msg = ", ".join(f"{n}={c}" for n, c in bad)
        raise ValueError(
            f"Letters with cell count outside [{min_cells}, {max_cells}]: {msg}"
        )


# Always validate at import time so typos in the art surface early.
validate_letters()


if __name__ == "__main__":
    print(f"Total letters: {len(LETTERS)}")
    print(f"Cell counts: min={MIN_CELLS}, max={MAX_CELLS}")
    print(f"Drone fleet size (= MAX_CELLS): {MAX_CELLS}\n")

    # Sort by cell count to see range visually
    by_count = sorted(LETTERS.items(), key=lambda kv: len(kv[1]))
    for name, cells in by_count:
        n = len(cells)
        h, w = letter_bbox(name)
        print(f"'{name}'  cells={n:>2}  bbox={h}x{w}  (will pad with {MAX_CELLS - n} wait cells)")
        print(render_letter(name))
        print()
