"""Bitmap-based formation definitions and ordered shape paths for drone-show transitions.

This version keeps the project API unchanged:
    formation_seq.py -> comm_env.py -> comm_train.py / comm_eval.py

User-facing naming remains ``shapes``.
A comma-separated shapes argument such as ``GROUND,A,B,C`` means:
    start at GROUND -> move to A -> move to B -> move to C

Default scale:
    20 drones on a 25x25 grid.

Alphabet formations A-Z are defined as 5x7 bitmaps in LETTER_BITMAPS.
Each bitmap is converted into exactly n_agents target cells by:
    1. reading lit bitmap pixels,
    2. centering them on the grid,
    3. scaling bitmap pixels into grid coordinates,
    4. deterministically trimming or expanding to exactly n_agents cells.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import matplotlib.pyplot as plt
import math
GridPos = tuple[int, int]


@dataclass(frozen=True)
class Formation:
    """A named formation made of one grid cell per drone."""

    name: str
    cells: list[GridPos]


@dataclass(frozen=True)
class FormationPath:
    """Ordered formation path: start formation + target formations."""

    start: Formation
    targets: list[Formation]

    @property
    def names(self) -> list[str]:
        return [self.start.name] + [formation.name for formation in self.targets]

    @property
    def label(self) -> str:
        return "->".join(self.names)

    def __len__(self) -> int:
        return len(self.targets)


# 5x7 pixel-font alphabet. "1" means a lit pixel / target candidate.
LETTER_BITMAPS: dict[str, list[str]] = {
    "A": [
        "01110",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],
    "B": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10001",
        "10001",
        "11110",
    ],
    "C": [
        "01111",
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "01111",
    ],
    "D": [
        "11110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "11110",
    ],
    "E": [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "11111",
    ],
    "F": [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "10000",
    ],
    "G": [
        "01111",
        "10000",
        "10000",
        "10111",
        "10001",
        "10001",
        "01111",
    ],
    "H": [
        "10001",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],
    "I": [
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "11111",
    ],
    "J": [
        "00111",
        "00010",
        "00010",
        "00010",
        "00010",
        "10010",
        "01100",
    ],
    "K": [
        "10001",
        "10010",
        "10100",
        "11000",
        "10100",
        "10010",
        "10001",
    ],
    "L": [
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "11111",
    ],
    "M": [
        "10001",
        "11011",
        "10101",
        "10101",
        "10001",
        "10001",
        "10001",
    ],
    "N": [
        "10001",
        "11001",
        "10101",
        "10011",
        "10001",
        "10001",
        "10001",
    ],
    "O": [
        "01110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "01110",
    ],
    "P": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10000",
        "10000",
        "10000",
    ],
    "Q": [
        "01110",
        "10001",
        "10001",
        "10001",
        "10101",
        "10010",
        "01101",
    ],
    "R": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10100",
        "10010",
        "10001",
    ],
    "S": [
        "01111",
        "10000",
        "10000",
        "01110",
        "00001",
        "00001",
        "11110",
    ],
    "T": [
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
    ],
    "U": [
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "01110",
    ],
    "V": [
        "10001",
        "10001",
        "10001",
        "10001",
        "01010",
        "01010",
        "00100",
    ],
    "W": [
        "10001",
        "10001",
        "10001",
        "10101",
        "10101",
        "10101",
        "01010",
    ],
    "X": [
        "10001",
        "01010",
        "01010",
        "00100",
        "01010",
        "01010",
        "10001",
    ],
    "Y": [
        "10001",
        "01010",
        "01010",
        "00100",
        "00100",
        "00100",
        "00100",
    ],
    "Z": [
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "10000",
        "11111",
    ],
}


# Extra non-alphabet formations. These are kept as explicit 20-cell shapes because
# they are not naturally represented as 5x7 letters.
EXTRA_SHAPES_20: dict[str, list[GridPos]] = {
    "-": [(12, c) for c in range(3, 23)],
    "PLUS": [
        (6, 12), (7, 12), (8, 12), (9, 12), (10, 12),
        (11, 12), (12, 12), (13, 12), (14, 12), (15, 12), (16, 12),
        (12, 8), (12, 9), (12, 10), (12, 11),
        (12, 13), (12, 14), (12, 15), (12, 16), (12, 17),
    ],
    "+": [
        (6, 12), (7, 12), (8, 12), (9, 12), (10, 12),
        (11, 12), (12, 12), (13, 12), (14, 12), (15, 12), (16, 12),
        (12, 8), (12, 9), (12, 10), (12, 11),
        (12, 13), (12, 14), (12, 15), (12, 16), (12, 17),
    ],
    "CROSS": [
        (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
        (10, 10), (11, 11), (12, 12), (13, 13), (14, 14),
        (5, 18), (6, 17), (7, 16), (8, 15), (9, 14),
        (10, 13), (11, 12), (12, 11), (13, 10), (14, 9),
    ],
    "DIAMOND": [
        (4, 12),
        (5, 11), (5, 13),
        (6, 10), (6, 14),
        (7, 9), (7, 15),
        (8, 8), (8, 16),
        (9, 9), (9, 15),
        (10, 10), (10, 14),
        (11, 11), (11, 13),
        (12, 12),
        (13, 11), (13, 13),
        (14, 10), (14, 14),
    ],
    "ARROW_UP": [
        (4, 12),
        (5, 11), (5, 12), (5, 13),
        (6, 10), (6, 11), (6, 12), (6, 13), (6, 14),
        (7, 12), (8, 12), (9, 12), (10, 12), (11, 12),
        (12, 12), (13, 12), (14, 12), (15, 12), (16, 12), (17, 12),
    ],
    "SQUARE": [
        (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12),
        (14, 7), (14, 8), (14, 9), (14, 10), (14, 11), (14, 12),
        (7, 7), (8, 7), (9, 7), (10, 7),
        (7, 12), (8, 12), (9, 12), (10, 12),
    ],
}


def _lit_bitmap_cells(bitmap: list[str]) -> list[tuple[int, int]]:
    """Return lit bitmap cells as (bitmap_row, bitmap_col)."""
    cells: list[tuple[int, int]] = []
    for row, line in enumerate(bitmap):
        for col, value in enumerate(line):
            if value not in {"0", ".", " "}:
                cells.append((row, col))
    return cells


def _centered_grid_cell(
    bitmap_row: int,
    bitmap_col: int,
    bitmap_height: int,
    bitmap_width: int,
    grid_size: int,
    scale: int,
) -> GridPos:
    """Map one bitmap cell to one centered grid coordinate."""
    shape_height = (bitmap_height - 1) * scale + 1
    shape_width = (bitmap_width - 1) * scale + 1
    top = (grid_size - shape_height) // 2
    left = (grid_size - shape_width) // 2
    return (top + bitmap_row * scale, left + bitmap_col * scale)


def _rank_cells_for_trim(cells: list[GridPos]) -> list[GridPos]:
    """Deterministically rank cells when a bitmap produces too many candidates.

    Preference keeps visually important points: endpoints and spread-out boundary
    cells tend to appear early because they have fewer lit neighbors in the
    candidate set. Ties are resolved by distance from center and then row/col.
    """
    cell_set = set(cells)
    center_r = sum(r for r, _ in cells) / len(cells)
    center_c = sum(c for _, c in cells) / len(cells)

    def score(cell: GridPos) -> tuple[int, float, int, int]:
        r, c = cell
        neighbors = sum(
            (r + dr, c + dc) in cell_set
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, 0), (1, 0), (0, -1), (0, 1)]
        )
        dist_center = abs(r - center_r) + abs(c - center_c)
        return (neighbors, -dist_center, r, c)

    # Low neighbor count first keeps stroke endpoints/corners; reverse distance
    # favors preserving outer silhouette if trimming is needed.
    return sorted(cells, key=score)


def _expand_to_n_cells(cells: list[GridPos], n_cells: int, grid_size: int) -> list[GridPos]:
    """Add nearby cells until exactly n_cells unique grid cells exist."""
    result: list[GridPos] = []
    seen: set[GridPos] = set()
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            result.append(cell)

    if len(result) >= n_cells:
        return result[:n_cells]

    queue = deque(result)
    directions = [
        (0, 1), (1, 0), (0, -1), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]
    while queue and len(result) < n_cells:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in seen:
                seen.add((nr, nc))
                result.append((nr, nc))
                queue.append((nr, nc))
                if len(result) == n_cells:
                    break

    if len(result) != n_cells:
        raise ValueError(f"Could not expand formation to {n_cells} cells")
    return result


def bitmap_to_n_cells(
    bitmap: list[str],
    n_cells: int = 20,
    grid_size: int = 25,
    scale: int = 2,
) -> list[GridPos]:
    """Convert a 5x7 bitmap into exactly n_cells centered grid coordinates."""
    if not bitmap:
        raise ValueError("bitmap must not be empty")
    width = len(bitmap[0])
    if any(len(line) != width for line in bitmap):
        raise ValueError(f"bitmap rows must have equal width: {bitmap}")

    lit = _lit_bitmap_cells(bitmap)
    if not lit:
        raise ValueError("bitmap has no lit cells")

    raw_cells = [
        _centered_grid_cell(
            bitmap_row=row,
            bitmap_col=col,
            bitmap_height=len(bitmap),
            bitmap_width=width,
            grid_size=grid_size,
            scale=scale,
        )
        for row, col in lit
    ]

    # Remove duplicates while preserving bitmap scan order.
    unique_cells = list(dict.fromkeys(raw_cells))

    if len(unique_cells) > n_cells:
        ranked = _rank_cells_for_trim(unique_cells)
        selected = ranked[:n_cells]
        # Return in stable row-major order for deterministic observation order.
        return sorted(selected)

    if len(unique_cells) < n_cells:
        return sorted(_expand_to_n_cells(unique_cells, n_cells, grid_size))

    return sorted(unique_cells)


def build_letter_shapes(
    n_cells: int = 20,
    grid_size: int = 25,
    scale: int = 2,
) -> dict[str, list[GridPos]]:
    """Build A-Z grid-coordinate shapes from LETTER_BITMAPS."""
    return {
        letter: bitmap_to_n_cells(bitmap, n_cells=n_cells, grid_size=grid_size, scale=scale)
        for letter, bitmap in LETTER_BITMAPS.items()
    }


# This is where bitmap letters are converted into actual target coordinates and
# registered as shapes.
SHAPES: dict[str, list[GridPos]] = build_letter_shapes(n_cells=20, grid_size=25, scale=2)
SHAPES.update(EXTRA_SHAPES_20)

# LOVE aliases keep old command examples readable.
SHAPES["LOVE_L"] = list(SHAPES["L"])
SHAPES["LOVE_O"] = list(SHAPES["O"])
SHAPES["LOVE_V"] = list(SHAPES["V"])
SHAPES["LOVE_E"] = list(SHAPES["E"])


def _validate_shape_bank(expected_size: int = 20) -> None:
    bad = {
        name: len(cells)
        for name, cells in SHAPES.items()
        if len(set(cells)) != expected_size or len(cells) != expected_size
    }
    if bad:
        raise ValueError(
            f"Every non-GROUND shape must have {expected_size} unique cells. Bad shapes: {bad}"
        )


_validate_shape_bank(expected_size=20)


def available_shape_names(include_ground: bool = True) -> list[str]:
    names = list(SHAPES)
    return (["GROUND"] + names) if include_ground else names


def render_shape_ascii(name: str, grid_size: int = 25) -> str:
    """Render a shape as ASCII for quick debugging in Colab."""
    if name not in SHAPES:
        raise ValueError(f"Unknown shape {name!r}. Available: {available_shape_names()}")
    grid = [["."] * grid_size for _ in range(grid_size)]
    for row, col in SHAPES[name]:
        if 0 <= row < grid_size and 0 <= col < grid_size:
            grid[row][col] = "#"
    return "\n".join("".join(row) for row in grid)


def make_ground_line(
    grid_size: int,
    n_agents: int,
    row: int | None = None,
    start_col: int | None = None,
    name: str = "GROUND",
) -> Formation:
    """Create the bottom-row ground formation: (grid_size - 1, x)."""
    ground_row = grid_size - 1 if row is None else row
    first_col = (grid_size - n_agents) // 2 if start_col is None else start_col
    last_col = first_col + n_agents - 1

    if not (0 <= ground_row < grid_size):
        raise ValueError(
            f"ground row must be in [0, {grid_size - 1}], got {ground_row}"
        )
    if first_col < 0 or last_col >= grid_size:
        raise ValueError(
            "ground line is outside the grid: "
            f"start_col={first_col}, last_col={last_col}, grid_size={grid_size}"
        )

    return Formation(
        name=name,
        cells=[(ground_row, col) for col in range(first_col, first_col + n_agents)],
    )


def get_shape(
    name: str,
    grid_size: int,
    n_agents: int,
    ground_row: int | None = None,
    ground_start_col: int | None = None,
) -> Formation:
    """Return a named formation."""
    if name == "GROUND":
        return make_ground_line(
            grid_size=grid_size,
            n_agents=n_agents,
            row=ground_row,
            start_col=ground_start_col,
        )

    if name not in SHAPES:
        raise ValueError(
            f"Unknown shape {name!r}. Available: {available_shape_names()}"
        )

    cells = list(SHAPES[name])
    if len(cells) != n_agents:
        raise ValueError(
            f"Shape {name!r} has {len(cells)} cells, but n_agents={n_agents}. "
            "The current bitmap bank is generated for 20 drones."
        )

    for row, col in cells:
        if not (0 <= row < grid_size and 0 <= col < grid_size):
            raise ValueError(
                f"Shape {name!r} contains cell {(row, col)} outside {grid_size}x{grid_size}."
            )

    return Formation(name=name, cells=cells)


def normalize_shape_names(names: list[str]) -> list[str]:
    """Ensure the path starts from GROUND."""
    clean = [name.strip() for name in names if name.strip()]
    if not clean:
        clean = ["GROUND", "X"]
    if clean[0] != "GROUND":
        clean = ["GROUND"] + clean
    if len(clean) < 2:
        raise ValueError("shapes must contain at least a start shape and one target")
    return clean


def build_shapes(
    names: list[str],
    grid_size: int,
    n_agents: int,
    ground_row: int | None = None,
    ground_start_col: int | None = None,
) -> FormationPath:
    """Build an ordered formation path from shape names."""
    normalized = normalize_shape_names(names)
    formations = [
        get_shape(
            name=name,
            grid_size=grid_size,
            n_agents=n_agents,
            ground_row=ground_row,
            ground_start_col=ground_start_col,
        )
        for name in normalized
    ]
    return FormationPath(start=formations[0], targets=formations[1:])


def parse_shapes_arg(
    shapes: str,
    grid_size: int,
    n_agents: int,
    ground_row: int | None = None,
    ground_start_col: int | None = None,
) -> FormationPath:
    names = [name.strip() for name in shapes.split(",") if name.strip()]
    return build_shapes(
        names=names,
        grid_size=grid_size,
        n_agents=n_agents,
        ground_row=ground_row,
        ground_start_col=ground_start_col,
    )

def plot_all_letters(grid_size: int = 25):
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    n_cols = 4
    n_rows = math.ceil(len(letters) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for ax, name in zip(axes, letters):
        coords = SHAPES[name]

        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linewidth=0.3)
        ax.set_aspect("equal")
        ax.set_title(f"{name} ({len(coords)})")

        ax.axhline((grid_size - 1) / 2, linestyle="--", linewidth=0.8)
        ax.axvline((grid_size - 1) / 2, linestyle="--", linewidth=0.8)

        for r, c in coords:
            ax.scatter(c, r, s=80)

    for ax in axes[len(letters):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


plot_all_letters()