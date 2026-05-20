"""Formation definitions and ordered shape paths for drone-show transitions.

Naming note
-----------
This file intentionally keeps the user-facing name ``shapes`` instead of
``sequence``. A comma-separated shapes argument such as ``GROUND,X,I,-`` means:

    start at GROUND -> move to X -> move to I -> move to -

If the first name is not ``GROUND``, ``GROUND`` is automatically prepended.
For example, ``X,I,-`` is interpreted as ``GROUND,X,I,-``.

Default scale
-------------
The current default bank is for 20 drones on a 25x25 grid. The GROUND formation
is generated dynamically from ``grid_size`` and ``n_agents``.
"""
from __future__ import annotations

from dataclasses import dataclass

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


# 20-cell target shapes on a 25x25 grid, centered around row/col 12.
# Every non-GROUND formation must have exactly 20 cells.
SHAPES_20: dict[str, list[GridPos]] = {
    "I": [(r, 12) for r in range(3, 23)],

    "-": [(12, c) for c in range(3, 23)],

    "+": [
        # vertical arm: 11 cells
        (6, 12), (7, 12), (8, 12), (9, 12), (10, 12),
        (11, 12), (12, 12), (13, 12), (14, 12), (15, 12), (16, 12),
        # horizontal arm: 10 cells, center (12, 12) already included
        (12, 8), (12, 9), (12, 10), (12, 11),
        (12, 13), (12, 14), (12, 15), (12, 16), (12, 17),
    ],

    "X": [
        # left-top to right-bottom diagonal
        (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
        (10, 10), (11, 11), (12, 12), (13, 13), (14, 14),
        # right-top to left-bottom diagonal, shifted to avoid duplicate center
        (5, 18), (6, 17), (7, 16), (8, 15), (9, 14),
        (10, 13), (11, 12), (12, 11), (13, 10), (14, 9),
    ],

    "T": [
        (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12),
        (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18),
        (6, 12), (7, 12), (8, 12), (9, 12),
        (10, 12), (11, 12), (12, 12), (13, 12),
    ],

    "L": [
        (5, 6), (6, 6), (7, 6), (8, 6), (9, 6),
        (10, 6), (11, 6), (12, 6), (13, 6), (14, 6),
        (15, 6), (16, 6), (17, 6), (18, 6),
        (18, 7), (18, 8), (18, 9), (18, 10), (18, 11), (18, 12),
    ],

    "O": [
        (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14),
        (16, 9), (16, 10), (16, 11), (16, 12), (16, 13), (16, 14),
        (8, 9), (10, 9), (12, 9), (14, 9),
        (8, 14), (10, 14), (12, 14), (14, 14),
    ],

    "V": [
        (5, 5), (6, 5), (7, 6), (8, 6), (9, 7),
        (10, 8), (11, 9), (12, 10), (13, 11), (15, 12),
        (5, 19), (6, 19), (7, 18), (8, 18), (9, 17),
        (10, 16), (11, 15), (12, 14), (13, 13), (15, 13),
    ],

    "E": [
        (5, 6), (6, 6), (7, 6), (8, 6),
        (9, 6), (10, 6), (11, 6), (12, 6),
        (5, 7), (5, 8), (5, 9), (5, 10),
        (9, 7), (9, 8), (9, 9), (9, 10),
        (12, 7), (12, 8), (12, 9), (12, 10),
    ],

    "A": [
        (16, 5), (15, 6), (14, 7), (13, 8), (12, 9),
        (11, 10), (10, 11), (9, 12),
        (10, 13), (11, 14), (12, 15), (13, 16),
        (14, 17), (15, 18), (16, 19),
        (13, 9), (13, 10), (13, 11), (13, 12), (13, 13),
    ],

    "H": [
        (5, 6), (6, 6), (7, 6), (8, 6), (9, 6),
        (10, 6), (11, 6), (12, 6),
        (5, 16), (6, 16), (7, 16), (8, 16), (9, 16),
        (10, 16), (11, 16), (12, 16),
        (9, 7), (9, 8), (9, 9), (9, 10),
    ],

    "N": [
        (5, 6), (6, 6), (7, 6), (8, 6), (9, 6),
        (10, 6), (11, 6), (12, 6),
        (5, 17), (6, 17), (7, 17), (8, 17), (9, 17),
        (10, 17), (11, 17), (12, 17),
        (7, 8), (8, 10), (9, 12), (10, 14),
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

# Compatibility alias: current project uses SHAPES as the active bank.
SHAPES: dict[str, list[GridPos]] = SHAPES_20

# LOVE aliases keep old command examples readable.
SHAPES["LOVE_L"] = list(SHAPES["L"])
SHAPES["LOVE_O"] = list(SHAPES["O"])
SHAPES["LOVE_V"] = list(SHAPES["V"])
SHAPES["LOVE_E"] = list(SHAPES["E"])


def _validate_shape_bank() -> None:
    bad = {name: len(cells) for name, cells in SHAPES.items() if len(set(cells)) != 20 or len(cells) != 20}
    if bad:
        raise ValueError(f"Every 20-drone shape must have 20 unique cells. Bad shapes: {bad}")


_validate_shape_bank()


def available_shape_names(include_ground: bool = True) -> list[str]:
    names = list(SHAPES)
    return (["GROUND"] + names) if include_ground else names


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
            "Use the 20-drone defaults or define a matching formation bank."
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
