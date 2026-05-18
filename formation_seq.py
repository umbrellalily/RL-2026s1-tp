"""Formation definitions and ordered shape paths for drone-show transitions.

Naming note
-----------
This file intentionally keeps the user-facing name ``shapes`` instead of
``sequence``. A comma-separated shapes argument such as ``GROUND,X,I,-`` means:

    start at GROUND -> move to X -> move to I -> move to -

If the first name is not ``GROUND``, ``GROUND`` is automatically prepended.
For example, ``X,I,-`` is interpreted as ``GROUND,X,I,-``.
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


# Ten-cell target shapes on a 15x15 grid, centered around (7, 7).
# GROUND is generated dynamically by make_ground_line because it depends on
# grid_size and n_agents.
SHAPES: dict[str, list[GridPos]] = {
    "I": [
        (3, 7), (4, 7), (5, 7), (6, 7), (7, 7),
        (8, 7), (9, 7), (10, 7), (11, 7), (12, 7),
    ],
    "-": [
        (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
        (7, 8), (7, 9), (7, 10), (7, 11), (7, 12),
    ],
    "L": [
        (3, 5), (4, 5), (5, 5), (6, 5), (7, 5),
        (8, 5), (9, 5), (9, 6), (9, 7), (9, 8),
    ],
    "T": [
        (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
        (5, 7), (6, 7), (7, 7), (8, 7), (9, 7),
    ],
    "+": [
        (4, 7), (5, 7), (6, 7),
        (7, 5), (7, 6), (7, 7), (7, 8), (7, 9),
        (8, 7), (9, 7),
    ],
    "X": [
        (4, 4), (5, 5), (5, 9), (6, 6), (6, 8),
        (7, 7), (8, 6), (8, 8), (9, 5), (9, 9),
    ],

    "O": [
        (4, 5), (4, 6), (4, 7),
        (5, 5),         (5, 8),
        (6, 5),         (6, 8),
        (7, 5), (7, 6), (7, 7),
    ],

    "V": [
        (3, 4),
        (4, 4),
        (5, 5),
        (6, 5),
        (7, 6),
        (8, 7),
        (7, 8),
        (6, 9),
        (5, 9),
        (4, 10),
    ],

    "E": [
        (3, 5), (3, 6), (3, 7),
        (4, 5),
        (5, 5), (5, 6),
        (6, 5),
        (7, 5), (7, 6), (7 ,7),
    ],
}




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
            f"Shape {name!r} has {len(cells)} cells, but n_agents={n_agents}"
        )
    return Formation(name=name, cells=cells)


def normalize_shape_names(names: list[str]) -> list[str]:
    """Ensure the path starts from GROUND.

    ``['X', 'I']`` becomes ``['GROUND', 'X', 'I']``.
    ``['GROUND', 'X']`` is left unchanged.
    """
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
 names = [s.strip() for s in sequence.split(",") if s.strip()]
    return build_sequence(names=names, grid_size=grid_size, n_agents=n_agents)