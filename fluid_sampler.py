#!/usr/bin/env python3
"""Sample a toy 2D pipe-flow velocity field with THRML.

The model is intentionally discrete and hackathon-sized:
- each grid cell is a categorical velocity (u, v),
- u and v each take values in {-3, ..., 3},
- obstacle/wall/inlet constraints are unary factors,
- local flow consistency is represented by pairwise factors.

This keeps the factor tensors small enough for a laptop while still making the
energy derive from recognizable finite-difference flow constraints.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

_mpl_config_dir = Path("fluid_flow_thrml/.matplotlib").resolve()
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from thrml import (
    Block,
    BlockGibbsSpec,
    CategoricalNode,
    FactorSamplingProgram,
    SamplingSchedule,
    sample_states,
)
from thrml.models import CategoricalEBMFactor, CategoricalGibbsConditional

if TYPE_CHECKING:
    from jaxtyping import Float

@dataclass(frozen=True)
class Grid:
    width: int
    height: int
    solid: np.ndarray
    round_obstacles: tuple[tuple[float, float, float], ...]
    rect_obstacles: tuple[tuple[int, int, int, int], ...]

    @property
    def n_cells(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class Weights:
    drive: float = 0.22
    inlet: float = 8.0
    outlet: float = 5.0
    wall: float = 10.0
    obstacle: float = 14.0
    no_penetration: float = 6.0
    smooth: float = 0.08
    flux: float = 0.30
    streamline: float = 0.12
    diagonal_flux: float = 0.08
    diagonal_streamline: float = 0.04
    obstacle_deflect: float = 0.9
    obstacle_tangent: float = 1.8
    rect_deflect: float = 1.5


def cell_index(x: int, y: int, width: int) -> int:
    return y * width + x


def cell_xy(index: int, width: int) -> tuple[int, int]:
    return index % width, index // width


def velocity_categories(max_speed: int) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    values = np.arange(-max_speed, max_speed + 1, dtype=np.int16)
    cats = np.asarray([(u, v) for u in values for v in values], dtype=np.int16)
    index = {(int(u), int(v)): i for i, (u, v) in enumerate(cats)}
    return cats, index


def pipe_walls(width: int, height: int) -> np.ndarray:
    solid = np.zeros((height, width), dtype=bool)
    solid[0, :] = True
    solid[-1, :] = True
    return solid


def make_pipe_with_boulder(width: int, height: int, radius: float) -> Grid:
    solid = pipe_walls(width, height)

    # A circular obstacle near the center.
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    for y in range(height):
        for x in range(width):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                solid[y, x] = True

    return Grid(
        width=width,
        height=height,
        solid=solid,
        round_obstacles=((cx, cy, radius),),
        rect_obstacles=(),
    )


def make_pipe_with_twin_boulders(width: int, height: int, radius: float) -> Grid:
    solid = pipe_walls(width, height)
    obstacles = (
        (width * 0.38, height * 0.35, radius),
        (width * 0.62, height * 0.65, radius),
    )

    for cx, cy, obstacle_radius in obstacles:
        for y in range(height):
            for x in range(width):
                if (x - cx) ** 2 + (y - cy) ** 2 <= obstacle_radius**2:
                    solid[y, x] = True

    return Grid(
        width=width,
        height=height,
        solid=solid,
        round_obstacles=obstacles,
        rect_obstacles=(),
    )


def make_pipe_with_alternating_fins(width: int, height: int) -> Grid:
    solid = pipe_walls(width, height)
    fin_length = max(3, int(round(height * 0.34)))
    fin_width = 2
    start_x = max(4, width // 5)
    spacing = max(5, width // 5)

    rects: list[tuple[int, int, int, int]] = []
    for i, x0 in enumerate(range(start_x, width - 3, spacing)):
        x1 = min(width - 1, x0 + fin_width)
        if i % 2 == 0:
            y0 = height - fin_length - 1
            y1 = height - 1
        else:
            y0 = 1
            y1 = fin_length + 1
        solid[y0:y1, x0:x1] = True
        rects.append((x0, x1, y0, y1))

    return Grid(
        width=width,
        height=height,
        solid=solid,
        round_obstacles=(),
        rect_obstacles=tuple(rects),
    )


def make_grid(geometry: str, width: int, height: int, radius: float) -> Grid:
    if geometry == "boulder":
        return make_pipe_with_boulder(width, height, radius)
    if geometry == "fins":
        return make_pipe_with_alternating_fins(width, height)
    if geometry == "twin-boulders":
        return make_pipe_with_twin_boulders(width, height, radius)
    raise ValueError(f"unknown geometry: {geometry}")


def target_penalty(cats: np.ndarray, target: tuple[int, int], scale: float) -> np.ndarray:
    target_arr = np.asarray(target, dtype=np.float32)
    diff = cats.astype(np.float32) - target_arr[None, :]
    return scale * np.sum(diff * diff, axis=1)


def build_unary_energy(
    grid: Grid,
    cats: np.ndarray,
    cat_index: dict[tuple[int, int], int],
    weights: Weights,
    inflow: int,
) -> np.ndarray:
    k = len(cats)
    energy = np.zeros((grid.n_cells, k), dtype=np.float32)
    zero = cat_index[(0, 0)]

    for y in range(grid.height):
        for x in range(grid.width):
            idx = cell_index(x, y, grid.width)
            is_solid = bool(grid.solid[y, x])

            if is_solid:
                energy[idx, :] += weights.obstacle
                energy[idx, zero] = 0.0
                continue

            if x == 0:
                energy[idx, :] += target_penalty(cats, (inflow, 0), weights.inlet)
            elif x == grid.width - 1:
                energy[idx, :] += target_penalty(cats, (inflow, 0), weights.outlet)
            else:
                # Weak left-to-right pressure-gradient surrogate.
                energy[idx, :] += target_penalty(cats, (inflow, 0), weights.drive)

            # No-penetration into adjacent solid cells.
            u = cats[:, 0]
            v = cats[:, 1]
            if x + 1 >= grid.width or grid.solid[y, x + 1]:
                energy[idx, :] += weights.no_penetration * np.maximum(u, 0) ** 2
            if x - 1 < 0 or grid.solid[y, x - 1]:
                energy[idx, :] += weights.no_penetration * np.maximum(-u, 0) ** 2
            if y + 1 >= grid.height or grid.solid[y + 1, x]:
                energy[idx, :] += weights.no_penetration * np.maximum(v, 0) ** 2
            if y - 1 < 0 or grid.solid[y - 1, x]:
                energy[idx, :] += weights.no_penetration * np.maximum(-v, 0) ** 2

            # Local inviscid "turning" surrogate: if a right-moving fluid parcel
            # would hit a solid cell soon, bias it into an open diagonal lane.
            # This is the piece that creates visible vertical velocity around the boulder.
            ahead_is_blocked = x + 1 < grid.width and grid.solid[y, x + 1]
            if ahead_is_blocked:
                can_go_up = y + 1 < grid.height and not grid.solid[y + 1, x]
                can_go_down = y - 1 >= 0 and not grid.solid[y - 1, x]
                if can_go_up or can_go_down:
                    if can_go_up and not can_go_down:
                        sign = 1
                    elif can_go_down and not can_go_up:
                        sign = -1
                    else:
                        sign = 1 if y >= (grid.height - 1) / 2.0 else -1
                    target = (max(1, inflow - 1), sign * max(1, inflow - 1))
                    energy[idx, :] += target_penalty(cats, target, weights.obstacle_deflect)

            # Boundary-layer surrogate around round obstacles. In an inviscid
            # model the obstacle does not create viscosity, but it should still
            # split streamlines around blocked regions and bend them back.
            for cx, cy, radius in grid.round_obstacles:
                dx = x - cx
                dy = y - cy
                distance = float(np.hypot(dx, dy))
                shell_width = 3.0
                in_obstacle_shell = radius < distance <= radius + shell_width
                if in_obstacle_shell and abs(dy) > 0.2:
                    if dx <= 0:
                        vertical_sign = 1 if dy > 0 else -1
                    else:
                        vertical_sign = -1 if dy > 0 else 1

                    # Strongest near the solid boundary, weaker at the outer shell.
                    closeness = (radius + shell_width - distance) / shell_width
                    tangent_weight = weights.obstacle_tangent * max(0.0, closeness)
                    target = (max(1, inflow), vertical_sign * max(1, inflow))
                    energy[idx, :] += target_penalty(cats, target, tangent_weight)

            # Rectangular fin surrogate: cells close to a protruding solid fin
            # are biased away from the fin and toward the open lane.
            for x0, x1, y0, y1 in grid.rect_obstacles:
                closest_x = min(max(x, x0), x1 - 1)
                closest_y = min(max(y, y0), y1 - 1)
                distance = float(np.hypot(x - closest_x, y - closest_y))
                shell_width = 3.0
                if 0.0 < distance <= shell_width:
                    cy = (y0 + y1 - 1) / 2.0
                    vertical_sign = 1 if y > cy else -1
                    closeness = (shell_width - distance) / shell_width
                    deflect_weight = weights.rect_deflect * max(0.0, closeness)
                    target = (max(1, inflow), vertical_sign * max(1, inflow))
                    energy[idx, :] += target_penalty(cats, target, deflect_weight)

    return energy


def neighbor_pairs(grid: Grid) -> tuple[list[tuple[int, int]], list[str]]:
    pairs: list[tuple[int, int]] = []
    orientations: list[str] = []
    for y in range(grid.height):
        for x in range(grid.width):
            here = cell_index(x, y, grid.width)
            if x + 1 < grid.width:
                pairs.append((here, cell_index(x + 1, y, grid.width)))
                orientations.append("h")
            if y + 1 < grid.height:
                pairs.append((here, cell_index(x, y + 1, grid.width)))
                orientations.append("v")
            if x + 1 < grid.width and y + 1 < grid.height:
                pairs.append((here, cell_index(x + 1, y + 1, grid.width)))
                orientations.append("du")
            if x + 1 < grid.width and y - 1 >= 0:
                pairs.append((here, cell_index(x + 1, y - 1, grid.width)))
                orientations.append("dd")
    return pairs, orientations


def grid_color_indices(grid: Grid) -> tuple[list[int], list[int], list[int], list[int]]:
    groups: tuple[list[int], list[int], list[int], list[int]] = ([], [], [], [])
    for y in range(grid.height):
        for x in range(grid.width):
            idx = cell_index(x, y, grid.width)
            groups[(x % 2) + 2 * (y % 2)].append(idx)
    return groups


def pair_energy_table(cats: np.ndarray, orientation: str, weights: Weights) -> np.ndarray:
    a = cats[:, None, :].astype(np.float32)
    b = cats[None, :, :].astype(np.float32)
    diff = a - b

    speed_diff = np.sum(diff * diff, axis=2)
    energy = weights.smooth * speed_diff

    if orientation == "h":
        # Horizontal continuity: neighboring cells should agree on horizontal flux.
        energy += weights.flux * (a[:, :, 0] - b[:, :, 0]) ** 2

        # If left cell flows right, downstream velocity should stay similar.
        energy += weights.streamline * np.maximum(a[:, :, 0], 0) * speed_diff
        # If right cell flows left, upstream velocity should stay similar.
        energy += weights.streamline * np.maximum(-b[:, :, 0], 0) * speed_diff
    elif orientation == "v":
        # Vertical continuity: neighboring cells should agree on vertical flux.
        energy += weights.flux * (a[:, :, 1] - b[:, :, 1]) ** 2

        # Same streamline idea for vertical flow.
        energy += weights.streamline * np.maximum(a[:, :, 1], 0) * speed_diff
        energy += weights.streamline * np.maximum(-b[:, :, 1], 0) * speed_diff
    else:
        if orientation == "du":
            projection_a = (a[:, :, 0] + a[:, :, 1]) / np.sqrt(2.0)
            projection_b = (b[:, :, 0] + b[:, :, 1]) / np.sqrt(2.0)
        elif orientation == "dd":
            projection_a = (a[:, :, 0] - a[:, :, 1]) / np.sqrt(2.0)
            projection_b = (b[:, :, 0] - b[:, :, 1]) / np.sqrt(2.0)
        else:
            raise ValueError(f"unknown pair orientation: {orientation}")

        # Diagonal correlations provide a cheap 9-point-stencil flavor. They
        # make diagonal bypass lanes around obstacles coherent instead of noisy.
        energy += weights.diagonal_flux * (projection_a - projection_b) ** 2
        energy += weights.diagonal_streamline * np.maximum(projection_a, 0) * speed_diff

    return energy.astype(np.float32)


def build_program(
    grid: Grid,
    cats: np.ndarray,
    cat_index: dict[tuple[int, int], int],
    weights: Weights,
    inflow: int,
    beta: float,
):
    nodes = [CategoricalNode() for _ in range(grid.n_cells)]
    free_index_groups = grid_color_indices(grid)
    free_blocks = [Block([nodes[i] for i in group]) for group in free_index_groups]

    unary_energy = build_unary_energy(grid, cats, cat_index, weights, inflow)
    unary_factor = CategoricalEBMFactor([Block(nodes)], -beta * jnp.asarray(unary_energy))

    pairs, orientations = neighbor_pairs(grid)
    head_nodes = [nodes[a] for a, _ in pairs]
    tail_nodes = [nodes[b] for _, b in pairs]
    pair_tables = np.stack(
        [pair_energy_table(cats, orientation, weights) for orientation in orientations],
        axis=0,
    )
    pair_factor = CategoricalEBMFactor(
        [Block(head_nodes), Block(tail_nodes)],
        -beta * jnp.asarray(pair_tables),
    )

    spec = BlockGibbsSpec(free_blocks, clamped_blocks=[])
    samplers = [CategoricalGibbsConditional(len(cats)) for _ in free_blocks]
    program = FactorSamplingProgram(spec, samplers, [unary_factor, pair_factor], [])
    return nodes, program


def initial_assignment(grid: Grid, cat_index: dict[tuple[int, int], int], inflow: int) -> np.ndarray:
    zero = cat_index[(0, 0)]
    flow = cat_index[(inflow, 0)]
    assignment = np.empty((grid.n_cells,), dtype=np.uint8)
    for y in range(grid.height):
        for x in range(grid.width):
            assignment[cell_index(x, y, grid.width)] = zero if grid.solid[y, x] else flow
    return assignment


def assignment_to_block_state(grid: Grid, assignment: np.ndarray) -> list[jnp.ndarray]:
    return [
        jnp.asarray([int(assignment[i]) for i in group], dtype=jnp.uint8)
        for group in grid_color_indices(grid)
    ]


def beta_schedule(beta_start: float, beta_end: float, stages: int) -> np.ndarray:
    if stages == 1:
        return np.asarray([beta_end], dtype=np.float32)
    return np.geomspace(beta_start, beta_end, stages).astype(np.float32)


def score_assignment(
    assignment: np.ndarray,
    grid: Grid,
    cats: np.ndarray,
    cat_index: dict[tuple[int, int], int],
    weights: Weights,
    inflow: int,
) -> float:
    unary = build_unary_energy(grid, cats, cat_index, weights, inflow)
    total = float(np.sum(unary[np.arange(grid.n_cells), assignment]))
    pairs, orientations = neighbor_pairs(grid)
    tables = {
        orientation: pair_energy_table(cats, orientation, weights)
        for orientation in set(orientations)
    }
    for (a, b), orientation in zip(pairs, orientations):
        total += float(tables[orientation][assignment[a], assignment[b]])
    return total


def sample_flow(args: argparse.Namespace):
    grid = make_grid(args.geometry, args.width, args.height, args.obstacle_radius)
    cats, cat_index = velocity_categories(args.max_speed)
    weights = Weights()
    betas = beta_schedule(args.beta_start, args.beta_end, args.anneal_stages)

    init = assignment_to_block_state(grid, initial_assignment(grid, cat_index, args.inflow))
    best_assignment: np.ndarray | None = None
    best_score = float("inf")

    key = jax.random.key(args.seed)
    for stage, beta in enumerate(betas, start=1):
        nodes, program = build_program(grid, cats, cat_index, weights, args.inflow, float(beta))
        schedule = SamplingSchedule(
            n_warmup=args.warmup,
            n_samples=args.samples_per_stage,
            steps_per_sample=args.steps_per_sample,
        )
        key, subkey = jax.random.split(key)
        samples = sample_states(subkey, program, schedule, init, [], [Block(nodes)])[0]
        samples_np = np.asarray(samples, dtype=np.uint8)

        for assignment in samples_np:
            score = score_assignment(assignment, grid, cats, cat_index, weights, args.inflow)
            if score < best_score:
                best_score = score
                best_assignment = assignment.copy()

        init = assignment_to_block_state(grid, samples_np[-1])
        print(f"stage {stage}/{len(betas)} beta={float(beta):.3g} best_energy={best_score:.2f}")

    assert best_assignment is not None
    return grid, cats, best_assignment, best_score


def plot_flow(
    grid: Grid,
    u: Float[np.ndarray, "n_x n_y"],
    v: Float[np.ndarray, "n_x n_y"],
    out_path: Path,
    title: str,
) -> None:
    speed = np.sqrt(u * u + v * v)

    y, x = np.mgrid[0 : grid.height, 0 : grid.width]
    mask = grid.solid
    u_plot = np.where(mask, 0.0, u)
    v_plot = np.where(mask, 0.0, v)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(speed, origin="lower", cmap="Blues", alpha=0.7)
    ax.imshow(np.where(mask, 1.0, np.nan), origin="lower", cmap="gray_r", alpha=0.95)
    ax.quiver(x, y, u_plot, v_plot, color="black", pivot="middle", scale=55, width=0.003)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(-0.5, grid.height - 0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(color="white", linewidth=0.35, alpha=0.5)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample a toy 2D flow field with THRML.")
    parser.add_argument(
        "--geometry",
        choices=("boulder", "fins", "twin-boulders"),
        default="boulder",
        help="pipe obstacle layout to sample",
    )
    parser.add_argument("--width", type=int, default=22)
    parser.add_argument("--height", type=int, default=11)
    parser.add_argument("--obstacle-radius", type=float, default=2.2)
    parser.add_argument("--max-speed", type=int, default=3)
    parser.add_argument("--inflow", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--samples-per-stage", type=int, default=120)
    parser.add_argument("--steps-per-sample", type=int, default=2)
    parser.add_argument("--anneal-stages", type=int, default=6)
    parser.add_argument("--beta-start", type=float, default=0.25)
    parser.add_argument("--beta-end", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fluid_flow_thrml/outputs/flow.png"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    grid, cats, assignment, energy = sample_flow(args)

    field = cats[assignment].reshape(grid.height, grid.width, 2).astype(float)
    u = field[:, :, 0]
    v = field[:, :, 1]
    title_geometry = args.geometry.replace("-", " ")
    plot_flow(grid, u, v, args.output,
        title=f"THRML sampled toy inviscid pipe flow: {title_geometry}, energy={energy:.2f}"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
