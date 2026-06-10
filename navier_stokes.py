import numpy as np
from dataclasses import dataclass

from fluid_sampler import (
    Grid,
)

# -----------------------------
# Utility
# -----------------------------

def idx(i, j, w):
    return j * w + i


def clamp(x, a, b):
    return max(a, min(b, x))


def bilinear_sample(field, x, y, w, h):
    """Simple bilinear sampling for semi-Lagrangian advection."""
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = clamp(x0, 0, w - 1)
    x1 = clamp(x1, 0, w - 1)
    y0 = clamp(y0, 0, h - 1)
    y1 = clamp(y1, 0, h - 1)

    sx = x - np.floor(x)
    sy = y - np.floor(y)

    f00 = field[y0 * w + x0]
    f10 = field[y0 * w + x1]
    f01 = field[y1 * w + x0]
    f11 = field[y1 * w + x1]

    return (
        f00 * (1 - sx) * (1 - sy)
        + f10 * sx * (1 - sy)
        + f01 * (1 - sx) * sy
        + f11 * sx * sy
    )


# -----------------------------
# Solver
# -----------------------------

@dataclass
class FluidState:
    u: np.ndarray  # x-velocity
    v: np.ndarray  # y-velocity
    p: np.ndarray  # pressure


class NavierStokesSolver:
    def __init__(self, grid: Grid, viscosity=0.001, dt=0.1):
        self.grid = grid
        self.w = grid.width
        self.h = grid.height
        self.n = grid.n_cells

        self.viscosity = viscosity
        self.dt = dt

        self.state = FluidState(
            u=np.zeros(self.n, dtype=np.float32),
            v=np.zeros(self.n, dtype=np.float32),
            p=np.zeros(self.n, dtype=np.float32),
        )

    # -----------------------------
    # Boundary handling
    # -----------------------------

    def apply_solid(self, u, v):
        """Zero velocity inside solids."""
        for i in range(self.n):
            if self.grid.solid.flat[i]:
                u[i] = 0.0
                v[i] = 0.0

    # -----------------------------
    # Advection (semi-Lagrangian)
    # -----------------------------

    def advect(self, u, v):
        w, h = self.w, self.h
        new_u = np.zeros_like(u)
        new_v = np.zeros_like(v)

        for y in range(h):
            for x in range(w):
                i = idx(x, y, w)

                if self.grid.solid[y, x]:
                    continue

                # backtrace
                x_back = x - self.dt * u[i]
                y_back = y - self.dt * v[i]

                new_u[i] = bilinear_sample(u, x_back, y_back, w, h)
                new_v[i] = bilinear_sample(v, x_back, y_back, w, h)

        return new_u, new_v

    # -----------------------------
    # Diffusion (explicit)
    # -----------------------------

    def diffuse(self, f):
        w, h = self.w, self.h
        out = f.copy()

        a = self.viscosity * self.dt

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                i = idx(x, y, w)

                if self.grid.solid[y, x]:
                    continue

                lap = (
                    f[idx(x - 1, y, w)]
                    + f[idx(x + 1, y, w)]
                    + f[idx(x, y - 1, w)]
                    + f[idx(x, y + 1, w)]
                    - 4 * f[i]
                )

                out[i] = f[i] + a * lap

        return out

    # -----------------------------
    # Pressure projection
    # -----------------------------

    def build_pressure_matrix(self):
        """
        Dense Poisson matrix for ∇²p = div(u).
        """
        n = self.n
        w, h = self.w, self.h

        A = np.zeros((n, n), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                i = idx(x, y, w)

                if self.grid.solid[y, x]:
                    A[i, i] = 1.0
                    continue

                A[i, i] = -4.0

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if not self.grid.solid[ny, nx]:
                            j = idx(nx, ny, w)
                            A[i, j] = 1.0

        return A

    def compute_divergence(self, u, v):
        w, h = self.w, self.h
        div = np.zeros(self.n, dtype=np.float32)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                i = idx(x, y, w)

                if self.grid.solid[y, x]:
                    continue

                div[i] = (
                    u[idx(x + 1, y, w)] - u[idx(x - 1, y, w)]
                    + v[idx(x, y + 1, w)] - v[idx(x, y - 1, w)]
                ) * 0.5

        return div

    def project(self, u, v):
        div = self.compute_divergence(u, v)
        A = self.build_pressure_matrix()

        p = np.linalg.solve(A, div)

        # subtract gradient
        w, h = self.w, self.h

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                i = idx(x, y, w)

                if self.grid.solid[y, x]:
                    continue

                dpdx = p[idx(x + 1, y, w)] - p[idx(x - 1, y, w)]
                dpdy = p[idx(x, y + 1, w)] - p[idx(x, y - 1, w)]

                u[i] -= 0.5 * dpdx
                v[i] -= 0.5 * dpdy

        return u, v, p

    # -----------------------------
    # External forcing (inflow)
    # -----------------------------

    def apply_inflow(self, u, v, inflow_speed=1.0):
        w, h = self.w, self.h

        for y in range(h):
            for x in range(w):
                i = idx(x, y, w)

                # simple left boundary inflow
                if x == 0 and not self.grid.solid[y, x]:
                    u[i] = inflow_speed
                    v[i] = 0.0

    # -----------------------------
    # Step
    # -----------------------------

    def step(self, inflow_speed=1.0):
        u, v = self.state.u, self.state.v

        # 1. add inflow
        self.apply_inflow(u, v, inflow_speed)

        # 2. advect velocity
        u, v = self.advect(u, v)

        # 3. diffusion
        u = self.diffuse(u)
        v = self.diffuse(v)

        # 4. projection
        u, v, p = self.project(u, v)

        # 5. enforce solids
        self.apply_solid(u, v)

        self.state = FluidState(u=u, v=v, p=p)

        return self.state


if __name__ == "__main__":
    from pathlib import Path
    from argparse import ArgumentParser

    from tqdm import tqdm

    from fluid_sampler import make_grid, plot_flow

    # -----------------------------
    # Create simulation
    # -----------------------------
    parser = ArgumentParser()
    parser.add_argument(
        "--geometry",
        choices=("boulder", "fins", "twin-boulders"),
        default="boulder",
        help="pipe obstacle layout to sample",
    )
    parser.add_argument("--width", type=int, default=22)
    parser.add_argument("--height", type=int, default=11)
    parser.add_argument("--obstacle-radius", type=float, default=2.2)
    args = parser.parse_args()

    grid = make_grid(args.geometry, args.width, args.height, args.obstacle_radius)

    solver = NavierStokesSolver(
        grid,
        viscosity=0.002,
        dt=0.8
    )

    # -----------------------------
    # Run simulation
    # -----------------------------
    steps = 80

    for _ in tqdm(range(steps)):
        state = solver.step(inflow_speed=2.0)

    u = state.u.reshape((grid.height, grid.width))
    v = state.v.reshape((grid.height, grid.width))

    out_path = Path("fluid_flow_thrml/outputs/navier_stokes.png")
    # -----------------------------
    # Save PNG
    # -----------------------------
    plot_flow(
        grid,
        u,
        v,
        out_path=out_path,
        title = "Navier–Stokes Flow Field (Velocity + Obstacles)",
    )
    print(f"Saved {out_path}")
