# THRML Toy Fluid Flow

> Disclaimer: This project code was generated with assistance from OpenAI Codex.

This is a deliberately small THRML experiment for sampling a 2D velocity field in
a pipe with an obstacle.

It is **not** a production Navier-Stokes solver. It is an energy-based toy model
that borrows local residual ideas from steady, inviscid, incompressible flow.

Each grid cell has one categorical velocity state:

```text
(u, v) where u, v in {-3, -2, -1, 0, 1, 2, 3}
```

So each non-obstacle cell has 49 possible velocity categories.

The energy contains:

- inlet/outlet/wall/obstacle boundary penalties,
- no-penetration penalties near solid cells,
- pairwise smoothness between neighboring velocities,
- approximate flux-continuity penalties,
- diagonal 9-point-stencil-style correlations,
- a streamline/advection-style penalty that discourages velocity from changing
  along the direction of flow,
- a local obstacle-deflection term that biases right-moving flow into open
  diagonal lanes when a boulder is directly ahead,
- an obstacle-shell tangent-flow prior that splits streamlines around the
  boulder and bends them back afterward.

THRML samples from:

```text
P(velocity_field) proportional to exp(-beta * E(field))
```

The script anneals `beta`, keeps the lowest-energy sampled field, and writes a
quiver plot.

## Run

From the repository root:

```bash
source .venv/bin/activate
pip install -r fluid_flow_thrml/requirements.txt
python fluid_flow_thrml/fluid_sampler.py
```

The output image is written to:

```text
fluid_flow_thrml/outputs/flow.png
```

Try a longer run:

```bash
python fluid_flow_thrml/fluid_sampler.py \
  --width 26 \
  --height 13 \
  --samples-per-stage 250 \
  --anneal-stages 8 \
  --beta-end 6.0
```
