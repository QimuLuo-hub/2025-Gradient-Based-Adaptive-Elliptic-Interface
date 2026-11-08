# Gradient-Based Adaptive Sampling for Elliptic Interface Problems

PyTorch code for a **deep unfitted Nitsche method** with **gradient-based adaptive sampling**, applied to elliptic interface problems in 2D and 3D:

$$
-\nabla\cdot(\beta\nabla u) = f \ \text{ in } \Omega_1\cup\Omega_2, \qquad
u = g \ \text{ on } \partial\Omega, \qquad
[\![u]\!] = p,\ \ [\![\beta\,\partial_n u]\!] = q \ \text{ on } \Gamma,
$$

where the coefficient $\beta$ is piecewise constant and can jump sharply across the interface $\Gamma$.

This is the cleaned-up 2025 code base for the SURF 2024 project [Deep Learning for Plasma Simulation](https://github.com/QimuLuo-hub/DL-for-Plasma-Simulation) (poster, slides and report are there). The energy-based extension of the method was published in *East Asian Journal on Applied Mathematics* ([paper](https://doi.org/10.4208/eajam.2025-069.100925), [code](https://github.com/QimuLuo-hub/Energy-Based-Adaptive-Deep-Unfitted-Nitsche-Method-for-Elliptic-Interface-Problems)).

## Method

- **Two networks, one per subdomain.** Fully connected tanh networks $u^{+}_\theta$ and $u^{-}_\theta$ (`Fc_Net.py`) are coupled only through the interface terms.
- **Nitsche energy loss** (`Loss_fun.py`). This is a Monte-Carlo approximation of the Nitsche energy functional. It includes subdomain energy terms, weighted averages $\{\!\{\cdot\}\!\}$ on $\Gamma$, an interface penalty $\gamma_f$ and a boundary penalty $\gamma_b$, each scaled by the measure of the subdomain, interface or boundary.
- **Gradient indicator** (`indicator.py`). It computes $\lVert\nabla u_\theta\rVert$ at the interior, boundary and interface points; on $\Gamma$ it takes the larger of the two networks' values.
- **Adaptive refinement loop** (`2d/*_adapt.py`, `3d/*_adapt.py`):
  1. start from a coarse uniform grid of $h^d$ points and $N_f$ interface points;
  2. train for a fixed number of epochs;
  3. rank the points by the indicator and refine around the top-ranked ones by a factor `re_factor`, inserting `num_insert` new points between neighbouring interface points;
  4. repeat for `re_level` levels, recording the $L^2$ error, point counts and training time at each level.

## Test problems

| Dim | Interface | Defined in | Adaptive driver |
|---|---|---|---|
| 2D | Flower $r = 0.5 + \sin(5\theta)/7$ | `2d/pde_2d.py` | `2d/flower_adapt.py` |
| 2D | Sunflower $r = 0.4 + 0.2\sin(15\theta)$, $\beta^{+}/\beta^{-} = 10$ | `2d/pde_2d.py` | `2d/sunflower_adapt.py` |
| 2D | Circle and straight interfaces | `2d/pde_2d.py` | — |
| 3D | Ellipsoid, $\beta^{+}/\beta^{-} = 60/2$ | `3d/pde_ellip.py` | `3d/ellip_adapt.py` |
| 3D | Popcorn-shaped interface | `3d/pde_pop.py` | `3d/pop_adapt.py` |

Each PDE class provides the level-set function `phi`, the coefficients, the exact solution, the right-hand side, the jumps $p, q$, the interface normal and point samplers.

## Repository structure

```
code/
├── Fc_Net.py                 # fully connected network
├── Loss_fun.py               # Nitsche energy loss
├── indicator.py              # gradient-magnitude indicator
├── visualize_all.py          # 2D surfaces / 3D (PyVista) solution and error plots
├── train_nitsche.py          # non-adaptive baseline (see note below)
├── sampling_comparison_2d.py # illustration: uniform vs. adaptive point sets (2D)
├── sampling_comparison_3d.py # same, 3D
├── 2d/                       # 2D problems, adaptive drivers and result figures
└── 3d/                       # 3D problems, adaptive drivers and result figures
sampling_compare/             # figures from the sampling-comparison scripts
```

## Running

```bash
pip install torch numpy pandas matplotlib pyvista openpyxl
```

The scripts import shared modules as `Adaptive_final_2025.<module>`, which was the name of the original project folder. To run them, make `code/` importable under that name, for example:

```bash
ln -s code Adaptive_final_2025
export PYTHONPATH=$PWD
cd code/2d && python sunflower_adapt.py
```

The hyperparameters for each run (`re_level`, `epochs`, `h`, `Nf`, `re_factor`, `num_insert`, `gamma_b`, `gamma_f`, network width `m` and `depth`, learning rate) are set at the bottom of each driver script. The plotting code uses LaTeX fonts (`text.usetex = True`), so a LaTeX installation is needed for the figures.

**Note:** `train_nitsche.py` (the non-adaptive baseline) still imports an older internal package (`DUNM.Module.*`) that is not part of this repository. The adaptive drivers in `2d/` and `3d/` do not need it.

## Results

Over the whole project (this gradient-based indicator and the later energy-based one), adaptive sampling gave more than **3× better accuracy** than non-adaptive sampling, with **over 80% fewer training points**, in both 2D and 3D. The published paper has the exact settings. Example figures (sampling at each refinement level, exact vs. predicted solutions, absolute error) are in `code/2d/`, `code/3d/` and `sampling_compare/`.
