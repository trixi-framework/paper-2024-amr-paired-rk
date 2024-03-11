# Multirate Time-Integration based on Dynamic ODE Partitioning through Adaptively Refined Meshes for Compressible Fluid Dynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10792779.svg)](https://doi.org/10.5281/zenodo.10792779)

This repository contains information and code to reproduce the results presented in the article
```bibtex
@online{doehring2024multirate,
  title={Multirate Time-Integration based on Dynamic ODE Partitioning
         through Adaptively Refined Meshes for Compressible Fluid Dynamics},
  author={Doehring, Daniel and Schlottke-Lakemper, Michael and Gassner, Gregor J.
          and Torrilhon, Manuel},
  year={2024},
  eprint={2403.05144},
  eprinttype={arxiv},
  eprintclass={math.NA},
  url={https://arxiv.org/abs/2403.05144}
}
```
If you find these results useful, please cite the article mentioned above. If you use the implementations provided here, please also cite this repository as
```bibtex
@misc{doehring2024multirateRepro,
  title={Reproducibility repository for "{M}ultirate Time-Integration based on
         Dynamic ODE Partitioning through Adaptively Refined Meshes for
         Compressible Fluid Dynamics"},
  author={Doehring, Daniel and Schlottke-Lakemper, Michael and Gassner, Gregor J.
          and Torrilhon, Manuel},
  year={2024},
  howpublished={\url{https://github.com/trixi-framework/paper-2024-amr-paired-rk}},
  doi={https://doi.org/10.5281/zenodo.10792779}
}
```

## Abstract

In this paper, we apply the Paired-Explicit Runge-Kutta (P-ERK) schemes by Vermeire et. al. [[1](https://doi.org/10.1016/j.jcp.2019.05.014),[2](https://doi.org/10.1016/j.jcp.2022.111470)] to dynamically partitioned systems arising from adaptive mesh refinement.
The P-ERK schemes enable multirate time-integration with no changes in the spatial discretization methodology, making them readily implementable in existing codes that employ a method-of-lines approach.

We show that speedup compared to a range of state of the art Runge-Kutta methods can be realized, despite additional overhead due to the dynamic re-assignment of flagging variables and restricting nonlinear stability properties.
The effectiveness of the approach is demonstrated for a range of simulation setups for viscous and inviscid convection-dominated compressible flows for which we provide a reproducibility repository.

In addition, we perform a thorough investigation of the nonlinear stability properties of the Paired-Explicit Runge-Kutta schemes regarding limitations due to the violation of monotonicity properties of the underlying spatial discretization.
Furthermore, we present a novel approach for estimating the relevant eigenvalues of large Jacobians required for the optimization of stability polynomials.

## Reproducing the results

### Installation

To download the code using `git`, use 

```bash
git clone git@github.com:trixi-framework/paper-2024-amr-paired-rk.git
``` 

If you do not have git installed you can obtain a `.zip` and unpack it:
```bash
wget https://github.com/trixi-framework/paper-2024-amr-paired-rk/archive/main.zip
unzip paper-2024-amr-paired-rk.zip
```

To instantiate the environment execute the following two commands:
```bash
cd paper-2024-amr-paired-rk/elixirs
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Note that the results are obtained using Julia 1.9.4, which is also set in the `Manifest.toml`.
Thus, you might need to install the [old Julia 1.9.4 release](https://julialang.org/downloads/oldreleases/) first
and *replace* the `julia` calls from this README with
`/YOUR/PATH/TO/julia-1.9.4/bin/julia`

### Project initialization

If you installed Trixi.jl this way, you always have to start Julia with the `--project` flag set to your `elixirs` directory, e.g.,
```bash
julia --project=.
```
if already inside the `elixirs` directory.

If you do not execute from the `paper-2024-amr-paired-rk/elixirs/` directory, you have to call `julia` with
```bash
julia --project=/YOUR/PATH/TO/paper-2024-amr-paired-rk/elixirs/
```

### Running the code

The scripts for validations and applications are located in the `elixirs` directory.

To execute them provide the respective path:

```bash
julia --project=. ./sec5_validation/error_comparison/PERK2_3.jl
```

For all cases in the `applications` directory the solution has been computed using a specific number of 
threads.
To specify the number of threads the [`--threads` flag](https://docs.julialang.org/en/v1/manual/multi-threading/#Starting-Julia-with-multiple-threads) needs to be specified, i.e., 
```bash
julia --project=. --threads 8 ./sec7_applications/hyperbolic_parabolic/doubly_periodic_shear_layer/PERK3_3_4_7.jl
```
The precise number of threads for the different cases is given in `elixirs/sec7_applications/README.md`.

## Authors

* [Daniel Doehring](https://www.acom.rwth-aachen.de/the-lab/team-people/name:daniel_doehring) (Corresponding Author)
* [Michael Schlottke-Lakemper](https://lakemper.eu/)
* [Gregor J. Gassner](https://www.mi.uni-koeln.de/NumSim/gregor-gassner/)
* [Manuel Torrilhon](https://www.acom.rwth-aachen.de/the-lab/team-people/name:manuel_torrilhon)

## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
