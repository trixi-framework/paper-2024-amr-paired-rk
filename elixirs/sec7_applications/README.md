# Applications

This directory contains the applications of Section 7.
The hyperbolic-parabolic problems from Section 7.1 are contained in `sec_7.1_hyperbolic_parabolic` 
while the purely hyperbolic problems from Section 7.2 are given in `sec_7.2_hyperbolic`

## Executing the code with the right number of threads

The presented runtimes have been obtained with specific numbers of threads.
For maximum reproducibility there is an `@assert` barrier in every elixir ensuring that the correct number of threads is set via
```bash
julia --threads NUMTHREADS
```

Remember, the following commands assume the curren working directory is `paper-2024-amr-paired-rk/elixirs`

### Hyperbolic-Parabolic

```bash
julia --project=. --threads 8 ./sec7_applications/hyperbolic_parabolic/doubly_periodic_shear_layer/PERK3_3_4_7.jl
```

```bash
julia --project=. --threads 16 ./sec7_applications/hyperbolic_parabolic/taylor_green_vortex/PERK3_3_4_6.jl
```

```bash
julia --project=. --threads 16 ./sec7_applications/hyperbolic_parabolic/visco_resistive_orszag_tang/PERK3_3_4_6.jl
```

### Hyperbolic

```bash
julia --project=. --threads 8 ./sec7_applications/hyperbolic/kelvin_helmholtz/PERK3_4_6_11.jl
```

```bash
julia --project=. --threads 8 ./sec7_applications/hyperbolic/mhd_rotor/PERK3_4_6_10.jl
```

```bash
julia --project=. --threads 16 ./sec7_applications/hyperbolic/rayleigh_taylor/PERK3_3_4_6.jl
```