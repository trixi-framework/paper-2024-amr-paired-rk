
using Trixi, OrdinaryDiffEq, Plots

###############################################################################
# semidiscretization of the compressible Euler equations

gamma = 5/3
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_rayleigh_taylor_instability(coordinates, t, equations::CompressibleEulerEquations2D)

Setup used for the Rayleigh-Taylor instability. Initial condition adapted from
- Shi, Jing, Yong-Tao Zhang, and Chi-Wang Shu (2003).
  Resolution of high order WENO schemes for complicated flow structures.
  [DOI](https://doi.org/10.1016/S0021-9991(03)00094-9).

This should be used together with `source_terms_rayleigh_taylor_instability`, which is
defined below.
"""
@inline function initial_condition_rayleigh_taylor_instability(x, t,
                                                               equations::CompressibleEulerEquations2D,
                                                               slope=1000)
  tol = 1e2*eps()

  if x[2] < 0.5
    p = 2*x[2] + 1
  else
    p = x[2] + 3/2
  end

  # smooth the discontinuity to avoid ambiguity at element interfaces
  smoothed_heaviside(x, left, right) = left + 0.5*(1 + tanh(slope * x)) * (right-left)
  rho = smoothed_heaviside(x[2] - 0.5, 2.0, 1.0)

  c = sqrt(equations.gamma * p / rho)
  v = -0.025 * c * cos(8*pi*x[1])
  u = 0.0

  return prim2cons(SVector(rho, u, v, p), equations)
end

@inline function boundary_condition_dirichlet_top(x, t,
                                                  equations::CompressibleEulerEquations2D)
  rho = 1.0
  u = 0.0
  v = 0.0
  p = 2.5
  return prim2cons(SVector(rho, u, v, p), equations)
end

@inline function boundary_condition_dirichlet_bottom(x, t,
                                                     equations::CompressibleEulerEquations2D)
  rho = 2.0
  u = 0.0
  v = 0.0
  p = 1.0
  return prim2cons(SVector(rho, u, v, p), equations)
end

@inline function source_terms_rayleigh_taylor_instability(u, x, t,
                                                          equations::CompressibleEulerEquations2D)
  g = 1.0
  rho, rho_v1, rho_v2, rho_e = u

  return SVector(0.0, 0.0, g*rho, g*rho_v2)
end

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

surface_flux = flux_hlle
volume_flux = flux_ranocha
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max=0.5,
                                            alpha_min=0.001,
                                            alpha_smooth=true,
                                            variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux, volume_integral=volume_integral)

num_elements = 12
trees_per_dimension = (num_elements, 4 * num_elements)
mesh = P4estMesh(trees_per_dimension,
                 polydeg=3, initial_refinement_level=0,
                 coordinates_min=(0.0, 0.0), coordinates_max=(0.25, 1.0),
                 periodicity=false)

initial_condition = initial_condition_rayleigh_taylor_instability

# For reflective BCs on all walls
boundary_conditions = Dict( :x_neg => boundary_condition_slip_wall,
                            :y_neg => BoundaryConditionDirichlet(boundary_condition_dirichlet_bottom),
                            :y_pos => BoundaryConditionDirichlet(boundary_condition_dirichlet_top),
                            :x_pos => boundary_condition_slip_wall
                            )

semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition, solver;
                                    boundary_conditions=boundary_conditions,
                                    source_terms = source_terms_rayleigh_taylor_instability)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.95) # https://doi.org/10.1016/S0021-9991(03)00094-9

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = Symbol[])

stepsize_callback = StepsizeCallback(cfl=1.35) # p = 3, E = 3, 4, 6

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=0.5,
                                          alpha_min=0.001,
                                          alpha_smooth=true,
                                          variable=Trixi.density)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=0,
                                      med_level =3, med_threshold=0.00125,
                                      max_level =6, max_threshold=0.0025)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=20, # PERK 3, 4, 6 
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback,
                        amr_callback)

###############################################################################
# run the simulation

Stages = [6, 4, 3]

cd(@__DIR__)
ode_algorithm = PERK3_Multi(Stages, "./data/")

@assert Threads.nthreads() == 16 "Provided data obtained on 16 threads"
sol = Trixi.solve(ode, ode_algorithm, dt = 42.0,
                  save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
plot(sol)

pd = PlotData2D(sol)
plot(pd["rho"], title = "\$ ρ, t_f = 1.95 \$", c = :jet, xticks = [0.0, 0.25], titlefontsize = 10)
plot(getmesh(pd), xlabel = "\$x\$", ylabel="\$y\$", title = "Mesh at \$t_f = 1.95\$", xticks = [0.0, 0.25], titlefontsize = 10)