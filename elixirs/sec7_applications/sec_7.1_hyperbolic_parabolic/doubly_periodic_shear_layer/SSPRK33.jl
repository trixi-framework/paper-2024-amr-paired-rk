
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations


prandtl_number() = 0.72
mu() = 1.0/40000

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu=mu(),
                                                          Prandtl=prandtl_number())
"""
A compressible version of the double shear layer initial condition. Adapted from
Brown and Minion (1995).

- David L. Brown and Michael L. Minion (1995)
  Performance of Under-resolved Two-Dimensional Incompressible Flow Simulations.
  [DOI: 10.1006/jcph.1995.1205](https://doi.org/10.1006/jcph.1995.1205)
"""
function initial_condition_shear_layer(x, t, equations::CompressibleEulerEquations2D)
  # Shear layer parameters
  k = 80
  delta = 0.05
  u0 = 1.0
  
  Ms = 0.1 # maximum Mach number

  rho = 1.0
  v1  = x[2] <= 0.5 ? u0 * tanh(k*(x[2] - 0.25)) : u0 * tanh(k*(0.75 -x[2]))
  v2  = u0 * delta * sin(2*pi*(x[1]+ 0.25))
  p   = (u0 / Ms)^2 * rho / equations.gamma # scaling to get Ms

  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_shear_layer

volume_flux = flux_ranocha
solver = DGSEM(polydeg=3, surface_flux=flux_hllc,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
InitialRefinement = 3
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=InitialRefinement,
                n_cells_max=100_000)


semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

#tspan = (0.0, 0.8) # For plot only
tspan = (0.0, 1.2)

ode = semidiscretize(semi, tspan) #  For ODE.jl methods

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = Symbol[])

alive_callback = AliveCallback(analysis_interval=analysis_interval,)

amr_indicator = IndicatorLöhner(semi, variable=v1)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = InitialRefinement,
                                      med_level  = InitialRefinement+4, med_threshold=0.15,
                                      max_level  = InitialRefinement+6, max_threshold=0.3)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=60, # SSPRK33
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

cfl = 1.0 # SSPRK33

stepsize_callback = StepsizeCallback(cfl=cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback,
                        amr_callback)

###############################################################################
# run the simulation

@assert Threads.nthreads() == 8 "Provided data obtained on 8 threads"

sol = solve(ode, SSPRK33(;thread = OrdinaryDiffEq.True());
            dt = 1.0,
            ode_default_options()..., callback=callbacks);

summary_callback() # print the timer summary