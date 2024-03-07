
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations


prandtl_number() = 0.72
mu() = 6.25e-4 # equivalent to Re = 1600

equations = CompressibleEulerEquations3D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu=mu(),
                                                          Prandtl=prandtl_number())

"""
    initial_condition_taylor_green_vortex(x, t, equations::CompressibleEulerEquations3D)

The classical viscous Taylor-Green vortex, as found for instance in

- Jonathan R. Bull and Antony Jameson
  Simulation of the Compressible Taylor Green Vortex using High-Order Flux Reconstruction Schemes
  [DOI: 10.2514/6.2014-3210](https://doi.org/10.2514/6.2014-3210)
"""
# See also https://www.sciencedirect.com/science/article/pii/S187775031630299X?via%3Dihub#sec0070
function initial_condition_taylor_green_vortex(x, t, equations::CompressibleEulerEquations3D)
  A  = 1.0 # magnitude of speed
  Ms = 0.1 # maximum Mach number

  rho = 1.0
  v1  =  A * sin(x[1]) * cos(x[2]) * cos(x[3])
  v2  = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
  v3  = 0.0
  p   = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
  p   = p + 1.0/16.0 * A^2 * rho * (cos(2*x[1])*cos(2*x[3]) + 2*cos(2*x[2]) + 2*cos(2*x[1]) + cos(2*x[2])*cos(2*x[3]))

  return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end
initial_condition = initial_condition_taylor_green_vortex

volume_flux = flux_ranocha
solver = DGSEM(polydeg=2, surface_flux=flux_hlle,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0, -1.0) .* pi
coordinates_max = ( 1.0,  1.0,  1.0) .* pi
InitialRefinement = 4 # For real run
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=InitialRefinement,
                n_cells_max=2500000)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver)
        
###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan; split_form = false) # PERK

summary_callback = SummaryCallback()

interval = 20 # PERK 6, 4, 3

analysis_callback = AnalysisCallback(semi, interval=interval, save_analysis=true,
                                     analysis_errors = Symbol[],
                                     analysis_integrals=(energy_kinetic,
                                     enstrophy))

amr_indicator = IndicatorMax(semi, variable = Trixi.enstrophy)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=InitialRefinement,
                                      med_level =InitialRefinement+1, med_threshold=10.0,
                                      max_level =InitialRefinement+2, max_threshold=30.0)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = interval,
                           adapt_initial_condition=false,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=4.1) # PERK 3, 4, 6

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback,
                        amr_callback)

###############################################################################
# run the simulation

Stages = [6, 4, 3] # Have three levels anyway

cd(@__DIR__)
ode_algorithm = PERK3_Para_Multi(Stages, "./data/")

@assert Threads.nthreads() == 16 "Provided data obtained on 16 threads"
sol = Trixi.solve(ode, ode_algorithm, dt = 42.0,
                  save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary