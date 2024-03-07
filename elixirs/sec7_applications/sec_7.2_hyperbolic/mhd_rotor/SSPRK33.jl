
using OrdinaryDiffEq
using Trixi


###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
equations = IdealGlmMhdEquations2D(1.4)

"""
    initial_condition_rotor(x, t, equations::IdealGlmMhdEquations2D)

The classical MHD rotor test case. Here, the setup is taken from
- Dominik Derigs, Gregor J. Gassner, Stefanie Walch & Andrew R. Winters (2018)
  Entropy Stable Finite Volume Approximations for Ideal Magnetohydrodynamics
  [doi: 10.1365/s13291-018-0178-9](https://doi.org/10.1365/s13291-018-0178-9)
"""
function initial_condition_rotor(x, t, equations::IdealGlmMhdEquations2D)
  # setup taken from Derigs et al. DMV article (2018)
  # domain must be [0, 1] x [0, 1], γ = 1.4
  dx = x[1] - 0.5
  dy = x[2] - 0.5
  r = sqrt(dx^2 + dy^2)
  f = (0.115 - r)/0.015
  if r <= 0.1
    rho = 10.0
    v1 = -20.0*dy
    v2 = 20.0*dx
  elseif r >= 0.115
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
  else
    rho = 1.0 + 9.0*f
    v1 = -20.0*f*dy
    v2 = 20.0*f*dx
  end
  v3 = 0.0
  p = 1.0
  B1 = 5.0/sqrt(4.0*pi)
  B2 = 0.0
  B3 = 0.0
  psi = 0.0
  return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end
initial_condition = initial_condition_rotor

# Original publication [doi: 10.1365/s13291-018-0178-9](https://doi.org/10.1365/s13291-018-0178-9) uses outflow BCs
@inline function boundary_condition_outflow(u_inner, orientation, direction, x, t,
                                            surface_flux_function,
                                            equations::IdealGlmMhdEquations2D)
  # Use simple outflow, also for the divergence-clearing varaible psi.
  # This follows from eq. (49) from [DOI: 10.1006/jcph.2001.6961](https://doi.org/10.1006/jcph.2001.6961)
  return surface_flux_function(u_inner, u_inner, orientation, equations)
end

boundary_conditions = (x_neg=boundary_condition_outflow,
                       x_pos=boundary_condition_outflow,
                       y_neg=boundary_condition_outflow,
                       y_pos=boundary_condition_outflow)

surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell)
volume_flux  = (flux_central, flux_nonconservative_powell)
polydeg = 4
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000,
                periodicity=false) # Use outflow BC

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.15) # final time from paper

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = Symbol[])

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=0.5,
                                          alpha_min=0.001,
                                          alpha_smooth=false,
                                          variable=density_pressure)
# For density_pressure
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=3,
                                      med_level =7, med_threshold=0.0025,
                                      max_level =9, max_threshold=0.25)                                           
                                      
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 20, # SSPRK33
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

cfl = 0.57 # SSPRK33

stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        amr_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

@assert Threads.nthreads() == 1 "Provided data obtained on one thread"
  
sol = solve(ode, SSPRK33(;thread = OrdinaryDiffEq.True());
            dt=42.0,
            save_everystep=false, callback=callbacks,
            ode_default_options()...);

summary_callback() # print the timer summary