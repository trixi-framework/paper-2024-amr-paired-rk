# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

"""
    PERK3_Para()

The following structures and methods provide a minimal implementation of
the paired explicit Runge-Kutta method (https://doi.org/10.1016/j.jcp.2019.05.014)
optimized for a certain simulation setup (PDE, IC & BC, Riemann Solver, DG Solver).

This is using the same interface as OrdinaryDiffEq.jl, copied from file "methods_2N.jl" for the
CarpenterKennedy2N{54, 43} methods.
"""

mutable struct PERK3_Para
  const NumStages::Int

  AMatrix::Matrix{Float64}
  c::Vector{Float64}

  # Constructor for previously computed A Coeffs
  function PERK3_Para(NumStages_::Int, BasePathMonCoeffs_::AbstractString, cS2_::Float64=1.0)

    newPERK3 = new(NumStages_)

    newPERK3.AMatrix, newPERK3.c = 
      ComputePERK3_ButcherTableau(NumStages_, BasePathMonCoeffs_, cS2_)
    return newPERK3
  end
end # struct PERK3_Para


# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PERK3_IntegratorPara{RealT<:Real, uType, Params, Sol, F, Alg, PERK_IntegratorOptions}
  u::uType
  du::uType
  u_tmp::uType
  t::RealT
  dt::RealT # current time step
  dtcache::RealT # ignored
  iter::Int # current number of time steps (iteration)
  p::Params # will be the semidiscretization from Trixi
  sol::Sol # faked
  f::F
  alg::Alg # This is our own class written above; Abbreviation for ALGorithm
  opts::PERK_IntegratorOptions
  finalstep::Bool # added for convenience
  # PERK stages:
  k1::uType
  k_higher::uType
  k_S1::uType # Required for third order
  t_stage::RealT
  du_ode_hyp::uType
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::PERK3_IntegratorPara, field::Symbol)
  if field === :stats
    return (naccept = getfield(integrator, :iter),)
  end
  # general fallback
  return getfield(integrator, field)
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::PERK3_Para;
               dt, callback=nothing, kwargs...)

  u0    = copy(ode.u0)
  du    = zero(u0) #previously: similar(u0) 
  u_tmp = zero(u0)

  # PERK stages
  k1       = zero(u0)
  k_higher = zero(u0)
  k_S1     = zero(u0)

  du_ode_hyp = zero(u0)

  t0 = first(ode.tspan)
  iter = 0

  integrator = PERK3_IntegratorPara(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                                (prob=ode,), ode.f, alg,
                                PERK_IntegratorOptions(callback, ode.tspan; kwargs...), false,
                                k1, k_higher, k_S1, t0, du_ode_hyp)
            
  # initialize callbacks
  if callback isa CallbackSet
    for cb in callback.continuous_callbacks
      error("unsupported")
    end
    for cb in callback.discrete_callbacks
      cb.initialize(cb, integrator.u, integrator.t, integrator)
    end
  elseif !isnothing(callback)
    error("unsupported")
  end

  solve!(integrator)
end

function solve!(integrator::PERK3_IntegratorPara)
  @unpack prob = integrator.sol
  @unpack alg = integrator
  t_end = last(prob.tspan)
  callbacks = integrator.opts.callback

  integrator.finalstep = false

  #@trixi_timeit timer() "main loop" while !integrator.finalstep
  while !integrator.finalstep
    if isnan(integrator.dt)
      error("time step size `dt` is NaN")
    end

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end || isapprox(integrator.t + integrator.dt, t_end)
      integrator.dt = t_end - integrator.t
      terminate!(integrator)
    end

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
      # k1:
      integrator.f(integrator.du, integrator.u, prob.p, integrator.t, integrator.du_ode_hyp)
      @threaded for i in eachindex(integrator.du)
        integrator.k1[i] = integrator.du[i] * integrator.dt
      end

      # k2
      integrator.t_stage = integrator.t + alg.c[2] * integrator.dt
    
      # Construct current state
      @threaded for i in eachindex(integrator.du)
        integrator.u_tmp[i] = integrator.u[i] + alg.c[2] * integrator.k1[i]
      end

      integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, integrator.du_ode_hyp)

      @threaded for i in eachindex(integrator.du)
        integrator.k_higher[i] = integrator.du[i] * integrator.dt
      end

      if alg.NumStages == 3
        @threaded for i in eachindex(integrator.du)
          integrator.k_S1[i] = integrator.k_higher[i]
        end
      end
      
      # Higher stages
      for stage = 3:alg.NumStages
        integrator.t_stage = integrator.t + alg.c[stage] * integrator.dt

        # Construct current state
        @threaded for i in eachindex(integrator.du)
          integrator.u_tmp[i] = integrator.u[i] + alg.AMatrix[stage - 2, 1] * integrator.k1[i] + 
                                                  alg.AMatrix[stage - 2, 2] * integrator.k_higher[i]
        end

        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, integrator.du_ode_hyp)

        @threaded for i in eachindex(integrator.du)
          integrator.k_higher[i] = integrator.du[i] * integrator.dt
        end

        # TODO: Stop for loop at NumStages -1 to avoid if
        if stage == alg.NumStages - 1
          @threaded for i in eachindex(integrator.du)
            integrator.k_S1[i] = integrator.k_higher[i]
          end
        end
      end

      @threaded for i in eachindex(integrator.u)
        # Proposed PERK
        #integrator.u[i] += 0.75 * integrator.k_S1[i] + 0.25 * integrator.k_higher[i]
        # Own PERK based on SSPRK33
        integrator.u[i] += (integrator.k1[i] + integrator.k_S1[i] + 4.0 * integrator.k_higher[i])/6.0
      end
    end # PERK step timer

    integrator.iter += 1
    integrator.t += integrator.dt

    # handle callbacks
    if callbacks isa CallbackSet
      for cb in callbacks.discrete_callbacks
        if cb.condition(integrator.u, integrator.t, integrator)
          cb.affect!(integrator)
        end
      end
    end

    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
      @warn "Interrupted. Larger maxiters is needed."
      terminate!(integrator)
    end
  end # "main loop" timer
  
  return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                (prob.u0, integrator.u),
                                integrator.sol.prob)
end

# get a cache where the RHS can be stored
get_du(integrator::PERK3_IntegratorPara) = integrator.du
get_tmp_cache(integrator::PERK3_IntegratorPara) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::PERK3_IntegratorPara, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::PERK3_IntegratorPara, dt)
  integrator.dt = dt
end

function get_proposed_dt(integrator::PERK3_IntegratorPara)
  return integrator.dt
end

# stop the time integration
function terminate!(integrator::PERK3_IntegratorPara)
  integrator.finalstep = true
  empty!(integrator.opts.tstops)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PERK3_IntegratorPara, new_size)
  resize!(integrator.u, new_size)
  resize!(integrator.du, new_size)
  resize!(integrator.u_tmp, new_size)

  resize!(integrator.k1, new_size)
  resize!(integrator.k_higher, new_size)
  resize!(integrator.k_S1, new_size)

  # TODO: Move this into parabolic cache or similar
  resize!(integrator.du_ode_hyp, new_size)
end

end # @muladd