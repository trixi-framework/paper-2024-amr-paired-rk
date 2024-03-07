# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

mutable struct PERK3_Para_Multi{StageCallbacks}
  const NumStageEvalsMin::Int64
  const NumMethods::Int64
  const NumStages::Int64
  stage_callbacks::StageCallbacks

  AMatrices::Array{Float64, 3}
  c::Vector{Float64}
  ActiveLevels::Vector{Vector{Int64}}
  HighestActiveLevels::Vector{Int64}
  HighestEvalLevels::Vector{Int64}

  function PERK3_Para_Multi(NumStageEvalsMin_::Int, NumMethods_::Int,
                       BasePathMonCoeffs_::AbstractString, cS2_::Float64=1.0;
                       stage_callbacks=())

    newPERK3_Multi = new{typeof(stage_callbacks)}(NumStageEvalsMin_, NumMethods_,
                        # Current convention: NumStages = MaxStages = S;
                        # TODO: Allow for different S >= Max {Stage Evals}
                        NumStageEvalsMin_ * 2^(NumMethods_ - 1),
                        stage_callbacks)

    newPERK3_Multi.AMatrices, newPERK3_Multi.c, newPERK3_Multi.ActiveLevels, 
    newPERK3_Multi.HighestActiveLevels, newPERK3_Multi.HighestEvalLevels = 
      ComputePERK3_Multi_ButcherTableau(NumMethods_, newPERK3_Multi.NumStages, BasePathMonCoeffs_, cS2_)

    return newPERK3_Multi
  end

  function PERK3_Para_Multi(Stages_::Vector{Int64},
                       BasePathMonCoeffs_::AbstractString, cS2_::Float64=1.0;
                       stage_callbacks=())

    newPERK3_Multi = new{typeof(stage_callbacks)}(minimum(Stages_),
                          length(Stages_),
                          maximum(Stages_),
                          stage_callbacks)

    newPERK3_Multi.AMatrices, newPERK3_Multi.c, newPERK3_Multi.ActiveLevels, 
    newPERK3_Multi.HighestActiveLevels, newPERK3_Multi.HighestEvalLevels = 
      ComputePERK3_Multi_ButcherTableau(Stages_, newPERK3_Multi.NumStages, BasePathMonCoeffs_, cS2_)

    return newPERK3_Multi
  end
end # struct PERK3_Para_Multi


# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PERK3_Multi_Para_Integrator{RealT<:Real, uType, Params, Sol, F, Alg, PERK_IntegratorOptions}
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
  # PERK3_Para_Multi stages:
  k1::uType
  k_higher::uType
  k_S1::uType # Required for third order
  # Variables managing level-depending integration
  level_info_elements::Vector{Vector{Int64}}
  level_info_elements_acc::Vector{Vector{Int64}}
  level_info_interfaces_acc::Vector{Vector{Int64}}
  level_info_boundaries_acc::Vector{Vector{Int64}}
  level_info_boundaries_orientation_acc::Vector{Vector{Vector{Int64}}}
  level_info_mortars_acc::Vector{Vector{Int64}}
  level_u_indices_elements::Vector{Vector{Int64}}
  t_stage::RealT
  coarsest_lvl::Int64
  n_levels::Int64
  du_ode_hyp::uType
  AddRHSCalls::Float64
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::PERK3_Multi_Para_Integrator, field::Symbol)
  if field === :stats
    return (naccept = getfield(integrator, :iter),)
  end
  # general fallback
  return getfield(integrator, field)
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::PERK3_Para_Multi;
               dt, callback=nothing, kwargs...)

  u0    = copy(ode.u0)
  du    = zero(u0) #previously similar(u0)
  u_tmp = zero(u0)

  # PERK3_Para_Multi stages
  k1       = zero(u0)
  k_higher = zero(u0)
  k_S1     = zero(u0)

  du_ode_hyp = zero(u0) # TODO: Not best solution since this is not needed for hyperbolic problems

  t0 = first(ode.tspan)
  iter = 0

  ### Set datastructures for handling of level-dependent integration ###
  mesh, _, dg, cache = mesh_equations_solver_cache(ode.p)
  @unpack elements, interfaces, boundaries = cache

  if typeof(mesh) <:TreeMesh
    n_elements   = length(elements.cell_ids)
    n_interfaces = length(interfaces.orientations)
    n_boundaries = length(boundaries.orientations) # TODO Not sure if adequate, especially multiple dimensions

    # NOTE: For really different grid sizes
    
    min_level = minimum_level(mesh.tree)
    max_level = maximum_level(mesh.tree)
    
    # NOTE: For testcase with artificial assignment
    #=
    Random.seed!(42)
    min_level = 1 # Hard-coded to our convergence study testcase
    max_level = 3 # Hard-coded to our convergence study testcase
    =#
    n_levels = max_level - min_level + 1
    
    # Initialize storage for level-wise information
    level_info_elements     = [Vector{Int64}() for _ in 1:n_levels]
    level_info_elements_acc = [Vector{Int64}() for _ in 1:n_levels]

    # Determine level for each element
    for element_id in 1:n_elements
      # Determine level
      # NOTE: For really different grid sizes
      level = mesh.tree.levels[elements.cell_ids[element_id]]
      # NOTE: For testcase with artificial assignment
      #level = rand((1, 2, 3))

      # Convert to level id
      level_id = max_level + 1 - level

      push!(level_info_elements[level_id], element_id)
      # Add to accumulated container
      for l in level_id:n_levels
        push!(level_info_elements_acc[l], element_id)
      end
    end
    @assert length(level_info_elements_acc[end]) == 
      n_elements "highest level should contain all elements"


    level_info_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]
    # Determine level for each interface
    for interface_id in 1:n_interfaces
      # Get element id: Interfaces only between elements of same size
      element_id  = interfaces.neighbor_ids[1, interface_id]

      # Determine level
      level = mesh.tree.levels[elements.cell_ids[element_id]]

      # Higher element's level determines this interfaces' level
      level_id = max_level + 1 - level
      

      # NOTE: For testcase with artificial assignment
      #=
      if element_id_left in level_info_elements[1]
        level_id_left = 1
      elseif element_id_left in level_info_elements[2]
        level_id_left = 2
      elseif element_id_left in level_info_elements[3]
        level_id_left = 3
      end

      if element_id_right in level_info_elements[1]
        level_id_right = 1
      elseif element_id_right in level_info_elements[2]
        level_id_right = 2
      elseif element_id_right in level_info_elements[3]
        level_id_right = 3
      end
      level_id = min(level_id_left, level_id_right)
      =#

      for l in level_id:n_levels
        push!(level_info_interfaces_acc[l], interface_id)
      end
    end
    @assert length(level_info_interfaces_acc[end]) == 
      n_interfaces "highest level should contain all interfaces"


    level_info_boundaries_acc = [Vector{Int64}() for _ in 1:n_levels]
    # For efficient treatment of boundaries we need additional datastructures
    n_dims = ndims(mesh.tree) # Spatial dimension
    level_info_boundaries_orientation_acc = [[Vector{Int64}() for _ in 1:2*n_dims] for _ in 1:n_levels]

    # Determine level for each boundary
    for boundary_id in 1:n_boundaries
      # Get element id (boundaries have only one unique associated element)
      element_id = boundaries.neighbor_ids[boundary_id]

      # Determine level
      level = mesh.tree.levels[elements.cell_ids[element_id]]

      # Convert to level id
      level_id = max_level + 1 - level

      # Add to accumulated container
      for l in level_id:n_levels
        push!(level_info_boundaries_acc[l], boundary_id)
      end

      # For orientation-side wise specific treatment
      if boundaries.orientations[boundary_id] == 1 # x Boundary
        if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
          for l in level_id:n_levels
            push!(level_info_boundaries_orientation_acc[l][2], boundary_id)
          end
        else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
          for l in level_id:n_levels
            push!(level_info_boundaries_orientation_acc[l][1], boundary_id)
          end
        end
      elseif boundaries.orientations[boundary_id] == 2 # y Boundary
        if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
          for l in level_id:n_levels
            push!(level_info_boundaries_orientation_acc[l][4], boundary_id)
          end
        else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
          for l in level_id:n_levels
            push!(level_info_boundaries_orientation_acc[l][3], boundary_id)
          end
        end
      elseif boundaries.orientations[boundary_id] == 3 # z Boundary
        if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
          for l in level_id:n_levels
            push!(level_info_boundaries_orientation_acc[l][6], boundary_id)
          end
        else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
          for l in level_id:n_levels
            push!(level_info_boundaries_orientation_acc[l][5], boundary_id)
          end
        end 
      end
    end
    @assert length(level_info_boundaries_acc[end]) == 
      n_boundaries "highest level should contain all boundaries"


    level_info_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]
    if n_dims > 1
      @unpack mortars = cache
      n_mortars = length(mortars.orientations)

      for mortar_id in 1:n_mortars
        # This is by convention always one of the finer elements
        element_id  = mortars.neighbor_ids[1, mortar_id]

        # Determine level
        level  = mesh.tree.levels[elements.cell_ids[element_id]]

        # Higher element's level determines this mortars' level
        level_id = max_level + 1 - level
        # Add to accumulated container
        for l in level_id:n_levels
          push!(level_info_mortars_acc[l], mortar_id)
        end
      end
      @assert length(level_info_mortars_acc[end]) == 
        n_mortars "highest level should contain all mortars"
    end
  elseif typeof(mesh) <:P4estMesh
    @unpack interfaces, boundaries = cache

    nnodes = length(mesh.nodes)
    n_elements = nelements(dg, cache)
    h_min = 42;
    h_max = 0;

    h_min_per_element = zeros(n_elements)

    if typeof(mesh) <:P4estMesh{2}
      for element_id in 1:n_elements
        # pull the four corners numbered as right-handed
        P0 = cache.elements.node_coordinates[:, 1     , 1     , element_id]
        P1 = cache.elements.node_coordinates[:, nnodes, 1     , element_id]
        #P2 = cache.elements.node_coordinates[:, nnodes, nnodes, element_id]
        #P3 = cache.elements.node_coordinates[:, 1     , nnodes, element_id]
        # compute the four side lengths and get the smallest
        #L0 = sqrt( sum( (P1-P0).^2 ) )
        L0 = abs(P1[1] - P0[1])
        #=
        L1 = sqrt( sum( (P2-P1).^2 ) )
        L2 = sqrt( sum( (P3-P2).^2 ) )
        L3 = sqrt( sum( (P0-P3).^2 ) )
        =#
        #h = min(L0, L1, L2, L3)
        h = L0
        h_min_per_element[element_id] = h
        if h > h_max 
          h_max = h
        end
        if h < h_min
          h_min = h
        end
      end
    else # typeof(mesh) <:P4estMesh{3}
      for element_id in 1:n_elements
        # pull the four corners numbered as right-handed
        P0 = cache.elements.node_coordinates[:, 1     , 1     , 1, element_id]
        P1 = cache.elements.node_coordinates[:, nnodes, 1     , 1, element_id]
        #P2 = cache.elements.node_coordinates[:, nnodes, nnodes, element_id]
        #P3 = cache.elements.node_coordinates[:, 1     , nnodes, element_id]
        # compute the four side lengths and get the smallest
        #L0 = sqrt( sum( (P1-P0).^2 ) )
        L0 = abs(P1[1] - P0[1])
        #=
        L1 = sqrt( sum( (P2-P1).^2 ) )
        L2 = sqrt( sum( (P3-P2).^2 ) )
        L3 = sqrt( sum( (P0-P3).^2 ) )
        =#
        #h = min(L0, L1, L2, L3)
        h = L0
        h_min_per_element[element_id] = h
        if h > h_max 
          h_max = h
        end
        if h < h_min
          h_min = h
        end
      end
    end

    #=
    S_min = alg.NumStageEvalsMin
    S_max = alg.NumStages
    n_levels = Int((S_max - S_min)/2) + 1 # Linearly increasing levels
    h_bins = LinRange(h_min, h_max, n_levels+1) # These are the intervals
    =#
    
    n_levels = Int(log2(round(h_max / h_min))) + 1
    if n_levels == 1
      h_bins = [h_max]
    else
      h_bins = [ceil(h_min, digits = 10) * 2^i for i = 0:n_levels-1]
    end

    println("h_min: ", h_min, " h_max: ", h_max)
    println("h_max/h_min: ", h_max/h_min)
    println("h_bins:")
    display(h_bins)
    println("\n")

    level_info_elements = [Vector{Int64}() for _ in 1:n_levels]
    level_info_elements_acc = [Vector{Int64}() for _ in 1:n_levels]
    for element_id in 1:n_elements
      h = h_min_per_element[element_id]

      level = findfirst(x-> x >= h, h_bins)
      #=
      level = findfirst(x-> x >= h, h_bins) - 1
      # Catch case h = h_min
      if level == 0
        level = 1
      end
      =#
      append!(level_info_elements[level], element_id)

      for l in level:n_levels
        push!(level_info_elements_acc[l], element_id)
      end
    end
    level_info_elements_count = Vector{Int64}(undef, n_levels)
    for i in eachindex(level_info_elements)
      level_info_elements_count[i] = length(level_info_elements[i])
    end

    n_interfaces = last(size(interfaces.u))

    level_info_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]
    # Determine level for each interface
    for interface_id in 1:n_interfaces
      # For interfaces: Elements of same size
      element_id = interfaces.neighbor_ids[1, interface_id]
      h = h_min_per_element[element_id]
      
      level = findfirst(x-> x >= h, h_bins)
      #=
      level = findfirst(x-> x >= h, h_bins) - 1
      # Catch case h = h_min
      if level == 0
        level = 1
      end
      =#

      for l in level:n_levels
        push!(level_info_interfaces_acc[l], interface_id)
      end
    end
    @assert length(level_info_interfaces_acc[end]) == 
      n_interfaces "highest level should contain all interfaces"

    n_boundaries = last(size(boundaries.u))
    level_info_boundaries_acc = [Vector{Int64}() for _ in 1:n_levels]
    # For efficient treatment of boundaries we need additional datastructures
    n_dims = ndims(mesh) # Spatial dimension
    # TODO: Not yet adapted for P4est!
    level_info_boundaries_orientation_acc = [[Vector{Int64}() for _ in 1:2*n_dims] for _ in 1:n_levels]

    # Determine level for each boundary
    for boundary_id in 1:n_boundaries
      # Get element id (boundaries have only one unique associated element)
      element_id = boundaries.neighbor_ids[boundary_id]
      h = h_min_per_element[element_id]

      # Determine level
      level = findfirst(x-> x >= h, h_bins)
      #=
      level = findfirst(x-> x >= h, h_bins) - 1
      # Catch case h = h_min
      if level == 0
        level = 1
      end
      =#

      # Add to accumulated container
      for l in level:n_levels
        push!(level_info_boundaries_acc[l], boundary_id)
      end
    end
    @assert length(level_info_boundaries_acc[end]) == 
      n_boundaries "highest level should contain all boundaries"

    @unpack mortars = cache # TODO: Could also make dimensionality check
    level_info_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]
    @unpack mortars = cache
    n_mortars = last(size(mortars.u))

    for mortar_id in 1:n_mortars
      # Get element ids
      element_id_lower  = mortars.neighbor_ids[1, mortar_id]
      h_lower = h_min_per_element[element_id_lower]

      element_id_higher = mortars.neighbor_ids[2, mortar_id]
      h_higher = h_min_per_element[element_id_higher]

      # Determine level
      level = findfirst(x-> x >= h, h_bins)
      #=
      level = findfirst(x-> x >= h, h_bins) - 1
      # Catch case h = h_min
      if level == 0
        level = 1
      end
      =#

      # Add to accumulated container
      for l in level:n_levels
        push!(level_info_mortars_acc[l], mortar_id)
      end
    end
    @assert length(level_info_mortars_acc[end]) == 
      n_mortars "highest level should contain all mortars"
  end

  #=
  println("level_info_elements:")
  display(level_info_elements); println()
  println("level_info_elements_acc:")
  display(level_info_elements_acc); println()

  println("level_info_interfaces_acc:")
  display(level_info_interfaces_acc); println()

  println("level_info_boundaries_acc:")
  display(level_info_boundaries_acc); println()
  println("level_info_boundaries_orientation_acc:")
  display(level_info_boundaries_orientation_acc); println()
  
  println("level_info_mortars_acc:")
  display(level_info_mortars_acc); println()
  =#
  
  # Set initial distribution of DG Base function coefficients 
  @unpack equations, solver = ode.p
  u = wrap_array(u0, mesh, equations, solver, cache)

  level_u_indices_elements = [Vector{Int64}() for _ in 1:n_levels]

  # Have if outside for performance reasons (this is also used in the AMR calls)
  if n_dims == 1
    for level in 1:n_levels
      for element_id in level_info_elements[level]
        # First dimension of u: nvariables, following: nnodes (per dim) last: nelements                                    
        indices = vec(transpose(LinearIndices(u)[:, :, element_id]))
        append!(level_u_indices_elements[level], indices)
      end
      sort!(level_u_indices_elements[level])
      @assert length(level_u_indices_elements[level]) == 
              nvariables(equations) * Trixi.nnodes(solver)^ndims(mesh) * length(level_info_elements[level])
    end
  elseif n_dims == 2
    for level in 1:n_levels
      for element_id in level_info_elements[level]
        # First dimension of u: nvariables, following: nnodes (per dim) last: nelements
        indices = collect(Iterators.flatten(LinearIndices(u)[:, :, :, element_id]))
        append!(level_u_indices_elements[level], indices)
      end
      sort!(level_u_indices_elements[level])
      @assert length(level_u_indices_elements[level]) == 
              nvariables(equations) * Trixi.nnodes(solver)^ndims(mesh) * length(level_info_elements[level])
    end
  elseif n_dims == 3
    for level in 1:n_levels
      for element_id in level_info_elements[level]
        # First dimension of u: nvariables, following: nnodes (per dim) last: nelements
        indices = collect(Iterators.flatten(LinearIndices(u)[:, :, :, :, element_id]))
        append!(level_u_indices_elements[level], indices)
      end
      sort!(level_u_indices_elements[level])
      @assert length(level_u_indices_elements[level]) == 
              nvariables(equations) * Trixi.nnodes(solver)^ndims(mesh) * length(level_info_elements[level])
    end
  end

  println("level_u_indices_elements:")
  display(level_u_indices_elements); println()

  ### Done with setting up for handling of level-dependent integration ###

  integrator = PERK3_Multi_Para_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                (prob=ode,), ode.f, alg,
                PERK_IntegratorOptions(callback, ode.tspan; kwargs...), false,
                k1, k_higher, k_S1,
                level_info_elements, level_info_elements_acc, 
                level_info_interfaces_acc, 
                level_info_boundaries_acc, level_info_boundaries_orientation_acc,
                level_info_mortars_acc, 
                level_u_indices_elements,
                t0, -1, n_levels, du_ode_hyp, 0.0)
            
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

  # Start actual solve
  solve!(integrator)
end


function solve!(integrator::PERK3_Multi_Para_Integrator)
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

    #@trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
      
      # k1: Evaluated on entire domain / all levels
      integrator.f(integrator.du, integrator.u, prob.p, integrator.t, integrator.du_ode_hyp)
      
      @threaded for i in eachindex(integrator.du)
        integrator.k1[i] = integrator.du[i] * integrator.dt
      end
      
      integrator.t_stage = integrator.t + alg.c[2] * integrator.dt
      # k2: Here always evaluated for finest scheme (Allow currently only max. stage evaluations)
      @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] + alg.c[2] * integrator.k1[i]
      end

      #=
      for stage_callback in alg.stage_callbacks
        stage_callback(integrator.u_tmp, integrator, prob.p, integrator.t_stage)
      end
      =#

      # CARE: This does not work if we have only one method but more than one grid level
      integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, 
                   integrator.level_info_elements_acc[1],
                   integrator.level_info_interfaces_acc[1],
                   integrator.level_info_boundaries_acc[1],
                   integrator.level_info_boundaries_orientation_acc[1],
                   integrator.level_info_mortars_acc[1],
                   integrator.level_u_indices_elements, 1,
                   integrator.du_ode_hyp)
      
      @threaded for u_ind in integrator.level_u_indices_elements[1] # Update finest level
        integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
      end

      for stage = 3:alg.NumStages
        # Construct current state
        @threaded for i in eachindex(integrator.u)
          integrator.u_tmp[i] = integrator.u[i]
        end

        # Loop over different methods with own associated level
        for level = 1:min(alg.NumMethods, integrator.n_levels)
          @threaded for u_ind in integrator.level_u_indices_elements[level]
            integrator.u_tmp[u_ind] += alg.AMatrices[level, stage - 2, 1] * integrator.k1[u_ind]
          end
        end
        for level = 1:min(alg.HighestEvalLevels[stage], integrator.n_levels)
          @threaded for u_ind in integrator.level_u_indices_elements[level]
            integrator.u_tmp[u_ind] += alg.AMatrices[level, stage - 2, 2] * integrator.k_higher[u_ind]
          end
        end

        # "Remainder": Non-efficiently integrated
        for level = alg.NumMethods+1:integrator.n_levels
          @threaded for u_ind in integrator.level_u_indices_elements[level]
            integrator.u_tmp[u_ind] += alg.AMatrices[alg.NumMethods, stage - 2, 1] * integrator.k1[u_ind]
          end
        end
        if alg.HighestEvalLevels[stage] == alg.NumMethods
          for level = alg.HighestEvalLevels[stage]+1:integrator.n_levels
            @threaded for u_ind in integrator.level_u_indices_elements[level]
              integrator.u_tmp[u_ind] += alg.AMatrices[alg.NumMethods, stage - 2, 2] * integrator.k_higher[u_ind]
            end
          end
        end

        integrator.t_stage = integrator.t + alg.c[stage] * integrator.dt

        # "coarsest_lvl" cannot be static for AMR, has to be checked with available levels
        integrator.coarsest_lvl = min(alg.HighestActiveLevels[stage], integrator.n_levels)

        # Check if there are fewer integrators than grid levels (non-optimal method)
        if integrator.coarsest_lvl == alg.NumMethods
          integrator.coarsest_lvl = integrator.n_levels
        end

        # For statically refined meshes:
        #integrator.coarsest_lvl = alg.HighestActiveLevels[stage]

        #=
        for stage_callback in alg.stage_callbacks
          stage_callback(integrator.u_tmp, integrator, prob.p, integrator.t_stage)
        end
        =#
        
        # Joint RHS evaluation with all elements sharing this timestep
        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, 
                    integrator.level_info_elements_acc[integrator.coarsest_lvl],
                    integrator.level_info_interfaces_acc[integrator.coarsest_lvl],
                    integrator.level_info_boundaries_acc[integrator.coarsest_lvl],
                    integrator.level_info_boundaries_orientation_acc[integrator.coarsest_lvl],
                    integrator.level_info_mortars_acc[integrator.coarsest_lvl],
                    integrator.level_u_indices_elements, integrator.coarsest_lvl,
                    integrator.du_ode_hyp)
        

        # Update k_higher of relevant levels
        for level in 1:integrator.coarsest_lvl
          @threaded for u_ind in integrator.level_u_indices_elements[level]
            integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
          end
        end

        if stage == alg.NumStages - 1
          @threaded for i in eachindex(integrator.du)
            integrator.k_S1[i] = integrator.k_higher[i]
          end
        end
      end
      
      @threaded for u_ind in eachindex(integrator.u)
        # Proposed PERK
        #integrator.u[i] += 0.75 * integrator.k_S1[i] + 0.25 * integrator.k_higher[i]

        # Own PERK based on SSPRK33
        integrator.u[u_ind] += (integrator.k1[u_ind] + integrator.k_S1[u_ind] + 4.0 * integrator.k_higher[u_ind])/6.0
      end
      
      #=
      for stage_callback in alg.stage_callbacks
        stage_callback(integrator.u, integrator, prob.p, integrator.t_stage)
      end
      =#
    #end # PERK3_Para_Multi step timer

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

    #=
    for stage_callback in alg.stage_callbacks
      stage_callback(integrator.u, integrator, prob.p, integrator.t_stage)
    end
    =#

    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
      @warn "Interrupted. Larger maxiters is needed."
      terminate!(integrator)
    end
  end # "main loop" timer

  println("Additional RHS Calls: ", integrator.AddRHSCalls)
  
  return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                (prob.u0, integrator.u),
                                integrator.sol.prob)
end

# get a cache where the RHS can be stored
get_du(integrator::PERK3_Multi_Para_Integrator) = integrator.du
get_tmp_cache(integrator::PERK3_Multi_Para_Integrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::PERK3_Multi_Para_Integrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::PERK3_Multi_Para_Integrator, dt)
  integrator.dt = dt
end

function get_proposed_dt(integrator::PERK3_Multi_Para_Integrator)
  return integrator.dt
end

# stop the time integration
function terminate!(integrator::PERK3_Multi_Para_Integrator)
  integrator.finalstep = true
  empty!(integrator.opts.tstops)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PERK3_Multi_Para_Integrator, new_size)
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