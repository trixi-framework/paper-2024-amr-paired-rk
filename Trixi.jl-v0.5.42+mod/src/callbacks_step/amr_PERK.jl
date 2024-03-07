# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

# Custom implementation for PERK integrator
function (amr_callback::AMRCallback)(integrator::Union{PERK_Multi_Integrator, 
                                                       PERK3_Multi_Integrator,
                                                       PERK3_Multi_Para_Integrator}; kwargs...)
  u_ode = integrator.u
  semi = integrator.p

  @trixi_timeit timer() "AMR" begin
    has_changed = amr_callback(u_ode, semi,
                               integrator.t, integrator.iter; kwargs...)

    if has_changed
      resize!(integrator, length(u_ode))
      u_modified!(integrator, true)

      ### PERK addition ###
      # TODO: Need to make this much less allocating!
      @trixi_timeit timer() "PERK stage identifiers update" begin
        mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
        @unpack elements, interfaces, boundaries = cache

        if typeof(mesh) <:TreeMesh
          n_elements   = length(elements.cell_ids)
          n_interfaces = length(interfaces.orientations)
          n_boundaries = length(boundaries.orientations) # TODO Not sure if adequate
      
          # TODO: Not sure if this still returns the correct number of ACTIVE Levels
          min_level = minimum_level(mesh.tree)
          max_level = maximum_level(mesh.tree)
          integrator.n_levels = max_level - min_level + 1

          n_dims = ndims(mesh.tree) # Spatial dimension

          # Initialize storage for level-wise information
          if integrator.n_levels != length(integrator.level_info_elements_acc)
            integrator.level_info_elements = [Vector{Int64}() for _ in 1:integrator.n_levels]
            integrator.level_info_elements_acc = [Vector{Int64}() for _ in 1:integrator.n_levels]
            integrator.level_info_interfaces_acc = [Vector{Int64}() for _ in 1:integrator.n_levels]
            integrator.level_info_boundaries_acc = [Vector{Int64}() for _ in 1:integrator.n_levels]
            # For efficient treatment of boundaries we need additional datastructures
            integrator.level_info_boundaries_orientation_acc = [[Vector{Int64}() for _ in 1:2*n_dims] for _ in 1:integrator.n_levels]
            integrator.level_info_mortars_acc = [Vector{Int64}() for _ in 1:integrator.n_levels]
            integrator.level_u_indices_elements = [Vector{Int64}() for _ in 1:integrator.n_levels]
            #resize!(integrator.level_info_elements_acc, integrator.n_levels) # TODO: Does unfortunately not work
          else # Just empty datastructures
            for level in 1:integrator.n_levels
              empty!(integrator.level_info_elements[level])
              empty!(integrator.level_info_elements_acc[level])
              empty!(integrator.level_info_interfaces_acc[level])
              empty!(integrator.level_info_boundaries_acc[level])
              for dim in 1:2*n_dims
                empty!(integrator.level_info_boundaries_orientation_acc[level][dim])
              end
              empty!(integrator.level_info_mortars_acc[level])
              empty!(integrator.level_u_indices_elements[level])
            end
          end

          # Determine level for each element
          for element_id in 1:n_elements
            # Determine level
            level = mesh.tree.levels[elements.cell_ids[element_id]]
            # Convert to level id
            level_id = max_level + 1 - level

            push!(integrator.level_info_elements[level_id], element_id)
            # Add to accumulated container
            for l in level_id:integrator.n_levels
              push!(integrator.level_info_elements_acc[l], element_id)
            end
          end
          @assert length(integrator.level_info_elements_acc[end]) == 
            n_elements "highest level should contain all elements"

          
          # NOTE: Additional RHS Call computation
          # CARE: Hard-coded for each case 
          #Stages = [6, 4, 3] # VRMHD O-T, Taylor-Green
          #Stages = [7, 4, 3] # Shearlayer
          #CFL_Act_Ideal = 1.0

          Stages = [11, 6, 4] # Kelvin-Helmholtz
          CFL_Act_Ideal = 0.7037037037037036

          Stages = [10, 6, 4] # MHD Rotor
          CFL_Act_Ideal = 1.0
          
          MaxStage = maximum(Stages)
          MinStage = minimum(Stages)
          integrator_levels = length(Stages)          

          
          ### Compute number of performed (scalar) RHS evals ###
          for level = 1:min(integrator_levels, integrator.n_levels)
            integrator.AddRHSCalls += amr_callback.interval * 
                                      Stages[level] * length(integrator.level_info_elements[level])
          end
          
          # Contribution from non-represented levels
          for level = integrator_levels+1:integrator.n_levels
            integrator.AddRHSCalls += amr_callback.interval * MinStage * 
                                      length(integrator.level_info_elements[level])
          end
          
          ### Subtract number of ideal RHS calls ###
          # Contribution from non-ideally scaling levels
          for level = 1:integrator.n_levels
            integrator.AddRHSCalls -= CFL_Act_Ideal * amr_callback.interval * MaxStage / (2.0^(level - 1)) * 
                                      length(integrator.level_info_elements[level])
          end

          # Determine level for each interface
          for interface_id in 1:n_interfaces
            # Get element id: Interfaces only between elements of same size
            element_id  = interfaces.neighbor_ids[1, interface_id]

            # Determine level
            level = mesh.tree.levels[elements.cell_ids[element_id]]

            # Higher element's level determines this interfaces' level
            level_id = max_level + 1 - level
            for l in level_id:integrator.n_levels
              push!(integrator.level_info_interfaces_acc[l], interface_id)
            end
          end
          @assert length(integrator.level_info_interfaces_acc[end]) == 
            n_interfaces "highest level should contain all interfaces"


          # Determine level for each boundary
          for boundary_id in 1:n_boundaries
            # Get element id (boundaries have only one unique associated element)
            element_id = boundaries.neighbor_ids[boundary_id]

            # Determine level
            level = mesh.tree.levels[elements.cell_ids[element_id]]

            # Convert to level id
            level_id = max_level + 1 - level

            # Add to accumulated container
            for l in level_id:integrator.n_levels
              push!(integrator.level_info_boundaries_acc[l], boundary_id)
            end

            # For orientation-side wise specific treatment
            if boundaries.orientations[boundary_id] == 1 # x Boundary
              if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
                for l in level_id:integrator.n_levels
                  push!(integrator.level_info_boundaries_orientation_acc[l][2], boundary_id)
                end
              else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
                for l in level_id:integrator.n_levels
                  push!(integrator.level_info_boundaries_orientation_acc[l][1], boundary_id)
                end
              end
            elseif boundaries.orientations[boundary_id] == 2 # y Boundary
              if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
                for l in level_id:integrator.n_levels
                  push!(integrator.level_info_boundaries_orientation_acc[l][4], boundary_id)
                end
              else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
                for l in level_id:integrator.n_levels
                  push!(integrator.level_info_boundaries_orientation_acc[l][3], boundary_id)
                end
              end
            elseif boundaries.orientations[boundary_id] == 3 # z Boundary
              if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
                for l in level_id:integrator.n_levels
                  push!(integrator.level_info_boundaries_orientation_acc[l][6], boundary_id)
                end
              else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
                for l in level_id:integrator.n_levels
                  push!(integrator.level_info_boundaries_orientation_acc[l][5], boundary_id)
                end
              end 
            end
          end # 1:n_boundaries
          @assert length(integrator.level_info_boundaries_acc[end]) == 
            n_boundaries "highest level should contain all boundaries"

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
              for l in level_id:integrator.n_levels
                push!(integrator.level_info_mortars_acc[l], mortar_id)
              end
            end
            @assert length(integrator.level_info_mortars_acc[end]) == 
              n_mortars "highest level should contain all mortars"
          end
        #elseif typeof(mesh) <:P4estMesh{2}
        elseif typeof(mesh) <:P4estMesh
            nnodes = length(mesh.nodes)
            n_elements = nelements(solver, cache)
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
            integrator.n_levels = Int((S_max - S_min)/2) + 1 # Linearly increasing levels
            h_bins = LinRange(h_min, h_max, integrator.n_levels+1) # These are the intervals
            =#
            
            integrator.n_levels = Int(log2(round(h_max / h_min))) + 1
            if integrator.n_levels == 1
              h_bins = [h_max]
            else
              h_bins = [ceil(h_min, digits = 10) * 2^i for i = 0:integrator.n_levels-1]
            end
            #println(h_bins)

            n_dims = ndims(mesh) # Spatial dimension

            # Initialize storage for level-wise information
            if integrator.n_levels != length(integrator.level_info_elements_acc)
              integrator.level_info_elements = [Vector{Int64}() for _ in 1:integrator.n_levels]
              integrator.level_info_elements_acc = [Vector{Int64}() for _ in 1:integrator.n_levels]
              integrator.level_info_interfaces_acc = [Vector{Int64}() for _ in 1:integrator.n_levels]
              integrator.level_info_boundaries_acc = [Vector{Int64}() for _ in 1:integrator.n_levels]
              # For efficient treatment of boundaries we need additional datastructures
              integrator.level_info_boundaries_orientation_acc = [[Vector{Int64}() for _ in 1:2*n_dims] for _ in 1:integrator.n_levels]
              integrator.level_info_mortars_acc = [Vector{Int64}() for _ in 1:integrator.n_levels]
              integrator.level_u_indices_elements = [Vector{Int64}() for _ in 1:integrator.n_levels]
              #resize!(integrator.level_info_elements_acc, integrator.n_levels) # TODO: Does unfortunately not work
            else # Just empty datastructures
              for level in 1:integrator.n_levels
                empty!(integrator.level_info_elements[level])
                empty!(integrator.level_info_elements_acc[level])
                empty!(integrator.level_info_interfaces_acc[level])
                empty!(integrator.level_info_boundaries_acc[level])
                for dim in 1:2*n_dims
                  empty!(integrator.level_info_boundaries_orientation_acc[level][dim])
                end
                empty!(integrator.level_info_mortars_acc[level])
                empty!(integrator.level_u_indices_elements[level])
              end
            end

            for element_id in 1:n_elements
              h = h_min_per_element[element_id]

              level = findfirst(x-> x >= h, h_bins)

              append!(integrator.level_info_elements[level], element_id)

              for l in level:integrator.n_levels
                push!(integrator.level_info_elements_acc[l], element_id)
              end
            end
            @assert length(integrator.level_info_elements_acc[end]) == 
            n_elements "highest level should contain all elements"

            
            # NOTE: Additional RHS Call computation
            # CARE: Hard-coded for each case 
            Stages = [6, 4, 3] # Rayleigh-Taylor
            CFL_Act_Ideal = 0.9642857142857144


            MaxStage = maximum(Stages)
            MinStage = minimum(Stages)
            integrator_levels = length(Stages)

             ### Compute number of performed (scalar) RHS evals ###
            for level = 1:min(integrator_levels, integrator.n_levels)
              integrator.AddRHSCalls += amr_callback.interval * 
                                        Stages[level] * length(integrator.level_info_elements[level])
            end
            
            # Contribution from non-represented levels
            for level = integrator_levels+1:integrator.n_levels
              integrator.AddRHSCalls += amr_callback.interval * MinStage * 
                                        length(integrator.level_info_elements[level])
            end
            
            ### Subtract number of ideal RHS calls ###
            # Contribution from non-ideally scaling levels
            for level = 1:integrator.n_levels
              integrator.AddRHSCalls -= CFL_Act_Ideal * amr_callback.interval * MaxStage / (2.0^(level - 1)) * 
                                        length(integrator.level_info_elements[level])
            end


            n_interfaces = last(size(interfaces.u))

            # Determine level for each interface
            for interface_id in 1:n_interfaces
              # For interfaces: Elements of same size
              element_id = interfaces.neighbor_ids[1, interface_id]
              h = h_min_per_element[element_id]

              # Determine level
              level = findfirst(x-> x >= h, h_bins)

              for l in level:integrator.n_levels
                push!(integrator.level_info_interfaces_acc[l], interface_id)
              end
            end
            @assert length(integrator.level_info_interfaces_acc[end]) == 
              n_interfaces "highest level should contain all interfaces"

            n_boundaries = last(size(boundaries.u))
            # For efficient treatment of boundaries we need additional datastructures
            # Determine level for each boundary
            for boundary_id in 1:n_boundaries
              # Get element id (boundaries have only one unique associated element)
              element_id = boundaries.neighbor_ids[boundary_id]
              h = h_min_per_element[element_id]

              # Determine level
              level = findfirst(x-> x >= h, h_bins)

              # Add to accumulated container
              for l in level:integrator.n_levels
                push!(integrator.level_info_boundaries_acc[l], boundary_id)
              end
            end
            @assert length(integrator.level_info_boundaries_acc[end]) == 
              n_boundaries "highest level should contain all boundaries"

            @unpack mortars = cache # TODO: Could also make dimensionality check
            n_mortars = last(size(mortars.u))

            for mortar_id in 1:n_mortars
              # This is by convention always one of the finer elements
              element_id  = mortars.neighbor_ids[1, mortar_id]
              h = h_min_per_element[element_id]

              # Determine level
              level = findfirst(x-> x >= h, h_bins)

              # Add to accumulated container
              for l in level:integrator.n_levels
                push!(integrator.level_info_mortars_acc[l], mortar_id)
              end
            end
            @assert length(integrator.level_info_mortars_acc[end]) == 
              n_mortars "highest level should contain all mortars"

            #=
            println("level_info_elements:")
            display(integrator.level_info_elements); println()
            println("level_info_elements_acc:")
            display(integrator.level_info_elements_acc); println()
          
            println("level_info_interfaces_acc:")
            display(integrator.level_info_interfaces_acc); println()
          
            println("level_info_boundaries_acc:")
            display(integrator.level_info_boundaries_acc); println()
            println("level_info_boundaries_orientation_acc:")
            display(integrator.level_info_boundaries_orientation_acc); println()
            
            println("level_info_mortars_acc:")
            display(integrator.level_info_mortars_acc); println()
            =#            
        end

        u = wrap_array(u_ode, mesh, equations, solver, cache)

        if n_dims == 1
          for level in 1:integrator.n_levels
            for element_id in integrator.level_info_elements[level]
              # First dimension of u: nvariables, following: nnodes (per dim) last: nelements
              indices = vec(transpose(LinearIndices(u)[:, :, element_id]))
              append!(integrator.level_u_indices_elements[level], indices)
            end
            sort!(integrator.level_u_indices_elements[level])
          end
        elseif n_dims == 2
          for level in 1:integrator.n_levels
            for element_id in integrator.level_info_elements[level]
              # First dimension of u: nvariables, following: nnodes (per dim) last: nelements
              indices = collect(Iterators.flatten(LinearIndices(u)[:, :, :, element_id]))
              append!(integrator.level_u_indices_elements[level], indices)
            end
            sort!(integrator.level_u_indices_elements[level])
          end
        elseif n_dims == 3
          for level in 1:integrator.n_levels
            for element_id in integrator.level_info_elements[level]
              # First dimension of u: nvariables, following: nnodes (per dim) last: nelements
              indices = collect(Iterators.flatten(LinearIndices(u)[:, :, :, :, element_id]))
              append!(integrator.level_u_indices_elements[level], indices)
            end
            sort!(integrator.level_u_indices_elements[level])
          end
        end
        
      end # "PERK stage identifiers update" timing
    end # if has changed
  end # "AMR" timing

  return has_changed
end

end # @muladd