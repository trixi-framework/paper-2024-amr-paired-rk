mutable struct ViscousContainer2D{uEltype <: Real}
    u_transformed::Array{uEltype, 4}
    # IDEA: Use SVector for fixed sized vectors?
    gradients::Vector{Array{uEltype, 4}}
    flux_viscous::Vector{Array{uEltype, 4}}

    # internal `resize!`able storage
    _u_transformed::Vector{uEltype}
    _gradients::Vector{Vector{uEltype}}
    _flux_viscous::Vector{Vector{uEltype}}

    function ViscousContainer2D{uEltype}(n_vars::Integer, n_nodes::Integer,
                                         n_elements::Integer) where {uEltype <: Real}
        new(Array{uEltype, 4}(undef, n_vars, n_nodes, n_nodes, n_elements),
            [Array{uEltype, 4}(undef, n_vars, n_nodes, n_nodes, n_elements) for _ in 1:2],
            [Array{uEltype, 4}(undef, n_vars, n_nodes, n_nodes, n_elements) for _ in 1:2],
            Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements),
            [Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements) for _ in 1:2],
            [Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements) for _ in 1:2])
    end
end

function init_viscous_container_2d(n_vars::Integer, n_nodes::Integer,
                                   n_elements::Integer,
                                   ::Type{uEltype}) where {uEltype <: Real}
    return ViscousContainer2D{uEltype}(n_vars, n_nodes, n_elements)
end

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(viscous_container::ViscousContainer2D, equations, dg, cache)
    capacity = nvariables(equations) * nnodes(dg) * nnodes(dg) * nelements(dg, cache)
    resize!(viscous_container._u_transformed, capacity)
    for dim in 1:2
        resize!(viscous_container._gradients[dim], capacity)
        resize!(viscous_container._flux_viscous[dim], capacity)
    end

    viscous_container.u_transformed = unsafe_wrap(Array,
                                                  pointer(viscous_container._u_transformed),
                                                  (nvariables(equations),
                                                   nnodes(dg), nnodes(dg),
                                                   nelements(dg, cache)))

    for dim in 1:2
        viscous_container.gradients[dim] = unsafe_wrap(Array,
                                                       pointer(viscous_container._gradients[dim]),
                                                       (nvariables(equations),
                                                        nnodes(dg), nnodes(dg),
                                                        nelements(dg, cache)))

        viscous_container.flux_viscous[dim] = unsafe_wrap(Array,
                                                          pointer(viscous_container._flux_viscous[dim]),
                                                          (nvariables(equations),
                                                           nnodes(dg), nnodes(dg),
                                                           nelements(dg, cache)))
    end
    return nothing
end
