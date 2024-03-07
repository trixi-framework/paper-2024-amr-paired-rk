# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Wrapper type for solutions from Trixi.jl's own time integrators, partially mimicking
# SciMLBase.ODESolution
struct TimeIntegratorSolution{tType, uType, P}
    t::tType
    u::uType
    prob::P
end

include("methods_2N.jl")
include("methods_3Sstar.jl")

include("methods_PERK.jl")
include("methods_PERKMulti.jl")


include("methods_PERK3.jl")
include("methods_PERK3_Para.jl")

include("methods_PERK3Multi.jl")
include("methods_PERK3_Para_Multi.jl")


include("methods_SSP.jl")
end # @muladd
