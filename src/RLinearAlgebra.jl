module RLinearAlgebra

using LinearAlgebra

include("tools.jl")
include("solvers.jl")

###########
# Exports #
###########

# Linear Solver exports
export rsolve, rsolve!
export LinearSolver
export TypeRPM, TypeBlendenpik, TypeRGS
export SamplerKaczmarzWR, SamplerKaczmarzCYC, SamplerMotzkin, SamplerGaussSketch
export RPMSamplers
export SVDistribution, UFDistribution
export ProjectionStdCore

end # module
