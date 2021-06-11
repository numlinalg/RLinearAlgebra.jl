module RLinearAlgebra

using LinearAlgebra, Random, Distributions

include("tools.jl")
include("linear_samplers.jl")

###########
# Exports #
###########

# Linear Solver exports
export rsolve, rsolve!
export LinearSolver

###########################################
# Linear Sampler Exports
###########################################

# Abstract Types
export LinSysSampler, LinSysSketch, LinSysSelect
export LinSysVecRowSampler, LinSysVecRowSketch, LinSysVecRowSelect
export LinSysVecColSampler, LinSysVecColSketch, LinSysVecColSelect
export LinSysBlkRowSampler, LinSysBlkRowSketch, LinSysBlkRowSelect
export LinSysBlkColSampler, LinSysBlkColSketch, LinSysBlkColSelect

# Vector Row Samplers
export LinSysVecRowDetermCyclic, LinSysVecRowHopRandCyclic, LinSysVecRowOneRandCyclic,
    LinSysVecRowPropToNormSampler, LinSysVecRowSVSampler, LinSysVecRowRandCyclic,
    LinSysVecRowUnidSampler

# Vector Column Samplers
export LinSysVecColDetermCyclic

end # module
