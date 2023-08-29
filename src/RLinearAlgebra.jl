############################################################################################
## This file is part of RLinearAlgebra.jl
##
## Overview: abstractions and methods for performing various randomized linear algebra
## computations
##
## Contents
## - Dependencies
## - Export Statements
##  + Linear Sampler
##  + Linear Solver Routine
##  + Linear Solver Log
##  + Linear Solver Stopping Criteria
## - Source File Inclusions
##
############################################################################################

module RLinearAlgebra

###########################################
# Dependencies
###########################################

using LinearAlgebra, Random, Distributions

###########################################
# Exports
###########################################

#*****************************************#
# Linear Sampler Exports
#*****************************************#

# Abstract Types
export LinSysSampler, LinSysSketch, LinSysSelect
export LinSysVecRowSampler, LinSysVecRowSketch, LinSysVecRowSelect
export LinSysVecColSampler, LinSysVecColSketch, LinSysVecColSelect
export LinSysBlkRowSampler, LinSysBlkRowSketch, LinSysBlkRowSelect
export LinSysBlkColSampler, LinSysBlkColSketch, LinSysBlkColSelect

# Vector Row Samplers
export LinSysVecRowDetermCyclic, LinSysVecRowHopRandCyclic, LinSysVecRowOneRandCyclic,
    LinSysVecRowPropToNormSampler, LinSysVecRowSVSampler, LinSysVecRowRandCyclic,
    LinSysVecRowUnidSampler, LinSysVecRowUnifSampler, LinSysVecRowGaussSampler,
    LinSysVecRowSparseUnifSampler, LinSysVecRowSparseGaussSampler, LinSysVecRowMaxResidual,
    LinSysVecRowMaxDistance, LinSysVecRowResidCyclic, LinSysVecRowDistCyclic

# Vector Column Samplers
export LinSysVecColDetermCyclic

#*****************************************#
# Linear Solver Routine Exports
#*****************************************#

# Abstract Types
export LinSysSolveRoutine, LinSysVecRowProjection, LinSysVecColProjection,
    LinSysBlkRowProjection, LinSysBlkColProjection, LinSysPreconKrylov

# Vector Row Projection
export LinSysVecRowProjStd, Kaczmarz, ART, LinSysVecRowProjPO, LinSysVecRowProjFO

# Vector Column Projection
export LinSysVecColProjStd, CoordinateDescent, GaussSeidel, LinSysVecColProjPO,
    LinSysVecColProjFO

#*****************************************#
# Linear Solver Log Exports
#*****************************************#
export LinSysSolverLog
export LSLogOracle, LSLogFull

#*****************************************#
# Linear Solver Stopping Criteria Exports
#*****************************************#
export LinSysStopCriterion
export LSStopMaxIterations

#*****************************************#
# Randomized Linear Solver Exports
#*****************************************#
export RLSSolver, rsolve, rsolve!


###########################################
# Source File Inclusions
###########################################

include("tools.jl")
include("linear_samplers.jl")
include("linear_solver_routines.jl")
include("linear_solver_logs.jl")
include("linear_solver_stops.jl")
include("linear_rsolve.jl")


end # module
