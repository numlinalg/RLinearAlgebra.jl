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

import SparseArrays: sprandn, SparseMatrixCSC

import Hadamard: hadamard

import SparseArrays: sprandn, SparseMatrixCSC

import Hadamard: hadamard
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
    LinSysVecRowMaxDistance, LinSysVecRowResidCyclic, LinSysVecRowDistCyclic, LinSysBlockRowSRHT,LinSysBlockRowFJLT

# Vector Column Samplers
export LinSysVecColDetermCyclic, LinSysVecColOneRandCyclic
#Vector Block Row Samplers
export LinSysBlkRowGaussSampler, LinSysBlkRowRandCyclic, LinSysBlkRowReplace
#Vector Block Column Samplers
export LinSysBlkColRandCyclic, LinSysBlkColGaussSampler, LinSysBlkColReplace
export LinSysVecColDetermCyclic, LinSysVecColOneRandCyclic, LinSysBlockColSRHT, LinSysBlockColFJLT

#*****************************************#
# Linear Solver Routine Exports
#*****************************************#

# Abstract Types
export LinSysSolveRoutine, LinSysVecRowProjection, LinSysVecColProjection,
    LinSysBlkRowProjection, LinSysBlkColProjection, LinSysPreconKrylov

# Vector Row Projection
export LinSysVecRowProjStd, Kaczmarz, ART, LinSysVecRowProjPO, LinSysVecRowProjFO

# Vector Block Row Projection
export LinSysBlkRowLQ, BlockKaczmarz, IterativeHessianSketch

# Vector Column Projection
export LinSysVecColProjStd, CoordinateDescent, GaussSeidel, LinSysVecColProjPO,
    LinSysVecColProjFO

# Vector Block Column Projection
export LinSysBlkColGent, BlockCoordinateDescent 
#*****************************************#
# Linear Solver Log Exports
#*****************************************#
export LinSysSolverLog
export LSLogOracle, LSLogFull, LSLogMA
export get_uncertainty
#*****************************************#
# Linear Solver Stopping Criteria Exports
#*****************************************#
export LinSysStopCriterion
export LSStopMaxIterations
export LSStopThreshold, LSStopMA 
export iota_threshold
#*****************************************#
# Randomized Linear Solver Exports
#*****************************************#
export RLSSolver, rsolve, rsolve!

###########################################
# Low Rank Approximation Exports
###########################################
export ApproxMethod, RangeFinderMethod, IntDecompMethod, NystromMethod

# Rangefinder methods

# Interpolatory decomposition methods

# Nystrom methods

# Function to perform the approximation
export rapproximate


###########################################
# Low Rank Approximation Error Exports
###########################################
export ApproxError, RangeError
###########################################
# Source File Inclusions
###########################################

include("tools.jl")
include("linear_samplers.jl")
include("linear_solver_routines.jl")
include("linear_solver_logs.jl")
include("linear_solver_stops.jl")
include("linear_rsolve.jl")
include("low_rank_approx.jl")

end # module
