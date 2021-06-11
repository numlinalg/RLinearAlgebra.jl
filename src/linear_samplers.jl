############################################################################################
## This file is part of RLinearAlgebra.jl
##
## Overview: abstractions and methods for sampling, sketching or selecting from linear
## systems of equations.
##
## Contents
## - Abstract Types
## - Vector Row Sampler/Sketch/Selector
## - Vector Column Sampler/Sketch/Selector
## - Block Row Sampler/Sketch/Selector
## - Block Column Sampler/Sketch/Selector
## - Export Statements
##
############################################################################################

# Dependencies: Random, Distributions, LinearAlgebra

#############################################
# Abstract Types
#############################################

"""
    LinSysSampler

Abstract supertype for sampling, sketching or deterministically selecting components
    of a linear system.

# Aliases
- `LinSysSketch`
- `LinSysSelect`
"""
abstract type LinSysSampler end
LinSysSketch = LinSysSampler
LinSysSelect = LinSysSampler

"""
    LinSysVecRowSampler <: LinSysSampler

Abstract supertype for sampling, sketching or deterministically selecting a row space
    element from a linear system.

# Aliases
- `LinSysVecRowSketch`
- `LinSysVecRowSelect`
"""
abstract type LinSysVecRowSampler <: LinSysSampler end
LinSysVecRowSketch = LinSysVecRowSampler
LinSysVecRowSelect = LinSysVecRowSampler


"""
    LinSysVecColSampler <: LinSysSampler

Abstract supertype for sampling, sketching or deterministically selecting a column space
    element from a linear system.

# Aliases
- `LinSysVecColSketch`
- `LinSysVecColSelect`
"""
abstract type LinSysVecColSampler <: LinSysSampler end
LinSysVecColSketch = LinSysVecColSampler
LinSysVecColSelect = LinSysVecColSampler

"""
    LinSysBlkRowSampler <: LinSysSampler

Abstract supertype for sampling, sketching or deterministically selecting a collection of
    row space elements from a linear system.

# Aliases
- `LinSysBlkRowSketch`
- `LinSysBlkRowSelect`
"""
abstract type LinSysBlkRowSampler <: LinSysSampler end
LinSysBlkRowSketch = LinSysBlkRowSampler
LinSysBlkRowSelect = LinSysBlkRowSampler

"""
    LinSysBlkColSampler <: LinSysSampler

Abstract supertype for sampling, sketching or deterministically selecting a collection of
    column space elements from a linear system.

# Aliases
- `LinSysBlkColSketch`
- `LinSysBlkColSelect`
"""
abstract type LinSysBlkColSampler <: LinSysSampler end
LinSysBlkColSketch = LinSysBlkColSampler
LinSysBlkColSelect = LinSysBlkColSampler

#############################################
# Vector Row Sampler/Sketch/Selector
#############################################
# Non-adaptive Cyclic Methods
include("linear_samplers/vec_row_deterministic_cyclic.jl")
include("linear_samplers/vec_row_one_rand_cyclic.jl")
include("linear_samplers/vec_row_hop_rand_cyclic.jl")
include("linear_samplers/vec_row_rand_cyclic.jl")

# Non-adaptive Sampling (with replacement)
include("linear_samplers/vec_row_unid_replace.jl")
include("linear_samplers/vec_row_prop_to_norm_replace.jl")

# Non-adaptive Sketching
#include("linear_samplers/vec_row_uniform.jl")
#include("linear_samplers/vec_row_uniform_sym.jl")
#include("linear_samplers/vec_row_gauss.jl")
#include("linear_samplers/vec_row_uniform_sparse.jl")
#include("linear_samplers/vec_row_uniform_sym_sparse.jl")
#include("linear_samplers/vec_row_gauss_sparse.jl")

# Adaptive Deterministic Selection
#include("linear_samplers/vec_row_max_residual.jl")
#include("linear_samplers/vec_row_max_distance.jl")

# Adaptive Cyclic Methods
#include("linear_samplers/vec_row_residual_cycle.jl")
#include("linear_samplers/vec_row_distance_cycle.jl")

#############################################
# Vector Column Sampler/Sketch/Selector
#############################################
# Non-adaptive Cyclic Methods
include("linear_samplers/vec_col_deterministic_cyclic.jl")

#############################################
# Block Row Sampler/Sketch/Selector
#############################################

#############################################
# Block Column Sampler/Sketch/Selector
#############################################

#############################################
# Compositional Sampler/Sketch/Selector
#############################################
# Wishlist


#############################################
# Export Statements
#############################################
# See RLinearAlgebra.jl
#export LinSysSampler, LinSysSketch, LinSysSelect
#export LinSysVecRowSampler, LinSysVecRowSketch, LinSysVecRowSelect
#export LinSysVecColSampler, LinSysVecColSketch, LinSysVecColSelect
#export LinSysBlkRowSampler, LinSysBlkRowSketch, LinSysBlkRowSelect
#export LinSysBlkColSampler, LinSysBlkColSketch, LinSysBlkColSelect
