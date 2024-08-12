############################################################################################
## This file is part of RLinearAlgebra.jl
##
## Overview: abstractions and methods for sampling, sketching or selecting from linear
## systems of equations.
##
## Contents
## - Abstract Types
## - `sample` function documentation
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
# `sample` Function Documentation
#############################################
"""
    sample(type::T where T<:LinSysSampler,
        A::AbstractArray,
        b::AbstractVector,
        x::AbstractVector,
        iter::Int64)

A common interface for specifying different strategies for sampling, selecting or sketching
    a linear system specified by `A` and `b`. The `type` argument is used to select the an
    appropriately defined strategy. The argument `x` is the current iterate value for the
    solution. The arguent `iter` is the iteration counter.

The value(s) returned by sample depend on the subtype of `LinSysSampler` being used.
    Specifically,
- For `T<:LinSysVecRowSampler`, a vector in the row space of `A` and constant are returned
- For `T<:LinSysVecColSampler`, a vector of `length(x)`, the matrix `A`,
    and a scalar-valued residual are returned.
"""
function sample(
    type::Nothing,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    return nothing
end

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
include("linear_samplers/vec_row_uniform.jl")
include("linear_samplers/vec_row_gaussian.jl")
include("linear_samplers/vec_row_uniform_sparse.jl")
include("linear_samplers/vec_row_gaussian_sparse.jl")
include("linear_samplers/block_row_count_sketch.jl")
include("linear_samplers/block_row_sample_wo_replacement.jl")
#include("linear_samplers/vec_row_uniform_sym_sparse.jl")
#include("linear_samplers/vec_row_uniform_sym.jl")

# Adaptive Deterministic Selection
include("linear_samplers/vec_row_max_residual.jl")
include("linear_samplers/vec_row_max_distance.jl")

# Adaptive Cyclic Methods
include("linear_samplers/vec_row_cyclic_residual.jl")
include("linear_samplers/vec_row_cyclic_distance.jl")

#############################################
# Vector Column Sampler/Sketch/Selector
#############################################
# Non-adaptive Cyclic Methods
include("linear_samplers/vec_col_deterministic_cyclic.jl")
include("linear_samplers/vec_col_one_rand_cyclic.jl")
include("linear_samplers/block_col_sample_wo_replacement.jl")
# Non-adaptive Sampling (with replacement)
#Leventhal-Lewis (non-symmetric)

#############################################
# Block Row Sampler/Sketch/Selector
#############################################
include("linear_samplers/block_row_gaussian.jl")
include("linear_samplers/block_row_rand_cyclic.jl")
include("linear_samplers/block_row_rand_replace.jl")
#############################################
# Block Column Sampler/Sketch/Selector
#############################################
include("linear_samplers/block_col_rand_cyclic.jl")
include("linear_samplers/block_col_gaussian.jl")
include("linear_samplers/block_col_rand_replace.jl")
#############################################
# Compositional Sampler/Sketch/Selector
#############################################
# Wishlist

#############################################
# Sampler Helper Functions
#############################################
include("linear_samplers/linear_sampler_helpers/helpers_cyclic.jl")
#############################################
# Export Statements
#############################################
# See RLinearAlgebra.jl
#export LinSysSampler, LinSysSketch, LinSysSelect
#export LinSysVecRowSampler, LinSysVecRowSketch, LinSysVecRowSelect
#export LinSysVecColSampler, LinSysVecColSketch, LinSysVecColSelect
#export LinSysBlkRowSampler, LinSysBlkRowSketch, LinSysBlkRowSelect
#export LinSysBlkColSampler, LinSysBlkColSketch, LinSysBlkColSelect
