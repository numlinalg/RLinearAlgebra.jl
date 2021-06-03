############################################################################################
##
## Overview: abstractions and methods for sampling, sketching or selecting from linear
## systems of equations.
##
## Contents
## - Test Sets
## - Abstract Types
## - Vector Row Sampler/Sketch/Selector
## - Vector Column Sampler/Sketch/Selector
## - Block Row Sampler/Sketch/Selector
## - Block Column Sampler/Sketch/Selector
## - Export Statements
##
############################################################################################

# Dependencies: Random, Distributions
# Additional dependnecies for testing: Test

#############################################
# Test Sets
#############################################
# Procedural tests -- usually inexpensive tests, usually deterministic
linear_samplers_testset_proc = Dict{String,Vector{Expr}}()

# Property tests -- usually expensive to run, usually complex or non-determinism
linear_samplers_testset_prop = Dict{String,Vector{Expr}}()

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

# Tests
push!(linear_samplers_testset_proc,
    "Linear System Sampler Super Type" => [
        :(@test LinSysSketch == LinSysSampler), #Check alias
        :(@test LinSysSelect == LinSysSampler), #Check alias
    ]
)

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

# Tests
push!(linear_samplers_testset_proc,
    "Linear System Vector Row Sampler Super Type" => [
        :(@test supertype(LinSysVecRowSampler) == LinSysSampler), #Check parent type
        :(@test LinSysVecRowSketch == LinSysVecRowSampler), #Check alias
        :(@test LinSysVecRowSelect == LinSysVecRowSampler), #Check alias
    ]
)

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

# Tests
push!(linear_samplers_testset_proc,
    "Linear System Vector Column Sampler Super Type" => [
        :(@test supertype(LinSysVecColSampler) == LinSysSampler), #Check parent type
        :(@test LinSysVecColSketch == LinSysVecColSampler), #Check alias
        :(@test LinSysVecColSelect == LinSysVecColSampler), #Check alias
    ]
)

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

# Tests
push!(linear_samplers_testset_proc,
    "Linear System Block Row Sampler Super Type" => [
        :(@test supertype(LinSysBlkRowSampler) == LinSysSampler), #Check parent type
        :(@test LinSysBlkRowSketch == LinSysBlkRowSampler), #Check alias
        :(@test LinSysBlkRowSelect == LinSysBlkRowSampler), #Check alias
    ]
)


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

# Tests
push!(linear_samplers_testset_proc,
    "Linear System Vector Column Sampler Super Type" => [
        :(@test supertype(LinSysBlkColSampler) == LinSysSampler), #Check parent type
        :(@test LinSysBlkColSketch == LinSysBlkColSampler), #Check alias
        :(@test LinSysBlkColSelect == LinSysBlkColSampler), #Check alias
    ]
)


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
include("linear_samplers/vec_row_prop-to-norm_replace.jl")

# Non-adaptive Sketching
include("linear_samplers/vec_row_uniform.jl")
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

export LinSysSampler, LinSysSketch, LinSysSelect
export LinSysVecRowSampler, LinSysVecRowSketch, LinSysVecRowSelect
export LinSysVecColSampler, LinSysVecColSketch, LinSysVecColSelect
export LinSysBlkRowSampler, LinSysBlkRowSketch, LinSysBlkRowSelect
export LinSysBlkColSampler, LinSysBlkColSketch, LinSysBlkColSelect
