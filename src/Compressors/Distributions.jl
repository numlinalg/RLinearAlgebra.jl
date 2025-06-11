"""
    Distribution

An abstract supertype for structures specifying distribution for indices in subcompression methods.
"""
abstract type Distribution end

"""
    DistributionRecipe

An abstract supertype for structures with pre-allocated memory for distribution function
    subcompression methods.
"""
abstract type DistributionRecipe end

# Docstring Components
distribution_arg_list = Dict{Symbol,String}(
    :distribution => "`distribution::Distribution`, a user-specified distribution function for subcompression.",
    :distribution_recipe => "`distribution::DistributionRecipe`, a fully initialized realization of distribution.",
    :A => "`A::AbstractMatrix`, a coefficient matrix.",
)

distribution_output_list = Dict{Symbol,String}(
    :distribution_recipe => "A `DistributionRecipe` object."
)

distribution_method_description = Dict{Symbol,String}(
    :complete_distribution => "A function that generates a `DistributionRecipe` given the 
    arguments.",
    :update_distribution => "A function that updates the `Distribution` in place given 
    arguments.",
)
"""
    complete_distribution(distribution::Distribution, A::AbstractMatrix)

$(distribution_method_description[:complete_distribution])

# Arguments
- $(distribution_arg_list[:distribution])
- $(distribution_arg_list[:A]) 

# Outputs
- $(distribution_output_list[:distribution_recipe])
"""
function complete_distribution(distribution::Distribution, A::AbstractMatrix)
    return throw(ArgumentError("No `complete_distribution` method defined for a distribution of type \
    $(typeof(distribution)) and $(typeof(A))."))
end

function complete_distribution(distribution::Distribution, A::AbstractMatrix, b::AbstractVector)
    complete_distribution(distribution, A)
    return nothing
end

function complete_distribution(distribution::Distribution, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    complete_distribution(distribution, A, b)
    return nothing
end

"""
    update_distribution!(distribution::DistributionRecipe, A::AbstractMatrix)

$(distribution_method_description[:update_distribution])

# Arguments
- $(distribution_arg_list[:distribution_recipe])
- $(distribution_arg_list[:A]) 

# Outputs
- Modifies the `DistributionRecipe` in place and returns nothing.
"""
function update_distribution!(distribution::DistributionRecipe, A::AbstractMatrix)
    return throw(ArgumentError("No `update_distribution!` method defined for a distribution of type \
    $(typeof(distribution)) and $(typeof(A))."))
end

function update_distribution!(distribution::DistributionRecipe, A::AbstractMatrix, b::AbstractVector)
    update_distribution!(distribution, A)
    return nothing
end

function update_distribution!(distribution::DistributionRecipe, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    update_distribution!(distribution, A, b)
    return nothing
end

function sample_distribution!(x::AbstractVector, distribution::DistributionRecipe)
    wsample!(distribution.state_space, distribution.weights, x, ordered = true, replace = distribution.replace)
    return throw(ArgumentError("No `sample_distribution!` method defined for a distribution of type \
    $(typeof(distribution)) and $(typeof(x))."))
end

###########################################
# Include Distribution files
###########################################
include("Distributions/uniform.jl")