"""
    Uniform <: Distribution

Uniform distribution over the row/column index set of a matrix.

# Mathematical Description

During the subcompression, the uniform distribution is defined on the domain of row/column 
indices. If it's compressing from the left, then it means every row index has the same 
probability weight. If it's compressing from the right, then it means every column index 
has the same probability weight.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()` 
    or `Undef`.
- `replace::Bool`, an option to replace or not during the sampling process based 
    on the given weights.

# Constructor

    Uniform(;cardinality=Undef(), replace = false)

## Returns
- A `Uniform` object.
"""
mutable struct Uniform <: Distribution
    cardinality::Cardinality
    replace::Bool
end

function Uniform(; cardinality = Undef(), replace = false)
    return Uniform(cardinality, replace)
end

"""
    UniformRecipe <: DistributionRecipe

The recipe containing all allocations and information for the uniform distribution.

# Fields
- `cardinality::C where C<:Cardinality`, the cardinality of the compressor. The
    value is either `Left()` or `Right()` or `Undef()`.
- `replace::Bool`, an option to replace or not during the sampling process based 
    on the given weights.
- `state_space::Vector{Int64}`, the row/column index set.
- `weights::ProbabilityWeights`, the weights of each element in the state space.
"""
mutable struct UniformRecipe <: DistributionRecipe
    cardinality::Cardinality
    replace::Bool
    state_space::Vector{Int64}
    weights::ProbabilityWeights
end

function complete_distribution(distribution::Uniform, A::AbstractMatrix)
    cardinality = distribution.cardinality
    if cardinality == Left()
        n_rows = size(A, 1)
        state_space = collect(1: n_rows)
        weights = ProbabilityWeights(ones(n_rows))
    elseif cardinality == Right()
        n_cols = size(A, 2)
        state_space = collect(1: n_cols)
        weights = ProbabilityWeights(ones(n_cols))
    end

    return UniformRecipe(cardinality, distribution.replace, state_space, weights)
end

function update_distribution!(ingredients::UniformRecipe, A::AbstractMatrix)
    if ingredients.cardinality == Left()
        n_rows = size(A, 1)
        ingredients.state_space = collect(1: n_rows)
        ingredients.weights = ProbabilityWeights(ones(n_rows))
    elseif ingredients.cardinality == Right()
        n_cols = size(A, 2)
        ingredients.state_space = collect(1: n_cols)
        ingredients.weights = ProbabilityWeights(ones(n_cols))
    end

    return nothing
end