using LinearAlgebra, Random, Distributions

################################
#  DISTRIBUTIONS FOR KACZMARZ  #
################################
"""
    RowDistributionType

Abstract supertype for sampling over the equations of a linear system.
"""
abstract type RowDistributionType end

"""
    UFDistribution <: RowDistributionType

Abstract specification of a discrete uniform distribution over a system of linear equations.
There are no fields.
"""
struct UFDistribution <: RowDistributionType end

"""
    SVDistribution <: RowDistributionType

Abstract specification of a Strohmer-Vershynin sampling scheme over a system of linear
equations. There are no fields.

See Strohmer, T., Vershynin, R. A Randomized Kaczmarz Algorithm with Exponential Convergence.
J Fourier Anal Appl 15, 262 (2009). https://doi.org/10.1007/s00041-008-9030-4
"""
struct SVDistribution <: RowDistributionType end


"""
    distribution(type::T, A::Matrix) where T<:RowDistributionType

Generates a vector in the probability simplex corresponding to (1) a linear system whose
coefficient matrix is given by argument `A` and (2) the abstract distribution specified by
`type`.
"""
function distribution(type::UFDistribution, A::Matrix)
    p = ones(Float64, size(A, 1))
    return p./sum(p)
end

function distribution(type::SVDistribution, A::Matrix)
    p = zeros(Float64, size(A, 1))
    for i=1:size(A, 1)
        p[i] = norm(A[i,:], 2)
    end
    return p./sum(p)
end

#################
#  RPM Samplers #
#################

# Type structure
"""
    RPMSamplerType

Abstract supertype for vector sketching (including discrete equation samplers).
"""
abstract type RPMSamplerType end


"""
    SamplerKaczmarzWR <: RPMSamplerType

A mutable structure that specifies sampling with replacement over the equations of a linear
systems according to its field `distribution_type`.

# Fields

- `distribution_type::RowDistributionType`, a discrete distribution
- `dist::Union{Distributions.Categorical{Float64, Vector{Float64}}, Nothing}`, a vector
    corresponding to the sampling probabilities over the equations of a system

Calling `SamplerKaczmarzWR()` defaults to uniform sampling over the equations of a system.
"""
mutable struct SamplerKaczmarzWR <: RPMSamplerType
    distribution_type::RowDistributionType
    dist::Union{Distributions.Categorical{Float64, Vector{Float64}}, Nothing}
end
SamplerKaczmarzWR() = SamplerKaczmarzWR(UFDistribution(), nothing)

"""
    SamplerKaczmarzCYC <: RPMSamplerType

A mutable structure that specifies cyclic sampling over the equations of linear system.
The cycle is first set randomly and then is kept fixed.

# Fields

- `perm::Union{Vector{Int64}, Nothing}` is a vector specifying the ordering over the
    equations of the system.

Calling `SamplerKaczmarzCYC()` defaults to setting `perm` to `nothing`.
"""
mutable struct SamplerKaczmarzCYC <: RPMSamplerType
    perm::Union{Vector{Int64}, Nothing}
end
SamplerKaczmarzCYC() = SamplerKaczmarzCYC(nothing)

"""
    SamplerMotzkin <: RPMSamplerType

A mutuable structure that specifies Motzkin (maximum absolute residual) equation selection.
If the field `sampled` is `true`, then the maximum absolute residual equation is selected
from a unfirormly randomly sampled subset.

# Fields

- `sampled::Bool`, if `false` then equation is selected from all equations in the system. if
    `true` then the equation is selected from a random subset

Calling `SamplerMotzkin()` defaults to setting `sampled` to `false`.
"""
mutable struct SamplerMotzkin <: RPMSamplerType
    sampled::Bool
end
SamplerMotzkin() = SamplerMotzkin(false)

"""
    SamplerGaussSketch <: RPMSamplerType

A structure that specifies the Gaussian vector sketching.
"""
struct SamplerGaussSketch <: RPMSamplerType end

RPMSamplers() = [SamplerKaczmarzWR(), SamplerKaczmarzCYC(), SamplerMotzkin(), SamplerGaussSketch()]

# Implementation

"""
    sample(type::T, A::Matrix{Float64}, b::Vector{Float64}, x::Vector{Float64}, iter::Int64)
        where T <: RPMSamplerType

Implements the sampling scheme specified by `type` for a linear system with coefficient
matrix `A` and constant vector `b` at iterate `x` and iterate counter `iter`. Returns a
vector-scalar pair that specifies a hyper plane.
"""
function sample(
    type::SamplerKaczmarzWR,
    A::Matrix{Float64},
    b::Vector{Float64},
    x::Vector{Float64},
    iter::Int64
)
    if iter == 1
        p = distribution(type.distribution_type, A)
        type.dist = Categorical(p)
    end

    w_ind = rand(type.dist,1)
    return A[w_ind[1],:], b[w_ind[1]]
end
function sample(
        type::SamplerKaczmarzCYC,
        A::Matrix{Float64},
        b::Vector{Float64},
        x::Vector{Float64},
        iter::Int64
    )
    if iter == 1
        type.perm = randperm(length(b))
    end
    i = mod(iter, length(b))
    row = type.perm[i + 1]

    return A[row,:], b[row]
end
function sample(
        type::SamplerGaussSketch,
        A::Matrix{Float64},
        b::Vector{Float64},
        x::Vector{Float64},
        iter::Int64
    )
    N = length(b)
    w = randn(N)
    return A'*w, dot(b,w)
end
function sample(
    type::SamplerMotzkin,
    A::Matrix{Float64},
    b::Vector{Float64},
    x::Vector{Float64},
    iter::Int64
)
    if type.sampled == true
        rows = sample_subset(length(b))
        r = A[rows, :]*x - b[rows]
        j = argmax(abs.(r))
        i = rows[j]
    else
        r = A*x - b
        i = argmax(abs.(r))
    end
    return A[i,:], b[i]
end

"""
    sample_subset(n::Int64)

Generates a random set of values from `1` to `n` of a random size.
"""
function sample_subset(n::Int64)
    p = randperm(n)
    low = rand(1:n)
    up = rand(low:n)
    return p[low:up]
end



"""
    count_sketch(A :: Matrix{Float64}, b :: Vector{Float64}, e :: Int64 = 5)

Impelments Count sketch sampling scheme.

# Arguments
- `A :: Matrix{Float64}`, coefficient matrix
- `b :: Vector{Float64}`, constant vector

# Keywords
- `e :: Int64 = 5`, size of unrepeated count-sketch  chunk.

# Returns
- `:: Function`, argument free function that returns a pair (q, s) where q is the sketched row,
                and s is the corresponding sketched constant
"""
function count_sketch(A::Matrix{Float64}, b::Vector{Float64}, e::Int64 = 10)
    N = length(b)

    dist = Categorical(ones(Float64,e)/e)
    W = Vector{Int64}[]
    state = 0

    function genSample()
        #Generate W if it is empty
        if state == 0
            indx = rand(dist,N)
            W = map(λ -> findall(indx .== λ), 1:e)
        end

        #Pop w from W
        w = popfirst!(W)

        state = mod(state+1,e)
        return sum(A[w,:],dims=1)', sum(b[w])
    end

    return genSample
end
