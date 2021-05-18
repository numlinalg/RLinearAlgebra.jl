using LinearAlgebra, Random, Distributions

################################
#  DISTRIBUTIONS FOR KACZMARZ  #
################################
abstract type RowDistributionType end

struct UFDistribution <: RowDistributionType end
function distribution(type::UFDistribution, A::Matrix)
    p = ones(Float64, size(A, 1))
    return p./sum(p)
end

struct SVDistribution <: RowDistributionType end
"""
    distribution(type::SVDistribution, A :: Matrix{Float64})

    Implements the Strohmer and Vershynin sampler of:
    > Strohmer, T., Vershynin, R. A Randomized Kaczmarz Algorithm with Exponential Convergence. 
    J Fourier Anal Appl 15, 262 (2009). https://doi.org/10.1007/s00041-008-9030-4

# Arguments
- `A::Matrix{Float64}`, coefficient matrix

# Returns

- `:: Vector{Float64}`, discrete probability distribution p

"""
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
abstract type RPMSamplerType end

mutable struct SamplerKaczmarzWR <: RPMSamplerType
    distribution_type::RowDistributionType
    dist::Union{Distributions.Categorical{Float64, Vector{Float64}}, Nothing}
end
SamplerKaczmarzWR() = SamplerKaczmarzWR(UFDistribution(), nothing)

mutable struct SamplerKaczmarzCYC <: RPMSamplerType
    perm::Union{Vector{Int64}, Nothing}
end
SamplerKaczmarzCYC() = SamplerKaczmarzCYC(nothing)

mutable struct SamplerMotzkin <: RPMSamplerType
    sampled::Bool
end
SamplerMotzkin() = SamplerMotzkin(false)


struct SamplerGaussSketch <: RPMSamplerType end

RPMSamplers() = [SamplerKaczmarzWR(), SamplerKaczmarzCYC(), SamplerMotzkin(), SamplerGaussSketch()]

# Implementation

"""
    sample(type :: SamplerKaczmarzWR, A :: Matrix{Float64}, b :: Vector{Float64},
        x :: Vector{Float64}, iter :: Int64)

    Implements Kaczmarz sampling with replacement scheme

# Arguments
- `A::Matrix{Float64}`, coefficient matrix
- `b::Vector{Float64}`, constant vector

# Returns

- `:: Function`, argument free function that returns a pair (q, s) where q is the sampled
                    row, and s is the corresponding sampled constant vector

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

"""
    kaczmarzCyc(A :: Matrix{Float64}, b :: Vector{Float64})

Implements Kaczmarz sampling under random permutation ordering.

# Arguments
- `A::Matrix{Float64}`, coefficient matrix
- `b::Vector{Float64}`, constant vector

# Returns
- `::Function`, argument free function that returns a pair (q,s) where q is the sampled row,
                and s is the corresponding sampled constant vector.
"""
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

"""
    gauss(A :: Matrix{Float64}, b :: Vector{Float64})`

Implements Gaussian sketching sampling scheme

# Arguments
- `A :: Matrix{Float64}`, coefficient matrix
- `b :: Vector{Float64}`, constant vector

# Returns
- `:: Function`, argument free function that returns a pair (q, s) where q is the sketched row,
            and s is the corresponding sketched constant

"""
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

function sample_subset(n::Int64)
    p = randperm(n)
    low = rand(1:n)
    up = rand(low:n)
    return p[low:up]
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
