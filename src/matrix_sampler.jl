using LinearAlgebra, Random, Distributions

abstract type AbstractRowDistribution end


struct UFDistribution <: AbstractRowDistribution end
function distribution(A::Matrix, type::UFDistribution)
    p = ones(Float64, size(A, 1))
    return p./sum(p)
end

struct SVDistribution <: AbstractRowDistribution end
"""
    distribution(A :: Matrix{Float64}, )

    Implements the Strohmer and Vershynin sampler of:
    > Strohmer, T., Vershynin, R. A Randomized Kaczmarz Algorithm with Exponential Convergence. J Fourier Anal Appl 15, 262 (2009). https://doi.org/10.1007/s00041-008-9030-4

# Arguments
- `A::Matrix{Float64}`, coefficient matrix

# Returns

- `:: Vector{Float64}`, discrete probability distribution p

"""
function distribution(A::Matrix, type::SVDistribution)
    p = zeros(Float64, size(A, 1))
    for i=1:size(A, 1)
        p[i] = norm(A[i,:], 2)
    end
    return p./sum(p)
end


"""
    kaczmarzWR(A :: Matrix{Float64}, b :: Vector{Float64}, p :: Vector{Float64} = ones(Float64, length(b))/length(b))

    Implements Kaczmarz sampling with replacement scheme according to distribution p

# Arguments
- `A::Matrix{Float64}`, coefficient matrix
- `b::Vector{Float64}`, constant vector
- `row_sampler::AbstractRowSampler`, sampler type (defaults to uniform) 

# Returns

- `:: Function`, argument free function that returns a pair (q, s) where q is the sampled
                    row, and s is the corresponding sampled constant vector

"""
function kaczmarzWR(
    A::Matrix{Float64},
    b::Vector{Float64},
    dist_type::AbstractRowDistribution=UFDistribution()
)
    p = distribution(A, dist_type)

    dist = Categorical(p)

    function genSample()
        w_ind = rand(dist,1)
        return A[w_ind[1],:], b[w_ind[1]]
    end

    return genSample
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
function kaczmarzCyc(A::Matrix{Float64}, b::Vector{Float64})
    counter_max = length(b)

    counter = 0
    ordering = Int64[]

    function genSample()
        if counter == 0
            ordering = randperm(counter_max)
        end

        w = popfirst!(ordering)

        counter = mod(counter+1,counter_max)
        return A[w,:], b[w]
    end

    return genSample
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
function gauss(A::Matrix{Float64}, b::Vector{Float64})
    N = length(b)
    function genSample()
        w = randn(N)
        return A'*w, dot(b,w)
    end
    return genSample
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
