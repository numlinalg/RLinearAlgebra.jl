# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

# using Random, Distributions

"""
    LinSysVecRowPropToNormSampler{T} <: LinSysVecRowSampler where T <: Categorical

A parametric mutable structure that specifies sampling from the rows of the equation where
the probability of selecting a given equation is proportional to the sum of squares of the
coefficients of the given equation. The solver will appropriately initialize the
distribution.

See Strohmer, T., Vershynin, R. A Randomized Kaczmarz Algorithm with Exponential
Convergence. J Fourier Anal Appl 15, 262 (2009). https://doi.org/10.1007/s00041-008-9030-4

# Aliases
- `LinSysVecRowSVSampler`

# Fields
- `dist::T`, a categorical probability distribution.

Calling `LinSysVecRowPropToNormSampler()` or `LinSysVecRowSVSampler()` defaults `dist` to
`Categorical(1.0)`.
"""
mutable struct LinSysVecRowPropToNormSampler{T} <: LinSysVecRowSampler where T <:
    Categorical

    dist::T
end
LinSysVecRowSVSampler = LinSysVecRowPropToNormSampler
LinSysVecRowPropToNormSampler() = LinSysVecRowPropToNormSampler(Categorical(1.0))

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowSVSampler{T} where T <: Categorical,
    A::Matrix{Float64},
    b::Vector{Float64},
    x::Vector{Float64},
    iter::Int64
)
    if iter == 1
        prob = sum(A.^2, dims=2)[:,1]
        type.dist = Categorical(prob ./ sum(prob))
    end

    eqn_ind = rand(type.dist)

    return A[eqn_ind, :], b[eqn_ind]
end

#export LinSysVecRowPropToNormSampler, LinSysVecRowSVSampler

if @isdefined linear_samplers_testset_proc

    tsts = Expr[]

    # Test super type
    push!(tsts, :(@test supertype(LinSysVecRowPropToNormSampler) == LinSysVecRowSampler))

    # Test alias
    push!(tsts, :(@test LinSysVecRowPropToNormSampler == LinSysVecRowSVSampler))

    # Test construction
    let tst
        tst = quote
            A = rand(10,3)
            b = rand(10)
            x = rand(3)

            samp = LinSysVecRowSVSampler()

            α, β = RLinearAlgebra.sample(samp, A, b, x, 1)

            true
        end

        push!(tsts, :(@test $tst))
    end

    push!(linear_samplers_testset_proc,
        "LSVR Strohmer-Vershynin Sampling -- Procedural" => tsts
    )



end
