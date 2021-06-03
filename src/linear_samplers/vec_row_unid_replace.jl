# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

"""
    LinSysVecRowUnidSampler <: LinSysVecRowSampler

An immutable structure without fields that specifies randomly cycling from the rows of a
linear system with uniform probability and with replacement.
"""
struct LinSysVecRowUnidSampler <: LinSysVecRowSampler end

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowUnidSampler,
    A::Matrix{Float64},
    b::Vector{Float64},
    x::Vector{Float64},
    iter::Int64
)
    eqn_ind = rand(Base.OneTo(length(b)))

    return A[eqn_ind, :], b[eqn_ind]
end

export LinSysVecRowUnidSampler

if @isdefined linear_samplers_testset_proc

    # Test for appropriate super type
    tsts = Expr[:(@test supertype(LinSysVecRowUnidSampler) == LinSysVecRowSampler)]

    # Test construction
    let tst
        tst = quote
            A = rand(10,3)
            b = rand(10)
            x = rand(3)

            samp = LinSysVecRowUnidSampler()

            α, β = RLinearAlgebra.sample(samp, A, b, x, 1)

            true
        end

        push!(tsts, :(@test $tst))
    end

    push!(linear_samplers_testset_proc,
        "LSVR Uniform Discrete Sampling -- Procedural" => tsts
    )
end
