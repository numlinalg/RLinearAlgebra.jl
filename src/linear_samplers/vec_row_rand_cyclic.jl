# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

# using Random

"""
    LinSysVecRowRandCyclic <: LinSysVecRowSelect

A mutable structure with a field to store a cycling order. Randomly specifies a cycling
order the equations of a linear system. Once this ordering is exhausted by the solver,
a new random ordering is specified. This process is repeated

# Fields
- `order::Union{Vector{Int64},Nothing}`

Calling `LinSysVecRowOneRandCyclic()` defaults to setting `order` to `nothing`. The `sample`
function will handle the re-initialization of the fields once the system is provided.
"""
mutable struct LinSysVecRowRandCyclic <: LinSysVecRowSelect
    order::Union{Vector{Int64},Nothing}
end
LinSysVecRowRandCyclic() = LinSysVecRowRandCyclic(nothing)

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowRandCyclic,
    A::Matrix{Float64},
    b::Vector{Float64},
    x::Vector{Float64},
    iter::Int64
)

    mod_ind = mod(iter, 1:length(b))
    if mod_ind == 1
        type.order = randperm(length(b))
    end

    eqn_ind = type.order[mod_ind]
    return A[eqn_ind, :], b[eqn_ind]
end

export LinSysVecRowRandCyclic

# Tests
if @isdefined linear_samplers_testset_proc

    # Test for appropriate super type
    tsts = Expr[:(@test supertype(LinSysVecRowRandCyclic) == LinSysVecRowSampler)]

    # Make sure ordering changes after exhaustion
    let tst
        tst = quote
            A = rand(10, 3)
            b = rand(10)
            x = rand(3)

            cyc = LinSysVecRowRandCyclic()

            # Generate random ordering
            α, β = RLinearAlgebra.sample(cyc, A, b, x, 1)
            order = copy(cyc.order)

            flag = true
            for j = 2:10
                α, β = RLinearAlgebra.sample(cyc, A, b, x, j)
                # Order should remain fixed
                flag = flag & (order == cyc.order)
            end

            # Order should change
            α, β = RLinearAlgebra.sample(cyc, A, b, x, 11)
            flag = flag & (order != cyc.order)
            flag = flag & (α == A[cyc.order[1],:])
            flag = flag & (β == b[cyc.order[1]])

            flag
        end

        push!(tsts, :(@test $tst))
    end

    push!(linear_samplers_testset_proc,
        "LSVR Random Cyclic -- Procedural" => tsts
    )
end
