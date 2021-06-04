# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

# using Random

"""
    LinSysVecRowOneRandCyclic <: LinSysVecRowSelect

A mutable structure with a field to store a cycling order. Randomly specifies a cycling
order over the equations of a linear system. Once this ordering is specified, the ordering
is kept fixed.

# Fields
- `order::Union{Vector{Int64},Nothing}`

Calling `LinSysVecRowOneRandCyclic()` defaults to setting `order` to `nothing`. The `sample`
function will handle the re-initialization of the fields once the system is provided.
"""
mutable struct LinSysVecRowOneRandCyclic <: LinSysVecRowSelect
    order::Union{Vector{Int64},Nothing}
end
LinSysVecRowOneRandCyclic() = LinSysVecRowOneRandCyclic(nothing)

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowOneRandCyclic,
    A::Matrix{Float64},
    b::Vector{Float64},
    x::Vector{Float64},
    iter::Int64
)
    if iter == 1
        type.order = randperm(length(b))
    end

    eqn_ind = type.order[mod(iter, 1:length(b))]
    return A[eqn_ind, :], b[eqn_ind]
end

#export LinSysVecRowOneRandCyclic

# Tests
if @isdefined linear_samplers_testset_prop

    # Test for appropriate super type
    tsts = Expr[:(@test supertype(LinSysVecRowOneRandCyclic) == LinSysVecRowSampler)]

    # Make sure that the ordering remains unchanged
    let tst
        tst = quote
            A = rand(10,3)
            b = rand(10)
            x = rand(3)

            cyc = LinSysVecRowOneRandCyclic()

            α, β = RLinearAlgebra.sample(cyc, A, b, x, 1)

            order = copy(cyc.order)

            flag = true
            for j = 2:100
                α, β = RLinearAlgebra.sample(cyc, A, b, x, j)

                #Ordering should not change
                flag = flag & (cyc.order == order)
            end

            flag
        end

        push!(tsts, :(@test $tst))
    end

    push!(linear_samplers_testset_proc,
        "LSVR One Hop Random Cyclic -- Procedural" => tsts
    )
end
