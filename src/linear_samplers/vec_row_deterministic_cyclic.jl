# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type
# 4. Tests

# using Random

"""
    LinSysVecRowDetermCyclic <: LinSysVecRowSelect

An immutable structure without any fields. Specifies deterministic cycling through the
equations of a linear system.
"""
struct LinSysVecRowDetermCyclic <: LinSysVecRowSelect end

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowDetermCyclic,
    A::Matrix{Float64},
    b::Vector{Float64},
    x::Vector{Float64},
    iter::Int64
)

    eqn_ind = mod(iter, 1:length(b))
    return A[eqn_ind, :], b[eqn_ind]
end

#export LinSysVecRowDetermCyclic

# Tests
if @isdefined linear_samplers_testset_proc

    # Test for appropriate super type
    tsts = Expr[:(@test supertype(LinSysVecRowDetermCyclic) == LinSysVecRowSampler)]

    # Test for determining whether the rows are being cycled through correctly
    let tst
        tst = quote
            A = rand(10,3)
            b = rand(10)
            x = rand(3) #Irrelevant

            cyc = LinSysVecRowDetermCyclic()

            flag = true

            for i = 11:20
                α, β = RLinearAlgebra.sample(cyc, A, b, x, i)
                flag = flag & (α == A[i-10,:])
                flag = flag & (β == b[i-10])
            end

            flag
        end

        push!(tsts, :(@test $tst))
    end

    push!(linear_samplers_testset_proc,
        "LSVR Deterministic Cyclic -- Procedural" => tsts
    )
end
