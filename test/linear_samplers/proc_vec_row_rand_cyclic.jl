# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVRRandCyclic

using Test, RLinearAlgebra, Random

Random.seed!(1010)

@testset "LSVR Random Cyclic -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysVecRowRandCyclic) == LinSysVecRowSampler

    # Test whether ordering is appropriately sampled, and changes after exhaustion
    @test let
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

end

end
