# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVRDetermCycle

using Test, RLinearAlgebra, Random

Random.seed!(1010)

@testset "LSVR Deterministic Cyclic -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysVecRowDetermCyclic) == LinSysVecRowSampler

    # Test for appropriate cycling of rows
    @test let
        A = rand(10,3)
        b = rand(10)
        x = rand(3)

        cyc = LinSysVecRowDetermCyclic()

        flag = true

        for i = 11:20
            α, β = RLinearAlgebra.sample(cyc, A, b, x, i)
            flag = flag & (α == A[i-10,:])
            flag = flag & (β == b[i-10])
        end

        flag
    end

end

end
