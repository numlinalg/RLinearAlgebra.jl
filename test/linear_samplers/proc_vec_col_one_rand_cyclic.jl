# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVCOneRandCyclic

using Test, RLinearAlgebra, Random

Random.seed!(1010)

@testset "LSVC One Hop Random Cyclic -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysVecColOneRandCyclic) == LinSysVeccolSampler

    # Test whether row ordering remains fixed
    @test let
        A = rand(10,3)
        b = rand(10)
        x = rand(3)

        cyc = LinSysVecColOneRandCyclic()

        v, M, res = RLinearAlgebra.sample(cyc, A, b, x, 1)

        order = copy(cyc.order)

        for j = 2:lenght(cyc.order)
            v, M, res = RLinearAlgebra.sample(cyc, A, b, x, 1)
            @assert v[order[j]] == 1.0
        end

    end

end

end
