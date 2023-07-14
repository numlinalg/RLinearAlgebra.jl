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

end

end
