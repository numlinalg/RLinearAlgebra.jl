# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVR

using Test, RLinearAlgebra, Random

Random.seed!(1010)

@testset "LSVR Hop Random Cyclic -- Procedural" begin

    # Verify appropraite super type
    @test supertype(LinSysVecRowHopRandCyclic) == LinSysVecRowSampler

    # Test whether cycles are changed after appropriate hop period
    for k = 1:10
        @test let
            A = rand(10,3)
            b = rand(10)
            x = rand(3)

            cyc = LinSysVecRowHopRandCyclic(k)

            α, β = RLinearAlgebra.sample(cyc, A, b, x, 1)

            order = copy(cyc.order)

            flag = true
            for j = 2:(cyc.hop * 10)
                α, β = RLinearAlgebra.sample(cyc, A, b, x, j)

                # Ordering should not change
                flag = flag & (cyc.order == order)
            end

            α, β = RLinearAlgebra.sample(cyc, A, b, x, cyc.hop * 10 + 1)

            # Ordering should change
            flag = flag & (cyc.order != order)

            flag
        end
    end
end

end
